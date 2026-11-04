"""One shared approach to a gate and a branch for every place the gate might turn out to be.

`topt_planner` solves the nominal track, so the approach to an unseen gate is fast for the nominal
pose and only a speed cap and an entry cone keep the other poses reachable. A gate is revealed about
0.27 s before the drone gets there, and by then the approach has decided whether the crossing works.

Here the approach is solved together with what follows it. The program has one stem, up to where the
gate comes into sensor range and for the reaction time after it, and one branch per sampled gate
pose. A stem cannot depend on the sample, because the drone does not know it yet, so the solver has
to find the fastest approach from which every sample can still be crossed. No cap or cone is needed
in front of the reveal: the branches are what limit the stem's speed.

    plan up to the previous gate -> stem -> reaction -+-> branch 0 -> crossing 0 -> rest of the lap
                                                      +-> branch 1 -> crossing 1 -> rest of the lap
                                                      +-> ...

The legs are point-mass legs under the limits of `topt_planner`. The result is one whole-lap
`Trajectory` per sample, all equal up to the end of the reaction.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.topt_planner import (
    GRAVITY,
    PLAN_VERSION,
    PlannerConfig,
    Track,
    Trajectory,
    _rk4,
    cache_dir,
)
from lsy_drone_racing.control.track_geometry import gate_capsules

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EDGES = ((0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, -0.15, 0.0))


@dataclass(frozen=True)
class BranchConfig:
    """Options of the branching program.

    Attributes:
        pass_index: The pass whose gate is sampled.
        offsets: Sampled gate offsets in the gate's frame ``(along, sideways, up)`` [m]; the first
            is the nominal pose.
        reveal_radius: Horizontal distance from the nominal gate at which the stem ends [m].
        reaction: Time after that for which the stem continues, unchanged by the reveal [s].
        stem_nodes: Nodes of the stem.
        reaction_nodes: Nodes of the reaction.
        branch_nodes: Nodes of every branch and of every leg after it.
        entry_nodes: Last nodes of a branch that stay inside the entry cone.
        entry_lateral: Half angle of that cone sideways [rad].
        entry_vertical: Half angle of that cone vertically [rad].
        thrust_acc_max: Thrust limit of the branches [m/s^2], below the stem's so that the drone
            has some left for tracking error when it has to answer a reveal.
        tilt_max: Tilt limit of the branches [rad], for the same reason.
        max_iter: IPOPT iterations.
        start_node: Node of the base plan the stem starts from; -1 for the previous gate's crossing.
        rest_legs: Legs solved after the crossing, the last ending on the base plan, which is then
            followed for the rest of the lap; -1 solves every leg to the end of the lap.
        guard_rest: Hold the legs after the crossing to the plan's speed cap and entry cone in front
            of a gate that is still unseen.
    """

    pass_index: int = 3
    offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), *_EDGES)
    reveal_radius: float = 0.7
    reaction: float = 0.08
    stem_nodes: int = 15
    reaction_nodes: int = 3
    branch_nodes: int = 15
    entry_nodes: int = 5
    entry_lateral: float = 0.7
    entry_vertical: float = 0.2
    thrust_acc_max: float = 15.0
    tilt_max: float = 1.1
    max_iter: int = 3000
    start_node: int = -1
    rest_legs: int = -1
    guard_rest: bool = True


@dataclass
class BranchLibrary:
    """Solved branches.

    Attributes:
        offsets: Gate offset of every sample in the gate's frame ``(k, 3)``.
        trajectories: Whole-lap trajectory of every sample.
        stem_end: Node of the trajectories at which they start to differ.
    """

    offsets: NDArray[np.floating]
    trajectories: list[Trajectory]
    stem_end: int

    def blend(self, offset: NDArray[np.floating]) -> tuple[Trajectory, NDArray[np.floating]]:
        """Return the trajectory for a gate offset, and the offset it was made for.

        All trajectories have the same nodes, so they can be mixed node by node. Each sampled offset
        adds the difference of its branch to the nominal one, the first entry of `trajectories`, in
        proportion to how far the gate moved its way along that axis.

        Args:
            offset: The revealed gate offset in the gate's frame.

        Returns:
            The mixed trajectory and the offset of the gate it crosses.
        """
        nominal = self.trajectories[0]
        pos, vel, stamps = nominal.pos.copy(), nominal.vel.copy(), nominal.time.copy()
        made_for = np.zeros(3)
        for sample, trajectory in zip(self.offsets[1:], self.trajectories[1:], strict=True):
            size = float(sample @ sample)
            weight = float(np.clip(offset @ sample / size, 0.0, 1.0))
            pos += weight * (trajectory.pos - nominal.pos)
            vel += weight * (trajectory.vel - nominal.vel)
            stamps += weight * (trajectory.time - nominal.time)
            made_for += weight * sample
        return Trajectory(pos, vel, stamps, nominal.crossings), made_for


def _resample(points: NDArray[np.floating], count: int) -> NDArray[np.floating]:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-9]
    knots, values = arc[keep], points[keep]
    samples = np.linspace(0.0, knots[-1], count)
    if len(knots) < 3:
        return np.stack([np.interp(samples, knots, values[:, a]) for a in range(3)], axis=1)
    return CubicSpline(knots, values)(samples)


def solve_branches(
    track: Track, base: Trajectory, planner: PlannerConfig, config: BranchConfig
) -> BranchLibrary | None:
    """Solve the stem and the branches.

    Args:
        track: The nominal track.
        base: The whole-lap nominal plan; everything up to the previous gate's crossing is kept.
        planner: Limits of the point-mass legs.
        config: The branching options.

    Returns:
        The library, or None if IPOPT did not converge.
    """
    p = config.pass_index
    first = config.start_node if config.start_node >= 0 else int(base.crossings[p - 1])
    gate = int(track.sequence[p])
    centre = track.gates_pos[gate]
    n_rest = len(track.sequence) - p
    to_base = 0 <= config.rest_legs < n_rest - 1
    if to_base:
        n_rest = config.rest_legs + 1
    offsets = np.asarray(config.offsets, dtype=float)
    step = _rk4(np.asarray(planner.drag, dtype=float))
    cos_cross2 = np.cos(planner.crossing_angle_max) ** 2
    tan_lat, tan_up = np.tan(config.entry_lateral), np.tan(config.entry_vertical)
    opti = ca.Opti()

    def nearby(gates: NDArray[np.floating], points: NDArray[np.floating]) -> tuple[list, list]:
        poles = [
            q
            for q in track.poles_pos
            if np.min(np.linalg.norm(points[:, :2] - q[:2], axis=1)) < 0.9
        ]
        caps = [
            (a, b, r)
            for g, a, b, r in gate_capsules(gates, track.gates_rot)
            if np.min(np.linalg.norm(points - gates[g], axis=1)) < 0.9
        ]
        return caps, poles

    def leg(
        n: int,
        duration: ca.MX | float,
        guess: NDArray[np.floating],
        gates: NDArray[np.floating],
        thrust_max: float,
        tilt_max: float,
    ) -> ca.MX:
        cos_tilt2 = np.cos(tilt_max) ** 2
        caps, poles = nearby(gates, guess)
        x, u = opti.variable(9, n + 1), opti.variable(3, n)
        h = duration / n
        for k in range(n):
            opti.subject_to(x[:, k + 1] == step(x[:, k], u[:, k], h))
            opti.subject_to(opti.bounded(-planner.jerk_max, u[:, k], planner.jerk_max))
        for k in range(1, n + 1):
            q, f = x[0:3, k], x[6:9, k]
            opti.subject_to(ca.sumsqr(f) <= thrust_max**2)
            opti.subject_to(f[2] >= 0.0)
            opti.subject_to(f[2] ** 2 >= cos_tilt2 * ca.sumsqr(f))
            opti.subject_to(q[2] >= 0.05)
            for pole in poles:
                opti.subject_to(ca.sumsqr(q[0:2] - pole[0:2]) >= planner.pole_clearance**2)
            for a, b, r in caps:
                d = b - a
                t = ca.fmin(ca.fmax(ca.dot(q - a, d) / float(d @ d), 0.0), 1.0)
                opti.subject_to(ca.sumsqr(q - (a + t * d)) >= (r + planner.frame_clearance) ** 2)
        length = float(np.sum(np.linalg.norm(np.diff(guess, axis=0), axis=1)))
        t_guess = max(length / 2.5, 0.05)
        opti.set_initial(x[0:3, :], guess.T)
        opti.set_initial(x[3:6, :], (np.gradient(guess, axis=0) / (t_guess / n)).T)
        opti.set_initial(x[6:9, :], np.tile(GRAVITY, (n + 1, 1)).T)
        return x

    def crossing(
        end: ca.MX, g_index: int, normal: NDArray[np.floating], gates: NDArray[np.floating]
    ):
        rot = track.gates_rot[g_index]
        rel = end[0:3] - (gates[g_index] + planner.crossing_height * rot[:, 2])
        v = end[3:6]
        lat, ver = planner.crossing_lateral, planner.crossing_vertical
        opti.subject_to(ca.dot(rel, normal) == 0.0)
        opti.subject_to(opti.bounded(-lat, ca.dot(rel, rot[:, 1]), lat))
        opti.subject_to(opti.bounded(-ver, ca.dot(rel, rot[:, 2]), ver))
        opti.subject_to(ca.dot(v, normal) >= 0.0)
        opti.subject_to(ca.dot(v, normal) ** 2 >= cos_cross2 * ca.sumsqr(v))

    def duration_variable(initial: float) -> ca.MX:
        t = opti.variable()
        opti.subject_to(opti.bounded(0.05, t, 5.0))
        opti.set_initial(t, initial)
        return t

    # The stem follows the nominal plan up to the reveal radius.
    ahead = base.pos[first : int(base.crossings[p]) + 1]
    in_range = np.linalg.norm(ahead[:, :2] - centre[:2], axis=1) < config.reveal_radius
    reveal = max(int(np.argmax(in_range)), 2)
    stem_guess = _resample(ahead[: reveal + 1], config.stem_nodes + 1)
    t_stem = duration_variable(float(base.time[first + reveal] - base.time[first]))
    limits = (planner.thrust_acc_max, planner.tilt_max)
    stem = leg(config.stem_nodes, t_stem, stem_guess, track.gates_pos, *limits)
    opti.subject_to(stem[0:3, 0] == base.pos[first])
    opti.subject_to(stem[3:6, 0] == base.vel[first])
    opti.subject_to(ca.sumsqr(stem[0:2, -1] - centre[:2]) == config.reveal_radius**2)

    # The reaction: whatever the sample, the drone has not answered yet.
    react_guess = _resample(ahead[reveal : reveal + 3], config.reaction_nodes + 1)
    react = leg(config.reaction_nodes, config.reaction, react_guess, track.gates_pos, *limits)
    opti.subject_to(react[:, 0] == stem[:, -1])

    cost = t_stem + config.reaction
    legs = []
    normal = track.normal(p)
    rot = track.gates_rot[gate]
    for offset in offsets:
        gates = track.gates_pos.copy()
        gates[gate] = centre + rot @ offset
        c = gates[gate]
        # Branch: from the end of the reaction to the sampled crossing.
        tail = react_guess[-1]
        guess = _resample(
            np.array([tail, 0.5 * (tail + c - 0.3 * normal), c - 0.15 * normal, c]),
            config.branch_nodes + 1,
        )
        t_b = duration_variable(max(float(np.linalg.norm(c - tail)) / 2.5, 0.1))
        x = leg(config.branch_nodes, t_b, guess, gates, config.thrust_acc_max, config.tilt_max)
        opti.subject_to(x[:, 0] == react[:, -1])
        for k in range(config.branch_nodes + 1 - config.entry_nodes, config.branch_nodes + 1):
            v_n = ca.dot(x[3:6, k], normal)
            lateral, vertical = ca.dot(x[3:6, k], rot[:, 1]), ca.dot(x[3:6, k], rot[:, 2])
            opti.subject_to(opti.bounded(-tan_lat * v_n, lateral, tan_lat * v_n))
            opti.subject_to(opti.bounded(-tan_up * v_n, vertical, tan_up * v_n))
        crossing(x[:, -1], gate, normal, gates)
        parts = [(t_b, x)]
        cost += t_b / len(offsets)
        for r in range(1, n_rest):
            a, b = int(base.crossings[p + r - 1]), int(base.crossings[p + r])
            base_leg = base.pos[a : b + 1]
            fade = np.linspace(1.0, 0.0, len(base_leg))[:, None] if r == 1 else 0.0
            guess = _resample(base_leg + (c - centre) * fade, config.branch_nodes + 1)
            t_r = duration_variable(float(base.time[b] - base.time[a]))
            xr = leg(config.branch_nodes, t_r, guess, gates, *limits)
            opti.subject_to(xr[:, 0] == parts[-1][1][:, -1])
            if config.guard_rest and track.unseen[p + r]:
                length = np.linalg.norm(np.diff(guess, axis=0), axis=1)
                remaining = np.r_[np.cumsum(length[::-1])[::-1], 0.0]
                cap = planner.unseen_speed[p + r]
                n_rest_axis = track.normal(p + r)
                rest_rot = track.gates_rot[int(track.sequence[p + r])]
                rest_lat = np.tan(planner.approach_lateral)
                rest_up = np.tan(planner.approach_vertical)
                for k in range(1, config.branch_nodes + 1):
                    v = xr[3:6, k]
                    if cap > 0 and remaining[k] < planner.unseen_length:
                        opti.subject_to(ca.sumsqr(v) <= cap**2)
                    if remaining[k] < planner.approach_length[p + r]:
                        v_n = ca.dot(v, n_rest_axis)
                        side_v, up_v = ca.dot(v, rest_rot[:, 1]), ca.dot(v, rest_rot[:, 2])
                        opti.subject_to(opti.bounded(-rest_lat * v_n, side_v, rest_lat * v_n))
                        opti.subject_to(opti.bounded(-rest_up * v_n, up_v, rest_up * v_n))
            if to_base and r == n_rest - 1:
                opti.subject_to(xr[0:3, -1] == base.pos[b])
                opti.subject_to(xr[3:6, -1] == base.vel[b])
            else:
                crossing(xr[:, -1], int(track.sequence[p + r]), track.normal(p + r), gates)
            parts.append((t_r, xr))
            cost += t_r / len(offsets)
        legs.append(parts)
    opti.minimize(cost)
    opti.solver(
        "ipopt",
        {"print_time": False, "expand": True},
        {
            "max_iter": config.max_iter,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-5,
            "acceptable_tol": 1e-3,
        },
    )
    try:
        sol = opti.solve()
    except RuntimeError as err:
        logger.warning("Branching program failed: %s", err)
        return None

    t0 = float(base.time[first])
    ts = float(sol.value(t_stem))
    stem_x, react_x = np.asarray(sol.value(stem)), np.asarray(sol.value(react))
    pos = [base.pos[: first + 1], stem_x[0:3, 1:].T, react_x[0:3, 1:].T]
    vel = [base.vel[: first + 1], stem_x[3:6, 1:].T, react_x[3:6, 1:].T]
    stamps = [
        base.time[: first + 1],
        t0 + np.linspace(0.0, ts, config.stem_nodes + 1)[1:],
        t0 + ts + np.linspace(0.0, config.reaction, config.reaction_nodes + 1)[1:],
    ]
    stem_end = sum(len(a) for a in pos) - 1
    trajectories = []
    for parts in legs:
        p_all, v_all, t_all = list(pos), list(vel), list(stamps)
        now = t0 + ts + config.reaction
        crossings = list(base.crossings[:p])
        for t_var, x_var in parts:
            dt = float(sol.value(t_var))
            x = np.asarray(sol.value(x_var))
            p_all.append(x[0:3, 1:].T)
            v_all.append(x[3:6, 1:].T)
            t_all.append(now + np.linspace(0.0, dt, x.shape[1])[1:])
            now += dt
            crossings.append(sum(len(a) for a in p_all) - 1)
        if to_base:
            # The rest of the lap is the base plan's.
            b = int(base.crossings[p + n_rest - 1])
            p_all.append(base.pos[b + 1 :])
            v_all.append(base.vel[b + 1 :])
            t_all.append(now + base.time[b + 1 :] - base.time[b])
            shift = crossings[-1] - b
            crossings += [int(k) + shift for k in base.crossings[p + n_rest :]]
        trajectories.append(
            Trajectory(
                pos=np.concatenate(p_all),
                vel=np.concatenate(v_all),
                time=np.concatenate(t_all),
                crossings=np.array(crossings),
            )
        )
    return BranchLibrary(offsets=offsets, trajectories=trajectories, stem_end=stem_end)


def splice(first: Trajectory, last: Trajectory, base: Trajectory, node: int) -> Trajectory:
    """Join two trajectories that both follow the base plan around a node of it.

    Args:
        first: Differs from the base plan only before the node.
        last: Differs from the base plan only after the node.
        base: The base plan.
        node: Node of the base plan where the first ends and the last takes over.

    Returns:
        The first up to the node and the last after it.
    """
    shift = len(first.pos) - len(base.pos)
    n = node + shift
    crossings = np.r_[
        first.crossings[first.crossings <= n], last.crossings[last.crossings > node] + shift
    ]
    return Trajectory(
        pos=np.r_[first.pos[: n + 1], last.pos[node + 1 :]],
        vel=np.r_[first.vel[: n + 1], last.vel[node + 1 :]],
        time=np.r_[first.time[: n + 1], first.time[n] + last.time[node + 1 :] - last.time[node]],
        crossings=crossings,
    )


def plan_branches(
    track: Track, base: Trajectory, planner: PlannerConfig, config: BranchConfig | None = None
) -> BranchLibrary | None:
    """Return the branches of a track, from the disk cache when they were solved before."""
    config = config or BranchConfig()
    blob = repr(
        (
            np.round(base.pos, 4).tobytes(),
            np.round(track.gates_pos, 3).tobytes(),
            np.round(track.poles_pos, 3).tobytes(),
            sorted(asdict(planner).items()),
            sorted(asdict(config).items()),
            PLAN_VERSION,
        )
    ).encode()
    path = cache_dir() / f"branches_{hashlib.sha1(blob).hexdigest()[:20]}.npz"
    if path.exists():
        with np.load(path) as z:
            trajectories = [
                Trajectory(z[f"pos{j}"], z[f"vel{j}"], z[f"time{j}"], z[f"crossings{j}"])
                for j in range(len(z["offsets"]))
            ]
            return BranchLibrary(z["offsets"], trajectories, int(z["stem_end"]))
    t0 = time.perf_counter()
    library = solve_branches(track, base, planner, config)
    logger.info("Branches solved in %.1f s: %s", time.perf_counter() - t0, library is not None)
    if library is None:
        return None
    arrays = {"offsets": library.offsets, "stem_end": library.stem_end}
    for j, t in enumerate(library.trajectories):
        arrays |= {
            f"pos{j}": t.pos,
            f"vel{j}": t.vel,
            f"time{j}": t.time,
            f"crossings{j}": t.crossings,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return library
