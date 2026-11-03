"""Time-optimal point-mass trajectory through the gates of the nominal track.

The plan is the fastest motion of a point mass whose acceleration is a bounded thrust vector minus
gravity and linear drag, through every gate pass in order, clear of every pole and every bar and
stand of every gate. It is solved once per track as one nonlinear program (IPOPT) and cached on
disk: the nominal track is the same in every episode, so in flight the plan only has to be bent to
the poses the sensor reveals (`path_bend`), never solved again.

Per gate pass there is one segment with its own duration ``T_i`` and ``n`` nodes. The state of a
node is ``[p, v, f]`` - position, velocity and the specific thrust vector - and the input is the
jerk ``u = df/dt``, held constant over an interval and integrated with RK4:

    dp/dt = v,   dv/dt = f - g - drag * v,   df/dt = u

The constraints, per node:

* thrust: ``|f| <= thrust_acc_max``, and the tilt ``angle(f, z) <= tilt_max``, so the plan never
  asks for an attitude or a thrust the controller cannot deliver with some left in reserve;
* jerk: ``|u_i| <= jerk_max`` per axis, which bounds how fast the thrust vector turns;
* clearance: horizontal distance to every nearby pole axis, and distance to the centre line of every
  nearby gate capsule, each at least the part's radius plus the drone's reach;
* floor: a climb cone off the start, then a minimum height;
* unseen gates: over the last ``unseen_length`` metres before a gate the drone has not seen yet the
  speed is capped, and over the last ``approach_length`` metres the velocity is held inside a cone
  about the gate axis, so a pose revealed late moves the arrival point and not the arrival
  direction.

At the end of each segment the position lies in the gate plane within ``crossing_lateral`` /
``crossing_vertical`` of the centre, and the velocity points through the gate within
``crossing_angle_max`` of its axis.

IPOPT only finds local optima, so the program is started from several initial guesses - splines
through the gates whose detours around poles and frames are built with different radii - and the
fastest solution wins.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.track_geometry import DRONE_RADIUS, POLE_RADIUS, gate_capsules

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

GRAVITY = np.array([0.0, 0.0, 9.81])
#: Bump when the formulation changes, so cached plans of the old program are not reused.
PLAN_VERSION = "topt-1"


@dataclass(frozen=True)
class PlannerConfig:
    """Limits and options of the time-optimal program.

    Attributes:
        nodes: Nodes per gate-pass segment.
        thrust_acc_max: Bound on the specific thrust ``|f|`` [m/s^2]. The motors give 0.8 N, i.e.
            18.4 m/s^2 for the nominal 43.4 g and 16.5 m/s^2 for the heaviest randomized drone; the
            rest is the controller's reserve.
        tilt_max: Bound on the angle between the thrust and the vertical [rad].
        jerk_max: Bound on each component of ``df/dt`` [m/s^3].
        drag: Linear drag of the airframe [1/s] per axis, identified by
            `scripts/identify_attitude.py`.
        crossing_angle_max: Largest angle between the velocity and the gate axis at a crossing
            [rad].
        crossing_lateral: Largest sideways offset of a crossing from the gate centre [m].
        crossing_vertical: Largest vertical offset of a crossing from ``crossing_height`` [m].
        crossing_height: Height of the aimed crossing above the gate centre [m].
        pole_clearance: Smallest horizontal distance from a pole axis [m].
        frame_clearance: Smallest distance from a gate capsule's surface [m].
        z_min: Minimum height once the drone has left the climb cone [m].
        climb_slope: Over the first third of the first segment the height gain must be at least this
            times the horizontal distance from the start.
        unseen_speed: Speed cap in front of a gate the drone has not seen yet [m/s]. 0 disables.
        unseen_length: Distance before such a gate over which the cap applies [m].
        approach_length: Distance before such a gate over which the arrival cone applies [m].
        approach_lateral: Half angle of the arrival cone sideways [rad].
        approach_vertical: Half angle of the arrival cone vertically [rad].
        guesses: Initial guesses as ``(gate offset, pole detour radius, frame detour radius)``.
        max_iter: IPOPT iteration limit.
    """

    nodes: int = 25
    thrust_acc_max: float = 15.0
    tilt_max: float = 1.0
    jerk_max: float = 150.0
    drag: tuple[float, float, float] = (0.50, 0.50, 0.57)
    crossing_angle_max: float = 0.9
    crossing_lateral: float = 0.03
    crossing_vertical: float = 0.03
    crossing_height: float = 0.01
    pole_clearance: float = DRONE_RADIUS + POLE_RADIUS + 0.03
    frame_clearance: float = DRONE_RADIUS
    z_min: float = 0.15
    climb_slope: float = 0.5
    unseen_speed: float = 2.0
    unseen_length: float = 0.7
    approach_length: float = 0.4
    approach_lateral: float = 0.5
    approach_vertical: float = 0.1
    guesses: tuple[tuple[float, float, float], ...] = field(
        default=((0.3, 0.25, 0.5), (0.5, 0.3, 0.5), (0.2, 0.35, 0.6))
    )
    max_iter: int = 1500


@dataclass
class Trajectory:
    """A planned trajectory, one row per node.

    Attributes:
        pos: Positions ``(m, 3)``.
        vel: Velocities ``(m, 3)``.
        time: Time stamps ``(m,)``; all zero for the fallback guess.
        crossings: Node index of every gate pass ``(n_passes,)``.
        solved: Whether this is an IPOPT solution rather than the fallback guess.
    """

    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    time: NDArray[np.floating]
    crossings: NDArray[np.integer]
    solved: bool = True

    @property
    def duration(self) -> float:
        """Planned time from the first node to the last gate pass."""
        return float(self.time[self.crossings[-1]])


@dataclass(frozen=True)
class Track:
    """What the planner needs to know about the track.

    Attributes:
        gates_pos: Gate centres ``(n_gates, 3)``.
        gates_rot: Gate rotation matrices ``(n_gates, 3, 3)``.
        poles_pos: Pole tops ``(n_poles, 3)``.
        sequence: Gate index of every pass ``(n_passes,)``.
        directions: Passing direction of every pass, ``+1`` or ``-1``.
        unseen: Whether the gate of a pass will still be unseen when the drone gets there.
    """

    gates_pos: NDArray[np.floating]
    gates_rot: NDArray[np.floating]
    poles_pos: NDArray[np.floating]
    sequence: NDArray[np.integer]
    directions: NDArray[np.floating]
    unseen: tuple[bool, ...]

    def normal(self, index: int) -> NDArray[np.floating]:
        """Passing axis of pass ``index``, signed with its direction."""
        return self.gates_rot[self.sequence[index]][:, 0] * self.directions[index]


# region initial guess


def _guess_waypoints(
    track: Track, start: NDArray[np.floating], offset: float, r_pole: float, r_frame: float
) -> NDArray[np.floating]:
    """Return waypoints through the gates, with detours around poles and frames inserted.

    Each pass contributes a point ``offset`` before the gate, the gate centre and a point ``offset``
    behind it. The spline through them is then checked against the poles and frames, and the worst
    violation is fixed by moving or inserting a free waypoint, until nothing is violated.
    """
    points, fixed = [np.asarray(start, dtype=float)], [True]
    for i, g in enumerate(track.sequence):
        n = track.normal(i)
        centre = track.gates_pos[g]
        points += [centre - offset * n, centre, centre + offset * n]
        fixed += [False, True, False]
    points.append(points[-1] + 0.6 * track.normal(len(track.sequence) - 1))
    fixed.append(False)
    waypoints, fixed_arr = np.array(points), np.array(fixed)
    for _ in range(12):
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        if np.any(seg < 1e-3):
            break
        knots = np.r_[0.0, np.cumsum(seg)]
        ss = np.linspace(0.0, knots[-1], max(100, int(knots[-1] / 0.03)))
        pts = CubicSpline(knots, waypoints)(ss)
        worst, at, via = 0.0, None, None
        for pole in track.poles_pos:
            d = np.linalg.norm(pts[:, :2] - pole[:2], axis=1)
            k = int(np.argmin(d))
            if r_pole - d[k] > worst:
                away = (pts[k, :2] - pole[:2]) / max(d[k], 1e-6)
                worst, at, via = r_pole - d[k], k, np.r_[pole[:2] + r_pole * away, pts[k, 2]]
        for centre, rot in zip(track.gates_pos, track.gates_rot, strict=True):
            local = (pts - centre) @ rot
            off = np.abs(local[:, 1:]).max(axis=1)
            bad = np.flatnonzero((np.abs(local[:, 0]) < 0.1) & (off > 0.12) & (off < r_frame))
            for k in bad:
                if r_frame - off[k] <= worst:
                    continue
                y, z = local[k, 1], local[k, 2]
                if z < 0 and abs(z) >= abs(y):  # leaving under the frame: go round the stand
                    y = np.copysign(r_frame, y)
                else:
                    y, z = y * r_frame / off[k], z * r_frame / off[k]
                worst, at = r_frame - off[k], k
                via = centre + rot @ np.array([local[k, 0], y, z])
        if at is None:
            break
        i = int(np.clip(np.searchsorted(knots, ss[at]), 1, len(waypoints) - 1))
        near = i if abs(knots[i] - ss[at]) < abs(knots[i - 1] - ss[at]) else i - 1
        if abs(knots[near] - ss[at]) < 0.05:
            if fixed_arr[near]:
                break
            waypoints[near] = via
        else:
            waypoints = np.insert(waypoints, i, via, axis=0)
            fixed_arr = np.insert(fixed_arr, i, False)
    return waypoints


def _guess_segments(
    track: Track, start: NDArray[np.floating], nodes: int, guess: tuple[float, float, float]
) -> list[NDArray[np.floating]] | None:
    """Split a guess spline at the gate planes into one ``(nodes + 1, 3)`` array per pass."""
    waypoints = _guess_waypoints(track, start, *guess)
    seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.any(seg < 1e-6):
        return None
    knots = np.r_[0.0, np.cumsum(seg)]
    spline = CubicSpline(knots, waypoints)
    ss = np.linspace(0.0, knots[-1], int(knots[-1] / 0.005) + 2)
    pts = spline(ss)
    bounds, i0 = [0.0], 0
    for i, g in enumerate(track.sequence):
        x = (pts[i0:] - track.gates_pos[g]) @ track.normal(i)
        near = np.linalg.norm(pts[i0:] - track.gates_pos[g], axis=1) < 0.3
        hits = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0) & near[:-1])
        if len(hits) == 0:
            return None
        idx = int(hits[0]) + i0
        bounds.append(ss[idx + 1])
        i0 = idx + 1
    return [
        spline(np.linspace(bounds[i], bounds[i + 1], nodes + 1)) for i in range(len(bounds) - 1)
    ]


# endregion

# region program


def _rk4(drag: NDArray[np.floating]) -> ca.Function:
    """One RK4 step of the point-mass dynamics over ``h`` with the jerk held constant."""
    x, u, h = ca.SX.sym("x", 9), ca.SX.sym("u", 3), ca.SX.sym("h")

    def f(state: ca.SX) -> ca.SX:
        return ca.vertcat(state[3:6], state[6:9] - GRAVITY - drag * state[3:6], u)

    k1 = f(x)
    k2 = f(x + h / 2 * k1)
    k3 = f(x + h / 2 * k2)
    k4 = f(x + h * k3)
    return ca.Function("rk4", [x, u, h], [x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])


def solve_time_optimal(
    track: Track,
    start: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    guess: list[NDArray[np.floating]],
    config: PlannerConfig,
) -> Trajectory | None:
    """Solve the time-optimal program from one initial guess.

    Args:
        track: The track.
        start: Position of the first node.
        start_vel: Velocity of the first node.
        guess: Initial positions per segment, see `_guess_segments`.
        config: Limits of the program.

    Returns:
        The trajectory, or None if IPOPT did not converge.
    """
    n, n_seg = config.nodes, len(track.sequence)
    drag = np.asarray(config.drag, dtype=float)
    step = _rk4(drag)
    opti = ca.Opti()
    durations = opti.variable(n_seg)
    states = [opti.variable(9, n + 1) for _ in range(n_seg)]
    inputs = [opti.variable(3, n) for _ in range(n_seg)]
    capsules = gate_capsules(track.gates_pos, track.gates_rot)
    cos_tilt2 = np.cos(config.tilt_max) ** 2
    cos_cross2 = np.cos(config.crossing_angle_max) ** 2
    cost = ca.sum1(durations)
    for i in range(n_seg):
        x, u, h = states[i], inputs[i], durations[i] / n
        opti.subject_to(opti.bounded(0.1, durations[i], 5.0))
        for k in range(n):
            opti.subject_to(x[:, k + 1] == step(x[:, k], u[:, k], h))
            opti.subject_to(opti.bounded(-config.jerk_max, u[:, k], config.jerk_max))
            cost += 1e-6 * ca.sumsqr(u[:, k]) * h
        g = guess[i]
        remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1)[::-1])[::-1], 0.0]
        # Only the objects the guess passes near are constrained; the rest cannot become active.
        near_poles = [
            p for p in track.poles_pos if np.min(np.linalg.norm(g[:, :2] - p[:2], axis=1)) < 0.9
        ]
        near_caps = [
            (a, b, r)
            for gate, a, b, r in capsules
            if np.min(np.linalg.norm(g - track.gates_pos[gate], axis=1)) < 0.9
        ]
        rot = track.gates_rot[track.sequence[i]]
        normal, lateral, up = track.normal(i), rot[:, 1], rot[:, 2]
        for k in range(n + 1):
            p, v, f = x[0:3, k], x[3:6, k], x[6:9, k]
            opti.subject_to(ca.sumsqr(f) <= config.thrust_acc_max**2)
            opti.subject_to(f[2] >= 0.0)
            opti.subject_to(f[2] ** 2 >= cos_tilt2 * ca.sumsqr(f))
            if i == 0 and k <= n // 3:
                opti.subject_to(p[2] >= start[2])
                opti.subject_to(
                    (p[2] - start[2]) ** 2 >= config.climb_slope**2 * ca.sumsqr(p[0:2] - start[0:2])
                )
            else:
                opti.subject_to(p[2] >= config.z_min)
            for pole in near_poles:
                opti.subject_to(ca.sumsqr(p[0:2] - pole[0:2]) >= config.pole_clearance**2)
            for a, b, r in near_caps:
                d = b - a
                t = ca.fmin(ca.fmax(ca.dot(p - a, d) / float(d @ d), 0.0), 1.0)
                opti.subject_to(ca.sumsqr(p - (a + t * d)) >= (r + config.frame_clearance) ** 2)
            if track.unseen[i] and config.unseen_speed > 0 and remaining[k] < config.unseen_length:
                opti.subject_to(ca.sumsqr(v) <= config.unseen_speed**2)
            if track.unseen[i] and remaining[k] < config.approach_length:
                v_n = ca.dot(v, normal)
                tan_lat, tan_up = np.tan(config.approach_lateral), np.tan(config.approach_vertical)
                opti.subject_to(opti.bounded(-tan_lat * v_n, ca.dot(v, lateral), tan_lat * v_n))
                opti.subject_to(opti.bounded(-tan_up * v_n, ca.dot(v, up), tan_up * v_n))
        if i > 0:
            opti.subject_to(x[:, 0] == states[i - 1][:, n])
        rel = x[0:3, n] - (track.gates_pos[track.sequence[i]] + config.crossing_height * up)
        v_end = x[3:6, n]
        opti.subject_to(ca.dot(rel, normal) == 0.0)
        opti.subject_to(
            opti.bounded(-config.crossing_lateral, ca.dot(rel, lateral), config.crossing_lateral)
        )
        opti.subject_to(
            opti.bounded(-config.crossing_vertical, ca.dot(rel, up), config.crossing_vertical)
        )
        opti.subject_to(ca.dot(v_end, normal) >= 0.0)
        opti.subject_to(ca.dot(v_end, normal) ** 2 >= cos_cross2 * ca.sumsqr(v_end))
    opti.subject_to(states[0][0:3, 0] == start)
    opti.subject_to(states[0][3:6, 0] == start_vel)
    opti.minimize(cost)

    for i in range(n_seg):
        g = guess[i]
        length = float(np.sum(np.linalg.norm(np.diff(g, axis=0), axis=1)))
        t_i = max(length / 2.5, 0.3)
        opti.set_initial(states[i][0:3, :], g.T)
        opti.set_initial(states[i][3:6, :], (np.gradient(g, axis=0) / (t_i / n)).T)
        opti.set_initial(states[i][6:9, :], np.tile(GRAVITY, (n + 1, 1)).T)
        opti.set_initial(durations[i], t_i)
    opti.solver(
        "ipopt",
        {"print_time": False, "expand": True},
        {
            "max_iter": config.max_iter,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
        },
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        return None
    pos, vel, stamps, crossings = [], [], [], []
    t0 = 0.0
    for i in range(n_seg):
        x = np.asarray(sol.value(states[i]))
        t_i = float(sol.value(durations[i]))
        first = 0 if i == 0 else 1
        pos.append(x[0:3, first:].T)
        vel.append(x[3:6, first:].T)
        stamps.append(t0 + np.linspace(0.0, t_i, n + 1)[first:])
        t0 += t_i
        crossings.append(sum(len(p) for p in pos) - 1)
    return Trajectory(
        pos=np.concatenate(pos),
        vel=np.concatenate(vel),
        time=np.concatenate(stamps),
        crossings=np.array(crossings),
    )


# endregion

# region cache

_MEMORY_CACHE: dict[str, Trajectory] = {}


def cache_dir() -> Path:
    """Directory of the on-disk plan cache (outside the repository)."""
    return Path(os.environ.get("TOPT_CACHE_DIR", Path(tempfile.gettempdir()) / "lsy_topt_cache"))


def _cache_key(
    track: Track,
    start: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    config: PlannerConfig,
) -> str:
    numbers = np.round(
        np.r_[
            track.gates_pos.ravel(),
            track.gates_rot.ravel(),
            track.poles_pos.ravel(),
            track.sequence,
            track.directions,
            np.asarray(track.unseen, dtype=float),
            start,
            start_vel,
        ],
        3,
    )
    blob = numbers.tobytes() + repr((sorted(asdict(config).items()), PLAN_VERSION)).encode()
    return hashlib.sha1(blob).hexdigest()[:20]


def plan(
    track: Track,
    start: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    config: PlannerConfig | None = None,
) -> Trajectory:
    """Return the time-optimal trajectory of a track, from the cache when it was solved before.

    Every initial guess in ``config.guesses`` is solved and the fastest solution kept. If none
    converges, the first guess is returned at a nominal 2 m/s so that the controller still has a
    path to follow; `Trajectory.solved` is False then.

    Args:
        track: The track.
        start: Position of the first node.
        start_vel: Velocity of the first node.
        config: Limits of the program; the defaults when None.

    Returns:
        The trajectory.
    """
    config = config or PlannerConfig()
    key = _cache_key(track, start, start_vel, config)
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]
    path = cache_dir() / f"{key}.npz"
    if path.exists():
        with np.load(path) as z:
            result = Trajectory(
                pos=z["pos"],
                vel=z["vel"],
                time=z["time"],
                crossings=z["crossings"],
                solved=bool(z["solved"]),
            )
        _MEMORY_CACHE[key] = result
        return result

    t_start = time.perf_counter()
    best: Trajectory | None = None
    for guess_args in config.guesses:
        guess = _guess_segments(track, start, config.nodes, guess_args)
        if guess is None:
            continue
        result = solve_time_optimal(track, start, start_vel, guess, config)
        if result is not None and (best is None or result.duration < best.duration):
            best = result
    if best is None:
        guess = _guess_segments(track, start, config.nodes, config.guesses[0])
        if guess is None:
            raise RuntimeError("Could not build even an initial guess through the gates")
        pos = np.concatenate([guess[0]] + [g[1:] for g in guess[1:]])
        vel = np.gradient(pos, axis=0)
        vel *= 2.0 / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
        best = Trajectory(
            pos=pos,
            vel=vel,
            time=np.zeros(len(pos)),
            crossings=np.arange(1, len(guess) + 1) * config.nodes,
            solved=False,
        )
        logger.warning("Time-optimal planning failed; flying the initial guess at 2 m/s")
    else:
        logger.info(
            "Time-optimal plan: %.2f s (solved in %.1f s)",
            best.duration,
            time.perf_counter() - t_start,
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            pos=best.pos,
            vel=best.vel,
            time=best.time,
            crossings=best.crossings,
            solved=best.solved,
        )
    except OSError:
        pass
    _MEMORY_CACHE[key] = best
    return best


# endregion
