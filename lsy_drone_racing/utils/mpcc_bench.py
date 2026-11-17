"""Fly the MPCC along simple paths with nothing in the way, to see how much of the drone it uses.

A race lap mixes everything, so a low thrust can come from the plan, the speed caps, the reveals or
the MPCC. Here there is only the MPCC: a line, a turn, a hairpin, a loop, an S-bend, with a speed
limit at the drone's physical limit. Whatever is left unused is the MPCC's own choice.

A primitive's limit comes from a point mass at full thrust, so `Flight.time_ratio` says how much
slower than that the MPCC flew. The point mass ignores rotation and thrust lag, so a ratio of 1 is
not reachable; it is a common ruler for the settings, not a target.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from crazyflow.sim import Sim
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.path_bend import ArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.topt_mpcc import MpccConfig, ToptMPCC

# The same as in `topt_mpcc`, which is only imported when the MPCC flies: training a policy on these
# paths should not need the solver toolchain.
DRAG = np.array([0.50, 0.50, 0.57])
GRAVITY = np.array([0.0, 0.0, 9.81])
MASS = 0.0434
SIM_FREQ = 500
CTRL_FREQ = 50
HEIGHT = 1.2
#: The thrust the motors can give, as an acceleration [m/s^2]: 4 * 0.2 N / 43.4 g.
THRUST_ACC_MAX = 18.4
#: What counts as flying at high thrust [m/s^2].
HIGH_THRUST = 15.0
SPACING = 0.05


def _arc(start: NDArray, heading: float, radius: float, angle: float) -> tuple[NDArray, float]:
    """Return beads of a circular arc, turning left for a positive angle, and the final heading."""
    sign = np.sign(angle)
    centre = start + radius * np.array([-np.sin(heading), np.cos(heading)]) * sign
    n = max(int(abs(angle) * radius / SPACING), 2)
    a = heading - sign * np.pi / 2 + np.linspace(0.0, angle, n + 1)[1:]
    return centre + radius * np.stack((np.cos(a), np.sin(a)), axis=1), heading + angle


def _line(start: NDArray, heading: float, length: float) -> NDArray:
    n = max(int(length / SPACING), 1)
    return start + np.outer(np.linspace(0.0, length, n + 1)[1:], [np.cos(heading), np.sin(heading)])


def _build(pieces: list[tuple]) -> NDArray:
    """Chain ``("line", length)`` and ``("arc", radius, angle)`` pieces into beads."""
    pts, heading = [np.zeros((1, 2))], 0.0
    for piece in pieces:
        if piece[0] == "line":
            pts.append(_line(pts[-1][-1], heading, piece[1]))
        else:
            arc, heading = _arc(pts[-1][-1], heading, piece[1], piece[2])
            pts.append(arc)
    xy = np.concatenate(pts)
    return np.c_[xy, np.full(len(xy), HEIGHT)]


def _climb(length: float, slope: float) -> NDArray:
    s = np.arange(0.0, length, SPACING)
    return np.c_[s * np.cos(slope), np.zeros_like(s), HEIGHT - 0.6 + s * np.sin(slope)]


PRIMITIVES: dict[str, NDArray] = {
    "line": _build([("line", 4.0)]),
    "climb": _climb(3.5, 0.25),
    "turn90": _build([("line", 1.5), ("arc", 0.4, np.pi / 2), ("line", 1.5)]),
    "hairpin": _build([("line", 1.5), ("arc", 0.35, np.pi), ("line", 1.5)]),
    "loop": _build([("line", 1.0), ("arc", 0.5, 4 * np.pi), ("line", 1.0)]),
    "s_bend": _build(
        [("line", 1.0), ("arc", 0.5, np.pi / 2), ("arc", 0.5, -np.pi / 2), ("line", 1.0)]
    ),
}


def _curvature(beads: NDArray) -> NDArray:
    span = 2
    chords = beads[2 * span :] - beads[: -2 * span]
    tangents = chords / np.maximum(np.linalg.norm(chords, axis=1, keepdims=True), 1e-9)
    step = np.linalg.norm(np.diff(beads, axis=0), axis=1).mean()
    kappa = np.linalg.norm(np.diff(tangents, axis=0), axis=1) / step
    return np.r_[np.full(span, kappa[0]), kappa, np.full(span + 1, kappa[-1])]


def speed_profile(beads: NDArray, v_max: float = 7.0) -> tuple[NDArray, float]:
    """Return a point mass's fastest speed at every bead at full thrust, and its time.

    Args:
        beads: The path.
        v_max: Ceiling of the speed [m/s].

    Returns:
        The speed per bead, and the time from standstill to the last bead.
    """
    level = float(np.sqrt(THRUST_ACC_MAX**2 - GRAVITY[2] ** 2))
    kappa = _curvature(beads)
    v = np.minimum(np.sqrt(level / np.maximum(kappa, 1e-3)), v_max)
    ds = np.linalg.norm(np.diff(beads, axis=0), axis=1)
    drag = float(DRAG[0])
    v[0] = 0.0
    for i in range(len(ds)):  # as fast as the leftover acceleration allows
        lateral = v[i] ** 2 * kappa[i]
        along = np.sqrt(max(level**2 - lateral**2, 0.0)) - drag * v[i]
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2.0 * max(along, 0.0) * ds[i], 0.0)))
    for i in reversed(range(len(ds))):  # and braking in time; drag helps here
        lateral = v[i + 1] ** 2 * kappa[i + 1]
        along = np.sqrt(max(level**2 - lateral**2, 0.0)) + drag * v[i + 1]
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * along * ds[i]))
    return v, float(np.sum(2.0 * ds / np.maximum(v[:-1] + v[1:], 0.05)))


def make_tracker(cfg: MpccConfig, beads: NDArray, limit: NDArray) -> ToptMPCC:
    """Return the race controller on a given path, with the plan and the bend taken out."""
    from lsy_drone_racing.control.topt_mpcc import ToptMPCC, get_solver

    c = ToptMPCC.__new__(ToptMPCC)
    c.cfg = cfg
    c._freq = CTRL_FREQ
    c._dt = 1.0 / CTRL_FREQ
    c._randomized = False
    c._solver = get_solver(cfg, c._dt)
    c._solver.reset()
    c._start = beads[0].copy()
    c._tick = 0
    c._airborne = True
    c._s = 0.0
    c._s_remap = None
    c._extra_iterations = 0
    c._warm = False
    c._ds_pred = np.zeros(cfg.horizon + 1)
    c._v_pred = None
    c._dist = np.zeros(3)
    c._thrust_factor = 1.0
    c._f_est = MASS * GRAVITY[2]
    c._f_cmd = c._f_est
    c._acc = np.zeros(2)
    c._failures = 0
    c._last_status = 0
    c._last_ms = 0.0
    c._last_cmd = np.zeros(4)
    c.plan_log = []
    # A straight run-out, so the end of the path is not a wall for the horizon.
    tail = beads[-1] - beads[-2]
    tail = tail / np.linalg.norm(tail)
    extra = np.outer(np.arange(1, 41) * SPACING, tail)
    path = ArcPath(np.r_[beads, beads[-1] + extra])
    c._path = path
    c._tunnel = np.full(len(path.ss), cfg.tunnel_max)
    limit = np.r_[limit, np.full(40, limit[-1])]
    c._vmax = np.minimum(cfg.v_max, np.interp(path.ss, path.knot_s, limit))
    return c


@dataclass
class Flight:
    """What a flight along a primitive did.

    Attributes:
        finished: Whether it reached the end of the path in time, still under control.
        time: Time to the end of the path [s].
        time_ratio: That time over the point mass's.
        error_p95: 95th percentile of the distance to the path [m].
        error_max: Largest distance to the path [m].
        thrust_mean: Mean thrust acceleration, from the measured motion [m/s^2].
        thrust_high: Share of the time above `HIGH_THRUST`.
        tilt_max: Largest tilt [rad].
        height_drop: Largest fall below the path [m].
        failures: Solver failures.
    """

    finished: bool
    time: float
    time_ratio: float
    error_p95: float
    error_max: float
    thrust_mean: float
    thrust_high: float
    tilt_max: float
    height_drop: float
    failures: int


def fly(
    name: str, cfg: MpccConfig | None = None, sim: Sim | None = None, tracker_fn: Any = None
) -> Flight:
    """Fly one primitive from standstill and report how it went.

    Args:
        name: A key of `PRIMITIVES`.
        cfg: The MPCC settings; the defaults when None.
        sim: A simulator to reuse, one world and one drone.
        tracker_fn: Makes the controller from the beads and the speed limit; the MPCC when None.

    Returns:
        The flight's numbers.
    """
    if tracker_fn is None and cfg is None:
        from lsy_drone_racing.control.topt_mpcc import MpccConfig

        cfg = MpccConfig()
    beads = PRIMITIVES[name]
    limit, ideal = speed_profile(beads)
    limit = limit + 0.3  # the limit must not stop the drone from starting, or brake it to zero
    tracker = tracker_fn(beads, limit) if tracker_fn else make_tracker(cfg, beads, limit)
    length = float(np.sum(np.linalg.norm(np.diff(beads, axis=0), axis=1)))
    if sim is None:
        sim = Sim(
            n_worlds=1,
            n_drones=1,
            drone="cf21B_500",
            control="attitude",
            dynamics="first_principles",
            freq=SIM_FREQ,
            attitude_freq=SIM_FREQ,
        )
    sim.reset()
    states = sim.data.states
    sim.data = sim.data.replace(states=states.replace(pos=states.pos.at[0, 0].set(beads[0])))
    sub = SIM_FREQ // CTRL_FREQ
    max_ticks = int(CTRL_FREQ * max(4.0 * ideal, 3.0))
    pos, vel, tilts = [], [], []
    done = False
    tick = 0
    for tick in range(max_ticks):
        s = sim.data.states
        p = np.asarray(s.pos).reshape(3).astype(float)
        v = np.asarray(s.vel).reshape(3).astype(float)
        q = np.asarray(s.quat).reshape(4).astype(float)
        w = np.asarray(s.ang_vel).reshape(3).astype(float)
        pos.append(p)
        vel.append(v)
        tilts.append(float(np.arccos(np.clip(R.from_quat(q).apply([0, 0, 1])[2], -1, 1))))
        if tracker._s >= length:
            done = True
            break
        if tilts[-1] > 1.5 or abs(p[2] - beads[:, 2].mean()) > 1.5 or not np.all(np.isfinite(p)):
            break
        action = tracker.compute_control({"pos": p, "vel": v, "quat": q, "ang_vel": w})
        sim.attitude_control(action.reshape(1, 1, 4).astype(np.float32))
        sim.step(sub)
    pos, vel = np.asarray(pos), np.asarray(vel)
    dv = np.gradient(vel, 1.0 / CTRL_FREQ, axis=0)
    thrust = np.linalg.norm(dv + GRAVITY + DRAG * vel, axis=1)
    moving = np.arange(len(thrust)) > 3
    distances = np.linalg.norm(pos[:, None] - tracker._path(tracker._path.ss)[None], axis=2)
    nearest = distances.argmin(axis=1)
    error = distances.min(axis=1)
    below = tracker._path(tracker._path.ss)[nearest, 2] - pos[:, 2]
    t = tick / CTRL_FREQ
    return Flight(
        finished=done,
        time=t,
        time_ratio=t / ideal,
        error_p95=float(np.percentile(error, 95)),
        error_max=float(error.max()),
        thrust_mean=float(thrust[moving].mean()) if moving.any() else 0.0,
        thrust_high=float(np.mean(thrust[moving] > HIGH_THRUST)) if moving.any() else 0.0,
        tilt_max=float(max(tilts)),
        height_drop=float(below.max()),
        failures=int(tracker._failures),
    )


def with_settings(**overrides: float) -> MpccConfig:
    """Return the default settings with some fields changed."""
    from lsy_drone_racing.control.topt_mpcc import MpccConfig

    return replace(MpccConfig(), **overrides)
