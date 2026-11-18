"""What a learned racing controller sees and must obey: the track, its rules and the features.

Unlike `path_policy`, where the policy's job is to follow a path, here the path is only a hint. The
policy also sees the gates and the obstacles, and it answers for them: it is free to leave the path
as long as it passes the gates in order without touching anything.

Every function takes NumPy or JAX arrays, so that training compiles the same code that flight runs.
The gates and poles of a `RaceTrack` may be one layout for all drones, or have a leading world axis
so that every drone flies its own.
The geometry and the rules follow the race itself:

* a drone collides as a sphere of radius 0.086 m, 0.02 m above its body origin
  (``col_sphere`` in ``crazyflow/drones/cf21B_500.xml``);
* a gate is four bars and a stand, a pole a capsule down from its top (`track_geometry`);
* a gate is passed by crossing its plane in the right direction inside a 0.45 m square
  (`lsy_drone_racing.envs.utils.gate_passed`).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.track_geometry import GATE_BOXES
from lsy_drone_racing.utils.path_policy import _xp, nearest, resample
from lsy_drone_racing.utils.path_policy import features as path_features

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

#: Collision sphere of the drone, in its body frame [m].
DRONE_SPHERE_RADIUS = 0.086
DRONE_SPHERE_HEIGHT = 0.02
#: A pole is a capsule of this radius and length, from its reported top downwards [m].
POLE_RADIUS = 0.015
POLE_LENGTH = 1.6
#: Half size of the square in which a gate crossing counts [m].
GATE_PASS_HALF = 0.225
#: `path_policy.features`, the two next gates, the two nearest poles, the nearest gate, the position
#: and the share of the lap that is done.
OBS_DIM = 53 + 12 + 6 + 4 + 3 + 1
#: Where the guide path keeps going past the last gate, so that the look-ahead has something to see.
RUN_OUT_BEADS = 80
GRAVITY = np.array([0.0, 0.0, 9.81])
#: The guide starts a little above the drone, already climbing, as the MPCC's plan does [m, m/s].
TAKEOFF_LIFT = 0.02
TAKEOFF_SPEED = 0.7


class RaceTrack(NamedTuple):
    """The track, the guide path and the limits, as arrays.

    Attributes:
        gates_pos: Gate centres ``(g, 3)``.
        gates_rot: Gate rotation matrices ``(g, 3, 3)``; the first column is the passing axis.
        poles: Pole tops ``(p, 3)``.
        sequence: Gate of every pass ``(s,)``.
        directions: Direction of every pass, +1 or -1 ``(s,)``.
        beads: Guide path at an even spacing ``(m, 3)``.
        speeds: The plan's speed at every bead ``(m,)``.
        velocities: The plan's velocity at every bead ``(m, 3)``.
        quats: The plan's attitude at every bead at zero yaw ``(m, 4)``.
        crossings: Bead of every pass ``(s,)``.
        boxes_center: Centres of a gate's collision boxes in its frame ``(b, 3)``.
        boxes_half: Half extents of those boxes ``(b, 3)``.
        pos_low: Lower position limit ``(3,)``.
        pos_high: Upper position limit ``(3,)``.
        start: Where the drone starts ``(3,)``.
    """

    gates_pos: NDArray
    gates_rot: NDArray
    poles: NDArray
    sequence: NDArray
    directions: NDArray
    beads: NDArray
    speeds: NDArray
    velocities: NDArray
    quats: NDArray
    crossings: NDArray
    boxes_center: NDArray
    boxes_half: NDArray
    pos_low: NDArray
    pos_high: NDArray
    start: NDArray


def build_track(
    track_cfg: dict,
    spacing: float = 0.05,
    config: object | None = None,
    plan_fn: Callable | None = None,
    smooth: bool | None = None,
) -> RaceTrack:
    """Return a track with its guide path, from the planner's cache when it was solved before.

    Args:
        track_cfg: The ``env.track`` section of a race configuration.
        spacing: Distance between the beads of the guide path; `path_policy` assumes 0.05 m.
        config: The planner's `PlannerConfig`; its defaults if None.
        plan_fn: Plans instead of `topt_planner.plan`, called as
            ``plan_fn(track, start, start_vel, config)``.
        smooth: Build the guide along the curve the plan flies; by default only with a `config`.

    Returns:
        The track.
    """
    # Only here because the planner needs casadi, and only to make the guide.
    from lsy_drone_racing.control.topt_planner import Track, plan
    from lsy_drone_racing.envs.utils import load_gate_order

    gates_pos = np.array([g["pos"] for g in track_cfg["gates"]], dtype=float)
    gates_rot = Rotation.from_euler("xyz", [g["rpy"] for g in track_cfg["gates"]]).as_matrix()
    poles = np.array([o["pos"] for o in track_cfg["obstacles"]], dtype=float)
    sequence, directions = load_gate_order(track_cfg, len(gates_pos))
    sequence = np.asarray(sequence, dtype=int)
    directions = np.asarray(directions, dtype=float)
    start = np.asarray(track_cfg["drones"][0]["pos"], dtype=float)
    planner_track = Track(
        gates_pos, gates_rot, poles, sequence, directions, tuple(False for _ in sequence)
    )
    trajectory = (plan_fn or plan)(
        planner_track,
        start + np.array([0.0, 0.0, TAKEOFF_LIFT]),
        np.array([0.0, 0.0, TAKEOFF_SPEED]),
        config,
    )
    # The policies of the fixed levels were trained on straight lines between nodes.
    guide = guide_arrays(
        trajectory, spacing, smooth=config is not None if smooth is None else smooth
    )
    limits = track_cfg["safety_limits"]
    return RaceTrack(
        gates_pos=gates_pos,
        gates_rot=gates_rot,
        poles=poles,
        sequence=sequence,
        directions=directions,
        **guide,
        boxes_center=np.array([b[0] for b in GATE_BOXES]),
        boxes_half=np.array([b[1] for b in GATE_BOXES]),
        pos_low=np.asarray(limits["pos_limit_low"], dtype=float),
        pos_high=np.asarray(limits["pos_limit_high"], dtype=float),
        start=start,
    )


def smooth_plan(trajectory: object, steps: int = 10) -> SimpleNamespace:
    """Return a plan with points added between its nodes, along the curve it really flies.

    Nodes on a long leg lie up to 35 cm apart, and a straight line between them can cut through a
    pole the plan itself goes around. The curve follows from the positions and velocities at the
    nodes.

    Args:
        trajectory: A `topt_planner.Trajectory`, or anything with its ``pos``, ``vel``, ``time`` and
            ``crossings``.
        steps: Points per interval between nodes.

    Returns:
        The same fields, with ``steps`` times as many rows.
    """
    p, v, t = trajectory.pos, trajectory.vel, trajectory.time
    h = np.diff(t)[:, None, None]
    s = (np.arange(steps) / steps)[None, :, None]
    pos = (
        (2 * s**3 - 3 * s**2 + 1) * p[:-1, None]
        + (s**3 - 2 * s**2 + s) * h * v[:-1, None]
        + (-2 * s**3 + 3 * s**2) * p[1:, None]
        + (s**3 - s**2) * h * v[1:, None]
    )
    vel = (1.0 - s) * v[:-1, None] + s * v[1:, None]
    time = t[:-1, None] + np.diff(t)[:, None] * s[0, :, 0]
    return SimpleNamespace(
        pos=np.r_[pos.reshape(-1, 3), p[-1:]],
        vel=np.r_[vel.reshape(-1, 3), v[-1:]],
        time=np.r_[time.ravel(), t[-1:]],
        crossings=np.asarray(trajectory.crossings) * steps,
    )


def guide_arrays(
    trajectory: object, spacing: float = 0.05, smooth: bool = False
) -> dict[str, NDArray]:
    """Return a solved plan as a guide: beads at an even spacing, with a run-out past the last gate.

    Args:
        trajectory: A `topt_planner.Trajectory`, or anything with its ``pos``, ``vel``, ``time`` and
            ``crossings``.
        spacing: Distance between the beads; `path_policy` assumes 0.05 m.
        smooth: Follow the curve the plan flies, instead of straight lines between its nodes.

    Returns:
        The ``beads``, ``speeds``, ``velocities``, ``quats`` and ``crossings`` of a `RaceTrack`.
    """
    from lsy_drone_racing.control.topt_planner import PlannerConfig

    if smooth:
        trajectory = smooth_plan(trajectory)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(trajectory.pos, axis=0), axis=1))]
    beads = resample(trajectory.pos, spacing)
    grid = np.arange(len(beads)) * spacing
    velocities = np.stack([np.interp(grid, arc, trajectory.vel[:, i]) for i in range(3)], axis=1)
    time = np.interp(grid, arc, trajectory.time)
    crossings = np.round(arc[trajectory.crossings] / spacing).astype(int)
    crossings = np.minimum(crossings, len(beads) - 1)

    # What the plan's attitude must have been: the thrust that gives its acceleration.
    acceleration = np.gradient(velocities, time + np.arange(len(time)) * 1e-9, axis=0)
    thrust = acceleration + GRAVITY + np.asarray(PlannerConfig().drag) * velocities
    thrust /= np.maximum(np.linalg.norm(thrust, axis=1, keepdims=True), 1e-6)
    roll = -np.arcsin(np.clip(thrust[:, 1], -1.0, 1.0))
    pitch = np.arctan2(thrust[:, 0], thrust[:, 2])
    quats = Rotation.from_euler("xyz", np.c_[roll, pitch, np.zeros(len(roll))]).as_quat()

    tangent = beads[-1] - beads[-2]
    tangent /= np.linalg.norm(tangent)
    run_out = beads[-1] + np.outer(np.arange(1, RUN_OUT_BEADS + 1) * spacing, tangent)
    all_velocities = np.r_[velocities, np.tile(velocities[-1], (RUN_OUT_BEADS, 1))]
    return {
        "beads": np.r_[beads, run_out],
        "speeds": np.linalg.norm(all_velocities, axis=1),
        "velocities": all_velocities,
        "quats": np.r_[quats, np.tile(quats[-1], (RUN_OUT_BEADS, 1))],
        "crossings": crossings,
    }


def euler_matrix(rpy: NDArray) -> NDArray:
    """Return the rotation matrices of ``Rotation.from_euler("xyz", rpy)``, ``(..., 3, 3)``."""
    xp = _xp(rpy)
    cr, sr = xp.cos(rpy[..., 0]), xp.sin(rpy[..., 0])
    cp, sp = xp.cos(rpy[..., 1]), xp.sin(rpy[..., 1])
    cy, sy = xp.cos(rpy[..., 2]), xp.sin(rpy[..., 2])
    rows = (
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
    return xp.stack(rows, axis=-1).reshape((*rpy.shape[:-1], 3, 3))


def euler_quat(rpy: NDArray) -> NDArray:
    """Return the ``xyzw`` quaternions of ``Rotation.from_euler("xyz", rpy)``, ``(..., 4)``."""
    xp = _xp(rpy)
    cr, sr = xp.cos(rpy[..., 0] / 2), xp.sin(rpy[..., 0] / 2)
    cp, sp = xp.cos(rpy[..., 1] / 2), xp.sin(rpy[..., 1] / 2)
    cy, sy = xp.cos(rpy[..., 2] / 2), xp.sin(rpy[..., 2] / 2)
    return xp.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def _layout(track: RaceTrack, n: int) -> tuple[NDArray, NDArray, NDArray]:
    """Return the gate positions, gate rotations and poles of ``n`` drones, with a world axis."""
    xp = _xp(track.gates_pos)
    gates, rotations, poles = track.gates_pos, track.gates_rot, track.poles
    if gates.ndim == 2:
        gates = xp.broadcast_to(gates, (n, *gates.shape))
        rotations = xp.broadcast_to(rotations, (n, *rotations.shape))
        poles = xp.broadcast_to(poles, (n, *poles.shape))
    return gates, rotations, poles


def body_center(pos: NDArray, quat: NDArray) -> NDArray:
    """Return the centre of the collision sphere of drones at ``pos`` with ``xyzw`` ``quat``."""
    xp = _xp(pos)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    up = xp.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), axis=-1)
    return pos + DRONE_SPHERE_HEIGHT * up


def clearances(center: NDArray, track: RaceTrack) -> tuple[NDArray, NDArray]:
    """Return the free distance from collision spheres to every pole and gate; below 0 is a hit.

    Args:
        center: Sphere centres ``(n, 3)``.
        track: The track.

    Returns:
        Distances to the poles ``(n, p)`` and to the gates ``(n, g)`` [m].
    """
    xp = _xp(center)
    gates_pos, gates_rot, pole_tops = _layout(track, center.shape[0])
    below = xp.clip(pole_tops[:, :, 2] - center[:, None, 2], 0.0, POLE_LENGTH)
    axis = xp.stack((pole_tops[:, :, 0], pole_tops[:, :, 1], pole_tops[:, :, 2] - below), axis=-1)
    poles = xp.linalg.norm(center[:, None] - axis, axis=-1) - POLE_RADIUS - DRONE_SPHERE_RADIUS

    # A signed distance to every box of every gate, in the frame of the gate.
    local = xp.einsum("ngj,ngji->ngi", center[:, None] - gates_pos, gates_rot)
    q = xp.abs(local[:, :, None] - track.boxes_center[None, None]) - track.boxes_half[None, None]
    outside = xp.linalg.norm(xp.maximum(q, 0.0), axis=-1)
    inside = xp.minimum(q.max(axis=-1), 0.0)
    gates = (outside + inside).min(axis=-1) - DRONE_SPHERE_RADIUS
    return poles, gates


def crossed_gate(
    last: NDArray, pos: NDArray, gate_pos: NDArray, gate_rot: NDArray, direction: NDArray
) -> NDArray:
    """Return whether drones passed their gates on this step, as the race decides it.

    Args:
        last: Positions before the step ``(n, 3)``.
        pos: Positions after the step ``(n, 3)``.
        gate_pos: Centre of every drone's gate ``(n, 3)``.
        gate_rot: Rotation of every drone's gate ``(n, 3, 3)``.
        direction: Direction every drone must pass in, +1 or -1 ``(n,)``.

    Returns:
        Booleans ``(n,)``.
    """
    xp = _xp(pos)
    before = xp.einsum("nj,nji->ni", last - gate_pos, gate_rot)
    after = xp.einsum("nj,nji->ni", pos - gate_pos, gate_rot)
    forward = direction > 0
    plane = xp.where(
        forward, (before[:, 0] < 0) & (after[:, 0] > 0), (before[:, 0] > 0) & (after[:, 0] < 0)
    )
    span = after[:, 0] - before[:, 0]
    alpha = -before[:, 0] / xp.where(span == 0, 1.0, span)
    y = alpha * after[:, 1] + (1 - alpha) * before[:, 1]
    z = alpha * after[:, 2] + (1 - alpha) * before[:, 2]
    return plane & (xp.abs(y) < GATE_PASS_HALF) & (xp.abs(z) < GATE_PASS_HALF)


def advance_guide(
    pos: NDArray, index: NDArray, passed: NDArray, track: RaceTrack
) -> tuple[NDArray, NDArray]:
    """Move every drone's place on the guide, which stays between its last and its next gate.

    The bound means that the guide cannot be followed past a gate that was not passed, so progress
    along it cannot be earned by flying around a gate.

    Args:
        pos: Positions ``(n, 3)``.
        index: Current beads ``(n,)``.
        passed: Gates passed so far ``(n,)``.
        track: The track.

    Returns:
        The new beads and the distance to them.
    """
    xp = _xp(pos)
    count = track.sequence.shape[0]
    rows = xp.zeros(pos.shape[0], dtype=index.dtype)
    new, _ = nearest(pos, track.beads[None], index, rows)
    low = xp.where(passed > 0, track.crossings[xp.maximum(passed - 1, 0)], 0)
    high = track.crossings[xp.minimum(passed, count - 1)]
    new = xp.clip(new, low, high)
    return new, xp.linalg.norm(track.beads[new] - pos, axis=-1)


def race_features(
    pos: NDArray,
    vel: NDArray,
    quat: NDArray,
    ang_vel: NDArray,
    previous: NDArray,
    index: NDArray,
    passed: NDArray,
    track: RaceTrack,
) -> NDArray:
    """Return the policy's input, ``(n, OBS_DIM)``.

    `track` holds what the drone knows of the gates and poles, which is what the race's sensor shows
    it, and not necessarily where they really are.
    """
    xp = _xp(pos)
    n = pos.shape[0]
    worlds = xp.arange(n)
    rows = xp.zeros(n, dtype=index.dtype)
    guide = path_features(
        pos, vel, quat, ang_vel, previous, track.beads[None], track.speeds[None], index, rows
    )
    count = track.sequence.shape[0]
    next_passes = (xp.minimum(passed, count - 1), xp.minimum(passed + 1, count - 1))
    gates_pos, gates_rot, pole_tops = _layout(track, n)
    gates = []
    for i in next_passes:
        gate = track.sequence[i]
        gates.append(gates_pos[worlds, gate] - pos)
        gates.append(gates_rot[worlds, gate][:, :, 0] * track.directions[i][:, None])
    poles, frames = clearances(body_center(pos, quat), track)
    order = xp.argsort(poles, axis=1)[:, :2]
    pole_rel = (pole_tops[worlds[:, None], order][..., :2] - pos[:, None, :2]).reshape(n, 4)
    pole_gap = xp.take_along_axis(poles, order, axis=1)
    near = frames.argmin(axis=1)
    done = (passed / count)[:, None]
    return xp.concatenate(
        [
            guide,
            *gates,
            pole_rel,
            pole_gap,
            gates_pos[worlds, near] - pos,
            frames[worlds, near][:, None],
            pos,
            done,
        ],
        axis=1,
    ).astype(np.float32)
