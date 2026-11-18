"""Fast point-mass guide through a track: closed-form bang-bang segments, a few gradient steps.

`topt_planner` takes seconds per track. This planner gives up its jerk limit and its drag, to spend
milliseconds instead. Between two waypoints every axis moves bang-bang, which has a
closed form, and the slowest axis sets the time of the segment. The other axes use just the
acceleration that makes them arrive at the same moment.

The variables are the free waypoints and the velocity at every waypoint. The gates themselves stay
fixed. The stages of one plan:

1. **Route:** `topt_planner._guess_waypoints`, which already goes around poles and frames.
2. **Speeds:** a search over a grid of speeds along the route's direction at every waypoint.
3. **Refine:** gradient steps on the total time plus a squared hinge on clearance, floor, walls and
   the gate crossing angle. A hinge is zero once clear, so the rest of the path is free to be fast.
4. **Check:** the same check as `scripts/solve_scene_library.py`, on the true collision shapes, and
   that the path stays inside the arena. The hinge is soft, so a plan is only trusted after this.

Stage 2 alone (``steps=0``) gives the cheapest guide. Drag is not part of the segments, so
`required_thrust` measures what a plan would really ask for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.control.topt_planner import PlannerConfig, Track, _guess_waypoints
from lsy_drone_racing.control.track_geometry import GATE_BOXES
from lsy_drone_racing.utils.race_policy import (
    DRONE_SPHERE_HEIGHT,
    DRONE_SPHERE_RADIUS,
    POLE_LENGTH,
    POLE_RADIUS,
    clearances,
    guide_arrays,
)

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray

_GRAVITY = 9.81
_BOXES_CENTER = np.array([b[0] for b in GATE_BOXES], dtype=np.float32)
_BOXES_HALF = np.array([b[1] for b in GATE_BOXES], dtype=np.float32)


@dataclass(frozen=True)
class FastPlannerConfig:
    """Limits and budget of a fast plan.

    Attributes:
        thrust: Limit on the specific thrust [m/s^2]. Below `topt_planner`'s 15, because the
            segments have no drag to pay for.
        tilt: Limit on the angle between the thrust and the vertical [rad].
        clearance: Free distance the hinge asks for from a collision sphere [m]. It is above the
            3 cm of the final check, because the hinge is soft and only looks at samples.
        clearance_tolerance: How far the hinge may be violated for an iterate to count as clear [m].
        crossing_angle_max: Largest angle between the velocity and the gate axis at a pass [rad].
        crossing_height: Height of the pass above the gate centre [m].
        z_min: Least height once clear of the climb cone off the start [m].
        climb_slope: Height gain per horizontal metre near the start.
        arena_low: Lower corner of the room the path must stay in, ``(3,)`` [m].
        arena_high: Upper corner of that room, ``(3,)`` [m].
        speeds: Speed grid of the search of stage 2 [m/s].
        steps: Gradient steps; 0 stops after stage 2.
        weights: Weights of the hinge, raised in equal blocks of steps.
        step_pos: First step size of a waypoint [m].
        step_vel: First step size of a velocity [m/s].
        max_waypoints: Waypoints a plan is padded to, because compiled shapes are fixed.
        samples: Points of the path the hinge looks at, in all.
        guess: The route's ``(gate offset, pole detour radius, frame detour radius)``.
        drop_gate_offsets: Remove the route's points before and behind gates that no detour moved.
        other_guesses: Routes `plan_guide` tries once the first one has failed.
    """

    thrust: float = 14.0
    tilt: float = 1.1
    clearance: float = 0.04
    clearance_tolerance: float = 0.005
    crossing_angle_max: float = 0.9
    crossing_height: float = 0.01
    z_min: float = 0.15
    climb_slope: float = 0.5
    arena_low: tuple[float, float, float] = (-2.5, -1.5, 0.0)
    arena_high: tuple[float, float, float] = (2.5, 1.5, 2.0)
    speeds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0)
    steps: int = 60
    weights: tuple[float, ...] = (100.0, 1000.0, 10000.0)
    step_pos: float = 0.01
    step_vel: float = 0.1
    max_waypoints: int = 32
    samples: int = 320
    guess: tuple[float, float, float] = (0.3, 0.25, 0.5)
    drop_gate_offsets: bool = True
    other_guesses: tuple[tuple[float, float, float], ...] = ((0.5, 0.3, 0.5), (0.2, 0.35, 0.6))


# region closed-form segments


def _sqrt(x: Array) -> Array:
    """Return a square root whose gradient stays finite at and below zero."""
    positive = x > 1e-9
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def axis_min_time(distance: Array, v0: Array, v1: Array, accel: Array) -> Array:
    """Return the shortest time of a move along one axis, at full acceleration with one switch."""
    forward = jnp.where(v1 >= v0, 1.0, -1.0)
    straight = (v1**2 - v0**2) / (2.0 * accel * forward)
    sign = jnp.where(distance - straight >= 0.0, 1.0, -1.0)
    peak = _sqrt(0.5 * (v0**2 + v1**2) + sign * accel * distance)
    return (2.0 * peak - sign * (v0 + v1)) / accel


def axis_accel_for_time(distance: Array, v0: Array, v1: Array, duration: Array) -> Array:
    """Return the signed first-phase acceleration that makes one axis take exactly ``duration``.

    Accelerating at ``x`` for ``t1`` and then at ``-x`` for ``t2`` gives
    ``T^2 x^2 - 4 e x - dv^2 = 0`` with ``e = d - (v0 + v1) T / 2``, and the larger root is the one
    with ``t1, t2 >= 0``.
    """
    excess = distance - 0.5 * (v0 + v1) * duration
    change = v1 - v0
    sign = jnp.where(excess >= 0.0, 1.0, -1.0)
    root = _sqrt(4.0 * excess**2 + duration**2 * change**2)
    return (2.0 * excess + sign * root) / jnp.maximum(duration**2, 1e-6)


def accel_headroom(first: Array, limits: Array) -> Array:
    """Return by how much the accelerations of a segment could be scaled within the thrust limits.

    A bang-bang axis uses both signs, so the vertical part counts against the limit in whichever
    direction hurts: up for the size of the thrust, down for the tilt.

    Args:
        first: First-phase acceleration per axis ``(..., 3)``.
        limits: Thrust limit [m/s^2] and the tangent of the tilt limit ``(2,)``.

    Returns:
        The scale ``(...,)``; below 1 means the segment asks for too much.
    """
    thrust, tan_tilt = limits[0], limits[1]
    flat = (first[..., :2] ** 2).sum(axis=-1)
    vertical = jnp.abs(first[..., 2])
    total = jnp.maximum(flat + vertical**2, 1e-9)
    size = (
        _sqrt(_GRAVITY**2 * vertical**2 + total * (thrust**2 - _GRAVITY**2)) - _GRAVITY * vertical
    ) / total
    tilt = tan_tilt * _GRAVITY / jnp.maximum(_sqrt(flat) + tan_tilt * vertical, 1e-6)
    return jnp.minimum(size, tilt)


def segment_time(
    p0: Array, p1: Array, v0: Array, v1: Array, limits: Array, passes: int = 4
) -> tuple[Array, Array]:
    """Return the time of a segment and the first-phase acceleration of every axis.

    It starts from the time each axis would need with the whole thrust to itself, a lower bound, and
    stretches that until the axes together fit the limits. Stretching only ever goes up, because
    an axis that keeps moving fast cannot take every time above its shortest one: taking a little
    longer means turning back, which can need more than going fast did.

    Args:
        p0: Start position ``(..., 3)``.
        p1: End position ``(..., 3)``.
        v0: Start velocity ``(..., 3)``.
        v1: End velocity ``(..., 3)``.
        limits: Thrust limit [m/s^2] and the tangent of the tilt limit ``(2,)``.
        passes: How many times to stretch.

    Returns:
        The time ``(...,)`` and the accelerations ``(..., 3)``.
    """
    distance = p1 - p0
    alone = _sqrt(limits[0] ** 2 - _GRAVITY**2)
    alone = jnp.stack((alone, alone, limits[0] - _GRAVITY))
    duration = axis_min_time(distance, v0, v1, alone).max(axis=-1)
    for i in range(passes):
        first = axis_accel_for_time(distance, v0, v1, duration[..., None])
        headroom = jnp.minimum(accel_headroom(first, limits), 1.0)
        # Accelerations go with 1/T to 1/T^2. A first gentle step keeps the time near the lower
        # bound, and full ones then get out of the times an axis cannot take. Measured on 20000
        # random segments, this order leaves 99.5% within 5% of the limit, 0.7% above the true time.
        duration = duration / (jnp.sqrt(headroom) if i == 0 else headroom)
    return duration, axis_accel_for_time(distance, v0, v1, duration[..., None])


def segment_state(
    p0: Array, p1: Array, v0: Array, v1: Array, duration: Array, first: Array, t: Array
) -> tuple[Array, Array, Array]:
    """Return position, velocity and acceleration at time ``t`` of a segment; arrays broadcast."""
    safe = jnp.where(jnp.abs(first) < 1e-6, 1e-6, first)
    switch = 0.5 * (duration + (v1 - v0) / safe)
    early = t <= switch
    left = duration - t
    pos = jnp.where(early, p0 + v0 * t + 0.5 * first * t**2, p1 - v1 * left - 0.5 * first * left**2)
    vel = jnp.where(early, v0 + first * t, v1 + first * left)
    return pos, vel, jnp.where(early, first, -first)


# endregion

# region loss


class Problem(NamedTuple):
    """One track as fixed-size arrays. ``K`` is `FastPlannerConfig.max_waypoints`."""

    waypoints: Array  # (K, 3)
    free: Array  # (K,), whether a waypoint may move
    active: Array  # (K - 1,), whether a segment is real rather than padding
    tangents: Array  # (K, 3), direction of the speed search
    gate_normals: Array  # (K, 3), passing axis at a gate waypoint, else zero
    sample_grid: Array  # (N,), even shares of the length of the path
    start: Array  # (3,)
    start_vel: Array  # (3,)
    gates_pos: Array  # (g, 3)
    gates_rot: Array  # (g, 3, 3)
    poles: Array  # (p, 3)


def path_clearance(center: Array, problem: Problem) -> Array:
    """Return the free distance from collision spheres to the nearest object, ``(n,)``.

    The same geometry as `race_policy.clearances`, so what the hinge pushes away from is what the
    final check measures.
    """
    below = jnp.clip(problem.poles[:, 2] - center[:, None, 2], 0.0, POLE_LENGTH)
    axis = jnp.stack(
        (
            jnp.broadcast_to(problem.poles[:, 0], below.shape),
            jnp.broadcast_to(problem.poles[:, 1], below.shape),
            problem.poles[:, 2] - below,
        ),
        axis=-1,
    )
    offset = center[:, None] - axis
    pole = _sqrt((offset**2).sum(axis=-1)) - POLE_RADIUS - DRONE_SPHERE_RADIUS
    local = jnp.einsum("ngj,gji->ngi", center[:, None] - problem.gates_pos, problem.gates_rot)
    q = jnp.abs(local[:, :, None] - _BOXES_CENTER) - _BOXES_HALF
    outside = _sqrt((jnp.maximum(q, 0.0) ** 2).sum(axis=-1))
    box = outside + jnp.minimum(q.max(axis=-1), 0.0)
    gate = box.min(axis=(1, 2)) - DRONE_SPHERE_RADIUS
    return jnp.minimum(pole.min(axis=1), gate)


def _violations(
    waypoints: Array, velocities: Array, problem: Problem, limits: Array, statics: tuple
) -> tuple[Array, Array, Array]:
    """Return the segment times, and how far samples and gate passes break their limits."""
    clearance, z_min, slope, low, high, cos_cross = statics
    p0, p1, v0, v1 = waypoints[:-1], waypoints[1:], velocities[:-1], velocities[1:]
    duration, first = segment_time(p0, p1, v0, v1, limits)
    duration = duration * problem.active
    # Clearance only holds where it is looked at. Samples are shared out by the length a segment has
    # now: a segment that has grown into a long arc would otherwise step over a pole between them.
    switch_speed = jnp.linalg.norm(
        segment_state(p0, p1, v0, v1, duration[:, None], first, 0.5 * duration[:, None])[1], axis=-1
    )
    length = (
        0.25
        * duration
        * (jnp.linalg.norm(v0, axis=-1) + 2.0 * switch_speed + jnp.linalg.norm(v1, axis=-1))
    )
    edges = jax.lax.stop_gradient(jnp.concatenate((jnp.zeros(1), jnp.cumsum(length))))
    along = problem.sample_grid * edges[-1]
    k = jnp.clip(jnp.searchsorted(edges, along, side="right") - 1, 0, length.shape[0] - 1)
    share = (along - edges[k]) / jnp.maximum(edges[k + 1] - edges[k], 1e-6)
    t = (share * duration[k])[:, None]
    pos, _, _ = segment_state(p0[k], p1[k], v0[k], v1[k], duration[k][:, None], first[k], t)
    center = pos + jnp.array([0.0, 0.0, DRONE_SPHERE_HEIGHT])
    # A sample of a padded segment sits on the last gate, which the hinge ignores like any other.
    hit = jnp.maximum(clearance - path_clearance(center, problem), 0.0)
    climb = problem.start[2] + slope * _sqrt(((pos[:, :2] - problem.start[:2]) ** 2).sum(axis=-1))
    floor = jnp.maximum(jnp.minimum(z_min, climb) - pos[:, 2], 0.0)
    walls = jnp.maximum(low - center, 0.0) + jnp.maximum(center - high, 0.0)
    sampled = jnp.stack((hit, floor, walls.max(axis=-1)), axis=-1)
    along = (velocities * problem.gate_normals).sum(axis=-1)
    is_gate = (problem.gate_normals**2).sum(axis=-1) > 0.5
    speed = _sqrt((velocities**2).sum(axis=-1))
    cone = jnp.where(is_gate, jnp.maximum(cos_cross * speed - along, 0.0), 0.0)
    return duration, sampled, cone


def initial_velocities(problem: Problem, limits: Array, speeds: Array) -> Array:
    """Return the fastest velocities along the route's tangents on a grid of speeds, ``(K, 3)``.

    A shortest-path search over one speed per waypoint, so it is exact for that grid and needs no
    starting point. The start keeps its given velocity.
    """
    count = speeds.shape[0]

    def leg(p0: Array, p1: Array, t0: Array, t1: Array, active: Array) -> Array:
        leave = speeds[:, None] * t0
        arrive = speeds[:, None] * t1
        duration, _ = segment_time(p0, p1, leave[:, None], arrive[None], limits)
        return jnp.where(active, duration, jnp.where(jnp.eye(count, dtype=bool), 0.0, 1e6))

    legs = jax.vmap(leg)(
        problem.waypoints[:-1],
        problem.waypoints[1:],
        problem.tangents[:-1],
        problem.tangents[1:],
        problem.active,
    )
    first, _ = segment_time(
        problem.waypoints[0],
        problem.waypoints[1],
        problem.start_vel,
        speeds[:, None] * problem.tangents[1],
        limits,
    )

    def forward(best: Array, matrix: Array) -> tuple[Array, Array]:
        total = best[:, None] + matrix
        return total.min(axis=0), total.argmin(axis=0)

    best, back = jax.lax.scan(forward, first, legs[1:])

    def backward(choice: Array, pointers: Array) -> tuple[Array, Array]:
        previous = pointers[choice]
        return previous, previous

    last = best.argmin()
    _, chosen = jax.lax.scan(backward, last, back, reverse=True)
    chosen = jnp.concatenate((chosen, last[None]))
    velocities = speeds[chosen][:, None] * problem.tangents[1:]
    return jnp.concatenate((problem.start_vel[None], velocities))


# endregion

# region solve


@jax.jit
def _solve(
    problem: Problem,
    limits: Array,
    speeds: Array,
    statics: tuple,
    schedule: Array,
    step_pos: Array,
    step_vel: Array,
    tolerance: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Search the speeds, then take one Adam step per entry of ``schedule``."""
    velocities = initial_velocities(problem, limits, speeds)
    start = jnp.concatenate((problem.waypoints.ravel(), velocities.ravel()))
    moves = jnp.concatenate(
        (
            (problem.free[:, None] * jnp.ones((1, 3))).ravel() * step_pos,
            jnp.ones_like(velocities).at[0].set(0.0).ravel() * step_vel,
        )
    )
    shape = problem.waypoints.shape

    def unpack(z: Array) -> tuple[Array, Array]:
        x = start + z * moves
        return x[: shape[0] * 3].reshape(shape), x[shape[0] * 3 :].reshape(shape)

    def loss(z: Array, weight: Array) -> tuple[Array, tuple[Array, Array]]:
        waypoints, vel = unpack(z)
        duration, sampled, cone = _violations(waypoints, vel, problem, limits, statics)
        total = duration.sum()
        penalty = (sampled**2).sum() + (cone**2).sum()
        worst = jnp.maximum(sampled.max(), cone.max())
        return total + weight * penalty, (total, worst)

    grad = jax.grad(loss, has_aux=True)

    def score(total: Array, worst: Array) -> Array:
        # Any iterate within the tolerance beats any iterate outside it, then the fastest wins.
        return jnp.where(worst <= tolerance, total, 1e3 + worst)

    def step(state: tuple, inputs: tuple[Array, Array]) -> tuple[tuple, None]:
        z, m1, m2, best_z, best_score = state
        weight, index = inputs
        g, (total, worst) = grad(z, weight)
        now = score(total, worst)
        better = now < best_score
        best_z = jnp.where(better, z, best_z)
        best_score = jnp.where(better, now, best_score)
        g = jnp.nan_to_num(g)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g**2
        hat1 = m1 / (1.0 - 0.9 ** (index + 1.0))
        hat2 = m2 / (1.0 - 0.999 ** (index + 1.0))
        z = z - hat1 / (jnp.sqrt(hat2) + 1e-8)
        return (z, m1, m2, best_z, best_score), None

    zeros = jnp.zeros_like(start)
    state = (zeros, zeros, zeros, zeros, jnp.inf)
    indices = jnp.arange(schedule.shape[0], dtype=schedule.dtype)
    state, _ = jax.lax.scan(step, state, (schedule, indices))
    z, *_, best_z, best_score = state
    # The last iterate has not been scored yet.
    _, (total, worst) = loss(z, schedule[-1] if schedule.shape[0] else 0.0)
    final = jnp.where(score(total, worst) < best_score, z, best_z)
    waypoints, vel = unpack(final)
    duration, sampled, cone = _violations(waypoints, vel, problem, limits, statics)
    _, first = segment_time(waypoints[:-1], waypoints[1:], vel[:-1], vel[1:], limits)
    return waypoints, vel, duration, first, sampled.max(), cone.max()


def _route(
    track: Track, start: NDArray, config: FastPlannerConfig
) -> tuple[NDArray, NDArray, NDArray]:
    """Return the route's waypoints up to the last pass, what may move, and each pass's waypoint."""
    points = _guess_waypoints(track, start, *config.guess)
    if config.drop_gate_offsets:
        # The guess puts a point before and behind every gate to shape its spline. Those that no
        # detour had to move only make segments shorter, and each segment brakes on its own.
        offset = config.guess[0]
        keep = np.ones(len(points), dtype=bool)
        for i, g in enumerate(track.sequence):
            for side in (-offset, offset):
                spot = track.gates_pos[g] + side * track.normal(i)
                keep &= np.linalg.norm(points - spot, axis=1) > 1e-9
        points = points[keep]
    passes, cursor = [], 1
    for i, g in enumerate(track.sequence):
        offset = np.linalg.norm(points[cursor:] - track.gates_pos[g], axis=1)
        at = cursor + int(np.argmax(offset < 1e-9))
        passes.append(at)
        points[at] = track.gates_pos[g] + config.crossing_height * track.gates_rot[g][:, 2]
        cursor = at + 1
    points = points[: passes[-1] + 1]
    free = np.ones(len(points), dtype=bool)
    free[0] = False
    free[passes] = False
    return points, free, np.array(passes)


def build_problem(
    track: Track, start: NDArray, start_vel: NDArray, config: FastPlannerConfig
) -> tuple[Problem, NDArray, int]:
    """Return the arrays of one track, its passes' waypoints, and how many waypoints are real.

    Raises:
        ValueError: If the route has more waypoints than ``config.max_waypoints``.
    """
    points, free, passes = _route(track, start, config)
    count = len(points)
    size = config.max_waypoints
    if count > size:
        raise ValueError(f"Route of {count} waypoints, more than the {size} that are compiled")
    pad = size - count
    waypoints = np.r_[points, np.tile(points[-1], (pad, 1))]
    free_all = np.r_[free, np.zeros(pad, dtype=bool)]
    active = np.arange(size - 1) < count - 1

    tangents = np.zeros((size, 3))
    tangents[1 : count - 1] = points[2:] - points[:-2]
    tangents[count - 1 :] = points[-1] - points[-2]
    normals = np.zeros((size, 3))
    for i, at in enumerate(passes):
        normals[at] = track.normal(i)
        tangents[at] = normals[at]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    # The velocity search keeps the last waypoint's speed along padding, which needs one direction.
    tangents[count - 1 :] = tangents[count - 1]

    problem = Problem(
        *(
            jnp.asarray(a)
            for a in (
                waypoints.astype(np.float32),
                free_all,
                active,
                tangents.astype(np.float32),
                normals.astype(np.float32),
                np.linspace(0.0, 1.0, config.samples, dtype=np.float32),
                np.asarray(start, dtype=np.float32),
                np.asarray(start_vel, dtype=np.float32),
                track.gates_pos.astype(np.float32),
                track.gates_rot.astype(np.float32),
                track.poles_pos.astype(np.float32),
            )
        )
    )
    return problem, passes, count


def _schedule(config: FastPlannerConfig) -> Array:
    """Return the hinge's weight at every step, raised in equal blocks."""
    blocks = np.array_split(np.arange(config.steps), len(config.weights))
    return jnp.asarray(
        np.concatenate([np.full(len(b), w) for b, w in zip(blocks, config.weights, strict=True)]),
        dtype=jnp.float32,
    )


def _trajectory(
    waypoints: NDArray,
    velocities: NDArray,
    duration: NDArray,
    first: NDArray,
    count: int,
    passes: NDArray,
    dt: float = 0.004,
) -> SimpleNamespace:
    """Sample the plan in time, with the fields of a `topt_planner.Trajectory`."""
    pos, vel, acc, stamps, ends = [], [], [], [], []
    t0, rows = 0.0, 0
    for k in range(count - 1):
        n = max(int(np.ceil(duration[k] / dt)), 1)
        t = (np.arange(n) * duration[k] / n)[:, None]
        # Plain NumPy: the formulas of `segment_state`, without a compiled call per segment.
        x = np.where(np.abs(first[k]) < 1e-6, 1e-6, first[k])
        early = t <= 0.5 * (duration[k] + (velocities[k + 1] - velocities[k]) / x)
        left = duration[k] - t
        pos.append(
            np.where(
                early,
                waypoints[k] + velocities[k] * t + 0.5 * first[k] * t**2,
                waypoints[k + 1] - velocities[k + 1] * left - 0.5 * first[k] * left**2,
            )
        )
        vel.append(
            np.where(early, velocities[k] + first[k] * t, velocities[k + 1] + first[k] * left)
        )
        acc.append(np.where(early, first[k], -first[k]))
        stamps.append(t0 + t[:, 0])
        t0 += float(duration[k])
        rows += n
        ends.append(rows)
    pos.append(waypoints[count - 1][None])
    vel.append(velocities[count - 1][None])
    acc.append(acc[-1][-1:])
    stamps.append(np.array([t0]))
    return SimpleNamespace(
        pos=np.concatenate(pos, dtype=float),
        vel=np.concatenate(vel, dtype=float),
        acc=np.concatenate(acc, dtype=float),
        time=np.concatenate(stamps, dtype=float),
        crossings=np.array([ends[p - 1] for p in passes]),
    )


# endregion

# region check


def smooth_velocities(velocities: NDArray, times: NDArray, window: float) -> NDArray:
    """Return velocities averaged over a window in time, for a guide's attitude to follow.

    Rows of a plan are not evenly spaced, so the average comes from the integral of the velocity.
    The window shrinks at both ends, where it would leave the plan.

    Args:
        velocities: Velocities ``(n, 3)`` [m/s].
        times: Their times ``(n,)`` [s], not decreasing.
        window: Width of the window [s].

    Returns:
        The averaged velocities ``(n, 3)``.
    """
    steps = np.diff(times)[:, None]
    integral = np.r_[
        np.zeros((1, 3)), np.cumsum(0.5 * (velocities[1:] + velocities[:-1]) * steps, 0)
    ]
    low = np.maximum(times - 0.5 * window, times[0])
    high = np.minimum(times + 0.5 * window, times[-1])
    area = np.stack(
        [
            np.interp(high, times, integral[:, i]) - np.interp(low, times, integral[:, i])
            for i in range(3)
        ],
        axis=1,
    )
    width = (high - low)[:, None]
    return np.where(width > 1e-9, area / np.maximum(width, 1e-9), velocities)


def guide_gaps(trajectory: object, track: Track, smooth: bool = False) -> tuple[float, float]:
    """Return the least free distance of a plan's guide to the poles and to the gates [m].

    The check of `scripts/solve_scene_library.py`, on the same beads a policy would be given.

    Args:
        trajectory: The plan.
        track: The track.
        smooth: Build the guide as `load_library` does. A plan's rows are close enough that this
            only
            matters once its velocities no longer match its positions.

    Returns:
        The least distance to the poles and to the gates.
    """
    guide = guide_arrays(trajectory, smooth=smooth)
    beads = guide["beads"][: guide["crossings"][-1] + 1]
    n = len(beads)
    layout = SimpleNamespace(
        gates_pos=np.tile(track.gates_pos, (n, 1, 1)),
        gates_rot=np.tile(track.gates_rot, (n, 1, 1, 1)),
        poles=np.tile(track.poles_pos, (n, 1, 1)),
        boxes_center=_BOXES_CENTER,
        boxes_half=_BOXES_HALF,
    )
    poles, gates = clearances(beads + [0.0, 0.0, DRONE_SPHERE_HEIGHT], layout)
    return float(poles.min()), float(gates.min())


def required_thrust(
    trajectory: object, drag: tuple[float, ...] | None = None
) -> tuple[NDArray, NDArray]:
    """Return the specific thrust [m/s^2] and the tilt [rad] a plan asks for at each of its rows."""
    drag = np.asarray(PlannerConfig().drag if drag is None else drag)
    thrust = trajectory.acc + np.array([0.0, 0.0, _GRAVITY]) + drag * trajectory.vel
    size = np.linalg.norm(thrust, axis=1)
    return size, np.arccos(np.clip(thrust[:, 2] / np.maximum(size, 1e-9), -1.0, 1.0))


# endregion


@dataclass
class FastPlan:
    """A fast plan and how it went.

    Attributes:
        trajectory: ``pos``, ``vel``, ``acc``, ``time`` and ``crossings``, sampled in time.
        lap_time: Planned time to the last pass [s].
        pole_gap: Least free distance of the guide to the poles [m].
        gate_gap: Least free distance of the guide to the gates [m].
        inside: Whether the path stays inside the arena.
        clear: Whether the guide stays inside and keeps at least ``required_gap`` from everything.
        timing_ms: Milliseconds of ``route``, ``solve``, ``sample`` and ``check``.
    """

    trajectory: SimpleNamespace
    lap_time: float
    pole_gap: float
    gate_gap: float
    inside: bool
    clear: bool
    timing_ms: dict[str, float]


def plan_guide(
    track: Track,
    start: NDArray,
    start_vel: NDArray,
    config: FastPlannerConfig | None = None,
    required_gap: float = 0.03,
) -> tuple[FastPlan, int]:
    """Plan a guide, and try other routes if it fails its check.

    Segments between gates are fast, but a sharp arrival can leave the hinge stuck against a frame,
    so the first retry keeps the route's points before and behind every gate: slower, and more
    reliably clear. A route on the wrong side of something stays stuck whatever the budget, so the
    next ones start from different detours. A hard track takes several full budgets.

    Args:
        track: The track.
        start: Position of the first waypoint.
        start_vel: Velocity at the first waypoint.
        config: Limits and budget of the first try; the defaults if None.
        required_gap: Free distance the final check asks for [m].

    Returns:
        The first clear plan, or the last one if none is, and how many tries it took.
    """
    config = config or FastPlannerConfig()
    tries = [config]
    if config.drop_gate_offsets:
        tries.append(replace(config, drop_gate_offsets=False))
    tries += [replace(config, guess=guess) for guess in config.other_guesses]
    spent = dict.fromkeys(("route", "solve", "sample", "check"), 0.0)
    for count, attempt in enumerate(tries, 1):
        plan = plan_fast(track, start, start_vel, attempt, required_gap)
        spent = {k: spent[k] + plan.timing_ms[k] for k in spent}
        if plan.clear:
            break
    plan.timing_ms = spent
    return plan, count


def plan_fast(
    track: Track,
    start: NDArray,
    start_vel: NDArray,
    config: FastPlannerConfig | None = None,
    required_gap: float = 0.03,
) -> FastPlan:
    """Plan a guide through a track.

    The first call for a given ``config.steps`` compiles for seconds, so make it before the race.

    Args:
        track: The track.
        start: Position of the first waypoint.
        start_vel: Velocity at the first waypoint.
        config: Limits and budget; the defaults if None.
        required_gap: Free distance the final check asks for [m].

    Returns:
        The plan, also when its check failed: look at `FastPlan.clear`.
    """
    config = config or FastPlannerConfig()
    t0 = time.perf_counter()
    problem, passes, count = build_problem(track, start, start_vel, config)
    t1 = time.perf_counter()
    statics = (
        jnp.float32(config.clearance),
        jnp.float32(config.z_min),
        jnp.float32(config.climb_slope),
        jnp.asarray(config.arena_low, dtype=jnp.float32),
        jnp.asarray(config.arena_high, dtype=jnp.float32),
        jnp.float32(np.cos(config.crossing_angle_max)),
    )
    result = _solve(
        problem,
        jnp.asarray([config.thrust, np.tan(config.tilt)], dtype=jnp.float32),
        jnp.asarray(config.speeds, dtype=jnp.float32),
        statics,
        _schedule(config),
        jnp.float32(config.step_pos),
        jnp.float32(config.step_vel),
        jnp.float32(config.clearance_tolerance),
    )
    waypoints, velocities, duration, first, _, _ = jax.block_until_ready(result)
    t2 = time.perf_counter()
    trajectory = _trajectory(
        np.asarray(waypoints),
        np.asarray(velocities),
        np.asarray(duration),
        np.asarray(first),
        count,
        passes,
    )
    t3 = time.perf_counter()
    pole_gap, gate_gap = guide_gaps(trajectory, track)
    inside = bool(
        np.all(trajectory.pos >= config.arena_low) and np.all(trajectory.pos <= config.arena_high)
    )
    t4 = time.perf_counter()
    return FastPlan(
        trajectory=trajectory,
        lap_time=float(trajectory.time[-1]),
        pole_gap=pole_gap,
        gate_gap=gate_gap,
        inside=inside,
        clear=bool(inside and min(pole_gap, gate_gap) >= required_gap),
        timing_ms={
            "route": (t1 - t0) * 1e3,
            "solve": (t2 - t1) * 1e3,
            "sample": (t3 - t2) * 1e3,
            "check": (t4 - t3) * 1e3,
        },
    )


# region runtime guides
#: Settings of the guides `RLScene` plans in flight, and of the libraries its runtime policy was
#: trained on (`scripts/solve_fast_library.py` defaults). Policy and guide belong together.
RUNTIME_GUIDE_CONFIG = FastPlannerConfig(steps=60, thrust=15.0)
#: Width of the smoothing of a runtime guide's velocities [s], against attitude jumps.
RUNTIME_VELOCITY_WINDOW = 0.15


def runtime_plan(
    track: Track,
    start: NDArray,
    start_vel: NDArray,
    config: FastPlannerConfig = RUNTIME_GUIDE_CONFIG,
    velocity_window: float = RUNTIME_VELOCITY_WINDOW,
) -> tuple[SimpleNamespace, FastPlan, int]:
    """Plan a guide as `RLScene` flies it: `plan_guide`, with its velocities smoothed.

    Args:
        track: The track as currently known.
        start: Position of the first waypoint.
        start_vel: Velocity at the first waypoint.
        config: Limits and budget.
        velocity_window: Width of the smoothing of the velocities [s].

    Returns:
        The trajectory for `race_policy.guide_arrays`, the plan with its check, and its tries.
    """
    plan, tries = plan_guide(track, start, start_vel, config)
    t = plan.trajectory
    trajectory = SimpleNamespace(
        pos=t.pos,
        vel=smooth_velocities(t.vel, t.time, velocity_window),
        time=t.time,
        crossings=t.crossings,
    )
    return trajectory, plan, tries
