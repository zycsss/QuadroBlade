"""The closed-form segments must be exact, and a plan must keep clear of what the check measures."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.topt_planner import Track
from lsy_drone_racing.utils.fast_runtime_planner import (
    FastPlannerConfig,
    _trajectory,
    _violations,
    accel_headroom,
    axis_accel_for_time,
    axis_min_time,
    build_problem,
    guide_gaps,
    path_clearance,
    plan_fast,
    plan_guide,
    required_thrust,
    segment_state,
    segment_time,
    smooth_velocities,
)
from lsy_drone_racing.utils.race_policy import clearances

LIMITS = jnp.array([15.0, float(np.tan(1.1))])
MOVES = [
    (1.0, 0.0, 0.0),
    (0.5, 0.0, 1.0),
    (2.0, 3.0, 1.0),
    (-1.0, 2.0, 2.0),
    (0.1, 3.0, 3.0),
    (0.0, 1.0, -1.0),
]


def _integrate(first: float, duration: float, v0: float, v1: float) -> tuple[float, float]:
    """Step through the two phases numerically, and return the distance and the final speed."""
    switch = 0.5 * (duration + (v1 - v0) / first) if abs(first) > 1e-6 else duration
    dt = duration / 200000
    t = (np.arange(200000) + 0.5) * dt
    speed = v0 + np.cumsum(np.where(t <= switch, first, -first)) * dt
    return float(speed.sum() * dt), float(speed[-1])


@pytest.mark.parametrize(("distance", "v0", "v1"), MOVES)
def test_shortest_time_uses_exactly_the_limit(distance: float, v0: float, v1: float):
    """At the shortest time the acceleration that fits is the limit itself."""
    accel = 9.0
    t = axis_min_time(jnp.array(distance), jnp.array(v0), jnp.array(v1), jnp.array(accel))
    x = axis_accel_for_time(jnp.array(distance), jnp.array(v0), jnp.array(v1), t)
    assert abs(float(x)) == pytest.approx(accel, rel=1e-3)


@pytest.mark.parametrize(("distance", "v0", "v1"), MOVES)
@pytest.mark.parametrize("stretch", [1.0, 1.5, 3.0])
def test_any_longer_time_reaches_the_end_state(
    distance: float, v0: float, v1: float, stretch: float
):
    """The two phases of the closed form, integrated, cover the distance and end at the speed."""
    t = axis_min_time(jnp.array(distance), jnp.array(v0), jnp.array(v1), jnp.array(9.0))
    duration = float(t) * stretch
    x = float(axis_accel_for_time(jnp.array(distance), jnp.array(v0), jnp.array(v1), duration))
    covered, final = _integrate(x, duration, v0, v1)
    assert covered == pytest.approx(distance, abs=1e-3)
    assert final == pytest.approx(v1, abs=1e-3)


def test_segment_meets_its_ends_without_a_jump():
    """Position and velocity are continuous from one end through the switches to the other."""
    p0, p1 = jnp.array([0.0, 0.0, 0.5]), jnp.array([1.5, -0.7, 1.0])
    v0, v1 = jnp.array([1.0, 0.5, 0.0]), jnp.array([0.0, -2.0, 0.5])
    duration, first = segment_time(p0, p1, v0, v1, LIMITS)
    t = jnp.linspace(0.0, duration, 2001)[:, None]
    pos, vel, _ = segment_state(p0, p1, v0, v1, duration, first, t)
    assert np.allclose(pos[0], p0, atol=1e-4)
    assert np.allclose(pos[-1], p1, atol=1e-4)
    assert np.allclose(vel[0], v0, atol=1e-3)
    assert np.allclose(vel[-1], v1, atol=1e-3)
    assert np.abs(np.diff(np.asarray(pos), axis=0)).max() < 0.01
    assert np.abs(np.diff(np.asarray(vel), axis=0)).max() < 0.05


def test_segment_stays_within_the_thrust_limits():
    """Random segments ask for no more than the limit, apart from what the last stretch leaves."""
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.uniform(-1.5, 1.5, (2000, 3)), dtype=jnp.float32)
    v0 = jnp.asarray(rng.uniform(-4.0, 4.0, (2000, 3)), dtype=jnp.float32)
    v1 = jnp.asarray(rng.uniform(-4.0, 4.0, (2000, 3)), dtype=jnp.float32)
    _, first = segment_time(jnp.zeros((2000, 3)), p, v0, v1, LIMITS)
    headroom = np.asarray(accel_headroom(first, LIMITS))
    assert np.mean(headroom >= 0.95) > 0.99


def test_headroom_of_one_axis():
    """Alone, a horizontal axis may use what is left of the thrust after holding the weight."""
    flat = np.sqrt(15.0**2 - 9.81**2)
    assert float(accel_headroom(jnp.array([flat, 0.0, 0.0]), LIMITS)) == pytest.approx(
        1.0, abs=1e-3
    )
    assert float(accel_headroom(jnp.array([0.0, 0.0, 15.0 - 9.81]), LIMITS)) == pytest.approx(
        1.0, abs=1e-3
    )


def _track(poles: np.ndarray | None = None) -> Track:
    """Return four gates around a loop with the race's order of passes."""
    gates = np.array([[1.0, -0.8, 0.7], [1.5, 0.7, 1.2], [-0.2, 0.9, 0.7], [-1.2, -0.5, 1.2]])
    yaw = np.array([0.5, 1.6, 3.1, -1.6])
    rotations = Rotation.from_euler("z", yaw[:, None]).as_matrix()
    if poles is None:
        poles = np.array(
            [[2.0, -1.0, 1.55], [0.3, -1.2, 1.55], [-2.0, 1.0, 1.55], [2.2, 1.2, 1.55]]
        )
    sequence = np.array([0, 1, 2, 3, 1])
    return Track(gates, rotations, poles, sequence, np.ones(5), tuple(False for _ in sequence))


START = np.array([-1.0, -1.0, 0.02])
START_VEL = np.array([0.0, 0.0, 0.7])


def test_hinge_measures_the_same_distance_as_the_check():
    """The optimiser's clearance is the training check's, or it would push at the wrong shapes."""
    track = _track()
    problem, _, _ = build_problem(track, START, START_VEL, FastPlannerConfig())
    rng = np.random.default_rng(1)
    centers = rng.uniform([-2.5, -1.5, 0.1], [2.5, 1.5, 1.8], (500, 3))
    ours = np.asarray(path_clearance(jnp.asarray(centers, dtype=jnp.float32), problem))
    n = len(centers)
    from types import SimpleNamespace

    from lsy_drone_racing.utils.fast_runtime_planner import _BOXES_CENTER, _BOXES_HALF

    layout = SimpleNamespace(
        gates_pos=np.tile(track.gates_pos, (n, 1, 1)),
        gates_rot=np.tile(track.gates_rot, (n, 1, 1, 1)),
        poles=np.tile(track.poles_pos, (n, 1, 1)),
        boxes_center=_BOXES_CENTER,
        boxes_half=_BOXES_HALF,
    )
    poles, gates = clearances(centers, layout)
    # Far tighter than the 3 cm that matter; 0.0003 mm apart on CPU, 0.1 mm apart on the GPU.
    assert np.allclose(ours, np.minimum(poles.min(axis=1), gates.min(axis=1)), atol=1e-3)


def test_plan_passes_every_gate_in_order_and_clear():
    """A plan goes through the centre of every gate, in order, and touches nothing."""
    track = _track()
    config = FastPlannerConfig(steps=60)
    plan, _ = plan_guide(track, START, START_VEL, config)
    assert plan.clear
    t = plan.trajectory
    assert np.allclose(t.pos[0], START, atol=1e-5)
    assert np.allclose(t.vel[0], START_VEL, atol=1e-5)
    for i, row in enumerate(t.crossings):
        gate = track.gates_pos[track.sequence[i]]
        assert np.linalg.norm(t.pos[row] - gate) < 0.02
        assert t.vel[row] @ track.normal(i) > 0.0
    assert np.all(np.diff(t.crossings) > 0)
    assert np.all(np.diff(t.time) >= 0.0)


def test_refining_goes_around_a_pole_on_the_route():
    """A pole across a leg is routed around, and the check agrees with the plan's own gap."""
    poles = np.array([[1.25, -0.05, 1.55], [0.3, -1.2, 1.55], [-2.0, 1.0, 1.55], [2.2, 1.2, 1.55]])
    track = _track(poles)
    plan, _ = plan_guide(track, START, START_VEL, FastPlannerConfig(steps=60))
    assert plan.clear
    pole_gap, gate_gap = guide_gaps(plan.trajectory, track)
    assert min(pole_gap, gate_gap) >= 0.03


def test_final_check_rejects_a_path_through_a_pole():
    """The check is the last word, whatever the hinge believed."""
    track = _track()
    straight = np.linspace(
        track.poles_pos[0] - [1.0, 0.0, 0.5], track.poles_pos[0] - [0.0, 0.0, 0.5], 200
    )
    trajectory = type(
        "Flight",
        (),
        {
            "pos": straight,
            "vel": np.tile([2.0, 0.0, 0.0], (200, 1)),
            "time": np.linspace(0.0, 0.5, 200),
            "crossings": np.array([199]),
        },
    )
    pole_gap, _ = guide_gaps(trajectory, track)
    assert pole_gap < 0.0


def test_samples_follow_a_segment_that_grows():
    """A short hop flown at speed swings out in a long arc, and must not step over a pole there."""
    config = FastPlannerConfig(steps=0)
    far_poles = np.array([[2.4, 1.4, 1.55]] * 4)
    problem, _, _ = build_problem(_track(far_poles), START, START_VEL, config)
    # All but this hop are padding, as if the optimiser had let one segment take over the path.
    size = config.max_waypoints
    waypoints = np.tile(np.array([0.3, -1.0, 1.0], dtype=np.float32), (size, 1))
    waypoints[0] = [-0.3, -1.0, 1.0]
    velocities = np.zeros((size, 3), dtype=np.float32)
    velocities[0] = [0.0, 5.0, 0.0]
    velocities[1:] = [0.0, -5.0, 0.0]
    active = np.zeros(size - 1, dtype=bool)
    active[0] = True
    duration, first = segment_time(
        jnp.asarray(waypoints[0]),
        jnp.asarray(waypoints[1]),
        jnp.asarray(velocities[0]),
        jnp.asarray(velocities[1]),
        LIMITS,
    )
    tips = segment_state(
        jnp.asarray(waypoints[0]),
        jnp.asarray(waypoints[1]),
        jnp.asarray(velocities[0]),
        jnp.asarray(velocities[1]),
        duration,
        first,
        0.5 * duration,
    )[0]
    assert float(tips[1]) > -0.6  # The arc really reaches far from the short chord.
    pole = np.array([float(tips[0]), float(tips[1]), 1.55], dtype=np.float32)
    problem = problem._replace(
        waypoints=jnp.asarray(waypoints),
        active=jnp.asarray(active),
        poles=jnp.tile(jnp.asarray(pole), (4, 1)),
        gates_pos=problem.gates_pos + 50.0,  # Out of the way.
    )
    statics = (0.04, -10.0, 0.5, jnp.full(3, -50.0), jnp.full(3, 50.0), 0.6)
    _, sampled, _ = _violations(
        jnp.asarray(waypoints), jnp.asarray(velocities), problem, LIMITS, statics
    )
    assert float(sampled[:, 0].max()) > 0.1


def test_required_thrust_of_a_hover_is_the_weight():
    """Standing still needs exactly gravity, straight up."""
    still = type("Still", (), {"acc": np.zeros((3, 3)), "vel": np.zeros((3, 3))})
    size, tilt = required_thrust(still)
    assert np.allclose(size, 9.81)
    assert np.allclose(tilt, 0.0)


def test_trajectory_ends_at_each_waypoint():
    """Rows of a sampled plan land exactly on its passes."""
    waypoints = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [2.0, 1.0, 1.0]], dtype=np.float32)
    velocities = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    duration, first = segment_time(
        jnp.asarray(waypoints[:-1]),
        jnp.asarray(waypoints[1:]),
        jnp.asarray(velocities[:-1]),
        jnp.asarray(velocities[1:]),
        LIMITS,
    )
    flight = _trajectory(
        waypoints, velocities, np.asarray(duration), np.asarray(first), 3, np.array([1, 2])
    )
    assert np.allclose(flight.pos[flight.crossings], waypoints[1:], atol=1e-4)
    assert flight.time[-1] == pytest.approx(float(duration.sum()), abs=1e-5)


def test_retry_only_runs_after_a_failed_check():
    """One clear plan is the end of it."""
    plan, tries = plan_guide(_track(), START, START_VEL, FastPlannerConfig(steps=60))
    assert plan.clear
    assert tries == 1
    single = plan_fast(_track(), START, START_VEL, FastPlannerConfig(steps=60))
    assert single.lap_time == pytest.approx(plan.lap_time, abs=1e-3)  # Single precision.


def test_a_plan_outside_the_arena_is_not_clear():
    """Clearance alone is not enough: leaving the arena ends the race too."""
    small = FastPlannerConfig(steps=30, arena_low=(-0.5, -0.5, 0.0), arena_high=(0.5, 0.5, 2.0))
    plan = plan_fast(_track(), START, START_VEL, small)
    assert not plan.inside
    assert not plan.clear


def test_retries_stop_at_the_budget_and_return_the_last_plan():
    """A track nothing can clear ends after every route, and says so."""
    config = FastPlannerConfig(steps=5)
    plan, tries = plan_guide(_track(), START, START_VEL, config, required_gap=10.0)
    assert not plan.clear
    assert tries == 4


def test_smoothing_turns_a_velocity_step_into_a_ramp():
    """A jump in velocity spreads over the window, and a constant velocity stays as it is."""
    times = np.r_[np.arange(0.0, 0.5, 0.004), np.arange(0.5, 1.0, 0.001)]
    velocities = np.zeros((len(times), 3))
    velocities[:, 0] = np.where(times < 0.5, 1.0, 3.0)
    velocities[:, 1] = 2.0
    smoothed = smooth_velocities(velocities, times, 0.1)
    assert np.allclose(smoothed[:, 1], 2.0)
    assert smoothed[np.searchsorted(times, 0.5), 0] == pytest.approx(2.0, abs=0.1)
    assert smoothed[0, 0] == pytest.approx(1.0)
    assert smoothed[-1, 0] == pytest.approx(3.0)
    assert np.abs(np.diff(smoothed[:, 0])).max() < 0.1
