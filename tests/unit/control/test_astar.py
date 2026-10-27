"""Unit tests for the A* route search used by the path planner."""

from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.control.astar import (
    ArenaGrid,
    AstarConfig,
    points_to_segment_distance,
    segment_distance,
)
from lsy_drone_racing.control.path_planning import GateFrames, PathConfig, PathPlanner
from lsy_drone_racing.envs.utils import load_gate_order, load_track
from lsy_drone_racing.utils import load_config

CONFIG = load_config(Path(__file__).parents[3] / "config" / "level2.toml")
EMPTY = np.zeros((0, 3))


def make_obs() -> dict[str, np.ndarray]:
    """Build a level-2 observation dictionary."""
    gates, obstacles, drones = load_track(CONFIG.env.track)
    order, sign = load_gate_order(CONFIG.env.track, len(CONFIG.env.track.gates))
    return {
        "pos": np.asarray(drones.pos[0], dtype=float),
        "vel": np.zeros(3),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gates_pos": np.asarray(gates.pos, dtype=float),
        "gates_quat": np.asarray(gates.quat, dtype=float),
        "obstacles_pos": np.asarray(obstacles.pos, dtype=float),
        "gate_sequence": order,
        "gate_sequence_direction": sign,
    }


def small_config(resolution: float = 0.05) -> AstarConfig:
    """Return a configuration whose arena is the unit box, for the synthetic geometry tests."""
    return AstarConfig(resolution=resolution, bounds_low=(-1.0, -1.0), bounds_high=(1.0, 1.0))


@pytest.mark.unit
def test_distance_helpers():
    """The segment distances must match their closed-form values."""
    distance = points_to_segment_distance(
        np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros(2), np.array([1.0, 0.0])
    )
    np.testing.assert_allclose(distance, [1.0, 0.0], atol=1e-9)
    # Crossing segments touch, parallel ones keep their offset.
    assert (
        segment_distance(
            np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.array([0.0, 1.0])
        )
        == 0.0
    )
    assert segment_distance(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.5]), np.array([1.0, 0.5])
    ) == pytest.approx(0.5)


@pytest.mark.unit
def test_grid_occupies_poles_and_gate_posts():
    """The grid must block the poles and the posts while leaving the opening passable."""
    grid = ArenaGrid(
        np.array([[0.7, 0.7, 0.7]]),
        np.array([[0.0, 0.0, 0.7]]),
        np.array([[0.0, 1.0, 0.0]]),
        small_config(resolution=0.02),
    )
    assert grid.occupied[grid.cell_of(np.array([0.7, 0.7]))], "the pole is not blocked"
    assert not grid.occupied[grid.cell_of(np.array([0.0, 0.0]))], "the opening is blocked"
    assert grid.occupied[grid.cell_of(np.array([0.0, 0.28]))], "the gate post is not blocked"
    assert grid.occupied[grid.cell_of(np.array([0.0, -0.28]))], "the gate post is not blocked"
    # A chord along the gate plane through the centre passes, one through a post does not.
    assert grid.line_is_clear(np.array([-0.3, 0.0]), np.array([0.3, 0.0]))
    assert not grid.line_is_clear(np.array([0.0, 0.0]), np.array([0.0, 0.6]))
    # A chord through the pole does not pass either.
    assert not grid.line_is_clear(np.array([0.2, 0.7]), np.array([1.0, 0.7]))


@pytest.mark.unit
def test_search_routes_around_a_pole():
    """A blocked chord must be replaced by a route that keeps the clearance everywhere."""
    grid = ArenaGrid(
        np.array([[0.0, 0.0, 0.7]]),
        EMPTY,
        EMPTY,
        AstarConfig(resolution=0.05, bounds_low=(-2.0, -1.0), bounds_high=(2.0, 1.0)),
    )
    start, goal = np.array([-1.5, 0.0]), np.array([1.5, 0.0])
    assert not grid.line_is_clear(start, goal), "the test chord should be blocked"
    leg = grid.search(start, goal)
    assert leg is not None, "no route around a single pole"
    np.testing.assert_allclose(leg[0], start, atol=1e-9)
    np.testing.assert_allclose(leg[-1], goal, atol=1e-9)
    assert len(leg) >= 3, "a route around a pole needs intermediate points"
    for begin, end in zip(leg[:-1], leg[1:], strict=True):
        assert grid.line_is_clear(begin, end), "the route cuts through the pole"
    # The route has to leave the blocked circle, and does so by roughly its radius.
    assert np.max(np.abs(leg[:, 1])) >= 0.9 * grid.pole_inflate, "the route does not go around"


@pytest.mark.unit
def test_search_reports_an_unreachable_goal():
    """A sealed wall of poles must yield no route, and opening it must yield one."""
    wall_y = np.arange(-1.0, 1.01, 0.15)
    wall = np.stack([np.zeros_like(wall_y), wall_y, np.full_like(wall_y, 0.7)], axis=1)
    start, goal = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
    sealed = ArenaGrid(wall, EMPTY, EMPTY, small_config())
    assert sealed.search(start, goal) is None, "a sealed wall was crossed"
    # Removing the poles around the middle opens a channel wider than two blocked circles.
    opened = ArenaGrid(wall[np.abs(wall[:, 1]) > 0.5], EMPTY, EMPTY, small_config())
    assert opened.search(start, goal) is not None, "the open channel was not found"


@pytest.mark.unit
def test_route_keeps_the_clearance_on_every_leg():
    """Every segment of every returned route must satisfy the clearance requirement."""
    obs = make_obs()
    frames = GateFrames.from_quaternions(obs["gates_quat"])
    grid = ArenaGrid(
        obs["obstacles_pos"], obs["gates_pos"], frames.y_axes, AstarConfig(resolution=0.05)
    )
    waypoints = [obs["pos"]] + [obs["gates_pos"][gate] for gate in obs["gate_sequence"]]
    for start, goal in zip(waypoints[:-1], waypoints[1:], strict=True):
        leg = grid.search(start[:2], goal[:2])
        assert leg is not None, f"no route from {start} to {goal}"
        for begin, end in zip(leg[:-1], leg[1:], strict=True):
            assert grid.line_is_clear(begin, end), f"route from {start} to {goal} is not clear"


@pytest.mark.unit
def test_planner_routes_around_the_obstacles():
    """Enabling A* must keep the gate passes and improve the clearance."""
    obs = make_obs()
    heuristic_path, heuristic_info = PathPlanner(PathConfig(use_astar=False)).plan(obs, obs["pos"])
    routed_path, routed_info = PathPlanner(PathConfig(use_astar=True)).plan(obs, obs["pos"])
    np.testing.assert_allclose(routed_info.gate_centers, heuristic_info.gate_centers)
    np.testing.assert_allclose(routed_info.gate_centers, obs["gates_pos"][obs["gate_sequence"]])
    routed = routed_path.pos(np.linspace(0.0, routed_path.length, 6000))
    heuristic = heuristic_path.pos(np.linspace(0.0, heuristic_path.length, 6000))
    routed_clearance = PathPlanner.clearance(routed, routed_info.obstacles)
    heuristic_clearance = PathPlanner.clearance(heuristic, heuristic_info.obstacles)
    assert routed_clearance >= PathConfig().safety_distance - 0.01, (
        f"the routed path violates the safety distance: {routed_clearance:.3f} m"
    )
    # Not strictly greater: a chord that already satisfies the clearance is deliberately left
    # alone, so on a track where the heuristic happens to comply everywhere the two are equal.
    assert routed_clearance >= heuristic_clearance, "routing made the clearance worse"
    assert routed_clearance >= heuristic_clearance, "routing made the clearance worse"
