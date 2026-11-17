"""The racing policy's rules must be the race's rules, in both NumPy and JAX."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.track_geometry import GATE_BOXES
from lsy_drone_racing.envs.utils import gate_passed
from lsy_drone_racing.utils.race_policy import (
    DRONE_SPHERE_RADIUS,
    OBS_DIM,
    POLE_RADIUS,
    RaceTrack,
    advance_guide,
    clearances,
    crossed_gate,
    euler_matrix,
    euler_quat,
    race_features,
    smooth_plan,
)


@pytest.fixture
def track() -> RaceTrack:
    """Return a straight guide through two gates, with two poles beside it."""
    beads = np.arange(100)[:, None] * np.array([0.05, 0.0, 0.0]) + np.array([0.0, 0.0, 1.0])
    return RaceTrack(
        gates_pos=np.array([[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]),
        gates_rot=np.stack([np.eye(3), np.eye(3)]),
        poles=np.array([[2.0, 1.0, 1.55], [0.0, -2.0, 1.55]]),
        sequence=np.array([0, 1]),
        directions=np.array([1.0, 1.0]),
        beads=beads,
        speeds=np.ones(100),
        velocities=np.tile([1.0, 0.0, 0.0], (100, 1)),
        quats=np.tile([0.0, 0.0, 0.0, 1.0], (100, 1)),
        crossings=np.array([20, 60]),
        boxes_center=np.array([b[0] for b in GATE_BOXES]),
        boxes_half=np.array([b[1] for b in GATE_BOXES]),
        pos_low=np.array([-3.0, -3.0, 0.0]),
        pos_high=np.array([5.0, 3.0, 2.0]),
        start=np.array([0.0, 0.0, 0.01]),
    )


def test_pole_clearance_is_the_distance_to_the_surface(track: RaceTrack):
    """Beside a pole, the free distance is the gap between the pole and the sphere."""
    poles, _ = clearances(np.array([[2.0, 0.5, 1.0]]), track)
    assert poles[0, 0] == pytest.approx(0.5 - POLE_RADIUS - DRONE_SPHERE_RADIUS)


def test_pole_clearance_counts_the_height_above_its_top(track: RaceTrack):
    """Above a pole's top, the distance runs to the top."""
    poles, _ = clearances(np.array([[2.0, 1.0, 1.85]]), track)
    assert poles[0, 0] == pytest.approx(0.3 - POLE_RADIUS - DRONE_SPHERE_RADIUS)


def test_gate_centre_is_free_and_a_bar_is_a_hit(track: RaceTrack):
    """The centre of the opening clears the bars by 0.2 m less the sphere; a bar itself does not."""
    _, gates = clearances(np.array([[1.0, 0.0, 1.0], [1.0, 0.28, 1.0]]), track)
    assert gates[0, 0] == pytest.approx(0.2 - DRONE_SPHERE_RADIUS)
    assert gates[1, 0] < 0.0


def test_gate_clearance_follows_the_rotation_of_the_gate(track: RaceTrack):
    """A rotated gate has its bars where the rotation puts them."""
    rot = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_matrix()
    turned = track._replace(gates_rot=np.stack([rot, rot]))
    # The left bar of a gate turned by 90 degrees lies along the world's x axis.
    _, gates = clearances(np.array([[1.28, 0.0, 1.0]]), turned)
    assert gates[0, 0] < 0.0


@pytest.mark.parametrize("seed", range(4))
def test_crossing_matches_the_race(seed: int):
    """Random steps through a gate pass or fail exactly as the race says."""
    rng = np.random.default_rng(seed)
    n = 512
    rot = Rotation.from_euler("xyz", rng.uniform(-3.0, 3.0, (n, 3)) * [0.1, 0.1, 1.0])
    gate = rng.uniform(-1.0, 1.0, (n, 3))
    last = gate + rng.uniform(-0.4, 0.4, (n, 3))
    pos = gate + rng.uniform(-0.4, 0.4, (n, 3))
    direction = rng.choice([-1.0, 1.0], n)
    last, pos, gate = (v.astype(np.float32) for v in (last, pos, gate))
    ours = crossed_gate(last, pos, gate, rot.as_matrix().astype(np.float32), direction)
    theirs = [
        bool(
            gate_passed(
                pos[i].astype(np.float32),
                last[i].astype(np.float32),
                gate[i].astype(np.float32),
                rot[i].as_quat().astype(np.float32),
                bool(direction[i] < 0),
                (0.45, 0.45),
            )
        )
        for i in range(n)
    ]
    assert np.array_equal(ours, theirs)
    assert 0 < ours.sum() < n


def test_guide_cannot_be_followed_past_a_gate_that_was_not_passed(track: RaceTrack):
    """Flying around a gate earns no progress along the guide."""
    pos = np.array([[1.5, 1.0, 1.0]])
    index, _ = advance_guide(pos, np.array([18]), np.array([0]), track)
    assert index[0] == track.crossings[0]
    index, _ = advance_guide(pos, np.array([18]), np.array([1]), track)
    assert index[0] > track.crossings[0]


def test_features_agree_between_numpy_and_jax(track: RaceTrack):
    """The compiled features are the ones the policy sees in flight."""
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(4, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    args = (
        rng.normal(1.0, 0.3, (4, 3)).astype(np.float32),
        rng.normal(size=(4, 3)).astype(np.float32),
        quat.astype(np.float32),
        rng.normal(size=(4, 3)).astype(np.float32),
        rng.uniform(-1, 1, (4, 7)).astype(np.float32),
        np.array([0, 10, 50, 99]),
        np.array([0, 1, 1, 1]),
    )
    a = race_features(*args, track)
    jax_track = RaceTrack(*(jnp.asarray(v) for v in track))
    b = race_features(*(jnp.asarray(v) for v in args), jax_track)
    assert a.shape == (4, OBS_DIM)
    assert np.allclose(a, np.asarray(b), atol=1e-4)


def test_euler_conversions_match_scipy():
    """Gates and starts are turned the way the race turns them."""
    rpy = np.random.default_rng(0).uniform(-1.0, 1.0, (64, 3))
    rot = Rotation.from_euler("xyz", rpy)
    assert np.allclose(euler_matrix(rpy), rot.as_matrix())
    assert np.allclose(euler_quat(rpy), rot.as_quat())


def test_a_layout_per_world_matches_the_same_layout_for_all(track: RaceTrack):
    """Giving each drone its own copy of a layout changes nothing."""
    n = 4
    rng = np.random.default_rng(0)
    copies = track._replace(
        gates_pos=np.tile(track.gates_pos, (n, 1, 1)),
        gates_rot=np.tile(track.gates_rot, (n, 1, 1, 1)),
        poles=np.tile(track.poles, (n, 1, 1)),
    )
    center = rng.normal(1.0, 0.3, (n, 3))
    for a, b in zip(clearances(center, track), clearances(center, copies), strict=True):
        assert np.array_equal(a, b)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    args = (center, np.ones((n, 3)), quat, np.zeros((n, 3)), np.zeros((n, 7)))
    args = (*args, np.array([0, 10, 50, 99]), np.array([0, 1, 1, 1]))
    assert np.array_equal(race_features(*args, track), race_features(*args, copies))


def test_each_world_sees_its_own_layout(track: RaceTrack):
    """A moved gate changes the distances and features of its own world only."""
    n = 2
    moved = np.tile(track.gates_pos, (n, 1, 1))
    moved[1, 0] += [0.0, 0.1, 0.0]
    own = track._replace(
        gates_pos=moved,
        gates_rot=np.tile(track.gates_rot, (n, 1, 1, 1)),
        poles=np.tile(track.poles, (n, 1, 1)),
    )
    center = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    _, gates = clearances(center, own)
    assert gates[0, 0] == pytest.approx(0.2 - DRONE_SPHERE_RADIUS)
    assert gates[1, 0] == pytest.approx(0.1 - DRONE_SPHERE_RADIUS)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    x = race_features(
        center,
        np.zeros((n, 3)),
        quat,
        np.zeros((n, 3)),
        np.zeros((n, 7)),
        np.zeros(n, dtype=int),
        np.zeros(n, dtype=int),
        own,
    )
    assert np.allclose(x[0, 53:56], [0.0, 0.0, 0.0])
    assert np.allclose(x[1, 53:56], [0.0, 0.1, 0.0])


def test_smoothing_a_plan_keeps_its_nodes_and_follows_the_curve():
    """A quarter circle stays on its circle between nodes, where straight lines would cut inside."""
    angle = np.linspace(0.0, np.pi / 2, 6)
    speed = np.pi / 2
    plan = SimpleNamespace(
        pos=np.c_[np.cos(angle), np.sin(angle), np.zeros(6)],
        vel=speed * np.c_[-np.sin(angle), np.cos(angle), np.zeros(6)],
        time=angle / speed,
        crossings=np.array([2, 5]),
    )
    smooth = smooth_plan(plan, steps=10)
    assert np.allclose(smooth.pos[::10], plan.pos)
    assert np.allclose(smooth.time[::10], plan.time)
    assert np.array_equal(smooth.crossings, [20, 50])
    assert np.allclose(np.linalg.norm(smooth.pos[:, :2], axis=1), 1.0, atol=1e-3)
    assert np.linalg.norm((plan.pos[0, :2] + plan.pos[1, :2]) / 2) < 0.99
