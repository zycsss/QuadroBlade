"""The scene policy must not be able to tell which track it is on or which way a scene faces."""

from __future__ import annotations

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_scene import _reported_track
from lsy_drone_racing.control.topt_planner import (
    FAST_GUIDE_CONFIG,
    RANDOM_TRACK_CONFIG,
    Track,
    _guess_segments,
    solve_time_optimal,
)
from lsy_drone_racing.control.track_geometry import GATE_BOXES
from lsy_drone_racing.utils.path_policy import ACT_DIM
from lsy_drone_racing.utils.race_policy import (
    DRONE_SPHERE_RADIUS,
    TAKEOFF_LIFT,
    TAKEOFF_SPEED,
    guide_arrays,
)
from lsy_drone_racing.utils.scene_generator import (
    ARENA_HIGH,
    ARENA_LOW,
    GATE_BORDER,
    GATE_GAP,
    POLE_FRAME_GAP,
    POLE_TO_GATE,
    _frame_distance,
    sample_layout,
)
from lsy_drone_racing.utils.scene_policy import (
    OBS_DIM,
    GuideLibrary,
    advance_scene_guide,
    closest_frame_point,
    scene_action,
    scene_clearances,
    scene_features,
    scene_heading,
)


def _library(angle: float = 0.0) -> GuideLibrary:
    """Return a straight guide through two gates in the middle of a very large arena, turned."""
    beads = np.arange(200)[:, None] * np.array([0.05, 0.0, 0.0]) + np.array([-5.0, 0.0, 1.0])
    gates = np.array([[-3.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    poles = np.array([[-2.0, 1.0, 1.55], [-1.0, -0.5, 1.55]])
    turn = Rotation.from_euler("z", angle)
    return GuideLibrary(
        beads=turn.apply(beads)[None],
        speeds=np.ones((1, 200)),
        velocities=turn.apply(np.tile([1.0, 0.0, 0.0], (200, 1)))[None],
        quats=np.tile([0.0, 0.0, 0.0, 1.0], (1, 200, 1)),
        crossings=np.array([[40, 100]]),
        starts=np.array([[-5.0, 0.0, 0.01]]),
        gates_pos=turn.apply(gates)[None],
        gates_rpy=np.array([[[0.0, 0.0, angle], [0.0, 0.0, angle]]]),
        poles=turn.apply(poles)[None],
        sequence=np.array([0, 1]),
        directions=np.array([1.0, 1.0]),
        boxes_center=np.array([b[0] for b in GATE_BOXES]),
        boxes_half=np.array([b[1] for b in GATE_BOXES]),
        pos_low=np.array([-100.0, -100.0, 0.0]),
        pos_high=np.array([100.0, 100.0, 2.0]),
    )


def _features(lib: GuideLibrary, angle: float) -> np.ndarray:
    """Return the features of one drone, placed and moving as the turned scene says."""
    turn = Rotation.from_euler("z", angle)
    pos = turn.apply(np.array([[-2.5, 0.1, 1.1]]))
    vel = turn.apply(np.array([[2.0, 0.3, -0.1]]))
    ang_vel = turn.apply(np.array([[0.1, 0.2, 0.3]]))
    quat = (turn * Rotation.from_euler("xyz", [0.1, -0.2, 0.0])).as_quat()[None]
    gates_rot = Rotation.from_euler("xyz", lib.gates_rpy[0]).as_matrix()
    return scene_features(
        pos,
        vel,
        quat,
        ang_vel,
        np.zeros((1, ACT_DIM)),
        np.array([50]),
        np.array([1]),
        np.array([0]),
        lib.gates_pos,
        gates_rot[None],
        lib.poles,
        lib,
    )


@pytest.mark.parametrize("angle", [0.7, -2.0, np.pi])
def test_a_turned_scene_looks_the_same(angle: float):
    """Whichever way a scene faces, the policy sees the same numbers."""
    a = _features(_library(), 0.0)
    b = _features(_library(angle), angle)
    assert a.shape == (1, OBS_DIM)
    # The walls are far away and clipped in every direction.
    assert np.allclose(a, b, atol=1e-4)


def test_the_output_turns_with_the_frame():
    """A lean along the frame's forward axis is a lean along the guide in the world."""
    u = np.zeros((1, ACT_DIM))
    # Offset the first column downwards: the thrust axis then leans forwards in the frame.
    u[0, 2] = -0.5
    for angle in (0.0, 0.7, -2.0):
        action = scene_action(u, np.array([np.cos(angle)]), np.array([np.sin(angle)]))[0]
        assert action[2] == 0.0
        up = Rotation.from_euler("xyz", [action[0], action[1], 0.0]).as_matrix()[:, 2]
        assert np.allclose(
            up[:2] / np.linalg.norm(up[:2]), [np.cos(angle), np.sin(angle)], atol=1e-5
        )


def test_level_output_flies_level():
    """A network that outputs zero asks for level flight at any heading."""
    action = scene_action(np.zeros((1, ACT_DIM)), np.array([0.6]), np.array([0.8]))[0]
    assert np.allclose(action[:3], 0.0, atol=1e-6)


def test_features_agree_between_numpy_and_jax():
    """The compiled features are the ones the policy sees in flight."""
    lib = _library(0.3)
    a = _features(lib, 0.3)
    turn = Rotation.from_euler("z", 0.3)
    b = scene_features(
        jnp.asarray(turn.apply(np.array([[-2.5, 0.1, 1.1]]))),
        jnp.asarray(turn.apply(np.array([[2.0, 0.3, -0.1]]))),
        jnp.asarray((turn * Rotation.from_euler("xyz", [0.1, -0.2, 0.0])).as_quat()[None]),
        jnp.asarray(turn.apply(np.array([[0.1, 0.2, 0.3]]))),
        jnp.zeros((1, ACT_DIM)),
        jnp.array([50]),
        jnp.array([1]),
        jnp.array([0]),
        jnp.asarray(lib.gates_pos),
        jnp.asarray(Rotation.from_euler("xyz", lib.gates_rpy[0]).as_matrix()[None]),
        jnp.asarray(lib.poles),
        GuideLibrary(*(jnp.asarray(v) for v in lib)),
    )
    assert np.allclose(a, np.asarray(b), atol=1e-3)


def test_closest_frame_point_lies_on_the_frame():
    """The vector to the closest bar is as long as the clearance to it, and points at it."""
    lib = _library()
    rot = Rotation.from_euler("z", 0.5).as_matrix()[None]
    gate = np.array([[0.0, 0.0, 1.0]])
    # In the opening, a little off centre, the closest bar is on the side the drone is nearest to.
    center = gate + (rot[0] @ np.array([0.0, 0.1, 0.0]))[None]
    vector = closest_frame_point(center, gate, rot, lib)
    assert (rot[0].T @ vector[0])[1] > 0.0
    assert np.linalg.norm(vector[0]) == pytest.approx(0.1, abs=1e-6)
    rot_all = np.tile(rot, (1, 2, 1, 1))
    gates_all = np.array([[gate[0], [50.0, 50.0, 1.0]]])
    _, frames = scene_clearances(center, gates_all, rot_all, lib.poles, lib)
    assert frames[0, 0] == pytest.approx(np.linalg.norm(vector[0]) - DRONE_SPHERE_RADIUS, abs=1e-6)


def test_guide_stays_between_the_gates():
    """Flying around a gate earns no progress along the guide."""
    lib = _library()
    pos = np.array([[-2.0, 1.0, 1.0]])
    index, _ = advance_scene_guide(pos, np.array([38]), np.array([0]), np.array([0]), lib)
    assert index[0] == 40


def test_heading_follows_the_guide():
    """The frame points along the guide."""
    lib = _library(0.7)
    c, s = scene_heading(np.array([10]), np.array([0]), lib)
    assert np.allclose([c[0], s[0]], [np.cos(0.7), np.sin(0.7)])


@pytest.mark.parametrize("seed", range(40))
def test_generated_layouts_keep_their_distances(seed: int):
    """Gates and poles stay in the arena, apart, and clear of the frames."""
    layout = sample_layout(np.random.default_rng(seed))
    gates, poles = layout.gates_pos[:, :2], layout.poles[:, :2]
    assert (gates >= ARENA_LOW + GATE_BORDER - 1e-9).all()
    assert (gates <= ARENA_HIGH - GATE_BORDER + 1e-9).all()
    distances = np.linalg.norm(gates[:, None] - gates[None], axis=-1) + 10 * np.eye(4)
    assert distances.min() >= GATE_GAP - 1e-9
    for pole in poles:
        assert np.linalg.norm(gates - pole, axis=1).min() >= POLE_TO_GATE[0] - 1e-9
        for gate, yaw in zip(gates, layout.gates_yaw, strict=True):
            assert _frame_distance(pole, gate, yaw) >= POLE_FRAME_GAP - 1e-9


def test_generator_reaches_what_level2_does():
    """Close poles and short first legs, which level 3's generator never makes, do come up."""
    rng = np.random.default_rng(0)
    near, short = [], []
    for _ in range(200):
        layout = sample_layout(rng)
        gaps = np.linalg.norm(layout.gates_pos[:, None, :2] - layout.poles[None, :, :2], axis=-1)
        near.append(gaps.min(axis=1))
        short.append(np.linalg.norm(layout.gates_pos[0, :2] - layout.start[:2]))
    assert np.percentile(np.concatenate(near), 50) < 0.8
    assert min(short) < 0.8


def test_a_generated_track_is_planned_from_what_the_race_reports():
    """The configuration only holds placeholders on a generated track; a fixed one is used as is."""
    limits = {"pos_limit_low": [-2.5, -1.5, 0.0], "pos_limit_high": [2.5, 1.5, 2.0]}
    track = ConfigDict(
        {
            "randomize": True,
            "gate_order": [1],
            "safety_limits": limits,
            "gates": [],
            "obstacles": [],
        }
    )
    yaw = Rotation.from_euler("z", 0.4)
    obs = {
        "gates_pos": np.array([[1.0, 0.5, 0.7]]),
        "gates_quat": yaw.as_quat()[None],
        "obstacles_pos": np.array([[0.0, 1.0, 1.55]]),
        "pos": np.array([0.1, -0.1, 0.01]),
    }
    reported = _reported_track(obs, track)
    assert np.allclose(reported["gates"][0]["pos"], [1.0, 0.5, 0.7])
    assert np.allclose(reported["gates"][0]["rpy"], [0.0, 0.0, 0.4])
    assert np.allclose(reported["obstacles"][0]["pos"], [0.0, 1.0, 1.55])
    assert np.allclose(reported["drones"][0]["pos"], [0.1, -0.1, 0.01])
    fixed = ConfigDict({"randomize": False})
    assert _reported_track(obs, fixed) is fixed


def test_a_plan_on_a_random_track_keeps_its_stride_and_stays_clear():
    """The solver may not spread its nodes to step over a pole; this is the fix, on a real solve."""
    start, gates, yaw, poles = (
        (layout := sample_layout(np.random.default_rng(3))).start,
        layout.gates_pos,
        layout.gates_yaw,
        layout.poles,
    )
    rot = Rotation.from_euler("xyz", np.c_[np.zeros((4, 2)), yaw]).as_matrix()
    sequence, directions = np.array([0, 1, 2, 3, 1]), np.ones(5)
    track = Track(gates, rot, poles, sequence, directions, (False,) * 5)
    config = RANDOM_TRACK_CONFIG
    first = start + np.array([0.0, 0.0, TAKEOFF_LIFT])
    for guess_args in config.guesses:
        guess = _guess_segments(track, first, config.nodes, guess_args)
        result = guess and solve_time_optimal(
            track, first, np.array([0.0, 0.0, TAKEOFF_SPEED]), guess, config
        )
        if result:
            break
    assert result, "no guess solved this track"
    assert (
        np.linalg.norm(np.diff(result.pos, axis=0), axis=1).max() <= config.node_spacing_max + 1e-3
    )
    beads = guide_arrays(result, smooth=True)["beads"]
    assert np.linalg.norm(beads[:, None, :2] - poles[None, :, :2], axis=-1).min() > 0.1


def test_faster_guides_only_raise_the_limits():
    fast, usual = asdict(FAST_GUIDE_CONFIG), asdict(RANDOM_TRACK_CONFIG)
    assert {k for k in fast if fast[k] != usual[k]} == {"tilt_max", "thrust_acc_max"}
    assert FAST_GUIDE_CONFIG.tilt_max > RANDOM_TRACK_CONFIG.tilt_max
    assert FAST_GUIDE_CONFIG.thrust_acc_max > RANDOM_TRACK_CONFIG.thrust_acc_max
