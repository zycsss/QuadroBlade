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
    BEND_DIM,
    BEND_FADE,
    BEND_POINTS,
    BENDER_DIM,
    OBS_DIM,
    TEACH_AHEAD,
    GuideLibrary,
    advance_scene_guide,
    bend_offset,
    bend_path,
    bender_features,
    closest_frame_point,
    joined_bead,
    make_bend,
    pop_up_poles,
    replanned_gap,
    reveal_order,
    scene_action,
    scene_clearances,
    scene_features,
    scene_heading,
    staged_layout,
    teacher_bend,
    zero_bend,
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


def _features(lib: GuideLibrary, angle: float, bend: object = None) -> np.ndarray:
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
        bend,
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


def test_reveal_order_follows_the_guide():
    beads = np.arange(200)[:, None] * np.array([0.05, 0.0, 0.0]) + np.array([0.0, 0.0, 1.0])
    # Off the bead grid by 2 cm, so that no bead sits exactly at the sensor's range.
    gates = np.array([[8.02, 0.0, 1.0], [3.02, 0.0, 1.0]])
    poles = np.array([[5.02, 0.5, 1.55], [5.02, 3.0, 1.55]])
    order, entry = reveal_order(beads, gates, poles, 0.7)
    # Gate 1 at x 3, the near pole at x 5, gate 0 at x 8, and the far pole never comes into range.
    assert order.tolist() == [1, 2, 0, 3]
    assert entry[:3].tolist() == [47, 91, 147]
    assert entry[3] >= len(beads)


def test_a_stage_knows_the_first_objects_of_the_order():
    nominal = (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
    true = (np.ones((2, 3)), np.full((2, 3), 0.1), np.full((2, 3), 2.0))
    order = np.array([1, 2, 0, 3])
    gates_pos, gates_rpy, poles = staged_layout(nominal, true, order, 2)
    assert gates_pos.tolist() == [[0, 0, 0], [1, 1, 1]]
    assert gates_rpy[1].tolist() == [0.1, 0.1, 0.1] and gates_rpy[0].tolist() == [0, 0, 0]
    assert poles.tolist() == [[2, 2, 2], [0, 0, 0]]
    # The nominal layout itself is left alone.
    assert not nominal[0].any()


def _bend(
    u: np.ndarray, pos: list, vel: list, limit: float = 0.3, xp: object = np, **kw: object
) -> object:
    lib = _library()
    if xp is jnp:
        lib = GuideLibrary(*(jnp.asarray(v) for v in lib))
    return make_bend(
        xp.asarray(u)[None],
        limit,
        xp.asarray([pos], dtype=float),
        xp.asarray([vel], dtype=float),
        xp.array([50]),
        xp.array([0]),
        lib,
        **kw,
    )


def test_no_bend_leaves_the_features_as_they_are():
    lib = _library()
    plain = _features(lib, 0.0)
    assert np.array_equal(_features(lib, 0.0, zero_bend(np.array([50]))), plain)


def test_a_zero_bend_of_a_drone_on_the_guide_is_the_guide():
    # Bead 50 is at x = -2.5; the drone sits on it and flies along the guide.
    bend = _bend(np.zeros(BEND_DIM), [-2.5, 0.0, 1.0], [2.0, 0.0, 0.0])
    beads = np.arange(50, 200)[None]
    assert np.allclose(bend_offset(bend, beads), 0.0)
    assert bend.scale[0] == pytest.approx(1.0)


def test_a_zero_bend_off_the_guide_is_still_the_guide():
    bend = _bend(np.zeros(BEND_DIM), [-2.5, 0.1, 1.05], [2.0, 1.0, 0.0])
    assert np.allclose(bend_offset(bend, np.arange(50, 200)[None]), 0.0)


def test_a_bend_from_the_drone_starts_there_and_leaves_along_its_velocity():
    bend = _bend(np.zeros(BEND_DIM), [-2.5, 0.1, 1.05], [2.0, 1.0, 0.0], from_drone=True)
    offsets = bend_offset(bend, np.arange(50, 70)[None])[0]
    assert offsets[0] == pytest.approx([0.0, 0.1, 0.05])
    # Moving sideways, the bend first leads further that way, then blends back into the guide.
    assert offsets[2, 1] > offsets[0, 1]
    assert offsets[-1] == pytest.approx(np.zeros(3))


def test_a_bend_moves_the_guide_where_the_drone_is_and_what_lies_ahead():
    lib = _library()
    u = np.zeros((1, BEND_DIM))
    u[0, len(BEND_POINTS)] = 1.0 / 3.0  # sideways 0.1 m at the drone's own bead
    pos = np.array([[-2.5, 0.1, 1.0]])
    bend = make_bend(u, 0.3, pos, np.array([[2.0, 0.0, 0.0]]), np.array([50]), np.array([0]), lib)
    first = 3 + 3 + 9 + ACT_DIM
    plain, bent = _features(lib, 0.0), _features(lib, 0.0, bend)
    # `_features` places the drone 0.1 m to the side of the planned guide and 0.1 m above it: the
    # bent guide runs under it.
    assert plain[0, first : first + 3] == pytest.approx([0.0, 0.1, 0.1], abs=1e-6)
    assert bent[0, first : first + 3] == pytest.approx([0.0, 0.0, 0.1], abs=1e-6)
    near = slice(first + 3, first + 6)
    assert np.linalg.norm(bent[0, near] - [0.2, 0.0, 0.0]) < np.linalg.norm(
        plain[0, near] - [0.2, 0.0, 0.0]
    )


def test_control_points_set_the_offsets_and_fade_out():
    count = len(BEND_POINTS)
    u = np.zeros(BEND_DIM)
    u[2] = 1.0  # forward, at the third control point
    u[count + 2] = 1.0  # sideways, at the third
    u[2 * count + 3] = -0.5  # down, at the fourth
    u[-1] = 1.0
    bend = _bend(u, [-2.5, 0.0, 1.0], [1.0, 0.0, 0.0])
    third, fourth = 50 + BEND_POINTS[2], 50 + BEND_POINTS[3]
    offsets = bend_offset(bend, np.array([[50, third, fourth, 50 + BEND_FADE]]))[0]
    assert offsets[0] == pytest.approx(np.zeros(3))
    assert offsets[1] == pytest.approx([0.3, 0.3, 0.0])
    assert offsets[2] == pytest.approx([0.0, 0.0, -0.15])
    assert offsets[3] == pytest.approx(np.zeros(3))
    assert bend.scale[0] == pytest.approx(1.3)


def test_no_offset_exceeds_the_limit():
    bend = _bend(np.full(BEND_DIM, 5.0), [-2.5, 0.0, 1.0], [8.0, 0.0, 0.0], limit=0.2)
    assert np.all(np.abs(bend.points) <= 0.2 + 1e-9)


def test_the_bend_and_the_bender_inputs_agree_between_numpy_and_jax():
    u = np.linspace(-1.0, 1.0, BEND_DIM)
    pos, vel = [-2.4, 0.1, 1.1], [2.0, 0.5, -0.2]
    a = _bend(u, pos, vel, from_drone=True)
    b = _bend(u, pos, vel, xp=jnp, from_drone=True)
    beads = np.arange(40, 160)[None]
    assert np.allclose(
        bend_offset(a, beads), np.asarray(bend_offset(b, jnp.asarray(beads))), atol=1e-5
    )
    lib = _library()
    feats = _features(lib, 0.0, a)
    args = (
        np.asarray([pos]),
        np.array([50]),
        np.array([1]),
        np.array([0]),
        lib,
        u[None],
        np.array([[True, False]]),
        np.array([[False, True]]),
        lib.poles,
    )
    x = bender_features(feats, *args)
    jlib = GuideLibrary(*(jnp.asarray(v) for v in lib))
    jargs = tuple(jnp.asarray(v) if isinstance(v, np.ndarray) else jlib for v in args)
    y = bender_features(jnp.asarray(feats), *jargs)
    assert x.shape == (1, BENDER_DIM)
    assert np.allclose(x, np.asarray(y), atol=1e-4)
    # The flags: the next gate (gate 1, after one pass) is unseen, the one after it too.
    assert x[0, -4:-2].tolist() == [0.0, 0.0]


def _two_guides(beads: np.ndarray, crossings: list[int]) -> GuideLibrary:
    """Return the straight library with a second row: guide ``beads`` passing at ``crossings``."""
    lib = _library()
    return lib._replace(
        beads=np.concatenate((lib.beads, beads[None])),
        speeds=np.ones((2, 200)),
        velocities=np.tile(lib.velocities, (2, 1, 1)),
        quats=np.tile(lib.quats, (2, 1, 1)),
        crossings=np.array([lib.crossings[0], crossings]),
        starts=np.tile(lib.starts, (2, 1)),
        gates_pos=np.tile(lib.gates_pos, (2, 1, 1)),
        gates_rpy=np.tile(lib.gates_rpy, (2, 1, 1)),
        poles=np.tile(lib.poles, (2, 1, 1)),
    )


def _teach(lib: GuideLibrary, reached: int, xp: object = np) -> tuple:
    if xp is jnp:
        lib = GuideLibrary(*(jnp.asarray(v) for v in lib))
    pos, vel = xp.asarray([[-2.5, 0.0, 1.0]]), xp.asarray([[2.0, 0.0, 0.0]])
    index, passed, rows = xp.asarray([50]), xp.asarray([1]), xp.asarray([0])
    return teacher_bend(0.3, pos, vel, index, passed, rows, xp.asarray([reached]), lib)


def test_the_teacher_does_not_bend_a_guide_onto_itself():
    lib = _two_guides(_library().beads[0], [40, 100])
    u, left, clipped = _teach(lib, 0)
    assert np.allclose(u, 0.0) and left[0] == pytest.approx(0.0) and clipped[0] == 0.0


def test_a_drone_joins_the_other_guide_at_its_nearest_bead():
    # The second guide is the first one 10 beads later.
    later = np.concatenate((np.repeat(_library().beads[0, :1], 10, axis=0), _library().beads[0]))
    lib = _two_guides(later[:200], [50, 110])
    pos = np.array([[-2.5, 0.02, 1.0]])  # bead 50 of the first guide, bead 60 of the second
    assert joined_bead(pos, np.array([50]), np.array([1]), np.array([1]), lib).tolist() == [60]


def test_the_teacher_bends_the_guide_onto_a_shifted_one():
    shift = np.array([0.0, 0.1, 0.05])
    lib = _two_guides(_library().beads[0] + shift, [40, 100])
    u, left, clipped = _teach(lib, 1)
    assert clipped[0] == 0.0 and u[0, -1] == 0.0  # the speed scale isn't taught
    assert left[0] == pytest.approx(0.0, abs=1e-6)
    vel, index, rows = np.array([[2.0, 0.0, 0.0]]), np.array([50]), np.array([0])
    pos, passed = np.array([[-2.5, 0.0, 1.0]]), np.array([1])
    gap, cap = replanned_gap(0.3, pos, vel, index, passed, rows, np.array([1]), lib)
    assert np.allclose(gap[0], shift)  # the guide runs along x: its frame is the world's
    # The bend `make_bend` makes carries the guide onto the other, from the drone's bead on.
    bend = make_bend(u, 0.3, pos, vel, index, rows, lib)
    offsets = bend_offset(bend, index[:, None] + TEACH_AHEAD)[0]
    assert np.allclose(offsets, shift, atol=1e-9)
    assert np.allclose(bend_path(u, cap)[0], offsets, atol=1e-9)
    ju, jleft, _ = _teach(lib, 1, jnp)
    assert np.allclose(u, np.asarray(ju), atol=1e-5)
    assert float(jleft[0]) == pytest.approx(0.0, abs=1e-5)


def test_the_frame_follows_the_bent_guide():
    count = len(BEND_POINTS)
    u = np.zeros(BEND_DIM)
    u[count + 2] = 1.0  # 0.3 m to the left at the third control point, 20 beads (1 m) on
    bend = _bend(u, [-2.5, 0.0, 1.0], [2.0, 0.0, 0.0])
    lib = _library()
    c, s = scene_heading(np.array([50]), np.array([0]), lib)
    assert (c[0], s[0]) == pytest.approx((1.0, 0.0))
    c, s = scene_heading(np.array([50]), np.array([0]), lib, bend)
    # The chord of 20 beads now leads 1 m on and 0.3 m to the left.
    assert np.arctan2(s[0], c[0]) == pytest.approx(np.arctan2(0.3, 1.0))


def test_pop_up_poles_move_onto_the_guide_within_the_shift():
    beads = np.c_[np.linspace(0.0, 3.0, 61), np.zeros(61), np.ones(61)]
    # One pole 0.1 m beside the guide, one 0.4 m beside it.
    poles = np.array([[1.0, 0.1, 1.5], [2.0, -0.4, 1.5]])
    shift = np.array([0.15, 0.15, 0.05])
    true = pop_up_poles(beads, poles, shift, np.random.default_rng(0), share=1.0)
    assert abs(true[0, 1]) <= 0.02  # within reach: lands on the guide
    np.testing.assert_allclose(true[1, 1], -0.25)  # out of reach: as close as the shift allows
    assert np.all(np.abs(true - poles) <= shift + 1e-12)
    # No share draws the race's shifts, with the same random numbers as before.
    rng = np.random.default_rng(1)
    expected = poles + np.random.default_rng(1).uniform(-shift, shift, poles.shape)
    np.testing.assert_allclose(pop_up_poles(beads, poles, shift, rng, share=0.0), expected)
