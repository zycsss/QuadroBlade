"""How `RLScene` plans its guide again when the sensor reveals something."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ml_collections import ConfigDict

from lsy_drone_racing.control.rl_scene import (
    SHIFTED_RACE_DEFAULTS,
    RLScene,
    _controller_settings,
    _index_between_gates,
    _reported_track,
    _revealed,
)
from lsy_drone_racing.utils.race_policy import RUN_OUT_BEADS


def _obs(gates_seen: tuple[bool, ...], poles_seen: tuple[bool, ...]) -> dict:
    return {
        "pos": np.array([0.1, 0.2, 0.05]),
        "gates_pos": np.array([[1.0, 0.0, 0.7], [2.0, 1.0, 1.2]]),
        "gates_quat": np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        "obstacles_pos": np.array([[0.5, 0.5, 1.55]]),
        "gates_visited": np.array(gates_seen),
        "obstacles_visited": np.array(poles_seen),
    }


def _straight_library(crossings: list[int]) -> SimpleNamespace:
    beads = np.arange(100)[:, None] * np.array([0.05, 0.0, 0.0])
    return SimpleNamespace(beads=beads[None], crossings=np.array([crossings]))


def test_revealed_lists_gates_then_poles():
    assert _revealed(_obs((True, False), (True,))) == (True, False, True)


def test_a_new_guide_is_joined_between_the_last_and_the_next_pass():
    lib = _straight_library([30, 60, 90])
    # Nearest overall would be bead 70, but the next pass is the second, so it stays at 60 or less.
    assert _index_between_gates(lib, np.array([3.5, 0.0, 0.0]), passed=1)[0] == 60
    assert _index_between_gates(lib, np.array([2.0, 0.1, 0.0]), passed=1)[0] == 40
    assert _index_between_gates(lib, np.array([2.0, 0.0, 0.0]), passed=0)[0] == 30


def test_replanning_needs_the_runtime_planner():
    config = ConfigDict({"controller": {"replan_on_reveal": True}, "env": {"track": {}}})
    with pytest.raises(ValueError, match="replan_on_reveal"):
        RLScene(_obs((False, False), (False,)), {}, config)
    config = ConfigDict({"controller": {"guide_planner": "spline"}, "env": {"track": {}}})
    with pytest.raises(ValueError, match="guide_planner"):
        RLScene(_obs((False, False), (False,)), {}, config)


def _scene(monkeypatch: pytest.MonkeyPatch, clear: bool) -> tuple[RLScene, SimpleNamespace]:
    scene = RLScene.__new__(RLScene)
    scene._known = (False, False, False)
    scene._in_order, scene._from_state = False, False
    scene._library = _straight_library([30, 60, 90])
    scene._library.sequence = np.array([0, 1, 0])
    scene._index = np.array([10])
    scene.replans = []
    new = _straight_library([40, 70, 95])
    monkeypatch.setattr(RLScene, "_runtime_library", lambda self, obs: (new, clear))
    return scene, new


def test_a_reveal_brings_a_new_guide(monkeypatch: pytest.MonkeyPatch):
    scene, new = _scene(monkeypatch, clear=True)
    scene._replan_if_revealed(_obs((True, False), (False,)), np.array([[1.0, 0.0, 0.0]]), 0)
    assert scene._library is new
    assert scene._index[0] == 20
    assert len(scene.replans) == 1 and scene.replans[0]["revealed"] == 1
    # Nothing new revealed: no plan.
    scene._replan_if_revealed(_obs((True, False), (False,)), np.array([[1.0, 0.0, 0.0]]), 0)
    assert len(scene.replans) == 1


def test_a_plan_that_fails_its_check_is_not_flown(monkeypatch: pytest.MonkeyPatch):
    scene, new = _scene(monkeypatch, clear=False)
    old = scene._library
    scene._replan_if_revealed(_obs((False, True), (True,)), np.array([[1.0, 0.0, 0.0]]), 0)
    assert scene._library is old
    assert scene._index[0] == 10
    assert scene.replans[0]["clear"] is False


def test_a_fixed_track_is_replanned_on_what_the_sensor_shows():
    track = ConfigDict(
        {
            "randomize": False,
            "gate_order": [1, 2],
            "safety_limits": {"pos_limit_low": [-1, -1, 0], "pos_limit_high": [1, 1, 2]},
        }
    )
    obs = _obs((True, False), (False,))
    reported = _reported_track(obs, track, start=np.zeros(3), observed=True)
    assert np.allclose(reported["gates"][1]["pos"], obs["gates_pos"][1])
    # The guide still starts where the drone started, not where it is now.
    assert np.allclose(reported["drones"][0]["pos"], 0.0)


def test_in_order_replanning_waits_for_the_next_object_in_order(monkeypatch: pytest.MonkeyPatch):
    scene, new = _scene(monkeypatch, clear=True)
    planned = []
    monkeypatch.setattr(
        RLScene, "_runtime_library", lambda self, layout: (planned.append(layout), (new, True))[1]
    )
    scene._in_order, scene._stage = True, 0
    scene._order = np.array([1, 2, 0])
    nominal = _obs((False, False), (False,))
    scene._nominal = (nominal["gates_pos"] - 0.1, np.zeros((2, 3)), nominal["obstacles_pos"] - 0.1)
    pos = np.array([[1.0, 0.0, 0.0]])
    # The pole comes second in order, so revealing it first changes nothing.
    scene._replan_if_revealed(_obs((False, False), (True,)), pos, 0)
    assert planned == []
    # Gate 1 completes the first two: the plan uses gate 1 and the pole as seen, gate 0 as reported.
    scene._replan_if_revealed(_obs((False, True), (True,)), pos, 0)
    layout = planned[-1]
    seen = _obs((False, True), (True,))
    assert np.allclose(layout["gates_pos"][1], seen["gates_pos"][1])
    assert np.allclose(layout["gates_pos"][0], scene._nominal[0][0])
    assert np.allclose(layout["obstacles_pos"], seen["obstacles_pos"])
    assert scene._stage == 2


def test_in_order_replanning_needs_replanning():
    config = ConfigDict(
        {
            "controller": {"guide_planner": "fast_runtime", "replan_in_order": True},
            "env": {"track": {}},
        }
    )
    with pytest.raises(ValueError, match="replan_in_order"):
        RLScene(_obs((False, False), (False,)), {}, config)


def test_the_style_selector_needs_the_runtime_planner_and_no_single_policy():
    config = ConfigDict({"controller": {"style_selector": True}, "env": {"track": {}}})
    with pytest.raises(ValueError, match="fast_runtime"):
        RLScene(_obs((False, False), (False,)), {}, config)
    section = {"style_selector": True, "guide_planner": "fast_runtime", "policy": "other.npz"}
    config = ConfigDict({"controller": section, "env": {"track": {}}})
    with pytest.raises(ValueError, match="style_policies"):
        RLScene(_obs((False, False), (False,)), {}, config)


def _pilot(policy: object, bender: object = None, hold: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        bender=bender,
        bend=None,
        u=np.zeros((1, 9)),
        hold=hold,
        limit=0.3,
        from_drone=False,
    )


def test_the_selector_decides_every_hold_steps_and_the_chosen_style_flies():
    scene = RLScene.__new__(RLScene)
    # Style i always commands i / 10 on every axis.
    scene._pilots = [_pilot(lambda x, i=i: np.full((1, 4), i / 10.0)) for i in range(3)]
    picks = iter([2, 0])
    decisions = []

    def selector(inputs: np.ndarray) -> np.ndarray:
        decisions.append(inputs.copy())
        return np.eye(3)[[next(picks)]]

    scene._selector, scene._hold = selector, 3
    scene._style, scene._tick = 1, 0
    x = np.zeros((1, 86))
    flown = []
    for _ in range(5):
        flown.append(scene._select(x, scene._propose(x, None, None, None))[0, 0])
        scene._tick += 1
    assert flown == pytest.approx([0.2, 0.2, 0.2, 0.0, 0.0])
    assert len(decisions) == 2
    # The first decision sees the style flown before it and every proposal.
    assert decisions[0][0, 86:89].tolist() == [0.0, 1.0, 0.0]
    assert decisions[0][0, 89:].tolist() == pytest.approx([0.0] * 4 + [0.1] * 4 + [0.2] * 4)
    assert decisions[1][0, 86:89].tolist() == [0.0, 0.0, 1.0]


def test_a_bender_bends_every_hold_steps_and_its_policy_flies_the_bent_guide():
    scene = RLScene.__new__(RLScene)
    seen, bends = [], []

    def bender(inputs: np.ndarray) -> np.ndarray:
        bends.append(inputs[0, 0])
        return np.full((1, 9), 2.0)

    def policy(x: np.ndarray) -> np.ndarray:
        seen.append(x[0, 0])
        return np.zeros((1, 4))

    scene._pilots = [_pilot(policy, bender, hold=2)]
    plain = np.zeros((1, 86))

    def features(bend: tuple) -> np.ndarray:
        return np.full((1, 86), bend[1])

    def make(u: np.ndarray, limit: float, from_drone: bool) -> tuple:
        # The output reaches the bend clipped to [-1, 1].
        assert u.max() == 1.0
        return ("bend", len(bends))

    scene._tick = 0
    for _ in range(5):
        scene._propose(plain, features, make, lambda previous: previous)
        scene._tick += 1
    # Decisions at steps 0, 2 and 4; each sees the output before it.
    assert bends == [0.0, 1.0, 1.0]
    assert seen == [1, 1, 2, 2, 3]


def _config(controller: dict, randomizations: dict | None) -> ConfigDict:
    env = {"track": {}} | ({"randomizations": randomizations} if randomizations else {})
    return ConfigDict({"controller": controller, "env": env})


def test_a_shifted_race_flies_the_style_selector_by_default():
    shifts = {"gate_pos": {}, "obstacle_pos": {}}
    settings = _controller_settings(_config({"file": "rl_scene.py"}, shifts))
    assert settings == {"file": "rl_scene.py"} | SHIFTED_RACE_DEFAULTS


def test_a_race_without_shifts_keeps_the_solved_guide():
    assert _controller_settings(_config({"file": "rl_scene.py"}, None)) == {"file": "rl_scene.py"}
    # Disturbances alone don't move what the sensor reports.
    other = {"drone_mass": {}}
    assert _controller_settings(_config({}, other)) == {}


def test_settings_given_replace_the_shifted_race_defaults():
    shifts = {"gate_rpy": {}}
    chosen = {"guide_planner": "topt"}
    assert _controller_settings(_config(chosen, shifts)) == chosen
    chosen = {"guide_planner": "fast_runtime", "policy": "other.npz"}
    assert _controller_settings(_config(chosen, shifts)) == chosen


def _four_pass_library() -> SimpleNamespace:
    lib = _straight_library([20, 40, 60, 80])
    lib.sequence = np.array([0, 1, 0, 1])
    lib.directions = np.ones(4)
    return SimpleNamespace(_replace=lambda **kw: SimpleNamespace(**(vars(lib) | kw)), **vars(lib))


def _straight_plan(monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    from lsy_drone_racing.utils import fast_runtime_planner

    def plan(track: object, start: np.ndarray, start_vel: np.ndarray) -> tuple:
        seen.append((track.sequence.tolist(), start.tolist(), start_vel.tolist()))
        pos = start + np.linspace(0.0, 2.0, 21)[:, None] * np.array([1.0, 0.0, 0.0])
        trajectory = SimpleNamespace(
            pos=pos,
            vel=np.tile([1.0, 0.0, 0.0], (21, 1)),
            time=np.linspace(0.0, 2.0, 21),
            crossings=np.array([10]),
        )
        return trajectory, SimpleNamespace(clear=True), 1

    monkeypatch.setattr(fast_runtime_planner, "runtime_plan", plan)


def test_a_replan_from_the_state_starts_at_the_drone_and_covers_the_next_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = []
    _straight_plan(monkeypatch, seen)
    scene = RLScene.__new__(RLScene)
    scene._library, scene._replan_gates = _four_pass_library(), 1
    pos, vel = np.array([0.5, 0.0, 1.0]), np.array([0.3, 0.1, 0.0])
    library, clear = scene._state_library(_obs((True, False), (False,)), pos, vel, passed=1)
    assert clear
    # Only the second pass is planned, from where the drone is and as fast as it flies.
    assert seen == [([1], pos.tolist(), vel.tolist())]
    assert library.beads[0, 0] == pytest.approx(pos)
    crossings = library.crossings[0]
    assert crossings[0] == 0
    assert crossings[1] == pytest.approx(20, abs=1)
    # Passes beyond the plan cross at its last bead, before the run-out.
    last = library.beads.shape[1] - RUN_OUT_BEADS - 1
    assert crossings[2:].tolist() == [last, last]


def test_a_pass_brings_a_new_plan_from_the_state(monkeypatch: pytest.MonkeyPatch):
    scene, new = _scene(monkeypatch, clear=True)
    scene._known = (True, False, False)
    scene._from_state, scene._replan_gates, scene._planned_at_pass = True, 1, 0
    calls = []

    def state_library(obs: dict, pos: np.ndarray, vel: np.ndarray, passed: int) -> tuple:
        calls.append(passed)
        return new, True

    scene._state_library = state_library
    obs = _obs((True, False), (False,)) | {"vel": np.zeros(3)}
    scene._replan_if_revealed(obs, np.zeros((1, 3)), 0)
    assert calls == []
    scene._replan_if_revealed(obs, np.zeros((1, 3)), 1)
    assert calls == [1]
    assert scene._library is new
    assert scene._index.tolist() == [0]
    # The same pass doesn't plan again.
    scene._replan_if_revealed(obs, np.zeros((1, 3)), 1)
    assert calls == [1]
