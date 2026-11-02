"""Regression checks for MPCC snapshots and the flight-reporting boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def observation(x: float, passed: int = 0, sequence: tuple = (0,), direction: tuple = (1,)) -> dict:
    """A gate at x=0, whose plane has an analytically known intersection."""
    return {
        "pos": np.array([x, 0.0, 1.0]),
        "vel": np.array([2.0, 0.0, 0.0]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "n_gates_passed": passed,
        "gate_sequence": np.array(sequence),
        "gate_sequence_direction": np.array(direction),
        "gates_pos": np.array([[0.0, 0.0, 1.0]]),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "obstacles_pos": np.zeros((0, 3)),
    }


def test_controller_loads_and_snapshot_mutation_cannot_change_flight(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exports must not let a plotting consumer change active planning or settings."""
    from lsy_drone_racing.control import my_controller
    from lsy_drone_racing.utils import load_config, load_controller

    # Only replace the native compilation boundary; the planner and controller init stay real.
    solver = SimpleNamespace(reset=lambda: None, set=lambda *args: None)
    monkeypatch.setattr(my_controller, "get_solver", lambda *args: (solver, None))
    cls = load_controller(ROOT / "lsy_drone_racing/control/mpcc_controller.py")
    config = load_config(ROOT / "config/level0.toml")
    controller = cls(observation(-1.0), {}, config)
    first = controller.reference_snapshot()
    assert first["points"].shape[1] == 3
    np.testing.assert_allclose(first["points"][0], [-1.0, 0.0, 1.0])
    first["points"][:] = 99
    first["gate_centers"][:] = 99
    again = controller.reference_snapshot()
    np.testing.assert_allclose(again["points"][0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(again["gate_centers"], [[0.0, 0.0, 1.0]])
    settings = controller.settings_snapshot()
    settings["mpcc"]["v_max"] = 99
    assert controller.settings_snapshot()["mpcc"]["v_max"] < 99
    assert controller.step_callback(np.zeros(4), observation(-1), 0, False, False, {}) is False


def test_pass_time_and_speed_are_interpolated_not_rounded_to_next_sample():
    """Crossing one quarter through a 0.5 s interval occurs at 0.125 s."""
    module = importlib.import_module("tests.mpcc_evaluation")
    before, after = observation(-1), observation(3, 1)
    after["vel"] = np.array([6.0, 4.0, 0.0])
    recorder = module.EpisodeRecorder(before, freq=2)
    recorder.record_step(np.zeros(4), after, {"solver_ms": 1.0})
    row = recorder.gates[0]
    assert row["time_s"] == pytest.approx(0.125)
    assert row["split_s"] == pytest.approx(0.125)
    assert row["speed_mps"] == pytest.approx(np.sqrt(10))
    assert row["vx_mps"] == pytest.approx(3)
    assert row["vy_mps"] == pytest.approx(1)


def test_repeated_reverse_pass_is_separate_and_uses_the_right_direction():
    """Deduplicating gate IDs or dropping the final sample loses the second event."""
    module = importlib.import_module("tests.mpcc_evaluation")
    sequence, direction = (0, 0), (1, -1)
    recorder = module.EpisodeRecorder(observation(-1, 0, sequence, direction), freq=2)
    recorder.record_step(np.zeros(4), observation(1, 1, sequence, direction), {})
    recorder.record_step(np.zeros(4), observation(-1, 2, sequence, direction), {})
    assert [row["pass_index"] for row in recorder.gates] == [1, 2]
    assert [row["gate_id"] for row in recorder.gates] == [0, 0]
    assert [row["direction"] for row in recorder.gates] == [1, -1]
    assert [row["time_s"] for row in recorder.gates] == pytest.approx([0.25, 0.75])
    assert [row["split_s"] for row in recorder.gates] == pytest.approx([0.25, 0.5])


@pytest.mark.parametrize("reason", ["FINISHED", "TIMEOUT", "CONTROLLER_STOPPED"])
def test_runner_records_final_event_and_closes_environment(reason: str):
    """A terminal step must be recorded before the loop exits, with resources closed."""
    module = importlib.import_module("tests.mpcc_evaluation")
    initial = observation(-1)
    final = observation(1, int(reason == "FINISHED"))
    state = {"closed": False}
    env = SimpleNamespace(
        step=lambda action: (final, 0.0, reason == "FINISHED", reason == "TIMEOUT", {}),
        close=lambda: state.update(closed=True),
        unwrapped=SimpleNamespace(data=SimpleNamespace(disabled_drones=np.array([[False]]))),
    )
    ctrl = SimpleNamespace(
        compute_control=lambda obs, info: np.zeros(4),
        diagnostics=lambda: {},
        step_callback=lambda *args: reason == "CONTROLLER_STOPPED",
    )
    recorder = module.EpisodeRecorder(initial, freq=2)
    assert module.fly_episode(env, ctrl, initial, {}, recorder) == reason
    assert state["closed"]
    assert recorder.steps == 1
    assert recorder.gates[0]["status"] == ("passed" if reason == "FINISHED" else "not_passed")


def test_failure_report_excludes_warped_pose_and_retains_missing_gate(tmp_path: Path):
    """Disabled-drone reset coordinates must never become a flight segment."""
    module = importlib.import_module("tests.mpcc_evaluation")
    initial = observation(-1)
    recorder = module.EpisodeRecorder(initial, freq=2)
    warped = observation(-1)
    warped["pos"][:] = -1
    recorder.record_step(np.zeros(4), warped, {}, valid_state=False)
    reference = {
        "points": np.array([[-1.0, 0, 1], [1.0, 0, 1]]),
        "gate_centers": initial["gates_pos"],
        "gates_pos": initial["gates_pos"],
        "gates_quat": initial["gates_quat"],
        "obstacles": np.zeros((0, 3)),
    }
    summary = module.write_report(
        recorder,
        reference,
        {"mpcc": {}},
        [],
        {"status": "TERMINATED", "seed": 0, "config": "synthetic"},
        tmp_path,
    )
    assert summary["completed"] is False
    assert summary["lap_time_s"] is None
    assert recorder.gates[0]["time_s"] is None
    assert (tmp_path / "trajectory.png").read_bytes().startswith(b"\x89PNG")
    with np.load(tmp_path / "telemetry.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["pos"][0], [-1, 0, 1])
        assert np.isnan(data["pos"][1]).all()
        assert data["valid_state"].tolist() == [True, False]
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "TERMINATED"
    assert "not_passed" in (tmp_path / "gates.csv").read_text()


def test_runner_closes_environment_when_compute_raises():
    """Native solver exceptions must not leak the environment."""
    module = importlib.import_module("tests.mpcc_evaluation")
    state = {"closed": False}
    env = SimpleNamespace(close=lambda: state.update(closed=True))

    def fail(obs: dict, info: dict):
        raise RuntimeError("solver unavailable")

    with pytest.raises(RuntimeError, match="solver unavailable"):
        module.fly_episode(
            env,
            SimpleNamespace(compute_control=fail),
            observation(-1),
            {},
            module.EpisodeRecorder(observation(-1), freq=2),
        )
    assert state["closed"]
