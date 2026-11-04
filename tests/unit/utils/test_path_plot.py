"""The debug-run figure: it has to draw every kind of log, and only draw a height that is one."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from lsy_drone_racing.utils.path_plot import _plot_arrays, position_command

if TYPE_CHECKING:
    from pathlib import Path


def synthetic_run(steps: int = 60) -> dict:
    """Build the arrays of a finished run: a short climb through two gates past one pole."""
    times = np.linspace(0.0, 3.0, steps)
    phase = np.linspace(0.0, 2.0, steps)
    pos = np.stack([phase - 1.0, 0.2 * np.sin(2.0 * phase), 0.5 + 0.3 * phase], axis=1)
    passed = np.zeros(steps, dtype=int)
    passed[steps // 3 :] = 1
    passed[(2 * steps) // 3 :] = 2
    return {
        "t": times,
        "pos": pos,
        "speed": np.linalg.norm(np.gradient(pos, times, axis=0), axis=1),
        "des": pos + np.array([0.02, -0.02, 0.05]),
        "n": passed,
        "gate_sequence": np.array([0, 1]),
        "gates_pos": np.array([[0.0, 0.0, 0.6], [0.5, 0.2, 0.8]]),
        "gates_quat": np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        "true_gates_pos": np.array([[0.05, 0.02, 0.62], [0.52, 0.18, 0.79]]),
        "obstacles_pos": np.array([[-0.3, 0.4, 1.0]]),
        "true_obstacles_pos": np.array([[-0.28, 0.38, 1.0]]),
        "bounds_low": np.array([-2.0, -1.0, 0.0]),
        "bounds_high": np.array([2.0, 1.0, 2.0]),
        "failure_kind": np.array("none"),
        "failure_object": np.array(-1),
        "failure_pos": pos[-1],
        "failure_t": np.array(float(times[-1])),
        "failure_distance": np.array(0.0),
    }


@pytest.mark.unit
def test_figure_written_for_a_finished_run(tmp_path: Path):
    arrays = synthetic_run()
    out = tmp_path / "run.png"
    _plot_arrays(arrays, out, title="synthetic", dpi=60)
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
@pytest.mark.parametrize(("kind", "index"), [("HIT_OBSTACLE", 0), ("HIT_GATE", 1)])
def test_figure_written_for_a_failed_run(tmp_path: Path, kind: str, index: int):
    arrays = synthetic_run()
    arrays["failure_kind"] = np.array(kind)
    arrays["failure_object"] = np.array(index)
    arrays["failure_t"] = np.array(2.5)
    arrays["failure_distance"] = np.array(0.05)
    out = tmp_path / f"{kind}.png"
    _plot_arrays(arrays, out, title=kind, dpi=60)
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_position_command_needs_state_mode():
    arrays = synthetic_run()
    # In attitude mode ``des`` is roll/pitch/yaw; drawing its third component as a height would be a
    # plausible-looking lie, so nothing may be returned.
    assert position_command(arrays, "attitude") is None
    assert position_command(arrays, "state") is not None
    arrays["control_mode"] = np.array("state")
    assert position_command(arrays) is not None
    arrays["control_mode"] = np.array("attitude")
    assert position_command(arrays) is None
    del arrays["control_mode"]
    # A log written before the mode was recorded cannot prove its ``des`` is a position.
    assert position_command(arrays) is None


@pytest.mark.unit
def test_position_command_rejects_mismatched_des():
    arrays = synthetic_run()
    arrays["des"] = arrays["des"][:-3]
    assert position_command(arrays, "state") is None
    arrays["des"] = arrays["pos"][:, :2]
    assert position_command(arrays, "state") is None
