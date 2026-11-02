"""Opt-in fixed-seed speed acceptance; no randomized reliability claim."""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from tests.mpcc_evaluation import ROOT, evaluate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_MPCC_BUILD") != "1",
        reason="set RUN_MPCC_BUILD=1 to compile and fly the sub6 MPCC",
    ),
]


@pytest.mark.parametrize("repeat", range(3))
def test_level2_seed0_completes_below_six_seconds(repeat: int):
    """A full ordered flight must beat six seconds on each fresh instance."""
    config = ROOT / "config/level2.toml"
    original = config.read_bytes()
    summary = evaluate(
        config,
        seed=0,
        output_dir=ROOT / "plots/mpcc/sub6/acceptance" / str(repeat),
        controller_file="mpcc_level2_sub6_controller.py",
    )
    assert config.read_bytes() == original
    assert summary["completed"] and summary["status"] == "FINISHED", summary
    assert summary["gates_passed"] == summary["gates_required"] == 5
    assert summary["sim_time_s"] < 6.0
    assert summary["solver_failures"] == 0
    directory = Path(summary["output_dir"])
    with (directory / "gates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["gate_number"]) for row in rows] == [1, 2, 3, 4, 2]
    assert all(row["status"] == "passed" for row in rows)
    assert (np.diff([float(row["time_s"]) for row in rows]) > 0).all()
    assert float(rows[-1]["time_s"]) == pytest.approx(summary["lap_time_s"])
    with np.load(directory / "telemetry.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["action"]).all()
        assert np.isfinite(arrays["pos"]).all()
        assert arrays["n_gates_passed"][-1] == 5
    assert (directory / "trajectory.png").read_bytes().startswith(b"\x89PNG")
