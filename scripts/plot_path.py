"""Script for drawing a run from the log that ``debug_run.py`` wrote.

The figure is built entirely from the recorded data, so it can be regenerated offline, compared
between runs, and never disagrees with the numbers it came from. See
``lsy_drone_racing.utils.path_plot``.

Run it from the repository root, e.g.

    pixi run python scripts/plot_path.py --log logs/level2_my_controller_seed0.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from lsy_drone_racing.utils.path_plot import plot_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def newest_log(directory: Path) -> Path:
    """Return the most recently written run log in a directory.

    Args:
        directory: Directory to search.

    Returns:
        The newest ``.npz``.

    Raises:
        FileNotFoundError: If the directory holds no run logs.
    """
    logs = sorted(directory.glob("*.npz"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"no .npz run logs in {directory}; run scripts/debug_run.py first")
    return logs[-1]


def main(
    log: str | None = None,
    out: str | None = None,
    planned: bool = False,
    grid: bool = True,
    resolution: float = 0.10,
    dpi: int = 140,
) -> None:
    """Draw one logged run.

    Args:
        log: Path of the ``.npz`` written by ``debug_run.py``. Defaults to the newest in `logs/`.
        out: Output image path. Defaults to `plots/<log name>.png`.
        planned: Whether to draw the last path the controller committed to, from the log.
        grid: Whether to draw the occupancy overlay of the A* search.
        resolution: Edge length [m] of the A* grid cell used for the occupancy overlay.
        dpi: Resolution of the saved figure.
    """
    log_path = Path(log) if log is not None else newest_log(REPO_ROOT / "logs")
    out_path = Path(out) if out is not None else REPO_ROOT / "plots" / f"{log_path.stem}.png"
    written = plot_run(
        log_path, out_path, planned=planned, grid=grid, resolution=resolution, dpi=dpi
    )

    summary_path = log_path.with_suffix(".json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"{log_path.name}")
        print(
            f"  verdict    {summary.get('verdict')} "
            f"({summary.get('n_final')}/{summary.get('n_required')} passes)"
        )
        print(f"  finish     {summary.get('sim_time'):.2f} s over {summary.get('steps')} steps")
        print(f"  plans      {summary.get('n_plans')}")
        print(
            f"  solver     {summary.get('solver_ms_mean'):.2f} ms mean, "
            f"{summary.get('solver_ms_p95'):.2f} ms p95, "
            f"{summary.get('solver_failures')} failures"
        )
        print(
            f"  peaks      {summary.get('speed_max'):.2f} m/s, {summary.get('acc_max'):.1f} m/s^2"
        )
    print(f"\nplot -> {written}")


if __name__ == "__main__":
    fire.Fire(main)
