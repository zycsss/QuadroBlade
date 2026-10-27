"""Script for visualizing the planned path of a race configuration.

The planner decides the route through the track, but the simulation output starts after the plan is
made. This script plans the nominal track of a configuration without running the simulation and
draws what the planner produced, using the same figure that `scripts/debug_run.py` writes per
episode (see `lsy_drone_racing.utils.path_plot`).

Run it from the repository root, e.g.

    pixi run python scripts/plot_path.py --config level2.toml
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np

from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.path_plot import observation_from_config, plot_paths

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(
    config: str = "level2.toml",
    out: str | None = None,
    compare: bool = True,
    resolution: float = 0.10,
    grid: bool = True,
    dpi: int = 140,
) -> None:
    """Plot the planned path of a race configuration.

    Args:
        config: Name of the race configuration. Assumed to be in `config/`.
        out: Output image path. Defaults to `plots/path_<config>.png`.
        compare: Whether to plan with and without the A* route search. Disable to plot only the
            planner default.
        resolution: Edge length [m] of the A* grid cell used for the occupancy overlay.
        grid: Whether to draw the occupancy overlay of the A* search.
        dpi: Resolution of the saved figure.
    """
    conf = load_config(REPO_ROOT / "config" / config)
    obs = observation_from_config(conf)
    out_path = REPO_ROOT / "plots" / f"path_{Path(config).stem}.png" if out is None else Path(out)
    records = plot_paths(
        obs,
        out_path,
        compare=compare,
        resolution=resolution,
        grid=grid,
        dpi=dpi,
        config=conf,
        title=f"Path planning on {config}",
    )

    print(f"{config}: {len(obs['gate_sequence'])} gate passes, {len(obs['obstacles_pos'])} poles")
    print(f"{'route':12s} {'length':>9s} {'min clear':>10s} {'at arc':>8s} {'plan':>9s}")
    for record in records:
        index = int(np.argmin(record.clearance))
        name = "astar" if record.use_astar else "heuristic"
        print(
            f"{name:12s} {record.path.length:8.2f} m {record.clearance[index]:9.3f} m "
            f"{record.arc[index]:7.2f} m {record.plan_ms:8.1f} ms"
        )
    print(f"\nplot -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
