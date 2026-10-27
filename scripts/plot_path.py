"""Script for visualizing the planned path of a race configuration.

The planner decides the route through the track, but nothing in the simulation output shows that
decision: the episode plots of ``debug_run.py`` start after the plan is made. This script plans the
nominal track of a configuration without running the simulation and draws what the planner produced,
which is the quickest way to see the difference between the heuristic obstacle avoidance and the A*
route (:mod:`lsy_drone_racing.control.astar`).

It writes one figure with three panels:

* the planned path from above, over the occupancy grid the A* search sees, with the gates and their
  crossing direction. The path is coloured by arc length so that a track which visits a gate twice
  (or crosses itself) can still be followed from start to finish;
* the lateral clearance to the nearest pole along the path, which is the quantity the route search
  is supposed to improve;
* the altitude along the path, which the two dimensional search leaves to the waypoint template.

Run it from the repository root, e.g.

    pixi run python scripts/plot_path.py --config level2.toml
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np

from lsy_drone_racing.control.astar import (
    GATE_FRAME_HALF,
    GATE_OPENING_HALF,
    POLE_RADIUS,
    ArenaGrid,
    AstarConfig,
)
from lsy_drone_racing.control.path_planning import GateFrames, PathConfig, PathPlanner
from lsy_drone_racing.envs.utils import load_gate_order, load_track
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def observation(config: ConfigDict) -> dict[str, np.ndarray]:
    """Build the observation the planner needs from the nominal track of a configuration.

    Args:
        config: The loaded race configuration.

    Returns:
        A dictionary with the observation keys [PathPlanner.plan][] reads.
    """
    gates, obstacles, drones = load_track(config.env.track)
    order, sign = load_gate_order(config.env.track, len(config.env.track.gates))
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


def pole_clearance(points: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """Return the XY distance from every path point to the nearest obstacle.

    Args:
        points: Path points, shape ``(n, 3)``.
        obstacles: Obstacle positions, shape ``(m, 3)``.

    Returns:
        The clearance of every point, shape ``(n, )``, or all ``inf`` without obstacles.
    """
    if len(obstacles) == 0:
        return np.full(len(points), np.inf)
    delta = points[:, None, :2] - np.asarray(obstacles, dtype=float)[None, :, :2]
    return np.linalg.norm(delta, axis=-1).min(axis=1)


def plan_all(obs: dict[str, np.ndarray], modes: list[bool]) -> list[dict]:
    """Plan every requested mode and collect what the figure and the summary need.

    Args:
        obs: The observation built by [observation][__main__.observation].
        modes: One entry per plan, ``True`` to use the A* route search.

    Returns:
        One record per plan with the path, its metadata and its timing.
    """
    records = []
    for use_astar in modes:
        planner = PathPlanner(PathConfig(use_astar=use_astar))
        start = time.perf_counter()
        path, info = planner.plan(obs, obs["pos"])
        elapsed = (time.perf_counter() - start) * 1e3
        arc = np.linspace(0.0, path.length, 4000)
        points = path.pos(arc)
        records.append(
            {
                "use_astar": use_astar,
                "path": path,
                "info": info,
                "arc": arc,
                "points": points,
                "clearance": pole_clearance(points, info.obstacles),
                "plan_ms": elapsed,
            }
        )
    return records


def draw_gates(ax: Axes, obs: dict[str, np.ndarray]) -> None:
    """Draw the gate frames, their opening and their crossing direction.

    Args:
        ax: The axes to draw on.
        obs: The observation holding the gate poses.
    """
    from scipy.spatial.transform import Rotation as R

    for gate_id, (centre, quat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"], strict=True)):
        rotation = R.from_quat(quat)
        lateral = rotation.apply([0.0, 1.0, 0.0])
        normal = rotation.apply([1.0, 0.0, 0.0])
        # The frame is a square ring 0.2 m to 0.36 m from the centre. Seen from above, only the two
        # side posts of the ring are visible; the opening between them is what the drone crosses.
        for sign in (1.0, -1.0):
            post = np.stack(
                [
                    centre + sign * GATE_OPENING_HALF * lateral,
                    centre + sign * GATE_FRAME_HALF * lateral,
                ]
            )
            ax.plot(post[:, 0], post[:, 1], color="0.20", lw=5.0, solid_capstyle="butt", zorder=4)
        opening = np.stack(
            [centre - GATE_OPENING_HALF * lateral, centre + GATE_OPENING_HALF * lateral]
        )
        ax.plot(opening[:, 0], opening[:, 1], color="tab:orange", lw=1.3, ls=":", zorder=4)
        ax.annotate(
            f"g{gate_id}",
            xy=(centre[0], centre[1]),
            xytext=(centre[0] + 0.09, centre[1] + 0.10),
            color="tab:orange",
            fontsize=9.5,
            zorder=6,
        )
        ax.arrow(
            centre[0],
            centre[1],
            0.30 * normal[0],
            0.30 * normal[1],
            color="tab:orange",
            width=0.006,
            head_width=0.05,
            zorder=4,
        )


def draw_poles(ax: Axes, obstacles: np.ndarray, inflate: float | None) -> None:
    """Draw the obstacles and, optionally, the radius the A* grid blocks around them.

    Args:
        ax: The axes to draw on.
        obstacles: Obstacle positions, shape ``(n, 3)``.
        inflate: Radius [m] of the blocking circle the search uses, or ``None`` to omit it.
    """
    import matplotlib.pyplot as plt

    for pole_id, pole in enumerate(obstacles):
        if inflate is not None:
            ax.add_patch(
                plt.Circle(
                    (pole[0], pole[1]),
                    inflate,
                    fill=False,
                    ec="tab:red",
                    ls="--",
                    lw=1.0,
                    alpha=0.75,
                    zorder=2,
                )
            )
        ax.add_patch(
            plt.Circle((pole[0], pole[1]), POLE_RADIUS, fc="tab:purple", ec="none", zorder=5)
        )
        ax.annotate(
            f"o{pole_id}",
            xy=(pole[0], pole[1]),
            xytext=(pole[0] + 0.07, pole[1] - 0.16),
            color="tab:purple",
            fontsize=8,
            zorder=6,
        )


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    base = PathConfig()
    conf = load_config(REPO_ROOT / "config" / config)
    obs = observation(conf)
    modes = [False, True] if compare else [base.use_astar]
    records = plan_all(obs, modes)
    # The coloured route is the interesting one: the A* route when it is enabled, else the default.
    featured = next((record for record in records if record["use_astar"]), records[-1])

    overlay = None
    if grid:
        frames = GateFrames.from_quaternions(obs["gates_quat"])
        overlay = ArenaGrid(
            obs["obstacles_pos"],
            obs["gates_pos"],
            frames.y_axes,
            AstarConfig(
                resolution=resolution,
                pole_clearance=base.safety_distance,
                # Must match what the planner passes, or the drawn occupancy is more optimistic
                # than the grid the route was actually searched on.
                overshoot=base.push_scale,
                bounds_low=base.astar_bounds[0],
                bounds_high=base.astar_bounds[1],
            ),
        )

    figure = plt.figure(figsize=(16.0, 5.8))
    layout = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.40, wspace=0.22)
    top = figure.add_subplot(layout[:, 0])
    clearance_axis = figure.add_subplot(layout[0, 1])
    height_axis = figure.add_subplot(layout[1, 1])

    # --- panel 1: the route from above ---------------------------------------------------------
    if overlay is not None:
        low = np.asarray(overlay.config.bounds_low, dtype=float)
        high = np.asarray(overlay.config.bounds_high, dtype=float)
        top.imshow(
            overlay.occupied.T,
            origin="lower",
            extent=(low[0], high[0], low[1], high[1]),
            cmap="Greys",
            alpha=0.22,
            zorder=0,
            interpolation="nearest",
        )
    draw_poles(top, obs["obstacles_pos"], overlay.pole_inflate if overlay is not None else None)
    draw_gates(top, obs)

    for record in records:
        if record is featured:
            continue
        top.plot(
            record["points"][:, 0], record["points"][:, 1], color="0.45", ls="--", lw=1.5, zorder=3
        )
    # The featured route is coloured by arc length so a self-crossing track stays readable.
    points = featured["points"]
    segments = np.stack([points[:-1, :2], points[1:, :2]], axis=1)
    collection = LineCollection(segments, cmap="viridis", zorder=5, linewidths=2.6)
    collection.set_array(featured["arc"][:-1])
    top.add_collection(collection)
    figure.colorbar(collection, ax=top, fraction=0.045, pad=0.02, label="arc length [m]")

    tightest = int(np.argmin(featured["clearance"]))
    if np.isfinite(featured["clearance"][tightest]):
        top.plot(
            points[tightest, 0],
            points[tightest, 1],
            "o",
            ms=10,
            mfc="none",
            mec="tab:red",
            mew=1.8,
            zorder=7,
        )
        top.annotate(
            f"tightest {featured['clearance'][tightest] * 100:.0f} cm",
            xy=(points[tightest, 0], points[tightest, 1]),
            xytext=(points[tightest, 0] - 0.62, points[tightest, 1] + 0.18),
            color="tab:red",
            fontsize=8.5,
            zorder=8,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.8,
                "color": "tab:red",
                "shrinkA": 0,
                "shrinkB": 6,
            },
        )
    top.plot(
        obs["pos"][0], obs["pos"][1], "*", ms=17, color="tab:green", mec="k", mew=0.6, zorder=7
    )
    top.plot(points[-1, 0], points[-1, 1], "s", ms=8, color="tab:green", mec="k", mew=0.6, zorder=7)
    top.annotate(
        "end of path",
        xy=(points[-1, 0], points[-1, 1]),
        xytext=(points[-1, 0] + 0.12, points[-1, 1] + 0.06),
        color="tab:green",
        fontsize=8.5,
        zorder=8,
    )
    top.set_aspect("equal")
    top.set_xlabel("x [m]")
    top.set_ylabel("y [m]")
    title = "Planned path from above"
    if overlay is not None:
        title += (
            f"\ngrey = A* occupancy ({resolution:.2f} m cells), "
            f"red dashed = blocked pole radius {overlay.pole_inflate:.2f} m"
        )
    top.set_title(title, fontsize=10.5)
    top.grid(alpha=0.22)
    handles = [
        Line2D([], [], color=plt.cm.viridis(0.6), lw=2.6, label="A* route (colour = arc length)"),
        Line2D([], [], color="0.45", ls="--", lw=1.5, label="heuristic push-out"),
        Line2D([], [], color="tab:green", marker="*", ms=13, ls="none", label="start"),
        Line2D([], [], color="tab:green", marker="s", ms=8, ls="none", label="end of path"),
    ]
    if not any(record["use_astar"] for record in records):
        handles = [handles[1], handles[2]]
    top.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95)

    # --- panel 2: lateral clearance along the path --------------------------------------------
    for record in records:
        clearance_axis.plot(
            record["arc"],
            record["clearance"],
            color="tab:blue" if record["use_astar"] else "0.45",
            ls="-" if record["use_astar"] else "--",
            lw=2.0 if record["use_astar"] else 1.5,
            label="A* route" if record["use_astar"] else "heuristic push-out",
        )
    clearance_axis.axhline(
        base.safety_distance,
        color="tab:red",
        ls=":",
        lw=1.2,
        label=f"safety_distance {base.safety_distance:.2f} m",
    )
    clearance_axis.set_xlabel("arc length [m]")
    clearance_axis.set_ylabel("clearance to nearest pole [m]")
    clearance_axis.set_ylim(bottom=0.0)
    clearance_axis.set_title("Lateral clearance along the path", fontsize=10.5)
    clearance_axis.grid(alpha=0.22)
    clearance_axis.legend(fontsize=8, loc="lower right", framealpha=0.95)

    # --- panel 3: altitude along the path ------------------------------------------------------
    for record in records:
        height_axis.plot(
            record["arc"],
            record["points"][:, 2],
            color="tab:blue" if record["use_astar"] else "0.45",
            ls="-" if record["use_astar"] else "--",
            lw=2.0 if record["use_astar"] else 1.5,
            label="A* route" if record["use_astar"] else "heuristic push-out",
        )
    for gate_id in range(len(obs["gates_pos"])):
        height_axis.axhline(
            obs["gates_pos"][gate_id][2], color="tab:orange", lw=0.7, ls=":", alpha=0.6, zorder=0
        )
    height_axis.set_xlabel("arc length [m]")
    height_axis.set_ylabel("z [m]")
    height_axis.set_title("Altitude along the path (dotted = gate heights)", fontsize=10.5)
    height_axis.grid(alpha=0.22)
    height_axis.legend(fontsize=8, loc="upper left", framealpha=0.95)

    figure.suptitle(f"Path planning on {config}", fontsize=12)
    out_path = REPO_ROOT / "plots" / f"path_{Path(config).stem}.png" if out is None else Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # --- terminal summary ---------------------------------------------------------------------
    print(f"{config}: {len(obs['gate_sequence'])} gate passes, {len(obs['obstacles_pos'])} poles")
    print(
        f"safety_distance = {base.safety_distance:.2f} m, A* pole radius = "
        f"{base.safety_distance * base.push_scale:.2f} m (clearance x push_scale)"
    )
    print(f"{'mode':12s} {'length':>9s} {'min clear':>10s} {'at arc':>8s} {'plan':>9s}")
    for record in records:
        clearance = record["clearance"]
        tightest_arc = int(np.argmin(clearance))
        name = "astar" if record["use_astar"] else "heuristic"
        print(
            f"{name:12s} {record['path'].length:8.2f} m {clearance[tightest_arc]:9.3f} m "
            f"{record['arc'][tightest_arc]:7.2f} m {record['plan_ms']:8.1f} ms"
        )
    print(f"\nplot -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
