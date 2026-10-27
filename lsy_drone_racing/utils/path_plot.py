"""Figures for a planned path.

The planner decides the route through the track, but the simulation output did not show that
decision, and it did not show how the drone actually flew it either. This module draws both, for
``scripts/plot_path.py`` and for the per-episode figures of ``scripts/debug_run.py``, so there is a
single implementation of the picture.

One view from above, showing

* the planned path as a black line, over the occupancy grid the A* search sees, with the gates and
  their crossing direction;
* the achieved trajectory of an episode, when there is one, coloured by its speed — the quantity
  that says how the flight went, rather than how it was supposed to go;
* every pole with two circles: the solid one is ``safety_distance``, the clearance the path has to
  keep, and the dotted one is the slightly larger radius the A* search blocks so that its route
  survives the smoothing pass. Drawing only the larger radius made the path look as if it cut
  through its own constraint.

The legend sits outside the axes so it never covers the track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.astar import (
    GATE_FRAME_HALF,
    GATE_OPENING_HALF,
    POLE_RADIUS,
    ArenaGrid,
    AstarConfig,
)
from lsy_drone_racing.control.path_planning import (
    GateFrames,
    PathConfig,
    PathPlanner,
    path_config_from,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


@dataclass
class PathRecord:
    """One planned route and the numbers the summaries need.

    Attributes:
        use_astar: Whether this route came from the A* search rather than the push-out heuristic.
        path: The planned path.
        gate_centers: Gate centres in passing order, shape ``(n_passes, 3)``.
        obstacles: Obstacle positions used for the plan, shape ``(n_obstacles, 3)``.
        arc: Arc length samples, shape ``(n, )``.
        points: Path positions at ``arc``, shape ``(n, 3)``.
        clearance: XY distance to the nearest pole at every sample, shape ``(n, )``.
        plan_ms: Wall-clock time of the plan in milliseconds.
    """

    use_astar: bool
    path: object
    gate_centers: NDArray[np.floating]
    obstacles: NDArray[np.floating]
    arc: NDArray[np.floating]
    points: NDArray[np.floating]
    clearance: NDArray[np.floating]
    plan_ms: float


def observation_from_config(config: ConfigDict) -> dict[str, NDArray[np.floating]]:
    """Build the observation the planner needs from the nominal track of a configuration.

    Args:
        config: The loaded race configuration.

    Returns:
        A dictionary with the observation keys the planner reads.
    """
    from lsy_drone_racing.envs.utils import load_gate_order, load_track

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


def pole_clearance(
    points: NDArray[np.floating], obstacles: NDArray[np.floating]
) -> NDArray[np.floating]:
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


def plan_route(
    obs: dict[str, NDArray[np.floating]], config: ConfigDict | None = None, samples: int = 4000
) -> PathRecord:
    """Plan one route with the configured planner and collect what the figure and the summary need.

    Args:
        obs: Observation dictionary, see [observation_from_config][...observation_from_config].
        config: Race configuration whose planner settings apply, or ``None`` for the defaults.
        samples: Number of arc length samples to evaluate.

    Returns:
        The planned route.
    """
    import time

    settings = path_config_from(config)
    planner = PathPlanner(settings)
    start = time.perf_counter()
    path, info = planner.plan(obs, obs["pos"])
    elapsed = (time.perf_counter() - start) * 1e3
    arc = np.linspace(0.0, path.length, samples)
    points = path.pos(arc)
    return PathRecord(
        use_astar=settings.use_astar,
        path=path,
        gate_centers=info.gate_centers,
        obstacles=info.obstacles,
        arc=arc,
        points=points,
        clearance=pole_clearance(points, info.obstacles),
        plan_ms=elapsed,
    )


def _draw_gates(ax: Axes, obs: dict[str, NDArray[np.floating]]) -> None:
    """Draw the gate frames, their opening and their crossing direction."""
    from scipy.spatial.transform import Rotation as R

    for gate_id, (centre, quat) in enumerate(zip(obs["gates_pos"], obs["gates_quat"], strict=True)):
        rotation = R.from_quat(quat)
        lateral = rotation.apply([0.0, 1.0, 0.0])
        normal = rotation.apply([1.0, 0.0, 0.0])
        # The frame is a square ring from GATE_OPENING_HALF to GATE_FRAME_HALF. From above only the
        # two side posts are visible; the opening between them is what the drone crosses.
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


def _draw_poles(ax: Axes, obstacles: NDArray[np.floating], settings: PathConfig) -> None:
    """Draw every obstacle with its required clearance and the radius the search blocks."""
    import matplotlib.pyplot as plt

    required = settings.safety_distance
    searched = required * settings.push_scale
    for pole_id, pole in enumerate(obstacles):
        ax.add_patch(
            plt.Circle((pole[0], pole[1]), required, fill=False, ec="tab:red", lw=1.3, zorder=2)
        )
        ax.add_patch(
            plt.Circle(
                (pole[0], pole[1]),
                searched,
                fill=False,
                ec="tab:red",
                ls=":",
                lw=1.0,
                alpha=0.65,
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


def plot_paths(
    obs: dict[str, NDArray[np.floating]],
    out_path: str | Path,
    *,
    resolution: float = 0.10,
    grid: bool = True,
    dpi: int = 140,
    flown: NDArray[np.floating] | None = None,
    flown_speed: NDArray[np.floating] | None = None,
    extra: list[tuple[str, NDArray[np.floating]]] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    config: ConfigDict | None = None,
    heading: str | None = None,
) -> PathRecord:
    """Draw the planned path of a track, and the flight that followed it, and save the figure.

    Args:
        obs: Observation dictionary, see [observation_from_config][...observation_from_config].
        out_path: Where to write the image. Parent directories are created.
        resolution: Edge length [m] of the A* grid cell used for the occupancy overlay.
        grid: Whether to draw the occupancy overlay of the A* search.
        dpi: Resolution of the saved figure.
        flown: Optional achieved trajectory of an episode, shape ``(n, 3)``. Drawn coloured by
            speed.
        flown_speed: Speed at every ``flown`` sample, shape ``(n, )``. Without it the flown
            trajectory is coloured by arc length instead.
        extra: Optional further paths to draw as thin lines, each a ``(label, points)`` pair.
        title: Figure title. Defaults to "Planned path".
        subtitle: Optional second title line, e.g. the verdict of an episode.
        config: Race configuration whose planner settings apply, or ``None`` for the defaults.
        heading: Heading of the plot. Defaults to "Track from above".

    Returns:
        The planned route.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    settings = path_config_from(config)
    record = plan_route(obs, config)

    overlay = None
    if grid:
        frames = GateFrames.from_quaternions(obs["gates_quat"])
        overlay = ArenaGrid(
            obs["obstacles_pos"],
            obs["gates_pos"],
            frames.y_axes,
            AstarConfig(
                resolution=resolution,
                pole_clearance=settings.safety_distance,
                frame_margin=settings.astar_frame_margin,
                # Must match what the planner passes, or the drawn occupancy is more optimistic
                # than the grid the route was actually searched on.
                overshoot=settings.push_scale,
                bounds_low=settings.astar_bounds[0],
                bounds_high=settings.astar_bounds[1],
            ),
        )

    # The legend lives outside the axes, so the track needs room to its left and the figure room to
    # its right.
    figure, top = plt.subplots(figsize=(13.0, 6.4))
    handles: list = []

    if overlay is not None:
        low = np.asarray(overlay.config.bounds_low, dtype=float)
        high = np.asarray(overlay.config.bounds_high, dtype=float)
        top.imshow(
            overlay.occupied.T,
            origin="lower",
            extent=(low[0], high[0], low[1], high[1]),
            cmap="Greys",
            alpha=0.30,
            zorder=0,
            interpolation="nearest",
        )
        handles.append(
            Patch(facecolor="0.55", alpha=0.45, label=f"A* occupancy ({resolution:.2f} m cells)")
        )

    _draw_poles(top, obs["obstacles_pos"], settings)
    _draw_gates(top, obs)
    handles.extend(
        [
            Line2D([], [], color="0.20", lw=5.0, label="gate frame"),
            Line2D([], [], color="tab:orange", lw=1.3, ls=":", label="gate opening + direction"),
            Line2D([], [], marker="o", color="tab:purple", ls="none", ms=5, label="pole"),
            Line2D(
                [],
                [],
                color="tab:red",
                lw=1.3,
                label=f"required clearance {settings.safety_distance:.2f} m",
            ),
            Line2D(
                [],
                [],
                color="tab:red",
                ls=":",
                lw=1.0,
                label=f"A* search margin {settings.safety_distance * settings.push_scale:.2f} m",
            ),
        ]
    )

    for label, points_extra in extra or []:
        top.plot(points_extra[:, 0], points_extra[:, 1], color="tab:cyan", ls="-", lw=1.5, zorder=3)
        handles.append(Line2D([], [], color="tab:cyan", lw=1.5, label=label))

    # The planned path is the reference: one black line, no colour coding.
    top.plot(
        record.points[:, 0],
        record.points[:, 1],
        color="k",
        lw=2.0,
        zorder=6,
        solid_capstyle="round",
    )
    handles.insert(0, Line2D([], [], color="k", lw=2.0, label="planned path"))

    if flown is not None and len(flown) > 1:
        segments = np.stack([flown[:-1, :2], flown[1:, :2]], axis=1)
        collection = LineCollection(segments, cmap="turbo", zorder=7, linewidths=2.8)
        if flown_speed is not None and len(flown_speed) >= len(segments):
            collection.set_array(np.asarray(flown_speed, dtype=float)[: len(segments)])
            colour_label = "flown speed [m/s]"
        else:
            travelled = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(flown, axis=0), axis=1)))
            )
            collection.set_array(travelled[: len(segments)])
            colour_label = "flown arc length [m]"
        top.add_collection(collection)
        bar = figure.colorbar(collection, ax=top, fraction=0.030, pad=0.015)
        bar.set_label(colour_label, fontsize=9)
        bar.ax.tick_params(labelsize=8)
        handles.insert(
            1, Line2D([], [], color="tab:orange", lw=2.8, label="flown (colour = speed)")
        )

    top.plot(
        obs["pos"][0], obs["pos"][1], "*", ms=18, color="tab:green", mec="k", mew=0.6, zorder=8
    )
    top.plot(
        record.points[-1, 0],
        record.points[-1, 1],
        "s",
        ms=8,
        color="tab:green",
        mec="k",
        mew=0.6,
        zorder=8,
    )
    handles.extend(
        [
            Line2D([], [], marker="*", color="tab:green", ls="none", ms=14, label="start"),
            Line2D([], [], marker="s", color="tab:green", ls="none", ms=8, label="end of path"),
        ]
    )

    tightest = int(np.argmin(record.clearance))
    if np.isfinite(record.clearance[tightest]):
        top.plot(
            record.points[tightest, 0],
            record.points[tightest, 1],
            "o",
            ms=10,
            mfc="none",
            mec="tab:red",
            mew=1.8,
            zorder=9,
        )
        top.annotate(
            f"tightest {record.clearance[tightest] * 100:.0f} cm",
            xy=(record.points[tightest, 0], record.points[tightest, 1]),
            xytext=(record.points[tightest, 0] - 0.62, record.points[tightest, 1] + 0.18),
            color="tab:red",
            fontsize=8.5,
            zorder=9,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.8,
                "color": "tab:red",
                "shrinkA": 0,
                "shrinkB": 6,
            },
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                mec="tab:red",
                mew=1.8,
                ms=9,
                label="tightest planned clearance",
            )
        )

    top.set_aspect("equal")
    top.set_xlabel("x [m]")
    top.set_ylabel("y [m]")
    top.set_title(heading or "Track from above", fontsize=11)
    top.grid(alpha=0.20)
    top.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        fontsize=8.5,
        framealpha=1.0,
        borderaxespad=0.0,
    )

    figure.suptitle(title or "Planned path", fontsize=12, y=1.0)
    if subtitle:
        figure.text(0.42, 0.955, subtitle, ha="center", va="top", fontsize=10, color="0.25")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return record
