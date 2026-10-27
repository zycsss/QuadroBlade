"""Figures for a planned path.

The planner decides the route through the track, but the simulation output did not show that
decision. This module draws it, for ``scripts/plot_path.py`` and for the per-episode figures of
``scripts/debug_run.py``, so there is a single implementation of the picture.

One figure, three panels:

* the planned path from above, over the occupancy grid the A* search sees, with the gates and their
  crossing direction. The path is coloured by arc length so that a track which visits a gate twice
  (or crosses itself) can still be followed from start to finish. If the achieved trajectory of an
  episode is given, it is drawn as a thin black line on top.
* the lateral clearance to the nearest pole along the path, which is the quantity the route search
  is supposed to improve;
* the altitude along the path, which the two dimensional search leaves to the waypoint template.

Two circles are drawn around every pole, and they are not the same thing: the solid one is
``safety_distance``, the clearance the path has to keep, and the dotted one is the slightly larger
radius the A* search blocks so that its route survives the smoothing pass. Drawing only the larger
radius made the path look as if it cut through its own constraint.
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
    """One planned route and the numbers the figure and summaries need.

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


def plan_routes(
    obs: dict[str, NDArray[np.floating]],
    modes: list[bool],
    config: ConfigDict | None = None,
    samples: int = 4000,
) -> list[PathRecord]:
    """Plan every requested mode and collect what the figure and the summaries need.

    Args:
        obs: Observation dictionary, see [observation_from_config][...observation_from_config].
        modes: One entry per plan, ``True`` to use the A* route search.
        config: Race configuration whose planner settings apply, or ``None`` for the defaults.
        samples: Number of arc length samples to evaluate for the plots.

    Returns:
        One record per plan, in the order of ``modes``.
    """
    import time

    records = []
    for use_astar in modes:
        planner = PathPlanner(path_config_from(config, use_astar=use_astar))
        start = time.perf_counter()
        path, info = planner.plan(obs, obs["pos"])
        elapsed = (time.perf_counter() - start) * 1e3
        arc = np.linspace(0.0, path.length, samples)
        points = path.pos(arc)
        records.append(
            PathRecord(
                use_astar=use_astar,
                path=path,
                gate_centers=info.gate_centers,
                obstacles=info.obstacles,
                arc=arc,
                points=points,
                clearance=pole_clearance(points, info.obstacles),
                plan_ms=elapsed,
            )
        )
    return records


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
    compare: bool = True,
    resolution: float = 0.10,
    grid: bool = True,
    dpi: int = 140,
    flown: NDArray[np.floating] | None = None,
    extra: list[tuple[str, NDArray[np.floating]]] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    config: ConfigDict | None = None,
    heading: str | None = None,
) -> list[PathRecord]:
    """Draw the planned path of a track and save it as one figure.

    Args:
        obs: Observation dictionary, see [observation_from_config][...observation_from_config].
        out_path: Where to write the image. Parent directories are created.
        compare: Whether to plan with and without the A* route search. Disable to draw only the
            planner default.
        resolution: Edge length [m] of the A* grid cell used for the occupancy overlay.
        grid: Whether to draw the occupancy overlay of the A* search.
        dpi: Resolution of the saved figure.
        flown: Optional achieved trajectory of an episode, shape ``(n, 3)``, drawn on top.
        extra: Optional further paths to draw as thin lines, each a ``(label, points)`` pair. Used
            by ``scripts/debug_run.py`` to show the plan the episode started from next to the plan
            for the poses it ended up facing.
        title: Figure title. Defaults to "Planned path".
        subtitle: Optional second title line, e.g. the verdict of an episode.
        config: Race configuration whose planner settings apply, or ``None`` for the defaults.
        heading: Heading of the top view panel. Defaults to "Planned path from above".

    Returns:
        The planned routes, in the order they were planned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    settings = path_config_from(config)
    modes = [False, True] if compare else [settings.use_astar]
    records = plan_routes(obs, modes, config)
    # The coloured route is the interesting one: the A* route when it is enabled, else the default.
    featured = next((record for record in records if record.use_astar), records[-1])

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

    figure = plt.figure(figsize=(16.0, 5.8))
    layout = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.40, wspace=0.22)
    top = figure.add_subplot(layout[:, 0])
    clearance_axis = figure.add_subplot(layout[0, 1])
    height_axis = figure.add_subplot(layout[1, 1])

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
    _draw_poles(top, obs["obstacles_pos"], settings)
    _draw_gates(top, obs)

    for record in records:
        if record is featured:
            continue
        top.plot(record.points[:, 0], record.points[:, 1], color="0.45", ls="--", lw=1.5, zorder=3)
    if flown is not None and len(flown) > 1:
        top.plot(flown[:, 0], flown[:, 1], color="k", lw=1.1, alpha=0.85, zorder=6)
    for _, points_extra in extra or []:
        top.plot(
            points_extra[:, 0],
            points_extra[:, 1],
            color="tab:cyan",
            ls="-",
            lw=1.6,
            alpha=0.9,
            zorder=4,
        )
    # The featured route is coloured by arc length so a self-crossing track stays readable.
    points = featured.points
    segments = np.stack([points[:-1, :2], points[1:, :2]], axis=1)
    collection = LineCollection(segments, cmap="viridis", zorder=5, linewidths=2.6)
    collection.set_array(featured.arc[:-1])
    top.add_collection(collection)
    figure.colorbar(collection, ax=top, fraction=0.045, pad=0.02, label="arc length [m]")

    tightest = int(np.argmin(featured.clearance))
    if np.isfinite(featured.clearance[tightest]):
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
            f"tightest {featured.clearance[tightest] * 100:.0f} cm",
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
    heading = heading or "Planned path from above"
    if overlay is not None:
        heading += f"\ngrey = A* occupancy ({resolution:.2f} m cells)"
    top.set_title(heading, fontsize=10.5)
    top.grid(alpha=0.22)
    handles = [
        Line2D([], [], color=plt.cm.viridis(0.6), lw=2.6, label="A* route (colour = arc length)"),
        Line2D([], [], color="0.45", ls="--", lw=1.5, label="heuristic push-out"),
        Line2D([], [], color="tab:green", marker="*", ms=13, ls="none", label="start"),
        Line2D([], [], color="tab:green", marker="s", ms=8, ls="none", label="end of path"),
        Line2D([], [], color="tab:red", lw=1.3, label=f"required {settings.safety_distance:.2f} m"),
        Line2D(
            [],
            [],
            color="tab:red",
            ls=":",
            lw=1.0,
            label=f"A* search margin {settings.safety_distance * settings.push_scale:.2f} m",
        ),
    ]
    if flown is not None and len(flown) > 1:
        handles.insert(2, Line2D([], [], color="k", lw=1.1, label="flown"))
    for label, _ in extra or []:
        handles.insert(0, Line2D([], [], color="tab:cyan", lw=1.6, label=label))
    if not any(record.use_astar for record in records):
        handles = [handle for handle in handles if "A*" not in handle.get_label()]
    top.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)

    for record in records:
        clearance_axis.plot(
            record.arc,
            record.clearance,
            color="tab:blue" if record.use_astar else "0.45",
            ls="-" if record.use_astar else "--",
            lw=2.0 if record.use_astar else 1.5,
            label="A* route" if record.use_astar else "heuristic push-out",
        )
    clearance_axis.axhline(
        settings.safety_distance,
        color="tab:red",
        ls=":",
        lw=1.2,
        label=f"safety_distance {settings.safety_distance:.2f} m",
    )
    clearance_axis.set_xlabel("arc length [m]")
    clearance_axis.set_ylabel("clearance to nearest pole [m]")
    clearance_axis.set_ylim(bottom=0.0)
    clearance_axis.set_title("Lateral clearance along the path", fontsize=10.5)
    clearance_axis.grid(alpha=0.22)
    clearance_axis.legend(fontsize=8, loc="lower right", framealpha=0.95)

    for record in records:
        height_axis.plot(
            record.arc,
            record.points[:, 2],
            color="tab:blue" if record.use_astar else "0.45",
            ls="-" if record.use_astar else "--",
            lw=2.0 if record.use_astar else 1.5,
            label="A* route" if record.use_astar else "heuristic push-out",
        )
    for centre in obs["gates_pos"]:
        height_axis.axhline(centre[2], color="tab:orange", lw=0.7, ls=":", alpha=0.6, zorder=0)
    height_axis.set_xlabel("arc length [m]")
    height_axis.set_ylabel("z [m]")
    height_axis.set_title("Altitude along the path (dotted = gate heights)", fontsize=10.5)
    height_axis.grid(alpha=0.22)
    height_axis.legend(fontsize=8, loc="upper left", framealpha=0.95)

    figure.suptitle(title or "Planned path", fontsize=12, y=0.99)
    if subtitle:
        # Anchored at its top edge below the title so the two never collide, which they did when the
        # subtitle was centred a few hundredths of a figure unit under the suptitle.
        figure.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=10, color="0.25")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return records
