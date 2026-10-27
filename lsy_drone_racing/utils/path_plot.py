"""Figures for a logged run.

The planner decides the route through the track and the controller decides how to fly it, but the
simulation output showed neither. This module draws a run from the ``.npz`` that
``scripts/debug_run.py`` writes, so the picture can be regenerated offline, compared between runs,
and never disagrees with the data it came from.

One view from above, over the track the drone actually faced:

* the achieved trajectory, coloured by its speed — the quantity that says how the flight went
  rather than how it was supposed to go, with the finishing time in the title and beside the finish
  marker;
* every pole with a circle at ``POLE_RADIUS + DRONE_RADIUS``: the distance at which the drone would
  touch it. The planner's safety margin and the radius its A* search blocks are deliberately not
  drawn, because they made a 3 cm pole look like a 42 cm one;
* optionally the last path the controller committed to, which the log stores as ``plan_points``.

The legend sits outside the axes so it never covers the track.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lsy_drone_racing.control.astar import (
    DRONE_RADIUS,
    GATE_FRAME_HALF,
    GATE_OPENING_HALF,
    POLE_RADIUS,
    ArenaGrid,
    AstarConfig,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Planner defaults used only to place the A* occupancy overlay. The log does not record the planner
#: settings, and the overlay is a hint about what the search saw, not part of the measurement.
_OVERLAY_CLEARANCE = 0.15
_OVERLAY_OVERSHOOT = 1.4
_OVERLAY_BOUNDS = ((-2.45, -1.45), (2.45, 1.45))


def load_log(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one run log.

    Args:
        path: Path of the ``.npz`` written by ``scripts/debug_run.py``. A ``.json`` of the same name
            next to it is read too when present, for the verdict and the timing.

    Returns:
        The arrays of the run and its summary. The summary is empty if there is no ``.json``.

    Raises:
        FileNotFoundError: If the ``.npz`` does not exist.
    """
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(f"no run log at {log_path}")
    with np.load(log_path) as data:
        arrays = {key: data[key] for key in data.files}
    summary_path = log_path.with_suffix(".json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return arrays, summary


def finish_time(arrays: dict[str, Any], summary: dict[str, Any]) -> float:
    """Return the finishing time of a run in seconds.

    Args:
        arrays: The arrays of the run.
        summary: Its summary, empty if there was no ``.json``.

    Returns:
        The simulated time at the end of the run.
    """
    if "sim_time" in summary:
        return float(summary["sim_time"])
    times = np.asarray(arrays["t"], dtype=float)
    if not len(times):
        return 0.0
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return float(times[-1] + step)


def _draw_gates(ax: Axes, gates_pos: np.ndarray, gates_quat: np.ndarray) -> None:
    """Draw the gate frames, their opening and their crossing direction."""
    from scipy.spatial.transform import Rotation as R

    for gate_id, (centre, quat) in enumerate(zip(gates_pos, gates_quat, strict=True)):
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
        ax.plot(opening[:, 0], opening[:, 1], color="tab:grey", lw=1.3, ls=":", zorder=4)
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


def _draw_poles(ax: Axes, obstacles: np.ndarray) -> None:
    """Draw every obstacle with the radius at which the drone would touch it."""
    import matplotlib.pyplot as plt

    contact = POLE_RADIUS + DRONE_RADIUS
    for pole_id, pole in enumerate(obstacles):
        ax.add_patch(
            plt.Circle((pole[0], pole[1]), contact, fill=False, ec="tab:red", lw=1.3, zorder=2)
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


def failure_of(arrays: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ending of a run when it did not finish, otherwise ``None``.

    Args:
        arrays: The arrays of a run.

    Returns:
        A record with the ``kind`` of ending, the ``object`` it happened against, its ``pos`` and
        ``t``, and the ``distance`` to that object. ``None`` for a finished run or an older log.
    """
    if "failure_kind" not in arrays:
        return None
    kind = str(arrays["failure_kind"])
    if kind in ("", "none", "FINISHED"):
        return None
    return {
        "kind": kind,
        "object": int(arrays["failure_object"]),
        "pos": np.asarray(arrays["failure_pos"], dtype=float),
        "t": float(arrays["failure_t"]),
        "distance": float(arrays["failure_distance"]),
    }


def _draw_failure_overview(
    ax: Axes, failure: dict[str, Any], obstacles: np.ndarray, gates_pos: np.ndarray
) -> None:
    """Mark the ending on the track and highlight the object it happened against.

    Args:
        ax: The axes to draw on.
        failure: The record from [failure_of][...failure_of].
        obstacles: Obstacle positions, shape ``(n, 3)``.
        gates_pos: Gate positions, shape ``(n, 3)``.
    """
    import matplotlib.pyplot as plt

    pos = failure["pos"]
    kind = failure["kind"]
    index = failure["object"]
    ax.plot(pos[0], pos[1], "X", ms=15, color="tab:red", mec="k", mew=0.7, zorder=10)
    hit_gate = kind == "HIT_GATE" or (kind == "CONTACT" and index >= len(obstacles))
    if kind in ("HIT_OBSTACLE", "CONTACT") and index < len(obstacles):
        pole = obstacles[index]
        ax.add_patch(
            plt.Circle(
                (pole[0], pole[1]),
                POLE_RADIUS + DRONE_RADIUS,
                fill=False,
                ec="tab:red",
                lw=2.6,
                zorder=9,
            )
        )
        label = f"{kind} on pole o{index}"
    elif hit_gate and index < len(gates_pos):
        ax.plot(
            gates_pos[index, 0],
            gates_pos[index, 1],
            "o",
            ms=13,
            mfc="none",
            mec="tab:red",
            mew=2.4,
            zorder=9,
        )
        label = f"{kind} at gate g{index}"
    else:
        label = kind
    ax.annotate(
        f"{label} at {failure['t']:.2f} s",
        xy=(pos[0], pos[1]),
        xytext=(pos[0] - 0.15, pos[1] - 0.38),
        color="tab:red",
        fontsize=9.5,
        fontweight="bold",
        zorder=11,
        arrowprops={"arrowstyle": "-", "lw": 0.9, "color": "tab:red", "shrinkA": 0, "shrinkB": 6},
    )


def _draw_failure_detail(
    ax: Axes,
    failure: dict[str, Any],
    flown: np.ndarray,
    obstacles: np.ndarray,
    gates_pos: np.ndarray,
    gates_quat: np.ndarray,
) -> None:
    """Zoom on the ending: the object at true scale, either side-on for a pole or in the gate plane.

    For a pole the view is a local top-down window at real scale, which shows how far the drone got
    inside the contact radius. For a gate the view is the gate's own plane, lateral against vertical
    offset, with the square frame ring drawn where it is, so a miss through the posts is visible as
    a point inside the ring rather than next to the opening.

    Args:
        ax: The axes to draw the zoom on.
        failure: The record from [failure_of][...failure_of].
        flown: The achieved trajectory, shape ``(n, 3)``.
        obstacles: Obstacle positions, shape ``(n, 3)``.
        gates_pos: Gate positions, shape ``(n, 3)``.
        gates_quat: Gate orientations as xyzw quaternions, shape ``(n, 4)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.spatial.transform import Rotation as R

    kind, index = failure["kind"], failure["object"]
    if kind in ("HIT_OBSTACLE", "CONTACT") and index < len(obstacles):
        pole = obstacles[index]
        window = 0.45
        ax.plot(flown[:, 0], flown[:, 1], color="k", lw=1.2, zorder=3)
        ax.add_patch(plt.Circle((pole[0], pole[1]), POLE_RADIUS, fc="tab:purple", zorder=5))
        ax.add_patch(
            plt.Circle(
                (pole[0], pole[1]),
                POLE_RADIUS + DRONE_RADIUS,
                fill=False,
                ec="tab:red",
                lw=1.8,
                zorder=4,
            )
        )
        distance = np.linalg.norm(flown[:, :2] - pole[None, :2], axis=1)
        closest = int(np.argmin(distance))
        ax.plot(flown[closest, 0], flown[closest, 1], "o", ms=8, mfc="none", mec="tab:red", mew=1.8)
        ax.text(
            0.03,
            0.04,
            f"closest {distance[closest]:.3f} m\ncontact at {POLE_RADIUS + DRONE_RADIUS:.3f} m",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:red",
            va="bottom",
        )
        ax.set_xlim(pole[0] - window, pole[0] + window)
        ax.set_ylim(pole[1] - window, pole[1] + window)
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.set_title(f"Pole o{index} at true scale", fontsize=10)
    elif index < len(gates_pos):
        centre, quat = gates_pos[index], gates_quat[index]
        local = R.from_quat(quat).apply(flown - centre, inverse=True)
        ax.plot(local[:, 1], local[:, 2], color="k", lw=1.2, zorder=3)
        # The frame is a square ring: the solid band runs from the opening edge to the outer edge on
        # all four sides. Drawn in the gate's own plane, a miss through the posts is a point inside
        # the band rather than a line next to the opening.
        low, high = GATE_OPENING_HALF, GATE_FRAME_HALF
        bars = (
            (low, -high, high - low, 2 * high),
            (-high, -high, high - low, 2 * high),
            (-high, low, 2 * high, high - low),
            (-high, -high, 2 * high, high - low),
        )
        for x0, y0, width, height in bars:
            ax.add_patch(Rectangle((x0, y0), width, height, fc="0.55", ec="0.2", lw=0.8, zorder=4))
        extent = max(GATE_FRAME_HALF, np.abs(local[:, 1:]).max()) + 0.08
        ax.axhline(0.0, color="0.8", lw=0.6, zorder=1)
        ax.axvline(0.0, color="0.8", lw=0.6, zorder=1)
        # Where the path crosses this gate's plane is what decides a hit, not how close it ever got
        # to the centre somewhere else along the leg.
        crossing = int(np.argmin(np.abs(local[:, 0])))
        lateral, vertical = local[crossing, 1], local[crossing, 2]
        inside = max(abs(lateral), abs(vertical)) <= GATE_OPENING_HALF
        ax.plot(lateral, vertical, "o", ms=8, mfc="none", mec="tab:red", mew=1.8, zorder=6)
        ax.text(
            0.03,
            0.04,
            f"crossing {lateral:+.2f}, {vertical:+.2f} m\n"
            f"{'inside' if inside else 'outside'} the {2 * GATE_OPENING_HALF:.2f} m opening",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:red",
            va="bottom",
        )
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("lateral offset [m]", fontsize=9)
        ax.set_ylabel("vertical offset [m]", fontsize=9)
        ax.set_title(f"Gate g{index} in its own plane", fontsize=10)
    else:
        ax.axis("off")
        return
    ax.set_aspect("equal")
    ax.grid(alpha=0.20)


def _plot_arrays(
    arrays: dict[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    heading: str | None = None,
    planned: bool = False,
    grid: bool = True,
    resolution: float = 0.10,
    dpi: int = 140,
    finish: float | None = None,
) -> Path:
    """Draw a run from its logged arrays and save the figure.

    Args:
        arrays: The arrays of a run, as written by ``scripts/debug_run.py``.
        out_path: Where to write the image. Parent directories are created.
        title: Figure title. Defaults to "Run".
        subtitle: Optional second title line, e.g. the verdict.
        heading: Heading of the plot. Defaults to "Track from above".
        planned: Whether to draw the last path the controller committed to, from ``plan_points``.
        grid: Whether to draw the occupancy overlay of the A* search.
        resolution: Edge length [m] of the A* grid cell used for the occupancy overlay.
        dpi: Resolution of the saved figure.
        finish: Finishing time in seconds. Taken from the ``t`` array when omitted.

    Returns:
        The path of the written image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    obstacles = np.asarray(arrays["true_obstacles_pos"], dtype=float)
    gates_pos = np.asarray(arrays["true_gates_pos"], dtype=float)
    gates_quat = np.asarray(arrays["gates_quat"], dtype=float)
    flown = np.asarray(arrays["pos"], dtype=float)
    speed = np.asarray(arrays["speed"], dtype=float)
    elapsed = float(finish) if finish is not None else float(np.asarray(arrays["t"])[-1])

    overlay = None
    if grid:
        from scipy.spatial.transform import Rotation as R

        y_axes = R.from_quat(gates_quat).as_matrix()[:, :, 1]
        overlay = ArenaGrid(
            obstacles,
            gates_pos,
            y_axes,
            AstarConfig(
                resolution=resolution,
                pole_clearance=_OVERLAY_CLEARANCE,
                overshoot=_OVERLAY_OVERSHOOT,
                bounds_low=_OVERLAY_BOUNDS[0],
                bounds_high=_OVERLAY_BOUNDS[1],
            ),
        )

    # A failed run gets a second panel below the track, zoomed on what it failed against. It is a
    # subplot rather than an absolutely placed axes because the saved figure is cropped to its
    # content, which moves anything positioned in figure coordinates.
    failure = failure_of(arrays)
    if failure is None:
        figure, top = plt.subplots(figsize=(13.0, 6.4))
        detail_ax = None
    else:
        figure = plt.figure(figsize=(13.0, 8.6))
        layout = figure.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.28)
        top = figure.add_subplot(layout[0])
        detail_ax = figure.add_subplot(layout[1])
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

    _draw_poles(top, obstacles)
    _draw_gates(top, gates_pos, gates_quat)
    handles.extend(
        [
            Line2D([], [], color="0.20", lw=5.0, label="gate frame"),
            Line2D([], [], color="tab:grey", lw=1.3, ls=":", label="gate opening"),
            Line2D([], [], color="tab:orange", lw=1.0, label="gate crossing direction"),
            Line2D([], [], marker="o", color="tab:purple", ls="none", ms=5, label="pole"),
            Line2D(
                [],
                [],
                color="tab:red",
                lw=1.3,
                label=f"contact radius {POLE_RADIUS + DRONE_RADIUS:.3f} m (pole + drone)",
            ),
        ]
    )

    if planned and "plan_points" in arrays and len(arrays["plan_points"]):
        # Every plan starts at the original take-off point and is anchored by path progress, so most
        # of the last plan is already behind the drone. Draw only the part it still had to fly.
        last = np.asarray(arrays["plan_points"][-1], dtype=float)
        theta = float(np.asarray(arrays["plan_theta"])[-1])
        length = float(np.asarray(arrays["plan_length"])[-1])
        first = int(np.clip(round(theta / max(length, 1e-9) * (len(last) - 1)), 0, len(last) - 1))
        remaining = last[first:]
        top.plot(remaining[:, 0], remaining[:, 1], color="k", lw=1.8, zorder=3)
        handles.insert(
            0, Line2D([], [], color="k", lw=1.8, label="last plan, from where it was committed")
        )

    if len(flown) > 1:
        segments = np.stack([flown[:-1, :2], flown[1:, :2]], axis=1)
        collection = LineCollection(segments, cmap="turbo", zorder=7, linewidths=2.8)
        collection.set_array(speed[: len(segments)])
        top.add_collection(collection)
        bar = figure.colorbar(collection, ax=top, fraction=0.030, pad=0.015)
        bar.set_label("flown speed [m/s]", fontsize=9)
        bar.ax.tick_params(labelsize=8)
        handles.insert(
            int(planned), Line2D([], [], color="tab:orange", lw=2.8, label="flown (colour = speed)")
        )

    top.plot(flown[0, 0], flown[0, 1], "*", ms=18, color="tab:green", mec="k", mew=0.6, zorder=8)
    top.plot(flown[-1, 0], flown[-1, 1], "s", ms=8, color="tab:green", mec="k", mew=0.6, zorder=8)
    top.annotate(
        f"{'finish' if failure is None else 'ended'} {elapsed:.2f} s",
        xy=(flown[-1, 0], flown[-1, 1]),
        xytext=(flown[-1, 0] + 0.12, flown[-1, 1] - 0.04),
        color="tab:green" if failure is None else "tab:red",
        fontsize=9.5,
        zorder=9,
    )
    handles.extend(
        [
            Line2D([], [], marker="*", color="tab:green", ls="none", ms=14, label="start"),
            Line2D(
                [],
                [],
                marker="s" if failure is None else "X",
                color="tab:green" if failure is None else "tab:red",
                ls="none",
                ms=8 if failure is None else 13,
                label="finish" if failure is None else "where it failed",
            ),
        ]
    )

    if failure is not None:
        _draw_failure_overview(top, failure, obstacles, gates_pos)
        _draw_failure_detail(detail_ax, failure, flown, obstacles, gates_pos, gates_quat)

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

    outcome = "finished in" if failure is None else "failed after"
    figure.suptitle(f"{title or 'Run'} — {outcome} {elapsed:.2f} s", fontsize=12.5, y=0.99)
    if subtitle:
        figure.text(0.42, 0.945, subtitle, ha="center", va="top", fontsize=10, color="0.25")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def plot_run(
    log_path: str | Path,
    out_path: str | Path,
    *,
    title: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Path:
    """Draw a run from the log that ``scripts/debug_run.py`` wrote.

    This is the entry point: everything the figure shows is read back from ``log_path``, so a
    picture can never disagree with the run it describes and two runs are drawn from identical data.

    Args:
        log_path: Path of the ``.npz``. A ``.json`` of the same name next to it supplies the verdict
            and the finishing time.
        out_path: Where to write the image.
        title: Figure title. Defaults to the log file name.
        description: Optional second title line. Defaults to the verdict from the summary.
        **kwargs: Passed on to [_plot_arrays][..._plot_arrays], e.g. ``planned``, ``grid``,
            ``resolution`` or ``dpi``.

    Returns:
        The path of the written image.
    """
    arrays, summary = load_log(log_path)
    if description is None and summary:
        description = f"{summary.get('verdict', '?')}: {summary.get('detail', '')}".strip(": ")
    return _plot_arrays(
        arrays,
        out_path,
        title=title or Path(log_path).stem,
        subtitle=description,
        finish=finish_time(arrays, summary),
        **kwargs,
    )
