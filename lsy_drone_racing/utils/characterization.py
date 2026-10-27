"""Run and measure the scenario manoeuvres that characterise the drone.

Each scenario is one controlled flight with an analytic reference — a line, a circle, a corner or a
lateral step — flown by the racing controller with the planner out of the loop. The numbers this
module returns are the drone's capability envelope, which is what the controller and the planner
parameters should be set from.

The logged arrays use the same names as a race episode, so `path_plot.plot_run` and the run-log
tooling read them unchanged. The verdict differs on purpose: a scenario ends by finishing its path
or by running out of time, and neither is a race outcome.

Typical use is `scripts/characterize.py`, which runs the whole matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lsy_drone_racing.control.scenario_path import ScenarioPath, build, parse_spec

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Fraction of the peak speed above which a run counts as cruising, for steady-state averages.
CRUISE_FRACTION = 0.9

#: Fraction of a run skipped at each end when averaging a steady state.
STEADY_EDGE = 0.25

#: Thresholds at which a lateral step counts as arrived, in meters.
STEP_TOLERANCES = (0.05, 0.02)

#: Radius of the drone's collision box half-diagonal, used to relate tilt to reach.
BOX_CORNER = float(np.hypot(0.07, 0.07))


@dataclass
class ScenarioResult:
    """One measured scenario.

    Attributes:
        spec: The scenario specification that was flown.
        verdict: ``"SCENARIO_COMPLETE"``, ``"SCENARIO_TIMEOUT"`` or the environment's verdict.
        arrays: The logged time series, named like a race episode's.
        metrics: The measured quantities.
        path: Optional summary of the scenario geometry for logging.
    """

    spec: str
    verdict: str
    arrays: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _log_arrays(
    record: dict[str, list], diagnostics: list[dict[str, float]], path: ScenarioPath
) -> dict[str, Any]:
    """Assemble the arrays of a scenario run in the race-episode naming.

    Args:
        record: Per-step lists of the drone state.
        diagnostics: Per-step controller diagnostics.
        path: The scenario path, stored as the "plan" so the plot can draw it.

    Returns:
        The arrays, ready for `numpy.savez_compressed` and for `path_plot.plot_run`.
    """
    arrays: dict[str, Any] = {key: np.asarray(value) for key, value in record.items()}
    for key in diagnostics[0] if diagnostics else ():
        arrays[key] = np.array([entry[key] for entry in diagnostics])
    times = arrays["t"]
    arrays["acc"] = (
        np.gradient(arrays["vel"], times, axis=0) if len(times) > 1 else np.zeros((0, 3))
    )
    points = path.pos(np.linspace(0.0, path.length, 200))
    arrays["plan_points"] = points[None, ...]
    arrays["plan_t"] = np.array([0.0])
    arrays["plan_tick"] = np.array([0.0])
    arrays["plan_theta"] = np.array([0.0])
    arrays["plan_length"] = np.array([path.length])
    arrays["plan_n_waypoints"] = np.array([float(len(points))])
    arrays["plan_reason"] = np.array([f"scenario: {path.__class__.__name__}"], dtype="<U120")
    arrays["plan_drone_pos"] = points[:1].copy()
    arrays["true_gates_pos"] = np.zeros((0, 3))
    arrays["true_obstacles_pos"] = np.zeros((0, 3))
    arrays["gates_pos"] = np.zeros((0, 3))
    arrays["gates_quat"] = np.zeros((0, 4))
    arrays["gate_sequence"] = np.zeros(0, dtype=int)
    arrays["bounds_low"] = np.array([-2.45, -1.45, 0.0])
    arrays["bounds_high"] = np.array([2.45, 1.45, 2.0])
    arrays["failure_kind"] = np.array("none")
    arrays["failure_object"] = np.array(-1)
    arrays["failure_pos"] = arrays["pos"][-1]
    arrays["failure_t"] = np.array(float(times[-1]))
    arrays["failure_distance"] = np.array(0.0)
    return arrays


def _reference_at(arrays: dict[str, Any], path: ScenarioPath) -> tuple[NDArray, NDArray]:
    """Evaluate the analytic reference at the logged path progress.

    Args:
        arrays: The logged arrays.
        path: The scenario path.

    Returns:
        The reference positions ``(n, 3)`` and the reference curvature ``(n, )``.
    """
    theta = np.clip(np.asarray(arrays["theta"], dtype=float), 0.0, path.length)
    return path.pos(theta), path.curvature(theta)


def _steady_slice(length: int) -> slice:
    """Return the middle half of a run, which excludes the entry and exit transients.

    Args:
        length: Number of samples.

    Returns:
        The slice of the run used for the steady-state averages.
    """
    low = int(STEADY_EDGE * length)
    high = max(low + 1, int((1.0 - STEADY_EDGE) * length))
    return slice(low, high)


def _cruising(speed: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Return a mask of the samples where the drone is at its cruising speed.

    This is the window for the attitude, not for the speed: on a short straight the acceleration
    ramp is most of the flight, and the pitch held while accelerating from rest says nothing about
    the attitude held at speed, which is what sets the drone's effective collision reach.

    Args:
        speed: Measured speed, shape ``(n, )``.

    Returns:
        A boolean mask over the run; all true when the run never moves.
    """
    peak = float(np.max(speed)) if len(speed) else 0.0
    cruising = np.ones(len(speed), dtype=bool) if peak <= 0 else speed >= CRUISE_FRACTION * peak
    return cruising if np.any(cruising) else np.ones(len(speed), dtype=bool)


def scenario_metrics(spec: str, arrays: dict[str, Any]) -> dict[str, float]:
    """Measure a scenario run against its analytic reference.

    The reference is evaluated at the logged path progress rather than assumed, so a run that lags
    behind the reference is measured against where the reference actually was.

    Args:
        spec: The scenario specification, for the manoeuvre geometry.
        arrays: The logged arrays of the run.

    Returns:
        The measured quantities: tracking error, speed, lateral acceleration, attitude, step
        response and solver cost.
    """
    path = build(spec)
    name = parse_spec(spec)[0]
    position = np.asarray(arrays["pos"], dtype=float)
    quat = np.asarray(arrays["quat"], dtype=float)
    speed = np.asarray(arrays["speed"], dtype=float)
    reference, curvature = _reference_at(arrays, path)
    error = position - reference
    horizontal = np.linalg.norm(error[:, :2], axis=1)
    steady = _steady_slice(len(position))

    # Tilt about x and y from the quaternion, i.e. roll and pitch.
    roll = np.degrees(
        np.arctan2(
            2 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2),
        )
    )
    pitch = np.degrees(
        np.arcsin(np.clip(2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1.0, 1.0))
    )
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    lateral_acceleration = speed**2 * curvature
    solver = np.asarray(arrays.get("solver_ms", np.zeros(0)), dtype=float)

    metrics: dict[str, float] = {
        "steps": float(len(position)),
        "t_end": float(np.asarray(arrays["t"])[-1]),
        "path_length": float(path.length),
        "progress": float(np.asarray(arrays["theta"])[-1]),
        "tracking_rms": float(np.sqrt(np.mean(horizontal**2))),
        "tracking_max": float(np.max(horizontal)),
        "tracking_steady": float(np.mean(horizontal[steady])),
        "vertical_max": float(np.max(np.abs(error[:, 2]))),
        "speed_mean": float(np.mean(speed[steady])),
        "speed_max": float(np.max(speed)),
        "v_ref_mean": float(np.mean(np.asarray(arrays["v_ref"])[steady]))
        if "v_ref" in arrays
        else float("nan"),
        "a_lat_mean": float(np.mean(lateral_acceleration[steady])),
        "a_lat_max": float(np.max(lateral_acceleration)),
        "tilt_mean": float(np.mean(tilt[steady])),
        "tilt_max": float(np.max(tilt)),
        # The peak at the start of a run is the pitch transient of accelerating from rest, which is
        # a different quantity from the attitude the drone holds while flying the manoeuvre.
        # Median, not max: the ramp crosses the cruise band briefly and would otherwise dominate.
        "tilt_cruise": float(np.median(tilt[_cruising(speed)])),
        "solver_ms_mean": float(np.mean(solver)) if len(solver) else float("nan"),
        "solver_ms_p95": float(np.percentile(solver, 95)) if len(solver) else float("nan"),
        "solver_failures": float(np.sum(np.asarray(arrays.get("solver_status", [0])) != 0)),
    }
    if metrics["a_lat_mean"] > 1e-6:
        metrics["tilt_per_accel"] = float(metrics["tilt_cruise"] / metrics["a_lat_mean"])
        metrics["box_reach"] = float(BOX_CORNER + 0.02 * np.sin(np.radians(metrics["tilt_cruise"])))

    if name == "circle" and hasattr(path, "centre"):
        radius = np.linalg.norm(position[:, :2] - path.centre[None, :2], axis=1)
        # Signed: negative means the drone cut inside its own arc.
        metrics["radial_bias"] = float(np.mean(radius[steady] - path.radius))
    metrics.update(_settle_times(horizontal, np.asarray(arrays["t"], dtype=float)))
    return metrics


def _settle_times(lateral: NDArray[np.floating], times: NDArray[np.floating]) -> dict[str, float]:
    """Measure how long the drone needs to recover from a lateral disturbance.

    The disturbance is whatever pushed the lateral error above the largest tolerance — either the
    drone starting beside its path, or the reference stepping sideways. Settling is the first moment
    after the onset from which the error stays below the tolerance for the rest of the run, so a
    run that never settles reports nothing rather than a misleading short time.

    Args:
        lateral: Magnitude of the lateral tracking error, shape ``(n, )``.
        times: Time of each sample, shape ``(n, )``.

    Returns:
        One entry per tolerance that the run actually needed, in seconds from the onset.
    """
    largest = max(STEP_TOLERANCES)
    onset_steps = np.flatnonzero(lateral > largest)
    if not len(onset_steps):
        return {}
    onset = int(onset_steps[0])
    result: dict[str, float] = {}
    for tolerance in STEP_TOLERANCES:
        above = np.flatnonzero(lateral[onset:] > tolerance)
        settled = onset + (int(above[-1]) + 1 if len(above) else 0)
        if settled >= len(lateral):
            continue
        result[f"settle_time_{int(round(tolerance * 100)):02d}"] = float(
            times[settled] - times[onset]
        )
    return result


def _require_inside_arena(path: ScenarioPath, spec: str) -> None:
    """Refuse a scenario whose reference leaves the arena.

    A reference outside the arena cannot be flown: the optimal control problem's position bounds
    make it infeasible, the drone saturates and creeps, and the run ends in a timeout that looks
    like a control failure. Failing loudly here keeps that confusion out of the measurements.

    Args:
        path: The scenario path.
        spec: The specification, for the error message.

    Raises:
        ValueError: If any sample of the path is outside the arena bounds.
    """
    samples = path.pos(np.linspace(0.0, path.length, 400))
    low, high = np.array([-2.45, -1.45]), np.array([2.45, 1.45])
    outside = np.any(samples[:, :2] < low, axis=1) | np.any(samples[:, :2] > high, axis=1)
    if np.any(outside):
        first = samples[outside][0]
        where = f"({first[0]:.2f}, {first[1]:.2f}) m"
        raise ValueError(
            f"scenario {spec!r} leaves the arena at {where};"
            " the reference must stay inside +-2.45 x +-1.45 m"
        )


def run_scenario(
    spec: str, config: Any, seed: int = 0, device: str = "cpu", verbose: bool = False
) -> ScenarioResult:
    """Fly one scenario and measure it.

    The drone is placed at the scenario path's start, so the run begins exactly on the reference:
    any error that follows is tracking error, not an initialisation transient. The environment is
    rebuilt per run because the start position is part of the track configuration.

    Args:
        spec: The scenario specification.
        config: The loaded race configuration for ``config/scenario.toml``. Mutated per run.
        seed: Environment seed.
        device: ``"cpu"`` or ``"gpu"``.
        verbose: Print the step where the environment ends the run unexpectedly.

    Returns:
        The measured scenario.
    """
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_controller

    path = build(spec)
    _require_inside_arena(path, spec)
    options = parse_spec(spec)[1]
    start = path.pos(0.0)
    offset = float(options.get("start_offset", 0.0))
    if offset:
        # Begin beside the path rather than on it, so the initial tracking error is a lateral step
        # with no preview at all: that is the reveal case, not the planned-shift case.
        heading = path.tangent(0.0)
        left = np.array([-heading[1], heading[0], 0.0])
        start = start - offset * left
    config.controller.file = "scenario_controller.py"
    config.controller.scenario = spec
    config.env.seed = seed
    config.env.track.drones[0]["pos"] = [float(start[0]), float(start[1]), float(start[2])]
    config.env.track.drones[0]["vel"] = [0.0, 0.0, 0.0]
    config.env.track.drones[0]["ang_vel"] = [0.0, 0.0, 0.0]

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
        device=device,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=seed)
    controller_cls = load_controller(
        Path(__file__).resolve().parents[2] / "lsy_drone_racing/control/scenario_controller.py"
    )
    controller = controller_cls(obs, info, config)

    record: dict[str, list] = {
        "t": [],
        "pos": [],
        "vel": [],
        "quat": [],
        "ang_vel": [],
        "des": [],
        "n": [],
        "speed": [],
    }
    diagnostics: list[dict[str, float]] = []
    steps = 0
    verdict = "SCENARIO_TIMEOUT"
    try:
        while True:
            position = np.asarray(obs["pos"], dtype=float)
            velocity = np.asarray(obs["vel"], dtype=float)
            record["t"].append(steps / config.env.freq)
            record["pos"].append(position)
            record["vel"].append(velocity)
            record["quat"].append(np.asarray(obs["quat"], dtype=float))
            record["ang_vel"].append(np.asarray(obs["ang_vel"], dtype=float))
            record["n"].append(int(obs["n_gates_passed"]))
            record["speed"].append(float(np.linalg.norm(velocity)))

            action = np.asarray(controller.compute_control(obs, info), dtype=float)
            record["des"].append(action[:3].copy())
            diagnostics.append(dict(controller.diagnostics()))
            obs, reward, terminated, truncated, info = env.step(action)
            finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            steps += 1
            if terminated or truncated:
                verdict = "SCENARIO_ABORTED"
                if verbose:
                    print(f"  scenario {spec!r} ended by the environment after {steps} steps")
                break
            if finished:
                verdict = "SCENARIO_COMPLETE" if controller.completed else "SCENARIO_TIMEOUT"
                break
    finally:
        controller.episode_callback()
        controller.episode_reset()
        env.close()

    arrays = _log_arrays(record, diagnostics, path)
    return ScenarioResult(
        spec=spec, verdict=verdict, arrays=arrays, metrics=scenario_metrics(spec, arrays)
    )


def write_result(result: ScenarioResult, out_dir: Path) -> tuple[Path, Path]:
    """Write a scenario run's arrays and its metrics next to each other.

    Args:
        result: The measured scenario.
        out_dir: Directory for the outputs.

    Returns:
        The paths of the ``.npz`` and the ``.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.spec.replace(" ", "_").replace("=", "")
    stem = "".join(character for character in stem if character.isalnum() or character in "_-.")
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **result.arrays)
    summary = {"spec": result.spec, "verdict": result.verdict, **result.metrics}
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float))
    return npz_path, json_path


def plot_scenario(result: ScenarioResult, out_path: str | Path, *, dpi: int = 140) -> Path:
    """Draw one scenario: the geometry and the three time series that explain it.

    The four panels answer what the metric table only summarises: whether the reference was actually
    reached (speed against the requested speed), what the error was and when it happened, how much
    the drone had to tilt for it, and whether the solver kept up.

    Args:
        result: The measured scenario.
        out_path: Where to write the image.
        dpi: Resolution of the saved figure.

    Returns:
        The path of the written image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    arrays, metrics = result.arrays, result.metrics
    path = build(result.spec)
    times = np.asarray(arrays["t"])
    position = np.asarray(arrays["pos"])
    speed = np.asarray(arrays["speed"])
    quat = np.asarray(arrays["quat"])
    reference, curvature = _reference_at(arrays, path)
    error = np.linalg.norm((position - reference)[:, :2], axis=1)
    roll = np.degrees(
        np.arctan2(
            2 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2),
        )
    )
    pitch = np.degrees(
        np.arcsin(np.clip(2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1))
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    geometry, speed_ax, error_ax, attitude_ax = axes.ravel()

    samples = path.pos(np.linspace(0.0, path.length, 200))
    geometry.plot(samples[:, 0], samples[:, 1], color="0.55", lw=1.4, label="reference")
    if len(position) > 1:
        segments = np.stack([position[:-1, :2], position[1:, :2]], axis=1)
        collection = LineCollection(segments, cmap="turbo", linewidths=2.4, zorder=3)
        collection.set_array(speed[:-1])
        geometry.add_collection(collection)
        bar = figure.colorbar(collection, ax=geometry, fraction=0.046, pad=0.02)
        bar.set_label("flown speed [m/s]", fontsize=8)
        bar.ax.tick_params(labelsize=7)
    geometry.plot(position[0, 0], position[0, 1], "*", ms=14, color="tab:green", zorder=4)
    geometry.plot(position[-1, 0], position[-1, 1], "s", ms=6, color="tab:red", zorder=4)
    # A straight line has no extent across itself, and an equal-aspect axes would collapse to a
    # sliver, so pad the short axis.
    span = np.maximum(samples[:, :2].max(axis=0) - samples[:, :2].min(axis=0), 1e-3)
    centre = 0.5 * (samples[:, :2].max(axis=0) + samples[:, :2].min(axis=0))
    pad = np.maximum(0.5 * span, 0.35)
    geometry.set_xlim(centre[0] - pad[0], centre[0] + pad[0])
    geometry.set_ylim(centre[1] - pad[1], centre[1] + pad[1])
    geometry.set_aspect("equal")
    geometry.set_xlabel("x [m]")
    geometry.set_ylabel("y [m]")
    geometry.set_title(result.spec, fontsize=10)
    geometry.grid(alpha=0.2)

    speed_ax.plot(times, speed, color="tab:blue", lw=1.8, label="flown")
    if "v_ref" in arrays:
        speed_ax.plot(
            times, np.asarray(arrays["v_ref"]), color="0.4", ls="--", lw=1.2, label="reference"
        )
    speed_ax.set_xlabel("t [s]")
    speed_ax.set_ylabel("speed [m/s]")
    speed_ax.set_title(f"Speed — mean {metrics['speed_mean']:.2f} m/s", fontsize=10)
    speed_ax.grid(alpha=0.2)
    speed_ax.legend(fontsize=8)

    error_ax.plot(times, error * 1000, color="tab:red", lw=1.6)
    error_ax.set_xlabel("t [s]")
    error_ax.set_ylabel("horizontal error [mm]")
    error_ax.set_title(f"Tracking — rms {metrics['tracking_rms'] * 1000:.0f} mm", fontsize=10)
    error_ax.grid(alpha=0.2)

    attitude_ax.plot(times, roll, color="tab:orange", lw=1.5, label="roll")
    attitude_ax.plot(times, pitch, color="tab:purple", lw=1.5, label="pitch")
    attitude_ax.set_xlabel("t [s]")
    attitude_ax.set_ylabel("attitude [deg]")
    attitude_ax.set_title(f"Attitude — tilt max {metrics['tilt_max']:.1f} deg", fontsize=10)
    attitude_ax.grid(alpha=0.2)
    acceleration = attitude_ax.twinx()
    acceleration.plot(times, speed**2 * curvature, color="0.35", ls=":", lw=1.3, label="v^2 k")
    acceleration.set_ylabel("lateral accel [m/s^2]", color="0.35")
    acceleration.tick_params(axis="y", colors="0.35", labelsize=8)
    handles, labels = attitude_ax.get_legend_handles_labels()
    extra, extra_labels = acceleration.get_legend_handles_labels()
    attitude_ax.legend(handles + extra, labels + extra_labels, fontsize=8, loc="lower right")

    figure.suptitle(f"{result.verdict} — {metrics['t_end']:.2f} s", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return out_path


#: Column order and heading of the summary table.
_COLUMNS = (
    ("spec", "scenario"),
    ("verdict", "verdict"),
    ("speed_mean", "v [m/s]"),
    ("v_ref_mean", "v_ref"),
    ("tracking_rms", "err rms [m]"),
    ("tracking_max", "err max [m]"),
    ("radial_bias", "radial bias [m]"),
    ("a_lat_mean", "a_lat [m/s^2]"),
    ("tilt_cruise", "tilt [deg]"),
    ("settle_time_05", "settle 5cm [s]"),
    ("settle_time_02", "settle 2cm [s]"),
    ("solver_ms_p95", "solve p95 [ms]"),
)


def scenario_table(results: list[ScenarioResult]) -> str:
    """Render the measured scenarios as a markdown table.

    Args:
        results: The measured scenarios.

    Returns:
        A markdown table with one row per scenario.
    """
    header = "| " + " | ".join(title for _, title in _COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    rows = [header, rule]
    for result in results:
        cells = []
        for key, _ in _COLUMNS:
            value = (
                result.spec
                if key == "spec"
                else result.verdict
                if key == "verdict"
                else (result.metrics.get(key))
            )
            if isinstance(value, float):
                cells.append("—" if np.isnan(value) else f"{value:.2f}")
            else:
                cells.append("—" if value is None else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
