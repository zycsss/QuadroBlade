"""Headless MPCC evaluation with per-pass times, speeds and offline flight artifacts.

Run from the project root::

    pixi run python -m tests.mpcc_evaluation --config config/level0.toml --seed 0

Select the Level2 sub6 profile with ``--controller mpcc_level2_sub6_controller.py``.

Every run gets a unique directory under plots/mpcc (or --output-dir). Gate CSV timestamps and
velocities are linear-interpolation estimates between valid environment samples, not measurements
at the simulator's internal frequency. ``gate_id`` is zero based; ``gate_number`` is one based.
The configuration on disk is never edited. Integration tests explicitly opt into native compilation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quadroblade_mpcc_mpl"))

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
GATE_FIELDS = (
    "pass_index",
    "gate_id",
    "gate_number",
    "direction",
    "status",
    "time_s",
    "split_s",
    "speed_mps",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "x_m",
    "y_m",
    "z_m",
)


class EpisodeRecorder:
    """Record transitions, including the terminal transition and missing gate passes."""

    def __init__(self, obs: dict, freq: float):
        """Initialize from t=0; retain copies because environment buffers may be reused."""
        if not np.isfinite(freq) or freq <= 0:
            raise ValueError("freq must be finite and positive")
        self.freq = float(freq)
        self.steps = 0
        self.previous = {key: np.asarray(value).copy() for key, value in obs.items()}
        self.gates = []
        for index, (gate, direction) in enumerate(
            zip(obs["gate_sequence"], obs["gate_sequence_direction"], strict=True), start=1
        ):
            row = dict.fromkeys(GATE_FIELDS)
            row.update(
                pass_index=index,
                gate_id=int(gate),
                gate_number=int(gate) + 1,
                direction=int(direction),
                status="not_passed",
            )
            self.gates.append(row)
        self.states = [self._sample(obs, True)]
        self.actions: list[np.ndarray] = []
        self.diagnostics: list[dict] = []

    @staticmethod
    def _sample(obs: dict, valid: bool) -> dict:
        """Represent reset poses with NaN so plots cannot connect through them."""
        return {
            "pos": np.asarray(obs["pos"], dtype=float).copy() if valid else np.full(3, np.nan),
            "vel": np.asarray(obs["vel"], dtype=float).copy() if valid else np.full(3, np.nan),
            "quat": np.asarray(obs["quat"], dtype=float).copy() if valid else np.full(4, np.nan),
            "n_gates_passed": int(obs["n_gates_passed"]),
            "valid_state": valid,
        }

    def record_step(
        self, action: np.ndarray, obs: dict, diagnostics: dict, *, valid_state: bool = True
    ) -> None:
        """Confirm events by gate count, then interpolate the signed plane crossing."""
        old_count = int(self.previous["n_gates_passed"])
        new_count = int(obs["n_gates_passed"])
        if new_count < old_count or new_count > len(self.gates) or new_count - old_count > 1:
            raise ValueError("unexpected gate-count transition")
        if new_count > old_count:
            if not valid_state:
                raise ValueError("cannot estimate a crossing from a disabled-drone reset pose")
            row = self.gates[old_count]
            gate = row["gate_id"]
            rotation = Rotation.from_quat(np.asarray(obs["gates_quat"])[gate])
            centre = np.asarray(obs["gates_pos"])[gate]
            before = rotation.apply(self.previous["pos"] - centre, inverse=True)
            after = rotation.apply(np.asarray(obs["pos"]) - centre, inverse=True)
            sign = row["direction"]
            if not (sign * before[0] < 0 <= sign * after[0]):
                raise ValueError("gate-count increment does not match the signed gate plane")
            alpha = float(-before[0] / (after[0] - before[0]))
            crossing_time = (self.steps + alpha) / self.freq
            velocity = (1 - alpha) * self.previous["vel"] + alpha * np.asarray(obs["vel"])
            position = (1 - alpha) * self.previous["pos"] + alpha * np.asarray(obs["pos"])
            last_time = self.gates[old_count - 1]["time_s"] if old_count else 0.0
            row.update(
                status="passed",
                time_s=crossing_time,
                split_s=crossing_time - last_time,
                speed_mps=float(np.linalg.norm(velocity)),
                vx_mps=float(velocity[0]),
                vy_mps=float(velocity[1]),
                vz_mps=float(velocity[2]),
                x_m=float(position[0]),
                y_m=float(position[1]),
                z_m=float(position[2]),
            )
        self.steps += 1
        self.actions.append(np.asarray(action, dtype=float).copy())
        self.diagnostics.append(dict(diagnostics))
        self.states.append(self._sample(obs, valid_state))
        self.previous = {key: np.asarray(value).copy() for key, value in obs.items()}

    def arrays(self) -> dict[str, np.ndarray]:
        """Return numeric arrays; actions/diagnostics belong to the start of each interval."""
        arrays = {key: np.asarray([state[key] for state in self.states]) for key in self.states[0]}
        arrays["t"] = np.arange(self.steps + 1) / self.freq
        arrays["speed"] = np.linalg.norm(arrays["vel"], axis=1)
        arrays["action_t"] = arrays["t"][:-1]
        arrays["action"] = np.asarray(self.actions, dtype=float).reshape(-1, 4)
        for key in sorted({key for row in self.diagnostics for key in row}):
            arrays[f"diag_{key}"] = np.asarray(
                [row.get(key, np.nan) for row in self.diagnostics], dtype=float
            )
        return arrays


def fly_episode(env: Any, controller: Any, obs: dict, info: dict, recorder: EpisodeRecorder) -> str:
    """Fly until completion/failure, recording each result before termination; always close."""
    try:
        while True:
            control_start = time.perf_counter()
            action = np.asarray(controller.compute_control(obs, info))
            control_ms = (time.perf_counter() - control_start) * 1000
            if action.shape != (4,) or not np.isfinite(action).all():
                raise ValueError("MPCC must produce four finite attitude/thrust commands")
            if hasattr(env, "action_space") and not env.action_space.contains(action):
                raise ValueError("MPCC command is outside the environment action space")
            diagnostics = dict(controller.diagnostics())
            diagnostics["control_ms"] = control_ms
            obs, reward, terminated, truncated, info = env.step(action)
            valid = not bool(np.asarray(env.unwrapped.data.disabled_drones).any())
            recorder.record_step(action, obs, diagnostics, valid_state=valid)
            stopped = controller.step_callback(action, obs, reward, terminated, truncated, info)
            if int(obs["n_gates_passed"]) == len(recorder.gates):
                return "FINISHED"
            if truncated:
                return "TIMEOUT"
            if terminated:
                return "TERMINATED"
            if stopped:
                return "CONTROLLER_STOPPED"
    finally:
        env.close()


def _json_value(value: Any) -> Any:
    """Serialize scientific settings without object arrays or pickle."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def evaluation_geometry(env: Any) -> dict[str, np.ndarray]:
    """Copy static true geometry for reports only; never supply it to the controller."""
    data = env.unwrapped.data
    return {
        "true_gates_pos": np.asarray(data.gates_pos[0], dtype=float).copy(),
        "true_gates_quat": np.asarray(data.gates_quat[0], dtype=float).copy(),
        "true_obstacles": np.asarray(data.obstacles_pos[0], dtype=float).copy(),
    }


def _plot(arrays: dict, reference: dict, gates: list[dict], summary: dict, output: Path) -> None:
    """Draw a standalone headless figure from exactly the data saved in the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lsy_drone_racing.utils.path_plot import _draw_gates, _draw_poles

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    try:
        view3d = figure.add_subplot(221, projection="3d")
        top = figure.add_subplot(222)
        altitude = figure.add_subplot(223)
        speed = figure.add_subplot(224)
        planned, actual = reference["points"], arrays["pos"]
        gate_pos = reference.get("true_gates_pos", reference["gates_pos"])
        gate_quat = reference.get("true_gates_quat", reference["gates_quat"])
        obstacles = reference.get("true_obstacles", reference["obstacles"])
        view3d.plot(*planned.T, "--", color="tab:green", label="Planned")
        view3d.plot(*actual.T, color="tab:blue", label="Flown")
        top.plot(*planned[:, :2].T, "--", color="tab:green", label="Planned")
        top.plot(*actual[:, :2].T, color="tab:blue", label="Flown")
        _draw_gates(top, gate_pos, gate_quat)
        # The shared helper labels zero-based IDs; match the report's one-based gate numbers.
        for text in top.texts:
            label = text.get_text()
            if label.startswith("g") and label[1:].isdigit():
                text.set_text(f"G{int(label[1:]) + 1}")
        _draw_poles(top, obstacles)
        # Show the actual gate planes in 3D as well as their post positions in top view.
        corners = np.array(
            [
                [0, -0.36, -0.36],
                [0, 0.36, -0.36],
                [0, 0.36, 0.36],
                [0, -0.36, 0.36],
                [0, -0.36, -0.36],
            ]
        )
        for centre, quat in zip(gate_pos, gate_quat, strict=True):
            frame = Rotation.from_quat(quat).apply(corners) + centre
            view3d.plot(*frame.T, color="0.5", linewidth=1)
        altitude.plot(arrays["t"], actual[:, 2], color="tab:blue", label="Altitude")
        speed.plot(arrays["t"], arrays["speed"], color="tab:blue", label="Measured speed")
        if "diag_v_ref" in arrays:
            speed.plot(arrays["action_t"], arrays["diag_v_ref"], "--", label="Reference speed")
        marked_gates: set[int] = set()
        for row in gates:
            if row["status"] != "passed":
                continue
            point = [row["x_m"], row["y_m"], row["z_m"]]
            label = f"P{row['pass_index']} / G{row['gate_number']}"
            view3d.scatter(*point, color="tab:orange", s=18)
            top.scatter(*point[:2], color="tab:orange", s=18)
            if row["gate_id"] not in marked_gates:
                passes = ", ".join(
                    f"P{item['pass_index']}"
                    for item in gates
                    if item["gate_id"] == row["gate_id"] and item["status"] == "passed"
                )
                top.annotate(
                    passes, point[:2], xytext=(5, -12), textcoords="offset points", fontsize=8
                )
                marked_gates.add(row["gate_id"])
            for axis in (altitude, speed):
                axis.axvline(row["time_s"], color="0.6", alpha=0.5, linewidth=0.8)
            speed.scatter(row["time_s"], row["speed_mps"], color="tab:orange", zorder=3)
            speed.annotate(
                label,
                (row["time_s"], row["speed_mps"]),
                xytext=(3, 6),
                textcoords="offset points",
                fontsize=8,
            )
        view3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3D trajectory")
        top.set(xlabel="x [m]", ylabel="y [m]", title="Track and ordered passes")
        top.set_aspect("equal", adjustable="datalim")
        altitude.set(xlabel="Simulation time [s]", ylabel="z [m]", title="Altitude")
        speed.set(xlabel="Simulation time [s]", ylabel="Speed [m/s]", title="Speed at each pass")
        speed.margins(x=0.1)
        for axis in (view3d, top, altitude, speed):
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        figure.suptitle(
            f"MPCC | {summary['status']} | seed {summary.get('seed', '?')} | "
            f"{summary['gates_passed']}/{summary['gates_required']} passes | "
            f"{summary['sim_time_s']:.2f} s\nGate times and speeds are interpolated estimates"
        )
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)


def write_report(
    recorder: EpisodeRecorder,
    reference: dict,
    settings: dict,
    plan_log: list[dict],
    metadata: dict,
    output_dir: Path,
) -> dict:
    """Save raw data and a summary before drawing, so plotting errors preserve flight evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = recorder.arrays()
    for key, value in reference.items():
        arrays[f"reference_{key}"] = np.asarray(value)
    for key in ("tick", "t", "theta", "length", "n_waypoints", "drone_pos", "points", "reason"):
        arrays[f"plan_{key}"] = np.asarray([plan[key] for plan in plan_log])
    passed = [row for row in recorder.gates if row["status"] == "passed"]
    completed = len(passed) == len(recorder.gates)
    solver_ms = arrays.get("diag_solver_ms", np.array([]))
    solver_ms = solver_ms[np.isfinite(solver_ms)]
    summary = {
        **metadata,
        "completed": completed,
        "gates_passed": len(passed),
        "gates_required": len(recorder.gates),
        "steps": recorder.steps,
        "sim_time_s": recorder.steps / recorder.freq,
        "lap_time_s": passed[-1]["time_s"] if completed and passed else None,
        "solver_ms_mean": float(np.mean(solver_ms)) if solver_ms.size else None,
        "solver_ms_p95": float(np.percentile(solver_ms, 95)) if solver_ms.size else None,
        "solver_failures": int(np.count_nonzero(arrays.get("diag_solver_status", []) != 0))
        if "diag_solver_status" in arrays
        else 0,
        "settings": settings,
        "gate_statistics": "time and velocity linearly interpolated between valid observations",
        "output_dir": str(output_dir.resolve()),
    }
    for label in ("control", "planning"):
        values = arrays.get(f"diag_{label}_ms", np.array([]))
        values = values[np.isfinite(values)]
        for metric, fn in (
            ("mean", np.mean),
            ("p95", lambda x: np.percentile(x, 95)),
            ("max", np.max),
        ):
            summary[f"{label}_ms_{metric}"] = float(fn(values)) if values.size else None
    np.savez_compressed(output_dir / "telemetry.npz", **arrays)
    with (output_dir / "gates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GATE_FIELDS)
        writer.writeheader()
        writer.writerows(recorder.gates)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, default=_json_value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _plot(arrays, reference, recorder.gates, summary, output_dir / "trajectory.png")
    return summary


def evaluate(
    config_path: str | Path = "config/level0.toml",
    seed: int = 0,
    output_dir: str | Path = "plots/mpcc",
    controller_file: str = "mpcc_controller.py",
    controller_overrides: dict | None = None,
) -> dict:
    """Evaluate a selected controller and save a uniquely named reproducible run directory."""
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_config, load_controller

    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    config = load_config(path)
    if controller_overrides:
        from copy import deepcopy

        config = deepcopy(config)
        for key, value in deepcopy(controller_overrides).items():
            if key == "file":
                raise ValueError("select the entry using controller_file")
            config.controller[key] = value
    config.sim.render = False
    config.env.control_mode = "attitude"
    config.env.seed = seed
    controller_path = ROOT / "lsy_drone_racing/control" / controller_file
    config.controller.file = controller_path.name
    controller_cls = load_controller(controller_path)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{path.stem}_seed{seed}_", dir=output_root))
    start = time.perf_counter()
    env = JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=seed,
            device="cpu",
        )
    )
    recorder = controller = None
    geometry = {}
    status, error = "ERROR", None
    try:
        obs, info = env.reset(seed=seed)
        geometry = evaluation_geometry(env)
        recorder = EpisodeRecorder(obs, config.env.freq)
        init_start = time.perf_counter()
        controller = controller_cls(obs, info, config)
        controller_init_ms = (time.perf_counter() - init_start) * 1000
        status = fly_episode(env, controller, obs, info, recorder)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        if controller is not None and recorder is not None:
            try:
                summary = write_report(
                    recorder,
                    {**controller.reference_snapshot(), **geometry},
                    controller.settings_snapshot(),
                    controller.plan_log,
                    {
                        "status": status,
                        "error": error,
                        "seed": seed,
                        "config": config.to_dict(),
                        "config_path": str(path.resolve()),
                        "controller_file": str(controller_path),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "wall_time_s": time.perf_counter() - start,
                        "controller_init_ms": controller_init_ms,
                    },
                    run_dir,
                )
            finally:
                try:
                    controller.episode_callback()
                finally:
                    controller.episode_reset()
    print("Pass  Gate  Dir    Time [s]  Split [s]  Speed [m/s]")
    for row in recorder.gates:
        values = "  ".join(
            f"{row[key]:9.3f}" if row[key] is not None else "        -"
            for key in ("time_s", "split_s", "speed_mps")
        )
        print(f"{row['pass_index']:4d}  {row['gate_number']:4d}  {row['direction']:+3d}  {values}")
    print(
        f"{status}: {summary['gates_passed']}/{summary['gates_required']} passes; "
        f"simulation {summary['sim_time_s']:.3f} s"
    )
    print(f"Artifacts: {run_dir}")
    return summary


def main() -> None:
    """Parse the standalone evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/level0.toml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/mpcc"))
    parser.add_argument("--controller", default="mpcc_controller.py")
    parser.add_argument(
        "--controller-overrides",
        type=json.loads,
        default=None,
        help="JSON object of in-memory controller settings",
    )
    args = parser.parse_args()
    summary = evaluate(
        args.config, args.seed, args.output_dir, args.controller, args.controller_overrides
    )
    if not summary["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
