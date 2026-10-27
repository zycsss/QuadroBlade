"""Headless debugging harness for drone racing controllers.

This script runs a controller for one or more episodes, records the flight, classifies *why* an
episode ended, and plots the trajectory. It deliberately does not render anything, so it works over
SSH and on machines without a display.

Run as:

    $ python scripts/debug_run.py --config level0.toml --controller state_controller.py

inside the repository's pixi environment, or from outside of it with

    $ pixi run python scripts/debug_run.py --config level0.toml --controller my_controller.py

For GPU execution use the ``gpu`` environment and pass ``--device gpu`` explicitly. The environment
defaults to CPU and the TOML config cannot change that, so the ``gpu`` environment alone would still
run on CPU:

    $ pixi run -e gpu python scripts/debug_run.py --config level0.toml --device gpu

Output:

* A one-line verdict per episode printed to stdout.
* ``plots/<config>_<controller>_seed<seed>.png`` with three panels: top-down track, altitude
  and gate progress over time, and speed over time.

Note:
    When an episode terminates because the drone was disabled, the environment warps the drone
    below the ground (``pos = [-1, -1, -1]``) *before* reporting the observation. This script
    therefore records the state at the *start* of every step, so the last recorded pose is the last
    live pose of the drone, not the warped one.
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import logging
from pathlib import Path
from typing import Any

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_OPENING = 0.4  # m, outer frame is 0.72 m
GATE_OUTER = 0.72
GATE_PASS_BOX = 0.45  # m, the box used by the environment's pass check


def gate_geometry(config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal gate positions and xyzw quaternions from a config."""
    gates = config.env.track.gates
    pos = np.array([g["pos"] for g in gates], dtype=float)
    quat = R.from_euler("xyz", np.array([g["rpy"] for g in gates])).as_quat()
    return pos, quat


def crossing_margin(
    prev_pos: np.ndarray, pos: np.ndarray, gate_pos: np.ndarray, gate_quat: np.ndarray
) -> dict[str, float]:
    """Reproduce the environment's gate test for one step and report the margin.

    The environment crosses gates along the gate frame's x-axis and accepts the pass if the
    interpolated crossing point lies inside a ``GATE_PASS_BOX`` sized square in the gate's y/z
    plane. This returns the local crossing coordinates and the remaining margin in metres.
    """
    rot = R.from_quat(gate_quat)
    last_local = rot.apply(prev_pos - gate_pos, inverse=True)
    local = rot.apply(pos - gate_pos, inverse=True)
    denom = local[0] - last_local[0]
    alpha = -last_local[0] / denom if abs(denom) > 1e-12 else 0.0
    y_int = alpha * local[1] + (1 - alpha) * last_local[1]
    z_int = alpha * local[2] + (1 - alpha) * last_local[2]
    half = GATE_PASS_BOX / 2
    return {
        "y": float(y_int),
        "z": float(z_int),
        "margin": float(half - max(abs(y_int), abs(z_int))),
    }


def classify(
    n_passed: int,
    n_required: int,
    truncated: bool,
    last_pos: np.ndarray,
    last_vel: np.ndarray,
    gates_pos: np.ndarray,
    obstacles_pos: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
) -> tuple[str, str]:
    """Guess why an episode ended and explain the evidence.

    Args:
        n_passed: Number of gate-order entries passed at the end of the episode.
        n_required: Total number of entries that must be passed.
        truncated: Whether the episode hit the step limit.
        last_pos: Last live drone position (before any warping).
        last_vel: Last live drone velocity.
        gates_pos: Nominal gate positions.
        obstacles_pos: Nominal obstacle positions.
        bounds_low: Lower position bound at which the environment disables the drone.
        bounds_high: Upper position bound at which the environment disables the drone.

    Returns:
        A ``(verdict, detail)`` tuple.
    """
    if n_passed >= n_required:
        return "FINISHED", f"all {n_required} gate passes completed"
    if truncated:
        return "TIMEOUT", f"hit the step limit with {n_passed}/{n_required} passes"
    outside = last_pos < bounds_low
    outside |= last_pos > bounds_high
    if np.any(outside):
        axes = np.array(["x", "y", "z"])[outside]
        return "OUT_OF_BOUNDS", f"last live pos {np.round(last_pos, 2)} left bounds on {axes}"
    d_obs = np.linalg.norm(obstacles_pos - last_pos, axis=1)
    d_gate = np.linalg.norm(gates_pos - last_pos, axis=1)
    if d_obs.min() < 0.15:
        return "HIT_OBSTACLE", f"closest obstacle {d_obs.argmin()} at {d_obs.min():.3f} m"
    if d_gate.min() < 0.45:
        return "HIT_GATE", f"closest gate {d_gate.argmin()} centre at {d_gate.min():.3f} m"
    predicted = last_pos + last_vel / 50
    return (
        "CONTACT",
        f"pos {np.round(last_pos, 2)} vel {np.round(last_vel, 2)} -> {np.round(predicted, 2)}",
    )


def run_episode(
    config: Any, controller_cls: type, seed: int, verbose: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    """Run a single episode and record it.

    Args:
        config: Race configuration.
        controller_cls: Controller class to instantiate.
        seed: Random seed for the episode.
        verbose: Print per-gate-pass margins.
        device: ``"cpu"`` or ``"gpu"``. Note that the environment defaults to ``"cpu"``; the
            ``device`` cannot be selected from the TOML config.
    """
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
    controller = controller_cls(obs, info, config)
    gates_pos, gates_quat = gate_geometry(config)
    obstacles_pos = np.array([o["pos"] for o in config.env.track.obstacles], dtype=float)
    gate_sequence = np.abs(np.asarray(config.env.track.gate_order)) - 1
    bounds_low = np.asarray(env.unwrapped.data.pos_limit_low, dtype=float)
    bounds_high = np.asarray(env.unwrapped.data.pos_limit_high, dtype=float)

    rec: dict[str, list] = {
        "t": [],
        "pos": [],
        "vel": [],
        "quat": [],
        "ang_vel": [],
        "des": [],
        "n": [],
        "speed": [],
    }
    diag: list[dict[str, float]] = []
    n_steps = 0
    terminated = truncated = stopped = False
    try:
        while True:
            pos = np.asarray(obs["pos"], dtype=float)
            vel = np.asarray(obs["vel"], dtype=float)
            rec["t"].append(n_steps / config.env.freq)
            rec["pos"].append(pos)
            rec["vel"].append(vel)
            rec["quat"].append(np.asarray(obs["quat"], dtype=float))
            rec["ang_vel"].append(np.asarray(obs["ang_vel"], dtype=float))
            rec["n"].append(int(obs["n_gates_passed"]))
            rec["speed"].append(float(np.linalg.norm(vel)))

            action = np.asarray(controller.compute_control(obs, info), dtype=float)
            rec["des"].append(action[:3].copy())
            # Whatever the controller exposes about its own step, e.g. path progress, reference
            # speed, solve time and solver status. Optional, so other controllers still work.
            step_diagnostics = getattr(controller, "diagnostics", None)
            if step_diagnostics is not None:
                diag.append(dict(step_diagnostics()))
            obs, reward, terminated, truncated, info = env.step(action)
            finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            n_steps += 1
            # Mirror sim.py: a truthy step_callback return ends the episode. Note that the base
            # class default returns True, so a controller that forgets to override it stops after
            # the first step.
            if terminated or truncated or finished:
                stopped = bool(finished) and not (terminated or truncated)
                break
    finally:
        controller.episode_callback()
        controller.episode_reset()
        env.close()

    rec["pos"] = np.array(rec["pos"])
    rec["vel"] = np.array(rec["vel"])
    rec["quat"] = np.array(rec["quat"])
    rec["ang_vel"] = np.array(rec["ang_vel"])
    rec["des"] = np.array(rec["des"])
    rec["n"] = np.array(rec["n"])
    rec["speed"] = np.array(rec["speed"])
    rec["t"] = np.array(rec["t"])
    # The environment only reports position and velocity, so the acceleration is a finite difference
    # of the recorded velocity at the control rate.
    rec["acc"] = (
        np.gradient(rec["vel"], rec["t"], axis=0) if len(rec["t"]) > 1 else np.zeros((0, 3))
    )
    for key in diag[0] if diag else ():
        rec[key] = np.array([entry[key] for entry in diag])

    n_final = int(np.asarray(obs["n_gates_passed"]))
    if stopped and n_final < len(gate_sequence):
        verdict, detail = (
            "CONTROLLER_STOPPED",
            "step_callback returned True before the race finished "
            "(did you forget to override it? the base class returns True)",
        )
    else:
        verdict, detail = classify(
            n_final,
            len(gate_sequence),
            bool(truncated),
            rec["pos"][-1],
            rec["vel"][-1],
            gates_pos,
            obstacles_pos,
            bounds_low,
            bounds_high,
        )
    # Gate crossing quality: for every increment of n_gates_passed find the step pair that crossed.
    crossings = []
    for i in range(len(rec["n"]) - 1):
        for k in range(rec["n"][i], rec["n"][i + 1]):
            gate_id = int(gate_sequence[k])
            info_ = crossing_margin(
                rec["pos"][i], rec["pos"][i + 1], gates_pos[gate_id], gates_quat[gate_id]
            )
            crossings.append({"pass": k, "gate": gate_id, "t": rec["t"][i + 1], **info_})

    if verbose:
        for c in crossings:
            print(
                f"  pass {c['pass']} (gate {c['gate']}) at t={c['t']:.2f}s "
                f"offset y={c['y']:+.3f} z={c['z']:+.3f} m, margin {c['margin'] * 100:+.1f} cm"
            )

    return {
        "verdict": verdict,
        "detail": detail,
        "n_final": n_final,
        "n_required": len(gate_sequence),
        "steps": n_steps,
        "sim_time": n_steps / config.env.freq,
        "crossings": crossings,
        # The poses the drone actually faced, which differ from the nominal ones above on a
        # randomized track. The controller only learns them within the sensor range.
        "true_gates_pos": np.asarray(env.unwrapped.data.gates_pos[0], dtype=float),
        "true_obstacles_pos": np.asarray(env.unwrapped.data.obstacles_pos[0], dtype=float),
        # Every path the controller committed to, with when, where and why.
        "plan_log": list(getattr(controller, "plan_log", [])),
        "record": rec,
        "gates_pos": gates_pos,
        "gates_quat": gates_quat,
        "obstacles_pos": obstacles_pos,
        "gate_sequence": gate_sequence,
        "bounds_low": bounds_low,
        "bounds_high": bounds_high,
    }


def episode_arrays(ep: dict[str, Any]) -> dict[str, Any]:
    """Assemble the arrays of an episode, exactly as they are written to a run log.

    The arrays hold per control step: time, position, velocity, acceleration (a finite difference of
    the recorded velocity, since acceleration is not observed), attitude, angular rate, the applied
    attitude command, the achieved speed, the number of gates passed, and whatever the controller
    reports through its ``diagnostics``. Every path the controller committed to is stored separately
    as ``plan_*`` arrays, with the time, tick, reason, drone position, path progress and 200 samples
    of the path itself, so a run can be replayed against the plan that was active at any step. The
    controller's ``replans`` counter is the index of the active plan.

    Args:
        ep: The episode record returned by `run_episode`.

    Returns:
        The arrays, ready for `numpy.savez_compressed` and for the plot.
    """
    rec = ep["record"]
    arrays: dict[str, Any] = {key: np.asarray(value) for key, value in rec.items()}
    arrays.update(
        {
            "gates_pos": ep["gates_pos"],
            "gates_quat": ep["gates_quat"],
            "obstacles_pos": ep["obstacles_pos"],
            "true_gates_pos": ep["true_gates_pos"],
            "true_obstacles_pos": ep["true_obstacles_pos"],
            "gate_sequence": ep["gate_sequence"],
            "bounds_low": ep["bounds_low"],
            "bounds_high": ep["bounds_high"],
        }
    )
    plans = ep["plan_log"]
    arrays["plan_reason"] = np.array([plan["reason"] for plan in plans], dtype="<U120")
    for key in ("t", "tick", "theta", "length", "n_waypoints"):
        arrays[f"plan_{key}"] = np.array([plan[key] for plan in plans], dtype=float)
    arrays["plan_drone_pos"] = np.array([plan["drone_pos"] for plan in plans], dtype=float)
    arrays["plan_points"] = np.array([plan["points"] for plan in plans], dtype=float)
    return arrays


def write_log(
    ep: dict[str, Any], arrays: dict[str, Any], out_path: Path, meta: dict[str, Any]
) -> Path:
    """Write the arrays of an episode to ``.npz`` next to a ``.json`` summary.

    Args:
        ep: The episode record returned by `run_episode`.
        arrays: The arrays of the episode, see [episode_arrays][...episode_arrays].
        out_path: Path of the ``.npz``. The summary is written next to it with a ``.json`` suffix.
        meta: Extra metadata for the summary, e.g. config, controller file and seed.

    Returns:
        The path of the written ``.npz``.
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    rec = ep["record"]
    plans = ep["plan_log"]
    solver_ms = rec.get("solver_ms", np.zeros(0))
    summary = {
        **meta,
        "verdict": ep["verdict"],
        "detail": ep["detail"],
        "sim_time": ep["sim_time"],
        "steps": ep["steps"],
        "n_final": ep["n_final"],
        "n_required": ep["n_required"],
        "crossings": ep["crossings"],
        "n_plans": len(plans),
        "plan_reasons": [plan["reason"] for plan in plans],
        "solver_ms_mean": float(np.mean(solver_ms)) if len(solver_ms) else None,
        "solver_ms_p95": float(np.percentile(solver_ms, 95)) if len(solver_ms) else None,
        "solver_failures": int(np.sum(np.asarray(rec.get("solver_status", [0])) != 0)),
        "speed_max": float(np.max(rec["speed"])) if len(rec["speed"]) else None,
        "acc_max": float(np.max(np.linalg.norm(rec["acc"], axis=1))) if len(rec["acc"]) else None,
        "arrays": sorted(arrays),
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(summary, indent=2, default=float))
    return out_path


def plot_episode(log_path: Path, out_path: Path, title: str) -> Path:
    """Plot the run that a written log holds.

    The figure is built by `lsy_drone_racing.utils.path_plot` from the log file, not from anything
    held in memory, so the picture and the log are the same data by construction.

    Args:
        log_path: The ``.npz`` this episode was just logged to.
        out_path: Where to write the image.
        title: Figure title.

    Returns:
        The path of the written image.
    """
    from lsy_drone_racing.utils.path_plot import plot_run

    return plot_run(
        log_path, out_path, title=title, heading="Track from above: colour = how it flew"
    )


def resolve_device(device: str) -> str:
    """Validate the requested device and report what JAX actually sees.

    Args:
        device: Requested device, ``"cpu"`` or ``"gpu"``.

    Returns:
        The device string to hand to the environment.

    Raises:
        SystemExit: If a GPU is requested but JAX cannot see one, with a diagnosis.
    """
    import jax

    if device == "cpu":
        return device
    try:
        devices = jax.devices("gpu")
    except Exception as e:  # JAX raises RuntimeError when no GPU backend is present
        raise SystemExit(
            f"Requested --device gpu, but JAX sees no GPU.\n"
            f"  reason: {e}\n"
            f"  active backend: {jax.default_backend()}\n"
            f'Check with: pixi run -e gpu python -c "import jax; print(jax.devices())"\n'
            "Common causes: no NVIDIA driver, /dev/nvidia* not exposed to this process "
            "(containers/sandboxes), or the wrong pixi environment."
        ) from e
    print(f"gpu: {len(devices)} device(s) -> {devices[0]}")
    return device


def main(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    seed: int = 0,
    plot: bool = True,
    out_dir: str = "plots",
    log: bool = True,
    log_dir: str = "logs",
    verbose: bool = True,
    device: str = "cpu",
    control_mode: str | None = None,
):
    """Run a controller headless and report why each episode ended.

    Args:
        config: Race config file name in ``config/``.
        controller: Controller file name in ``lsy_drone_racing/control/``. Defaults to the one set
            in the config.
        n_runs: Number of episodes to run.
        seed: Base random seed. Episode ``i`` uses ``seed + i``.
        plot: Whether to write a PNG per episode. The figure is drawn from the run log by
            `lsy_drone_racing.utils.path_plot`, so the log is written even when `log` is False.
        out_dir: Directory for plots (``plots/`` is gitignored).
        log: Whether to report the ``.npz`` of the run and its ``.json`` summary per episode. The
            files are written whenever this or `plot` is set, because the figure needs them.
        log_dir: Directory for those logs (``logs/`` is gitignored).
        verbose: Print per-gate-pass margins.
        device: ``"cpu"`` or ``"gpu"``. The environment defaults to CPU and the TOML config cannot
            change this, so it must be requested explicitly.
        control_mode: Override the config's ``env.control_mode`` (``"state"`` or ``"attitude"``).
            Needed when a config ships with a mode that does not match the controller's action
            space, e.g. running the attitude-based MPCC on a ``"state"`` config.
    """
    logging.basicConfig(level=logging.WARNING)
    device = resolve_device(device)
    config_path = REPO_ROOT / "config" / config
    cfg = load_config(config_path)
    cfg.sim.render = False  # Never open a window in this harness
    if control_mode is not None:
        cfg.env.control_mode = control_mode
    controller_file = controller or cfg.controller.file
    controller_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller_file)

    print(
        f"config={config}  controller={controller_file}  runs={n_runs}  seed0={seed}  "
        f"device={device}  control_mode={cfg.env.control_mode}"
    )
    results = []
    for i in range(n_runs):
        ep_seed = seed + i
        ep = run_episode(cfg, controller_cls, ep_seed, verbose=verbose, device=device)
        results.append(ep)
        print(
            f"[seed {ep_seed}] {ep['verdict']:<13} {ep['n_final']}/{ep['n_required']} passes  "
            f"t={ep['sim_time']:.2f}s  | {ep['detail']}"
        )
        if plot or log:
            stem = f"{config_path.stem}_{Path(controller_file).stem}_seed{ep_seed}"
            log_path = write_log(
                ep,
                episode_arrays(ep),
                Path(log_dir) / f"{stem}.npz",
                {
                    "config": config,
                    "controller": controller_file,
                    "seed": ep_seed,
                    "device": device,
                    "control_mode": cfg.env.control_mode,
                },
            )
            if log:
                solver_ms = ep["record"].get("solver_ms", np.zeros(0))
                print(
                    f"          log  -> {log_path}  ({len(ep['plan_log'])} plans, "
                    f"{len(ep['record']['t'])} steps, solver "
                    + (f"{np.mean(solver_ms):.1f} ms mean" if len(solver_ms) else "n/a")
                    + ")"
                )
            if plot:
                # Plotted from the file that was just written, never from memory.
                out = Path(out_dir) / f"{stem}.png"
                plot_episode(
                    log_path, out, f"{config_path.stem} / {controller_file} / seed {ep_seed}"
                )
                print(f"          plot -> {out}  (from {log_path.name})")

    n_ok = sum(ep["verdict"] == "FINISHED" for ep in results)
    times = [ep["sim_time"] for ep in results if ep["verdict"] == "FINISHED"]
    print(f"\nfinished {n_ok}/{n_runs}" + (f", mean time {np.mean(times):.2f}s" if times else ""))


if __name__ == "__main__":
    fire.Fire(main)
