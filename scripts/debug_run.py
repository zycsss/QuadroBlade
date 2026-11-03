"""Headless debugging harness for drone racing controllers.

This script runs a controller for one or more episodes, records the flight, classifies *why* an
episode ended, and plots the trajectory. It deliberately does not render anything, so it works over
SSH and on machines without a display.

Run as:

    $ python scripts/debug_run.py --config level0.toml --controller my_controller.py

inside the repository's pixi environment, or from outside of it with

    $ pixi run python scripts/debug_run.py --config level0.toml --controller my_controller.py

For GPU execution use the ``gpu`` environment and pass ``--device gpu`` explicitly. The environment
defaults to CPU and the TOML config cannot change that, so the ``gpu`` environment alone would still
run on CPU:

    $ pixi run -e gpu python scripts/debug_run.py --config level0.toml --device gpu

Output:

* A one-line verdict per episode printed to stdout.
* ``plots/<config>_<controller>_seed<seed>.png`` with the track from above coloured by the speed it
  was flown at, and below it the altitude over time on the same speed colour scale. A run that did
  not finish also gets a zoom on the pole or gate it failed against.
* ``logs/<config>_<controller>_seed<seed>.npz`` (the arrays) and ``.json`` (the summary) that both
  the figure and the replay website in ``replay_site/`` are drawn from.

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


def failure_record(
    verdict: str,
    last_pos: np.ndarray,
    sim_time: float,
    gates_pos: np.ndarray,
    obstacles_pos: np.ndarray,
) -> dict[str, Any]:
    """Describe an unsuccessful ending in a form a figure can draw.

    The verdict alone says a run failed; this says where, when and against which object, which is
    what a zoomed view needs. It is written into the log as ``failure_*`` arrays.

    Args:
        verdict: The verdict from `classify`.
        last_pos: Last live drone position.
        sim_time: Time at which the episode ended.
        gates_pos: Gate positions, nominal as the environment started.
        obstacles_pos: Obstacle positions.

    Returns:
        A record with ``kind``, ``object`` (index, or ``-1``), ``pos``, ``t`` and the ``distance``
        to that object. ``kind`` is ``"none"`` for a finished run.
    """
    if verdict == "FINISHED":
        return {"kind": "none", "object": -1, "pos": last_pos, "t": sim_time, "distance": 0.0}
    obstacle_distance = np.linalg.norm(obstacles_pos - last_pos, axis=1)
    gate_distance = np.linalg.norm(gates_pos - last_pos, axis=1)
    if verdict == "HIT_OBSTACLE":
        index = int(obstacle_distance.argmin())
        distance = float(obstacle_distance[index])
    elif verdict == "HIT_GATE":
        index = int(gate_distance.argmin())
        distance = float(gate_distance[index])
    elif verdict == "CONTACT":
        # Nothing recognised the ending, so point at the nearest object it most likely ran into.
        if obstacle_distance.min() <= gate_distance.min():
            index, distance = int(obstacle_distance.argmin()), float(obstacle_distance.min())
        else:
            index, distance = int(gate_distance.argmin()), float(gate_distance.min())
    else:
        index, distance = -1, 0.0
    return {"kind": verdict, "object": index, "pos": last_pos, "t": sim_time, "distance": distance}


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
        # What the controller was allowed to know about the track at that step. An object reports
        # its nominal position until the drone comes within the sensor range, after which the true
        # one is revealed and stays revealed. Logged so the replay can show both, and when one
        # became the other, without re-deriving the environment's rule from the trajectory.
        "gates_seen": [],
        "poles_seen": [],
        "gates_visited": [],
        "poles_visited": [],
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
            rec["gates_seen"].append(np.asarray(obs["gates_pos"], dtype=float))
            rec["poles_seen"].append(np.asarray(obs["obstacles_pos"], dtype=float))
            rec["gates_visited"].append(np.asarray(obs["gates_visited"], dtype=bool))
            rec["poles_visited"].append(np.asarray(obs["obstacles_visited"], dtype=bool))

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
    rec["gates_seen"] = np.array(rec["gates_seen"], dtype=float)
    rec["poles_seen"] = np.array(rec["poles_seen"], dtype=float)
    rec["gates_visited"] = np.array(rec["gates_visited"], dtype=bool)
    rec["poles_visited"] = np.array(rec["poles_visited"], dtype=bool)
    # The range at which the environment reveals a true position, needed to read the two above.
    rec["sensor_range"] = np.array(float(config.env.sensor_range))
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
        # Where, when and against what the episode ended, so a figure can zoom on it.
        "failure": failure_record(
            verdict,
            rec["pos"][-1],
            n_steps / config.env.freq,
            np.asarray(env.unwrapped.data.gates_pos[0], dtype=float),
            np.asarray(env.unwrapped.data.obstacles_pos[0], dtype=float),
        ),
        "n_final": n_final,
        "n_required": len(gate_sequence),
        "steps": n_steps,
        "sim_time": n_steps / config.env.freq,
        "crossings": crossings,
        # The poses the drone actually faced, which differ from the nominal ones above on a
        # randomized track. The controller only learns them within the sensor range.
        "true_gates_pos": np.asarray(env.unwrapped.data.gates_pos[0], dtype=float),
        "true_gates_quat": np.asarray(env.unwrapped.data.gates_quat[0], dtype=float),
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
        # Which action space the controller drove, so a reader can tell whether ``des`` is a
        # position command (state mode) or a roll/pitch/yaw command (attitude mode).
        "control_mode": config.env.control_mode,
    }


def episode_arrays(ep: dict[str, Any]) -> dict[str, Any]:
    """Assemble the arrays of an episode, exactly as they are written to a run log.

    The arrays hold per control step: time, position, velocity, acceleration (a finite difference of
    the recorded velocity, since acceleration is not observed), attitude, angular rate, the applied
    attitude command, the achieved speed, the number of gates passed, the gate and pole poses as the
    controller saw them (nominal until revealed, true afterwards), the reveal flags and the sensor
    range that decides when a reveal happens, and whatever the controller reports through its
    ``diagnostics``. The action space is recorded as ``control_mode`` because it decides how ``des``
    must be read: a position command in ``state`` mode, roll/pitch/yaw in ``attitude`` mode. Every
    path the controller committed to is stored separately as ``plan_*`` arrays, with the time, tick,
    reason, drone position, path progress and 200 samples of the path itself, so a run can be
    replayed against the plan that was active at any step. The controller's ``replans`` counter is
    the index of the active plan.

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
            "true_gates_quat": ep["true_gates_quat"],
            "true_obstacles_pos": ep["true_obstacles_pos"],
            "gate_sequence": ep["gate_sequence"],
            "bounds_low": ep["bounds_low"],
            "bounds_high": ep["bounds_high"],
            "control_mode": np.array(ep["control_mode"]),
        }
    )
    plans = ep["plan_log"]
    arrays["plan_reason"] = np.array([plan["reason"] for plan in plans], dtype="<U120")
    for key in ("t", "tick", "theta", "length", "n_waypoints", "ms"):
        arrays[f"plan_{key}"] = np.array([plan[key] for plan in plans], dtype=float)
    arrays["plan_drone_pos"] = np.array([plan["drone_pos"] for plan in plans], dtype=float)
    arrays["plan_points"] = np.array([plan["points"] for plan in plans], dtype=float)
    failure = ep["failure"]
    arrays["failure_kind"] = np.array(failure["kind"])
    arrays["failure_object"] = np.array(failure["object"], dtype=int)
    arrays["failure_pos"] = np.asarray(failure["pos"], dtype=float)
    arrays["failure_t"] = np.array(failure["t"], dtype=float)
    arrays["failure_distance"] = np.array(failure["distance"], dtype=float)
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


def run_and_save(
    config: str,
    controller_file: str,
    control_mode: str | None,
    ep_seed: int,
    verbose: bool,
    device: str,
    plot: bool,
    out_dir: str,
    log: bool,
    log_dir: str,
) -> dict[str, Any]:
    """Run one episode, write its log and figure, and return a short summary.

    This is the unit of work for both the serial loop and the process pool, so a parallel run writes
    exactly the files a serial one does. Everything printed along the way is captured and handed
    back as ``text``, so episodes finishing in parallel do not interleave their output.

    Args:
        config: Race config file name in ``config/``.
        controller_file: Controller file name in ``lsy_drone_racing/control/``.
        control_mode: Override for ``env.control_mode``, or None to keep the config's.
        ep_seed: Seed of this episode.
        verbose: Print per-gate-pass margins.
        device: ``"cpu"`` or ``"gpu"``.
        plot: Whether to write the PNG.
        out_dir: Directory for plots.
        log: Whether to report the log path.
        log_dir: Directory for logs.

    Returns:
        ``seed``, ``verdict``, ``n_final``, ``n_required``, ``sim_time``, ``detail`` and ``text``.
    """
    import contextlib
    import io

    config_path = REPO_ROOT / "config" / config
    cfg = load_config(config_path)
    cfg.sim.render = False  # Never open a window in this harness
    if control_mode is not None:
        cfg.env.control_mode = control_mode
    controller_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller_file)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ep = run_episode(cfg, controller_cls, ep_seed, verbose=verbose, device=device)
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
                plan_ms = [entry["ms"] for entry in ep["plan_log"]]
                print(
                    f"          log  -> {log_path}  ({len(ep['plan_log'])} plans, "
                    f"{len(ep['record']['t'])} steps, plan {np.mean(plan_ms):.1f} ms mean, "
                    f"solver {np.mean(solver_ms):.1f} ms mean)"
                )
            if plot:
                # Plotted from the file that was just written, never from memory.
                out = Path(out_dir) / f"{stem}.png"
                plot_episode(
                    log_path, out, f"{config_path.stem} / {controller_file} / seed {ep_seed}"
                )
                print(f"          plot -> {out}  (from {log_path.name})")
    return {
        "seed": ep_seed,
        "verdict": ep["verdict"],
        "n_final": ep["n_final"],
        "n_required": ep["n_required"],
        "sim_time": ep["sim_time"],
        "detail": ep["detail"],
        "text": buffer.getvalue(),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval of a success rate (95 % for the default ``z``)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pass_probability(p: float, n: int = 20, threshold: float = 0.5) -> float:
    """Return the chance that ``n`` runs at success rate ``p`` finish ``threshold`` of them.

    This is the question the competition's ``scripts/evaluate.py`` asks: 20 random seeds, and the
    submission only counts with at least half of them finished.
    """
    from math import ceil, comb

    k_min = ceil(threshold * n)
    return float(sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(k_min, n + 1)))


def summarize(results: list[dict[str, Any]]) -> str:
    """Return the closing summary: success rate with its interval, and the lap time spread."""
    n = len(results)
    n_ok = sum(r["verdict"] == "FINISHED" for r in results)
    times = np.array([r["sim_time"] for r in results if r["verdict"] == "FINISHED"])
    mean = f", mean time {np.mean(times):.2f}s" if len(times) else ""
    lines = [f"\nfinished {n_ok}/{n}{mean}"]
    if n > 1:
        low, high = wilson_interval(n_ok, n)
        lines.append(
            f"  success {n_ok / n:.0%}  (95% CI {low:.0%}-{high:.0%}),  "
            f"P(>=50% of 20 in evaluate.py) = {pass_probability(n_ok / n):.0%}"
        )
    if len(times) > 1:
        sem = np.std(times, ddof=1) / np.sqrt(len(times))
        lines.append(
            f"  lap     mean {np.mean(times):.2f}s +- {1.96 * sem:.2f}s,  "
            f"median {np.median(times):.2f}s,  min {np.min(times):.2f}s,  max {np.max(times):.2f}s"
        )
    failures: dict[str, list[int]] = {}
    for r in results:
        if r["verdict"] != "FINISHED":
            failures.setdefault(r["verdict"], []).append(r["seed"])
    for verdict, seeds in sorted(failures.items()):
        lines.append(f"  {verdict:<13} seeds {sorted(seeds)}")
    return "\n".join(lines)


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
    jobs: int = 1,
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
        jobs: Number of worker processes; 0 uses one per CPU core. Each episode is independent and
            writes the same log and figure it would in a serial run. The first episode always runs
            in this process, so an OCP change is compiled once before the workers load it.
    """
    logging.basicConfig(level=logging.WARNING)
    device = resolve_device(device)
    cfg = load_config(REPO_ROOT / "config" / config)
    controller_file = controller or cfg.controller.file
    if jobs <= 0:
        jobs = os.cpu_count() or 1
    jobs = max(1, min(jobs, n_runs))

    print(
        f"config={config}  controller={controller_file}  runs={n_runs}  seed0={seed}  "
        f"device={device}  control_mode={control_mode or cfg.env.control_mode}  jobs={jobs}"
    )
    head = (config, controller_file, control_mode)
    tail = (verbose, device, plot, out_dir, log, log_dir)
    seeds = [seed + i for i in range(n_runs)]
    results = []

    def report(result: dict[str, Any]) -> None:
        results.append(result)
        print(result["text"], end="", flush=True)

    # The first episode builds the acados solver if the OCP changed; the workers then only load it,
    # so they cannot race each other on c_generated_code/.
    report(run_and_save(*head, seeds[0], *tail))
    if jobs == 1:
        for ep_seed in seeds[1:]:
            report(run_and_save(*head, ep_seed, *tail))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # One XLA/BLAS thread per worker: the pool is the parallelism, and N workers each starting
        # a machine-wide thread pool only fight over the same cores. Workers inherit this at spawn.
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_and_save, *head, s, *tail) for s in seeds[1:]]
            for future in as_completed(futures):
                report(future.result())
    results.sort(key=lambda r: r["seed"])
    print(summarize(results))


if __name__ == "__main__":
    fire.Fire(main)
