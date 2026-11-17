"""Compare `utils/fast_planner.py` against the `topt_planner` guides of a solved library.

Every seed of a library's range is planned, not only those `topt_planner` kept, because measuring on
its survivors alone would favour it. Timing is one process on one core, as in a race; pin it with
``taskset``, and look at the load printed alongside, because other processes slow it down.

Example::

    $ taskset -c 15 pixi run -e gpu python scripts/compare_fast_planner.py \
        --library .tmp/topt/rl2/scenes/fixed_level3_validation.npz --source level3 \
        --first_seed 300000 --count 200 --out level3_validation.json
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import json
import time
from pathlib import Path
from types import SimpleNamespace

import fire
import jax
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.topt_planner import Track
from lsy_drone_racing.envs.randomize import build_random_track_fn
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.fast_planner import FastPlannerConfig, plan_guide, required_thrust
from lsy_drone_racing.utils.race_policy import TAKEOFF_LIFT, TAKEOFF_SPEED, guide_arrays
from lsy_drone_racing.utils.scene_generator import sample_layout

SEQUENCE = np.array([0, 1, 2, 3, 1])
#: What a plan is asked for and may reach [m/s^2, rad, m/s^2].
THRUST_LIMIT = 15.0
TILT_LIMIT = 1.1
CROSSING_ANGLE_MAX = 0.9
#: Budgets compared: gradient steps and the thrust limit of the segments.
BUDGETS = (
    {"steps": 0, "thrust": 15.0},
    {"steps": 30, "thrust": 15.0},
    {"steps": 60, "thrust": 15.0},
    {"steps": 120, "thrust": 15.0},
    {"steps": 300, "thrust": 15.0},
    {"steps": 60, "thrust": 14.0},
)


def _layouts(source: str, first_seed: int, count: int) -> list[tuple]:
    """Return the layouts of a seed range, as `scripts/solve_scene_library.py` draws them."""
    if source != "level3":
        return [sample_layout(np.random.default_rng(first_seed + i)) for i in range(count)]
    config = load_config(Path(__file__).parents[1] / "config" / "level3.toml").env.track
    start = np.array(config.drones[0]["pos"], dtype=float)
    build = build_random_track_fn(
        [g["pos"][2] for g in config.gates],
        [o["pos"][2] for o in config.obstacles],
        config.safety_limits.pos_limit_low,
        config.safety_limits.pos_limit_high,
    )
    layouts = []
    for i in range(count):
        gates, quats, poles = build(start, jax.random.PRNGKey(first_seed + i))
        yaw = Rotation.from_quat(np.asarray(quats, dtype=float)).as_euler("xyz")[:, 2]
        layouts.append((start, np.asarray(gates, dtype=float), yaw, np.asarray(poles, dtype=float)))
    return layouts


def _library(path: str) -> dict[bytes, SimpleNamespace]:
    """Return the plans of a library, by the rounded position of their gates."""
    plans = {}
    with np.load(path) as z:
        for i in range(int(z["count"])):
            key = np.round(z[f"gates_pos_{i}"], 3).astype(np.float32).tobytes()
            plans[key] = SimpleNamespace(
                pos=z[f"pos_{i}"],
                vel=z[f"vel_{i}"],
                time=z[f"time_{i}"],
                crossings=z[f"crossings_{i}"],
            )
    return plans


def _tilt_jump(trajectory: object, smooth: bool) -> float:
    """Return the largest change of the guide's attitude between two beads [rad]."""
    guide = guide_arrays(trajectory, smooth=smooth)
    quats = guide["quats"][: guide["crossings"][-1] + 1]
    dots = np.abs((quats[1:] * quats[:-1]).sum(axis=1))
    return float(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)).max())


def _measure(layouts: list[tuple], reference: dict, config: FastPlannerConfig) -> dict:
    """Plan every layout under one budget, and summarize it against the library."""
    rows = []
    for k, (start, gates, yaw, poles) in enumerate(layouts):
        rotations = Rotation.from_euler("z", yaw[:, None]).as_matrix()
        track = Track(gates, rotations, poles, SEQUENCE, np.ones(5), tuple(False for _ in SEQUENCE))
        first = start + np.array([0.0, 0.0, TAKEOFF_LIFT])
        plan, tries = plan_guide(track, first, np.array([0.0, 0.0, TAKEOFF_SPEED]), config)
        if k == 0:  # The first call compiles, which a race would do before the start.
            # A retry compiles nothing new: only the arrays change, not their shapes.
            plan, tries = plan_guide(track, first, np.array([0.0, 0.0, TAKEOFF_SPEED]), config)
        t = plan.trajectory
        thrust, tilt = required_thrust(t)
        speed_at_pass = t.vel[t.crossings]
        along = (speed_at_pass * np.array([track.normal(i) for i in range(5)])).sum(axis=1)
        angle = np.arccos(np.clip(along / np.linalg.norm(speed_at_pass, axis=1), -1.0, 1.0))
        topt = reference.get(np.round(gates, 3).astype(np.float32).tobytes())
        rows.append(
            {
                "clear": plan.clear,
                "gap": min(plan.pole_gap, plan.gate_gap),
                "tries": tries,
                "lap": plan.lap_time,
                "topt_lap": float(topt.time[topt.crossings[-1]]) if topt else np.nan,
                "ms": sum(plan.timing_ms.values()),
                "route_ms": plan.timing_ms["route"],
                "solve_ms": plan.timing_ms["solve"],
                "thrust_peak": float(thrust.max()),
                "thrust_over": float(np.mean(thrust > THRUST_LIMIT)),
                "tilt_peak": float(tilt.max()),
                "cross_ok": bool(angle.max() <= CROSSING_ANGLE_MAX + 0.02),
                "inside": plan.inside,
                "tilt_jump": _tilt_jump(t, smooth=False),
                "path": float(np.linalg.norm(np.diff(t.pos, axis=0), axis=1).sum()),
            }
        )
    get = lambda name: np.array([r[name] for r in rows], dtype=float)  # noqa: E731
    clear, ratio = get("clear") > 0.5, get("lap") / get("topt_lap")
    both = clear & np.isfinite(ratio)
    return {
        "tracks": len(rows),
        "topt_kept": int(np.isfinite(get("topt_lap")).sum()),
        "clear_share": float(clear.mean()),
        "clear_share_of_topt_kept": float(clear[np.isfinite(ratio)].mean()),
        "clear_share_of_topt_dropped": (
            float(clear[~np.isfinite(ratio)].mean()) if np.any(~np.isfinite(ratio)) else None
        ),
        "first_try_share": float((get("tries") == 1).mean()),
        "tries_max": int(get("tries").max()),
        "tries_mean": float(get("tries").mean()),
        "gap_worst": float(get("gap").min()),
        "lap_ratio_p50": float(np.median(ratio[both])),
        "lap_ratio_p90": float(np.percentile(ratio[both], 90)),
        "lap_ratio_best": float(ratio[both].min()),
        "lap_p50": float(np.median(get("lap")[clear])),
        "topt_lap_p50": float(np.median(get("topt_lap")[both])),
        "ms_p50": float(np.median(get("ms"))),
        "ms_p95": float(np.percentile(get("ms"), 95)),
        "ms_max": float(get("ms").max()),
        "route_ms_p50": float(np.median(get("route_ms"))),
        "solve_ms_p50": float(np.median(get("solve_ms"))),
        "under_20_ms_share": float((get("ms") <= 20.0).mean()),
        "clear_under_20_ms_share": float(((get("ms") <= 20.0) & clear).mean()),
        "thrust_peak_p50": float(np.median(get("thrust_peak"))),
        "thrust_peak_max": float(get("thrust_peak").max()),
        "thrust_over_share_of_time": float(get("thrust_over").mean()),
        "tilt_peak_max": float(get("tilt_peak").max()),
        "crossing_angle_ok_share": float(get("cross_ok").mean()),
        "inside_arena_share": float(get("inside").mean()),
        "tilt_jump_p50": float(np.median(get("tilt_jump"))),
    }


def main(library: str, source: str, first_seed: int, count: int, out: str | None = None):
    """Compare the budgets on one seed range.

    Args:
        library: Library of `topt_planner` guides from those seeds.
        source: ``generated`` or ``level3``, as in `scripts/solve_scene_library.py`.
        first_seed: First seed of the range.
        count: Seeds to plan.
        out: JSON file for the results.
    """
    layouts = _layouts(source, first_seed, count)
    reference = _library(library)
    results = {"library": library, "first_seed": first_seed, "budgets": []}
    topt_jumps = [_tilt_jump(p, smooth=True) for p in list(reference.values())[:100]]
    results["topt_tilt_jump_p50"] = float(np.median(topt_jumps))
    results["topt_inside_arena_share"] = float(
        np.mean(
            [
                np.all(p.pos[:, 0] >= -2.5)
                and np.all(p.pos[:, 0] <= 2.5)
                and np.all(p.pos[:, 1] >= -1.5)
                and np.all(p.pos[:, 1] <= 1.5)
                and np.all(p.pos[:, 2] >= 0.0)
                and np.all(p.pos[:, 2] <= 2.0)
                for p in reference.values()
            ]
        )
    )
    print(f"topt inside the arena: {results['topt_inside_arena_share']:.3f}")
    print(f"topt attitude change between beads, median of worst: {np.median(topt_jumps):.2f} rad")
    for budget in BUDGETS:
        load = os.getloadavg()[0]
        t0 = time.time()
        summary = _measure(layouts, reference, FastPlannerConfig(**budget))
        summary |= {"budget": budget, "load_before": load, "load_after": os.getloadavg()[0]}
        results["budgets"].append(summary)
        print(json.dumps(summary), f"({time.time() - t0:.0f} s)", flush=True)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
