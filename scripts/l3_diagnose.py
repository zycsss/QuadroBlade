"""Run seeds of the stress test and classify every failure in detail.

For every run it prints the verdict, whether the time-optimal plan solved, how long it took, and for
a failure the nearest object of the *true* track: horizontal distance to every pole axis, distance
to every gate frame, the height and the distance to the safety limits.

    $ pixi run python scripts/l3_diagnose.py --seeds 0,1,2 --condition nominal
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from debug_run import run_episode  # noqa: E402
from sim2real_stress import CONDITIONS, parse_ctrl, perturbed  # noqa: E402

from lsy_drone_racing.control.track_geometry import gate_capsules, segment_distance  # noqa: E402
from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402


def cause(ep: dict) -> str:
    """Name the most likely cause of a failed episode from the true geometry."""
    if ep["verdict"] == "FINISHED":
        return "-"
    if ep["verdict"] == "TIMEOUT":
        return "timeout"
    if ep["verdict"] == "crash":
        return "crash"
    p = ep["record"]["pos"][-1]
    low, high = ep["bounds_low"], ep["bounds_high"]
    if np.any(p < low + 0.03) or np.any(p > high - 0.03):
        return "boundary" if p[2] > 0.05 else "ground"
    poles = ep["true_obstacles_pos"]
    d_pole = np.linalg.norm(poles[:, :2] - p[:2], axis=1)
    rot = R.from_quat(ep["true_gates_quat"]).as_matrix()
    d_frame = np.inf
    for _, a, b, radius in gate_capsules(ep["true_gates_pos"], rot):
        d, _ = segment_distance(p[None], a, b)
        d_frame = min(d_frame, float(d[0] - radius))
    if p[2] < 0.08:
        return "ground"
    if d_pole.min() < 0.14 and d_pole.min() <= d_frame + 0.05:
        return f"pole {int(d_pole.argmin())}"
    if d_frame < 0.12:
        return "gate frame"
    return f"other (pole {d_pole.min():.2f} frame {d_frame:.2f})"


def main(
    seeds: str = "0",
    condition: str = "nominal",
    config: str = "level3.toml",
    controller: str = "topt_mpcc.py",
    ctrl: str = "",
    out: str = "",
):
    """Run the seeds and print one detailed line each."""
    if isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    else:
        seed_list = [int(s) for s in str(seeds).split(",") if s.strip()]
    base = load_config(REPO_ROOT / "config" / config)
    for key, value in parse_ctrl(ctrl).items():
        base.controller[key] = list(value) if isinstance(value, tuple) else value
    base.sim.render = False
    base.env.control_mode = "attitude"
    spec = CONDITIONS[condition]
    if "dynamics" in spec:
        base.sim.dynamics = spec["dynamics"]
    cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller)
    rows = []
    for seed in seed_list:
        captured = {}

        class Spy(perturbed(cls, spec, "attitude", seed)):
            def __init__(self, obs: dict, info: dict, cfg: object):
                super().__init__(obs, info, cfg)
                captured["ctrl"] = self.inner

        t0 = time.perf_counter()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ep = run_episode(copy.deepcopy(base), Spy, seed)
        inner = captured.get("ctrl")
        row = {
            "seed": seed,
            "verdict": ep["verdict"],
            "passes": f"{ep['n_final']}/{ep['n_required']}",
            "t": ep["sim_time"],
            "cause": cause(ep),
            "plan_solved": bool(getattr(inner, "plan_solved", True)),
            "plan_s": float(getattr(inner, "plan_time", float("nan"))),
            "plan_dur": float(getattr(inner, "plan_duration", float("nan"))),
            "fails": int(getattr(inner, "_failures", -1)),
            "last_pos": np.round(ep["record"]["pos"][-1], 2).tolist(),
            "wall": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"seed {seed:<4} {row['verdict']:<13} {row['passes']:<5} t={row['t']:5.2f} "
            f"{row['cause']:<28} solved={row['plan_solved']} plan={row['plan_s']:.1f}s "
            f"dur={row['plan_dur']:.2f} mpcc_fail={row['fails']} pos={row['last_pos']}",
            flush=True,
        )
    if out:
        Path(REPO_ROOT / out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
