"""Sim-to-real stress test: fly a controller under conditions the real drone has and the sim lacks.

The race environment is ideal in ways the Crazyflie is not. This script wraps any controller and
re-runs the same seeds under perturbations taken one at a time and then together, so each gap's
cost is measured on its own:

* ``dynamics``   swap the first-principles model for the identified one with rotor lag and drag
                 (``so_rpy_rotor_drag``), i.e. a model the controller was *not* tuned on.
* ``obs_delay``  the controller sees the state ``k`` steps late (Vicon + estimator + radio).
* ``act_delay``  the command reaches the drone ``k`` steps late.
* ``thrust``     the delivered collective thrust is ``factor * command`` (battery sag, worn props).
* ``noise``      Gaussian noise on the observed position and velocity (estimator noise).

Nothing in ``config/`` is edited: the official config is loaded and modified in memory only.

Run as:

    $ pixi run python scripts/sim2real_stress.py --config level3.toml --controller topt_mpcc.py \
          --control_mode attitude --n_runs 10

or a single condition:

    $ pixi run python scripts/sim2real_stress.py --conditions obs_delay_2,thrust_90 --n_runs 10
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import ast
import contextlib
import copy
import io
import json
import logging
import re
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import fire
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from debug_run import run_episode  # noqa: E402

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402

logger = logging.getLogger(__name__)

# Each condition: a dict of perturbations. Units: delays in env steps (20 ms at 50 Hz), noise std in
# m and m/s. The realistic values are a first guess for a Crazyflie behind Vicon at 50 Hz: ~40 ms of
# total latency, 10 % thrust loss late in a battery, mm-level position noise.
CONDITIONS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "dynamics": {"dynamics": "so_rpy_rotor_drag"},
    "obs_delay_1": {"obs_delay": 1},
    "obs_delay_2": {"obs_delay": 2},
    "act_delay_1": {"act_delay": 1},
    "thrust_90": {"thrust": 0.90},
    "thrust_80": {"thrust": 0.80},
    "noise": {"pos_noise": 0.003, "vel_noise": 0.05},
    "realistic": {
        "obs_delay": 1,
        "act_delay": 1,
        "thrust": 0.90,
        "pos_noise": 0.003,
        "vel_noise": 0.05,
    },
    "realistic_rotor": {
        "dynamics": "so_rpy_rotor_drag",
        "obs_delay": 1,
        "act_delay": 1,
        "thrust": 0.90,
        "pos_noise": 0.003,
        "vel_noise": 0.05,
    },
}
DEFAULT_CONDITIONS = (
    "nominal,dynamics,obs_delay_1,obs_delay_2,act_delay_1,thrust_90,noise,realistic"
)


def perturbed(controller_cls: type, spec: dict[str, Any], control_mode: str, seed: int) -> type:
    """Return a controller class that sees and acts through the given perturbations."""
    obs_delay = int(spec.get("obs_delay", 0))
    act_delay = int(spec.get("act_delay", 0))
    thrust = float(spec.get("thrust", 1.0))
    pos_noise = float(spec.get("pos_noise", 0.0))
    vel_noise = float(spec.get("vel_noise", 0.0))

    class Perturbed:
        def __init__(self, obs: dict, info: dict, config: Any):
            self._rng = np.random.default_rng(seed + 7919)
            self._obs_buf: deque = deque(maxlen=obs_delay + 1)
            self._act_buf: deque = deque(maxlen=act_delay + 1)
            self.compute_ms: list[float] = []
            self.inner = controller_cls(self._observe(obs), info, config)

        def _observe(self, obs: dict) -> dict:
            obs = {k: np.array(v, copy=True) for k, v in obs.items()}
            if pos_noise > 0:
                obs["pos"] = obs["pos"] + self._rng.normal(0.0, pos_noise, obs["pos"].shape)
            if vel_noise > 0:
                obs["vel"] = obs["vel"] + self._rng.normal(0.0, vel_noise, obs["vel"].shape)
            self._obs_buf.append(obs)
            return self._obs_buf[0]  # oldest = obs_delay steps old once the buffer is full

        def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
            t0 = time.perf_counter()
            action = np.asarray(self.inner.compute_control(self._observe(obs), info), dtype=float)
            self.compute_ms.append(1e3 * (time.perf_counter() - t0))
            self._act_buf.append(action.copy())
            out = self._act_buf[0].copy()
            if control_mode == "attitude":
                out[3] *= thrust  # [roll, pitch, yaw, collective thrust]
            return out.astype(np.float32)

        def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
            # The inner controller is told the command it chose and the (delayed) state it saw.
            return self.inner.step_callback(
                self._act_buf[-1], self._obs_buf[-1], reward, terminated, truncated, info
            )

        def episode_callback(self):
            self.inner.episode_callback()

        def episode_reset(self):
            self.inner.episode_reset()

        @property
        def plan_log(self) -> list:
            return list(getattr(self.inner, "plan_log", []))

        def diagnostics(self) -> dict:
            inner = getattr(self.inner, "diagnostics", None)
            return dict(inner()) if inner is not None else {}

    return Perturbed


def parse_ctrl(ctrl: str | dict | None) -> dict[str, Any]:
    """Parse ``"key=value,key=value"`` into controller overrides (values as Python literals)."""
    if not ctrl:
        return {}
    if isinstance(ctrl, dict):
        return dict(ctrl)
    out = {}
    for item in re.split(r",(?=\s*[A-Za-z_]\w*\s*=)", str(ctrl)):  # commas before a key
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            out[key.strip()] = value.strip()
    return out


def run_condition(
    name: str,
    config: str,
    controller: str,
    control_mode: str,
    seeds: list[int],
    ctrl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all seeds under one condition and return the per-seed results."""
    spec = CONDITIONS[name]
    base = load_config(REPO_ROOT / "config" / config)
    for key, value in (ctrl or {}).items():  # in memory only, the file is untouched
        base.controller[key] = list(value) if isinstance(value, tuple) else value
    base.sim.render = False
    base.env.control_mode = control_mode
    if "dynamics" in spec:
        base.sim.dynamics = spec["dynamics"]
    controller_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / controller)
    rows = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        t0 = time.perf_counter()
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                ep = run_episode(
                    cfg, perturbed(controller_cls, spec, control_mode, seed), seed, verbose=False
                )
            row = {
                "seed": seed,
                "verdict": ep["verdict"],
                "passes": f"{ep['n_final']}/{ep['n_required']}",
                "finished": ep["verdict"] == "FINISHED",
                "sim_time": float(ep["sim_time"]),
                "detail": str(ep["detail"]),
            }
        except Exception as exc:  # a crash is a result too
            row = {
                "seed": seed,
                "verdict": "crash",
                "passes": "-",
                "finished": False,
                "sim_time": float("nan"),
                "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)
        print(
            f"  [{name:<16}] seed {seed:<4} {row['verdict']:<13} {row['passes']:<6} "
            f"t={row['sim_time']:.2f}s  {row['detail'][:90]}",
            flush=True,
        )
    return {"condition": name, "spec": spec, "ctrl": ctrl or {}, "rows": rows}


def summarize(results: list[dict[str, Any]]) -> str:
    lines = ["", f"{'condition':<18}{'success':>9}{'mean lap [s]':>14}  failure verdicts", "-" * 72]
    for res in results:
        rows = res["rows"]
        ok = [r for r in rows if r["finished"]]
        lap = f"{np.mean([r['sim_time'] for r in ok]):.2f}" if ok else "-"
        fails: dict[str, int] = {}
        for r in rows:
            if not r["finished"]:
                fails[r["verdict"]] = fails.get(r["verdict"], 0) + 1
        fail_txt = ", ".join(f"{k} x{v}" for k, v in sorted(fails.items())) or "-"
        lines.append(f"{res['condition']:<18}{len(ok):>4}/{len(rows):<4}{lap:>14}  {fail_txt}")
    return "\n".join(lines)


def main(
    config: str = "level3.toml",
    controller: str = "topt_mpcc.py",
    control_mode: str = "attitude",
    conditions: str = DEFAULT_CONDITIONS,
    n_runs: int = 10,
    seed: int = 0,
    out: str = "logs/sim2real_stress.json",
    ctrl: str = "",
):
    """Run the stress matrix and print a table: success and lap time per condition."""
    logging.basicConfig(level=logging.WARNING)
    names = [
        c.strip() for c in (conditions.split(",") if isinstance(conditions, str) else conditions)
    ]
    unknown = [n for n in names if n not in CONDITIONS]
    assert not unknown, f"unknown conditions {unknown}; choose from {list(CONDITIONS)}"
    seeds = [seed + i for i in range(n_runs)]
    overrides = parse_ctrl(ctrl)
    print(
        f"config={config} controller={controller} mode={control_mode} seeds={seeds[0]}..{seeds[-1]}"
        f" ctrl={overrides}"
    )
    results = [run_condition(n, config, controller, control_mode, seeds, overrides) for n in names]
    table = summarize(results)
    print(table)
    out_path = REPO_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
