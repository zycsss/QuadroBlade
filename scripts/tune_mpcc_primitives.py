"""Tune every MPCC setting on the simple paths of `lsy_drone_racing.utils.mpcc_bench`.

One setting at a time, each tried at a few values with the others at their best so far. A setting's
best value is kept if it lowers the score, which is how much slower than a point mass the flights
were, plus a penalty for straying from the path: speed that a drone cannot steer is worth nothing.

Example::

    $ pixi run python scripts/tune_mpcc_primitives.py --rounds 2 --out tuned.json
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np

from lsy_drone_racing.control.topt_mpcc import MpccConfig
from lsy_drone_racing.utils.mpcc_bench import PRIMITIVES, fly

FACTORS = (0.25, 0.5, 2.0, 4.0)
#: Settings scaled by `FACTORS`.
SCALED = (
    "q_contour",
    "q_contour_z",
    "q_lag",
    "q_progress",
    "q_rate",
    "q_torque",
    "q_dthrust",
    "q_dds",
    "q_vz",
    "k_dist",
    "k_thrust",
    "w_tunnel",
    "w_speed",
    "w_tilt",
    "w_az",
    "w_motor",
)
#: Settings tried at these values.
VALUES = {
    "horizon": (20, 30, 50),
    "progress_add": (0.0, 0.6, 1.0, 2.0),
    "az_max": (2.0, 6.0, 8.0, 12.0),
    "tilt_max": (0.9, 1.1, 1.25, 1.4),
    "tilt_lead": (0.0, 0.04, 0.12),
    "cmd_max": (1.1, 1.5),
    "thrust_max": (0.68, 0.77, 0.8),
    "dds_max": (15.0, 60.0),
    "motor_share": (0.25, 0.5, 0.75),
    "tunnel_max": (0.1, 0.2, 0.3),
    "dist_share": (0.0, 0.6),
}
ERROR_ALLOWED = 0.08


def evaluate(overrides: dict) -> dict:
    """Fly every primitive with some settings changed, and return the score and the numbers."""
    cfg = replace(MpccConfig(), **overrides)
    if "tilt_max" in overrides:
        cfg = replace(cfg, rescue_tilt=max(cfg.rescue_tilt, cfg.tilt_max + 0.15))
    flights = {name: fly(name, cfg) for name in PRIMITIVES}
    ratio = float(np.mean([f.time_ratio for f in flights.values()]))
    over = float(np.mean([max(f.error_max - ERROR_ALLOWED, 0.0) for f in flights.values()]))
    missed = sum(not f.finished for f in flights.values())
    return {
        "overrides": overrides,
        "score": ratio + 5.0 * over + 5.0 * missed,
        "ratio": ratio,
        "error_max": float(np.mean([f.error_max for f in flights.values()])),
        "thrust": float(np.mean([f.thrust_mean for f in flights.values()])),
        "high": float(np.mean([f.thrust_high for f in flights.values()])),
        "missed": missed,
        "failures": sum(f.failures for f in flights.values()),
        "flights": {k: vars(f) for k, f in flights.items()},
    }


def main(rounds: int = 2, out: str = "tuned_mpcc.json", jobs: int = 16, start: str | None = None):
    """Run the coordinate search.

    Args:
        rounds: Passes over all the settings.
        out: JSON file the search writes after every setting.
        jobs: Worker processes.
        start: JSON file with the overrides to start from.
    """
    best = json.loads(Path(start).read_text()) if start else {}
    history = []
    defaults = MpccConfig()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        current = evaluate(best)
        print(f"start: {_summary(current)}", flush=True)
        for pass_index in range(rounds):
            for name in (*SCALED, *VALUES):
                base = best.get(name, getattr(defaults, name))
                values = (
                    [type(base)(base * f) for f in FACTORS]
                    if name in SCALED
                    else list(VALUES[name])
                )
                values = [v for v in values if v != base]
                results = list(pool.map(evaluate, [{**best, name: v} for v in values]))
                top = min(results, key=lambda r: r["score"])
                row = " ".join(
                    f"{v:g}:{r['score']:.3f}" for v, r in zip(values, results, strict=True)
                )
                kept = top["score"] < current["score"] - 0.002
                if kept:
                    current, best = top, top["overrides"]
                kept_note = f" KEPT {best[name]:g}" if kept else ""
                print(
                    f"[{pass_index}] {name:12} base {base:g} now {current['score']:.3f} | "
                    f"{row}{kept_note}",
                    flush=True,
                )
                history.append(
                    {
                        "pass": pass_index,
                        "name": name,
                        "values": values,
                        "scores": [r["score"] for r in results],
                        "kept": kept,
                    }
                )
                Path(out).write_text(
                    json.dumps({"best": best, "current": current, "history": history}, indent=1)
                )
    print(f"final: {_summary(current)}")
    print("best", best)


def _summary(result: dict) -> str:
    """Return one line with the score and the bench measures of an evaluated configuration."""
    return (
        f"score {result['score']:.3f} ratio {result['ratio']:.3f} err {result['error_max']:.3f} "
        f"thrust {result['thrust']:.1f} high {result['high']:.2f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
