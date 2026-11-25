"""Plan, for shifted tracks, the guide the drone would fly at every stage of reveals.

On levels 2 and 3 gates and poles sit up to 15 cm from where they are reported, and the race reveals
each one only within 0.7 m. `RLScene` then plans its guide again with the runtime planner. Training
can't call the planner per step, so this script does it in advance: for every track of a solved
library and every draw of shifts, the guide through the layout with the first ``k`` objects known,
for ``k`` from 0 to all of them. Objects are taken in the order the stage-0 guide reaches their
reported poses (`scene_policy.reveal_order`), which flight knows from the start too.

The shifts are the race's: gates +-0.15 m in x and y, +-0.1 m in z, +-0.05, +-0.1 and +-0.2 rad of
roll, pitch and yaw; poles +-0.15 m in x and y and +-0.05 m in z. A stage whose plan fails its check
keeps the guide of the stage before, as flight does.

With ``pop_up`` a share of the poles is not drawn at random but moved towards the stage-0 guide,
as far as the shift allows (`pop_up_poles`). The sensor then reveals a pole standing on or next to
the guide the drone is flying, the event behind most pole crashes in races.

Example::

    $ pixi run -e gpu python scripts/solve_staged_library.py \
        --library .tmp/topt/rl2/scenes/fixed_level3_validation.npz --draws 4 --out staged.npz
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import multiprocessing as mp
import time
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.topt_planner import Track
from lsy_drone_racing.utils.fast_runtime_planner import runtime_plan
from lsy_drone_racing.utils.race_policy import TAKEOFF_LIFT, TAKEOFF_SPEED, guide_arrays
from lsy_drone_racing.utils.scene_policy import pop_up_poles, reveal_order, staged_layout

SEQUENCE = np.array([0, 1, 2, 3, 1])
DIRECTIONS = np.ones(5)
LAYOUT = ("start", "gates_pos", "gates_rpy", "poles")
#: The race's shifts, as `config/level2.toml` and `level3.toml` draw them.
GATE_SHIFT = np.array([0.15, 0.15, 0.1])
GATE_TURN = np.array([0.05, 0.1, 0.2])
POLE_SHIFT = np.array([0.15, 0.15, 0.05])
SENSOR_RANGE = 0.7
_PLAN_KEYS = ("pos", "vel", "time", "crossings")


def _plan(
    gates_pos: np.ndarray, gates_rpy: np.ndarray, poles: np.ndarray, start: np.ndarray
) -> tuple[object, bool]:
    track = Track(
        gates_pos,
        Rotation.from_euler("xyz", gates_rpy).as_matrix(),
        poles,
        SEQUENCE,
        DIRECTIONS,
        tuple(False for _ in SEQUENCE),
    )
    first = start + np.array([0.0, 0.0, TAKEOFF_LIFT])
    trajectory, planned, _ = runtime_plan(track, first, np.array([0.0, 0.0, TAKEOFF_SPEED]))
    return trajectory, planned.clear


def solve(job: tuple[dict, int, int, float]) -> tuple[list[dict], int]:
    """Return the scenarios of one track, and how many stage plans failed their check."""
    layout, draws, seed, pop_up = job
    nominal = (layout["gates_pos"], layout["gates_rpy"], layout["poles"])
    first, clear = _plan(*nominal, layout["start"])
    if not clear:
        return [], 0
    beads = guide_arrays(first, smooth=True)["beads"]
    order, entry = reveal_order(beads, layout["gates_pos"], layout["poles"], SENSOR_RANGE)
    rng = np.random.default_rng(seed)
    scenarios, failed = [], 0
    for _ in range(draws):
        gates, poles = len(layout["gates_pos"]), len(layout["poles"])
        true = (
            layout["gates_pos"] + rng.uniform(-GATE_SHIFT, GATE_SHIFT, (gates, 3)),
            layout["gates_rpy"] + rng.uniform(-GATE_TURN, GATE_TURN, (gates, 3)),
            pop_up_poles(beads, layout["poles"], POLE_SHIFT, rng, pop_up),
        )
        plans = [first]
        for stage in range(1, gates + poles + 1):
            trajectory, clear = _plan(*staged_layout(nominal, true, order, stage), layout["start"])
            failed += not clear
            plans.append(trajectory if clear else plans[-1])
        scenario = {k: layout[k] for k in LAYOUT}
        scenario |= {"true_gates_pos": true[0], "true_gates_rpy": true[1], "true_poles": true[2]}
        scenario |= {"order": order, "entry": entry, "plans": plans}
        scenarios.append(scenario)
    return scenarios, failed


def main(
    library: str, out: str, draws: int = 4, seed: int = 0, workers: int = 14, pop_up: float = 0.0
):
    """Solve every track of a library under ``draws`` draws of shifts.

    Args:
        library: A library written by `scripts/solve_scene_library.py`, which gives the tracks.
        out: Where to write the scenarios.
        draws: Draws of shifts per track.
        seed: Seed of the shifts.
        workers: Processes.
        pop_up: Chance of every pole to be moved towards the stage-0 guide (`pop_up_poles`).
    """
    t0 = time.time()
    with np.load(library) as data:
        layouts = [{k: data[f"{k}_{i}"] for k in LAYOUT} for i in range(int(data["count"]))]
    jobs = [(layout, draws, seed * 1_000_003 + i, pop_up) for i, layout in enumerate(layouts)]
    arrays, kept, failed = {}, 0, 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for done, (scenarios, fails) in enumerate(pool.imap(solve, jobs, chunksize=4), 1):
            failed += fails
            for scenario in scenarios:
                for k, v in scenario.items():
                    if k != "plans":
                        arrays[f"{k}_{kept}"] = v
                for stage, plan in enumerate(scenario["plans"]):
                    for k in _PLAN_KEYS:
                        arrays[f"{k}_{kept}_{stage}"] = getattr(plan, k)
                stages = len(scenario["plans"])
                kept += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} tracks, {kept} scenarios, {time.time() - t0:.0f} s")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, count=kept, stages=stages, **arrays)
    print(
        f"kept {kept} scenarios from {len(jobs)} tracks in {time.time() - t0:.0f} s; "
        f"{failed} stage plans failed their check and kept the stage before",
        flush=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
