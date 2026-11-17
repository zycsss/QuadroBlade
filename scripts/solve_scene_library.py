"""Solve `topt_planner` guides for generated tracks, ready for `scripts/train_scene_policy.py`.

A solve takes seconds, far too long for every episode, so a library is solved once. A track whose
guide cannot be made clear in any of the planner's initial guesses, or leaves the arena, is dropped
and counted.

Example::

    $ pixi run -e gpu python scripts/solve_scene_library.py --count 3000 --out scenes_train.npz
    $ pixi run -e gpu python scripts/solve_scene_library.py --count 200 --source level3 --out l3.npz
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import fire
import jax
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.topt_planner import (
    FAST_GUIDE_CONFIG,
    RANDOM_TRACK_CONFIG,
    Track,
    _guess_segments,
    solve_time_optimal,
)
from lsy_drone_racing.envs.randomize import build_random_track_fn
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.control.track_geometry import GATE_BOXES
from lsy_drone_racing.utils.race_policy import (
    DRONE_SPHERE_HEIGHT,
    TAKEOFF_LIFT,
    TAKEOFF_SPEED,
    clearances,
    guide_arrays,
)
from lsy_drone_racing.utils.scene_generator import ARENA_HIGH, ARENA_LOW, sample_layout

SEQUENCE = np.array([0, 1, 2, 3, 1])
DIRECTIONS = np.ones(5)
#: Least free distance of a guide that is kept [m]; a hit counts from 3 cm in training.
CLEAR_OF_POLES = 0.03
CLEAR_OF_FRAMES = 0.03


def _level3_layout(seed: int) -> tuple:
    config = load_config(Path(__file__).parents[1] / "config" / "level3.toml").env.track
    start = np.array(config.drones[0]["pos"], dtype=float)
    build = build_random_track_fn(
        [g["pos"][2] for g in config.gates],
        [o["pos"][2] for o in config.obstacles],
        config.safety_limits.pos_limit_low,
        config.safety_limits.pos_limit_high,
    )
    gates, quats, poles = build(start, jax.random.PRNGKey(seed))
    yaw = Rotation.from_quat(np.asarray(quats, dtype=float)).as_euler("xyz")[:, 2]
    return start, np.asarray(gates, dtype=float), yaw, np.asarray(poles, dtype=float)


def _is_clear(plan: object, gates: np.ndarray, rotations: np.ndarray, poles: np.ndarray) -> bool:
    """Return whether a plan's guide, on the curve it really flies, is free and in the arena."""
    guide = guide_arrays(plan, smooth=True)
    beads = guide["beads"][: guide["crossings"][-1] + 1]
    if (beads[:, :2] < ARENA_LOW).any() or (beads[:, :2] > ARENA_HIGH).any():
        return False
    count = len(beads)
    layout = SimpleNamespace(
        gates_pos=np.tile(gates, (count, 1, 1)),
        gates_rot=np.tile(rotations, (count, 1, 1, 1)),
        poles=np.tile(poles, (count, 1, 1)),
        boxes_center=np.array([b[0] for b in GATE_BOXES]),
        boxes_half=np.array([b[1] for b in GATE_BOXES]),
    )
    pole_gaps, frame_gaps = clearances(beads + [0.0, 0.0, DRONE_SPHERE_HEIGHT], layout)
    return bool(pole_gaps.min() >= CLEAR_OF_POLES and frame_gaps.min() >= CLEAR_OF_FRAMES)


def solve(job: tuple[str, int, bool]) -> dict | None:
    """Return one solved track, or None if its plan failed."""
    source, seed, fast_guides = job
    if source == "level3":
        start, gates, yaw, poles = _level3_layout(seed)
    else:
        start, gates, yaw, poles = sample_layout(np.random.default_rng(seed))
    rpy = np.c_[np.zeros((len(yaw), 2)), yaw]
    rotations = Rotation.from_euler("xyz", rpy).as_matrix()
    track = Track(gates, rotations, poles, SEQUENCE, DIRECTIONS, tuple(False for _ in SEQUENCE))
    config = FAST_GUIDE_CONFIG if fast_guides else RANDOM_TRACK_CONFIG
    first = start + np.array([0.0, 0.0, TAKEOFF_LIFT])
    velocity = np.array([0.0, 0.0, TAKEOFF_SPEED])
    # IPOPT only finds local optima: the first guess that gives a clear plan is kept.
    result = None
    for guess_args in config.guesses:
        guess = _guess_segments(track, first, config.nodes, guess_args)
        solved = solve_time_optimal(track, first, velocity, guess, config) if guess else None
        if solved is not None and _is_clear(solved, gates, rotations, poles):
            result = solved
            break
    if result is None:
        return None
    return {
        "pos": result.pos,
        "vel": result.vel,
        "time": result.time,
        "crossings": result.crossings,
        "start": start,
        "gates_pos": gates,
        "gates_rpy": rpy,
        "poles": poles,
    }


def main(
    out: str,
    count: int = 3000,
    source: str = "generated",
    first_seed: int = 0,
    workers: int = 12,
    fast_guides: bool = False,
):
    """Solve `count` tracks.

    Args:
        out: Where to write the plans.
        count: Tracks to try.
        source: ``generated`` for random scenes, or ``level3`` for the race's own generator.
        first_seed: Seed of the first track. Libraries with separate seeds share no track.
        workers: Processes.
        fast_guides: Plan with `FAST_GUIDE_CONFIG`, which uses more of the drone's reach.
    """
    t0 = time.time()
    jobs = [(source, first_seed + i, fast_guides) for i in range(count)]
    plans = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(solve, jobs), 1):
            if result is not None:
                plans.append(result)
            if i % 50 == 0 or i == count:
                print(
                    f"{i}/{count} tried, {len(plans)} solved, {time.time() - t0:.0f} s", flush=True
                )
    arrays = {f"{k}_{i}": v for i, plan in enumerate(plans) for k, v in plan.items()}
    np.savez_compressed(out, count=len(plans), **arrays)
    durations = [p["time"][-1] for p in plans]
    print(
        f"solved {len(plans)}/{count} in {time.time() - t0:.0f} s; plan lap time "
        f"median {np.median(durations):.2f} s, longest {np.max(durations):.2f} s",
        flush=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
