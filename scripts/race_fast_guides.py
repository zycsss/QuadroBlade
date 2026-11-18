"""Fly `RLScene` in the real race on generated tracks, with its guide from the fast planner.

Level 1's conditions on the race's own track generator: mass, inertia, start, action noise and
random force as the race draws them, and gates and poles stay where they are reported. The
controller is unchanged: where it asks `topt_planner` for a plan, it is given a fast one. A
track without a clear plan still flies the last plan tried, as a race would have to, and is
reported as such.

Example::

    $ pixi run -e gpu python scripts/race_fast_guides.py --policy policy.npz --first_seed 8000 \
        --count 128 --out race.json
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import json
import logging
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import topt_planner
from lsy_drone_racing.control.rl_scene import RLScene
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.fast_runtime_planner import (
    FastPlannerConfig,
    plan_guide,
    smooth_velocities,
)
from lsy_drone_racing.utils.race_policy import body_center
from lsy_drone_racing.utils.scene_policy import scene_clearances

PASSES = 5
#: From this free distance on, what ended the flight was not the nearest object [m].
TOUCHING = 0.12


def fly(job: tuple[int, str, int, float, float, float]) -> dict:
    """Fly one seed, and return how it ended."""
    seed, policy, steps, thrust, tilt, velocity_window = job
    logging.disable(logging.WARNING)
    config = load_config(Path("config/level3.toml"))
    config.sim.render = False
    config.env.control_mode = "attitude"
    for key in ("gate_pos", "gate_rpy", "obstacle_pos"):
        del config.env.randomizations[key]
    config.controller.policy = policy
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
    obs, info = env.reset(seed=seed)
    made = {}

    def fast_plan(track: object, start: object, start_vel: object, _: object = None) -> object:
        planned, tries = plan_guide(
            track, start, start_vel, FastPlannerConfig(steps=steps, thrust=thrust, tilt=tilt)
        )
        made.update(clear=planned.clear, tries=tries, lap=planned.lap_time)
        t = planned.trajectory
        return SimpleNamespace(
            pos=t.pos,
            vel=smooth_velocities(t.vel, t.time, velocity_window),
            time=t.time,
            crossings=t.crossings,
        )

    original = topt_planner.plan
    topt_planner.plan = fast_plan
    controller = RLScene(obs, info, config)
    topt_planner.plan = original
    lib = controller._library
    finished, end_time, passes, cause = False, 0.0, 0, "too slow"
    for tick in range(int(15 * config.env.freq)):
        action = controller.compute_control(obs, info)
        before = obs
        obs, _, terminated, truncated, info = env.step(action)
        passes = int(obs["n_gates_passed"])
        end_time = (tick + 1) / config.env.freq
        if passes >= PASSES:
            finished, cause = True, "finished"
            break
        if terminated or truncated:
            pos = np.asarray(before["pos"], float)[None]
            quat = np.asarray(before["quat"], float)[None]
            rotations = Rotation.from_quat(np.asarray(before["gates_quat"], float)).as_matrix()[
                None
            ]
            poles, gates = scene_clearances(
                body_center(pos, quat),
                np.asarray(before["gates_pos"], float)[None],
                rotations,
                np.asarray(before["obstacles_pos"], float)[None],
                lib,
            )
            target = int(lib.sequence[min(passes, PASSES - 1)])
            if poles.min() < TOUCHING and poles.min() <= gates.min():
                cause = "pole"
            elif gates.min() < TOUCHING:
                cause = "next gate" if int(gates[0].argmin()) == target else "other gate"
            else:
                cause = "out or floor"
            break
    env.close()
    return dict(seed=seed, finished=finished, time=end_time, passes=passes, cause=cause, **made)


def main(
    policy: str,
    first_seed: int,
    count: int = 128,
    out: str | None = None,
    workers: int = 8,
    steps: int = 60,
    thrust: float = 15.0,
    tilt: float = 1.1,
    velocity_window: float = 0.15,
):
    """Fly a range of seeds.

    Args:
        policy: A policy written by `scripts/train_scene_policy.py`.
        first_seed: First race seed.
        count: Seeds to fly.
        out: JSON file for the flights.
        workers: Processes.
        steps: Gradient steps per try of the fast plan.
        thrust: Thrust limit of the fast plan [m/s^2].
        tilt: Tilt limit of the fast plan [rad].
        velocity_window: Width of the smoothing of the velocities [s].
    """
    jobs = [(first_seed + i, policy, steps, thrust, tilt, velocity_window) for i in range(count)]
    with mp.get_context("spawn").Pool(workers) as pool:
        flights = pool.map(fly, jobs, chunksize=1)
    done = [f for f in flights if f["finished"]]
    causes = {}
    for f in flights:
        causes[f["cause"]] = causes.get(f["cause"], 0) + 1
    summary = {
        "policy": policy,
        "first_seed": first_seed,
        "finished": len(done),
        "count": count,
        "time": float(np.mean([f["time"] for f in done])) if done else None,
        "causes": causes,
        "plans_clear": sum(f["clear"] for f in flights),
        "finished_when_plan_clear": sum(f["finished"] for f in flights if f["clear"]),
        "planned_lap": float(np.median([f["lap"] for f in flights])),
    }
    print(json.dumps(summary), flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"summary": summary, "flights": flights}, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
