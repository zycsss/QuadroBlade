"""Fly one saved scene policy over the same tracks, on `topt_planner` guides and on fast guides.

This asks whether fast guides can stand in for solved ones, without training anything. A policy
trained on solved guides has never seen a fast one, so its result on them is a lower bound on what
training on them would give. Only tracks where the fast plan passes its check are flown, under both
guides, so the comparison is paired.

Example::

    $ pixi run -e gpu python scripts/fly_fast_guides.py --policy policy.npz \
        --library .tmp/topt/rl2/scenes/fixed_level3_validation.npz --out_dir fast_guides
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import json
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before
from scipy.spatial.transform import Rotation
from train_path_policy import SIM_FREQ, Norm
from train_scene_policy import SceneEnv, make_lap_check

from lsy_drone_racing.control.topt_planner import Track
from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.fast_runtime_planner import FastPlannerConfig, plan_guide
from lsy_drone_racing.utils.race_policy import TAKEOFF_LIFT, TAKEOFF_SPEED, build_track
from lsy_drone_racing.utils.scene_policy import load_library

SEQUENCE = np.array([0, 1, 2, 3, 1])
LAYOUT = ("start", "gates_pos", "gates_rpy", "poles")
#: The settings of `scripts/train_scene_policy.py` at level 1 difficulty.
ENV_CONFIG = dict(
    closer=10.0,
    progress=0.0,
    gate=20.0,
    finish=50.0,
    crash=20.0,
    hit_margin=0.03,
    obstacle=0.02,
    obstacle_scale=1.0,
    smooth=0.02,
    tick=0.05,
    slow=0.1,
    slow_speed=2.0,
    ground=0.25,
    randomize=0.0,
    start_vel=0.5,
    start_noise=0.05,
    episode_ticks=500,
    mass_range=0.005,
    inertia_range=1e-6,
)
FORCE_RANGE = 0.1


def _write_libraries(
    library: str,
    config: FastPlannerConfig,
    fast_out: Path,
    topt_out: Path,
    match_speed: bool = False,
) -> int:
    """Write the tracks with a clear fast plan, once per guide, and return how many there are.

    With ``match_speed`` the fast plan is retimed to the solved plan's lap: the same path, and only
    the speed it hints at changes.
    """
    fast, topt, count = {}, {}, 0
    with np.load(library) as data:
        for i in range(int(data["count"])):
            layout = {k: data[f"{k}_{i}"] for k in LAYOUT}
            rotations = Rotation.from_euler("xyz", layout["gates_rpy"]).as_matrix()
            track = Track(
                layout["gates_pos"],
                rotations,
                layout["poles"],
                SEQUENCE,
                np.ones(5),
                tuple(False for _ in SEQUENCE),
            )
            start = layout["start"] + np.array([0.0, 0.0, TAKEOFF_LIFT])
            plan, _ = plan_guide(track, start, np.array([0.0, 0.0, TAKEOFF_SPEED]), config)
            if not plan.clear:
                continue
            t = plan.trajectory
            stretch = plan.lap_time / float(data[f"time_{i}"][-1]) if match_speed else 1.0
            for name, value in (
                ("pos", t.pos),
                ("vel", t.vel * stretch),
                ("time", t.time / stretch),
                ("crossings", t.crossings),
                *layout.items(),
            ):
                fast[f"{name}_{count}"] = value
            for name in ("pos", "vel", "time", "crossings", *LAYOUT):
                topt[f"{name}_{count}"] = data[f"{name}_{i}"]
            count += 1
    np.savez(fast_out, count=count, **fast)
    np.savez(topt_out, count=count, **topt)
    return count


def main(
    policy: str,
    library: str,
    out_dir: str,
    worlds: int = 2048,
    seeds: int = 3,
    budgets: str = "60:15,60:14,300:15",
):
    """Fly a policy on solved guides and on fast guides of the same tracks.

    Each budget is flown three ways: solved guides, fast guides, and fast guides retimed to the
    solved plan's lap, which tells a path the policy cannot use from a speed it does not expect.

    Args:
        policy: A policy written by `scripts/train_scene_policy.py`.
        library: A library of solved guides.
        out_dir: Where the matched libraries and the results go.
        worlds: Drones per check. Each flies a track drawn from the library.
        seeds: Checks per guide, whose results are averaged.
        budgets: Fast plans to try, as ``steps:thrust`` separated by commas.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    race = load_config(Path(__file__).parents[1] / "config" / "level2.toml")
    reference = build_track(race.env.track)
    sim = Sim(
        n_worlds=worlds,
        n_drones=1,
        drone=race.sim.drone,
        control="attitude",
        dynamics=race.sim.dynamics,
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
    )

    def push(key: jax.Array, shape: tuple) -> jax.Array:
        return jax.random.uniform(key, shape, minval=-FORCE_RANGE, maxval=FORCE_RANGE)

    insert_fn_before(sim.step_pipeline, "integration", build_dynamics_disturbance_fn(push))
    sim.build_step_fn()
    cfg = dict(ENV_CONFIG, sensor_range=float(race.env.sensor_range))
    data = np.load(policy)
    actor = [
        (jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"]))
        for i in range(int(data["n_layers"]))
    ]
    std = jnp.asarray(data["obs_std"])
    norm = Norm(jnp.asarray(data["obs_mean"]), (std - 1e-6) ** 2, jnp.array(1.0))

    def fly(file: Path) -> dict:
        env = SceneEnv(sim, load_library(str(file), reference), cfg)
        check = make_lap_check(env, 1000)
        laps = [
            jax.device_get(check({"actor": actor}, norm, jax.random.PRNGKey(s)))
            for s in range(seeds)
        ]
        return {k: float(np.mean([lap[k] for lap in laps])) for k in laps[0]}

    results = {"policy": policy, "library": library, "worlds": worlds, "seeds": seeds, "runs": []}
    for budget in budgets.split(","):
        steps, thrust = budget.split(":")
        name = f"steps_{steps}_thrust_{thrust}"
        config = FastPlannerConfig(steps=int(steps), thrust=float(thrust))
        fast_file, topt_file = folder / f"fast_{name}.npz", folder / f"topt_matching_{name}.npz"
        count = _write_libraries(library, config, fast_file, topt_file)
        run = {"budget": name, "tracks": count, "topt": fly(topt_file), "fast": fly(fast_file)}
        retimed = folder / f"fast_retimed_{name}.npz"
        _write_libraries(library, config, retimed, topt_file, match_speed=True)
        run["fast_retimed_to_solved_lap"] = fly(retimed)
        results["runs"].append(run)
        print(json.dumps(run), flush=True)
        (folder / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
