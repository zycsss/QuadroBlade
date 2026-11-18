"""Fly `RLScene` on real level 2 or 3, where gates and poles move and are revealed late.

Each mode changes only how the guide is made, so the modes together say how much a correct guide is
worth and how much replanning recovers:

* ``topt``: `topt_planner`, once, through the reported layout (the committed final controller).
* ``topt_fast_limits``: the same with `FAST_GUIDE_CONFIG`.
* ``runtime``: the 20 ms runtime planner, once, through the reported layout.
* ``runtime_replan``: the runtime planner, again whenever the sensor reveals something.
* ``runtime_replan_in_order``: the same, but only once reveals complete the next stage of the order
  that `scripts/train_scene_policy.py --staged` trains with.
* ``runtime_true_layout``: the runtime planner, once, through the true layout. This cheats and is
  only an upper bound: the policy's inputs still show what the sensor shows.
* ``runtime_full_sensor``: the runtime planner, once, through the reported layout, while the
  policy's inputs show every true pose from the start. Cheats too: it isolates what late reveals
  cost the policy's inputs.
* ``runtime_true_layout_full_sensor``: both cheats, which should fly like level 1.

``--no_shifts`` keeps gates and poles where they are reported, as a check of the setup.
``--without gate_rpy`` (or ``gate_pos``, ``obstacle_pos``, comma-separated) removes one kind of
shift, and ``--gate_rpy_limit`` sets the largest gate roll, pitch and yaw shift.

For every flight that ends early it records what was hit, how far that object had moved from its
reported pose, and how long after its reveal the hit came.

Example::

    $ pixi run -e gpu python scripts/evaluate_reveals.py --level 3 --mode runtime_replan \
        --first_seed 9000 --count 128 --out level3_runtime_replan.json
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

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_scene import RLScene
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.race_policy import body_center
from lsy_drone_racing.utils.scene_policy import scene_clearances

MODES = (
    "topt",
    "topt_fast_limits",
    "runtime",
    "runtime_replan",
    "runtime_replan_in_order",
    "runtime_replan_selector",
    "runtime_true_layout",
    "runtime_full_sensor",
    "runtime_true_layout_full_sensor",
)
#: From this free distance on, what ended the flight was not the nearest object [m].
TOUCHING = 0.12
#: Longest race flown [s].
RACE_SECONDS = 15.0


def _controller_section(
    mode: str,
    policy: str | None,
    styles: tuple[str, ...] = (),
    replan_gates: int | None = None,
    benders: tuple[str, ...] = (),
) -> dict:
    section = {
        "topt": {"guide_planner": "topt"},
        "topt_fast_limits": {"guide_planner": "topt", "fast_guides": True},
        "runtime": {"guide_planner": "fast_runtime"},
        "runtime_replan": {"guide_planner": "fast_runtime", "replan_on_reveal": True},
        "runtime_replan_in_order": {
            "guide_planner": "fast_runtime",
            "replan_on_reveal": True,
            "replan_in_order": True,
        },
        "runtime_replan_selector": {
            "guide_planner": "fast_runtime",
            "replan_on_reveal": True,
            "style_selector": True,
        },
        "runtime_true_layout": {"guide_planner": "fast_runtime"},
        "runtime_full_sensor": {"guide_planner": "fast_runtime"},
        "runtime_true_layout_full_sensor": {"guide_planner": "fast_runtime"},
    }[mode]
    if replan_gates is not None:
        # Replans start at the drone and go through this many gates; 0 for the rest of the lap.
        section = section | {"replan_from_state": True, "replan_gates": replan_gates}
    if mode == "runtime_replan_selector":
        # The policy is the selector's, and the styles it chooses from can be given too.
        section |= {"selector": policy} if policy else {}
        section |= {"bender": True, "benders": list(benders)} if benders else {}
        return section | ({"style_policies": list(styles)} if styles else {})
    if benders:
        section |= {"bender": True, "bender_policy": benders[0]}
    return section | ({"policy": policy} if policy else {})


def _true_layout(env: object) -> dict[str, np.ndarray]:
    data = env.unwrapped.data
    return {
        "gates_pos": np.asarray(data.gates_pos[0], dtype=float),
        "gates_quat": np.asarray(data.gates_quat[0], dtype=float),
        "obstacles_pos": np.asarray(data.obstacles_pos[0], dtype=float),
    }


def fly(job: tuple) -> dict:
    """Fly one seed, and return how it ended."""
    seed, level, mode, policy, without, gate_rpy_limit, styles, replan_gates, benders = job
    logging.disable(logging.WARNING)
    config = load_config(Path(__file__).parents[1] / "config" / f"level{level}.toml")
    config.sim.render = False
    config.env.control_mode = "attitude"
    for key in without:
        del config.env.randomizations[key]
    if gate_rpy_limit is not None and "gate_rpy" in config.env.randomizations:
        kwargs = config.env.randomizations.gate_rpy.kwargs
        kwargs.minval = [-x for x in gate_rpy_limit]
        kwargs.maxval = list(gate_rpy_limit)
    for key, value in _controller_section(mode, policy, styles, replan_gates, benders).items():
        config.controller[key] = value
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
    true = _true_layout(env)
    shown = {k: np.asarray(obs[k], dtype=float) for k in true}
    moved_gates = np.linalg.norm(true["gates_pos"] - shown["gates_pos"], axis=1)
    moved_poles = np.linalg.norm(
        true["obstacles_pos"][:, :2] - shown["obstacles_pos"][:, :2], axis=1
    )
    true_guide = mode in ("runtime_true_layout", "runtime_true_layout_full_sensor")
    full_sensor = mode.endswith("full_sensor")
    controller = RLScene(obs | true if true_guide else obs, info, config)
    lib = controller._library
    n_gates = len(true["gates_pos"])
    revealed_at = np.full(n_gates + len(true["obstacles_pos"]), np.nan)
    passes, finished, end_time, cause, hit = 0, False, 0.0, "too slow", None
    for tick in range(int(RACE_SECONDS * config.env.freq)):
        seen = np.r_[
            np.asarray(obs["gates_visited"], bool), np.asarray(obs["obstacles_visited"], bool)
        ]
        revealed_at = np.where(seen & np.isnan(revealed_at), tick / config.env.freq, revealed_at)
        action = controller.compute_control(obs | true if full_sensor else obs, info)
        before = obs
        obs, _, terminated, truncated, info = env.step(action)
        passes = int(obs["n_gates_passed"])
        end_time = (tick + 1) / config.env.freq
        if passes >= len(lib.sequence):
            finished, cause = True, "finished"
            break
        if terminated or truncated:
            # Judged on the true layout, from the last step before the race ended it.
            pos = np.asarray(before["pos"], float)[None]
            quat = np.asarray(before["quat"], float)[None]
            rot = Rotation.from_quat(true["gates_quat"]).as_matrix()[None]
            poles, gates = scene_clearances(
                body_center(pos, quat),
                true["gates_pos"][None],
                rot,
                true["obstacles_pos"][None],
                lib,
            )
            target = int(lib.sequence[min(passes, len(lib.sequence) - 1)])
            if poles.min() < TOUCHING and poles.min() <= gates.min():
                cause, hit = "pole", n_gates + int(poles[0].argmin())
            elif gates.min() < TOUCHING:
                hit = int(gates[0].argmin())
                cause = "next gate" if hit == target else "other gate"
            else:
                cause = "out or floor"
            break
    env.close()
    replan_ms = [r["ms"] for r in controller.replans]
    moved = np.r_[moved_gates, moved_poles]
    return {
        "seed": seed,
        "finished": finished,
        "time": end_time,
        "passes": passes,
        "cause": cause,
        "hit_moved": None if hit is None else float(moved[hit]),
        "hit_after_reveal": None
        if hit is None or np.isnan(revealed_at[hit])
        else float(end_time - revealed_at[hit]),
        "first_plan_clear": bool(controller.first_plan_clear),
        "replans": len(replan_ms),
        "replans_failed": sum(not r["clear"] for r in controller.replans),
        "replan_ms": replan_ms,
        "style_ticks": controller.style_ticks.tolist(),
    }


def main(
    level: int = 3,
    mode: str = "topt",
    first_seed: int = 9000,
    count: int = 128,
    workers: int = 12,
    policy: str | None = None,
    out: str | None = None,
    no_shifts: bool = False,
    without: str | tuple[str, ...] = (),
    gate_rpy_limit: tuple[float, float, float] | None = None,
    styles: str | tuple[str, ...] = (),
    replan_gates: int | None = None,
    benders: str | tuple[str, ...] = (),
):
    """Fly `count` seeds of a level in one mode, and summarize how they ended.

    Args:
        level: 2 or 3.
        mode: How the guide is made, one of `MODES`.
        first_seed: Seed of the first race.
        count: Races.
        workers: Processes.
        policy: A policy file instead of the mode's default.
        out: Where to write every flight as JSON.
        no_shifts: Keep gates and poles where they are reported.
        without: Kinds of shift to remove: ``gate_pos``, ``gate_rpy`` or ``obstacle_pos``.
        gate_rpy_limit: Largest roll, pitch and yaw shift of a gate instead of the level's [rad].
        styles: With ``runtime_replan_selector``, the three style policies, comma separated, in
            place of the controller's; ``policy`` is then the selector.
        replan_gates: With a replanning mode, plan from the drone's position and velocity through
            this many gates, and again at every pass; 0 plans the rest of the lap.
        benders: Benders from `scripts/train_bender.py`, comma separated: one for ``policy``, or
            one per style with ``runtime_replan_selector``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if isinstance(without, str):
        without = tuple(w for w in without.split(",") if w)
    if no_shifts:
        without = ("gate_pos", "gate_rpy", "obstacle_pos")
    limit = None if gate_rpy_limit is None else tuple(float(x) for x in gate_rpy_limit)
    if isinstance(benders, str):
        benders = tuple(b for b in benders.split(",") if b)
    if isinstance(styles, str):
        styles = tuple(s for s in styles.split(",") if s)
    jobs = [
        (first_seed + i, level, mode, policy, tuple(without), limit)
        + (tuple(styles), replan_gates, tuple(benders))
        for i in range(count)
    ]
    with mp.get_context("spawn").Pool(workers) as pool:
        flights = pool.map(fly, jobs, chunksize=1)
    done = [f for f in flights if f["finished"]]
    causes = {}
    for f in flights:
        causes[f["cause"]] = causes.get(f["cause"], 0) + 1
    hits = [f for f in flights if f["hit_after_reveal"] is not None]
    replan_ms = [ms for f in flights for ms in f["replan_ms"]]
    summary = {
        "level": level,
        "mode": mode,
        "policy": policy,
        "without": list(without),
        "gate_rpy_limit": limit,
        "replan_gates": replan_gates,
        "benders": list(benders),
        "first_seed": first_seed,
        "finished": len(done),
        "count": count,
        "time": float(np.mean([f["time"] for f in done])) if done else None,
        "causes": causes,
        "first_plans_clear": sum(f["first_plan_clear"] for f in flights),
        "hit_moved_median_m": float(np.median([f["hit_moved"] for f in hits])) if hits else None,
        "hit_after_reveal_median_s": float(np.median([f["hit_after_reveal"] for f in hits]))
        if hits
        else None,
        "replans_per_race": float(np.mean([f["replans"] for f in flights])),
        "replans_failed": sum(f["replans_failed"] for f in flights),
        "replan_ms_median": float(np.median(replan_ms)) if replan_ms else None,
        "replan_ms_p95": float(np.percentile(replan_ms, 95)) if replan_ms else None,
        "style_share": (
            np.sum([f["style_ticks"] for f in flights], axis=0)
            / max(sum(sum(f["style_ticks"]) for f in flights), 1)
        ).tolist(),
    }
    print(json.dumps(summary), flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"summary": summary, "flights": flights}, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
