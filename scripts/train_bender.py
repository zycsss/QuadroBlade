"""Train a bender and its tracker together, with PPO, in one style.

The bender decides every ``hold`` control steps how to bend the guide ahead: a sideways and an
upward offset at four control points and a factor on the guide's speeds (`scene_policy.make_bend`).
Every bend starts at the drone, leaves along its velocity and asks only for offsets it can reach.
The tracker flies the bent guide at every step: it is a scene policy that sees the bent guide in
place of the planned one.

Both learn in one run, each with its own critic, advantage and reward:

* the bender from the style's race reward (`STYLES` of `scripts/train_scene_policy.py`), one
  semi-Markov transition per decision;
* the tracker from the same reward, minus ``follow`` times its distance from the bent guide, which
  keeps the bend meaningful. ``follow`` falls over the run, so that by the end the tracker may cut a
  bend where the race rewards it.

To keep two learners from chasing each other, the tracker starts from a trained style and the bender
from a bend of zero, so the pair first flies as that style does. For ``bender_only`` iterations only
the bender learns (and the tracker's critic). Then both do, the tracker at ``tracker_lr``. The
largest offset grows from ``limit_start`` to ``limit_end`` over ``limit_ramp`` iterations.

Example::

    $ pixi run -e gpu python scripts/train_bender.py --style safety_first \
        --tracker rl_scene_safety_first.npz --train staged_train.npz \
        --validation staged_validation.npz --out joint_bender_tracker_safety_first
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before

sys.path.insert(0, str(Path(__file__).parent))
from train_path_policy import Norm, forward, log_prob, normalize, update_norm  # noqa: E402
from train_race_policy import SIM_FREQ, summarize, zero_stats  # noqa: E402
from train_scene_policy import STYLES, SceneEnv, style_score  # noqa: E402

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402
from lsy_drone_racing.utils.path_policy import ACT_DIM, CONSTANT_INPUT_STD  # noqa: E402
from lsy_drone_racing.utils.race_policy import body_center, build_track  # noqa: E402
from lsy_drone_racing.utils.scene_policy import (  # noqa: E402
    BEND_DIM,
    BENDER_DIM,
    OBS_DIM,
    Bend,
    bend_offset,
    bend_path,
    bender_features,
    load_staged_library,
    make_bend,
    replanned_gap,
    scene_clearances,
    teacher_bend,
    zero_bend,
)

#: Offset from the replanned guide that costs an imitation weight of 1 [m].
PATH_SCALE = 0.1


class Scaling(NamedTuple):
    """The tracker's fixed input scaling, as `PathPolicy` applies it."""

    mean: jax.Array
    std: jax.Array
    constant: jax.Array


def tracker_input(feats: jax.Array, scaling: Scaling) -> jax.Array:
    """Return the tracker's scaled input, exactly as `PathPolicy` computes it in flight."""
    x = jnp.clip((feats - scaling.mean) / scaling.std, -10.0, 10.0)
    return jnp.where(scaling.constant, 0.0, x)


def _net(key: jax.Array, sizes: tuple, gain: float) -> list:
    orth = jax.nn.initializers.orthogonal
    layers = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        g = gain if i == len(sizes) - 2 else np.sqrt(2.0)
        layers.append((orth(g)(sub, (sizes[i], sizes[i + 1])), jnp.zeros(sizes[i + 1])))
    return layers


def init_params(
    key: jax.Array, tracker_file: str, hidden: tuple, critic_hidden: tuple, tracker_std: float
) -> tuple[dict, Scaling, float]:
    """Return a bender of zero bend and a tracker started from a trained policy.

    Returns:
        The parameters, the tracker's input scaling and its yaw limit, both kept from its file.
    """
    data = np.load(tracker_file)
    tracker = [
        (jnp.asarray(data[f"W{i}"]), jnp.asarray(data[f"b{i}"]))
        for i in range(int(data["n_layers"]))
    ]
    std = jnp.asarray(data["obs_std"])
    scaling = Scaling(jnp.asarray(data["obs_mean"]), std, std < CONSTANT_INPUT_STD)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "bender": {
            "actor": _net(k1, (BENDER_DIM, *hidden, BEND_DIM), 0.01),
            "critic": _net(k2, (BENDER_DIM, *critic_hidden, 1), 1.0),
            "log_std": jnp.full(BEND_DIM, -1.2),
        },
        "tracker": {
            "actor": tracker,
            "critic": _net(k3, (OBS_DIM, *critic_hidden, 1), 1.0),
            "log_std": jnp.full(ACT_DIM, float(np.log(tracker_std))),
        },
    }
    return params, scaling, float(data["yaw_max"]) if "yaw_max" in data.files else 0.0


def value(net: dict, x: jax.Array) -> jax.Array:
    """Return a critic's estimate."""
    return forward(net["critic"], x)[:, 0]


def ppo_loss(
    net: dict,
    batch: dict,
    clip: float,
    log_std_min: float,
    actor_scale: jax.Array,
    imitate: jax.Array = 0.0,
) -> jax.Array:
    """Return PPO's clipped objective and value loss; an ``actor_scale`` of 0 trains the critic.

    With ``gap`` and ``cap`` in the batch, ``imitate`` weighs how far the bend of the actor's mean
    leaves the guide from the replanned one (`replanned_gap`): per `PATH_SCALE`, squared, and
    averaged over the beads the bend covers.
    """
    log_std = jnp.maximum(net["log_std"], log_std_min)
    mean = forward(net["actor"], batch["obs"])
    ratio = jnp.exp(log_prob(mean, log_std, batch["act"]) - batch["logp"])
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
    vf = 0.5 * jnp.mean((value(net, batch["obs"]) - batch["ret"]) ** 2)
    loss = actor_scale * pg + 0.5 * vf
    if "gap" in batch:
        miss = bend_path(jnp.clip(mean, -1.0, 1.0), batch["cap"]) - batch["gap"]
        loss = loss + imitate * jnp.mean(jnp.sum(miss**2, axis=-1)) / PATH_SCALE**2
    return loss


def revealed_closeness(state: Any, lib: Any, margin: float, far: float) -> jax.Array:
    """Return how close drones are to what has been revealed, summed over poles and gate frames.

    Each counts 0 at ``far`` and beyond, rising linearly to 1 at ``margin``.
    """
    s = state.data.states
    poles, frames = scene_clearances(
        body_center(s.pos[:, 0], s.quat[:, 0]), state.gates_pos, state.gates_rot, state.poles, lib
    )

    def near(d: jax.Array, seen: jax.Array) -> jax.Array:
        return jnp.where(seen, jnp.clip((far - d) / (far - margin), 0.0, 1.0), 0.0).sum(axis=1)

    return near(poles, state.seen_poles) + near(frames, state.seen_gates)


def where_bend(mask: jax.Array, a: Bend, b: Bend) -> Bend:
    """Return bend ``a`` where ``mask`` is set and ``b`` elsewhere."""
    return Bend(*(jnp.where(mask.reshape(-1, *(1,) * (x.ndim - 1)), x, y) for x, y in zip(a, b)))


def _gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    next_values = jnp.concatenate([values[1:], last[None]])

    def back(gae: jax.Array, row: tuple) -> tuple:
        reward, val, next_val, done = row
        keep = 1.0 - done
        gae = reward + gamma * next_val * keep - val + gamma * lam * keep * gae
        return gae, gae

    _, adv = jax.lax.scan(
        back, jnp.zeros_like(last), (rewards, values, next_values, dones), reverse=True
    )
    return adv, adv + values


class Carry(NamedTuple):
    """What the rollout keeps between decisions."""

    state: Any
    bend: Bend
    bend_u: jax.Array


def _bender_input(env: SceneEnv, carry: Carry) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the bender's input, and every drone's position and velocity."""
    state = carry.state
    s = state.data.states
    pos, vel = s.pos[:, 0], s.vel[:, 0]
    _, _, poles = env.shown(state)
    obs = bender_features(
        env.observe(state),
        pos,
        state.index,
        state.passed,
        state.rows,
        env.lib,
        carry.bend_u,
        state.seen_gates,
        state.seen_poles,
        poles,
    )
    return obs, pos, vel


def make_train_step(
    env: SceneEnv,
    scaling: Scaling,
    cfg: dict,
    tx_bender: optax.GradientTransformation,
    tx_tracker: optax.GradientTransformation,
) -> Any:
    """Return one compiled iteration: ``horizon`` decisions in every world, then both updates."""
    hold, horizon = cfg["hold"], cfg["horizon"]
    gamma, lam, log_std_min = cfg["gamma"], cfg["lam"], cfg["log_std_min"]
    bender_gamma = cfg["bender_gamma"]
    teach = cfg["imitate"] > 0.0
    grad = jax.grad(ppo_loss)

    def decide(c: tuple, _: None) -> tuple:
        params, norm, carry, sched, stats, key = c
        limit, follow = sched[0], sched[1]
        key, k_b, k_ticks = jax.random.split(key, 3)
        obs, pos, vel = _bender_input(env, carry)
        norm = update_norm(norm, obs)
        xb = normalize(obs, norm)
        bender = params["bender"]
        mean_b = forward(bender["actor"], xb)
        log_std_b = jnp.maximum(bender["log_std"], log_std_min)
        ub = mean_b + jnp.exp(log_std_b) * jax.random.normal(k_b, mean_b.shape)
        lp_b, v_b = log_prob(mean_b, log_std_b, ub), value(bender, xb)
        state = carry.state
        bend = make_bend(ub, limit, pos, vel, state.index, state.rows, env.lib, cfg["from_drone"])
        if teach:
            reached = env.reached_rows(state)
            target = replanned_gap(
                limit, pos, vel, state.index, state.passed, state.rows, reached, env.lib
            )
        else:
            target = ()

        def tick(t: tuple, k: jax.Array) -> tuple:
            state, bend, bend_u, ret, disc, done, stats = t
            k_a, k_reset = jax.random.split(k)
            xt = tracker_input(env.observe(state, bend), scaling)
            tracker = params["tracker"]
            mean_t = forward(tracker["actor"], xt)
            log_std_t = jnp.maximum(tracker["log_std"], log_std_min)
            a = mean_t + jnp.exp(log_std_t) * jax.random.normal(k_a, mean_t.shape)
            lp_t, v_t = log_prob(mean_t, log_std_t, a), value(tracker, xt)
            new, r, term, trunc, step_stats = env.step(state, a, bend)
            r = jnp.nan_to_num(r)
            p = new.data.states.pos[:, 0]
            on_bend = env.lib.beads[new.rows, new.index] + bend_offset(bend, new.index)
            away = jnp.minimum(jnp.linalg.norm(p - on_bend, axis=-1), 0.5)
            r_t = jnp.nan_to_num(r - follow * away)
            r_b = r
            if cfg["clearance"] > 0.0:
                near = revealed_closeness(new, env.lib, cfg["hit_margin"], cfg["clearance_far"])
                r_b = r - cfg["clearance"] * near
            ended = term | trunc
            ret = ret + jnp.where(done, 0.0, disc * r_b)
            disc = disc * bender_gamma
            stats = jax.tree.map(jnp.add, stats, step_stats)
            # A new guide (a reveal's stage) or episode drops the bend until the next decision.
            moved = new.rows != state.rows
            state = env.reset(new, ended, k_reset)
            bend = where_bend(ended | moved, zero_bend(state.index), bend)
            bend_u = jnp.where(ended[:, None], 0.0, bend_u)
            out = (xt, a, lp_t, v_t, r_t, ended.astype(jnp.float32))
            return (state, bend, bend_u, ret, disc, done | ended, stats), out

        zeros = jnp.zeros(env.n)
        start = (
            state,
            bend,
            jnp.clip(ub, -1.0, 1.0),
            zeros,
            jnp.ones(env.n),
            jnp.zeros(env.n, dtype=bool),
            stats,
        )
        (state, bend, bend_u, ret, _, done, stats), track = jax.lax.scan(
            tick, start, jax.random.split(k_ticks, hold)
        )
        carry = Carry(state, bend, bend_u)
        out = (xb, ub, lp_b, v_b, ret, done.astype(jnp.float32), target)
        return (params, norm, carry, sched, stats, key), (out, track)

    def update(
        net: dict,
        opt: Any,
        tx: Any,
        data: dict,
        key: jax.Array,
        actor_scale: jax.Array,
        imitate: jax.Array = 0.0,
    ) -> tuple:
        size = data["obs"].shape[0]
        batch = size // cfg["minibatches"]

        def epoch(c: tuple, k: jax.Array) -> tuple:
            order = jax.random.permutation(k, size)[: cfg["minibatches"] * batch]

            def minibatch(c: tuple, idx: jax.Array) -> tuple:
                net, opt = c
                part = {name: v[idx] for name, v in data.items()}
                g = grad(net, part, cfg["clip"], log_std_min, actor_scale, imitate)
                updates, opt = tx.update(g, opt, net)
                return (optax.apply_updates(net, updates), opt), None

            return jax.lax.scan(minibatch, c, order.reshape(cfg["minibatches"], batch))[0], None

        return jax.lax.scan(epoch, (net, opt), jax.random.split(key, cfg["epochs"]))[0]

    @jax.jit
    def train_step(
        params: dict,
        opts: tuple,
        norm: Norm,
        carry: Carry,
        stats: Any,
        key: jax.Array,
        sched: jax.Array,
    ) -> tuple:
        c = (params, norm, carry, sched, stats, key)
        c, ((xb, ub, lp_b, v_b, ret_b, done_b, target), track) = jax.lax.scan(
            decide, c, None, length=horizon
        )
        params, norm, carry, _, stats, key = c
        # The tracker's steps in order: (horizon, hold, n, ...) -> (horizon * hold, n, ...).
        xt, a, lp_t, v_t, r_t, done_t = (v.reshape(horizon * hold, *v.shape[2:]) for v in track)
        last_t = value(
            params["tracker"], tracker_input(env.observe(carry.state, carry.bend), scaling)
        )
        adv_t, ret_t = _gae(r_t, v_t, done_t, last_t, gamma, lam)
        obs_b, _, _ = _bender_input(env, carry)
        last_b = value(params["bender"], normalize(obs_b, norm))
        adv_b, target_b = _gae(ret_b, v_b, done_b, last_b, bender_gamma**hold, lam)
        key, k_b, k_t = jax.random.split(key, 3)
        bender_data = {
            "obs": xb.reshape(-1, BENDER_DIM),
            "act": ub.reshape(-1, BEND_DIM),
            "logp": lp_b.ravel(),
            "adv": adv_b.ravel(),
            "ret": target_b.ravel(),
        }
        if teach:
            gap, cap = target
            bender_data["gap"] = gap.reshape(-1, *gap.shape[2:])
            bender_data["cap"] = cap.reshape(-1, cap.shape[-1])
        tracker_data = {
            "obs": xt.reshape(-1, OBS_DIM),
            "act": a.reshape(-1, ACT_DIM),
            "logp": lp_t.ravel(),
            "adv": adv_t.ravel(),
            "ret": ret_t.ravel(),
        }
        bender, opt_b = update(
            params["bender"], opts[0], tx_bender, bender_data, k_b, 1.0, sched[3]
        )
        tracker, opt_t = update(params["tracker"], opts[1], tx_tracker, tracker_data, k_t, sched[2])
        params = {"bender": bender, "tracker": tracker}
        return params, (opt_b, opt_t), norm, carry, stats, key

    return train_step


def make_lap_check(
    env: SceneEnv, scaling: Scaling, cfg: dict, decisions: int, teacher: bool = False
) -> Any:
    """Return compiled laps from the ground at rest, both networks without noise.

    With ``teacher`` the bends are the teacher's (`teacher_bend`) in place of the bender's, and the
    laps also report the offset the teacher's bends leave and the share of their points clipped.
    """
    hold = cfg["hold"]

    def lap(params: dict, norm: Norm, key: jax.Array, limit: jax.Array) -> dict:
        state = env.initial(key, ground=1.0, still=True)
        carry = Carry(state, zero_bend(state.index), jnp.zeros((env.n, BEND_DIM)))

        def decide(c: tuple, d: jax.Array) -> tuple:
            carry, gone, finish_time, gates, least, size, fit = c
            obs, pos, vel = _bender_input(env, carry)
            state = carry.state
            if teacher:
                reached = env.reached_rows(state)
                ub, left, clipped = teacher_bend(
                    limit, pos, vel, state.index, state.passed, state.rows, reached, env.lib
                )
                fit = fit + jnp.where(gone[:, None], 0.0, jnp.stack((left, clipped), axis=1))
            else:
                ub = forward(params["bender"]["actor"], normalize(obs, norm))
            bend = make_bend(
                ub, limit, pos, vel, state.index, state.rows, env.lib, cfg["from_drone"]
            )
            size = size + jnp.where(gone, 0.0, jnp.linalg.norm(bend.points, axis=-1).mean(axis=1))

            def tick(t: tuple, k: jax.Array) -> tuple:
                state, bend, gone, finish_time, gates, least = t
                x = tracker_input(env.observe(state, bend), scaling)
                a = forward(params["tracker"]["actor"], x)
                new, _, term, trunc, _ = env.step(state, a, bend)
                s = new.data.states
                poles, frames = scene_clearances(
                    body_center(s.pos[:, 0], s.quat[:, 0]),
                    new.gates_pos,
                    new.gates_rot,
                    new.poles,
                    env.lib,
                )
                gap = jnp.minimum(poles.min(axis=1), frames.min(axis=1))
                least = jnp.where(gone, least, jnp.minimum(least, gap))
                finished = new.passed >= env.lib.sequence.shape[0]
                finish_time = jnp.where(finished & ~gone, (d * hold + k + 1) / 50.0, finish_time)
                gates = jnp.where(gone, gates, new.passed)
                bend = where_bend(new.rows != state.rows, zero_bend(new.index), bend)
                return (new, bend, gone | term | trunc, finish_time, gates, least), None

            t0 = (carry.state, bend, gone, finish_time, gates, least)
            (state, bend, gone, finish_time, gates, least), _ = jax.lax.scan(
                tick, t0, jnp.arange(hold)
            )
            carry = Carry(state, bend, jnp.clip(ub, -1.0, 1.0))
            return (carry, gone, finish_time, gates, least, size, fit), None

        zeros = jnp.zeros(env.n)
        start = (
            carry,
            zeros.astype(bool),
            zeros,
            zeros.astype(jnp.int32),
            zeros + 9.0,
            zeros,
            jnp.zeros((env.n, 2)),
        )
        (_, _, finish_time, gates, least, size, fit), _ = jax.lax.scan(
            decide, start, jnp.arange(decisions)
        )
        finished = finish_time > 0
        laps = jnp.maximum(finished.sum(), 1)
        out = {
            "finish": finished.mean(),
            "time": jnp.sum(finish_time) / laps,
            "gates": gates.mean(),
            "clearance": jnp.sum(jnp.where(finished, least, 0.0)) / laps,
            "bend": size.mean() / decisions,
        }
        if teacher:
            out["fit_left"] = fit[:, 0].mean() / decisions
            out["clipped"] = fit[:, 1].mean() / decisions
        return out

    return jax.jit(lap)


def save_tracker(params: dict, scaling: Scaling, out: str, yaw_max: float):
    """Write the tracker for `PathPolicy`, with the input scaling it started with."""
    arrays = {
        "obs_mean": np.asarray(scaling.mean),
        "obs_std": np.asarray(scaling.std),
        "n_layers": len(params["tracker"]["actor"]),
        "yaw_max": yaw_max,
    }
    for i, (w, b) in enumerate(params["tracker"]["actor"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(out, **arrays)


def save_bender(params: dict, norm: Norm, out: str, hold: int, limit: float, from_drone: bool):
    """Write the bender for `PathPolicy`, with its hold, largest offset and where bends start."""
    arrays = {
        "obs_mean": np.asarray(norm.mean),
        "obs_std": np.asarray(jnp.sqrt(norm.var) + 1e-6),
        "n_layers": len(params["bender"]["actor"]),
        "hold": hold,
        "limit": limit,
        "from_drone": from_drone,
    }
    for i, (w, b) in enumerate(params["bender"]["actor"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(out, **arrays)


def main(
    train: str,
    validation: str,
    tracker: str,
    style: str = "middle_ground",
    out: str = "joint_bender_tracker",
    iterations: int = 4000,
    worlds: int = 8192,
    hold: int = 5,
    horizon: int = 8,
    epochs: int = 4,
    minibatches: int = 4,
    lr: float = 3e-4,
    tracker_lr: float = 1e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    seed: int = 0,
    eval_every: int = 100,
    bender_only: int = 500,
    limit_start: float = 0.05,
    limit_end: float = 0.3,
    limit_ramp: int = 2000,
    follow_start: float = 1.0,
    follow_end: float = 0.2,
    from_drone: bool = False,
    replan: bool = True,
    imitate: float = 0.0,
    imitate_fade: int = 1000,
    clearance: float = 0.0,
    clearance_far: float = 0.15,
    bender_gamma: float | None = None,
    teacher_check: bool = False,
    tracker_std: float = 0.1,
    log_std_min: float = -2.3,
    hidden: tuple[int, ...] = (128, 128),
    critic_hidden: tuple[int, ...] = (256, 256),
    ground: float = 0.25,
    episode_ticks: int = 500,
    mass_range: float = 0.005,
    inertia_range: float = 1e-6,
    force_range: float = 0.1,
    device: str = "gpu",
):
    """Train a bender and its tracker together, and keep the pair that does best on validation.

    Args:
        train: Staged libraries of `scripts/solve_staged_library.py`, comma separated.
        validation: Staged libraries that choose the pair that is kept.
        tracker: The trained style the tracker starts from.
        style: ``speed_first``, ``middle_ground`` or ``safety_first``: the reward of both networks,
            and how the kept pair is chosen (`style_score`).
        out: Prefix of the files written: ``<out>_tracker.npz`` and ``<out>_bender.npz``.
        iterations: PPO iterations.
        worlds: Simulated drones.
        hold: Control steps a bend holds for; 5 is 0.1 s.
        horizon: Decisions per world per iteration.
        epochs: Passes over each rollout, for each network.
        minibatches: Minibatches per pass.
        lr: The bender's learning rate.
        tracker_lr: The tracker's, once it learns.
        gamma: Discount per control step.
        lam: GAE parameter.
        clip: PPO clip.
        seed: Random seed.
        eval_every: Iterations between lap checks.
        bender_only: Iterations in which only the bender (and the tracker's critic) learns.
        limit_start: Largest offset of a bend at the start [m].
        limit_end: Largest offset once grown [m].
        limit_ramp: Iterations over which the largest offset grows.
        follow_start: The tracker's cost per metre from the bent guide, per step, at the start.
        follow_end: That cost at the end.
        from_drone: Start every bend at the drone, along its velocity, instead of on the guide.
        replan: Fly the guide of the stage that the reveals have reached, as flight does when it
            replans at every reveal. False keeps the guide planned through the layout as reported
            for the whole episode, so that only the bender adapts to what is revealed.
        imitate: Weight of how far the bender's bend leaves the guide from the one replanned at
            every reveal so far, over the beads the bend covers (`replanned_gap`). It fades
            linearly to 0 by ``imitate_fade`` iterations. Needs ``replan=False``.
        imitate_fade: Iterations over which the imitation fades out.
        clearance: Weight of the bender's local cost for closeness to what has been revealed: per
            pole and gate frame, 0 at ``clearance_far`` and beyond, 1 at the hit margin.
        clearance_far: Distance at which that cost starts [m].
        bender_gamma: The bender's discount per control step; the tracker's ``gamma`` if None.
        teacher_check: Only fly the validation laps with the teacher's bends, print them and stop.
        tracker_std: Action noise of the tracker at the start.
        log_std_min: Floor on the log of both networks' action noise.
        hidden: Hidden layers of the bender.
        critic_hidden: Hidden layers of both critics.
        ground: Share of episodes that start on the ground rather than along the guide.
        episode_ticks: Control steps per episode.
        mass_range: Largest mass offset of an episode [kg].
        inertia_range: Largest offset of each inertia entry [kg m^2].
        force_range: Largest random force per axis at each simulator step [N].
        device: Device to train on.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {tuple(STYLES)}")
    if (imitate > 0.0 or teacher_check) and replan:
        raise ValueError("the teacher bends a guide planned once: pass --replan=False")
    race = load_config(Path(__file__).parents[1] / "config" / "level2.toml")
    reference = build_track(race.env.track)
    libraries, scenarios = {}, {}
    for name, files in (("train", train), ("validation", validation)):
        libraries[name], scenarios[name] = load_staged_library(files, reference)
        print(f"{name}: {scenarios[name].gates_pos.shape[0]} shifted scenarios", flush=True)
    cfg = (
        dict(
            closer=10.0,
            progress=0.0,
            gate=20.0,
            finish=50.0,
            smooth=0.02,
            slow=0.1,
            slow_speed=2.0,
            ground=ground,
            randomize=1.0,
            start_vel=0.5,
            sensor_range=float(race.env.sensor_range),
            start_noise=0.05,
            episode_ticks=episode_ticks,
            mass_range=mass_range,
            inertia_range=inertia_range,
            hold=hold,
            horizon=horizon,
            gamma=gamma,
            lam=lam,
            clip=clip,
            epochs=epochs,
            minibatches=minibatches,
            log_std_min=log_std_min,
            from_drone=from_drone,
            replan=replan,
            imitate=imitate,
            clearance=clearance,
            clearance_far=clearance_far,
            bender_gamma=gamma if bender_gamma is None else bender_gamma,
        )
        | STYLES[style]
    )
    print(f"style {style}: {STYLES[style]}", flush=True)
    t0 = time.time()
    best, log = -1e9, []
    with jax.default_device(jax.devices(device)[0]):
        sim = Sim(
            n_worlds=worlds,
            n_drones=1,
            drone=race.sim.drone,
            control="attitude",
            dynamics=race.sim.dynamics,
            freq=SIM_FREQ,
            attitude_freq=SIM_FREQ,
            device=device,
        )
        if force_range > 0.0:

            def push(key: jax.Array, shape: tuple) -> jax.Array:
                return jax.random.uniform(key, shape, minval=-force_range, maxval=force_range)

            insert_fn_before(sim.step_pipeline, "integration", build_dynamics_disturbance_fn(push))
            sim.build_step_fn()
        env = SceneEnv(sim, libraries["train"], cfg, scenarios["train"])
        check_env = SceneEnv(sim, libraries["validation"], cfg, scenarios["validation"])
        key = jax.random.PRNGKey(seed)
        key, init, start, base = jax.random.split(key, 4)
        params, scaling, yaw_max = init_params(
            init, tracker, tuple(hidden), tuple(critic_hidden), tracker_std
        )
        check = make_lap_check(check_env, scaling, cfg, 1000 // hold)
        if teacher_check:
            taught = make_lap_check(check_env, scaling, cfg, 1000 // hold, teacher=True)
            for limit in (limit_end, 1.0):
                lap = jax.device_get(taught(params, Norm(0, 1, 1), base, jnp.asarray(limit)))
                lap = {k: float(v) for k, v in lap.items()}
                print(
                    f"teacher, limit {limit * 100:.0f} cm: {lap['finish'] * 100:5.1f}%"
                    f" {lap['gates']:.2f}g {lap['time']:.2f}s {lap['clearance'] * 100:.1f}cm"
                    f" bend {lap['bend'] * 100:.1f}cm left {lap['fit_left'] * 100:.1f}cm"
                    f" clipped {lap['clipped'] * 100:.1f}%",
                    flush=True,
                )
        tx_bender = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        tx_tracker = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(tracker_lr, eps=1e-5))
        opts = (tx_bender.init(params["bender"]), tx_tracker.init(params["tracker"]))
        norm = Norm(jnp.zeros(BENDER_DIM), jnp.ones(BENDER_DIM), jnp.array(1e-4))
        state = env.initial(start)
        carry = Carry(state, zero_bend(state.index), jnp.zeros((worlds, BEND_DIM)))
        stats = zero_stats()
        train_step = make_train_step(env, scaling, cfg, tx_bender, tx_tracker)

        # The started tracker on the validation laps, with no bend: what the pair has to beat.
        unbent = jax.device_get(check(params, norm, base, jnp.asarray(0.0)))
        unbent = {k: float(v) for k, v in unbent.items()}
        print(
            f"tracker alone, no bend: {unbent['finish'] * 100:5.1f}% {unbent['gates']:.2f}g"
            f" {unbent['time']:.2f}s {unbent['clearance'] * 100:.1f}cm"
            f" score {style_score(style, unbent):.2f}",
            flush=True,
        )
        if teacher_check:
            return
        timer = time.time()
        for it in range(iterations + 1):
            ramp = min(it / max(limit_ramp, 1), 1.0)
            limit = limit_start + (limit_end - limit_start) * ramp
            follow = follow_start + (follow_end - follow_start) * min(it / iterations, 1.0)
            actor = 0.0 if it < bender_only else 1.0
            taught = imitate * max(1.0 - it / max(imitate_fade, 1), 0.0)
            sched = jnp.asarray([limit, follow, actor, taught])
            params, opts, norm, carry, stats, key = train_step(
                params, opts, norm, carry, stats, key, sched
            )
            if it % eval_every != 0:
                continue
            jax.block_until_ready(params)
            rate = eval_every * worlds * horizon * hold / (time.time() - timer) if it else 0.0
            ep = summarize(stats)
            stats = zero_stats()
            val = jax.device_get(check(params, norm, base, jnp.asarray(limit)))
            val = {k: float(v) for k, v in val.items()}
            score = style_score(style, val)
            kept = score > best
            save_tracker(params, scaling, out + "_tracker.last.npz", yaw_max)
            save_bender(params, norm, out + "_bender.last.npz", hold, limit, from_drone)
            if kept:
                best = score
                save_tracker(params, scaling, out + "_tracker.npz", yaw_max)
                save_bender(params, norm, out + "_bender.npz", hold, limit, from_drone)
            log.append(
                dict(
                    it=it,
                    secs=round(time.time() - t0),
                    limit=limit,
                    follow=follow,
                    score=score,
                    validation=val,
                    episodes=ep,
                )
            )
            Path(out + ".log.json").write_text(json.dumps({"unbent": unbent, "log": log}))
            print(
                f"it {it:5} {log[-1]['secs']:5}s {rate / 1e6:4.1f}M/s limit {limit * 100:4.1f}cm"
                f" follow {follow:.2f} rew {ep['reward']:+7.3f} pole {ep['hit_pole'] * 100:3.0f}"
                f" gate {ep['hit_gate'] * 100:3.0f} fin {ep['finished'] * 100:3.0f}% | validation"
                f" {val['finish'] * 100:5.1f}% {val['gates']:.2f}g {val['time']:.2f}s"
                f" {val['clearance'] * 100:.1f}cm bend {val['bend'] * 100:.1f}cm"
                f" score {score:.2f}{' *' if kept else ''}",
                flush=True,
            )
            timer = time.time()


if __name__ == "__main__":
    fire.Fire(main)
