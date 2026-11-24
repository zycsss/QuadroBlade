"""Train the selector that picks one of the three style policies in flight, with PPO.

The three styles of `scripts/train_scene_policy.py` (``speed_first``, ``middle_ground`` and
``safety_first``) stay frozen. Every ``hold`` control steps the selector sees the scene features,
the style it flies now and what each style would command, and picks the style that flies the next
``hold`` steps. Its reward is the race's own objective and is the same whichever style it picks:
gates passed, a finish, a crash, and a cost per step. The selector therefore learns when each
style's risk is worth its speed.

A decision and the steps it holds for form one transition (a semi-Markov decision): the reward of
the steps in between is discounted step by step, and the next decision is discounted by ``gamma``.

The kept selector is the one whose deterministic laps of the validation scenarios score best by
``10 * finish - time + gates`` (finishing first, then time), the score of ``middle_ground``.

Example::

    $ pixi run -e gpu python scripts/train_style_selector.py \
        --train scenes/staged_level3_train.npz,scenes/staged_generated_train.npz \
        --validation scenes/staged_level3_validation.npz \
        --styles speed.npz,middle.npz,safety.npz --out style_selector.npz
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
from train_bender import where_bend  # noqa: E402
from train_path_policy import Norm, forward, normalize, update_norm  # noqa: E402
from train_race_policy import SIM_FREQ  # noqa: E402
from train_scene_policy import SceneEnv  # noqa: E402

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402
from lsy_drone_racing.utils.path_policy import ACT_DIM, CONSTANT_INPUT_STD  # noqa: E402
from lsy_drone_racing.utils.race_policy import build_track  # noqa: E402
from lsy_drone_racing.utils.scene_policy import (  # noqa: E402
    BEND_DIM,
    FIRST_STYLE,
    OBS_DIM,
    STYLE_NAMES,
    Bend,
    bender_features,
    load_staged_library,
    make_bend,
    selector_features,
    zero_bend,
)

N_STYLES = len(STYLE_NAMES)


class Styles(NamedTuple):
    """The frozen style policies, stacked along a first axis of three."""

    mean: jax.Array
    std: jax.Array
    constant: jax.Array
    layers: list


def load_styles(files: list[str]) -> Styles:
    """Load the three style policies in the order of `STYLE_NAMES`."""
    data = [np.load(f) for f in files]
    count = int(data[0]["n_layers"])
    layers = [
        (
            jnp.stack([jnp.asarray(d[f"W{i}"]) for d in data]),
            jnp.stack([jnp.asarray(d[f"b{i}"]) for d in data]),
        )
        for i in range(count)
    ]
    std = jnp.stack([jnp.asarray(d["obs_std"]) for d in data])
    return Styles(
        jnp.stack([jnp.asarray(d["obs_mean"]) for d in data]), std, std < CONSTANT_INPUT_STD, layers
    )


def style_actions(styles: Styles, feats: jax.Array) -> jax.Array:
    """Return every style's clipped action, ``(n, 3, ACT_DIM)``, as `PathPolicy` computes it.

    ``feats`` is ``(n, d)``, the same for every style, or ``(n, 3, d)``, one set per style.
    """
    feats = feats[:, None] if feats.ndim == 2 else feats
    x = jnp.clip((feats - styles.mean) / styles.std, -10.0, 10.0)
    x = jnp.where(styles.constant, 0.0, x)
    for i, (w, b) in enumerate(styles.layers):
        x = jnp.einsum("nsi,sio->nso", x, w) + b
        if i < len(styles.layers) - 1:
            x = jax.nn.elu(x)
    return jnp.clip(x, -1.0, 1.0)


class Benders(NamedTuple):
    """The frozen benders of the three styles, stacked as `Styles`, with how they bend."""

    nets: Styles
    hold: int
    limits: tuple[float, ...]
    from_drone: tuple[bool, ...]


class Pilots(NamedTuple):
    """Every world's bend of each style's guide, and each bender's last output ``(n, 3, d)``."""

    bends: tuple[Bend, ...]
    u: jax.Array


def load_benders(files: list[str]) -> Benders:
    """Load the benders of `scripts/train_bender.py` in the order of `STYLE_NAMES`."""
    holds = {int(np.load(f)["hold"]) for f in files}
    if len(holds) != 1:
        raise ValueError("the benders must all decide at the same rate")
    limits = tuple(float(np.load(f)["limit"]) for f in files)
    starts = tuple(bool(np.load(f).get("from_drone", False)) for f in files)
    return Benders(load_styles(files), holds.pop(), limits, starts)


def no_pilots(env: SceneEnv, index: jax.Array) -> Pilots:
    """Return no bend at all, for every style."""
    return Pilots(
        tuple(zero_bend(index) for _ in STYLE_NAMES), jnp.zeros((env.n, N_STYLES, BEND_DIM))
    )


def propose(
    env: SceneEnv, styles: Styles, benders: Benders | None, state: Any, pilots: Pilots | None
) -> tuple[jax.Array, jax.Array, Pilots | None]:
    """Return the features, every style's action ``(n, 3, ACT_DIM)`` and the bends flown.

    With benders, each bends its style's guide every ``hold`` steps of an episode, as `RLScene`.
    """
    feats = env.observe(state)
    if benders is None:
        return feats, style_actions(styles, feats), pilots
    s = state.data.states
    pos, vel = s.pos[:, 0], s.vel[:, 0]
    _, _, poles = env.shown(state)
    decide = state.ticks % benders.hold == 0
    inputs = jnp.stack(
        [
            bender_features(
                feats,
                pos,
                state.index,
                state.passed,
                state.rows,
                env.lib,
                pilots.u[:, i],
                state.seen_gates,
                state.seen_poles,
                poles,
            )
            for i in range(N_STYLES)
        ],
        axis=1,
    )
    outputs = style_actions(benders.nets, inputs)
    bends, bent = [], []
    for i in range(N_STYLES):
        made = make_bend(
            outputs[:, i],
            benders.limits[i],
            pos,
            vel,
            state.index,
            state.rows,
            env.lib,
            benders.from_drone[i],
        )
        bends.append(where_bend(decide, made, pilots.bends[i]))
        bent.append(env.observe(state, bends[-1]))
    u = jnp.where(decide[:, None, None], outputs, pilots.u)
    return feats, style_actions(styles, jnp.stack(bent, axis=1)), Pilots(tuple(bends), u)


def chosen_bend(pilots: Pilots | None, style: jax.Array) -> Bend | None:
    """Return every world's bend of the style it flies, whose frame its command is in."""
    if pilots is None:
        return None
    bend = pilots.bends[0]
    for i in range(1, N_STYLES):
        bend = where_bend(style == i, pilots.bends[i], bend)
    return bend


def drop_bends(pilots: Pilots | None, new: Any, old: Any, ended: jax.Array) -> Pilots | None:
    """Drop the bends of worlds on a new guide or a new episode; forget outputs of a new episode."""
    if pilots is None:
        return None
    moved = ended | (new.rows != old.rows)
    zero = zero_bend(new.index)
    bends = tuple(where_bend(moved, zero, b) for b in pilots.bends)
    return Pilots(bends, jnp.where(ended[:, None, None], 0.0, pilots.u))


def init_selector(key: jax.Array, obs_dim: int, hidden: tuple, critic_hidden: tuple) -> dict:
    """Return orthogonally initialized actor (logits over the styles) and critic weights."""
    orth = jax.nn.initializers.orthogonal

    def net(key: jax.Array, sizes: tuple, gain: float) -> list:
        layers = []
        for i in range(len(sizes) - 1):
            key, sub = jax.random.split(key)
            g = gain if i == len(sizes) - 2 else np.sqrt(2.0)
            layers.append((orth(g)(sub, (sizes[i], sizes[i + 1])), jnp.zeros(sizes[i + 1])))
        return layers

    a, c = jax.random.split(key)
    return {
        "actor": net(a, (obs_dim, *hidden, N_STYLES), 0.01),
        "critic": net(c, (obs_dim, *critic_hidden, 1), 1.0),
    }


def value(params: dict, x: jax.Array) -> jax.Array:
    """Return the critic's estimate."""
    return forward(params["critic"], x)[:, 0]


def selector_loss(params: dict, batch: dict, clip: float, ent: float) -> jax.Array:
    """Return PPO's clipped objective for a categorical policy, the value loss and entropy."""
    logp_all = jax.nn.log_softmax(forward(params["actor"], batch["obs"]))
    lp = jnp.take_along_axis(logp_all, batch["act"][:, None], axis=1)[:, 0]
    ratio = jnp.exp(lp - batch["logp"])
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
    vf = 0.5 * jnp.mean((value(params, batch["obs"]) - batch["ret"]) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(logp_all) * logp_all, axis=1))
    return pg + 0.5 * vf - ent * entropy


class Carry(NamedTuple):
    """What the rollout keeps between decisions."""

    state: Any
    style: jax.Array
    pilots: Pilots | None


def make_flight(env: SceneEnv, styles: Styles, cfg: dict, benders: Benders | None = None) -> Any:
    """Return a function flying one step with each world's style, and the race reward."""
    count = env.lib.sequence.shape[0]

    def step(state: Any, style: jax.Array, pilots: Pilots | None) -> tuple:
        _, proposals, pilots = propose(env, styles, benders, state, pilots)
        u = proposals[jnp.arange(env.n), style]
        new, _, term, trunc, _ = env.step(state, u, chosen_bend(pilots, style))
        finished = new.passed >= count
        crashed = term & ~finished
        reward = (
            cfg["gate"] * (new.passed - state.passed)
            + cfg["finish"] * finished
            - cfg["crash"] * crashed
            - cfg["tick"]
        )
        return new, reward, term, trunc, finished, crashed, pilots

    return step


def make_train_step(
    env: SceneEnv,
    styles: Styles,
    cfg: dict,
    tx: optax.GradientTransformation,
    horizon: int,
    epochs: int,
    minibatches: int,
    clip: float,
    ent: float,
    benders: Benders | None = None,
) -> Any:
    """Return one compiled PPO iteration: ``horizon`` decisions in every world, then the updates."""
    hold, gamma, lam = cfg["hold"], cfg["gamma"], cfg["lam"]
    gamma_tick = gamma ** (1.0 / hold)
    batch_size = env.n * horizon // minibatches
    fly = make_flight(env, styles, cfg, benders)
    grad = jax.grad(selector_loss)

    def observe(carry: Carry) -> jax.Array:
        feats, proposals, _ = propose(env, styles, benders, carry.state, carry.pilots)
        return selector_features(feats, carry.style, proposals)

    def decide(c: tuple, _: None) -> tuple:
        params, norm, carry, stats, key = c
        key, k_act, k_reset = jax.random.split(key, 3)
        obs = observe(carry)
        norm = update_norm(norm, obs)
        x = normalize(obs, norm)
        logits = forward(params["actor"], x)
        a = jax.random.categorical(k_act, logits)
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], axis=1)[:, 0]
        v = value(params, x)

        def tick(t: tuple, k: jax.Array) -> tuple:
            state, style, pilots, ret, disc, done, stats = t
            old = state
            state, r, term, trunc, finished, crashed, pilots = fly(state, style, pilots)
            ended = term | trunc
            ret = ret + jnp.where(done, 0.0, disc * jnp.nan_to_num(r))
            disc = disc * gamma_tick
            fresh = ended & ~done
            stats = {
                "episodes": stats["episodes"] + fresh.sum(),
                "finished": stats["finished"] + (fresh & finished).sum(),
                "crashed": stats["crashed"] + (fresh & crashed).sum(),
                "ticks": stats["ticks"] + jnp.where(fresh, state.ticks, 0).sum(),
                "chosen": stats["chosen"],
            }
            done = done | ended
            # A world that ended starts again at once, flying the first style until it decides.
            state = env.reset(state, ended, k)
            pilots = drop_bends(pilots, state, old, ended)
            style = jnp.where(ended, FIRST_STYLE, style)
            return (state, style, pilots, ret, disc, done, stats), None

        zeros = jnp.zeros(env.n)
        stats = stats | {"chosen": stats["chosen"] + jax.nn.one_hot(a, N_STYLES).sum(axis=0)}
        start = (
            carry.state,
            a,
            carry.pilots,
            zeros,
            jnp.ones(env.n),
            jnp.zeros(env.n, dtype=bool),
            stats,
        )
        (state, style, pilots, ret, _, done, stats), _ = jax.lax.scan(
            tick, start, jax.random.split(k_reset, hold)
        )
        carry = Carry(state, style, pilots)
        return (params, norm, carry, stats, key), (x, a, lp, v, ret, done)

    def returns(values: jax.Array, rewards: jax.Array, dones: jax.Array, last: jax.Array) -> tuple:
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

    def epoch(c: tuple, key: jax.Array, data: dict) -> tuple:
        order = jax.random.permutation(key, env.n * horizon)
        order = order[: minibatches * batch_size].reshape(minibatches, batch_size)

        def minibatch(c: tuple, idx: jax.Array) -> tuple:
            params, opt = c
            batch = {k: v[idx] for k, v in data.items()}
            updates, opt = tx.update(grad(params, batch, clip, ent), opt, params)
            return (optax.apply_updates(params, updates), opt), None

        return jax.lax.scan(minibatch, c, order)[0], None

    @jax.jit
    def train_step(
        params: dict, opt: Any, norm: Norm, carry: Carry, stats: dict, key: Any
    ) -> tuple:
        c = (params, norm, carry, stats, key)
        c, (x, a, lp, v, ret, done) = jax.lax.scan(decide, c, None, length=horizon)
        params, norm, carry, stats, key = c
        last = value(params, normalize(observe(carry), norm))
        adv, target = returns(v, ret, done.astype(jnp.float32), last)
        data = {
            "obs": x.reshape(env.n * horizon, -1),
            "act": a.ravel(),
            "logp": lp.ravel(),
            "adv": adv.ravel(),
            "ret": target.ravel(),
        }
        key, shuffle = jax.random.split(key)
        (params, opt), _ = jax.lax.scan(
            lambda c, k: epoch(c, k, data), (params, opt), jax.random.split(shuffle, epochs)
        )
        return params, opt, norm, carry, stats, key

    return train_step


def make_lap_check(
    env: SceneEnv, styles: Styles, cfg: dict, decisions: int, benders: Benders | None = None
) -> Any:
    """Return compiled laps from the ground at rest, choosing styles by the most likely one.

    ``forced`` of 0, 1 or 2 flies that style throughout instead, for comparison; -1 lets the
    selector choose.
    """
    hold = cfg["hold"]
    fly = make_flight(env, styles, cfg, benders)

    def lap(params: dict, norm: Norm, key: jax.Array, forced: jax.Array) -> dict:
        state = env.initial(key, ground=1.0, still=True)
        pilots = None if benders is None else no_pilots(env, state.index)

        def decide(c: tuple, d: jax.Array) -> tuple:
            state, style, pilots, gone, finish_time, gates, chosen = c
            feats, proposals, _ = propose(env, styles, benders, state, pilots)
            obs = selector_features(feats, style, proposals)
            picked = forward(params["actor"], normalize(obs, norm)).argmax(axis=1)
            style = jnp.where(forced >= 0, forced, picked)
            chosen = chosen + jnp.where(gone[:, None], 0.0, jax.nn.one_hot(style, N_STYLES))

            def tick(t: tuple, k: jax.Array) -> tuple:
                state, pilots, gone, finish_time, gates = t
                old = state
                state, _, term, trunc, finished, _, pilots = fly(state, style, pilots)
                pilots = drop_bends(pilots, state, old, jnp.zeros(env.n, dtype=bool))
                tick_count = d * hold + k + 1
                finish_time = jnp.where(finished & ~gone, tick_count / 50.0, finish_time)
                gates = jnp.where(gone, gates, state.passed)
                return (state, pilots, gone | term | trunc, finish_time, gates), None

            (state, pilots, gone, finish_time, gates), _ = jax.lax.scan(
                tick, (state, pilots, gone, finish_time, gates), jnp.arange(hold)
            )
            return (state, style, pilots, gone, finish_time, gates, chosen), None

        zeros = jnp.zeros(env.n)
        start = (
            state,
            jnp.full(env.n, FIRST_STYLE),
            pilots,
            jnp.zeros(env.n, dtype=bool),
            zeros,
            zeros.astype(jnp.int32),
            jnp.zeros((env.n, N_STYLES)),
        )
        (_, _, _, _, finish_time, gates, chosen), _ = jax.lax.scan(
            decide, start, jnp.arange(decisions)
        )
        finished = finish_time > 0
        laps = jnp.maximum(finished.sum(), 1)
        return {
            "finish": finished.mean(),
            "time": jnp.sum(finish_time) / laps,
            "gates": gates.mean(),
            "chosen": chosen.sum(axis=0) / jnp.maximum(chosen.sum(), 1.0),
        }

    return jax.jit(lap)


def score(lap: dict) -> float:
    """Return how a validation lap check ranks: finishing first, then time."""
    seconds = lap["time"] if lap["finish"] > 0 else 30.0
    return 10.0 * lap["finish"] - seconds + lap["gates"]


def save(params: dict, norm: Norm, out: str, styles: list[str], hold: int):
    """Write the selector in `PathPolicy`'s format, with its hold and the styles it chose from."""
    arrays = {
        "obs_mean": np.asarray(norm.mean),
        "obs_std": np.asarray(jnp.sqrt(norm.var) + 1e-6),
        "n_layers": len(params["actor"]),
        "hold": hold,
        "styles": np.array([Path(s).name for s in styles]),
    }
    for i, (w, b) in enumerate(params["actor"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(out, **arrays)


def main(
    train: str,
    validation: str,
    styles: str,
    benders: str = "",
    out: str = "style_selector.npz",
    iterations: int = 1500,
    worlds: int = 8192,
    hold: int = 10,
    horizon: int = 16,
    epochs: int = 4,
    minibatches: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.95,
    lam: float = 0.95,
    clip: float = 0.2,
    ent: float = 0.01,
    seed: int = 0,
    eval_every: int = 50,
    gate: float = 1.0,
    finish: float = 10.0,
    crash: float = 5.0,
    tick: float = 0.02,
    hit_margin: float = 0.02,
    ground: float = 0.5,
    episode_ticks: int = 600,
    hidden: tuple[int, ...] = (64, 64),
    critic_hidden: tuple[int, ...] = (256, 256),
    mass_range: float = 0.005,
    inertia_range: float = 1e-6,
    force_range: float = 0.1,
    device: str = "gpu",
):
    """Train the selector on staged scenarios and keep the one that does best on validation.

    Args:
        train: Staged libraries of `scripts/solve_staged_library.py` to train on, comma separated.
        validation: Staged libraries that choose the selector that is kept.
        styles: The ``speed_first``, ``middle_ground`` and ``safety_first`` policies, comma
            separated, in that order.
        benders: Their benders from `scripts/train_bender.py`, comma separated in the same order,
            when ``styles`` are the trackers trained with them.
        out: Where to write the best selector.
        iterations: PPO iterations.
        worlds: Simulated drones.
        hold: Control steps a choice holds for; 10 is 0.2 s.
        horizon: Decisions per world per iteration.
        epochs: Passes over each rollout.
        minibatches: Minibatches per pass.
        lr: Learning rate.
        gamma: Discount per decision.
        lam: GAE parameter.
        clip: PPO clip.
        ent: Entropy bonus.
        seed: Random seed.
        eval_every: Iterations between lap checks.
        gate: Reward for passing a gate.
        finish: Reward for the last pass.
        crash: Penalty for a crash.
        tick: Cost per control step: 0.02 makes a second worth 1% of the finishes.
        hit_margin: Clearance below which something counts as hit [m]; ``speed_first``'s.
        ground: Share of episodes that start on the ground rather than along the guide.
        episode_ticks: Control steps per episode.
        hidden: Hidden layers of the selector.
        critic_hidden: Hidden layers of the critic.
        mass_range: Largest mass offset of an episode [kg]; level 2's.
        inertia_range: Largest offset of each inertia entry [kg m^2]; level 2's.
        force_range: Largest random force per axis at each simulator step [N].
        device: Device to train on.
    """
    files = styles.split(",") if isinstance(styles, str) else list(styles)
    if len(files) != N_STYLES:
        raise ValueError(f"styles needs {N_STYLES} files, in the order {STYLE_NAMES}")
    race = load_config(Path(__file__).parents[1] / "config" / "level2.toml")
    reference = build_track(race.env.track)
    libraries, scenarios = {}, {}
    for name, f in (("train", train), ("validation", validation)):
        libraries[name], scenarios[name] = load_staged_library(f, reference)
        print(f"{name}: {scenarios[name].gates_pos.shape[0]} shifted scenarios", flush=True)
    cfg = dict(
        # Read by `SceneEnv`; its own reward terms are not used, only when an episode ends.
        closer=0.0,
        progress=0.0,
        gate=gate,
        finish=finish,
        crash=crash,
        hit_margin=hit_margin,
        obstacle=0.0,
        obstacle_scale=1.0,
        smooth=0.0,
        tick=tick,
        slow=0.0,
        slow_speed=1.0,
        ground=ground,
        randomize=1.0,
        start_vel=0.5,
        sensor_range=float(race.env.sensor_range),
        start_noise=0.05,
        episode_ticks=episode_ticks,
        mass_range=mass_range,
        inertia_range=inertia_range,
        hold=hold,
        gamma=gamma,
        lam=lam,
    )
    t0 = time.time()
    best, log = -1e9, []
    obs_dim = OBS_DIM + N_STYLES + N_STYLES * ACT_DIM
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
        frozen = load_styles(files)
        bender_files = benders.split(",") if isinstance(benders, str) else list(benders)
        bender_files = [b for b in bender_files if b]
        if bender_files and len(bender_files) != N_STYLES:
            raise ValueError(f"benders needs {N_STYLES} files, in the order {STYLE_NAMES}")
        bent = load_benders(bender_files) if bender_files else None
        if bent is not None and hold % bent.hold:
            raise ValueError("hold must be a multiple of the benders' hold")
        env = SceneEnv(sim, libraries["train"], cfg, scenarios["train"])
        check_env = SceneEnv(sim, libraries["validation"], cfg, scenarios["validation"])
        check = make_lap_check(check_env, frozen, cfg, 1000 // hold, bent)
        key = jax.random.PRNGKey(seed)
        key, init, start, base = jax.random.split(key, 4)
        params = init_selector(init, obs_dim, tuple(hidden), tuple(critic_hidden))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        opt = tx.init(params)
        norm = Norm(jnp.zeros(obs_dim), jnp.ones(obs_dim), jnp.array(1e-4))
        state = env.initial(start)
        pilots = None if bent is None else no_pilots(env, state.index)
        carry = Carry(state, jnp.full(worlds, FIRST_STYLE), pilots)

        # Every style alone on the same validation laps: what the selector has to beat.
        baselines = {}
        for i, name in enumerate(STYLE_NAMES):
            lap = jax.device_get(check(params, norm, base, jnp.asarray(i)))
            baselines[name] = {k: np.asarray(v).tolist() for k, v in lap.items()}
            b = baselines[name]
            print(
                f"{name:14} alone: {b['finish'] * 100:5.1f}% {b['gates']:.2f}g {b['time']:.2f}s"
                f" score {score(b):.2f}",
                flush=True,
            )

        def empty() -> dict:
            z = jnp.zeros(())
            return dict(episodes=z, finished=z, crashed=z, ticks=z, chosen=jnp.zeros(N_STYLES))

        stats = empty()
        train_step = make_train_step(
            env, frozen, cfg, tx, horizon, epochs, minibatches, clip, ent, bent
        )
        for it in range(iterations + 1):
            params, opt, norm, carry, stats, key = train_step(params, opt, norm, carry, stats, key)
            if it % eval_every != 0:
                continue
            ep = {k: np.asarray(v) for k, v in jax.device_get(stats).items()}
            stats = empty()
            n_ep = max(float(ep["episodes"]), 1.0)
            share = ep["chosen"] / max(ep["chosen"].sum(), 1.0)
            # The same laps as the baselines, so the comparison is exact.
            val = jax.device_get(check(params, norm, base, jnp.asarray(-1)))
            val = {k: np.asarray(v).tolist() for k, v in val.items()}
            s = score(val)
            kept = s > best
            save(params, norm, out + ".last.npz", files, hold)
            if kept:
                best = s
                save(params, norm, out, files, hold)
            log.append(dict(it=it, secs=round(time.time() - t0), score=s, validation=val))
            Path(out + ".log.json").write_text(json.dumps({"baselines": baselines, "log": log}))
            print(
                f"it {it:5} {log[-1]['secs']:5}s fin {ep['finished'] / n_ep * 100:3.0f}%"
                f" crash {ep['crashed'] / n_ep * 100:3.0f}% ep {ep['ticks'] / n_ep:5.1f}"
                f" picks {' '.join(f'{p * 100:3.0f}' for p in share)}"
                f" | validation {val['finish'] * 100:5.1f}% {val['gates']:.2f}g"
                f" {val['time']:.2f}s picks {' '.join(f'{p * 100:3.0f}' for p in val['chosen'])}"
                f" score {s:.2f}{' *' if kept else ''}",
                flush=True,
            )


if __name__ == "__main__":
    fire.Fire(main)
