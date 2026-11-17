"""Train a closed-loop path-following policy with PPO on lines, turns and random curves.

The reward pays for progress along the path and only punishes leaving a corridor around it, so the
policy is free to cut inside if that is faster. Checked regularly on `mpcc_bench`'s primitives.

Everything between two bench checks is one compiled function on the device: the rollout, the
returns and the updates. Nothing goes back to the CPU until the next check.

Example::

    $ pixi run -e gpu python scripts/train_path_policy.py --out policy.npz
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from crazyflow.sim import Sim
from crazyflow.sim import functional as F
from crazyflow.sim.pipeline import insert_fn_before

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn
from lsy_drone_racing.utils import mpcc_bench
from lsy_drone_racing.utils.path_policy import (
    ACT_DIM,
    OBS_DIM,
    SPACING,
    YAW_MAX,
    PathPolicy,
    PathPolicyTracker,
    features,
    nearest,
    resample,
    to_action,
)

if TYPE_CHECKING:
    from collections.abc import Callable

PATH_BEADS = 240
SIM_FREQ = 500
#: Simulator steps per control step: the policy runs at 50 Hz.
SUBSTEPS = SIM_FREQ // 50
#: Rotor speed that holds the drone up, measured in the simulator [RPM].
HOVER_RPM = 15900.0


def random_path(rng: np.random.Generator) -> np.ndarray:
    """Return a random chain of lines and arcs with a gentle height change, ``(PATH_BEADS, 3)``."""
    pieces = [("line", rng.uniform(0.3, 2.0))]
    needed = PATH_BEADS * SPACING + 1
    while sum(p[1] if p[0] == "line" else p[1] * abs(p[2]) for p in pieces) < needed:
        if rng.random() < 0.3:
            pieces.append(("line", rng.uniform(0.2, 1.5)))
        else:
            sign = rng.choice((-1.0, 1.0))
            pieces.append(("arc", rng.uniform(0.25, 1.2), sign * rng.uniform(0.3, np.pi)))
    beads = mpcc_bench._build(pieces)
    wave = np.linspace(0.0, rng.uniform(0.0, 6.0), len(beads)) + rng.uniform(0.0, 6.0)
    beads[:, 2] = 1.2 + 0.4 * np.sin(wave)
    heading = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(heading), np.sin(heading)
    beads[:, :2] = beads[:, :2] @ np.array([[c, s], [-s, c]])
    return resample(beads)[:PATH_BEADS]


def path_pool(
    count: int, rng: np.random.Generator, names: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return random paths and the plan's speed along them, with the bench's paths among them.

    With `names`, the pool is only those bench paths.
    """
    paths, speeds = [], []
    named = [mpcc_bench.PRIMITIVES[k] for k in names or mpcc_bench.PRIMITIVES]
    if names:
        count = len(named)
    for i in range(count):
        if i < 4 * len(named):
            beads = resample(named[i % len(named)])
            tail = (beads[-1] - beads[-2]) / np.linalg.norm(beads[-1] - beads[-2])
            short = PATH_BEADS - len(beads)
            if short > 0:
                beads = np.r_[beads, beads[-1] + np.outer(np.arange(1, short + 1) * SPACING, tail)]
            beads = beads[:PATH_BEADS]
        else:
            beads = random_path(rng)
        speed, _ = mpcc_bench.speed_profile(beads)
        paths.append(beads)
        speeds.append(speed)
    return np.array(paths), np.array(speeds)


class EnvState(NamedTuple):
    """Every world's simulator data and its place on its path."""

    data: Any
    pid: jax.Array
    index: jax.Array
    prev: jax.Array
    ticks: jax.Array


class Norm(NamedTuple):
    """Running mean and variance of the features."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array


class Stats(NamedTuple):
    """How episodes went since the last bench check."""

    episodes: jax.Array
    ticks: jax.Array
    tipped: jax.Array
    strayed: jax.Array
    arrived: jax.Array
    commanded: jax.Array
    reward: jax.Array
    samples: jax.Array
    yaw: jax.Array


def zero_stats() -> Stats:
    """Return empty statistics."""
    return Stats(*(jnp.zeros(()) for _ in Stats._fields))


def summarize(stats: Stats) -> dict:
    """Return the statistics as shares of the episodes that ended and means per control step."""
    s = jax.device_get(stats)
    count, samples = max(float(s.episodes), 1.0), max(float(s.samples), 1.0)
    return {
        "episodes": float(s.episodes),
        "length": float(s.ticks) / count,
        "tipped": float(s.tipped) / count,
        "strayed": float(s.strayed) / count,
        "arrived": float(s.arrived) / count,
        "commanded": float(s.commanded) / samples,
        "reward": float(s.reward) / samples,
        "yaw": float(s.yaw) / samples,
    }


class PathEnv:
    """Many drones in one simulator, each on a path of the pool, as pure functions."""

    def __init__(self, sim: Sim, paths: np.ndarray, speeds: np.ndarray, cfg: dict):
        """Keep the simulator's functions and put the pool on its device."""
        self.sim = sim
        self.n = sim.n_worlds
        self.paths = jnp.asarray(paths, dtype=jnp.float32)
        self.speeds = jnp.asarray(speeds, dtype=jnp.float32)
        self.cfg = cfg
        if (force := cfg["force_range"]) > 0.0:
            # The race pushes every drone with a new random force at every simulator step.
            def push(key: jax.Array, shape: tuple) -> jax.Array:
                return jax.random.uniform(key, shape, minval=-force, maxval=force)

            insert_fn_before(sim.step_pipeline, "integration", build_dynamics_disturbance_fn(push))
            sim.build_step_fn()

    def initial(self, key: jax.Array) -> EnvState:
        """Return all worlds at the start of an episode."""
        zeros = jnp.zeros(self.n, dtype=jnp.int32)
        # Every world gets its own mass and inertia, so the parameters need a world axis.
        p = self.sim.data.params
        params = p.replace(
            mass=jnp.broadcast_to(p.mass, (self.n, 1, 1)),
            J=jnp.broadcast_to(p.J, (self.n, 1, 3, 3)),
            J_inv=jnp.broadcast_to(p.J_inv, (self.n, 1, 3, 3)),
        )
        data = self.sim.data.replace(params=params)
        state = EnvState(data, zeros, zeros, jnp.zeros((self.n, ACT_DIM)), zeros)
        return self.reset(state, jnp.ones(self.n, dtype=bool), key)

    def reset(self, state: EnvState, mask: jax.Array, key: jax.Array) -> EnvState:
        """Start the masked worlds on a new path, some from rest and some moving along it."""
        cfg, n = self.cfg, self.n
        keys = jax.random.split(key, 8)
        data = self.sim._reset(state.data, self.sim.default_data, mask)
        pid = jax.random.randint(keys[0], (n,), 0, self.paths.shape[0])
        moving = jax.random.uniform(keys[1], (n,)) < cfg["moving"]
        index = jnp.where(moving, jax.random.randint(keys[2], (n,), 0, PATH_BEADS - 100), 0)
        here = self.paths[pid, index] + 0.02 * jax.random.normal(keys[3], (n, 3))
        tangent = self.paths[pid, jnp.minimum(index + 1, PATH_BEADS - 1)] - self.paths[pid, index]
        tangent /= jnp.maximum(jnp.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        scale = jax.random.uniform(keys[4], (n,), minval=0.3, maxval=1.0)
        speed = self.speeds[pid, index] * scale * moving
        # `sim.reset` stops the rotors, and spinning up costs about 0.25 m of height. A handover
        # has them running; the bench does not, so some episodes must learn that start too.
        spun = jax.random.uniform(keys[5], (n,)) < cfg["spun"]
        where = mask[:, None, None]
        states = data.states
        states = states.replace(
            pos=jnp.where(where, here[:, None], states.pos),
            vel=jnp.where(where, (tangent * speed[:, None])[:, None], states.vel),
            rotor_vel=jnp.where(where, (HOVER_RPM * spun)[:, None, None], states.rotor_vel),
        )
        # Mass and inertia as the race randomizes them: an offset drawn when the episode starts.
        default = self.sim.default_data.params
        dm, dj = cfg["mass_range"], cfg["inertia_range"]
        mass = default.mass + jax.random.uniform(keys[6], (n, 1, 1), minval=-dm, maxval=dm)
        inertia = default.J + jax.random.uniform(keys[7], (n, 1, 3, 3), minval=-dj, maxval=dj)
        world = mask[:, None, None]
        params = data.params.replace(
            mass=jnp.where(world, mass, data.params.mass),
            J=jnp.where(world[..., None], inertia, data.params.J),
            J_inv=jnp.where(world[..., None], jnp.linalg.inv(inertia), data.params.J_inv),
        )
        return EnvState(
            data.replace(states=states, params=params),
            jnp.where(mask, pid, state.pid),
            jnp.where(mask, index, state.index),
            jnp.where(mask[:, None], 0.0, state.prev),
            jnp.where(mask, 0, state.ticks),
        )

    def observe(self, state: EnvState) -> jax.Array:
        """Return the features of every world."""
        s = state.data.states
        return features(
            s.pos[:, 0],
            s.vel[:, 0],
            s.quat[:, 0],
            s.ang_vel[:, 0],
            state.prev,
            self.paths,
            self.speeds,
            state.index,
            state.pid,
        )

    def step(self, state: EnvState, u: jax.Array) -> tuple:
        """Advance one control step.

        Returns:
            The new state, the reward, terminated (crash, lost, arrived), truncated and the
            statistics of this step.
        """
        cfg = self.cfg
        u = jnp.clip(u, -1.0, 1.0)
        action = to_action(u, cfg["yaw_max"])
        data = F.attitude_control(state.data, action[:, None])
        data = self.sim._step(data, n_steps=SUBSTEPS)
        pos, quat = data.states.pos[:, 0], data.states.quat[:, 0]
        index, error = nearest(pos, self.paths, state.index, state.pid)
        progress = (index - state.index) * SPACING
        z_up = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
        tipped = z_up < jnp.cos(1.5)
        strayed = error > cfg["lost"]
        lost = strayed | tipped | ~jnp.isfinite(pos).all(axis=1)
        change = jnp.sum((u - state.prev) ** 2, axis=1)
        reward = (
            cfg["alive"]
            + cfg["progress"] * progress
            - cfg["corridor"] * jnp.maximum(error - cfg["corridor_width"], 0.0)
            - cfg["smooth"] * change
            - cfg["yaw"] * action[:, 2] ** 2
        )
        reward = jnp.nan_to_num(jnp.where(lost, reward - cfg["crash"], reward))
        ticks = state.ticks + 1
        arrived = index >= PATH_BEADS - 2
        truncated = (ticks >= cfg["episode_ticks"]) & ~arrived
        ended = lost | arrived | truncated
        stats = Stats(
            episodes=ended.sum(),
            ticks=(ticks * ended).sum(),
            tipped=tipped.sum(),
            strayed=(strayed & ~tipped).sum(),
            arrived=(arrived & ~lost).sum(),
            commanded=action[:, 3].mean() / mpcc_bench.MASS,
            reward=reward.mean(),
            samples=jnp.ones(()),
            yaw=jnp.abs(action[:, 2]).mean(),
        )
        # Arriving really ends the episode. Bootstrapping there would read the critic on a state it
        # never trains on, since the world resets at once, and the value then runs away.
        return EnvState(data, state.pid, index, u, ticks), reward, lost | arrived, truncated, stats


def init_params(
    key: jax.Array, hidden: tuple, critic_hidden: tuple, obs_dim: int = OBS_DIM
) -> dict:
    """Return orthogonally initialized actor and critic weights."""
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
        "actor": net(a, (obs_dim, *hidden, ACT_DIM), 0.01),
        "critic": net(c, (obs_dim, *critic_hidden, 1), 1.0),
        "log_std": jnp.full(ACT_DIM, -1.0),
    }


def forward(layers: list, x: jax.Array) -> jax.Array:
    """Run a network."""
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = jax.nn.elu(x)
    return x


def log_prob(mean: jax.Array, log_std: jax.Array, a: jax.Array) -> jax.Array:
    """Return the log probability of actions under a diagonal Gaussian."""
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.9189385, axis=-1)


def value(params: dict, x: jax.Array) -> jax.Array:
    """Return the critic's estimate."""
    return forward(params["critic"], x)[:, 0]


def ppo_loss(params: dict, batch: dict, clip: float, ent: float, log_std_min: float) -> jax.Array:
    """Return PPO's clipped objective with the value loss and the entropy bonus."""
    log_std = jnp.maximum(params["log_std"], log_std_min)
    mean = forward(params["actor"], batch["obs"])
    lp = log_prob(mean, log_std, batch["act"])
    ratio = jnp.exp(lp - batch["logp"])
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
    vf = 0.5 * jnp.mean((value(params, batch["obs"]) - batch["ret"]) ** 2)
    entropy = jnp.sum(log_std + 1.4189385)
    return pg + 0.5 * vf - ent * entropy


def update_norm(norm: Norm, obs: jax.Array) -> Norm:
    """Merge a batch of features into the running mean and variance."""
    b = obs.shape[0]
    d = obs.mean(0) - norm.mean
    total = norm.count + b
    mean = norm.mean + d * b / total
    var = (norm.var * norm.count + obs.var(0) * b + d**2 * norm.count * b / total) / total
    return Norm(mean, var, total)


def normalize(obs: jax.Array, norm: Norm) -> jax.Array:
    """Return features scaled by the running statistics."""
    return jnp.clip((obs - norm.mean) / (jnp.sqrt(norm.var) + 1e-6), -10.0, 10.0)


def make_train_step(
    env: PathEnv,
    tx: optax.GradientTransformation,
    horizon: int,
    epochs: int,
    minibatches: int,
    gamma: float,
    lam: float,
    clip: float,
    ent: float,
    log_std_min: float,
) -> Callable:
    """Return one compiled PPO iteration: a rollout, its returns and the updates."""
    batch_size = env.n * horizon // minibatches
    grad = jax.grad(ppo_loss)

    def tick(carry: tuple, _: None) -> tuple:
        params, norm, state, stats, key = carry
        key, noise, restart = jax.random.split(key, 3)
        obs = env.observe(state)
        norm = update_norm(norm, obs)
        x = normalize(obs, norm)
        mean = forward(params["actor"], x)
        log_std = jnp.maximum(params["log_std"], log_std_min)
        a = mean + jnp.exp(log_std) * jax.random.normal(noise, mean.shape)
        lp = log_prob(mean, log_std, a)
        v = value(params, x)
        state, reward, term, trunc, step_stats = env.step(state, a)
        # A step cut off by the clock still has a future worth counting.
        boot = value(params, normalize(env.observe(state), norm))
        target = reward + gamma * jnp.where(trunc & ~term, boot, 0.0)
        done = term | trunc
        state = env.reset(state, done, restart)
        stats = jax.tree.map(jnp.add, stats, step_stats)
        return (params, norm, state, stats, key), (x, a, lp, v, target, done)

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

    def epoch(carry: tuple, key: jax.Array, data: dict) -> tuple:
        order = jax.random.permutation(key, env.n * horizon)
        order = order[: minibatches * batch_size].reshape(minibatches, batch_size)

        def minibatch(carry: tuple, idx: jax.Array) -> tuple:
            params, opt = carry
            batch = {k: v[idx] for k, v in data.items()}
            updates, opt = tx.update(grad(params, batch, clip, ent, log_std_min), opt, params)
            return (optax.apply_updates(params, updates), opt), None

        return jax.lax.scan(minibatch, carry, order)[0], None

    @jax.jit
    def train_step(
        params: dict, opt: optax.OptState, norm: Norm, state: EnvState, stats: Stats, key: jax.Array
    ) -> tuple:
        carry = (params, norm, state, stats, key)
        carry, (x, a, lp, v, target, done) = jax.lax.scan(tick, carry, None, length=horizon)
        params, norm, state, stats, key = carry
        last = value(params, normalize(env.observe(state), norm))
        adv, ret = returns(v, target, done, last)
        data = {
            "obs": x.reshape(env.n * horizon, -1),
            "act": a.reshape(-1, ACT_DIM),
            "logp": lp.ravel(),
            "adv": adv.ravel(),
            "ret": ret.ravel(),
        }
        key, shuffle = jax.random.split(key)
        (params, opt), _ = jax.lax.scan(
            lambda c, k: epoch(c, k, data), (params, opt), jax.random.split(shuffle, epochs)
        )
        return params, opt, norm, state, stats, key

    return train_step


def save(params: dict, norm: Norm, out: str, yaw_max: float):
    """Write the actor and the feature scaling for `PathPolicy`."""
    std = jnp.sqrt(norm.var) + 1e-6
    arrays = {"obs_mean": np.asarray(norm.mean), "obs_std": np.asarray(std)}
    arrays["n_layers"] = len(params["actor"])
    arrays["yaw_max"] = yaw_max
    for i, (w, b) in enumerate(params["actor"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(out, **arrays)


def bench(file: str, sim: Sim, names: list[str], mass_offset: float = 0.0) -> dict:
    """Fly the bench paths with a saved policy and a mass that differs from nominal by an offset."""
    policy = PathPolicy(file)

    def tracker(beads: np.ndarray, limit: np.ndarray) -> PathPolicyTracker:
        return PathPolicyTracker(policy, beads, limit)

    default = sim.default_data
    mass = jnp.full_like(default.params.mass, mpcc_bench.MASS + mass_offset)
    # `fly` resets the simulator, which restores the default data.
    sim.default_data = default.replace(params=default.params.replace(mass=mass))
    return {name: mpcc_bench.fly(name, sim=sim, tracker_fn=tracker) for name in names}


def bench_score(flights: dict) -> float:
    """Return how far a set of flights is from fast and tight; lower is better."""
    fs = flights.values()
    ratio = np.mean([f.time_ratio for f in fs])
    over = np.mean([max(f.error_max - 0.08, 0.0) for f in fs])
    return float(ratio + 5.0 * over + 5.0 * (len(flights) - sum(f.finished for f in fs)))


def main(
    out: str = "path_policy.npz",
    iterations: int = 1500,
    worlds: int = 8192,
    horizon: int = 32,
    epochs: int = 5,
    minibatches: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    ent: float = 0.0,
    pool: int = 1500,
    seed: int = 0,
    eval_every: int = 100,
    progress: float = 10.0,
    corridor: float = 20.0,
    corridor_width: float = 0.08,
    smooth: float = 0.02,
    crash: float = 10.0,
    lost: float = 0.3,
    episode_ticks: int = 300,
    paths: str | tuple[str, ...] = (),
    moving: float = 0.5,
    device: str = "gpu",
    spun: float = 0.75,
    log_std_min: float = -20.0,
    alive: float = 0.0,
    hidden: tuple[int, ...] = (64, 64),
    critic_hidden: tuple[int, ...] = (256, 256, 256),
    yaw: float = 0.0,
    yaw_max: float = YAW_MAX,
    mass_range: float = 0.005,
    inertia_range: float = 1e-6,
    force_range: float = 0.1,
    bench_mass: float = 0.005,
):
    """Train and keep the policy that does best on the bench.

    Args:
        out: Where to write the best policy.
        iterations: PPO iterations.
        worlds: Simulated drones.
        horizon: Control steps per iteration.
        epochs: Passes over each rollout.
        minibatches: Minibatches per pass.
        lr: Learning rate.
        gamma: Discount; at 50 Hz, 0.99 looks about 2 s ahead.
        lam: GAE parameter.
        clip: PPO clip.
        ent: Entropy bonus.
        pool: Number of paths.
        seed: Random seed.
        eval_every: Iterations between bench checks.
        progress: Reward per metre of path.
        corridor: Penalty per metre outside the corridor, per step.
        corridor_width: Half width of the corridor [m].
        smooth: Penalty on the change of the action.
        crash: Penalty for tipping over or leaving `lost`.
        lost: Distance from the path that ends an episode [m].
        episode_ticks: Control steps per episode.
        paths: Train and check only on these bench paths; all bench and random paths if empty.
        moving: Share of episodes that start moving along the path instead of at rest.
        device: Device to train on.
        spun: Share of episodes that start with the rotors at hover speed instead of stopped.
        log_std_min: Floor on the log of the action noise, so exploration cannot die out.
        alive: Reward per control step for staying on the path.
        hidden: Hidden layers of the policy, the only network that flies.
        critic_hidden: Hidden layers of the critic, which only exists during training.
        yaw: Penalty per control step on the square of the yaw command.
        yaw_max: Limit on the yaw command [rad]; flight uses the same one.
        mass_range: Largest mass offset of an episode [kg]; the default is level 2's.
        inertia_range: Largest offset of each inertia entry [kg m^2]; the default is level 2's.
        force_range: Largest random force per axis at each simulator step [N]; level 2's default.
        bench_mass: The bench also flies this much lighter and heavier than nominal [kg].
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    only = [paths] if isinstance(paths, str) else list(paths)
    names = only or list(mpcc_bench.PRIMITIVES)
    pool_paths, pool_speeds = path_pool(pool, rng, only)
    print(f"{len(pool_paths)} paths in {time.time() - t0:.1f}s", flush=True)
    cfg = dict(
        progress=progress,
        corridor=corridor,
        corridor_width=corridor_width,
        smooth=smooth,
        crash=crash,
        lost=lost,
        episode_ticks=episode_ticks,
        moving=moving,
        spun=spun,
        alive=alive,
        yaw=yaw,
        yaw_max=yaw_max,
        mass_range=mass_range,
        inertia_range=inertia_range,
        force_range=force_range,
    )
    # The bench flies one world at a time, which is faster on the CPU.
    cpu = jax.devices("cpu")[0]
    eval_sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf21B_500",
        control="attitude",
        dynamics="first_principles",
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
        device="cpu",
    )
    best = 1e9
    log = []
    with jax.default_device(jax.devices(device)[0]):
        sim = Sim(
            n_worlds=worlds,
            n_drones=1,
            drone="cf21B_500",
            control="attitude",
            dynamics="first_principles",
            freq=SIM_FREQ,
            attitude_freq=SIM_FREQ,
            device=device,
        )
        env = PathEnv(sim, pool_paths, pool_speeds, cfg)
        key = jax.random.PRNGKey(seed)
        key, init, start = jax.random.split(key, 3)
        params = init_params(init, tuple(hidden), tuple(critic_hidden))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        opt = tx.init(params)
        norm = Norm(jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), jnp.array(1e-4))
        state = env.initial(start)
        stats = zero_stats()
        train_step = make_train_step(
            env, tx, horizon, epochs, minibatches, gamma, lam, clip, ent, log_std_min
        )
        timer, done_since = time.time(), 0
        for it in range(iterations + 1):
            params, opt, norm, state, stats, key = train_step(params, opt, norm, state, stats, key)
            done_since += 1
            if it % eval_every != 0:
                continue
            jax.block_until_ready(params)
            rate = done_since * worlds * horizon / (time.time() - timer)
            ep = summarize(stats)
            stats = zero_stats()
            tmp = out + ".tmp.npz"
            save(params, norm, tmp, yaw_max)
            with jax.default_device(cpu):
                flights = bench(tmp, eval_sim, names)
                light = bench(tmp, eval_sim, names, -bench_mass)
                heavy = bench(tmp, eval_sim, names, bench_mass)
            ratio = np.mean([f.time_ratio for f in flights.values()])
            err = np.mean([f.error_max for f in flights.values()])
            high = np.mean([f.thrust_high for f in flights.values()])
            thrust = np.mean([f.thrust_mean for f in flights.values()])
            ok = sum(f.finished for f in flights.values())
            # A policy must hold up at every mass the race can draw, not only the nominal one.
            score = np.mean([bench_score(f) for f in (flights, light, heavy)])
            extremes = {
                label: {
                    "ok": sum(f.finished for f in group.values()),
                    "ratio": float(np.mean([f.time_ratio for f in group.values()])),
                    "err": float(np.mean([f.error_max for f in group.values()])),
                    "worst_err": float(max(f.error_max for f in group.values())),
                }
                for label, group in (("light", light), ("heavy", heavy))
            }
            kept = score < best
            if kept:
                best = score
                save(params, norm, out, yaw_max)
            std = float(jnp.exp(jnp.maximum(params["log_std"], log_std_min)).mean())
            row = dict(
                it=it,
                steps=(it + 1) * worlds * horizon,
                secs=round(time.time() - t0),
                rate=rate,
                ok=ok,
                ratio=float(ratio),
                err=float(err),
                thrust=float(thrust),
                high=float(high),
                std=std,
                score=float(score),
                episodes=ep,
                extremes=extremes,
                flights={k: vars(f) for k, f in flights.items()},
            )
            log.append(row)
            Path(out + ".log.json").write_text(json.dumps(log))
            print(
                f"it {it:5} {row['steps'] / 1e6:7.1f}M {row['secs']:5}s {rate / 1e3:7.0f}k/s"
                f" rew {ep['reward']:+7.3f} ep {ep['length']:5.1f} tip {ep['tipped'] * 100:3.0f}%"
                f" stray {ep['strayed'] * 100:3.0f}% arr {ep['arrived'] * 100:3.0f}%"
                f" cmd {ep['commanded']:4.1f} yaw {ep['yaw']:.2f}"
                f" ok {ok}/{len(names)} ratio {ratio:.2f}"
                f" err {err:.3f} thrust {thrust:.1f} high {high * 100:3.0f}% std {std:.2f}"
                f" | light {extremes['light']['ok']} {extremes['light']['ratio']:.2f}"
                f" {extremes['light']['err']:.3f}"
                f" heavy {extremes['heavy']['ok']} {extremes['heavy']['ratio']:.2f}"
                f" {extremes['heavy']['err']:.3f}"
                f"{' *' if kept else ''}",
                flush=True,
            )
            timer, done_since = time.time(), 0


if __name__ == "__main__":
    fire.Fire(main)
