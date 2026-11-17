"""Train a racing policy with PPO on one track: the guide path is a hint, the gates are the goal.

The policy sees a guide path but is never held to it. The reward pays for getting closer to the next
gate, more for passing it and most for the finish. It charges for time, for being near anything that
can be hit, as ``exp(-k * d)`` of the distance to it, and for hitting it. So the policy decides for
itself when to follow the path, when to cut a corner, and when to do what the path does not say.

It also charges for flying slowly: without that and the cost of time, hovering far from everything
is the best a cautious policy can do. A cost per step adds up to about ``1 / (1 - gamma)`` times
itself, and that must stay below the crash penalty, or crashing becomes the way out of hovering.

Every episode gets its own moved gates and poles and a moved, moving start, by default as level 2
moves them. As in the race, the policy sees gates and poles at their nominal poses until it comes
within sensor range, while collisions and passes always use the true poses. The guide stays the plan
of the nominal track, like the plan the race's MPCC starts from.

Half of the episodes start on the ground as the race does, and the others at random places along the
guide with the plan's speed and attitude, so the later gates are met long before the first flights
get there. Regularly every world flies a lap from the ground without action noise, which measures
the finish rate and the lap time directly.

Example::

    $ pixi run -e gpu python scripts/train_race_policy.py --config level0.toml --out race.npz
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
import optax
from crazyflow.sim import Sim
from crazyflow.sim import functional as F
from crazyflow.sim.pipeline import insert_fn_before
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from train_path_policy import (  # noqa: E402
    HOVER_RPM,
    Norm,
    forward,
    init_params,
    make_train_step,
    normalize,
    save,
)

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402
from lsy_drone_racing.utils.mpcc_bench import MASS  # noqa: E402
from lsy_drone_racing.utils.path_policy import ACT_DIM, SPACING, to_action  # noqa: E402
from lsy_drone_racing.utils.race_policy import (  # noqa: E402
    OBS_DIM,
    RaceTrack,
    advance_guide,
    body_center,
    build_track,
    clearances,
    crossed_gate,
    euler_matrix,
    euler_quat,
    race_features,
)

SIM_FREQ = 500
#: Simulator steps per control step: the race runs at 50 Hz.
SUBSTEPS = SIM_FREQ // 50
#: The floor only ends a run away from the take-off platform [m].
PLATFORM_RADIUS = 0.2
#: How level 2 moves things: the largest offset of a gate's position and its roll, pitch and yaw, of
#: a pole, of the drone's start and of its attitude [m, rad].
GATE_SHIFT = (0.15, 0.15, 0.1)
GATE_TURN = (0.05, 0.1, 0.2)
POLE_SHIFT = (0.15, 0.15, 0.05)
DRONE_SHIFT = (0.1, 0.1)
DRONE_RISE = 0.02
DRONE_TILT = 0.1


class RaceState(NamedTuple):
    """Every world's simulator data and its progress through the race."""

    data: Any
    index: jax.Array
    passed: jax.Array
    prev: jax.Array
    last_pos: jax.Array
    ticks: jax.Array
    funnel: jax.Array
    gates_pos: jax.Array
    gates_rot: jax.Array
    poles: jax.Array
    seen_gates: jax.Array
    seen_poles: jax.Array


class RaceStats(NamedTuple):
    """How episodes went since the last check."""

    episodes: jax.Array
    ticks: jax.Array
    hit_pole: jax.Array
    hit_gate: jax.Array
    out: jax.Array
    tipped: jax.Array
    finished: jax.Array
    gates: jax.Array
    commanded: jax.Array
    reward: jax.Array
    gap: jax.Array
    guide: jax.Array
    samples: jax.Array


def zero_stats() -> RaceStats:
    """Return empty statistics."""
    return RaceStats(*(jnp.zeros(()) for _ in RaceStats._fields))


def summarize(stats: RaceStats) -> dict:
    """Return the statistics as shares of the episodes that ended and means per control step."""
    s = jax.device_get(stats)
    count, samples = max(float(s.episodes), 1.0), max(float(s.samples), 1.0)
    shares = ("hit_pole", "hit_gate", "out", "tipped", "finished")
    return {
        "episodes": float(s.episodes),
        "length": float(s.ticks) / count,
        **{k: float(getattr(s, k)) / count for k in shares},
        "gates": float(s.gates) / count,
        "commanded": float(s.commanded) / samples,
        "reward": float(s.reward) / samples,
        "gap": float(s.gap) / samples,
        "guide": float(s.guide) / samples,
    }


class RaceEnv:
    """Many drones racing one track in one simulator, as pure functions for `make_train_step`."""

    def __init__(self, sim: Sim, track: RaceTrack, cfg: dict):
        """Keep the simulator's functions and put the track on its device."""
        self.sim = sim
        self.n = sim.n_worlds
        self.cfg = cfg
        integers = {"sequence", "crossings"}
        self.track = RaceTrack(
            **{
                k: jnp.asarray(v, dtype=jnp.int32 if k in integers else jnp.float32)
                for k, v in track._asdict().items()
            }
        )
        self.gates_rpy = jnp.asarray(
            Rotation.from_matrix(track.gates_rot).as_euler("xyz"), dtype=jnp.float32
        )
        # The last pass a random start can be placed before.
        self.last_bead = int(track.crossings[-1])
        if (force := cfg["force_range"]) > 0.0:

            def push(key: jax.Array, shape: tuple) -> jax.Array:
                return jax.random.uniform(key, shape, minval=-force, maxval=force)

            insert_fn_before(sim.step_pipeline, "integration", build_dynamics_disturbance_fn(push))
            sim.build_step_fn()

    def initial(
        self, key: jax.Array, ground: float | None = None, still: bool = False
    ) -> RaceState:
        """Return all worlds at the start of an episode."""
        t, n = self.track, self.n
        zeros = jnp.zeros(n, dtype=jnp.int32)
        state = RaceState(
            self.sim.data,
            zeros,
            zeros,
            jnp.zeros((n, ACT_DIM)),
            jnp.zeros((n, 3)),
            zeros,
            jnp.zeros(n),
            jnp.broadcast_to(t.gates_pos, (n, *t.gates_pos.shape)),
            jnp.broadcast_to(t.gates_rot, (n, *t.gates_rot.shape)),
            jnp.broadcast_to(t.poles, (n, *t.poles.shape)),
            jnp.zeros((n, t.gates_pos.shape[0]), dtype=bool),
            jnp.zeros((n, t.poles.shape[0]), dtype=bool),
        )
        return self.reset(state, jnp.ones(n, dtype=bool), key, ground, still)

    def reset(
        self,
        state: RaceState,
        mask: jax.Array,
        key: jax.Array,
        ground: float | None = None,
        still: bool = False,
    ) -> RaceState:
        """Start the masked worlds on a new layout, on the ground or along the guide.

        Args:
            state: The current state.
            mask: The worlds to start again.
            key: Random key.
            ground: Share of starts on the ground; the configured share if None.
            still: Start at rest, as the race does, instead of with a random velocity.
        """
        t, cfg, n = self.track, self.cfg, self.n
        ground = cfg["ground"] if ground is None else ground
        size = cfg["randomize"]
        keys = jax.random.split(key, 12)
        data = self.sim._reset(state.data, self.sim.default_data, mask)
        count = t.gates_pos.shape[0]

        def spread(k: jax.Array, limit: tuple, shape: tuple) -> jax.Array:
            limit = size * jnp.asarray(limit)
            return jax.random.uniform(k, (*shape, len(limit)), minval=-limit, maxval=limit)

        gate_shift = spread(keys[4], GATE_SHIFT, (n, count))
        gates_pos = t.gates_pos + gate_shift
        gates_rot = euler_matrix(self.gates_rpy + spread(keys[5], GATE_TURN, (n, count)))
        poles = t.poles + spread(keys[6], POLE_SHIFT, (n, t.poles.shape[0]))

        on_ground = jax.random.uniform(keys[0], (n,)) < ground
        index = jax.random.randint(keys[1], (n,), 0, self.last_bead)
        index = jnp.where(on_ground, 0, index)
        passed = jnp.searchsorted(t.crossings, index, side="right")

        # A start along the guide moves with the gates around it, or it could lie inside a frame.
        last_pass = t.sequence.shape[0] - 1
        ahead = jnp.minimum(passed, last_pass)
        behind = jnp.maximum(passed - 1, 0)
        span = jnp.maximum(t.crossings[ahead] - t.crossings[behind], 1)
        share = jnp.where(passed > 0, (index - t.crossings[behind]) / span, 1.0)[:, None]
        worlds = jnp.arange(n)
        bend = share * gate_shift[worlds, t.sequence[ahead]]
        bend = bend + (1.0 - share) * jnp.where(
            passed[:, None] > 0, gate_shift[worlds, t.sequence[behind]], 0.0
        )
        scale = jax.random.uniform(keys[2], (n,), minval=0.5, maxval=1.0)
        noise = cfg["start_noise"] * size * jax.random.normal(keys[3], (n, 3))
        along = t.beads[index] + bend + noise
        along_vel = t.velocities[index] * scale[:, None]

        shift = spread(keys[7], DRONE_SHIFT, (n,))
        rise = size * DRONE_RISE * jax.random.uniform(keys[8], (n, 1))
        pos = jnp.where(on_ground[:, None], t.start + jnp.c_[shift, rise], along)

        # Standing on the floor it can only leave upwards.
        limit = cfg["start_vel"]
        velocity = jax.random.uniform(keys[9], (n, 3), minval=-limit, maxval=limit)
        takeoff = velocity * jnp.array([1.0, 1.0, 0.0]) + jnp.array([0.0, 0.0, 1.0]) * jnp.abs(
            velocity
        )
        vel = jnp.where(on_ground[:, None], takeoff, along_vel + velocity)
        vel = jnp.where(jnp.asarray(still) & on_ground[:, None], 0.0, vel)

        tilt = spread(keys[10], (DRONE_TILT, DRONE_TILT, DRONE_TILT), (n,))
        quat = jnp.where(on_ground[:, None], euler_quat(tilt), t.quats[index])
        rotor = jnp.where(on_ground, 0.0, HOVER_RPM)
        where = mask[:, None, None]
        s = data.states
        states = s.replace(
            pos=jnp.where(where, pos[:, None], s.pos),
            vel=jnp.where(where, vel[:, None], s.vel),
            quat=jnp.where(where, quat[:, None], s.quat),
            rotor_vel=jnp.where(where, rotor[:, None, None], s.rotor_vel),
        )
        seen_gates, seen_poles = self._in_range(pos, gates_pos, poles)
        return RaceState(
            data.replace(states=states),
            jnp.where(mask, index, state.index),
            jnp.where(mask, passed, state.passed),
            jnp.where(mask[:, None], 0.0, state.prev),
            jnp.where(mask[:, None], pos, state.last_pos),
            jnp.where(mask, 0, state.ticks),
            jnp.where(
                mask, self._funnel(jnp.linalg.norm(t.beads[index] - pos, axis=1)), state.funnel
            ),
            jnp.where(where, gates_pos, state.gates_pos),
            jnp.where(mask[:, None, None, None], gates_rot, state.gates_rot),
            jnp.where(where, poles, state.poles),
            jnp.where(mask[:, None], seen_gates, state.seen_gates),
            jnp.where(mask[:, None], seen_poles, state.seen_poles),
        )

    def _in_range(
        self, pos: jax.Array, gates_pos: jax.Array, poles: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return which gates and poles are within sensor range, measured in x and y as the race."""
        limit = self.cfg["sensor_range"]
        gates = jnp.linalg.norm(pos[:, None, :2] - gates_pos[..., :2], axis=-1) < limit
        return gates, jnp.linalg.norm(pos[:, None, :2] - poles[..., :2], axis=-1) < limit

    def _sensed(self, state: RaceState) -> RaceTrack:
        """Return the track as the sensor shows it: nominal until seen, then true."""
        t = self.track
        return t._replace(
            gates_pos=jnp.where(state.seen_gates[..., None], state.gates_pos, t.gates_pos),
            gates_rot=jnp.where(state.seen_gates[..., None, None], state.gates_rot, t.gates_rot),
            poles=jnp.where(state.seen_poles[..., None], state.poles, t.poles),
        )

    def _funnel(self, guide_error: jax.Array) -> jax.Array:
        """Return the part of the guide's potential that fades with the distance from it [m]."""
        width = self.cfg["funnel_width"]
        return self.cfg["funnel"] * jnp.exp(-(guide_error**2) / (2.0 * width**2))

    def observe(self, state: RaceState) -> jax.Array:
        """Return the features of every world."""
        s = state.data.states
        return race_features(
            s.pos[:, 0],
            s.vel[:, 0],
            s.quat[:, 0],
            s.ang_vel[:, 0],
            state.prev,
            state.index,
            state.passed,
            self._sensed(state),
        )

    def step(self, state: RaceState, u: jax.Array) -> tuple:
        """Advance one control step.

        Returns:
            The new state, the reward, terminated (crash or finish), truncated and the statistics of
            this step.
        """
        cfg, t = self.cfg, self.track
        u = jnp.clip(u, -1.0, 1.0)
        action = to_action(u, cfg["yaw_max"])
        data = F.attitude_control(state.data, action[:, None])
        data = self.sim._step(data, n_steps=SUBSTEPS)
        pos, quat = data.states.pos[:, 0], data.states.quat[:, 0]

        target = jnp.minimum(state.passed, t.sequence.shape[0] - 1)
        gate = t.sequence[target]
        worlds = jnp.arange(self.n)
        gate_pos = state.gates_pos[worlds, gate]
        crossed = crossed_gate(
            state.last_pos, pos, gate_pos, state.gates_rot[worlds, gate], t.directions[target]
        )
        passed = state.passed + crossed
        # How much closer this step brought the drone to the gate it was heading for.
        closer = jnp.linalg.norm(state.last_pos - gate_pos, axis=1)
        closer = closer - jnp.linalg.norm(pos - gate_pos, axis=1)
        index, guide_error = advance_guide(pos, state.index, passed, t)
        # The change of a potential: distance along the guide, plus a bounded funnel around it. The
        # funnel is added and not multiplied, so that leaving the guide can only ever cost `funnel`
        # metres of credit. Multiplied, the loss would grow along the lap and act as a corridor.
        funnel = self._funnel(guide_error)
        progress = (index - state.index) * SPACING + funnel - state.funnel
        finished = passed >= t.sequence.shape[0]

        true = t._replace(gates_pos=state.gates_pos, gates_rot=state.gates_rot, poles=state.poles)
        poles, frames = clearances(body_center(pos, quat), true)
        # Counted as hits with room to spare: left to the exact limit the policy skims everything,
        # and a centimetre then decides between this check and the race's own.
        hit_pole = (poles <= cfg["hit_margin"]).any(axis=1)
        hit_gate = (frames <= cfg["hit_margin"]).any(axis=1)
        away = jnp.linalg.norm(pos[:, :2] - t.start[:2], axis=1) > PLATFORM_RADIUS
        out = (
            (pos > t.pos_high).any(axis=1)
            | (pos[:, :2] < t.pos_low[:2]).any(axis=1)
            | ((pos[:, 2] <= 0.0) & away)
        )
        speed = jnp.linalg.norm(data.states.vel[:, 0], axis=1)
        z_up = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
        tipped = z_up < jnp.cos(1.5)
        crashed = hit_pole | hit_gate | out | tipped | ~jnp.isfinite(pos).all(axis=1)

        change = jnp.sum((u - state.prev) ** 2, axis=1)
        nearby = jnp.exp(-cfg["obstacle_scale"] * poles).sum(axis=1)
        nearby = nearby + jnp.exp(-cfg["obstacle_scale"] * frames).sum(axis=1)
        reward = (
            cfg["closer"] * closer
            + cfg["progress"] * progress
            + cfg["gate"] * crossed
            + cfg["finish"] * finished
            - cfg["obstacle"] * nearby
            - cfg["smooth"] * change
            - cfg["tick"]
            - cfg["slow"] * jnp.maximum(1.0 - speed / cfg["slow_speed"], 0.0)
        )
        reward = jnp.nan_to_num(jnp.where(crashed, reward - cfg["crash"], reward))

        ticks = state.ticks + 1
        terminated = crashed | finished
        truncated = (ticks >= cfg["episode_ticks"]) & ~terminated
        ended = terminated | truncated
        stats = RaceStats(
            episodes=ended.sum(),
            ticks=(ticks * ended).sum(),
            hit_pole=hit_pole.sum(),
            hit_gate=(hit_gate & ~hit_pole).sum(),
            out=(out & ~hit_pole & ~hit_gate).sum(),
            tipped=(tipped & ~hit_pole & ~hit_gate & ~out).sum(),
            finished=(finished & ~crashed).sum(),
            gates=crossed.sum(),
            commanded=action[:, 3].mean() / MASS,
            reward=reward.mean(),
            gap=jnp.minimum(poles.min(axis=1), frames.min(axis=1)).mean(),
            guide=guide_error.mean(),
            samples=jnp.ones(()),
        )
        gates_now, poles_now = self._in_range(pos, state.gates_pos, state.poles)
        new = state._replace(
            data=data,
            index=index,
            passed=passed,
            prev=u,
            last_pos=pos,
            ticks=ticks,
            funnel=funnel,
            seen_gates=state.seen_gates | gates_now,
            seen_poles=state.seen_poles | poles_now,
        )
        return new, reward, terminated, truncated, stats


def make_lap_check(env: RaceEnv, ticks: int) -> Any:
    """Return a compiled lap from the ground at rest in every world, with no action noise."""

    def lap(params: dict, norm: Norm, key: jax.Array) -> dict:
        state = env.initial(key, ground=1.0, still=True)

        def tick(carry: tuple, step: jax.Array) -> tuple:
            state, gone, finish_time, gates = carry
            x = normalize(env.observe(state), norm)
            state, _, term, trunc, _ = env.step(state, forward(params["actor"], x))
            done = state.passed >= env.track.sequence.shape[0]
            finish_time = jnp.where(done & ~gone, (step + 1) / 50.0, finish_time)
            gates = jnp.where(gone, gates, state.passed)
            gone = gone | term | trunc
            return (state, gone, finish_time, gates), None

        zeros = jnp.zeros(env.n)
        start = (state, jnp.zeros(env.n, dtype=bool), zeros, zeros.astype(jnp.int32))
        (_, _, finish_time, gates), _ = jax.lax.scan(tick, start, jnp.arange(ticks))
        finished = finish_time > 0
        return {
            "finish": finished.mean(),
            "time": jnp.sum(finish_time) / jnp.maximum(finished.sum(), 1),
            "best": jnp.where(finished, finish_time, 1e3).min(),
            "gates": gates.mean(),
        }

    return jax.jit(lap)


def main(
    config: str = "level0.toml",
    out: str = "race_policy.npz",
    iterations: int = 4000,
    worlds: int = 8192,
    horizon: int = 32,
    epochs: int = 5,
    minibatches: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    ent: float = 0.0,
    seed: int = 0,
    eval_every: int = 200,
    closer: float = 10.0,
    progress: float = 0.0,
    funnel: float = 0.0,
    funnel_width: float = 0.5,
    gate: float = 20.0,
    finish: float = 50.0,
    crash: float = 20.0,
    hit_margin: float = 0.03,
    obstacle: float = 0.02,
    obstacle_scale: float = 1.0,
    smooth: float = 0.02,
    tick: float = 0.05,
    slow: float = 0.1,
    slow_speed: float = 2.0,
    ground: float = 0.5,
    randomize: float = 1.0,
    start_vel: float = 0.5,
    start_noise: float = 0.05,
    episode_ticks: int = 500,
    log_std_min: float = -2.3,
    hidden: tuple[int, ...] = (64, 64),
    critic_hidden: tuple[int, ...] = (256, 256, 256),
    yaw_max: float = 0.0,
    force_range: float = 0.1,
    device: str = "gpu",
):
    """Train on one track and keep the policy that finishes most often, then fastest.

    Args:
        config: Race configuration in ``config/``.
        out: Where to write the best policy.
        iterations: PPO iterations.
        worlds: Simulated drones.
        horizon: Control steps per iteration.
        epochs: Passes over each rollout.
        minibatches: Minibatches per pass.
        lr: Learning rate.
        gamma: Discount.
        lam: GAE parameter.
        clip: PPO clip.
        ent: Entropy bonus.
        seed: Random seed.
        eval_every: Iterations between lap checks.
        closer: Reward per metre of getting closer to the next gate.
        progress: Reward per metre of the guide's potential; at zero the policy only sees the guide.
        funnel: Metres of potential that fade with the distance from the guide.
        funnel_width: Standard deviation of that fade [m].
        gate: Reward for passing a gate.
        finish: Reward for finishing the lap.
        crash: Penalty for a crash.
        hit_margin: Clearance below which something counts as hit [m].
        obstacle: Weight of the penalty ``sum(exp(-obstacle_scale * d))`` over poles and gates.
        obstacle_scale: How quickly that penalty fades with distance [1/m].
        smooth: Penalty on the change of the action.
        tick: Cost per control step, which puts a price on time directly.
        slow: Cost per control step of standing still, fading to nothing at `slow_speed`.
        slow_speed: Speed from which flying costs nothing extra [m/s].
        ground: Share of episodes that start on the ground rather than along the guide.
        randomize: Size of the changes to gates, poles and the drone's start, as a share of what
            level 2 does; zero flies the nominal track from the nominal start.
        start_vel: Largest random velocity per axis at the start of an episode [m/s]. Added to the
            plan's velocity along the guide. The lap checks start at rest, as the race does.
        start_noise: Standard deviation of the position of a start along the guide [m].
        episode_ticks: Control steps per episode.
        log_std_min: Floor on the log of the action noise.
        hidden: Hidden layers of the policy.
        critic_hidden: Hidden layers of the critic.
        yaw_max: Limit on the yaw command [rad].
        force_range: Largest random force per axis at each simulator step [N]; the race's default.
        device: Device to train on.
    """
    race = load_config(Path(__file__).parents[1] / "config" / config)
    track = build_track(race.env.track)
    print(
        f"guide {len(track.beads)} beads, crossings {track.crossings.tolist()}, "
        f"top speed {track.speeds.max():.1f} m/s",
        flush=True,
    )
    cfg = dict(
        closer=closer,
        progress=progress,
        funnel=funnel,
        funnel_width=funnel_width,
        gate=gate,
        finish=finish,
        crash=crash,
        hit_margin=hit_margin,
        obstacle=obstacle,
        obstacle_scale=obstacle_scale,
        smooth=smooth,
        tick=tick,
        slow=slow,
        slow_speed=slow_speed,
        ground=ground,
        randomize=randomize,
        start_vel=start_vel,
        sensor_range=float(race.env.sensor_range),
        start_noise=start_noise,
        episode_ticks=episode_ticks,
        yaw_max=yaw_max,
        force_range=force_range,
    )
    t0 = time.time()
    best = -1e9
    log = []
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
        env = RaceEnv(sim, track, cfg)
        key = jax.random.PRNGKey(seed)
        key, init, start = jax.random.split(key, 3)
        params = init_params(init, tuple(hidden), tuple(critic_hidden), OBS_DIM)
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        opt = tx.init(params)
        norm = Norm(jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), jnp.array(1e-4))
        state = env.initial(start)
        stats = zero_stats()
        train_step = make_train_step(
            env, tx, horizon, epochs, minibatches, gamma, lam, clip, ent, log_std_min
        )
        lap_check = make_lap_check(env, 1000)
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
            key, sub = jax.random.split(key)
            lap = {k: float(v) for k, v in jax.device_get(lap_check(params, norm, sub)).items()}
            # Finishing comes first, then the time: 1% more finishes outweighs 0.1 s.
            score = 10.0 * lap["finish"] - (lap["time"] if lap["finish"] > 0 else 30.0)
            score = score + lap["gates"]
            kept = score > best
            tmp = out + ".tmp.npz"
            save(params, norm, tmp, yaw_max)
            if kept:
                best = score
                save(params, norm, out, yaw_max)
            std = float(jnp.exp(jnp.maximum(params["log_std"], log_std_min)).mean())
            row = dict(
                it=it,
                steps=(it + 1) * worlds * horizon,
                secs=round(time.time() - t0),
                rate=rate,
                std=std,
                score=score,
                lap=lap,
                episodes=ep,
            )
            log.append(row)
            Path(out + ".log.json").write_text(json.dumps(log))
            print(
                f"it {it:5} {row['secs']:5}s {rate / 1e6:4.1f}M/s rew {ep['reward']:+7.3f}"
                f" ep {ep['length']:5.1f} gates {ep['gates']:4.2f}"
                f" pole {ep['hit_pole'] * 100:3.0f} gate {ep['hit_gate'] * 100:3.0f}"
                f" out {ep['out'] * 100:3.0f}"
                f" tip {ep['tipped'] * 100:3.0f} fin {ep['finished'] * 100:3.0f}%"
                f" gap {ep['gap']:.2f} guide {ep['guide']:.2f} cmd {ep['commanded']:4.1f}"
                f" std {std:.2f} | LAP fin {lap['finish'] * 100:5.1f}% gates {lap['gates']:.2f}"
                f" t {lap['time']:.2f} best {lap['best']:.2f}{' *' if kept else ''}",
                flush=True,
            )
            timer, done_since = time.time(), 0


if __name__ == "__main__":
    fire.Fire(main)
