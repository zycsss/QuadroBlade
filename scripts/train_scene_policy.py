"""Train a racing policy with PPO on random local scenes, never on the track it will be tested on.

Every world flies a generated track from `scripts/solve_scene_library.py`, with the race's gates,
poles, sensor reveal, disturbances, mass and inertia changes, and the rewards of
`scripts/train_race_policy.py`. An episode starts on the floor or anywhere along the guide, and the
last pass really ends it, so the critic is never read on a state it does not train on.

The policy that is kept is the one that does best on validation tracks it did not train on. Laps of
level 2 and level 3 are logged as well, but they choose nothing, so that the final test on level 2
says how well the policy carries over.

Example::

    $ pixi run -e gpu python scripts/train_scene_policy.py --train scenes_train.npz \
        --validation scenes_validation.npz --out local_scenes_no_level2.npz
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
from train_race_policy import (  # noqa: E402
    DRONE_RISE,
    DRONE_SHIFT,
    DRONE_TILT,
    GATE_SHIFT,
    GATE_TURN,
    PLATFORM_RADIUS,
    POLE_SHIFT,
    SIM_FREQ,
    SUBSTEPS,
    RaceStats,
    summarize,
    zero_stats,
)

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402
from lsy_drone_racing.utils.mpcc_bench import MASS  # noqa: E402
from lsy_drone_racing.utils.path_policy import ACT_DIM, SPACING  # noqa: E402
from lsy_drone_racing.utils.race_policy import (  # noqa: E402
    body_center,
    build_track,
    crossed_gate,
    euler_matrix,
    euler_quat,
)
from lsy_drone_racing.utils.scene_policy import (  # noqa: E402
    OBS_DIM,
    GuideLibrary,
    advance_scene_guide,
    load_library,
    scene_action,
    scene_clearances,
    scene_features,
    scene_heading,
    track_library,
)

#: How many places along the guide are tried for a start that touches nothing.
START_TRIES = 3


class SceneState(NamedTuple):
    """Every world's simulator data, its track and its progress along it."""

    data: Any
    rows: jax.Array
    index: jax.Array
    passed: jax.Array
    prev: jax.Array
    last_pos: jax.Array
    ticks: jax.Array
    gates_pos: jax.Array
    gates_rot: jax.Array
    poles: jax.Array
    seen_gates: jax.Array
    seen_poles: jax.Array


class SceneEnv:
    """Many drones in one simulator, each on a track of a library, as pure functions."""

    def __init__(self, sim: Sim, library: GuideLibrary, cfg: dict):
        """Keep the simulator's functions and put the library on its device."""
        self.sim = sim
        self.n = sim.n_worlds
        self.cfg = cfg
        self.lib = GuideLibrary(
            **{
                k: jnp.asarray(v, jnp.int32 if k in ("crossings", "sequence") else jnp.float32)
                for k, v in library._asdict().items()
            }
        )
        self.nominal_rot = euler_matrix(self.lib.gates_rpy)

    def initial(
        self, key: jax.Array, ground: float | None = None, still: bool = False
    ) -> SceneState:
        """Return all worlds at the start of an episode."""
        lib, n = self.lib, self.n
        zeros = jnp.zeros(n, dtype=jnp.int32)
        p = self.sim.data.params
        params = p.replace(
            mass=jnp.broadcast_to(p.mass, (n, 1, 1)),
            J=jnp.broadcast_to(p.J, (n, 1, 3, 3)),
            J_inv=jnp.broadcast_to(p.J_inv, (n, 1, 3, 3)),
        )
        gates, poles = lib.gates_pos.shape[1], lib.poles.shape[1]
        state = SceneState(
            self.sim.data.replace(params=params),
            zeros,
            zeros,
            zeros,
            jnp.zeros((n, ACT_DIM)),
            jnp.zeros((n, 3)),
            zeros,
            jnp.zeros((n, gates, 3)),
            jnp.zeros((n, gates, 3, 3)),
            jnp.zeros((n, poles, 3)),
            jnp.zeros((n, gates), dtype=bool),
            jnp.zeros((n, poles), dtype=bool),
        )
        return self.reset(state, jnp.ones(n, dtype=bool), key, ground, still)

    def _start_along_guide(
        self, key: jax.Array, rows: jax.Array, gate_shift: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return a place on the guide, moved with the gates around it, and the passes before it."""
        lib, n = self.lib, self.n
        last_pass = lib.sequence.shape[0] - 1
        k_index, k_noise = jax.random.split(key)
        index = (jax.random.uniform(k_index, (n,)) * lib.crossings[rows, last_pass]).astype(
            jnp.int32
        )
        passed = (lib.crossings[rows] <= index[:, None]).sum(axis=1)
        ahead = jnp.minimum(passed, last_pass)
        behind = jnp.maximum(passed - 1, 0)
        low = lib.crossings[rows, behind]
        span = jnp.maximum(lib.crossings[rows, ahead] - low, 1)
        share = jnp.where(passed > 0, (index - low) / span, 1.0)[:, None]
        worlds = jnp.arange(n)
        bend = share * gate_shift[worlds, lib.sequence[ahead]]
        bend = bend + (1.0 - share) * jnp.where(
            passed[:, None] > 0, gate_shift[worlds, lib.sequence[behind]], 0.0
        )
        noise = self.cfg["start_noise"] * jax.random.normal(k_noise, (n, 3))
        return lib.beads[rows, index] + bend + noise, index, passed

    def reset(
        self,
        state: SceneState,
        mask: jax.Array,
        key: jax.Array,
        ground: float | None = None,
        still: bool = False,
    ) -> SceneState:
        """Start the masked worlds on a new track, on the ground or along the guide.

        Args:
            state: The current state.
            mask: The worlds to start again.
            key: Random key.
            ground: Share of starts on the ground; the configured share if None.
            still: Start at rest, as the race does, instead of with a random velocity.
        """
        lib, cfg, n = self.lib, self.cfg, self.n
        ground = cfg["ground"] if ground is None else ground
        keys = jax.random.split(key, 14 + START_TRIES)
        data = self.sim._reset(state.data, self.sim.default_data, mask)
        rows = jax.random.randint(keys[0], (n,), 0, lib.beads.shape[0])

        def spread(k: jax.Array, limit: tuple, shape: tuple) -> jax.Array:
            limit = cfg["randomize"] * jnp.asarray(limit)
            return jax.random.uniform(k, (*shape, len(limit)), minval=-limit, maxval=limit)

        gates, count = lib.gates_pos.shape[1], lib.poles.shape[1]
        gate_shift = spread(keys[1], GATE_SHIFT, (n, gates))
        gates_pos = lib.gates_pos[rows] + gate_shift
        gates_rot = euler_matrix(lib.gates_rpy[rows] + spread(keys[2], GATE_TURN, (n, gates)))
        poles = lib.poles[rows] + spread(keys[3], POLE_SHIFT, (n, count))

        # A start along the guide must touch nothing, or the episode is over before it begins. One
        # that keeps touching after every try starts on the ground.
        along, index, passed = self._start_along_guide(keys[14], rows, gate_shift)
        quat_along = lib.quats[rows, index]
        free = self._free(along, quat_along, gates_pos, gates_rot, poles)
        for i in range(1, START_TRIES):
            a, x, p = self._start_along_guide(keys[14 + i], rows, gate_shift)
            q = lib.quats[rows, x]
            better = ~free & self._free(a, q, gates_pos, gates_rot, poles)
            along = jnp.where(better[:, None], a, along)
            index = jnp.where(better, x, index)
            passed = jnp.where(better, p, passed)
            quat_along = jnp.where(better[:, None], q, quat_along)
            free = free | better

        on_ground = (jax.random.uniform(keys[4], (n,)) < ground) | ~free
        index = jnp.where(on_ground, 0, index)
        passed = jnp.where(on_ground, 0, passed)

        shift = spread(keys[5], DRONE_SHIFT, (n,))
        rise = DRONE_RISE * jax.random.uniform(keys[6], (n, 1))
        pos = jnp.where(on_ground[:, None], lib.starts[rows] + jnp.c_[shift, rise], along)

        # Standing on the floor it can only leave upwards.
        limit = cfg["start_vel"]
        velocity = jax.random.uniform(keys[7], (n, 3), minval=-limit, maxval=limit)
        takeoff = velocity * jnp.array([1.0, 1.0, 0.0]) + jnp.array([0.0, 0.0, 1.0]) * jnp.abs(
            velocity
        )
        scale = jax.random.uniform(keys[8], (n,), minval=0.5, maxval=1.0)
        moving = lib.velocities[rows, index] * scale[:, None] + velocity
        vel = jnp.where(on_ground[:, None], takeoff, moving)
        vel = jnp.where(jnp.asarray(still) & on_ground[:, None], 0.0, vel)

        tilt = spread(keys[9], (DRONE_TILT, DRONE_TILT, DRONE_TILT), (n,))
        quat = jnp.where(on_ground[:, None], euler_quat(tilt), quat_along)
        rotor = jnp.where(on_ground, 0.0, HOVER_RPM)
        where = mask[:, None, None]
        s = data.states
        states = s.replace(
            pos=jnp.where(where, pos[:, None], s.pos),
            vel=jnp.where(where, vel[:, None], s.vel),
            quat=jnp.where(where, quat[:, None], s.quat),
            rotor_vel=jnp.where(where, rotor[:, None, None], s.rotor_vel),
        )
        # Mass and inertia as the race changes them: an offset drawn when the episode starts.
        default = self.sim.default_data.params
        dm, dj = cfg["mass_range"], cfg["inertia_range"]
        mass = default.mass + jax.random.uniform(keys[10], (n, 1, 1), minval=-dm, maxval=dm)
        inertia = default.J + jax.random.uniform(keys[11], (n, 1, 3, 3), minval=-dj, maxval=dj)
        params = data.params.replace(
            mass=jnp.where(where, mass, data.params.mass),
            J=jnp.where(where[..., None], inertia, data.params.J),
            J_inv=jnp.where(where[..., None], jnp.linalg.inv(inertia), data.params.J_inv),
        )
        seen_gates, seen_poles = self._in_range(pos, gates_pos, poles)
        return SceneState(
            data.replace(states=states, params=params),
            jnp.where(mask, rows, state.rows),
            jnp.where(mask, index, state.index),
            jnp.where(mask, passed, state.passed),
            jnp.where(mask[:, None], 0.0, state.prev),
            jnp.where(mask[:, None], pos, state.last_pos),
            jnp.where(mask, 0, state.ticks),
            jnp.where(where, gates_pos, state.gates_pos),
            jnp.where(mask[:, None, None, None], gates_rot, state.gates_rot),
            jnp.where(where, poles, state.poles),
            jnp.where(mask[:, None], seen_gates, state.seen_gates),
            jnp.where(mask[:, None], seen_poles, state.seen_poles),
        )

    def _free(
        self,
        pos: jax.Array,
        quat: jax.Array,
        gates_pos: jax.Array,
        gates_rot: jax.Array,
        poles: jax.Array,
    ) -> jax.Array:
        """Return whether drones at these places touch nothing, by the rule that ends an episode."""
        pole_gaps, frames = scene_clearances(
            body_center(pos, quat), gates_pos, gates_rot, poles, self.lib
        )
        margin = self.cfg["hit_margin"]
        return (pole_gaps > margin).all(axis=1) & (frames > margin).all(axis=1)

    def _in_range(
        self, pos: jax.Array, gates_pos: jax.Array, poles: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return which gates and poles are within sensor range, measured in x and y as the race."""
        limit = self.cfg["sensor_range"]
        gates = jnp.linalg.norm(pos[:, None, :2] - gates_pos[..., :2], axis=-1) < limit
        return gates, jnp.linalg.norm(pos[:, None, :2] - poles[..., :2], axis=-1) < limit

    def observe(self, state: SceneState) -> jax.Array:
        """Return the features of every world, with gates and poles as the sensor shows them."""
        lib = self.lib
        s = state.data.states
        rows = state.rows
        gates_pos = jnp.where(state.seen_gates[..., None], state.gates_pos, lib.gates_pos[rows])
        gates_rot = jnp.where(
            state.seen_gates[..., None, None], state.gates_rot, self.nominal_rot[rows]
        )
        poles = jnp.where(state.seen_poles[..., None], state.poles, lib.poles[rows])
        return scene_features(
            s.pos[:, 0],
            s.vel[:, 0],
            s.quat[:, 0],
            s.ang_vel[:, 0],
            state.prev,
            state.index,
            state.passed,
            rows,
            gates_pos,
            gates_rot,
            poles,
            lib,
        )

    def step(self, state: SceneState, u: jax.Array) -> tuple:
        """Advance one control step.

        Returns:
            The new state, the reward, terminated (crash or finish), truncated and the statistics of
            this step.
        """
        cfg, lib = self.cfg, self.lib
        u = jnp.clip(u, -1.0, 1.0)
        c, s = scene_heading(state.index, state.rows, lib)
        action = scene_action(u, c, s)
        data = F.attitude_control(state.data, action[:, None])
        data = self.sim._step(data, n_steps=SUBSTEPS)
        pos, quat = data.states.pos[:, 0], data.states.quat[:, 0]

        target = jnp.minimum(state.passed, lib.sequence.shape[0] - 1)
        gate = lib.sequence[target]
        worlds = jnp.arange(self.n)
        gate_pos = state.gates_pos[worlds, gate]
        crossed = crossed_gate(
            state.last_pos, pos, gate_pos, state.gates_rot[worlds, gate], lib.directions[target]
        )
        passed = state.passed + crossed
        closer = jnp.linalg.norm(state.last_pos - gate_pos, axis=1)
        closer = closer - jnp.linalg.norm(pos - gate_pos, axis=1)
        index, guide_error = advance_scene_guide(pos, state.index, passed, state.rows, lib)
        progress = (index - state.index) * SPACING
        finished = passed >= lib.sequence.shape[0]

        poles, frames = scene_clearances(
            body_center(pos, quat), state.gates_pos, state.gates_rot, state.poles, lib
        )
        hit_pole = (poles <= cfg["hit_margin"]).any(axis=1)
        hit_gate = (frames <= cfg["hit_margin"]).any(axis=1)
        start = lib.starts[state.rows]
        away = jnp.linalg.norm(pos[:, :2] - start[:, :2], axis=1) > PLATFORM_RADIUS
        out = (
            (pos > lib.pos_high).any(axis=1)
            | (pos[:, :2] < lib.pos_low[:2]).any(axis=1)
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
            seen_gates=state.seen_gates | gates_now,
            seen_poles=state.seen_poles | poles_now,
        )
        return new, reward, terminated, truncated, stats


def make_lap_check(env: SceneEnv, ticks: int) -> Any:
    """Return a compiled lap from the ground at rest in every world, with no action noise."""

    def lap(params: dict, norm: Norm, key: jax.Array) -> dict:
        state = env.initial(key, ground=1.0, still=True)

        def tick(carry: tuple, step: jax.Array) -> tuple:
            state, gone, finish_time, gates = carry
            x = normalize(env.observe(state), norm)
            state, _, term, trunc, _ = env.step(state, forward(params["actor"], x))
            done = state.passed >= env.lib.sequence.shape[0]
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
            "gates": gates.mean(),
        }

    return jax.jit(lap)


def main(
    train: str,
    validation: str,
    level3: str | None = None,
    out: str = "scene_policy.npz",
    iterations: int = 6000,
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
    ground: float = 0.25,
    randomize: float = 1.0,
    start_vel: float = 0.5,
    start_noise: float = 0.05,
    episode_ticks: int = 500,
    log_std_min: float = -2.3,
    hidden: tuple[int, ...] = (64, 64),
    critic_hidden: tuple[int, ...] = (256, 256, 256),
    mass_range: float = 0.005,
    inertia_range: float = 1e-6,
    force_range: float = 0.1,
    device: str = "gpu",
):
    """Train on generated tracks and keep the policy that does best on validation tracks.

    Args:
        train: Library of training tracks, or ``level2`` to train on that track alone.
        validation: Library of validation tracks, which choose the policy that is kept.
        level3: Library of level 3 tracks, flown and logged only.
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
        progress: Reward per metre along the guide, which leads around what stands in the way.
        gate: Reward for passing a gate.
        finish: Reward for the last pass.
        crash: Penalty for a crash.
        hit_margin: Clearance below which something counts as hit [m].
        obstacle: Weight of the penalty ``sum(exp(-obstacle_scale * d))`` over poles and gates.
        obstacle_scale: How quickly that penalty fades with distance [1/m].
        smooth: Penalty on the change of the action.
        tick: Cost per control step.
        slow: Cost per control step of standing still, fading to nothing at `slow_speed`.
        slow_speed: Speed from which flying costs nothing extra [m/s].
        ground: Share of episodes that start on the ground rather than along the guide.
        randomize: Size of the shifts to gates, poles and the start, as a share of level 2's.
        start_vel: Largest random velocity per axis at the start of an episode [m/s].
        start_noise: Standard deviation of the position of a start along the guide [m].
        episode_ticks: Control steps per episode.
        log_std_min: Floor on the log of the action noise.
        hidden: Hidden layers of the policy.
        critic_hidden: Hidden layers of the critic.
        mass_range: Largest mass offset of an episode [kg]; the default is level 2's.
        inertia_range: Largest offset of each inertia entry [kg m^2]; the default is level 2's.
        force_range: Largest random force per axis at each simulator step [N]; the race's default.
        device: Device to train on.
    """
    race = load_config(Path(__file__).parents[1] / "config" / "level2.toml")
    # Only the order of the gates, their boxes and the limits of level 2 go into training.
    reference = build_track(race.env.track)
    rpy = Rotation.from_matrix(reference.gates_rot).as_euler("xyz")
    # Training on level 2 itself is only a check of the inputs and for fine-tuning: it says nothing
    # about other tracks.
    level2 = track_library(reference, rpy)
    libraries = {
        "train": level2 if train == "level2" else load_library(train, reference),
        "validation": load_library(validation, reference),
        "level2": level2,
    }
    if level3:
        libraries["level3"] = load_library(level3, reference)
    for name, lib in libraries.items():
        print(f"{name}: {lib.beads.shape[0]} tracks, {lib.beads.shape[1]} beads", flush=True)
    cfg = dict(
        closer=closer,
        progress=progress,
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
        mass_range=mass_range,
        inertia_range=inertia_range,
    )
    t0 = time.time()
    best = -1e9
    log = []
    yaw_max = 0.0
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
        envs = {name: SceneEnv(sim, lib, cfg) for name, lib in libraries.items()}
        env = envs["train"]
        checks = {n: make_lap_check(e, 1000) for n, e in envs.items() if n != "train"}
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
            laps = {}
            for name, check in checks.items():
                key, sub = jax.random.split(key)
                laps[name] = {
                    k: float(v) for k, v in jax.device_get(check(params, norm, sub)).items()
                }
            val = laps["validation"]
            # Finishing comes first, then the time: 1% more finishes outweighs 0.1 s.
            score = (
                10.0 * val["finish"] - (val["time"] if val["finish"] > 0 else 30.0) + val["gates"]
            )
            kept = score > best
            save(params, norm, out + ".last.npz", yaw_max)
            if kept:
                best = score
                save(params, norm, out, yaw_max)
            std = float(jnp.exp(jnp.maximum(params["log_std"], log_std_min)).mean())
            log.append(
                dict(
                    it=it,
                    secs=round(time.time() - t0),
                    rate=rate,
                    std=std,
                    score=score,
                    laps=laps,
                    episodes=ep,
                )
            )
            Path(out + ".log.json").write_text(json.dumps(log))
            text = " ".join(
                f"{n} {v['finish'] * 100:5.1f}% {v['gates']:.2f}g {v['time']:.2f}s"
                for n, v in laps.items()
            )
            print(
                f"it {it:5} {log[-1]['secs']:5}s {rate / 1e6:4.1f}M/s rew {ep['reward']:+7.3f}"
                f" ep {ep['length']:5.1f} pole {ep['hit_pole'] * 100:3.0f} gate"
                f" {ep['hit_gate'] * 100:3.0f} out {ep['out'] * 100:3.0f}"
                f" tip {ep['tipped'] * 100:3.0f} fin {ep['finished'] * 100:3.0f}%"
                f" std {std:.2f} | {text}{' *' if kept else ''}",
                flush=True,
            )
            timer, done_since = time.time(), 0


if __name__ == "__main__":
    fire.Fire(main)
