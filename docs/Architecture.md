# Architecture

How this repository is built today. It describes what exists, not what is planned: plans and
experiment history live in [PLAN_path_rl.md](PLAN_path_rl.md), and how training scenarios are built
is in [training_scenarios.md](training_scenarios.md). Keep this file up to date whenever the
architecture changes.

## At a glance

```
                     config/level*.toml
                     (track, rules, randomization)
                               |
                               v
                 DroneRaceEnv (envs/, on crazyflow)
                 50 Hz: observation -> action -> physics at 500 Hz
                          ^                   |
       [roll, pitch, yaw, thrust]             | pos, vel, quat, ang_vel,
                          |                   | gates, poles, n_gates_passed
                          |                   v
       +------------------+-------------------+------------------+
       |                                                         |
  ToptMPCC (control/topt_mpcc.py)              RLRace (control/rl_race.py)
  plan -> bend -> MPCC                         guide + policy network
  for tracks with reveals (level 2)            levels 0 and 2
       |                                                         |
       +----------------> topt_planner <-------------------------+
                          time-optimal plan, cached on disk
```

**`RLScene` (`control/rl_scene.py`) is the final controller**: every `config/level*.toml` selects
it, so `pixi run python scripts/sim.py` flies it. Three controllers fly the race today, all using
the same planner: the MPCC follows its plan, and the
two RL controllers only see it as a hint. `RLRace` is trained on the track it flies. `RLScene`
(`control/rl_scene.py`, described below) is trained on generated tracks only.

| Controller | File | Tested on | Result in the real environment |
|---|---|---|---|
| Time-optimal MPCC | `control/topt_mpcc.py` | Level 2 (its default level) | 54/64 finished at 4.40 s |
| Time-optimal MPCC | `control/topt_mpcc.py` | Level 0 | 8/8 finished at 3.73 s |
| RL racing policy, fixed track | `control/rl_race.py` | Level 0 | 64/64 finished at 3.33 s |
| RL racing policy, randomized | `control/rl_race.py` | Level 2 | 64/64 finished at 4.66 s |
| RL scene policy, never saw the track | `control/rl_scene.py` | Level 0 | 64/64 finished at 4.03 s |
| RL scene policy, never saw the track | `control/rl_scene.py` | Level 2 | 0/64 finished |
| RL scene policy, never saw the track | `control/rl_scene.py` | Random tracks from the race's generator, level 1's conditions, no shifts | 243/256 finished at about 4.67 s |
| RL scene policy, trained and flown on fast guides, not the default | `control/rl_scene.py`, through `scripts/race_fast_guides.py` | Same | 248/256 finished at about 4.92 s |
| RL scene policy, trained and flown on solved guides with faster limits, not the default | `control/rl_scene.py` with `controller.fast_guides = true` | Same | 249/256 finished at about 4.46 s |

## The race environment

`lsy_drone_racing/envs/`, built on the `crazyflow` simulator.

- **Levels:** `config/level0.toml` to `level3.toml`. Level 0 has noise only. Level 1 adds mass and
  inertia changes. Level 2 also shifts gates and poles, revealing their true poses within 0.7 m.
  Level 3 generates a new layout each episode.
- **Timing:** the controller runs at 50 Hz and the physics at 500 Hz.
- **Actions:** `[roll, pitch, yaw, thrust]` in attitude mode, the angles being
  `Rotation.from_euler("xyz", ...)`, followed by the onboard Mellinger attitude law.
- **A lap:** the signed `gate_order`, `[1, 2, 3, 4, 2]` on the current levels, so one gate is flown
  twice.
- **A gate pass:** crossing the gate's plane in the right direction, inside a 0.45 m square.
- **A crash:** MuJoCo contact between the drone and a gate or pole, or leaving the safety limits. The
  drone collides as a sphere of radius 0.086 m, 0.02 m above its body origin. The floor does not
  count as a contact.
- **Disturbances:** a normal action noise of scale 0.001, and a new random force of up to 0.1 N per
  axis at every simulator step.
- **Track generator:** `envs/randomize.py`, `build_random_track_fn`, is written in JAX and runs per
  world.

## The time-optimal MPCC

| Stage | File | What it does |
|---|---|---|
| Plan | `control/topt_planner.py` | Fastest point-mass trajectory through the nominal track, solved with IPOPT and cached in `lsy_topt_cache/`. Random tracks use `RANDOM_TRACK_CONFIG`, which keeps clear of every object and limits node spacing to 25 cm, because the defaults could step over poles between nodes. `FAST_GUIDE_CONFIG` is the same with tilt 1.3 rad and thrust 16 m/s^2. Fixed levels keep the defaults. |
| Branches | `control/topt_branches.py` | One shared approach to an unseen gate, with a branch for each place the gate might be |
| Bend | `control/path_bend.py` | Bends the plan to the poses the sensor reveals, without re-planning |
| Fly | `control/topt_mpcc.py` | Contouring MPC (acados, `SQP_RTI`) maximizing progress inside a clearance tunnel, with a disturbance observer |
| Geometry | `control/track_geometry.py` | Sizes of gates, poles and the drone |

Solvers are generated into `c_generated_code/` and acados lives in `acados/`, both git-ignored and
built for the local platform.

## The RL racing controller

The current RL architecture. It picks its policy from the config: if gates or poles move,
`rl_race_randomized.npz` (trained on moved tracks), otherwise `rl_race_level0.npz` (trained on the
nominal track, faster there, and 0/64 on level 2).

```
state, guide, gates, poles --> 79 inputs --> 64 --> 64 --> 7 outputs --> to_action --> attitude and thrust
```

| Part | File |
|---|---|
| Controller used in the race | `control/rl_race.py` |
| Trained policies | `control/rl_race_randomized.npz`, `control/rl_race_level0.npz` |
| Track, rules and features | `utils/race_policy.py` |
| Attitude output, network in NumPy | `utils/path_policy.py` |
| Training | `scripts/train_race_policy.py` |
| PPO, shared by both trainers | `scripts/train_path_policy.py` |

**Inputs (79):**

| Inputs | Count |
|---|---|
| Velocity, angular velocity | 6 |
| Rotation matrix | 9 |
| Previous output | 7 |
| Offset from the guide | 3 |
| Guide ahead, 8 points from 0.2 to 3 m | 24 |
| Plan's speed ahead, 4 points | 4 |
| Next two gates: position relative to the drone and passing direction | 12 |
| Nearest two poles: relative x, y and clearance | 6 |
| Nearest gate: relative position and clearance | 4 |
| Position | 3 |
| Share of the lap done | 1 |

**Outputs (7):** the first two columns of the wanted rotation matrix, as an offset from level, and
the thrust. `to_action` cleans the columns up with Gram-Schmidt and converts them to angles. Roll
and pitch are limited to 1.4 rad. Yaw is limited by a setting stored in the policy file, and the
current policy always commands 0.

**Network:** the policy is 2 x 64 with ELU. The critic is 3 x 256 and exists only during training.

**The guide is only a hint.** It comes from `topt_planner`, but no reward depends on following it
and no episode ends for leaving it. It does set the input features and where mid-lap training
episodes start. The policy flew up to 38 cm off it.

**Reward (default settings):**

| Term | Value |
|---|---|
| Getting closer to the next gate | +10 per metre |
| Passing a gate | +20 |
| Finishing the lap | +50 |
| Time | -0.05 per step |
| Flying slowly | Up to -0.1 per step, 0 from 2 m/s |
| Near a pole or gate | -0.02 x sum of `exp(-d)` over all of them |
| Change of output | -0.02 x squared change |
| Crash | -20 |

Two rules hold this together:

- **Hovering must cost something.** Without the time and speed costs, the policy hovers far from
  everything. A per-step cost adds up to about 100 times itself at `gamma = 0.99`, and that total
  must stay below the crash penalty, or crashing becomes the cheap way out.
- **Hits are counted with a 3 cm margin.** Without it the policy skims gates by 1-3 cm and hits them
  in the real race.

A guide-based reward is available: `--progress` pays for distance along the guide, and `--funnel`
adds a bounded credit for staying near it. It is reliable but about 0.9 s slower, so it is off by
default.

**Training episodes:** half start on the ground, as the race does. The rest start at a random place
along the guide with the plan's velocity and attitude and the rotors at hover speed.

**Randomization (`--randomize`, default 1, as level 2 moves things; 0 flies the nominal track):**

| What | How much, per episode |
|---|---|
| Gate position | +-0.15 m in x and y, +-0.1 m in z |
| Gate roll, pitch, yaw | +-0.05, +-0.1, +-0.2 rad |
| Pole position | +-0.15 m in x and y, +-0.05 m in z |
| Drone start, on the ground | +-0.1 m in x and y, up to 0.02 m higher, attitude +-0.1 rad |
| Drone start, along the guide | Moves with the gates around it, plus noise of 0.05 m |
| Start velocity (`--start_vel`) | Up to +-0.5 m/s per axis (upwards only from the floor), added to the plan's along the guide |
| Random force | +-0.1 N per axis at every simulator step |

Mass and inertia are not randomized here. Lap checks start at rest, as the race does.

**What the policy sees:** gates and poles at nominal poses until within 0.7 m in x and y, then at
their true poses, as the race's sensor shows them. Collisions, passes and rewards use the true poses.
The guide stays the plan of the nominal track.

**Limits:** the guide is planned once for the nominal track, so it cannot be used on level 3 as it
is. Not trained on mass or inertia changes, although level 2 has them. Not flown on a real drone.

## The RL scene controller

The first step towards unseen tracks: a policy trained only on randomly built tracks, which works
when the track is where its guide says. It does not yet handle gates and poles that move.

```
generated layouts --> topt_planner (in advance) --> library of guides --> PPO --> rl_scene.npz
```

| Part | File |
|---|---|
| Controller used in the race | `control/rl_scene.py` |
| Trained policy | `control/rl_scene.npz` (`level1_random_tracks_fixed_guides`), trained with nothing moving |
| Inputs, output and guide library | `utils/scene_policy.py` |
| Random tracks | `utils/scene_generator.py` |
| Solving guides in advance | `scripts/solve_scene_library.py` |
| Training | `scripts/train_scene_policy.py` |

**Generated tracks** have the race's format: 4 gates, 4 poles and the race's order of passes, so one
gate is flown twice. Each leg is sampled on its own, and the ranges span levels 2 and 3 together,
because level 3 keeps poles 0.8 m from gates and level 2 has them closer. The whole process is in
[training_scenarios.md](training_scenarios.md).

**Inputs (86)** are those of `RLRace`, except:

- Absolute position and the share of the lap are gone, so it cannot recognize a track.
- Added: height, the distance to the walls in four directions, the nearest gate's axis, and the vector
  to that gate's closest bar.
- All directions are in a frame that turns with the guide, so a scene looks the same whichever way it
  faces. The drone stays at zero yaw, and the output's thrust axis is turned back into the world.

**Network and reward** match `RLRace`: 2 x 64, the same reward terms, and a real ending at the last
pass.

**Training episodes:** 25% start on the ground and the rest along the guide. Starts that touch something
are redrawn. Mass, inertia and the random force follow level 2. `--randomize` scales the shifts of gates,
poles and the start, and `rl_scene.npz` used 0. Validation tracks choose the policy that is kept;
level 2 and level 3 laps are only logged.

**Limits:**

- With level 2's shifts, the best run finished 8/32 on the real level 2. Every guide stays where it was
  planned, even after the sensor reveals a moved gate.
- About 14% of the guides in the libraries behind `rl_scene.npz` leave the arena, by up to 48 cm, because the library solver had lost its arena check. It has it again, but `topt_planner` itself has no arena limit, so such a guide can still come up in flight.
- On generated tracks `RLScene` plans once from the layout in the first observation, so it can start
  on level 3, but level 3's shifts are the same blocker as level 2's. That plan takes about 15 s on a
  new track (12-20 s), which has not been checked against the rules.

The network is plain NumPy and takes about 9 microseconds per call on one CPU core, against 20 ms per
control step, so flight needs no GPU.

## Training

Both trainers compile the whole iteration (rollout, returns and PPO updates) into one JAX function
on the GPU, and only return to the CPU for checks.

| | `scripts/train_path_policy.py` | `scripts/train_race_policy.py` |
|---|---|---|
| Task | Follow a path, on bench paths and random curves | Race one track |
| Check | `utils/mpcc_bench.py`, flown at nominal mass and +-5 g | A lap from the ground in every world |
| Speed | About 3M steps/s | About 1.7M steps/s |
| Default randomization | Mass, inertia, random force | Gates, poles, start, start velocity, random force |

Shared settings: 8192 worlds, 32 steps per iteration, 5 epochs of 4 minibatches, a floor of
`log_std >= -2.3` on the action noise. The feature functions accept NumPy or JAX arrays, so flight
and training run the same code. Tests check that they agree.

`scripts/train_scene_policy.py` uses the same PPO and reward, on a library of generated tracks instead
of one track, at about 1.6M steps/s.

`train_path_policy.py` trains the earlier path-following policy. It is not used by a race controller
yet; its PPO code is what `train_race_policy.py` reuses. Policies saved before the rotation matrix
change (`euler_*`, `noise_floor*`) no longer load with the current features.

Run outputs go to `.tmp/topt/rl2/`, which is git-ignored.

## Track grouping

`lsy_drone_racing/utils/track_groups.py` splits a track into local scenes: the fewest runs of
consecutive events that each span at most T seconds of flight. An event is a gate pass, or a pole
within 0.25 m of the straight route between gates. Passes count rather than gates, because a gate
flown twice is two tasks, and a pole beside two separate legs is met twice. What lies between groups
is a transit, with no event in it. Flight time comes from a fast point-mass estimate
(10 m/s^2 per axis, about 0.11 s of error per leg on level 0).

The same file also has `segment_path`, which splits an ideal path instead of grouping objects:
stretches that reach a curvature of 2 /m are hard, the rest easy. Hard stretches close together
merge, and long ones are split, to keep each one compact. Its input is a solved `topt_planner` path.

Both are stand-alone experiments: **no controller or trainer uses them yet.**

## Fast guide planner

`lsy_drone_racing/utils/fast_planner.py` plans a guide in milliseconds instead of seconds, for random
tracks. **No controller uses it by default.** The policy
`.tmp/topt/rl2/level1_random_tracks_fast_guides_60steps.npz` was trained on its guides and needs them:
it does not fly well on solved guides, and the policy in `control/rl_scene.npz` loses finishes on fast
ones. Results are in [PLAN_path_rl.md](PLAN_path_rl.md).

| Stage | What it does |
|---|---|
| Route | `topt_planner._guess_waypoints`, without the points before and behind gates that no detour moved |
| Speeds | A shortest-path search over a grid of speeds along the route at each waypoint |
| Refine | Adam steps on the waypoints and velocities: total time plus a squared hinge on clearance, floor, walls and the gate crossing angle. Gates stay fixed |
| Check | The true collision shapes on the guide's beads, and the arena |

- **Segments:** bang-bang per axis, in closed form. The time is stretched until the axes together fit the
  thrust and tilt limits. Drag and jerk are not modelled, so `required_thrust` measures what a plan asks
  for.
- **Compiled:** route and check run in Python and NumPy, and the speeds and refine stages in one JAX
  function of fixed size, padded to 32 waypoints with 320 clearance samples. Samples are shared out by
  each segment's current length.
- **Retries:** `plan_guide` tries up to 4 routes until one passes its check, and `plan_fast` is one try.
- **Output:** `pos`, `vel`, `acc`, `time` and `crossings`, which `guide_arrays` and `load_library` accept.

- **Attitude:** `smooth_velocities` averages a plan's velocities over a window before they become a
  guide, because bang-bang acceleration makes the guide's attitude jump. Positions stay as planned.

| Script | What it does |
|---|---|
| `scripts/compare_fast_planner.py` | Compares it with solved libraries |
| `scripts/fly_fast_guides.py` | Flies a saved policy on both kinds of guide, in the training environment |
| `scripts/solve_fast_library.py` | Writes a library of fast guides for the tracks of a solved one, for `scripts/train_scene_policy.py` |
| `scripts/race_fast_guides.py` | Flies `RLScene` in the real race, with fast plans in place of `topt_planner.plan` |

## Tools

| Tool | What it does |
|---|---|
| `scripts/sim.py` | Fly a controller with the viewer |
| `scripts/debug_run.py` | Fly headless, report why each episode ended, write logs to `logs/` |
| `replay_site/` | Replay those logs in 2D and 3D in a browser |
| `scripts/evaluate.py` | The official score |
| `scripts/identify_attitude.py` | Identifies the attitude dynamics the MPCC uses |
| `scripts/tune_mpcc_primitives.py` | Tune the MPCC |
| `scripts/deploy.py` and `ros_ws/` | Fly a real drone |

The other controllers in `control/` are the course's: the `Controller` base class,
`state_controller.py` and `attitude_controller.py`, which the environment tests fly, and
`attitude_mpc.py` with the `*_multi.py` pair, which the multi-drone configurations and their tests
use. Earlier work (the A*/B-spline path planner and its MPCC `my_controller.py`, the characterisation
scenarios and the course's PPO example) was removed and is kept in git history.

## Environment

- **Linux GPU:** use the pixi environments `gpu` and `gpu-tests`. `.pixi/envs/default` and
  `.pixi/envs/tests` in this checkout were built on Windows and do not run here.
- **Activating any `sim` environment** runs `tools/setup_acados.sh`, which builds acados if needed.
- **CPU runs on the GPU install:** set `JAX_PLATFORMS=cpu`.

```bash
pixi run -e gpu python scripts/sim.py --config level0.toml --controller rl_race.py
```

```bash
pixi run -e gpu python scripts/train_race_policy.py --config level0.toml --out race.npz
```

```bash
pixi run -e gpu-tests python -m pytest tests/unit/control
```
