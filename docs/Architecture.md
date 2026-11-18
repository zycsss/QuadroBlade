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
  ToptMPCC (control/topt_mpcc.py)              RLScene (control/rl_scene.py)
  plan -> bend -> MPCC                         guide + policy network
  for tracks with reveals (level 2)            any track, the final controller
       |                                                         |
       +----------------> topt_planner <-------------------------+
                          time-optimal plan, cached on disk
```

**`RLScene` (`control/rl_scene.py`) is the final controller**: every `config/level*.toml` selects
it, so `pixi run python scripts/sim.py` flies it. On a race that shifts gates or poles (levels 2
and 3) it defaults to the runtime planner replanned at every reveal, with the style selector
choosing between three policies; elsewhere it flies the solved guide with `rl_scene.npz` (see "The
RL scene controller"). Two controllers fly the race today, both using
the same planner: the MPCC follows its plan, and `RLScene`, trained on generated tracks only, sees it
as a hint. `RLRace`, an RL policy trained on the one track it flies, was removed in favour of
`RLScene`; its results stay in the table for comparison, and its code is in git history up to
a6ecd62.

| Controller | File | Tested on | Result in the real environment |
|---|---|---|---|
| Time-optimal MPCC | `control/topt_mpcc.py` | Level 2 (its default level) | 54/64 finished at 4.40 s |
| Time-optimal MPCC | `control/topt_mpcc.py` | Level 0 | 8/8 finished at 3.73 s |
| RL racing policy, fixed track (`RLRace`, removed) | `control/rl_race.py`, in git history | Level 0 | 64/64 finished at 3.33 s |
| RL racing policy, randomized (`RLRace`, removed) | `control/rl_race.py`, in git history | Level 2 | 64/64 finished at 4.66 s |
| RL scene policy, never saw the track | `control/rl_scene.py` | Level 0 | 64/64 finished at 4.03 s |
| RL scene policy, never saw the track | `control/rl_scene.py` | Level 2 | 0/64 finished |
| RL scene policy, never saw the track | `control/rl_scene.py` | Random tracks from the race's generator, level 1's conditions, no shifts | 243/256 finished at about 4.67 s |
| RL scene policy, trained and flown on fast guides, not the default | `control/rl_scene.py`, through `scripts/race_fast_guides.py` | Same | 248/256 finished at about 4.92 s |
| RL scene policy, trained and flown on solved guides with faster limits, not the default | `control/rl_scene.py` with `controller.fast_guides = true` | Same | 249/256 finished at about 4.46 s |
| RL scene policy on the solved guide, the default on levels 0 and 1 | `control/rl_scene.py` with `guide_planner = "topt"` | Real level 3, seeds 9000-9127 | 2/128 finished |
| RL scene policy on runtime guides, replanned at every reveal, not the default | `control/rl_scene.py` with `guide_planner = "fast_runtime"`, `replan_on_reveal = true` | Real level 3 / real level 2 | 55/128 at 5.39 s / 29/64 at 4.61 s |
| RL scene policies trained on staged reveals, replanned at every reveal, not the default | `control/rl_scene.py` with `guide_planner = "fast_runtime"`, `replan_on_reveal = true`, `policy = rl_scene_<style>.npz` | Real level 3 / real level 2 | `speed_first` 88/128 at 5.72 s / 31/64 at 4.97 s; `middle_ground` 93/128 at 6.20 s / 34/64 at 5.15 s; `safety_first` 107/128 at 7.45 s / 54/64 at 6.43 s |
| The agent choosing between those three styles in flight, **the default on levels 2 and 3** | `control/rl_scene.py` with `guide_planner = "fast_runtime"`, `replan_on_reveal = true`, `style_selector = true` | Real level 3 / real level 2 | 106/128 at 6.66 s / 54/64 at 6.09 s |

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

## The track-specific RL policy (removed), and what the scene controller builds on

`RLRace` flew one track it was trained on: `rl_race_randomized.npz` when gates or poles move,
otherwise `rl_race_level0.npz`. The controller and its policies were removed in favour of `RLScene`
and are in git history up to a6ecd62. Its features, rules and trainer stay, because the scene
controller below is built on them and its trainer imports them. The design is described here once.

```
state, guide, gates, poles --> 79 inputs --> 64 --> 64 --> 7 outputs --> to_action --> attitude and thrust
```

| Part | File |
|---|---|
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
| Policy for runtime guides | `control/rl_scene_runtime_guides.npz` (`level1_random_tracks_fast_guides_60steps`), trained on guides of the runtime planner, nothing moving |
| Three styles, for levels 2 and 3 | `control/rl_scene_{speed_first,middle_ground,safety_first}.npz` (`staged_reveal_guides_*`), trained on shifted scenarios with guides that follow the reveals |
| Style selector | `control/rl_scene_style_selector.npz` (`style_selector_hold_10_ticks_gamma_0.99`), picks one of the three styles every 0.2 s |
| Inputs, output and guide library | `utils/scene_policy.py` |
| Random tracks | `utils/scene_generator.py` |
| Solving guides in advance | `scripts/solve_scene_library.py` |
| Training | `scripts/train_scene_policy.py` |

**Generated tracks** have the race's format: 4 gates, 4 poles and the race's order of passes, so one
gate is flown twice. Each leg is sampled on its own, and the ranges span levels 2 and 3 together,
because level 3 keeps poles 0.8 m from gates and level 2 has them closer. The whole process is in
[training_scenarios.md](training_scenarios.md).

**Inputs (86)** are those of `RLRace` above, except:

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

**Where the guide comes from**, set in the `[controller]` section of a level file:

A race whose `env.randomizations` holds `gate_pos`, `gate_rpy` or `obstacle_pos` (levels 2 and 3)
takes `SHIFTED_RACE_DEFAULTS` (runtime planner, replanned at every reveal, style selector), unless
`[controller]` sets `guide_planner`, `policy`, `replan_on_reveal` or `style_selector` itself. Level
files may only change the controller file and the control mode, so the choice is made in code.

| Setting | Guide | Policy by default |
|---|---|---|
| `guide_planner = "topt"` (default) | `topt_planner`, once before the race; about 15 s on a new track | `rl_scene.npz` |
| `guide_planner = "topt"`, `fast_guides = true` | The same with `FAST_GUIDE_CONFIG` | Set `policy` to one trained on those guides |
| `guide_planner = "fast_runtime"` | The runtime planner, once before the race; about 20 ms | `rl_scene_runtime_guides.npz` |
| `guide_planner = "fast_runtime"`, `replan_on_reveal = true` | The runtime planner, again whenever the sensor reveals a gate or pole | `rl_scene_runtime_guides.npz` |
| The same, and `style_selector = true` | The same | The three styles, with the selector choosing; `selector` and `style_policies` point to other files |
| The same, and `bender = true` | The same, each policy flying its guide as its bender bends it (`policy` and `bender_policy`, or `benders` under the selector). Tried on `safety_first`: no gain, nothing installed | Set them |
| The same, and `replan_from_state = true` | Each replan starts at the drone's position and velocity and goes through the next `replan_gates` gates (1 by default, 0 for the rest of the lap), and also replans at every pass. Worse with today's policies: 97/128 on real level 3 for the rest of the lap, 24/128 for the next gate only | Unchanged |

A replan goes through the layout as the sensor now shows it, from where the drone started, so it is
the kind of guide the policy was trained on. The drone joins it at the nearest bead between its last
and its next pass. A plan that fails its check isn't flown, and the previous guide stays. Each replan
runs inside the control step and is recorded in `RLScene.replans`. All levels run in attitude mode;
level 3 was set to state mode until this was fixed, and failed on its first step.

**Limits:**

- With level 2's shifts, the best run finished 8/32 on the real level 2. Every guide stays where it was
  planned, even after the sensor reveals a moved gate.
- About 14% of the guides in the libraries behind `rl_scene.npz` leave the arena, by up to 48 cm, because the library solver had lost its arena check. It has it again, but `topt_planner` itself has no arena limit, so such a guide can still come up in flight.
- On generated tracks `RLScene` plans once from the layout in the first observation, so it can start
  on level 3, but level 3's shifts are the same blocker as level 2's. That plan takes about 15 s on a
  new track (12-20 s), which has not been checked against the rules.

The network is plain NumPy and takes about 9 microseconds per call on one CPU core, against 20 ms per
control step, so flight needs no GPU.

**Inputs that never varied in training** keep their training value in flight (`PathPolicy`, spread
below `CONSTANT_INPUT_STD`). Training gates stand upright, so the vertical part of a gate's axis is
always 0 there, and its weights were never trained. Levels 2 and 3 tilt gates by up to 0.1 rad, and
before this a tilt saturated those inputs and every flight crashed.

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

**Shifted scenarios with staged guides (`--staged`)** train for levels 2 and 3. `scripts/solve_staged_library.py`
draws the race's shifts for every track and plans, with the runtime planner, the guide at every stage
of reveals: stage `k` knows the first `k` objects in the order the stage-0 guide reaches them
(`scene_policy.reveal_order`, `staged_layout`, `load_staged_library`). In training a drone flies the
guide of the stage its reveals have completed, and joins a new guide at its nearest bead close by. In
flight `RLScene` follows the same rule with `replan_in_order = true`.

**Three styles (`--style`), always trained side by side:**

| Style | Crash | Hit margin | Obstacle term | Time cost per step | Kept policy chosen by |
|---|---|---|---|---|---|
| `speed_first` | -20 | 2 cm | 0 | 0.2 | Lap time, once 70% finish |
| `middle_ground` | -20 | 3 cm | 0.02 x `exp(-d)` | 0.05 | Finishing, then time |
| `safety_first` | -60 | 5 cm | 0.05 x `exp(-3 d)` | 0.05 | Finishing, then the smallest clearance |

The lap check reports each style's clearance: the mean, over finished laps, of the smallest free distance
to any gate or pole.

**Bender and tracker trained together (`scripts/train_bender.py`).** A bender decides every 5 steps
(0.1 s) on a bend of the guide ahead (`scene_policy.make_bend`). The bend has a sideways and an
upward offset at 0.5, 1, 2 and 3 m ahead, fading out by 4 m, and a factor of 0.7-1.3 on the guide's
speeds. Each offset is clipped to what the drone can reach at its speed, at 6 m/s^2 sideways, and
optionally (`from_drone`) the bend starts at the drone along its velocity. The tracker sees the bent
points ahead and the scaled speeds, but its offset from the guide as planned.

The bender's 117 inputs are the tracker's 86 features, 7 more guide points up to 4 m, its last output,
and whether the next two gates and two nearest poles have been revealed. One PPO run trains both
networks, each with its own critic. The bender gets the style's reward, one transition per decision.
The tracker gets the same reward minus a falling cost for leaving the bent guide.

The tracker starts from a trained style and the bender from no bend. For the first 500 iterations
only the bender learns, and the largest offset grows from 5 to 30 cm. On `safety_first` it found no
bend that helps (see `docs/PLAN_path_rl.md`), so nothing is installed. `train_style_selector.py
--benders` chooses between pairs in the same way.

**The style selector (`scripts/train_style_selector.py`)** chooses between the three frozen styles in
flight, so the agent itself decides how much risk to take. Every `hold` = 10 control steps it picks the
style for the next 10, from the 86 scene features, the current style as one-hot and the 12 numbers that
the three styles would command (`scene_policy.selector_features`, 101 inputs, 2 x 64 network, 3
logits). PPO with a categorical output treats a decision and its held steps as one transition, with a
discount of 0.99 per decision. The reward is the same for every style: +1 per gate, +10 to finish, -5
for a crash at a 2 cm margin, and -0.02 per step. The kept selector scores best on validation laps by
`10 * finish - time + gates`, and every style alone is flown on the same laps for comparison. In flight
`RLScene._select` runs all three styles every step (about 40 microseconds in all) and records the steps
flown in each style in `style_ticks`. Results: 106/128 at 6.66 s on real level 3 and 54/64 at 6.09 s
on level 2, against 107/128 at 7.45 s and 54/64 at 6.43 s for `safety_first` alone.

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

## Fast runtime planner

`lsy_drone_racing/utils/fast_runtime_planner.py` plans a guide in milliseconds instead of seconds, for
random tracks. `RLScene` uses it with `guide_planner = "fast_runtime"`, not by default. Its settings
for flight are `RUNTIME_GUIDE_CONFIG` and `RUNTIME_VELOCITY_WINDOW`, through `runtime_plan`, and they
match the libraries `rl_scene_runtime_guides.npz` was trained on. That policy needs these guides: it
does not fly well on solved guides, and `rl_scene.npz` loses finishes on runtime ones. Results are in
[PLAN_path_rl.md](PLAN_path_rl.md).

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
| `scripts/compare_fast_runtime_planner.py` | Compares it with solved libraries |
| `scripts/fly_fast_guides.py` | Flies a saved policy on both kinds of guide, in the training environment |
| `scripts/solve_fast_library.py` | Writes a library of fast guides for the tracks of a solved one, for `scripts/train_scene_policy.py` |
| `scripts/race_fast_guides.py` | Flies `RLScene` in the real race, with fast plans in place of `topt_planner.plan` |
| `scripts/evaluate_reveals.py` | Flies `RLScene` on real level 2 or 3 in one guide mode: solved once, runtime once, runtime replanned at every reveal (also with the style selector, `runtime_replan_selector`), or runtime through the true layout as an upper bound. Records what each crash hit, how far it had moved and how long after its reveal, and the share of time flown in each style |

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

From the course, `control/` keeps only the `Controller` base class, which every controller inherits
from, and `state_controller.py`, which the environment tests fly. Its attitude-mode examples
(`attitude_controller.py`, `attitude_mpc.py` and their `*_multi.py` versions) were removed, with the
tests that flew them. Earlier work (the A*/B-spline path planner and its MPCC `my_controller.py`, the characterisation
scenarios and the course's PPO example) was removed and is kept in git history.

## Environment

- **Linux GPU:** use the pixi environments `gpu` and `gpu-tests`. `.pixi/envs/default` and
  `.pixi/envs/tests` in this checkout were built on Windows and do not run here.
- **Activating any `sim` environment** runs `tools/setup_acados.sh`, which builds acados if needed.
- **CPU runs on the GPU install:** set `JAX_PLATFORMS=cpu`.

```bash
pixi run -e gpu python scripts/sim.py --config level0.toml
```

```bash
pixi run -e gpu python scripts/train_race_policy.py --config level0.toml --out race.npz
```

```bash
pixi run -e gpu-tests python -m pytest tests/unit/control
```
