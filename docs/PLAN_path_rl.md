# Plan: path-following RL controller, trained on the GPU

Goal: a policy that takes the drone's state and the path ahead and outputs attitude and thrust,
replacing the MPCC behind the existing plan, libraries and bender. Target is a 4.1 s lap at level 2
with at least 50% finishes, ideally about 80%.

## Where things stand

- Branch `topt-branches`, last commit `1e86f44`: 4.43 s at about 85% finishes on fresh seeds.
- **Uncommitted, and needed for this work:**
  - `lsy_drone_racing/utils/mpcc_bench.py`: bench paths, speed profile, `fly`
  - `lsy_drone_racing/utils/path_policy.py`: features and action mapping (NumPy or JAX), policy, tracker
  - `scripts/train_path_policy.py`: PPO, rollout and environment, compiled in JAX
  - `scripts/tune_mpcc_primitives.py`: coordinate search on the bench (has line-length lint errors)
  - `tests/unit/control/test_mpcc_primitives.py`: 20 tests, passing
  - `tests/unit/control/test_path_policy.py`: 4 tests that NumPy and JAX agree, passing
  - `lsy_drone_racing/control/topt_mpcc.py`: exposes hard-coded weights; defaults keep behaviour
- `.tmp/` is git-ignored, so anything there only moves if you copy it: test variants
  (`.tmp/topt/fast/variants.json`, including `C1`, `C3`, `C4`), analysis scripts (`.tmp/topt/acc/`),
  and policies (`.tmp/topt/rl/` failed, `.tmp/topt/rl2/` GPU runs).

## Why the first run failed

CPU run stopped at 24.6M steps after 588 s, about 42k steps/s:

    it 1500  rew -1.171  ok 0/6  ratio 0.20  err 0.164  thrust 6.1  high 0%  std 0.06

Two bugs, both fixed:

- `sim.reset` stops the rotors in mid-air. Spinning up costs about 0.26 m of height with a hover
  command, close to the 0.3 m `lost` limit, so episodes from rest could hardly succeed. Training now
  starts 75% of episodes with the rotors at hover speed. The bench still starts them stopped, for
  both controllers, so its numbers stay comparable.
- Arrival bootstrapped on a state the critic never trains on, and the value ran away (reward above
  +1M). Arrival now really ends the episode.

## Step 1: set up the GPU environment (done)

On native Ubuntu in this checkout, with `pixi install -e gpu`. JAX lists the RTX 5080. The checkout
had come from Windows: CRLF files broke `tools/setup_acados.sh`, which runs whenever a `sim`
environment activates, so acados is built even for training. Windows builds of acados and the solvers
are kept in `.tmp/acados_windows_backup` and `.tmp/c_generated_code_windows_backup`.

## Step 2: make the GPU actually help (done)

Rollout, returns and updates are one compiled function. Full training loop, including PPO:

| worlds | 512 | 2048 | 8192 | 32768 | 131072 |
|---|---|---|---|---|---|
| steps/s | 700k | 1.70M | 1.88M | 1.96M | 1.79M |

The default is 8192: more worlds add no speed and fewer updates per sample.

## Step 3: get it to fly (in this order, stop when a step works)

Use a short run each time. The log prints mean episode length and how episodes ended.

1. **Hover and a straight line only (done).** `--paths line`: 100% of episodes arrive in about 50 s.
   All six bench paths with `--paths line,climb,turn90,hairpin,loop,s_bend`, 2000 iterations: best 3/6
   finished. `climb`, `hairpin` and `loop` tip over, and noise falls to 0.03.
2. **Remove the incentive to crash.** Add a small reward for staying alive, lower `corridor` from 20
   to about 2-5, widen `lost`, and make sure a typical step's reward is positive.
3. **Stop the noise collapsing.** Put a floor on `log_std` or add a small `ent`.
4. **Copy the MPCC first.** Run the MPCC on bench and race paths, record features and actions, and
   train the actor to imitate them before PPO.
   Only keep this if steps 1-3 aren't enough, since it needs acados.

Then widen: all bench paths, then random curves, then sudden path jumps like a reveal's bend.

## Step 4: bench test (go or stop)

Run with `mpcc_bench.fly(name, sim=sim, tracker_fn=...)`. The policy must beat the tuned MPCC:

| metric (mean of 6 paths) | current MPCC | tuned MPCC | target |
|---|---|---|---|
| Time over ideal | 1.46 | 1.29 | **below 1.29** |
| Largest path error | 16 cm | 10 cm | **10 cm or less** |
| Time above 15 m/s^2 | 34% | 46% | **above 46%** |
| Mean thrust acceleration | 13.8 | 14.4 | **above 14.4** |
| Paths finished | 6/6 | 6/6 | **6/6** |

For reference, the RL policy in the race: 15.9 m/s^2 typical, 61% above 15 m/s^2.

If it can't get there, stop and rethink before moving to the race.

### Results so far

All with `--log_std_min -2.3 --crash 50`, 6000 iterations, one seed each, best checkpoint by the
training score. Bench-only runs are tested on the paths they trained on, so they flatter the policy.

| run | attitude in and out | policy | paths | ratio | error | above 15 | thrust | steps/s |
|---|---|---|---|---|---|---|---|---|
| `euler_3x256_bench` | roll, pitch, yaw | 3 x 256 | bench | **1.23** | **9.6 cm** | **67%** | **15.3** | 2.0M |
| `euler_3x256_pool` | roll, pitch, yaw | 3 x 256 | bench + random | 1.32 | **8.3 cm** | **49%** | 14.4 | 1.9M |
| `matrix_3x256_bench` | rotation matrix | 3 x 256 | bench | 1.34 | **8.5 cm** | 43% | 14.1 | 2.0M |
| `matrix_2x64_bench` | rotation matrix | 2 x 64 | bench | 1.32 | **8.7 cm** | **51%** | 14.2 | 3.2M |

All finished 6/6. Bold meets the Step 4 target. `matrix_3x256_bench` later reached 1.29, 9.7 cm, 65%, 14.8 at
iteration 5500, which the score did not pick because it weighs error above 8 cm.

- Softening `corridor` to 5 and `lost` to 0.5 raised the error to 13-15 cm, so both stay at 20 and 0.3.
- Shrinking the policy cost nothing here: `matrix_2x64_bench` matches `matrix_3x256_bench`, trains 1.6 times faster.
- The matrix version has not matched `euler_3x256_bench`. Its input and output changed together, so the cause
  is unknown. Its yaw is now free and reaches the 1.57 rad limit in flight, which is one suspect.
- A control step takes under 0.2 ms with either size, far below the 20 ms limit.

Limiting yaw, rotation matrix and 2 x 64 throughout. Commands are always sent with the yaw limit, so
`--yaw_max 0` keeps the matrix but sends zero yaw:

| run | yaw | paths | randomized | ratio | error | above 15 | thrust |
|---|---|---|---|---|---|---|---|
| `matrix_2x64_yaw_penalty` | penalty 0.2 | bench | no | 1.32 | **8.5 cm** | **56%** | 14.4 |
| `matrix_2x64_yaw_zero` | zero | bench | no | 1.30 | **8.8 cm** | **53%** | **14.6** |
| `matrix_2x64_yaw_zero_randomized_bench` | zero | bench | yes | 1.32 | **9.3 cm** | **55%** | **14.6** |
| `matrix_2x64_yaw_zero_randomized_pool` | zero | bench + random | yes | 1.40 | **7.5 cm** | 33% | 13.4 |

Zero yaw is the default choice from here. At 38.4 g and 48.4 g, mean largest error and worst path:

| run | 38.4 g | 48.4 g | ratio at 38.4 / 48.4 g |
|---|---|---|---|
| `matrix_2x64_yaw_zero` | 11.3 cm, worst 15 | 11.3 cm, worst 12 | 1.27 / 1.38 |
| `matrix_2x64_yaw_zero_randomized_bench` | 10.6 cm, worst 15.2 | 10.7 cm, worst 12.5 | 1.27 / 1.41 |
| `matrix_2x64_yaw_zero_randomized_pool` | 8.3 cm, worst 9.4 | 9.8 cm, worst 10.3 | 1.36 / 1.47 |

All finished 6/6 at every mass. Training now uses level 2's mass (+-5 g), inertia (+-1e-6 per entry)
and random force (+-0.1 N per axis, each simulator step). The bench itself has no random force.
`matrix_2x64_yaw_zero_randomized_pool` was still improving at every check when it stopped at 6000 iterations.

## Racing controller on the level 0 track

`lsy_drone_racing/control/rl_race.py`, trained by `scripts/train_race_policy.py`, with the rules and
features in `lsy_drone_racing/utils/race_policy.py`. It sees the state, the guide from `topt_planner`,
the next gates and the nearest obstacles, and it is never held to the guide. The attitude goes in and
out as a rotation matrix (yaw command 0), and the policy is 2 x 64.

Real race environment, 32 seeds, except the MPCC with 8:

| controller | finished | mean lap | reward for moving |
|---|---|---|---|
| `gate_distance_margin` (the default) | 32/32 | **3.33 s** | closer to next gate |
| `gate_distance_sharp_obstacle_no_margin` | 30/32 | 3.30 s | closer to next gate |
| `gate_distance_soft_obstacle_no_margin` | 24/32 | 3.20 s | closer to next gate |
| `path_progress_funnel` | 32/32 | 4.20 s | potential along the guide, with funnel |
| `path_progress` | 32/32 | 4.26 s | distance along the guide |
| `topt_mpcc` | 8/8 | 3.73 s | - |

The plan itself takes 3.21 s. The policy flies 10.6 m against the guide's 10.1 m, up to 38 cm off it.

- Without a cost for time and for flying slowly, the policy learns to hover far from everything.
- Without a hit margin it skims gates by 1-3 cm and hits them in the real race: 100% in training,
  75% in the race. Training now counts anything within 3 cm as a hit.
- Physics in training matches the race to about 1 cm over a lap. A 1500-point check found the
  collision rule slightly stricter than the race's, by up to 8 cm in places.
- Trained on the nominal track only. Not yet tried with level 2's moved gates, mass or inertia.

### Randomized gates, poles and starts

Trained 6000 iterations with gates, poles and the drone's start moved as level 2 moves them, start
velocity up to 0.5 m/s per axis, and the race's sensor reveal. No mass or inertia changes in training.
Real race environment, the same 64 seeds:

| controller | level 0 | level 2 |
|---|---|---|
| `randomized_gate_distance_margin` | 64/64, 4.46 s | **64/64, 4.66 s** |
| `gate_distance_margin` | 64/64, 3.33 s | 0/64 |
| `topt_mpcc` | - | 54/64, 4.40 s |

In the training lap check, which counts hits with a 3 cm margin, it reached 93.9% at 4.63 s and was
still improving. About 11% of starts along the guide already touch something and crash at once.

## Grouping a track into local scenes

`lsy_drone_racing/utils/track_groups.py`: the fewest runs of consecutive gate passes that each span
at most T seconds of flight, from a point-mass time estimate. Among equal splits, the one whose
shortest transit is longest. Nothing flies with this yet; it only splits tracks.

| T | level 2 groups | same at T +-10% | same if gates move | level 3 groups | same at T +-10% | same if gates move | level 3 poles in groups / transit / off route |
|---|---|---|---|---|---|---|---|
| 0.4 s | 5.00 | 100% | 100% | 5.00 | 100% | 100% | 0.00 / 2.80 / 1.20 |
| 0.6 s | 4.17 | 6% | 53% | 4.76 | 71% | 91% | 0.57 / 2.23 / 1.20 |
| 0.8 s | 3.00 | 100% | 94% | 4.08 | 50% | 87% | 2.05 / 0.75 / 1.20 |
| 1.0 s | 3.00 | 100% | 94% | 3.44 | 52% | 89% | 2.67 / 0.13 / 1.20 |

300 tracks each. There are 5 passes per lap, so 5 groups means every pass stands alone.

- Short T on level 3 leaves every pole in a transit, so a transit cannot be plain flight.
- The time estimate is off by about 0.11 s per leg against `topt_planner` on level 0 (legs of 0.54 to
  0.72 s), and that alone can change the split near a threshold.
- Groups can change when a gate is revealed, so they should be fixed at the start of a lap.

Poles grouped with passes, as events in time. At T = 0.4 s and a route width of 0.25 m, level 2 has
5.00 groups per lap and level 3 has 5.28. Scene types over 300 tracks:

| scene | level 2 | level 3 |
|---|---|---|
| 1 gate | 70% | 74% |
| 1 gate + 1 pole | 29% | 19% |
| Pole on its own | 0% | 6% |
| 1 gate + 2 poles | 1% | 1% |

Stability when gates and poles shift by level 2's amounts, 300 tracks:

| route width | T | which passes group together, level 2 / 3 | exact groups with poles, level 2 / 3 |
|---|---|---|---|
| 0.25 m | 0.4 s | 100% / 100% | 28% / 50% |
| 0.25 m | 0.8 s | 66% / 80% | 28% / 46% |
| 0.5 m | 0.4 s | 100% / 100% | 5% / 27% |
| 0.5 m | 0.8 s | 41% / 79% | 6% / 28% |

So the groups of passes can be relied on at T = 0.4 s, but which poles belong to them cannot, because
a shift easily moves a pole across the route width. Poles should not be a hard decision in flight: the
policy already sees the nearest poles in every phase.

### Segmenting the ideal path by curvature

`segment_path` in `track_groups.py`, on `topt_planner` plans solved for 40 shifted level 2 tracks and
40 generated level 3 tracks. One initial guess solved all 80 in about 2.5 s (5.4 s at most).

| curvature threshold | hard per lap, level 2 / 3 | hard time | hard length | speed hard / easy, level 3 | gates in hard, level 2 / 3 | hard with no gate, level 2 / 3 |
|---|---|---|---|---|---|---|
| 1.0 /m | 6.10 / 5.55 | 56% / 49% | 0.35 s | 2.53 / 3.52 | 39% / 57% | 68% / 49% |
| 2.0 /m | 3.33 / 2.90 | 30% / 22% | 0.33 s | 2.10 / 3.68 | 18% / 21% | 74% / 64% |
| 3.0 /m | 2.25 / 2.05 | 19% / 13% | 0.29 s | 1.82 / 3.67 | 9% / 12% | 80% / 72% |

- Curvature does separate slow from fast, and hard stretches come out compact, about 0.3 s long.
- The plan turns between gates and passes them fairly straight, so about 80% of gate passes count as
  easy at 2 /m. Median at a pass: 0.6-0.8 /m at 3.3-3.8 m/s.
- On level 0, the no-margin policy's real crashes were at gate 2, which curvature calls easy.
  Curvature alone therefore misses precision. Clearance is the likely second signal; not tested yet.
- Against the nominal level 2 plan, shifted plans have the same number of hard stretches 28% of the
  time, with a median overlap of 46% in time. Solving from scratch adds to this, so a plan should be
  fixed at the start of a lap and its segments kept.

### Labelling whole legs between gates

Proposal: two RL agents, one for low curvature legs and one for high curvature legs. Peak curvature
per leg on the same 80 plans has two clusters, about 0.5-1.5 /m and 3-6 /m, with few legs between.
At 2.5 /m level 0 alternates low, high, low, high, low, and 50% of legs are high on levels 2 and 3.

| threshold | mean speed low / high, level 2 | level 3 | slowest low / high, level 3 | leg time low / high, level 3 | within 0.3 /m of threshold, level 2 / 3 |
|---|---|---|---|---|---|
| 2.0 /m | 3.45 / 2.90 | 3.47 / 3.34 | 2.09 / 1.65 | 0.68 / 0.90 s | 10% / 10% |
| 2.5 /m | 3.44 / 2.82 | 3.45 / 3.34 | 2.10 / 1.56 | 0.70 / 0.92 s | 8% / 8% |

A high leg is a turn plus fast flight on both sides of it, so its mean speed is not much lower. Level
2 labels at 2 /m match nominal per leg 65-100% of the time, and for all legs 55% (2.5 /m not measured).

## Plan: train on randomized local scenes, test on level 2

**Status: built and tested. It fails the level 2 target, and the blocker is known. See "Results" at
the end of this section.**

**Goal:** a policy that has learned local tasks instead of one track, so that it can carry over to
unseen tracks. It trains on randomly constructed scenes and is tested on the real level 2. No level 2
layout is used in training or to pick the kept policy, so the level 2 result measures generalization.
Level 3 itself comes after this plan.

### What the scenes must cover

Measured per gate pass on 1000 layouts of each level (5th to 95th percentile, median in brackets):

| measure | level 2 | level 3 |
|---|---|---|
| Straight distance from the last gate [m] | 0.66-2.37 (1.84) | 1.07-3.70 (2.02) |
| Turn into the gate [deg] | 16-101 (36) | 2-81 (24) |
| Turn into the repeated gate, last pass [deg] | included above | 5-138 (51) |
| Turn out of the gate [deg] | 48-162 (77) | 48-176 (133) |
| Nearest pole to the gate [m] | 0.32-0.70 (0.51) | 0.81-1.95 (0.92) |
| Nearest pole to the straight route [m] | 0.09-0.48 (0.30) | 0.04-1.13 (0.44) |

Turn angles are between the straight line to or from the neighbouring gate and the gate's passing axis.

- **Copying the level 3 generator would not prepare it for level 2.** Level 3 keeps poles at least
  0.8 m from gates, while level 2 always has one closer than that. 23% of level 2's legs are also
  shorter than any on level 3.
- **Level 3 has harder turns.** The next gate is often behind the one just passed. On the repeated
  last pass, 20% of approaches come from behind the gate.
- So the scene ranges must span both levels, not just one.

### Step 1: scene generator

`utils/scene_generator.py` builds a short strip: a run-in point and three gates in a row. The middle
gate is the task, the first gives what lies behind, and the third gives the way out.

| scene setting | range |
|---|---|
| Leg length | 0.6-4.4 m |
| Turn into and out of a gate | 0-180 deg, weighted towards the common angles |
| Gate height | 0.6-1.3 m |
| Gate roll, pitch | +-0.05, +-0.1 rad |
| Poles near the task | 0, 1 or 2, in 30% / 50% / 20% of scenes (natural is about 72% / 26% / 1%) |
| Pole to gate | 0.3-2 m |
| Pole to route | 0-1.2 m |

Strips that overlap themselves or do not fit in the arena are thrown away. Take-off scenes (about 25%)
start from the platform, with the first gate as close as level 2's 0.74 m and a turn of up to 100 deg.

**Check:** plots of the ranges against the table above, and tests for the exclusions.

### Step 2: library of solved guides

`topt_planner` remains the guide. At about 2.5 s per solve it is too slow to run every episode, so it
is solved in advance: about 3000 strips on 12 of the 16 cores, which should take 10-15 minutes. Failed
solves are dropped and counted. 500 more strips are kept apart for validation.

The strip the plan was solved for is the nominal layout. Each episode adds level 2's shifts on top, and
the sensor reveals them as in the race, so the policy learns that a guide is out of date when a gate
turns up elsewhere.

### Step 3: inputs that only describe the scene

The policy can't generalize while it can recognize where it is on one track. So:

- **Removed:** absolute position and share of the lap done.
- **Added:** height, and the distance to the arena's walls ahead, behind and to each side.
- **Rotated:** all directions go into a frame that turns with the guide's heading, so that the same
  scene looks the same whichever way it faces. The output is rotated back. The guide's heading is used
  because, unlike the direction to the next gate, it does not jump at a pass.
- **Kept:** the 2 x 64 network and rotation matrix input and output. It only grows if Step 5 shows it
  is too small.

**Check:** NumPy and JAX tests as today, and a test that rotating a whole scene leaves the inputs
unchanged.

### Step 4: scene training

- Every world flies its own strip and guide (`nearest` already takes a row per world).
- An episode starts before the task gate, with the plan's velocity at 50-100% plus random velocity, and
  ends after the third gate, on a crash, or at a time limit.
- The end of a strip counts as a cut-off rather than a real ending, so the critic still expects the
  race to go on. **Risk:** similar bootstrapping once ran the critic away on the path-following task.
  If it does, use a fixed bonus and a real ending.
- Starts already touching something are redrawn (about 11% of mid-lap starts today).
- Mass and inertia are randomized as on level 2. The random force stays, and the reward stays as it is.
- The kept policy is the one that does best on the validation strips.
- The level 2 lap check still runs and is logged, but it does not choose the policy.

A run should take about 20 minutes at 6000 iterations.

### Step 5: test on level 2

Real race environment, the same 64 seeds as before, with these to beat:

| baseline | level 2 |
|---|---|
| `randomized_gate_distance_margin`, trained on level 2's track | 64/64, 4.66 s |
| `topt_mpcc` | 54/64, 4.40 s |

1. **Validation strips.** At least 90% of them flown without a crash, or stop and fix training first.
2. **Generalization test.** `local_scenes_no_level2`, at least 58/64 (90%).
3. **Price of generality.** Fine-tune a copy on level 2 as `local_scenes_tuned_level2`. Its gap to the
   untuned policy shows how much the track-specific knowledge was worth. It is kept separate, because
   it no longer measures generalization.
4. **Confirm.** The best policy on 256 fresh seeds, then `scripts/evaluate.py`.

Expect the untuned policy to be slower than 4.66 s at first, since that one was trained on this very
track. If it fails the 58/64, replay the crashes, find which scene type they are, and fix the ranges or
the mix. Adding level 2 layouts to training would hide the problem.

### Results

**Changes from the plan while building:**

- Generated tracks use the race's format (4 gates, 4 poles, passes `[1, 2, 3, 4, 2]`) instead of
  3-gate strips. This reuses the planner unchanged and gives the repeated-gate scenes by itself.
- The last pass really ends an episode from the start, since `train_path_policy.py` already records
  that bootstrapping at an arrival ran the critic away.
- The policy also sees the nearest gate's axis and the vector to its closest bar (86 inputs).

Library: 2899/4300 training, 462/700 validation and 241/300 level 3 tracks solved, in about 33 minutes.
Failed solves did not remove the hard scenes: the ranges before and after solving are nearly equal.

**Training lap checks.** Laps from rest, counting hits with a 3 cm margin, none of these trained on
level 2 unless stated. Lap times are over finishers only.

| run | iterations | validation | level 2 track | level 3 |
|---|---|---|---|---|
| `local_scenes_no_level2` (80 inputs) | 6000 | 12.8% | 6.7% | 19.3% |
| `local_scenes_guide_progress` (80 inputs) | 3000 | 10.7% | 9.2% | 18.2% |
| `local_scenes_2x256` (80 inputs) | 3000 | 12.8% | 12.0% | 20.1% |
| `local_scenes_nearest_gate_shape` | 6000 | 15.2% | 10.4% | 24.5% |
| `clear_guides_level2_shifts` | 4000 | 16.9% | 9.3% | 25.1% |
| **`clear_guides_no_shifts`, nothing moves in its checks** | 4000 | **60.8%** | **100%, 4.21 s** | **65.6%** |
| `scene_inputs_level2_only`, trained on the level 2 track alone | 3000 | 0% | 92.6%, 4.74 s | 0% |

The clear-guide runs train, validate and check level 3 on filtered libraries, so their columns are not
directly comparable with the rows above. `local_scenes_nearest_gate_shape`, trained on the unfiltered
library, reaches 21.2% on the filtered validation tracks.

**Real race environment, seeds from 1000:**

| policy | level 0 | level 2 |
|---|---|---|
| `clear_guides_no_shifts` | **64/64, 4.22 s** | 0/64 (49 gate hits, 15 contacts) |
| `clear_guides_level2_shifts` | - | 8/32, 5.38 s |
| `local_scenes_nearest_gate_shape` | - | 6/32, 5.41 s |
| `local_scenes_no_level2` | - | 5/16, 5.22 s |

The level 2 target of 58/64 was not met. The shorter runs stopped early because they were clearly short
of it, so they are not exact finish rates.

**What this shows:**

- **Generalization works when the track is where the guide says.** 64/64 at 4.22 s on a layout it never
  trained on, against 4.46 s for the policy trained on that track's moved versions.
- **The shifts are the blocker.** The same setup falls from about 61% to about 17% on validation once
  gates and poles move. Removing them is the only change that made a large difference.
- **It is not the inputs, reward or network size.** Same inputs on one track reach 92.6%. Guide progress
  and 2 x 256 are about the same as the first setup.
- **My missing-input theory was mostly wrong.** Adding the nearest gate's shape gave only 12.8% to 15.2%.
- **Bad guides are real but not the main cause.** 11% of generated and 8% of level 3 guides pass through
  a gate they are not targeting, and 7-9% through a pole, because the planner only enforces clearance
  from objects it picks as nearby, at its nodes. Filtering keeps about 45% of tracks, and it did not
  noticeably change the rate with shifts.
- **The policy trained on one track copes with shifts, the general one does not.** The likely reason, not
  yet tested, is that one track's corrections can be memorized. A general policy has to read them from
  a guide that stays wrong after the sensor has shown the true position.

**Never-seen random tracks, nothing moving.** `clear_guides_no_shifts` in the real race, on the race's
own level 3 generator with level 1's conditions: mass, inertia, start, action noise and random force
are drawn as usual, and gates and poles stay exactly where reported. Seeds 7100-7227, with the guide
planned from the reported layout using all of the planner's guesses (all 128 solved).

| guide | finished |
|---|---|
| All tracks | **105/128 (82%)**, mean 5.03 s, 3.84-6.84 s |
| Guide clear of contact on the nominal layout | 97/110 (88%) |
| Guide itself touches a pole or frame | 8/18 (44%) |

The 23 failures, rerun on the same seeds with the same outcome and classified from the last step
before the crash: 12 on a gate other than the next one, 9 on a pole, 2 out of the arena. 18 of the 23
had passed at most 3 gates. Guides that touch something account for 10 of the 23. The scratch script is
`.tmp/topt/rl2/random_track_eval.py`.

### Level 1 difficulty on random tracks, with the planner fixed

**The planner.** The default settings gave bad optima on random tracks: one "time-optimal" plan was 46 m
long, took 8.49 s, reached 12.9 m/s and had nodes 1.73 m apart, stepping over gates and poles between
nodes. Constraining every object and checking midpoints was not enough, since the solver only needed a
longer stride. `RANDOM_TRACK_CONFIG` limits nodes to 25 cm apart, with these results before frame
clearance was raised to 12 cm:

| One guess | Solved | Median plan | Guides through a pole | Guides through a frame |
|---|---|---|---|---|
| Generated, default settings on existing library | 462/700 | 3.94 s | 6.7% | 11.7% |
| Generated, every object and midpoints | 116/120 | 4.70 s | 19.8% | 16.4% |
| Generated, also limited stride | 115/120 | 3.99 s | 0% | 3.5% |
| Level 3, default settings on existing library | 241/300 | 4.03 s | 8.3% | 10.0% |
| Level 3, also limited stride | 120/120 | 3.87 s | 0% | 1.7% |

My earlier explanation, that clearance was only missing for far objects, was incomplete. Fixed levels
keep the default settings, and level 2's plan is unchanged. With frame clearance at 12 cm, across the
256 guides flown below, no guide came within 3.8 cm of a pole, and the worst frame gap was -0.6 cm. `plan()` with all guesses took a median of
14.7 s on a new track (11.8-19.6 s, 10 tracks, one process). My earlier 20-25 s was an estimate.

**Guides** now follow the curve the plan flies, rather than straight lines between nodes.

**Libraries:** 942/1200 from the race's generator and 597/800 from `scene_generator.py` for training,
157/200 from the race's generator for validation. The rest were dropped for having no clear guide.
These libraries were solved without the arena check, which I had dropped by mistake: 218 of 1539
training guides and 22 of 157 validation guides leave the arena, by up to 48 cm. The check is back for
future libraries. Training was not repeated.

**Training:** `level1_random_tracks_fixed_guides`, 8000 iterations, `--randomize 0`. Best validation lap
check 81.4%, levelling off at 78-81% from about iteration 3000.

**Real race, level 1's conditions on the race's own generator, no shifts, using all planner guesses:**

| Policy, guides | Seeds | Finished | Mean time | Ended by |
|---|---|---|---|---|
| `clear_guides_no_shifts`, first test with old guides | 7100-7227 | 105/128 | 5.03 s | 12 other gate, 9 pole, 2 out |
| `clear_guides_no_shifts`, fixed guides | 7100-7227 | 117/128 | 5.06 s | 10 other gate, 1 next gate |
| `level1_random_tracks_fixed_guides`, fixed guides | 7100-7227 | 120/128 | 4.72 s | 7 other gate, 1 next gate |
| `level1_random_tracks_fixed_guides`, fixed guides | 8000-8127 | **123/128** | 4.62 s | 2 other gate, 2 pole, 1 next gate |

- Together, 243/256 (95%) at about 4.67 s. Seed set 7100-7227 was already used as a comparison, so 8000-8127 is the clean test.
- The guides gave most of the gain in finishes: 105 to 117 with the same policy. New training added 3 more on the same seeds, which is too few to separate from noise (6 flights only the new policy finished, 3 only the old), but it was 0.34 s faster.
- Crashes mostly come from a gate other than the next one, and 7 of 8 and 4 of 5 happen after at least 3 passes.
- Real level 2, same policy: 0/64 (47 gate hits, 17 contacts), as expected of a policy trained with nothing moving.
- Real level 0: 64/64 at 4.03 s. The plan's own lap time on these tracks is a median of about 3.95 s from the libraries, so finishers are about 0.7 s behind it.

**Not done:** shifts, which remain the blocker for levels 2 and 3. Training is still not local per scene.

**Next, in the order to try:**

1. **Bend the guide to what the sensor shows,** so that it stays right after a reveal. `path_bend.py`
   already does this for the MPCC; a simple version moves the guide with the revealed gates.
2. **Bring the shifts in gradually,** continuing from `clear_guides_no_shifts`.
3. **Fix the planner's clearance** to all objects, which level 3 needs anyway.

### After this plan

- **Level 3.** `RLRace` builds its guide from the config, which on level 3 holds only placeholder
  positions. It has to plan from the generated layout at the start of each race, so whether that 2.5 s
  solve is allowed needs checking.
- **Two agents, low and high curvature.** Only if one policy can't do both. The library makes this
  cheap to compare, since each strip already has a plan to label.

## Can the policy find a better line than its guide?

Status: tested on level 1 difficulty, not flown in the real race. No policy or code changed: the two
options tried here gave no gain and were removed again, and `rl_scene.npz` stays as it was.

The policy's reward has no guide term, yet it stays close to the guide. Measured over 2048 laps from
the floor on validation tracks, with no action noise, counting finished laps only:

| Run, 6000 iterations (first row 8000) | Finished | Flown / plan time | From guide | Off centre at a pass |
|---|---|---|---|---|
| `level1_random_tracks_fixed_guides` (before) | 82.7% | 1.173 | 4.8 cm | 2.8 cm |
| `free_gate_opening`: no pull inside +-8 cm of the gate | 80.9% | 1.161 | 4.6 cm | 3.2 cm |
| `free_gate_opening_high_time_cost`: also tick cost 0.2 | 80.9% | 1.153 | 5.1 cm | 2.9 cm |
| `free_gate_opening_low_noise`: also action noise floor 0.03 | 67.4% | 1.237 | 7.3 cm | 3.0 cm |
| `free_gate_opening_coarse_guide`: also hides the guide up to 16 beads ahead | 63.3% | 1.244 | 7.5 cm | 2.8 cm |

No lap in any run beat its plan. Free room at a gate was left unused in all of them, also when nothing
in the reward pulled to the centre, and hiding the near guide only made the policy slower. Laps from
`free_gate_opening_high_time_cost` are at most 2% quicker, for 1.8 points fewer finishes.

The more useful finding is what that room is worth. The planner forces crossings within 3 cm of the
centre, so the same 48 validation layouts were solved again under other limits:

| Planner setting | Mean plan time | Median change |
|---|---|---|
| As used | 4.005 s | |
| Crossings within +-8 cm | 3.971 s | -0.9% |
| Crossing angle up to 1.4 rad | 4.010 s | 0.0% |
| Both | 3.970 s | -0.9% |
| Tilt 1.4 rad and thrust 17 m/s^2, instead of 1.1 and 15 | 3.619 s | -9.9% |
| All of them | 3.573 s | -10.6% |

So using the gate's whole opening saves about 0.03 s per lap at best, while the limits on tilt and
thrust are worth about 0.4 s. These are point-mass plans from local solves, so they compare settings
and aren't times a drone can fly. The highest limits also leave little of the motors' reserve:
the heaviest drone reaches 16.5 m/s^2.

The policy loses 0.09-0.18 s on each of the five legs, so the take-off isn't the main cost either.

### Faster guides

The policy follows its guide closely, so the next test was a faster guide, with limits raised to tilt
1.3 rad and thrust 16 m/s^2 for `topt_planner` (`FAST_GUIDE_CONFIG`). The heaviest drone reaches 16.5
m/s^2 and the policy's own tilt limit is 1.4 rad. On tracks shared with the earlier libraries, plans
are a median 5.7% quicker. Fewer tracks are kept: 128 against 157 validation tracks, and the new
libraries have the arena check, which the earlier ones don't.

All flights are in the real race with level 1's conditions on the race's generator, no shifts,
seeds 7100-7227 and 8000-8127. Fast-planner libraries use the same tracks as `fixed_*`.

| Policy | Guide in flight | Median planned lap | Seeds 7100+ | Seeds 8000+ | Together |
|---|---|---|---|---|---|
| `level1_random_tracks_fixed_guides` | `topt_planner`, as used | | 120/128 at 4.72 s | 123/128 at 4.62 s | 243/256 at about 4.67 s |
| **`level1_random_tracks_fast_limit_guides`** | **`topt_planner`, faster limits** | | **125/128 at 4.50 s** | **124/128 at 4.41 s** | **249/256 at about 4.46 s** |
| `level1_random_tracks_fast_guides_60steps` | Fast planner, thrust 15 | | 126/128 at 5.00 s | 122/128 at 4.84 s | 248/256 at about 4.92 s |
| `level1_random_tracks_fast_guides_thrust16_tilt13` | Fast planner, thrust 16, tilt 1.3 | 4.50-4.64 s | 122/128 at 5.00 s | 122/128 at 4.81 s | 244/256 at about 4.90 s |
| `level1_random_tracks_fast_guides_thrust18_tilt13` | Fast planner, thrust 18, tilt 1.3 | 4.10-4.15 s | 123/128 at 5.01 s | 123/128 at 4.88 s | 246/256 at about 4.94 s |

- **Solved guides with faster limits win.** On the 238 flights both finished, laps are 0.22 s quicker
  on average and quicker on 98% of them. 11 flights only the new policy finished, and 5 only the
  earlier one. The finish difference alone is within noise. It still needs the solve of about 15 s at
  race start, which hasn't been checked against the rules.
- **The fast planner gained nothing from higher limits.** Its plans got up to about 0.8 s quicker but
  its laps did not, so at these limits its hint is no longer what limits the lap. The cause isn't
  established. These plans are bang-bang with no jerk limit, which may be harder to follow.
- **Fast plans still work inside the race:** all 512 plans at raised limits were clear, 493 on the
  first try. Planning time in the race wasn't measured.
- **Not flown:** the new policy on real level 0 or level 2, or with shifts. Its validation lap check was
  88.8% at 4.54 s. On the shared validation guides it finished 88.0% against 85.8% for the earlier policy.
- `control/rl_scene.npz` stays as it was, and these policies are in `.tmp/topt/rl2/`.

## Redesign: a policy that bends its own guide, in three styles

**Status: proposed, nothing built.** Asked for: the policy sees more of the guide, decides itself
whether to bend it around obstacles or to go faster, and always comes as three policies: speed first,
middle ground and safety first.

### Why the current policy can't do this

It already may leave the guide, since its reward has no guide term, but it doesn't: it stays a median
4.8 cm from the guide, never beats the plan, and freeing the reward or hiding the near guide didn't
change that ("Can the policy find a better line" above). Leaving the guide is a decision made
implicitly, one attitude command at a time, so it never gets learned. A bend made explicit is one small
decision with a visible effect: "move this stretch 6 cm left" or "fly this stretch 10% faster".

### Recommended: two levels, a bender on top of a tracker

```
guide (20 ms planner)  -->  bender policy, 10 Hz  -->  bent guide  -->  tracker policy, 50 Hz  -->  attitude, thrust
                            sees more beads,           offsets and      sees the bent guide as
                            gates, poles, flags        speed scale      today's policy sees its guide
```

**The bend.** At 4 control points ahead (0.5, 1, 2 and 3 m along the guide), a sideways and a vertical
offset of up to 0.3 m in the frame that turns with the guide, plus one speed scale of 0.7-1.3 on the
guide's speeds. Offsets blend smoothly between control points. They are tied to places on the track,
not to the drone, so a bend stays put as the drone flies into it.

**Every bend starts from the drone's current state,** in three ways:

- **Where it starts:** the bent guide begins at the drone's position, not at the old guide, so it never
  asks the tracker to first fly back to a line it has already left.
- **Which way it leaves:** its first stretch leaves along the drone's velocity and blends into the
  control points, so it is a path the drone can follow from the way it is moving.
- **What it can reach:** each control point's offset is clipped to what the drone can still reach,
  given its speed and the time until it gets there: about `a_side * t^2 / 2` beyond the straight line
  its velocity points along, with `a_side` from the thrust left after gravity and the turn. A bend the
  drone can't fly is never handed to the tracker.

**The bender** decides every 5 control steps. Its inputs always start with the drone's current state:
its position relative to the guide (offset from the nearest bead), its velocity, its attitude and
angular velocity, its height, and the distances to the walls. All are in the frame that turns with the
guide, as the tracker's are, with no absolute x and y, so that no track can be recognized. Then more of
the guide: 24 beads up to 4 m ahead instead of 8, the speed at 8 places, and the current offsets. Then
the next two gates and the four nearest poles, each relative to the drone, with flags for whether each
has been revealed. It is trained with PPO on the race reward, with the tracker frozen. On level 3 this is what replanning would do, learned:
when a gate is revealed elsewhere, the bender bends towards it.

**The tracker** is today's scene policy, fine-tuned first on randomly bent guides with a reward for
progress along the guide it is shown. That reward matters: today's policy follows its guide only
because the guides were good. Shown random bends without such a reward, it would learn to ignore them,
and the bender would have no say. It keeps its crash penalty, so it still won't fly into a pole that
a bad bend points at.

**Three styles share one tracker,** and differ only in the bender's reward and in how the kept policy is
chosen:

| Style | Crash | Hit margin | Obstacle term | Time cost per step | Kept policy chosen by |
|---|---|---|---|---|---|
| `speed_first` | -20 | 2 cm | 0 | 0.2 | Lap time, from 70% finished |
| `middle_ground` | -20 | 3 cm | 0.02 | 0.05 | Today's score: finishing, then time |
| `safety_first` | -60 | 5 cm | 0.05, falling off faster | 0.05 | Finishing, then the smallest clearance |

The weights are a starting point. The three must end up ordered in the race: `speed_first` quickest,
`safety_first` finishing most. If two land on the same result, their weights move further apart.
`RLScene` picks one with `controller.style`, and `middle_ground` is the default.

### The alternative: one policy with both outputs

The scene policy gets the bend as extra outputs alongside the attitude and thrust, and sees its own bent
guide at the next step. It is a single network and a single training run per style. But nothing then
makes it follow its own bend, so it can learn to treat the bend as noise, and the bend loses its meaning.
A guide-progress reward fixes that and brings back the two-level design inside one network. I
recommend the two levels because the tracker is already proven, 249/256 on unshifted random tracks, and
each bender then has a small, clear job.

### Steps

1. **Bend and features** (about 2 hours): the bend in `scene_policy.py`, in JAX and NumPy, with tests that
   flight and training agree, that a bend starts at the drone's position along its velocity, that no
   control point exceeds what the drone can reach, and that a zero bend from a drone on the guide leaves
   the guide as it is. The bender's features in the same file.
2. **Tracker on bent guides** (about 1.5 hours): fine-tune `level1_random_tracks_fast_guides_60steps` on
   20 ms planner guides with random bends and a progress reward. Measure how closely it follows bends
   of each size. If it can't follow 15 cm reliably, the bender's range shrinks to what it can.
3. **Three benders** (about 2 hours, side by side): `speed_first_bender`, `middle_ground_bender`,
   `safety_first_bender`, first on unshifted random tracks, then with level 2's shifts and reveals.
4. **Real race** (about 1 hour): each style on real level 3, level 2 and unshifted random tracks, on
   fresh seeds, against the unbent tracker. A style replaces the default only if it beats it, and the
   three must be ordered as above.

### Risks

- **The tracker may not follow bends closely enough.** Gates leave about 8 cm of room, and the tracker
  stays a median 4.8 cm from guides it was trained on. Step 2 measures this before any bender is
  trained.
- **The bender learns slowly.** Its decisions act through a frozen tracker, 0.1 s apart. If PPO doesn't
  move it from a zero bend, it can start by copying the bend that the true layout's guide would need, on
  shifted tracks.
- **Starting each bend at the drone accepts its drift.** With a new bend every 0.1 s, the guide follows
  wherever the drone has drifted, and only the control points ahead pull it back. If drift builds up,
  the start blends from the drone's position back towards the previous bend over the first 0.5 m.
- **Three styles triple the training time.** They run side by side, at about 0.4M steps/s each.

### How it fits the level 3 plan

Step 0 and Step 1 of "Plan: real level 3" below don't depend on this and come first: they say how much a
correct guide is worth. The bender then takes the place of that plan's Step 2, the guides planned in
advance for every stage of reveals. If Step 1 shows that replanning alone almost closes the gap, the
two combine: replan at reveals and let the bender refine.

## Plan: an RL bender trained together with the tracker

**Status: built and tried on `safety_first` (2026-11-18): no gain, so the other two styles and the
selector over pairs were not trained. Results at the end of this section.** Asked for: a bender that is trained together with
the controller, not on top of a frozen one. It builds on "Redesign" above, which trained the tracker
first and the bender after. What carries over: the bend itself (offsets at control points ahead,
a speed scale) and the rule that every bend starts from the drone's position and velocity.

### Why train them together

- **Neither half is right alone.** A tracker trained on random bends learns to follow bends that no
  bender would make, and a bender over a frozen tracker can only ask for what that tracker already
  does. Trained together, the tracker learns to fly the bends that pay off, and the bender learns
  which bends the tracker can fly.
- **It replaces planning in training.** Flight now replans from the drone's state at every reveal and
  pass (`replan_from_state`), but training can't run the planner: its guides are planned from the
  start, in advance. A bend is computed inside the JAX step, from the drone's state, so in training
  it plays the part of that replan. In flight it refines the replanned guide.

### Structure

```
guide (training: staged, from the start; flight: replanned from the state)
   --> bender, every 5 steps (0.1 s): 4 control points x (side, up) offsets + 1 speed scale
   --> bent guide: starts at the drone, leaves along its velocity, offsets clipped to what it can reach
   --> tracker, every step: today's 86 scene features, computed on the bent guide --> attitude, thrust
```

- **Bender inputs:** the tracker's 86 features, plus 24 beads up to 4 m ahead, the next two gates and the
  four nearest poles with revealed flags, and the current bend. About 160 inputs, 2 x 128.
- **Tracker:** the same network shape as today, so a trained style can start it.

### One PPO run trains both

The two networks share one rollout and one update. Each has its own critic, its own advantage and its
own reward:

| | Acts | Reward |
|---|---|---|
| Bender | Every 5 steps, one semi-Markov transition, as the selector's | The style's race reward: gates, finish, crash, time cost, obstacle term |
| Tracker | Every step | The same race reward, plus progress along the **bent** guide and a small cost for leaving it |

The tracker's guide term is what keeps the bend meaningful. Without it the tracker learns to ignore
the bend, and the bender loses its say. Its weight falls over training (from 1 to 0.2), so that by the
end the tracker may cut a bend where the race rewards it.

**Keeping two learners stable:**

1. **Warm start.** The tracker starts from the style trained in Step 2, and the bender from zero
   output, a bend of zero. At the start the pair flies exactly as today.
2. **Bender first.** For the first 500 iterations only the bender learns, with the tracker frozen.
   Then both learn, the tracker at a third of the bender's learning rate.
3. **Bend size grows.** Offsets are capped at 5 cm at first, rising to 30 cm over 2000 iterations, so
   the tracker is never asked for a bend far from what it has learned.
4. **A check per network.** Every lap check also flies the new tracker on the unbent guide. If that
   gets worse by more than 5 finishes in 100, the tracker's learning rate halves.

### Three styles and the selector

Each style is one pair (bender and tracker), trained in one run with that style's reward from
`STYLES`. That keeps a style's risk in both halves. The selector of Step 2b then picks between the
three pairs, and is retrained on them (about 11 minutes).

### Steps

1. **Bend in JAX and NumPy** (about 2 hours), in `scene_policy.py`: control points, the start at the
   drone along its velocity, the reach clip, the speed scale. Tests: flight and training agree, a zero
   bend of a drone on the guide leaves the guide as it is, and no control point exceeds the reach.
2. **Joint trainer** (about 3 hours): `scripts/train_bender.py`, reusing `SceneEnv`, the staged
   scenarios and the semi-Markov PPO of `train_style_selector.py`. Smoke-test that a zero bender with
   a frozen tracker reproduces the Step 2 style's validation numbers.
3. **Three pairs** (about 1.5 hours, one at a time, since a staged run takes 12 GB of GPU memory):
   `joint_bender_tracker_speed_first`, `..._middle_ground`, `..._safety_first`.
4. **Selector over the pairs** (about 15 minutes).
5. **Real race** (about 1 hour): real levels 3 and 2 on the usual seeds, against today's default. In
   flight `RLScene` gets `controller.bender = true`. The bend runs after each replan from the state,
   and costs about 0.1 ms per step.

### Risks

- **The two learners chase each other.** The warm start, the bender-first phase and the slower tracker
  are there for this. If it still oscillates, alternate: 200 iterations of one, then of the other.
- **Bends in training and flight start from different guides.** Training guides were planned from the
  start, while flight's start at the drone. Near the drone the bend is the same either way, since it
  starts at the drone in both. Further ahead the replanned guide is already closer to right, so the
  bends flight needs are smaller than those training learns.
- **The bend may only learn to slow down.** The speed scale is the easiest output to use. If the pairs
  end up slower than today's styles with no more finishes, the speed scale gets a floor of 0.9.

### Results: built, tried on `safety_first`, no gain (2026-11-18)

**Built:**
- `Bend`, `make_bend`, `bend_offset`, `bender_features` and a ``bend`` argument of `scene_features` in
  `utils/scene_policy.py`, with tests that NumPy and JAX agree.
- `scripts/train_bender.py`, the joint trainer.
- `controller.bender` in `RLScene`: each policy as a pilot with its own bender, also under the selector.
- `--benders` in `scripts/train_style_selector.py` and `scripts/evaluate_reveals.py`.

The tracker always sees its offset from the guide as planned. A bend moves the points ahead and
scales the speeds.

**Two runs of 4000 iterations each (13 minutes), started from `rl_scene_safety_first.npz`:**

| Run | Validation, started tracker with no bend | Best validation | Real level 3 | Real level 2 |
|---|---|---|---|---|
| `safety_first` alone (Step 2) | | 43.7% at 7.19 s | 107/128 at 7.45 s | 54/64 at 6.43 s |
| Bend starts at the drone (`from_drone`) | 33.7% at 7.11 s | 43.8% at 7.72 s | 104/128 at 8.01 s | 50/64 at 7.23 s |
| Bend starts on the guide (the default) | 44.1% at 7.18 s | 44.6% at 7.19 s, iteration 0 | Not flown: the kept pair is the started tracker | |

- **Starting the bend at the drone breaks the warm start.** With no bend at all, the points ahead are
  pulled towards the drone, so the tracker reads a guide that it never trained on, and validation drops
  by 10 points. Joint training won that back, but the pair stayed slower and finished fewer real races
  than the style alone. With the offset from the planned guide also hidden, the started tracker fell
  to 28%. This run was stopped.
- **Starting the bend on the guide keeps the warm start exact,** but the bender found no bend that
  helps. Validation fell slowly as the bends grew to about 6 cm, from 44.6% to around 40-42%. The kept
  pair is iteration 0, which is the style unchanged.
- **Why the bender has nothing to add here:** in these scenarios the guide is already replanned at
  every stage of reveals. The bender knows no more than the planner did, so a better line has to come
  from flying, not from new information. Its PPO signal is also weak: 9 noisy outputs, each acting
  through the tracker for 0.1 s.
- **Not done, because the bender doesn't beat the style alone:** pairs for `speed_first` and
  `middle_ground`, and the selector over pairs. The code for both is in place.
- **Lost:** the weights of the run that started at the drone were deleted by a cleanup pattern that
  matched them. Its logs and race results are kept.

**What could still work:**
- Give the bender information that the guide lacks. For example, train on guides planned once from
  the start without staging, so that bending towards a revealed object is the bender's job.
- Use fewer outputs, such as one sideways offset at the next gate and a speed scale.
- Train it on the race reward alone, with no follow term for the tracker.

## Plan: real level 3, with a guide that follows what has been revealed

**Status: Steps 0, 1 and 2 done, with replanning in flight from Step 3 in a first form, and a selector that picks the style in flight (Step 2b). Results below.** Supersedes "Plan: perturbed arenas" below, which was written
before the 20 ms planner existed and left replanning in flight out for that reason.

**Goal:** the final controller, trained only on generated tracks, finishes real level 3 reliably: gates
and poles at random places, then moved by up to 15 cm and revealed only within 0.7 m. Level 2 is the
same problem on one fixed layout and is graded too, so it is tested alongside. Lap time comes second.

### What the race does on level 3

- The first observation reports a generated layout, which is all the controller knows at the start.
- The true poses differ by level 2's ranges: gates +-0.15 m in x and y, +-0.1 m in z, +-0.2 rad of yaw;
  poles +-0.15 m in x and y. An object's true pose is reported once the drone is within 0.7 m of it in
  x and y, and `gates_visited` and `obstacles_visited` say which ones those are.
- At 3-4 m/s, that is about 0.2-0.27 s before reaching a gate. A 15 cm sideways correction in 0.23 s
  needs about 5.7 m/s^2 on its own, before any turn.
- From the course's rules: grading uses levels 2 and 3 and real hardware. `env.control_mode` is one of
  the two settings a level file may change. `scripts/evaluate.py` scores level 2 and needs at least 50%
  of 20 runs. `scripts/sim.py` doesn't time the controller's constructor and only warns when a control
  step exceeds 20 ms. Real level 3 tracks are built from gates placed by hand, so their layouts may not
  match the race's generator.

### Where things stand

| Controller | Unshifted random tracks | Real level 2 | Real level 3 |
|---|---|---|---|
| `RLScene` with `rl_scene.npz`, the committed default | 243/256 at about 4.67 s | 0/64 | Never flown |
| `level1_random_tracks_fast_limit_guides`, solved guides with faster limits | 249/256 at about 4.46 s | Not flown | Never flown |
| `level1_random_tracks_fast_guides_60steps`, 20 ms planner | 248/256 at about 4.92 s | Not flown | Never flown |
| `clear_guides_level2_shifts`, trained with shifts | | 8/32 | Never flown |
| `RLRace`, trained on level 2's own track | | 64/64 at 4.66 s | Can't: one track |

**Blocker found while rescanning:** `config/level3.toml` still sets `control_mode = "state"`, while
`RLScene` returns an attitude command. Every result above overrode it to attitude, so the configured
final controller probably fails on level 3 as committed. Levels 0-2 already use attitude. Not run yet.

### Why the guide is the place to fix it

Every policy so far was trained on guides planned once, through the reported layout. After a reveal,
the guide still leads to where the gate was reported, and the only correction is what the policy does
on its own in about 0.2 s. `RLRace` manages that on one track it has learned by heart. A policy that
has never seen the track can't, which fits 0/64 on level 2.

The 20 ms planner can replan within a control step, or close to it: 17 ms median at 60 steps, 34-67 ms
at the 95th percentile. Training can't call it per step, but it can call it in advance for every layout
the drone could know about. So the guide can always be the plan through the layout as known so far,
in training and in flight alike. The policy then only has to handle the step when the guide jumps,
which is an offset from the guide, and it already learns those from noisy starts.

The 20 ms planner's policy is also the right one to build on. It finished as often as the solved-guide
policies and needs no 15 s solve at the start. A policy trained on one planner's guides loses finishes
on the other's, so training and flight must use the same planner.

### Step 0: make level 3 run as committed (minutes)

- Set `control_mode = "attitude"` in `config/level3.toml`, which the rules allow.
- Fly `pixi run python scripts/sim.py --config level3.toml` once, and add level 3 to
  `tests/integration/test_final_controller.py`, opt-in like level 0.

### Step 1: baselines and upper bounds, no training (about 1 hour)

All on fresh seeds: 128 of real level 3 from 9000, and 64 of real level 2 from 1000. Each row only
swaps how the guide is made, using hooks in the evaluation scripts:

| Flight | Guide | What it tells |
|---|---|---|
| Default `rl_scene.npz` | Solved once, at start | The baseline, never measured on level 3 |
| `fast_limit_guides` policy | Solved once, faster limits | Whether the faster policy holds up under shifts |
| `fast_guides_60steps` policy | 20 ms planner, once at start | Baseline for the 20 ms planner |
| `fast_guides_60steps` policy | 20 ms planner, again at every reveal | What replanning gives without retraining |
| `fast_guides_60steps` policy | 20 ms planner on the true layout, from the start | Upper bound: the right guide all along |

The last row cheats and is for evaluation only. If it isn't far above the first, the guide isn't the
main cause, and Step 2 changes to the inputs and the speed at unrevealed gates (Step 4) instead.

For every crash, record the object, how far it moved from its reported pose, and the time from its
reveal to the crash. That shows whether failures are reveals too late to react to, or wrong guides.
Also time each replan inside the race.

### Results of Steps 0 and 1 (2026-11-17)

**Step 0.** In state mode, `config/level3.toml` failed on its first step: `RLScene` returns 4 numbers
and state mode expects 13. It is now in attitude mode, and `test_final_controller.py` flies level 3
as configured. `RLScene` gained `guide_planner = "fast_runtime"` and `replan_on_reveal = true`, and
`scripts/evaluate_reveals.py` flies each guide mode.

**A bug that hid everything else.** Every policy trained on upright gates crashed whenever a gate was
tilted. Level 2 and 3 tilt gates by up to 0.05 rad of roll and 0.1 rad of pitch, and with them even a
flight given the true layout and every true pose from the start finished 0/64. The vertical part of a
gate's axis (inputs 58, 64 and 77) was always 0 in training, so its saved spread was 1.2e-6 and its
weights were never trained. A tilt of 0.1 rad became an input clipped to 10, through untrained
weights. `PathPolicy` now keeps any input whose training spread is below 1e-3 at its training value,
so the policy ignores the tilt, as it learned to. Found by removing one kind of shift at a time: with
yaw shifts alone, 61/64 finished; with roll and pitch alone, 0/64.

**Step 1, after that fix.** Seeds 9000-9127 of real level 3 and 1000-1063 of real level 2. The topt rows
fly `rl_scene.npz` and `level1_random_tracks_fast_limit_guides`, the runtime rows
`rl_scene_runtime_guides.npz`. "Hit after reveal" is the median time from the reveal of what was hit
to the crash.

| Guide | Level 3 | Level 2 | Hit after reveal, level 3 |
|---|---|---|---|
| `topt_planner`, once (the committed default) | 2/128 at 4.50 s | 1/64 at 4.08 s | 0.22 s |
| `topt_planner` with faster limits, once | 4/128 at 4.57 s | 0/64 | 0.21 s |
| Runtime planner, once | 5/128 at 5.23 s | 6/64 at 4.48 s | 0.26 s |
| **Runtime planner, replanned at every reveal** | **55/128 at 5.39 s** | **29/64 at 4.61 s** | 0.20 s |
| Runtime planner, true layout (cheat) | 120/128 at 5.11 s | 58/64 at 4.57 s | 0.21 s |
| Runtime planner, once, but the policy sees every true pose (cheat) | 5/128 at 5.22 s | 4/64 at 4.46 s | 0.26 s |
| Both cheats | 122/128 at 5.12 s | 59/64 at 4.57 s | 0.24 s |

With no shifts at all, the same script finished 63/64 with runtime guides and 60/64 with solved ones,
so the setup is sound.

- **The guide is the cause.** A guide through the true layout lifts level 3 from 5 to 120 of 128. Showing
  the policy every true pose without that guide changes nothing (5/128): it flies by its guide, not by
  the gates it sees.
- **Replanning at reveals recovers about half of that,** 55/128 and 29/64. What is left crashes about
  0.2 s after the reveal of what it hits, on objects that moved a median 15 cm: the corrected guide
  comes too late to follow. Poles cause 24 of the 73 level 3 crashes and next gates 38.
- **Replanning takes 34 ms median and 48 ms at the 95th percentile** here, over the 20 ms step, with 14
  races sharing the CPU. `scripts/sim.py` only warns, but Step 3's background thread is needed before
  hardware. About 5 replans per race on level 3 and 6 on level 2, and none failed its check.
- **Not yet in training:** the policy has never seen a guide jump, nor a gate revealed late. That is
  Step 2, and the gap between 55 and 120 is what it has to close.

### Results of Step 2: trained on guides that follow the reveals, in three styles (2026-11-17)

Built: `scripts/solve_staged_library.py`, `--staged` and `--style` in `scripts/train_scene_policy.py`,
and `replan_in_order` in `RLScene`. Revealed flags were left out for now, so the inputs stay at 86. Each
style trained 8000 iterations from scratch, on 6092 shifted scenarios with gate tilt. A staged run uses
about 12 GB of the GPU, so the three ran one after another, about 25 minutes each.

| Style | Validation lap check | Real level 3, replanned at every reveal | Real level 3, replanned in order | Real level 2, at every reveal | Real level 2, in order |
|---|---|---|---|---|---|
| `speed_first` | 45.6% at 5.57 s | 88/128 at 5.72 s | 75/128 at 5.71 s | 31/64 at 4.97 s | 31/64 at 4.97 s |
| `middle_ground` | 41.8% at 5.94 s | 93/128 at 6.20 s | 79/128 at 6.16 s | 34/64 at 5.15 s | 34/64 at 5.15 s |
| `safety_first` | 43.7% at 7.19 s | **107/128 at 7.45 s** | 92/128 at 7.47 s | **54/64 at 6.43 s** | 56/64 at 6.43 s |
| Untrained, for comparison | | 55/128 at 5.39 s | 47/128 at 5.33 s | 29/64 at 4.61 s | |
| **Best of the three per race** | | **120/128 at 6.13 s** | 106/128 at 6.18 s | **61/64 at 5.65 s** | 62/64 at 5.66 s |

Seeds 9000-9127 of level 3 and 1000-1063 of level 2, as in Step 1.

- **Training on reveals works.** Every style beats the untrained policy on level 3, `safety_first` by
  52 finishes.
- **The styles came out ordered as intended:** `speed_first` fastest, `safety_first` finishing most. Lap
  time costs about 1.7 s from one end to the other.
- **A selector has room to gain.** Taking the best style per race would finish 120/128 at 6.13 s on level
  3, against 107/128 at 7.45 s for the best single style. On level 2 it would be 61/64 at 5.65 s. This is
  an upper bound that knows each race's outcome, for whole races, not a result.
- **Replan at every reveal in flight, even though training replans in order:** level 3 gains 13-15
  finishes for every style, and level 2 is within 2. In-order replanning makes the drone wait for an
  object revealed out of order.
- **Poles end most of the remaining flights** of `speed_first` and `middle_ground` on level 2 (24 of 33
  and 30), about 0.25 s after their reveal. `safety_first` hits 8.
- **Replans took 32-34 ms median and 42-52 ms at the 95th percentile,** with 14 races sharing the CPU.

**Next: the agent chooses between the styles itself** (see "Redesign" above and the three-styles rule).
A selector would pick a style at events such as a reveal or a pass, with the three policies frozen
underneath, trained with PPO on one race objective. Its ceiling on these races is the best-of-three
row.

### Results of Step 2b: the agent chooses its style in flight (2026-11-18)

Built: `scripts/train_style_selector.py`, `scene_policy.selector_features`, and
`controller.style_selector` in `RLScene`, with the four files in `control/`.

**How it chooses.** The three styles of Step 2 stay frozen. Every 10 control steps (0.2 s) a small
network picks the style that flies the next 10. It sees:

- the 86 scene features,
- the style flying now, as one-hot,
- what each of the three styles would command now (12 numbers).

From the start until its first decision, `middle_ground` flies. In flight the first decision comes at the
first step.

**How it learns.** PPO with a categorical output, on the same staged scenarios as the styles. One
decision and the 10 steps it holds form one transition. The reward is the race's own and the same for
every style:

| Term | Value |
|---|---|
| Gate passed | +1 |
| Finish | +10 |
| Crash, at a 2 cm hit margin | -5 |
| Per control step | -0.02 (1 s costs as much as 1% of finishes) |

The kept selector scores best on validation laps by `10 * finish - time + gates`. Each style alone is
flown on the same validation laps for comparison. Training is 1500 iterations of 16 decisions in 8192
worlds, about 11 minutes.

**The discount sets the balance.** At 0.95 per decision (about 4 s ahead) a finish 7 s away hardly
counts, and the selector chose `speed_first` half the time. At 0.99 (about 20 s) it ranks whole laps.

| | Validation | Real level 3 | Real level 2 | Share of time per style on level 3 (speed / middle / safety) |
|---|---|---|---|---|
| `speed_first` alone | 45.0% at 5.58 s | 88/128 at 5.72 s | 31/64 at 4.97 s | |
| `middle_ground` alone | 49.5% at 5.95 s | 93/128 at 6.20 s | 34/64 at 5.15 s | |
| `safety_first` alone | 66.6% at 7.31 s | 107/128 at 7.45 s | 54/64 at 6.43 s | |
| Selector, discount 0.95 | 54.3% at 6.05 s | 103/128 at 6.16 s | 48/64 at 5.67 s | 53 / 30 / 17% |
| **Selector, discount 0.99 (installed)** | 62.9% at 6.55 s | **106/128 at 6.66 s** | **54/64 at 6.09 s** | 25 / 31 / 43% |
| Best of three per race (ceiling) | | 120/128 at 6.13 s | 61/64 at 5.65 s | |

The validation column uses a 2 cm hit margin, so it isn't comparable with Step 2's. Real races replan
at every reveal.

- **The installed selector matches `safety_first` on finishes and is faster:** 0.8 s on level 3 (106
  against 107 finishes) and 0.34 s on level 2 (54 against 54). It flies `safety_first` 43% of the time
  on level 3 and 71% on level 2, where the poles stand closer to the gates.
- **The ceiling is still 14 finishes away on level 3.** Poles end 18 of the 22 failed flights, as for the
  styles alone.
- **Cost in flight:** three style networks and the selector per step, about 40 microseconds in all,
  against 33 ms per replan.

Selector runs and results are in `.tmp/topt/level3/selector/` and `.tmp/topt/level3/results/`.

**Made the default on levels 2 and 3 (2026-11-18).** `RLScene` takes `SHIFTED_RACE_DEFAULTS` on any
race whose randomizations move gates or poles. Levels 0 and 1 keep the solved guide with
`rl_scene.npz`.

### Replanning from the drone's state (2026-11-18)

Built: `controller.replan_from_state` with `replan_gates` in `RLScene`, and `--replan_gates` in
`scripts/evaluate_reveals.py`. A replan starts at the drone's position and velocity and goes through
the next `replan_gates` gates (0 for the rest of the lap). It replans at every reveal, and at every pass
when the plan ends before the lap does. Flown with the installed selector, on the usual seeds:

| Replan | Real level 3 | Real level 2 | Failed plans (level 3 / level 2) | Replan time, median / 95th percentile |
|---|---|---|---|---|
| From the start, whole lap (the default) | **106/128 at 6.66 s** | **54/64 at 6.09 s** | 0 / 0 | 34 / 46 ms |
| From the state, rest of the lap | 97/128 at 6.63 s | 52/64 at 6.26 s | 7 / 4 | 31 / 41 ms |
| From the state, next 2 gates | 36/128 at 8.26 s | 6/64 at 8.09 s | 39 / 11 | 30 / 56 ms |
| From the state, next gate only | 24/128 at 8.33 s | 11/64 at 8.66 s | 991 / 36 | 63 / 131 ms |

Replan times are with 14 races sharing the CPU. Alone, a plan from the state took 13-16 ms on one
level 3 race.

- **Planning only to the next gate fails with today's policies**, for two reasons:
  - **The guide ends at the gate.** The policy reads the guide 3 m ahead, and past the plan's last gate
    it sees the straight run-out instead of the turn. Crashes into other gates dominate (38 and 61 on
    level 3).
  - **A plan from a drone near a gate often fails its check.** The plan is then retried at every step,
    so with the next gate only, 991 plans failed and the median time doubled.
- **From the state through the rest of the lap is close, but 9 finishes worse on level 3.** The styles
  were trained on guides planned from the start, which the drone joins partway along. A guide that
  starts at the drone is new to them.
- **The default is unchanged.** Planning from the state needs a policy trained on such guides. The
  joint bender plan below would provide one: its bend starts at the drone in training too. Without the
  bender, it needs guides planned from mid-flight states in the training library.

### Step 2: train with guides that follow the reveals (about 3 hours)

**Library.** For each training track, draw 4 sets of shifts. Order the track's 8 objects by where the
nominal guide enters their sensor range. Stage `k` is the layout with the first `k` objects at their
true poses and the rest as reported. Plan stages 0-8 with the 20 ms planner from the start, 9 guides
per track and draw. On 1,500 tracks that is about 54,000 plans, a few minutes on 14 cores, and
around 1 GB of guide arrays on the GPU. A stage whose plan fails its check keeps the stage before it.

**Training.** Shifts come from the 4 draws instead of a fresh draw per episode. A world flies the
guide of the longest stage whose objects have all been revealed. Its place on the new guide is found
with the usual nearest bead, within the current pair of gates. An object revealed out of order shows
in the inputs at once, and joins the guide when the stage before it completes. Flight uses exactly this
rule, so it never sees a guide training didn't.

**Inputs.** Flags for whether the next two gates and the two nearest poles have been revealed. Without
them a reported gate and a confirmed one look the same.

Three runs side by side, 6000 iterations, `--randomize 1` from the start, with matching names:

| Run | Guides by stage | Revealed flags |
|---|---|---|
| `staged_guides_with_seen_flags` | Yes | Yes |
| `staged_guides_only` | Yes | No |
| `seen_flags_only` | No, stage 0 throughout | Yes |

Choose the kept policy by a lap check on validation tracks with shifts and staged guides, as in flight.

### Step 3: replan in flight (about 1 hour)

- `RLScene` compiles the 20 ms planner in its constructor, plans stage 0 there, and plans the next stage
  when one completes. The guide's crossings stay in the full race order, so `passed` still indexes them.
- A replan over budget at the 95th percentile can run in a background thread, with the old guide
  flown until the new one is ready. Training then switches stages 0-3 ticks late, at random, to match.
- A plan that fails its check keeps the previous guide, as in training.

### Step 4: only if Step 3 is still short

One at a time, in this order:

1. **Caution at unrevealed gates:** give the 20 ms planner the speed cap before unrevealed gates that
   `topt_planner` has (`unseen_speed`), so arrivals leave time to correct. This costs lap time.
2. **Episodes around reveals:** start 1-1.5 m before an unrevealed gate and end after the next pass or
   two, so reveals make up more of training.
3. **Short memory:** the last few observations as input.
4. **Curriculum:** continue from Step 2's policy at 25-50% of the shifts, then raise them.

### Step 5: test, and only then change the default

On seeds no step tuned on:

| Test | Bar |
|---|---|
| Real level 3, 256 seeds from 9500 | At least 85% |
| Real level 2, 128 seeds from 2000 | At least 90%, as the old plan's 58/64 |
| Unshifted random tracks, 256 seeds | At least 240/256 |
| Level 0 | 64/64 |
| `scripts/evaluate.py` and `scripts/sim.py --config level3.toml` | Pass, as configured |

The bars are a proposal and may move after Step 1's baselines. The new policy replaces `rl_scene.npz`
only if it clears them all, and then README, `docs/Architecture.md` and the levels' configs follow.

### Left out

- **Lap time.** Faster limits didn't help the 20 ms planner, see "Faster guides" above. That comes
  after finishing.
- **Real hardware.** Hand-placed layouts may not match the generator, and on hardware a 20 ms control
  step may be firmer than in simulation. Step 3's timing figures are the input to that.
- **Separate agents for low and high curvature.**

### Risks

- **The jump when a guide switches** may unsettle the policy at the worst moment, just before a gate.
  The delay in Step 3 and the offset inputs are meant for this. Step 1's replanning row shows how bad it
  is without training.
- **Too little time to react.** If most crashes come within 0.1 s of a reveal, no guide helps, and
  Step 4.1 is the lever.
- **The stage rule is an approximation.** Reveals out of order wait for their stage, so the guide can lag
  what the inputs show.
- **GPU memory and time:** 9 guides per track and draw multiply the library, which may force fewer draws.

## Plan: perturbed arenas (levels 2 and 3)

**Status: superseded by "Plan: real level 3" above, before any step started.**

**Goal:** on arenas where gates and poles move from their reported poses, a policy trained only on
generated tracks finishes at least 58/64 on the real level 2, with no level 2 layout in training or
in choosing the policy. Then measure real level 3. Lap time comes second until that holds.

### What we know

| Policy, trained on generated tracks | Shifts in training | Validation lap check | Real level 2 |
|---|---|---|---|
| `clear_guides_no_shifts` | None | 60.8% (no shifts) | 0/64 |
| `level1_random_tracks_fixed_guides` | None | 81.4% (no shifts) | 0/64 |
| `clear_guides_level2_shifts` | Level 2's, from the start | 16.9% (with shifts) | 8/32 |

`randomized_gate_distance_margin`, trained on level 2's own track, gets 64/64 at 4.66 s, so these shifts
can be flown. The physics leaves little room: the opening gives about 11 cm of free space either side
of centre (20 cm less the 8.6 cm sphere), a gate can move 15 cm, and it is revealed about 0.2-0.27 s
before arrival.

### What may be causing it

None of these has been tested on its own:

1. **The policy can't tell what it knows.** A nominal gate and a revealed one look the same in its
   inputs, so it can't fly cautiously only when a gate is still unconfirmed. The race does report
   `gates_visited` and `obstacles_visited`.
2. **The guide stays wrong after a reveal.** The guide says one thing and the true gate another.
3. **The guide flies too fast at unseen gates.** Every plan here sets `unseen=False`, so none uses the
   speed cap and arrival cone the MPCC plans with.
4. **It was asked to learn too much at once.** Shifts were on from the first iteration, and crashes
   then end most episodes early.
5. **Too little practice at reveals.** Each episode has only a few, and full episodes dilute them.

### Step 1: baselines, no training (about 30 minutes)

- Real level 3 with `rl_scene.npz`, 64 seeds, which has never been measured. This needs only a config
  check that the first observation reports the nominal layout on level 3.
- Lap checks of the current policy with one shift at a time, at 25%, 50% and 100% of level 2's size:
  gate position, gate angle, pole position. This shows which shift does the damage.

### Step 2: clean libraries (about 40 minutes)

The current libraries have guides leaving the arena. Add the arena's walls to `RANDOM_TRACK_CONFIG` as a
constraint, so that these tracks get a valid plan and aren't dropped, and re-solve. Flight then gets
the same protection.

### Step 3: find the main cause (about 1 hour, three runs side by side)

Same libraries, 4000 iterations, shifts on from the start, compared with `clear_guides_level2_shifts`.
Each changes one thing:

| Run | Change | Tests cause |
|---|---|---|
| `shifts_with_seen_flags` | Inputs say whether the next gates and nearest poles have been revealed | 1 |
| `shifts_with_bent_guide` | The guide moves with what is revealed | 2 |
| `shifts_with_full_sensor` | Everything revealed from the start, with the guide unchanged | How much late reveals cost at all |

**The bent guide must run in training and in flight alike,** and `PathBender` can't: it is NumPy and
solves a large system per reveal. So a new closed-form bend, in `scene_policy.py`, computes only the
beads the policy looks at:

- each bead moves by a blend of the moves of the gates behind and ahead of it, rotation included;
- that blend starts from where the drone was at the reveal, so the guide never jumps beneath it;
- beads near a revealed pole are pushed out to clear it.

**Check first:** how often bent guides stay clear, compared with `PathBender`. It is an approximation,
and a pole pushed onto a gate's approach may not be solvable this way.

### Step 4: build on what worked (about 1 hour)

Combine what helped in Step 3, then compare two ways of reaching level 2's shifts:

| Run | How |
|---|---|
| `shifts_from_start` | Level 2's shifts from the first iteration |
| `shifts_gradual_from_level1` | Continues from `level1_random_tracks_fixed_guides`, raising shifts through 25%, 50%, 75% and 100% as validation holds |

The second needs a new `--init` option. Inputs must match, so a new input means a short retrain at no
shifts first, or widening the network's first layer.

### Step 5: only if it is still short of the target

Try these in order, one at a time:

1. **Cautious guides:** solve libraries with `unseen=True` on first passes. This should slow arrivals at
   unseen gates, at a cost in lap time.
2. **Local episodes around reveals:** start 1-1.5 m before an unrevealed gate, end after the next pass or
   two, and score each scene type. This also tests the local training idea directly.
3. **Short memory:** the last few observations as input, so it can tell a just-revealed correction from
   drift. The network stays small.

### Step 6: test

Real race, in this order, with nothing tuned on the test seeds:

| Test | Bar | To compare with |
|---|---|---|
| Lap check, validation tracks with shifts | At least 80%, or stop and diagnose | 16.9% |
| Real level 2, seeds 1000+ | **At least 58/64** | `RLRace` 64/64 at 4.66 s, MPCC 54/64 at 4.40 s |
| Real level 3, fresh seeds | Report; no bar yet | Step 1's baseline |
| Unshifted random tracks | Should not fall far below 243/256 | Level 1 result |
| Best policy, 256 fresh seeds, then `scripts/evaluate.py` | Confirmation | - |

### What is left out

- **Solve time at the start of a race:** about 15 s on a new track, unchecked against the rules. Fast
  guides aren't ready to replace it, at 60% against 78% under a policy not trained on them. The bend
  above is meant to be fast enough for flight either way.
- **Replanning in flight**, unless bending fails.
- **Separate agents for low and high curvature**, unless scene scores show one policy can't do both.

### Risks

- **Possibly no single main cause,** in which case Step 3 will show small gains from each and Step 4 has
  to combine them.
- **Slower laps than 4.66 s,** at least at first.
- **Overfitting to level 2's shift ranges:** level 3 uses the same ranges, so this shouldn't matter now.
- **Shared GPU:** three side-by-side runs are slower, and another session may be training too.

## Fast guide planner: a guide in one control step

`topt_planner` needs about 8 s per initial guess on a random track. The question was whether a usable
guide fits in a 20 ms control step. `utils/fast_runtime_planner.py` is our own: closed-form bang-bang segments
between waypoints, with gradient steps on `total time + w * sum(max(0, clearance - distance)^2)`.

**Measured on 200 seeds per source, including layouts `topt_planner` dropped.** Thrust limit 15 m/s^2.
Times are one pinned CPU core, the whole plan including the check and any retries, with machine load
of 2-4 out of 16 and GPU training running, so not an idle machine. In each pair, level 3 comes first,
then our generator.

| Planner | Clear and inside the arena | Lap vs `topt_planner` (median) | Time, median | Time, 95th percentile | Clear within 20 ms |
|---|---|---|---|---|---|
| `topt_planner` | 78.5% / 76.5% kept, before the arena | 1.00 (3.99 / 3.88 s) | Seconds | Seconds | 0% |
| 0 steps | 58% / 38.5% | 1.55 / 1.59 | 17 / 19 ms | 22 / 21 ms | 50.5% / 38% |
| 30 steps | 100% / 93.5% | 1.26 / 1.31 | 11 / 11 ms | 22 / 44 ms | 91% / 79.5% |
| 60 steps | 100% / 93.5% | 1.22 / 1.27 | 17 / 17 ms | 34 / 67 ms | 91% / 79% |
| 120 steps | 100% / 93.5% | 1.20 / 1.25 | 29 / 29 ms | 58 / 115 ms | 0% |
| 300 steps | 100% / 95% | 1.16 / 1.21 | 64 / 64 ms | 127 / 250 ms | 0% |
| 60 steps, thrust limit 14 | 100% / 94% | 1.32 / 1.38 | 17 / 18 ms | 34 / 71 ms | 90.5% / 77.5% |

Lap ratios are on tracks both planners solved. The 95th percentile is high because a failed check
retries, with up to 4 tries, and these are one full budget each. The first compile takes seconds and
happens before the race.

**What it costs:**

- **Lap time.** About 22-27% slower at 60 steps, and more steps only slowly close that. Path lengths
  match `topt_planner`, so the time is lost in speed, not in the route.
- **Thrust.** Drag isn't in the segments. At a limit of 15 the median plan peaks at 16.6 m/s^2 and spends
  27% of its time above 15. A limit of 14 brings that to 15.5 m/s^2 and 3-4%, but laps are slower. A few
  plans peak well above this, up to 25 m/s^2.
- **Attitude.** Bang-bang acceleration jumps: the guide's attitude changes up to 0.93 rad between two
  beads, against 0.29-0.33 rad for `topt_planner`. Training starts episodes from this attitude, so it
  would need a filter. Filtering only the attitude leaves the path and its clearance unchanged. Not
  built or measured.
- **Hard generated tracks.** 6.5% stay unclear after all 4 tries, by up to 10 cm. They are rejected.

**What fixed things, in order of effect:**

- Dropping the points before and behind gates: 1.34 to 1.21 at 60 steps, on 40 tracks. Each short segment
  brakes on its own.
- Limiting thrust as a whole rather than per axis. A per-axis box allowed 19 m/s^2 on diagonals and only
  8 along an axis.
- Sharing samples by each segment's current length. Before this, a segment that grew could step over a
  pole by 7 cm, as `topt_planner` used to between its nodes.

**Flying it, without training.** `level1_random_tracks_fixed_guides`, a snapshot taken mid-training, was
flown on 157 level 3 validation tracks. It has only seen `topt_planner` guides, so this is a lower bound
for fast guides. 1024 worlds, 2 seeds, on CPU.

| Guide | Finished | Lap |
|---|---|---|
| `topt_planner` | 78% | 5.00 s |
| Fast, 60 steps, thrust limit 14 | 63% | 5.85 s |
| Fast, 60 steps, thrust limit 15 | 60% | 5.72 s |
| Fast, 300 steps, thrust limit 15 | 52% | 5.48 s |
| Fast, retimed to the `topt_planner` lap | 38-40% | 5.10-5.16 s |

Faster hints crash more: retiming keeps the path and only raises the hinted speed, and it makes things
worse. So slow speed hints are not what costs the finishes. The policy does follow the speed it is
shown, which supports the design that speed is only a hint and must be learned as such. What this
doesn't separate is whether the remaining gap comes from the path's shape or from the policy never
having seen such guides.

**Verdict:**

- As a route within a control step: yes, on the race's own generator. Clear on all 200 level 3 layouts,
  including the 43 `topt_planner` dropped, 91% of them on the first try within 20 ms.
- As a replacement for solved guides under a policy already trained: no, 60% against 78%.
- As guides to train on: yes, at level 1 difficulty. See below.

**Training on fast guides.** `level1_random_tracks_fast_guides_60steps` uses the same settings and
tracks as `level1_random_tracks_fixed_guides`, 8000 iterations with `--randomize 0`. Only the guides
differ: 60 steps, thrust limit 15, velocities smoothed over 0.15 s. That smoothing brings the attitude
jump from 0.93 to 0.27 rad, against 0.29 rad for solved guides, and doesn't change positions or
clearance. 1523 of the 1539 training tracks and all 157 validation tracks had a clear fast plan, and
the libraries took 47 s on two cores.

| Policy | Guide | Validation lap check | Real race, seeds 7100-7227 | Real race, seeds 8000-8127 | Together |
|---|---|---|---|---|---|
| `level1_random_tracks_fixed_guides` | Solved | 81.4% at 4.73 s | 120/128 at 4.72 s | 123/128 at 4.62 s | **243/256 at about 4.67 s** |
| `level1_random_tracks_fast_guides_60steps` | Fast | 71.3% at 5.02 s | 126/128 at 5.00 s | 122/128 at 4.84 s | **248/256 at about 4.92 s** |
| `level1_random_tracks_fixed_guides` | Fast | | | 111/128 at 5.32 s | |

The real race is level 1's conditions on the race's own generator, with no shifts, and
`scripts/race_fast_guides.py` plans inside it. All 256 plans passed their check, 245 on the first try
and 11 on the second.

- **Finishes are about equal.** 7 flights only the solved-guide policy finished and 12 only the
  fast-guide policy finished, which a sign test can't separate from noise (p = 0.36). Seeds 7100-7227
  had been used as a comparison before, so 8000-8127 is the clean test: 122 against 123.
- **Laps are about 0.25 s slower,** on flights both finished, and faster on only 13% of them.
  Fast-guide flights end about 0.05 s after their plan's lap, while solved-guide flights end about 0.7 s
  after theirs, so the lap now seems limited by the hint. Higher limits didn't help, see "Faster guides" above.
- **Fewer episodes leave the arena in training:** 1.4% against 8.4%. Fast guides stay inside and 14% of
  the solved training guides don't, which fits a link but doesn't prove one.
- **Policy and guide belong together.** The fast-guide policy finishes 1-6% of the level 2 lap check on
  its solved guide, where the other finishes about 99%. The solved-guide policy loses 12 finishes on
  fast guides.
- **The validation lap check misled here.** It ranks fast guides 10 points lower, while the race shows no
  difference. Both policies score well above it in the race. It counts a hit from 3 cm and the race only
  on contact, and fast guides pass closer, which may explain this but wasn't tested. Use the race to
  compare guides.

**Not tested:** moved gates and poles, which remain the blocker for levels 2 and 3, and planning time
inside the race, which is only known from the table above. No controller uses fast guides by default.

**A finding about solved guides.** 12-18% of the kept `topt_planner` guides in `fixed_*` leave the arena,
by 6-8 cm at the median and up to 48 cm at worst. `_is_clear` has its arena check back for future
libraries, as recorded above, but `topt_planner` itself has no walls, so those tracks get dropped.

Run outputs are in `.tmp/topt/rl2/fast_planner/`. Reproduce with `scripts/compare_fast_runtime_planner.py`,
`scripts/fly_fast_guides.py`, `scripts/solve_fast_library.py` and `scripts/race_fast_guides.py`.

## Step 5: race

1. Train on real bent paths at level 2, with gates, poles and collisions, and randomize mass, inertia
   and gate and pole shifts. Bench training has no mass randomization yet.
2. Give it a faster reference: today's plan and speed limit would cap it. The 18 m/s^2, tilt 1.4 rad
   plan comes out at 2.7-3.3 s from handover. Treat speed as a hint, not a limit.
3. Add a controller class using the policy with the bender's path, and keep take-off as it is.
4. Check it runs inside the 20 ms control step. The network should be far below that.

## Step 6: compare

- Tune on 64 seeds per setting. Confirm on 256 fresh seeds, since 64 can't separate close settings.
- Compare on the same seeds against the current version and `C3` (74% at 4.34 s).
- Beware of seed variation: the same controller has scored 81-88% across fresh sets.
- `debug_run.py` writes logs readable by the replay site; run `scripts/evaluate.py` for the official
  score, which hasn't been done yet.

## Decisions still open

- `C3` as the MPCC default, or stay on the current version.
- Whether the first-run library solve of about 10 s is allowed in the competition. Check the rules.

## Decided

- We build and train our own RL controller. Another team's policy, code or weights are not used.
