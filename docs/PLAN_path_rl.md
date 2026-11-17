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

## Plan: perturbed arenas (levels 2 and 3)

**Status: proposed, not started.**

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
guide fits in a 20 ms control step. `utils/fast_planner.py` is our own: closed-form bang-bang segments
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

Run outputs are in `.tmp/topt/rl2/fast_planner/`. Reproduce with `scripts/compare_fast_planner.py`,
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
