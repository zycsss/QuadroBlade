# Incremental replanning: what is left to do

This page is the durable plan for the work that follows the A* + MPCC controller. It records what has
been measured, what is decided, and what each branch is for, so the reasoning does not have to be
reconstructed later.

## Where things stand

Merged into `main` from `A-star-mpcc`:

- **A* routing** for the obstacle detour (`control/astar.py`, `control/path_planning.py`).
- **Run instrumentation**: `debug_run.py` writes a `.npz` + `.json` per episode with the drone state,
  the controller diagnostics and every plan the controller committed to
  (`scripts/debug_run.py`, `control/my_controller.py`).
- **Log-driven figures**: `path_plot.plot_run` draws a run from its log, including the failure point
  and a zoomed pole or gate-plane view (`utils/path_plot.py`).
- **The characterisation instrument**: analytic scenarios, the scenario controller, the metrics and
  the measured envelope (`control/scenario_path.py`, `control/scenario_controller.py`,
  `utils/characterization.py`, `scripts/characterize.py`, `docs/control/characterisation.md`).

Measured baseline on `config/level2.toml`, seeds 0-19: **13/20 finished, mean 8.17 s**.
Re-measured at `caa8855`, the branch point of `seeded-replan`, on this machine: **14/20 at 8.99 s** on
seeds 0-19, **27/40 at 9.03 s** on seeds 0-39. The 13/20 is the pre-`clearance-fix` number.

## Measured: the seeded re-plan (branch `seeded-replan`)

The complaint this answers is that committed paths are not smooth and sometimes go backwards then come
back. It is the same complaint `fdc43c0` diagnosed and left unfixed, and the numbers below are that
diagnosis made quantitative: on the 200 samples of every plan the controller logs, **27 % of re-plans
turned by more than 120 deg in one 6 cm step, 42 % by more than 90 deg, and 133 deg of the mean worst
turn sat inside the first 0.35 m** — at the anchor.

The cause is that a re-plan was anchored where the drone happened to be and then stepped *toward the
next gate*. Two consequences: the first block of the first gate passage is laid `approach_distance` =
0.6 m before the gate, so a drone that is already 0.3 m into that approach gets a block *behind* it
and the polyline reverses within centimetres of the start; and the leaving direction is the direction
to the gate, not the direction the drone is flying, so a drone that is turning can be asked to turn
back.

The re-plan is now seeded (`PathSeed`, `PathPlanner.plan(seed=...)`):

1. the anchor is the projection of the *predicted* pose (`pos + vel / freq`) onto the **old path**, so
   the new path starts on the old one rather than beside it;
2. it leaves along the old path's **tangent**, not toward the next gate;
3. the part of the first passage behind the seed step is dropped, and the seed step is clamped so it
   never overshoots that passage (`PathConfig.trim_seeded_passage`);
4. the plan is a B-spline whose first three control points are the anchor and the drone's measured
   velocity and acceleration, so the start cannot be dragged away by a later stage: the pin is a
   boundary condition, not a correction applied afterwards.

That last point used to be a different mechanism. Before the B-spline the planner drew a natural cubic
spline, arc-length reparameterized it, and then moved the first 0.25 m of the finished path back onto
the old one with a smootherstep blend. It had to run on the dense samples rather than on the waypoints,
because the reparameterization refits the spline and resamples it up to twenty times, and its first
tangent drifts a little on every sweep: measured on a real re-plan, a start that was straight along the
old path came out **100 deg off it**, a fold recreated by the stage that was supposed to remove it. The
B-spline removes the need for that stage entirely - the first control points *are* the state - and the
pin, `_pin_anchor`, is gone.

Level 2, seeds 0-39:

| | before | after |
| --- | --- | --- |
| turns > 120 deg | 75 in 275 re-plans (27 %) | **4 in 285 (1.4 %)** |
| turns > 90 deg | 115 in 275 (42 %) | **6 in 285 (2.1 %)** |
| worst turn, mean | 149.5 deg | **57.6 deg** |
| worst turn in the first 0.35 m, mean | 133.1 deg | **40.5 deg** |
| success | 27/40 | 26/40 |
| mean lap time | 9.03 s | **8.71 s** |

So the geometry was not bought with pace: one seed in forty is inside the noise of a 40-seed binomial
(sd ~ 3.5 seeds) and the lap is 3.5 % faster. The failures change identity — the t ~ 1.3 s pole strikes
(seeds 9, 12, 14, 27) are gone, while seeds 7, 8, 28, 29, 31, 32, 38, 39 now end later, most of them on
a gate frame.

Two variants were measured, and the difference between them is the pin's target. Following the **old
path** (what the code does now) gives 26/40 at 8.71 s and the numbers above; following the **straight
tangent** gives 27/40 at 8.75 s with 4 folds in 289, worst turn 60.6 deg and anchor turn 42.9 deg. One
seed and one degree apart, so the old path is kept: it is the literal reading of "the new path starts
from the old path", and it inherits the old path's curvature at the anchor instead of a straight line's.

The trim is what buys the geometry. With `trim_seeded_passage = False` (pin only) the 20-seed screen is
**11/20 with 46 folds**, worse than the baseline on both axes: the untrimmed passage still folds, and
the pin only covers the first 0.25 m of it.

**What is left, in order:**

- **The remaining 4 folds** are all the case where the drone is already past the gate plane when the
  reveal arrives. The trim has to keep one approach sample or the gate would never be crossed, and that
  sample is behind the anchor. This needs the gate pass re-planned, not the anchor.
- **The 30-70 deg turns that remain at 0.25 m are not folds**: they are the correction biting where the
  pin lets go. Softening them means a longer `anchor_pin_length` (later correction) or a real
  trajectory optimisation, not a better anchor.
- **Mid-path reversals are untouched.** 3 of the 40 seeds still contain a 160-173 deg corner *between*
  two passages. That is the inter-gate turnaround the track asks for (order 1,2,3,4,2, gates facing
  different ways), and the fix is the fillet of radius `v^2 / a_lat` at the block join that `fdc43c0`
  already identified as belonging with the B-spline path, not an anchor fix.
- **The trim costs gate centring**, which is why the failures that remain are mostly `HIT_GATE`. A
  reveal at 0.7 m asks for a 0.21 m lateral correction that the drone needs 0.46 s to make, and a path
  that starts at the drone spends the first 0.3 m of that budget doing nothing. Shortening the deferral
  (`replan_defer_in_passage`) or capping the approach speed in front of an unrevealed gate (P1.3 below)
  attacks the same term from the information side.

## Measured: the failure classes, one by one (branch `seeded-replan`)

The 14 failures of the 40-seed screen were classified from their logs, at the **true** poses, because
the verdict itself is computed against the nominal ones (`classify` in `scripts/debug_run.py` names the
wrong object in 4 of the 14). The split:

| class | seeds | cause |
| --- | --- | --- |
| gate frame of a gate already passed | 7, 11, 17, 28, 29, 38, 39 | `_avoid_gate_frames` only ever saw the remaining passes |
| gate frame, tangential graze | 32 | the same stage only reacted to plane *crossings* |
| gate frame, pass completed on the same tick | 17 | the re-plan still contained the pass being flown |
| pole strike, drone 6-9 cm inside the plan | 6, 8, 9, 33 | contouring weight at a pole was half the gate weight |
| gate pass 0.14 m off centre, plan 0.08 s old | 13, 31 | the reveal lands at 0.7 m, the correction needs 0.46 s |

Four fixes, in that order:

1. **Every gate of the track is given to the frame-avoidance stage**, with the number of legitimate
   crossings per gate (`PathPlanner.plan` builds `crossing_count` from the remaining sequence). A gate
   the drone has already flown through is still a solid ring, and on a track that doubles back the
   route to the next gate crosses its plane - through the opening, because A* only models the posts.
2. **The test is a slab around the gate plane**, not a sign change (`gate_frame_slab` = 0.13 m), so a
   path that comes close and turns away is caught too. The push is capped at `gate_frame_step` = 0.15 m
   per pass over five passes, because clearing 0.5 m in one go left a kink where its taper ended, and
   the samples of the gate's own passage plus the first 0.3 m (the drone's own position) are protected.
3. **A pass the drone is completing is not re-flown** (`_completed_pass`): if the anchor is at or just
   past the first pass's plane, inside the opening and heading through it, the trim does not force a
   crossing (`crossing_required=False`). Forcing one is what made seed 17 reverse through the gate it
   had just left.
4. **The approach to a gate that is not yet seen is capped at 1.5 m/s** (`reveal_speed`,
   `reveal_window` = 1.2 m, held for `reveal_hold_time` = 0.5 s after a correction), and **the pole
   contouring weight is raised from 0.2 to 0.4**, matching the gate weight: a pole is a hard obstacle,
   not a corner to cut. The cap is disabled on a track without pose randomisation, where there is
   nothing to correct.

Level 2, seeds 0-39, same machine and session:

| | before | after |
| --- | --- | --- |
| success | 26/40 (65 %) | **31/40 (78 %)** |
| mean lap time | 8.71 s | 8.88 s |
| turns > 120 deg | 4 in 285 re-plans | **1 in 302** |
| turns > 90 deg | 6 in 285 | **3 in 302** |
| worst turn, mean | 57.6 deg | **52.9 deg** |
| worst turn in the first 0.35 m, mean | 40.5 deg | **40.2 deg** |

Five more finishes for 2 % of lap time, and the geometry does not regress. Level 0 and level 1 are
unchanged: 2/2 at 8.52 s and 2/2 at 8.50 s.

**What is left, all with the same signature - a gate pass 0.15-0.28 m off centre:**

- seeds 13, 29, 31, 39 fail at gate 2 with the plan crossing dead centre and the drone 0.11-0.16 m off
  it. The reveal arrives at 0.7 m and the 0.46 s correction is the whole distance left; even capped at
  1.5 m/s the drone reaches the plane before it has converged. This wants the correction to start
  earlier (`P1.2` uncertainty inflation), not another path tweak.
- seeds 4, 7, 8 clip a frame 0.28-0.38 m outside the opening, i.e. clipping the *outer* part of the
  ring while flying past a gate, and seed 9 still strikes pole 0 on the take-off leg. Seed 33 is the
  pole case that the weight change did not save.

## Measured: the B-spline trajectory, and the geometry relaxation (branch `seeded-replan`)

The path was a natural cubic spline in arc length, so the drone's state entered it only as a position
(the anchor) and a direction (a pin onto the old tangent). Two things followed, both visible in the
figures: the plan could contain curvature no speed can fly - the 0.19 m hairpin around pole 3 in seed
8, which needs 12 m/s² at 1.5 m/s - and the controller could only answer by cutting the corner.

`control/bspline_path.py` replaces that with a **clamped cubic B-spline in time**, following
Fast-Planner (Zhou et al., RA-L 2019):

- velocity and acceleration are B-splines of the same control points, `v_i = (Q_{i+1} - Q_i)/dt` and
  `a_i = (Q_{i+2} - 2Q_{i+1} + Q_i)/dt²`, so the convex hull property turns the dynamic limits into
  bounds on control points;
- the first three control points **are** the drone's state, `Q_0 = p`, `Q_1 = p + v dt/3`,
  `Q_2 = p + 2v dt/3 + a dt²/6`, so the plan starts with the measured velocity and acceleration
  instead of an approximation of the old path's tangent - this is what `PathSeed.velocity` and
  `.acceleration` now carry, taken from the observation and a finite difference of it;
- `retime` stretches the knot spacing until the speed and acceleration bounds hold, which is the
  cheapest form of Fast-Planner's time adjustment (Sec. V).

The controller is unchanged: `BsplinePath` implements the `ArcPath` duck type (`length`, `pos`,
`tangent`, `curvature`, `closest`), with an arc-length table so `theta` stays a length in metres.

Level 2, seeds 0-39, same machine and session:

| | arc path | **B-spline** |
| --- | --- | --- |
| success | 31/40 | **32/40** |
| mean lap time | 8.88 s | 8.86 s |
| worst turn, mean | 52.9 deg | **52.7 deg** |
| worst turn in the first 0.35 m, mean | 40.2 deg | **38.0 deg** |
| turns > 120 deg | 1 in 302 | 1 in 301 |

A wash on success, one degree better at the anchor, and it is the representation the next step needs
(local support for repair, dynamic limits at plan time).

### The relaxation, after the clearance projection was fixed

The first version of the relaxation projected *control points* onto `safety_distance` and lost three
finishes. Measuring the trade-off on the seeds it changed (`logs/relax_tradeoff.py`) showed why, and it
was not the gate passages (crossing offsets 0.000-0.011 m either way):

| | relaxation off | relaxation on, first version |
| --- | --- | --- |
| plan clearance to poles | 0.173-0.187 m | **0.145-0.149 m** |
| gate crossing offset | 0.000-0.011 m | 0.000-0.011 m |

The curve between control points lies inside their convex hull, and the hull of points outside a disc
still cuts the disc, so the control points satisfied the requirement while the plan did not. Projecting
onto `safety_distance + half the control spacing` fixes the 4 cm, and the relaxation becomes the best
configuration measured so far:

| level 2, seeds 0-39 | no relaxation | relaxation, no margin | **relaxation + margin** |
| --- | --- | --- | --- |
| success | 32/40 | 29/40 | **35/40** |
| mean lap time | 8.86 s | 8.45 s | **8.37 s** |
| worst turn, mean | 52.7 deg | 37.8 deg | 41.4 deg |
| worst turn in the first 0.35 m | 38.0 deg | 25.7 deg | **30.3 deg** |

35/40 clears the 17/20 screen this page asks for before a decision run. `bspline_relax_iters = 150`
is now the default, and the remaining failures are seeds 8, 9, 30 (take-off and first leg, 1.3-2.5 s)
and 11, 39.

### The three that did not pay

Each measured at seeds 0-39 and left off, with the number next to the flag:
**the OCP pole penalty** at `w_obstacle = 300`, `obstacle_radius = 0.16` gives 29/40 at 9.20 s against
32/40 at 8.86 and does not save the pole seeds (8, 9, 33); **asking the frame avoidance for 0.65 m**
to cover the ~0.1 m the fit loses costs 0.3 s a lap and seed 7 still clips the top bar; **capping the
approach at 1.1 m/s after a gate correction** costs 0.4 s a lap and seeds 4, 13, 31, 39 still cross
0.13-0.15 m off centre, because the correction is lateral and the path left is short.

**The earlier relaxation measurement, for the record.**
`relax_control_points` minimises the curvature energy of the control polygon with a fidelity term
onto the fitted polyline, projects control points out of every pole's clearance sphere, and locks the
points within 0.3 m of a gate centre. With 150 iterations the geometry improves a lot and the lap
gets *faster*:

| 150 relaxation steps | off | on |
| --- | --- | --- |
| success | **32/40** | 29/40 |
| mean lap time | 8.86 s | **8.45 s** |
| worst turn, mean | 52.7 deg | **37.8 deg** |
| worst turn in the first 0.35 m, mean | 38.0 deg | **25.7 deg** |
| turns > 90 deg | 4 in 301 | **2 in 293** |

It is 0.4 s a lap quicker and much smoother, and it loses three finishes: the new failures are gate
passes (6, 11, 28, 33, 34, 36, 39) and two early take-off contacts (30, and 8 at 2.5 s), i.e. the
relaxation pulls the curve off the passages and distorts the first metres after the anchor. It needs
a wider passage lock and a stronger fidelity term before it can be adopted; the knobs are in place.

## Measured: what it costs to go faster

Six variants on level 2, seeds 0-39. The baseline is the current default (35/40, 8.37 s):

| variant | success | mean lap | verdict |
| --- | --- | --- | --- |
| `mu_speed` 10 -> 16 | 35/40 | 8.36 s | no effect |
| `a_lat_max` 4 -> 6 | 34/40 | 8.84 s | **slower** |
| both | 34/40 | 8.84 s | slower |
| `replan_slowdown_time` 0.8 -> 0.35 s, factor 0.5 -> 0.7 | 30/40 | 7.90 s | 0.47 s for 5 finishes |
| post-replan slowdown off | 22/40 | 6.89 s | 1.5 s for 13 finishes |

The reference-speed knobs do nothing, or hurt. `mu_speed` only sets the closed-form speed inside
the high-weight zones, and the curvature and lead clamps dominate there; `a_lat_max` 6 raises the
corner speed limit, the drone tracks worse, and the extra corrections cost more than the speed they
buy. The only lever with a real effect is the post-replan slowdown, and it is exactly the margin the
rest of this page is about: 1.5 s of lap time buys 13 finishes. So the lap time is set by how long the
drone needs to correct after each reveal, not by how fast the reference asks it to fly - to go faster
the corrections have to get cheaper (uncertainty-aware planning) or rarer (coalescing, measured worse),
not the reference speed higher.

Two knobs were removed while measuring this. `gate_correction_hold_time` and `reveal_speed_corrected`
existed only to hold the approach cap after a gate correction; the 35/40 configuration runs with the
cap off, so they are gone and `reveal_hold_time` carries its measured default of 0.0 s.

## Measured: what A* and the polyline stages are worth

The planner has three switches (`PathConfig.use_astar`, `use_push_out`, `use_smoothing`) so the
question "can the B-spline connect the gate waypoints on its own?" can be answered rather than
argued. Seeds 0-19, level 2, baseline 17/20 at 7.58 s:

| configuration | success | mean lap | note |
| --- | --- | --- | --- |
| baseline: A* + push-out + smoothing | **17/20** | **7.58 s** | |
| no A* | 13/20 | 8.28 s | straight chords pass poles; the local push-out bends the line late, which is the kink A* was introduced to avoid |
| no A*, no push-out, no smoothing ("only the B-spline") | 0/20 | - | the fit does not visit the gates: see below |
| no smoothing (A* + push-out kept) | 17/20 | 7.58 s | the fit and the relaxation round the corners on their own |

The geometry of one plan (the seed-0 opening plan, measured directly, not through the simulator):

| configuration | waypoints | polyline | spline | max turn | pole clearance |
| --- | --- | --- | --- | --- | --- |
| baseline | 1500 | 11.03 m | 10.89 m | 3.8 deg | 0.149 m |
| no A* | 1500 | 11.04 m | 10.89 m | 4.1 deg | 0.147 m |
| only the B-spline | 24 | 10.77 m | **7.85 m** | 4.8 deg | 0.143 m |
| no push-out | 26 | 5.16 m | 381 m | 170 deg | 0.036 m |

Two things follow, and they are the answer to "can the spline be the only connector":

* **The fit is not a connector.** Fitted with well-conditioned settings (six control points for 24
  waypoints) it *shortcuts* the polyline - 7.85 m of spline through a 10.77 m waypoint line - so the
  passages it should fly through are cut off. With more control points it diverges: one control point
  per sample makes the least-squares problem square and degenerate, and the same plan came out as a
  **1.5e10 m** spline. The code now requires at least four samples per control point
  (`n_control <= len(waypoints) // 4`), which is inert for the normal path (169 control points against
  375 allowed) and is what keeps a sparse input conditioned.
* **The relaxation is not a clearance solver.** With the push-out off - so the pole projection inside
  `relax_control_points` is the only thing keeping the curve clear - the same plan diverges to 381 m
  with 4 turns over 90 deg and 3.6 cm of clearance. The projection is a local repair on top of a clear
  polyline; it cannot create one.

So A* is worth four finishes, and the polyline stages are what make the fit well posed. The one stage
that could go is the smoothing - the B-spline and its relaxation already round the corners - and it is
a clean removal (17/20 at 7.58 s on these seeds). **What cannot go is the dense, pole-clear polyline
that A* and the push-out produce.**

## Measured: joint tuning does not beat the hand setting

The eight weights were swept one at a time on this page, so `scripts/tune_mpcc.py` runs a trust-region
Bayesian optimiser (TuRBO-8) over them instead. Four things had to be fixed before it could say
anything, and each was a measurement:

* the incumbent must be the **shipped configuration**, evaluated first. Without that, twelve
  evaluations of a five-parameter space all landed 20-45 reward points *below* the default and the
  optimiser could not tell.
* the trust region has to **start narrow** (0.1, not 0.4 of the whole log range); at 0.4 the "local"
  search was sampling the whole space.
* `w_speed_gate` and `w_tunnel` came out of the space: the first is inert (4.0 and 2.0 give
  bit-identical runs, because the gate speed is clamped by `reveal_speed` and `v_max`), the second
  measures worse at every setting.
* the seeds must **rotate over a pool**. Scoring ten fixed seeds overfits, and the first run proved it.

Two runs, each validated on 40 unseen seeds (60-99) where the shipped configuration scores
**34/40 at 8.20 s**:

| run | what the optimiser saw | candidate | unseen seeds |
| --- | --- | --- | --- |
| 300 episodes, 10 fixed seeds | 80 % against the shipped 70 % | `q_contour_peak 3832, mu_speed 27.2, a_lat_max 5.4, slowdown 0.62, reveal 1.73` | **30/40, 8.55 s** |
| 600 episodes, 20 seeds rotating over 60 | 85 % against the shipped 65 % | `q_contour_peak 3906, mu_speed 19.3, a_lat_max 7.4, slowdown 0.55, reveal 1.87` | **32/40, 8.14 s** |

Both regress, and the reason is measurable rather than mysterious: at 20 episodes the standard error of
an 85 % success rate is about 8 %, i.e. **±2.6 seeds** - the same size as the "improvements" the
optimiser found - and the reward makes it worse by pricing one crash at 2.0, which is two seconds of
lap time. So the search is fitting seed noise, not the controller.

What would work: 100+ episodes per evaluation (a 1-2 seed difference is then resolvable) at ~10 s per
episode, i.e. a 20-30 minute evaluation per candidate and a budget of several hours. Or optimise lap
time only, on a seed set where every candidate is known to finish - which is not the problem we have.

The hand setting therefore stands, with one deliberate change: the post-re-plan slowdown is now
`factor 0.7`, which is **32/40 at 7.78 s** on seeds 0-39 against 35/40 at 8.37 s for 0.5 - 0.59 s a lap
for three finishes, taken because the lap time is what the competition scores. Everything else on this
page is unchanged.

## The numbers to build on

All measured, see `docs/control/characterisation.md` and the failure analysis.

| quantity | value | consequence |
| --- | --- | --- |
| sensor range / speed 1.7 m/s | 0.41 s of warning | the reveal window |
| 0.21 m lateral correction, no preview | **0.46 s** to 5 cm | the window is too short |
| same, with preview | 0.20 s | geometry known ahead helps |
| implied speed ceiling while a pose is unknown | **v <= 1.5 m/s** | `0.7 / v >= 0.46` |
| flown path vs plan, successful laps | 0.109 m inside the plan | clearance is spent by tracking, not planning |
| flown path vs plan, failed laps | up to 0.33 m inside | late replan dominates |
| circle radial bias | -0.04 m, every radius | same effect, isolated |
| steady lateral acceleration | ~6 m/s^2; overload peaks 9.9-11.4 | `a_lat_max = 8` is conservative |
| collision box (`cf21B_500.xml`) | 0.07 x 0.07 x 0.02 half extents | contact at 0.085-0.114 m |
| `DRONE_RADIUS` in the planner | 0.09 m (0.105 m contact) | up to 9 mm optimistic |
| full re-plan | **7.5 ms** after the smoothing speedup (see below) | fits a 20 ms control period |
| MPCC solve | 1.3 ms p95 | not the bottleneck |

Two consequences that shape everything below.

- **In simulation, replan cost is wall-clock only.** The environment advances 20 ms of simulated time
  per step regardless of how long the planner took, so a 7 ms repair and a 40 ms re-plan fly the same
  trajectory. Cheap replanning improves the success rate only if it lets the controller re-plan
  *more often* (as an observation is refined), not because it saves time.
- **On hardware it is a real cost, and it was the blocking one.** At 50 Hz the planner used to take
  ~40 ms per re-plan, holding the previous command for ~8 cm of flight at 2 m/s and overrunning the
  control period twice over. At 7.5 ms it fits. Local repair is still worth having, but it is no
  longer required to fit the loop.

## Measured: the clearance fix and the distance penalty (branch `clearance-fix`)

Two changes, both implemented and both measured on the 20-seed screen.

**The planner now delivers the clearance it promises.** `_smooth_path` used to project onto
`0.92 * safety_distance` = 0.138 m, i.e. it deliberately stopped inside the requirement, and nothing
re-checked clearance after the gate handling. Now the projection uses `smoothing_margin = 1.25` on a
randomized track (`1.0` where the poses are exact, so levels 0 and 1 are unchanged) and a final
`_enforce_clearance` pass runs after every stage that can move the path inward. Verified on a
simulated reveal: waypoints 0.187 m, fitted path 0.187 m, so the spline does not undo it.

| | before | after |
| --- | --- | --- |
| successes whose plan was below 0.15 m | **2 of 13** | **0 of 13** |
| level 2 | 13/20, 8.17 s | 13/20, 8.24 s |
| level 0 / level 1 | 8.65 / 8.64 s | 8.64 / 8.63 s |

**The success count did not move**, and the failures changed identity: seed 6 moved from a first-leg
pole strike to a mid-lap one, while 9, 12 and 14 still die at t ~ 1.3 s. Those three are not a
clearance problem. Their committed plan measures 0.102-0.150 m from the **true** pole while keeping
0.187 m from the **nominal** one, which is the signature of a plan built before the reveal: the
planner cannot keep a distance from a pose it does not know, and no amount of enforcing changes that.

**The distance penalty in the OCP is measured inert.** `w_obstacle` and `obstacle_radius` now exist,
`NP` carries six pole slots, and the term is `w * max(0, r - d)**2` over the horizon — continuously
differentiable at the boundary, unused slots parked at 1e3 m. At `w_obstacle = 500` and
`r = 0.14 m` the 20-seed screen was **identical**: same 13/20, same seeds, 8.22 s. The reason is the
same stale pose — the penalty pushes away from the nominal pole, which is where the plan already
keeps 0.187 m — and where the pose is known the path is outside the radius entirely, so the term
never fires. It defaults to `0.0`; the code path is kept because it is the right place to put a
barrier once the poses are trustworthy.

Both results point at the same conclusion: **the remaining failures are information-timing, not
geometry.** The planner and the OCP both act on the nominal pose until the drone is 0.7 m away, and
0.7 m at 1.7 m/s is 0.41 s against a 0.46 s correction. The next lever is therefore P1.2/P1.3 —
reserve the pose uncertainty in the plan while the object is unrevealed, and cap the approach speed
at 1.5 m/s where it is.

## Planning cost, before and after (branch `faster-planning`)

`_smooth_path` was 90 % of the plan: 600 Laplacian sweeps over 1500 points, 0.083 ms each. One sweep
only spreads the rounding by one point, so the rounded length is `spacing * sqrt(sweeps)` — the point
count and the sweep count are interchangeable through that product. Smoothing every 4th point for
`600 / 16 = 38` sweeps therefore rounds the same distance for 64 times less work, and the result is
resampled back onto the input spacing so nothing downstream changes.

| stage | before | after | share now |
| --- | --- | --- | --- |
| `_smooth_path` | 50.0 ms | **1.0 ms** | 13 % |
| `_avoid_obstacles` | 3.8 ms | **3.6 ms** | **48 %** |
| `_connect_blocks` (A*) | 1.6 ms | **1.5 ms** | **20 %** |
| `_to_arc_path` (now `_to_bspline_path`) | 0.8 ms | **0.7 ms** | 9 % |
| everything else | 0.7 ms | 0.7 ms | 9 % |
| **total** | **55.8 ms** | **7.5 ms** | |

The path is unchanged where it matters: length 13.522 -> 13.521 m, clearance 0.150 m, gate centring
within 4 mm, curvature p95 3.43 -> 3.45, kappa max 5.81 -> 6.46. The 20-seed screen still gives
**13/20 at 8.17 s** with the same failing seeds, and levels 0/1 are unchanged.

Two things follow for the branches below:

- A* is no longer 3 % of the plan, it is 20 %. The caching and bounding-box work listed under
  `dslite-replan` is now worth 1.5 ms per re-plan rather than noise.
- `_avoid_obstacles` is the new hotspot at half the plan, and it is the same shape of problem: a
  per-point push-out over the full 1500-point array, once per pole per iteration.

## Idea 1: remember observed poses: already done, do not rebuild

The environment latches observed poses (`_update_visited_objects` ORs the in-range mask, and the
observation builder reports `where(visited, true_pose, nominal_pose)`). Every later re-plan already
uses the true pose. The failure was never forgetting; it was committing the plan at `t = 0` against a
nominal pose.

The only place memory still buys something is on the real drone, where repeated noisy detections can
be fused per static landmark. That is a small (running average or a per-landmark Kalman filter) and
has no effect in simulation, where the first observation is exact.

## The 85 % goal and how it is judged

Two-tier, agreed protocol:

| stage | seeds | target |
| --- | --- | --- |
| screen per candidate | 0-19 | >= 17/20 to advance |
| screen, ambiguous | 0-39 | extend when the screen gives 15-16/20 |
| **decision** | **20-219** | **>= 170/200 (85 %)** |
| decision, inconclusive | 420-619 | 165-169/200 -> another 200 fresh seeds |
| regression | level 0, level 1 | 2/2, lap time unchanged |

```bash
pixi run python scripts/debug_run.py --config level2.toml --n_runs 20 --plot=False --no-log
pixi run python scripts/debug_run.py --config level2.toml --seed 20 --n_runs 200 --plot=False
```

## Competition constraint: never edit the provided configs

The online checker compares the **line count** of every provided config file and reports a difference
as `CHEAT DETECTED`. An earlier commit of this branch added fifteen lines of comment and override keys
to `config/level2.toml` and was rejected for exactly that reason. The file is now restored to the
official one (125 lines).

Consequences for everything below:

- The level 2 pace lives in `RANDOMIZED_TRACK_MPCC` in `my_controller.py`, not in the config. A
  `[controller]` entry that names an `MPCCConfig` field still overrides it, so tuning sweeps use a
  scratch config rather than a race config.
- The `PathConfig` defaults (`use_astar = True`, `safety_distance = 0.15`) carry the planner settings,
  so A* needs no config edit either.
- Our own configs live outside `config/` for the same reason: `scenarios/characterisation.toml` is the
  deterministic arena the characterisation runs in, and it is nowhere near the provided files.
  `config/` now contains only provided configs.
- The permitted edits are exactly two keys per file: `controller.file` and `env.control_mode`. Audit
  any submission with `git diff --stat <baseline> HEAD -- config/`, which must show no provided
  config at all.

Verified after the restore: level 2 seeds 0-19 still give **13/20 at 8.17 s**, so moving the tuning
changed no behaviour.

## Branch: `bspline-local-repair` (idea 2a)

**Goal.** A reveal repairs the path locally instead of re-planning 13 m, and the repair runs in
miiliseconds so it can run continuously rather than only on discrete reveal events.

**Why a B-spline.** `ArcPath` is a natural cubic spline: moving one knot changes the whole curve. A
uniform B-spline has local support, a span is controlled by `p + 1` successive control points, so a
repair touches only the control points near the obstacle. This is the structural change that makes
"partial replan" well defined rather than approximate. See EGO-Planner
([arXiv 2008.08835](https://ar5iv.labs.arxiv.org/html/2008.08835)): control points as decision
variables, collision cost from `{p, v}` pairs against a collision-free guiding path instead of an
ESDF, fast-restarting L-BFGS, and no requirement for a collision-free initialisation.

**Design.**

1. `control/bspline_path.py`: `BsplinePath`, uniform cubic, clamped ends, implementing the `ArcPath`
   duck type (`length`, `pos`, `tangent`, `curvature`, `closest` with the monotone `window`
   contract). `length` stays arc-length based, so the MPCC progress code is unchanged.
2. `PathPlanner` returns a `BsplinePath`, and it is the only representation left: the natural cubic
   spline, its arc-length reparameterization and the anchor pin that went with them were deleted once
   the B-spline replaced them.
3. `PathPlanner.repair(path, obs, theta)`:
   - window = control points covering `[theta - 0.2, theta + repair_window]` (`repair_window` ~3 m),
   - free variables = the control points in that window, everything else pinned,
   - cost = smoothness (acceleration and jerk of control points) + the EGO collision penalty from
     `{p, v}` pairs against poles inflated by `safety_distance` and the gate frame band +
     soft anchors at the gate passages inside the window,
   - solved with L-BFGS-B warm-started from the current control polygon,
   - continuity at both ends of the window is implicit because those control points are pinned.
4. `MPCCController._replan` calls `repair` when every changed object lies inside the repair window;
   otherwise it falls back to the full `plan`. The decision is recorded in the plan log
   (`plan_reason`), so the run log shows which path produced each segment.
5. Fall back to the full planner when the repair fails to clear the obstruction or the length changes
   by more than 5 %.

**Acceptance.** Repair under 2 ms p95; committed path keeps `safety_distance`; 20-seed screen not
below the current 13/20 before any tuning; then the P1 fixes below and the 200-seed decision run.
Also verify on `deploy.py` that a repair fits inside a 20 ms control period.

**Risks.** The B-spline's curvature is not `1/R` by construction, so the speed limit uses the
measured curvature (which it already does, from a finite difference); the gate straight passages must
be re-asserted after a repair; `closest` must stay monotone or the drone stalls.

## Branch: `dslite-replan` (idea 2b)

**Goal.** Repair the A* search instead of re-running it when a pole's pose is corrected.

**Design.**

1. D* Lite over the existing `ArenaGrid` occupancy, 8-connected, same cost model
   ([Koenig & Likhachev, AAAI 2002](https://aaai.org/papers/00476-AAAI02-072-d-lite/)).
2. `ArenaGrid.update(poles, ...)` recomputes the occupancy mask and reports the cells that changed;
   the solver updates only those `g`/`rhs` values and re-queues their neighbours.
3. `PathConfig.use_incremental = True` routes `_connect_blocks` through the incremental search,
   falling back to the current A* for a cold start or when the goal itself is unreachable.
4. Keep the whole-grid recompute as the reference implementation and assert identical routes on
   unchanged input.

**Acceptance.** Repair under 1 ms after a single-pole correction (full A* is ~5 ms); routes identical
to the reference on unchanged input; 20-seed screen unchanged.

**Risk.** Be honest about the ceiling: A* is ~5 ms of a ~58 ms plan, so the end-to-end gain is small
unless it is combined with 2a (where the whole plan is replaced) or with re-planning more often. If
the measured end-to-end gain is under 10 %, stop and record that here rather than polishing it.

## Pending success-rate work (P1), independent of both branches

Ordered, each validated on the 20-seed screen and promoted to the 200-seed decision run only if the
previous step misses 85 %.

- **P1.1 Clearance honesty and the collision radius.** Replace the hard-coded `safety_distance * 0.92`
  in `_smooth_path` with a named `smoothing_margin`; add an `_enforce_clearance` pass after
  `_avoid_gate_frames` so a committed path keeps `safety_distance`; set `DRONE_RADIUS` to the box
  corner 0.099 m; and fix `debug_run.classify` to use the true poses so failures are attributed
  correctly.
- **P1.2 Uncertainty inflation.** While a pole is unrevealed, inflate it by `pose_uncertainty = 0.212`
  in the chord test and the A* raster (needs per-pole `pole_inflate`, a contained change).
- **P1.3 Reveal-aware speed cap.** Cap the reference speed to **<= 1.5 m/s** over an arc-length
  window before an unrevealed pole. The ceiling comes from `0.7 / v >= 0.46`, not from a sweep.

Reference for the margin arithmetic: the flown path sits 0.109 m inside its plan, the contact radius
is 0.085-0.114 m, and the planned clearance is 0.15 m. Those three numbers are why the failures
cluster where they do.

## Also outstanding

- Scenario geometry unit tests (`test_scenario_path.py`): length, exact curvature, `closest`
  monotonicity and wrap safety.
- Arena replay: the characterisation metrics computed on level-2 race logs, plus gate-defined
  scenario configs (`config/scenario_*.toml`) so the same manoeuvres can be flown with `deploy.py`.
- Assets are gitignored: `logs/`, `plots/`. Regenerate with `scripts/characterize.py` and
  `scripts/debug_run.py`.
