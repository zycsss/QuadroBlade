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
| `_to_arc_path` | 0.8 ms | **0.7 ms** | 9 % |
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
2. `PathPlanner` returns a `BsplinePath` behind `PathConfig.path_kind = "bspline"`. The existing
   spline stays the default until the branch proves itself.
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
