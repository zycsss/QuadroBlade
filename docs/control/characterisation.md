# Characterising the drone

The controller is full of numbers that were chosen rather than measured: `a_lat_max = 8 m/s²`,
`v_max = 3.5 m/s`, a 0.15 m obstacle clearance, and the assumption that a path can be corrected
within the sensor range. Every tuning decision inherits those guesses, so this page measures them.

The method is to fly the racing controller against an **analytic reference** instead of a planned
path. A planned "circle" through gates is smoothed by the planner and speed-shaped by the gate
proximity weight, so it measures the planner. An analytic circle of radius `R` has curvature exactly
`1/R`, so what it measures is the drone.

## Method

`scripts/characterize.py` runs the matrix below. Each run:

- installs an analytic path (`line`, `circle`, `corner`, `shift`) as the reference,
- places the drone exactly on that path at `t = 0`, with no disturbances and no track
  randomisation, so repeated runs agree,
- keeps every other part of the racing controller — the optimal control problem, the solver, the
  progress projection, the curvature speed limit — and the same gains as `config/level2.toml`,
- ends when the path is flown, not on a gate count.

Tracking error is measured against the reference evaluated at the run's own path progress, so a
drone that lags behind is measured against where the reference actually was.

```bash
pixi run python scripts/characterize.py                          # the full matrix
pixi run python scripts/characterize.py --only circle            # one manoeuvre
pixi run python scripts/characterize.py --only line --speed 1.0,2.0
```

Outputs: `logs/scenarios/*.npz` with `.json` summaries, `plots/scenarios/*.png` per scenario, and
`plots/scenarios/summary.md` with the table. The table can be regenerated from the saved logs alone,
because `scenario_metrics` is a pure function of a spec and its arrays.

## Results

Measured with the level 2 gains (`a_lat_max = 8`, `v_max = 3.5`, `w_speed_gate = 4`). `v` is the mean
speed over the middle half of the run, `tilt` the median attitude at cruise, `a_lat` the mean
lateral acceleration `v²κ`, and the settle times are the time to return within 5 cm and 2 cm of the
reference after a 0.21 m lateral disturbance.

| scenario | v [m/s] | v_ref | err rms [m] | err max [m] | radial bias [m] | a_lat [m/s²] | tilt [°] | settle 5 cm [s] | settle 2 cm [s] | solve p95 [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| line, speed 1.0 | 0.93 | 1.00 | 0.00 | 0.00 | — | 0.00 | 6.1 | — | — | 1.25 |
| line, speed 1.5 | 1.40 | 1.50 | 0.00 | 0.00 | — | 0.00 | 4.1 | — | — | 1.24 |
| line, speed 2.0 | 1.87 | 2.00 | 0.00 | 0.00 | — | 0.00 | 5.5 | — | — | 1.28 |
| line, speed 2.5 | 2.33 | 2.50 | 0.00 | 0.00 | — | 0.00 | 7.5 | — | — | 1.28 |
| line, speed 3.0 | 2.68 | 2.95 | 0.00 | 0.00 | — | 0.00 | 9.4 | — | — | 1.26 |
| circle, R = 0.5 m | 1.69 | 1.99 | 0.03 | 0.05 | −0.04 | 5.80 | 34.4 | — | — | 1.37 |
| circle, R = 0.8 m | 2.16 | 2.47 | 0.03 | 0.05 | −0.04 | 5.94 | 35.1 | 0.16 | 0.32 | 1.30 |
| circle, R = 1.1 m | 2.52 | 2.86 | 0.03 | 0.06 | −0.03 | 5.90 | 34.4 | 0.22 | 0.36 | 1.29 |
| circle R = 0.8, 1.2× overload | 2.26 | 2.72 | 0.03 | 0.06 | −0.04 | 6.73 | 44.4 | 0.28 | 0.42 | 1.41 |
| circle R = 0.8, 1.5× overload | 2.20 | 2.80 | 0.04 | 0.07 | −0.04 | 6.42 | 49.1 | 0.28 | 0.40 | 1.40 |
| corner, R = 0.5 m, 90° | 1.83 | 2.00 | 0.02 | 0.05 | — | 3.47 | 18.1 | — | — | 1.30 |
| corner, R = 1.0 m, 90° | 2.27 | 2.66 | 0.03 | 0.05 | — | 4.38 | 34.9 | — | — | 1.30 |
| shift 0.21 m at 1.0 m/s | 0.95 | 1.00 | 0.02 | 0.10 | — | 0.00 | 6.1 | 0.20 | 0.28 | 1.25 |
| shift 0.21 m at 1.7 m/s | 1.60 | 1.70 | 0.02 | 0.10 | — | 0.00 | 5.3 | 0.20 | 0.28 | 1.26 |
| start 0.21 m off at 1.0 m/s | 0.93 | 1.00 | 0.05 | 0.21 | — | 0.00 | 10.1 | 0.46 | 0.52 | 1.25 |
| start 0.21 m off at 1.7 m/s | 1.59 | 1.70 | 0.07 | 0.21 | — | 0.00 | 4.7 | 0.46 | 0.54 | 1.29 |

## Findings

**1. On a straight the controller is essentially exact.** 0.00 m RMS tracking error at every speed,
and the flown speed lands 3–10 % under the request because the 4 m run spends a third of its length
accelerating. Nothing here needs fixing.

**2. The drone cuts every circle by about 4 cm.** The radial bias is −0.04 m at all three radii and
both overload cases — a consistent, curvature-independent shrink of the effective radius. This is the
same effect measured on the race track, where the flown path sat 0.109 m inside its plan on
successful laps and up to 0.33 m on failed ones. It means a planned clearance of 0.15 m to a pole is
worth 0.11 m in practice.

**3. The lateral acceleration in a steady circle is ~6 m/s², but the airframe is not the limit.**
Asking for 1.2× and 1.5× the curvature-limited speed raises the mean to 6.4–6.7 m/s² and the peak to
9.9–11.4 m/s² while tracking error stays at 3–4 cm. The reference is what limits the manoeuvre: the
drone never reaches the requested 2.7–2.8 m/s because the thrust that would accelerate it is spent
holding the turn. `a_lat_max = 8` is therefore **conservative rather than optimistic** — corner speeds
could be raised without the drone losing the path.

**4. A 0.21 m correction takes 0.46 s — longer than the 0.41 s the sensor gives at racing speed.**
This is the decisive measurement. With the new path known in advance (the reference steps ahead of
the drone, as after a re-plan) the error settles to 5 cm in 0.20 s. With no preview at all — the
drone starts beside its path, which is the worst case after a late reveal — it takes **0.46 s to
5 cm and 0.52–0.54 s to 2 cm**, and the speed barely matters (1.0 and 1.7 m/s give the same 0.46 s,
because the correction is limited by lateral authority, not by distance).

At 1.7 m/s and `sensor_range = 0.7 m` the drone has **0.41 s** between learning the true pose and
arriving. That is shorter than the 0.46 s the correction needs, which turns the failure mechanism
from a hypothesis into an arithmetic shortfall: the reveal is simply too late for a correction of
that size at that speed.

**5. The attitude at cruise sets the collision radius.** Median tilt is 4–9° on the straights and
34–49° in the turns, i.e. a bank of `atan(a/v²)` as expected. The drone's collision geometry is a
box of `0.07 × 0.07 × 0.02 m` half extents, whose horizontal reach is 0.070 m face-on but 0.099 m
corner-on; with a pole of 0.015 m radius, contact happens at 0.085–0.114 m centre-to-axis. The
simulation's own contact model is that box, while the planner assumes a 0.09 m sphere, i.e. 0.105 m.
At 34–49° of bank the box presents a corner, so the upper end of that range is the one that matters
near a pole.

**6. The solver is not a constraint.** 1.24–1.41 ms at p95 against a 20 ms control period, with no
failures in any scenario.

## What this means for the controller

- **The reveal window is too short for a full correction.** Either the drone must arrive more slowly
  where a pose is still unknown, or the path must be planned with the uncertainty included so that no
  correction of that size is needed. The 0.46 s measurement is the budget any fix has to beat.
- **The 4 cm inside-cut must be paid for out of the clearance budget.** A 0.15 m planned clearance is
  0.11 m flown; the collision box needs up to 0.114 m. Those two numbers being equal is why the
  failures cluster exactly where they do.
- **`a_lat_max = 8` can be raised.** It is not the airframe limit, and the measured 9.9–11.4 m/s²
  leaves headroom at the apex of a corner.
- **`DRONE_RADIUS = 0.09` should become the box corner, 0.099 m,** so the planner and the collision
  model agree about how wide the drone is.

## Limitations

- One deterministic run per scenario: the simulation has no noise here, so the numbers are repeatable
  but they are not a distribution.
- The scenarios are flown with the level 2 gains. Changing `w_speed_gate` or `mu_speed` changes the
  reference speed and moves every number that depends on speed.
- The analytic reference is not a substitute for the planner: these runs do not exercise A*, the
  smoothing, or the gate handling. They measure the drone and the MPCC, which is what the constants
  above depend on.
- The no-preview case is created by starting the drone beside its path, not by swapping the path
  in flight. A real re-plan also changes the path *ahead* of the drone, which is why the measured
  0.46 s is an upper bound on the raw correction and the 0.20 s preview case is the lower bound.
