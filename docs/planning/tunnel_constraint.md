# Gate safety as a constraint: the MPCC++ tunnel

The controller used to police the gates through the cost: a Gaussian contour weight that peaks at
each gate, plus a speed throttle that multiplies the reference down near the plane. That is the
formulation of the T-RO22 MPCC, and it has a structural weakness this page is about.

[MPCC++](https://rpg.ifi.uzh.ch/docs/RSS24_KrinnerRomero.pdf) (Krinner, Romero, Bauersfeld,
Zeilinger, Carron, Scaramuzza, RSS 2024) states it directly: the Gaussian contour weight "does not
explicitly prevent gate collisions, resulting in a sub-optimal trade-off between performance and
safety", it costs `2 n_gates + 4` hyperparameters, and it "requires specific tuning for each gate
configuration". Their alternative is a **prismatic tunnel** around the path, enforced on the state
rather than paid for in the cost, and it is what lets them fly 80 km/h with a 100 % success rate
against 59-89 % for the same controller without it.

## The constraint

Around the planned path, a corridor of half width `W` per stage:

```
   |(p - pd(theta_k)) . n(theta_k)| <= W(theta_k)          n, b: the two in-plane axes
   |(p - pd(theta_k)) . b(theta_k)| <= W(theta_k)          at the cross-section of stage k
```

The paper writes it as four half-spaces around the corner `p0 = c - W n - H b`; with `W = H`, which
is what a square 0.40 m opening calls for, those four collapse to the two above. The axis *along* the
corridor is deliberately unconstrained — that is what makes it a tunnel and not a box.

`W` is a funnel keyed on the distance to the gate plane: `tunnel_gate_half_width` (0.09 m = half the
opening minus the drone's 0.099 m contact radius) on the plane, `tunnel_nominal_half_width` (0.40 m)
at the edge of `tunnel_window` (0.6 m), and inactive beyond it. Between gates the drone is therefore
free to cut the corner; at a gate it must be inside the hole.

## Soft, not hard, and why

A hard constraint on a real-time SQP is fragile: the paper notes that model mismatch, disturbances or
state-estimation drift can leave the next state outside the set, and "employing real-time solvers does
not inherently ensure numerically stable solutions". They use the barrier

```
   p(h) = log(1 + exp(-alpha h)),   alpha = 100
```

on each slack `h`, which is nearly inert more than ~5 cm inside the tunnel and grows by `alpha` per
metre of violation. We do the same, with the exponent capped at 50 so the cost cannot overflow when
the drone is deep inside. Their hardware results use this barrier, not the hard version, so a "safety
guarantee" is not what is actually flying — it is a very steep cost, and we describe it that way.

## What we took, what we left

| | decision | reason |
| --- | --- | --- |
| tunnel barrier | **taken** | the gate opening becomes a property of the state, not of a tuned weight |
| soft barrier, `alpha = 100` | **taken** | their measured choice for real-time solvers; hard constraints add failure modes |
| square cross-section, gate frame | **adapted** | our openings are square, and the gate's own axes are exactly the hole's edges; a Frenet frame is undefined on a straight segment |
| low contour weight, no gate Gaussian | **taken** | with safety in the constraint the cost no longer has to hold the drone on the path |
| TuRBO joint tuning | **taken, separately** | 8 parameters tuned jointly with a lap-time reward, instead of one knob per screen |
| polynomial residual dynamics | **left** | it models aerodynamic effects identified from *real flight data*; level 2 is simulated, and the mismatch we do have (mass, inertia) is already handled by the online mass estimate |
| terminal set, recursive feasibility | **left** | it needs an offline *periodic feasible* trajectory through the gates; our reference is re-planned at every reveal, so a fixed terminal set does not survive a re-plan |

## Measured: it does not pay on this track

Seeds 0-19 are the fast screen used throughout the planning pages. The baseline is the configuration
this page started from - gate Gaussian, gate throttle, no tunnel - at **17/20 and 7.58 s**.

| configuration | success | mean lap |
| --- | --- | --- |
| baseline: Gaussian + throttle, no tunnel | **17/20** | **7.58 s** |
| tunnel 30, Gaussian and throttle kept | 16/20 | 8.32 s |
| tunnel 100, kept | 15/20 | 8.34 s |
| tunnel 300, kept | 16/20 | 8.21 s |
| tunnel 1000, kept | 13/20 | 8.18 s |
| tunnel 300, Gaussian and throttle **removed** (the paper's formulation) | 7/20 | 8.51 s |
| + pole barrier `w_obstacle=300`, radius 0.16 | 10/20 | 8.49 s |
| + pole barrier 1000, radius 0.18 | 10/20 | 8.34 s |
| + half the gate throttle back (`w_speed_gate=2`) | 10/20 | 8.49 s |

The tunnel alone is not free: it fights the gate Gaussian (both try to place the drone at the gate,
one by cost, one by constraint), which costs a finish and 0.6 s. Removing the Gaussian as well - the
step that should unlock the speed - costs **ten** finishes. The reason is the track, not the
formulation: the poles sit in the gate approaches, and the gate throttle doubles as pole protection.
Removing it frees the drone to cut corners at the curvature limit, and the pole barrier only recovers
three of the ten. The paper's track has no poles between gates, and its gate poses are known to the
controller at all times, so both of the mechanisms that were doing the work here are absent there.

The tunnel therefore ships **off** (`w_tunnel = 0.0`), measured, with the code path and the numbers in
place. It is not a dead end: it is the right constraint for the gate, and what fails is the assumption
that it can replace the slowdown while the poles are still only protected by that slowdown.

## What did come out of the paper

- **The weights are OCP parameters now.** `q_lag`, `q_contour`, `q_lag_peak`, `q_contour_peak`,
  `mu_speed`, `r_vtheta` and `w_tunnel` travel in the parameter vector (`WEIGHT_FIELDS`), so changing
  one no longer costs a one-minute solver rebuild: a weight change is 9 s instead of 90 s. This is
  what makes joint tuning possible at all.
- **`scripts/tune_mpcc.py`**: TuRBO-8 over those eight weights, with their lap-time reward extended by
  a solver-failure and crash penalty, and seeds drawn from a rotating pool so the tuner cannot
  specialise to ten tracks. A 600-episode budget is about 100 minutes.
- **`gate_weight` is a separate knob** in `track_reference`, so the Gaussian can be turned off without
  also deleting the pole proximity weighting, and the two formulations can be compared directly.
- **`tunnel_slack` in the run log**: the worst slack at the drone, which is the audit that separates
  "the constraint was not tight enough" from "the constraint was not the thing that failed".

## Operational notes

- The tunnel lives in the OCP parameters, two slots of ten values (`axis_a`, `axis_b`, centre, half
  width), so a horizon that spans two passages gets both. A third would be dropped, and that is
  logged rather than silent.
- The half width is a funnel of the **progress along the path**, not of the distance to the gate
  plane. The plane-distance version measured 0/20: the gates of this track are close enough in space
  that the drone sits within 0.6 m of a *later* gate's plane while flying past, and constraining that
  drove it into the wall.
- `w_tunnel = 0` disables it completely: every slot is sent inactive (`W = 1e3`, zero axes) and the
  barrier evaluates to exactly zero, which is how the parameter wiring was verified to be inert -
  the 40-seed screen reproduces 35/40 at 8.37 s with it off.
- `min_slack` is logged per tick (as `tunnel_slack` in the run record). A gate clip with a positive
  slack means the tunnel was not the constraint that failed, and that is the audit to run on any
  remaining gate failure.
