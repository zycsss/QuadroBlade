# Level2 sub6 MPCC

`mpcc_level2_sub6_controller.py` is an opt-in Level2 speed profile. It copies the caller's configuration, fills missing pace defaults for randomized gate/obstacle layouts, and preserves explicit overrides. It uses the existing MPCC dynamics, actuator limits, observed geometry and controller interface.

## Reported performance

User-provided performance summary: **Level 2 completes in under 7 seconds with a stable success rate of approximately 80%.**

The approximately 80% success rate has not been independently verified by this branch's fixed-seed automated tests.

## Run

From the repository root, with the project's simulation environment activated:

```bash
python scripts/sim.py --config level2.toml --controller mpcc_level2_sub6_controller.py
```

`config/level2.toml` also selects this controller by default. Its `seed = -1` generates a random track; the profile does not guarantee completion or a time below six seconds on every track.

For a reproducible headless run and trajectory, gate-time and telemetry exports:

```bash
python -m tests.mpcc_evaluation --config config/level2.toml --seed 0 \
  --controller mpcc_level2_sub6_controller.py
```

Reports are written under the ignored `plots/mpcc/` directory. The evaluator supports explicit in-memory settings with `--controller-overrides`; it does not edit the track file.

## Default pace settings

| Setting | Value |
| --- | --- |
| `v_max` | 5.0 |
| `a_lat_max` | 14.0 |
| `w_speed_gate` | 2.5 |
| `v_lead` | 1.0 |
| `l2_unseen_speed` | 1.8 |
| `replan_slowdown_time` | 0.15 |

These defaults were selected on a fixed seed. They are not a randomized reliability guarantee. Nonrandomized layouts retain the inherited defaults. Fully randomized Level3 tracks are rejected by the Level2 parent.

## Required control modules

The controller inherits through `mpcc_level2_controller.py`, `mpcc_controller.py` and `my_controller.py`. It also uses `level2_solver.py`, `path_planning.py`, `astar.py`, `controller.py` and the control package initializer. Keep these dependencies together with the existing simulation framework and acados runtime. Native solver artifacts are generated or reused under the ignored `c_generated_code/` directory.

## Verification

Run the controller and planning unit tests:

```bash
python -m pytest tests/unit/control -q
```

The fixed-seed integration test compiles/loads the native solver and flies three fresh instances. It checks ordered completion, simulation time below six seconds, finite telemetry, zero solver failures and unchanged configuration bytes:

```bash
RUN_MPCC_BUILD=1 python -m pytest tests/integration/test_mpcc_level2_sub6.py -q
```

This test uses the evaluator's explicit seed reset. `sim.py` uses a different reset sequence, so the same seed label across these entry points need not produce the same randomized track. The integration test establishes a fixed-seed result, not a randomized success rate.
