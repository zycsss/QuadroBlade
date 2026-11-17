# QuadroBlade: Autonomous Drone Racing

Team QuadroBlade's solution to the TUM [Autonomous Drone Racing](https://github.com/learnsyslab/lsy_drone_racing)
course, built on the [crazyflow](https://github.com/learnsyslab/crazyflow) simulator.

## Final controller

**`lsy_drone_racing/control/rl_scene.py` (`RLScene`)** is our final controller, and every
`config/level*.toml` selects it. It is a reinforcement-learning policy trained only on randomly
generated tracks, never on the track it flies:

1. From the first observation it plans a time-optimal guide through the gates
   (`control/topt_planner.py`, IPOPT, cached on disk).
2. At 50 Hz a small NumPy network (2 x 64) turns the drone state, the guide ahead and the nearby
   gates and poles into an attitude and thrust command. The trained weights are in
   `control/rl_scene.npz`.

Because the policy never sees anything that identifies a track, the same network flies level 0 and
generated level 3 tracks. How it is built and trained is described in
[docs/Architecture.md](docs/Architecture.md).

| Scenario | Result |
|---|---|
| Level 0 | 64/64 finished, 4.03 s |
| Random tracks (level 3 generator, level 1 conditions) | 243/256 finished, about 4.67 s |
| Level 2 (moved gates and poles) | 0/64 finished: the guide is not yet re-planned after a reveal |

## Quickstart

Install [pixi](https://pixi.sh), then from the repository root:

```bash
pixi run python scripts/sim.py
```

This flies the final controller on `config/level0.toml` with the viewer. Choose a level with `--config`:

```bash
pixi run python scripts/sim.py --config level3.toml
```

On a Linux machine with a GPU, use the `gpu` environment (`pixi run -e gpu ...`). Activating an
environment builds [acados](https://github.com/acados/acados) the first time, which the MPCC needs.

Run the tests:

```bash
pixi run -e tests tests
```

The integration test that flies the final controller is opt-in:

```bash
RUN_FINAL_CONTROLLER=1 pixi run -e tests pytest tests/integration/test_final_controller.py
```

## Difficulty levels

| Level | Randomized inertia | Moved gates and poles | Random tracks |
|---|---|---|---|
| [0](config/level0.toml) | No | No | No |
| [1](config/level1.toml) | Yes | No | No |
| [2](config/level2.toml) | Yes | Yes | No |
| [3](config/level3.toml) | Yes | Yes | Yes |

## Repository layout

| Path | Contents |
|---|---|
| `lsy_drone_racing/control/` | Controllers: `rl_scene.py` (final), `rl_race.py` and `topt_mpcc.py` (alternatives we compared), the time-optimal planner, and the course's base class and examples |
| `lsy_drone_racing/envs/` | The race environments (from the course) |
| `lsy_drone_racing/utils/` | Policy features, track generation, the fast guide planner and plotting |
| `scripts/` | Simulation, evaluation, training and analysis scripts |
| `tests/` | Unit and integration tests |
| `config/` | Level definitions |
| `docs/` | [Architecture](docs/Architecture.md), [experiment log](docs/PLAN_path_rl.md), [training scenarios](docs/training_scenarios.md) |
| `replay_site/` | Browser replay of the logs that `scripts/debug_run.py` writes |
| `ros_ws/`, `scripts/deploy.py` | Flying a real Crazyflie |

## Other controllers

| Controller | Use |
|---|---|
| `rl_race.py` | RL policy trained on the track it flies: 64/64 at 3.33 s on level 0 and 64/64 at 4.66 s on level 2. Its guide is planned for the nominal track, so it cannot fly level 3 |
| `topt_mpcc.py` | Contouring MPC that flies the time-optimal plan, bent to the revealed gates: 54/64 at 4.40 s on level 2 |

Fly one with `--controller`, for example
`pixi run python scripts/sim.py --config level2.toml --controller rl_race.py`.

## License

MIT, see [LICENSE](LICENSE). The environments, simulator interface and example controllers come from
the course's [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing) repository.
