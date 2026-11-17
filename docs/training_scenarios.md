# How the random training scenarios are generated

How the training scenarios for `RLScene` get built, from a random layout to one episode. The aim is
a policy that has learned local tasks, such as getting through a gate with a pole beside it, instead
of one track. Results are in [PLAN_path_rl.md](PLAN_path_rl.md), and the wider system is in
[Architecture.md](Architecture.md).

```
 1. Random layout               scene_generator.py, or the race's own generator
          |
 2. Solve a guide               topt_planner with RANDOM_TRACK_CONFIG, in advance
          |
 3. Check the guide             keep only guides that touch nothing
          |
 4. Library of tracks           .npz files, loaded onto the GPU
          |
 5. Start an episode            pick a track, a drone and a start, for each of 8192 worlds
```

Steps 1 to 4 run once, offline. Step 5 runs every time a world's episode ends.

## 1. Random layouts

A layout has the race's format: a drone start, 4 gates, 4 poles and the race's order of passes,
`[1, 2, 3, 4, 2]`. The same gate is flown twice, so one pass has an approach that nobody designed,
often from the side or behind the gate.

There are two sources, and neither one contains the other:

| Per gate pass (median) | `scene_generator.py` | Race's generator (level 3) | Level 2 |
|---|---|---|---|
| Nearest pole to the gate | 0.67 m | 0.92 m | 0.51 m |
| Distance from the last gate | 1.74 m | 2.02 m | 1.84 m |
| Turn into the gate | 36 deg | 24 deg | 36 deg |

The first column was measured on 1500 layouts before solving. The others were measured on 1000.

### Our generator: `utils/scene_generator.py`

Each leg is sampled on its own, so scene types come up in controlled amounts instead of being tied to
one track's shape.

**The start** is uniform in the arena, at least 0.5 m from the walls.

**Each gate**, in turn:

1. Draw the leg length from a triangular distribution: 0.6-3.0 m (most likely 1.5 m) for the first
   leg, and 0.9-4.4 m (most likely 1.8 m) for the others.
2. Draw the direction of the leg, up to 180 deg either side of the last gate's axis. This is the turn
   out of the last gate.
3. Keep the position only if it is at least 0.3 m from the walls and 0.9 m from every other gate.
4. Draw the gate's axis relative to the leg, which is the turn into the gate: up to 60 deg, or up to
   130 deg for 15% of gates.
5. Draw the height: 0.7 or 1.2 m, the race's two gate heights, plus up to 0.1 m either way.

**Each pole** is one of three kinds:

| Kind | Share | Where |
|---|---|---|
| Near a gate | 45% | 0.3-0.9 m from the centre of a random gate, in any direction |
| Near a route | 35% | Anywhere along a random leg, up to 0.6 m to either side |
| Anywhere | 20% | Uniform in the arena |

A pole is kept only if it is at least 0.3 m from the walls, 0.3 m from every gate centre, 0.12 m from
every gate frame, 0.3 m from other poles and 0.4 m from the start. Pole tops are at 1.55 m.

If a gate or pole finds no valid place in 50 tries, the whole layout starts over.

### The race's generator: `envs/randomize.py`

`build_random_track_fn` is what the race itself uses on level 3, so training on it directly targets
random tracks. It places objects on a grid with fixed exclusion distances: 1 m between gates, 0.8 m
from gates to poles, and 0.3 m between poles. Gate heights are fixed by the config, and each pole is
placed in the corridor to its gate when there is room.

## 2. Solving a guide

`topt_planner` solves the fastest point-mass trajectory through the layout, from the start at
0.7 m/s upwards. One initial guess takes about 8 s on one core, far too slow to run per episode, so it's done in
advance by `scripts/solve_scene_library.py`.

Random tracks use `RANDOM_TRACK_CONFIG`, because the default settings go wrong on them:

| Setting | Value | Why |
|---|---|---|
| `clear_everything` | True | Clearance from every object, not only those near the initial guess |
| `node_spacing_max` | 0.25 m | Stops the solver spreading nodes until it can step over a pole or bar |
| `between_nodes` | 1 | Also checks a point halfway between nodes |
| `frame_clearance` | 0.12 m | The drone's 8.6 cm sphere, plus the 3 cm training counts as a hit |

Without these, plans could reach 46 m with nodes 1.7 m apart, stepping straight through gates. The
fixed levels keep the defaults, because their plans are clear and their policies were trained on them.

`--fast_guides` solves with `FAST_GUIDE_CONFIG` instead, which raises the limits to tilt 1.3 rad and
thrust 16 m/s^2. A policy trained on those guides flies with `controller.fast_guides = true`.

Guides are built with `smooth=True`, which follows the curve the plan really flies. Straight lines
between nodes can cut through a pole the plan goes around.

## 3. Checking the guide

Every solved guide is checked against the true collision shapes of gates and poles, the ones the
training hit check uses. It must keep at least 3 cm from all of them and stay inside the arena.

Up to 3 of the planner's initial guesses are tried, and the first clear plan is kept. A track with no
clear plan is dropped, so the library is biased away from layouts that are hard to plan. About 20-25%
of tracks are dropped this way.

The libraries behind the current `rl_scene.npz` were solved without the arena check, so about 14% of
their guides leave the arena, by up to 48 cm. The check applies to libraries solved from now on.

## 4. The library

`load_library` stacks the files into one set of arrays, padding every guide to the same length along
its run-out. Each track holds its guide (position, speed, velocity and attitude at every 5 cm), the
bead of every gate pass, and the nominal poses of gates and poles.

The current training library has 942 tracks from the race's generator and 597 from
`scene_generator.py`, and the validation library has 157 from the race's generator.

Tracks used for training, for choosing the kept policy and for the final test are kept apart:

| Use | Source | Seeds |
|---|---|---|
| Training | Race's generator and `scene_generator.py` | 400000+ and 500000+ |
| Choosing the kept policy | Race's generator | 300000+ |
| Final test, in the real race | Race's generator, inside the race itself | Race seeds 7100+ and 8000+ |

Layouts for the first two rows are drawn with a generator key from the seed. The real race derives its
own, so its tracks are new to the policy.

## 5. Starting an episode

Whenever a world's episode ends, it draws afresh:

| What | How |
|---|---|
| Track | Uniform from the library |
| Mass | +-5 g |
| Inertia | +-1e-6 kg m^2, each entry |
| Where it starts | 25% on the floor, the rest at a random point along the guide before the last pass |
| Start along the guide | Plan's attitude, rotors at hover speed, 50-100% of the plan's velocity, 5 cm position noise |
| Start on the floor | +-0.1 m in x and y, up to 2 cm higher, attitude +-0.1 rad, rotors stopped |
| Extra velocity | Up to +-0.5 m/s per axis, upwards only from the floor |
| Random force | +-0.1 N per axis, new at every simulator step, throughout the episode |

A start along the guide is tried up to 3 times to find one that touches nothing, and otherwise the
episode starts on the floor. An episode ends with a crash, the last pass, or after 500 control steps.

### Shifts of gates and poles

`--randomize` scales how far gates and poles move from where the guide was planned, as a share of what
level 2 does:

| What | At `--randomize 1` |
|---|---|
| Gate position | +-0.15 m in x and y, +-0.1 m in z |
| Gate roll, pitch, yaw | +-0.05, +-0.1, +-0.2 rad |
| Pole position | +-0.15 m in x and y, +-0.05 m in z |

Level 1 difficulty on random tracks means `--randomize 0`: the track is exactly where the guide says.
This also removes the shift of the floor start, so that start is exact during training. The race still
draws one, so the test covers something training doesn't.

At any other value the policy sees gates and poles as the race's sensor shows them: at nominal poses
until within 0.7 m in x and y, then at their true poses. Collisions and passes always use the true
poses.

## How to reproduce it

Solve the libraries:

```bash
pixi run -e gpu python scripts/solve_scene_library.py --count 1200 --source level3 --first_seed 400000 --out level3_train.npz
```

```bash
pixi run -e gpu python scripts/solve_scene_library.py --count 800 --first_seed 500000 --out generated_train.npz
```

```bash
pixi run -e gpu python scripts/solve_scene_library.py --count 200 --source level3 --first_seed 300000 --out validation.npz
```

Train at level 1 difficulty:

```bash
pixi run -e gpu python scripts/train_scene_policy.py --train level3_train.npz,generated_train.npz --validation validation.npz --randomize 0 --out policy.npz
```
