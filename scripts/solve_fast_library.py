"""Plan fast guides for the tracks of a solved library, ready for `scripts/train_scene_policy.py`.

The tracks are those of the given library, so that a policy trained on the result compares
fairly with one trained on the solved guides. A track whose fast plan fails its check is dropped
and counted.

Bang-bang acceleration jumps, and the attitude of a guide comes from its acceleration. The
velocities are therefore smoothed over a short window before they are written. Positions stay as
planned, so the path and its clearance do not change.

Example::

    $ pixi run -e gpu python scripts/solve_fast_library.py \
        --library .tmp/topt/rl2/scenes/fixed_level3_validation.npz --out fast_validation.npz
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import lsy_drone_racing  # noqa: F401, I001  # must come first: sets up scipy for crazyflow
import time
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.topt_planner import Track
from lsy_drone_racing.utils.fast_planner import FastPlannerConfig, plan_guide, smooth_velocities
from lsy_drone_racing.utils.race_policy import TAKEOFF_LIFT, TAKEOFF_SPEED

SEQUENCE = np.array([0, 1, 2, 3, 1])
LAYOUT = ("start", "gates_pos", "gates_rpy", "poles")


def main(
    library: str,
    out: str,
    steps: int = 60,
    thrust: float = 15.0,
    tilt: float = 1.1,
    velocity_window: float = 0.15,
):
    """Plan every track of a library.

    Args:
        library: A library written by `scripts/solve_scene_library.py`, which gives the tracks.
        out: Where to write the fast guides.
        steps: Gradient steps per try.
        thrust: Thrust limit of the segments [m/s^2].
        tilt: Tilt limit of the segments [rad].
        velocity_window: Width of the smoothing of the velocities [s].
    """
    config = FastPlannerConfig(steps=steps, thrust=thrust, tilt=tilt)
    t0 = time.time()
    arrays, kept = {}, 0
    with np.load(library) as data:
        total = int(data["count"])
        for i in range(total):
            layout = {k: data[f"{k}_{i}"] for k in LAYOUT}
            track = Track(
                layout["gates_pos"],
                Rotation.from_euler("xyz", layout["gates_rpy"]).as_matrix(),
                layout["poles"],
                SEQUENCE,
                np.ones(5),
                tuple(False for _ in SEQUENCE),
            )
            first = layout["start"] + np.array([0.0, 0.0, TAKEOFF_LIFT])
            plan, _ = plan_guide(track, first, np.array([0.0, 0.0, TAKEOFF_SPEED]), config)
            if not plan.clear:
                continue
            t = plan.trajectory
            arrays |= {
                f"pos_{kept}": t.pos,
                f"vel_{kept}": smooth_velocities(t.vel, t.time, velocity_window),
                f"time_{kept}": t.time,
                f"crossings_{kept}": t.crossings,
                **{f"{k}_{kept}": v for k, v in layout.items()},
            }
            kept += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, count=kept, **arrays)
    print(f"kept {kept}/{total} tracks in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
