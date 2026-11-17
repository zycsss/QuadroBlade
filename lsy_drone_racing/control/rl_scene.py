"""A racing controller trained only on random local scenes, never on the track it flies.

The policy is trained by `scripts/train_scene_policy.py` and sees nothing that identifies a track,
see `lsy_drone_racing.utils.scene_policy`. The guide is planned once on the nominal track, and the
gates and poles come from the race's sensor: nominal at first, and true once they are in range.

Point ``controller.policy`` in the configuration to another file to fly a different policy. A policy
trained on faster guides needs ``controller.fast_guides = true`` as well, to fly the same kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.topt_planner import FAST_GUIDE_CONFIG, RANDOM_TRACK_CONFIG
from lsy_drone_racing.utils.path_policy import ACT_DIM, PathPolicy
from lsy_drone_racing.utils.race_policy import build_track
from lsy_drone_racing.utils.scene_policy import (
    advance_scene_guide,
    scene_action,
    scene_features,
    scene_heading,
    track_library,
)

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

DEFAULT_POLICY = Path(__file__).parent / "rl_scene.npz"


def _reported_track(obs: dict[str, NDArray[np.floating]], track: ConfigDict) -> dict:
    """Return the track to plan on: the configured one, or the generated one the race reports.

    On a generated track the configuration only holds placeholders. What the first observation shows
    is the generated layout, nominal where the sensor has not reached yet.

    Args:
        obs: The initial observation.
        track: The ``env.track`` section of the configuration.

    Returns:
        A track section for `build_track`.
    """
    if not track.get("randomize", False):
        return track
    rpy = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_euler("xyz")
    gates = np.asarray(obs["gates_pos"], dtype=float)
    return {
        "gates": [{"pos": p, "rpy": r} for p, r in zip(gates, rpy, strict=True)],
        "obstacles": [{"pos": p} for p in np.asarray(obs["obstacles_pos"], dtype=float)],
        "drones": [{"pos": np.asarray(obs["pos"], dtype=float)}],
        "gate_order": track["gate_order"],
        "safety_limits": track["safety_limits"],
    }


class RLScene(Controller):
    """Flies the track with a policy trained on other tracks."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Load the policy and plan the guide.

        Args:
            obs: The initial observation.
            info: The initial info dict.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        section = dict(config.get("controller") or {})
        self._policy = PathPolicy(section.get("policy", DEFAULT_POLICY))
        generated = config.env.track.get("randomize", False)
        limits = FAST_GUIDE_CONFIG if section.get("fast_guides", False) else RANDOM_TRACK_CONFIG
        track = build_track(
            _reported_track(obs, config.env.track), config=limits if generated else None
        )
        rpy = Rotation.from_matrix(track.gates_rot).as_euler("xyz")
        self._library = track_library(track, rpy)
        self._rows = np.zeros(1, dtype=int)
        self.episode_reset()

    def episode_reset(self):
        """Start again at the beginning of the guide."""
        self._index = np.zeros(1, dtype=int)
        self._previous = np.zeros((1, ACT_DIM), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Keep flying; the environment ends the race."""
        return False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, thrust]`` for the current observation."""
        keys = ("pos", "vel", "quat", "ang_vel")
        pos, vel, quat, ang_vel = (np.asarray(obs[k], dtype=float)[None] for k in keys)
        lib = self._library
        passed = np.array([max(int(obs["n_gates_passed"]), 0)])
        passed = np.minimum(passed, len(lib.sequence) - 1)
        self._index, _ = advance_scene_guide(pos, self._index, passed, self._rows, lib)
        gates_rot = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_matrix()
        x = scene_features(
            pos,
            vel,
            quat,
            ang_vel,
            self._previous,
            self._index,
            passed,
            self._rows,
            np.asarray(obs["gates_pos"], dtype=float)[None],
            gates_rot[None],
            np.asarray(obs["obstacles_pos"], dtype=float)[None],
            lib,
        )
        u = np.clip(self._policy(x), -1.0, 1.0)
        self._previous = u.astype(np.float32)
        c, s = scene_heading(self._index, self._rows, lib)
        return scene_action(u, c, s)[0].astype(np.float32)
