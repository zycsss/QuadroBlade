"""A learned racing controller: state, guide path, gates and obstacles in, attitude and thrust out.

The policy is trained by `scripts/train_race_policy.py`. The guide path comes from `topt_planner`
but only informs the policy, which decides for itself whether to follow it. Attitude goes in and out
as a rotation matrix, and the output is converted to the angles the attitude controller takes.

The guide is planned once on the nominal track. The gates and poles come from the race's sensor, so
the policy sees them where the race shows them: nominal at first, and true once they are in range.

Point ``controller.policy`` in the configuration to another file to fly a different policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.utils.path_policy import ACT_DIM, PathPolicy, to_action
from lsy_drone_racing.utils.race_policy import advance_guide, build_track, race_features

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

#: Trained on the one nominal track: faster there, and useless once gates move.
FIXED_TRACK_POLICY = Path(__file__).parent / "rl_race_level0.npz"
#: Trained with gates, poles and starts moved as on level 2.
MOVED_TRACK_POLICY = Path(__file__).parent / "rl_race_randomized.npz"


class RLRace(Controller):
    """Flies the track with a trained policy."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Load the policy and the guide path.

        Args:
            obs: The initial observation.
            info: The initial info dict.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        section = dict(config.get("controller") or {})
        randomizations = config.env.get("randomizations") or {}
        moved = any(k in randomizations for k in ("gate_pos", "gate_rpy", "obstacle_pos"))
        default = MOVED_TRACK_POLICY if moved else FIXED_TRACK_POLICY
        self._policy = PathPolicy(section.get("policy", default))
        self._track = build_track(config.env.track)
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
        passed = np.array([max(int(obs["n_gates_passed"]), 0)])
        passed = np.minimum(passed, len(self._track.sequence) - 1)
        self._index, _ = advance_guide(pos, self._index, passed, self._track)
        sensed = self._track._replace(
            gates_pos=np.asarray(obs["gates_pos"], dtype=float),
            gates_rot=Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_matrix(),
            poles=np.asarray(obs["obstacles_pos"], dtype=float),
        )
        x = race_features(pos, vel, quat, ang_vel, self._previous, self._index, passed, sensed)
        u = np.clip(self._policy(x), -1.0, 1.0)
        self._previous = u.astype(np.float32)
        return to_action(u, self._policy.yaw_max)[0].astype(np.float32)
