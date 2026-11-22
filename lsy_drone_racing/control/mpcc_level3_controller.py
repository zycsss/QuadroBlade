"""Level 3 (randomized track) controller built on the real-proven level 2 sprint MPCC.

Run without changing a provided configuration::

    pixi run python scripts/sim.py --config level3.toml \
        --controller mpcc_level3_controller.py

Everything level 3 specific lives in this file and in
[level3_planning][lsy_drone_racing.control.level3_planning]: the level 2 entry
(``mpcc_reliable_sprint_controller.py``) and every module it uses keep their behaviour. On a track
without ``env.track.randomize`` this controller is the level 2 sprint controller.

The level 2 parent rejects ``env.track.randomize = true``. Planning does not read the
``[env.track]`` entries of ``level3.toml`` (placeholders at the origin); it uses the observation,
which reports the generated layout until a gate or pole is in sensor range and its true pose
afterwards. The parent is therefore handed a config copy with the flag cleared.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import fields, replace

import numpy as np

from lsy_drone_racing.control.level3_planning import Level3Planner, Level3Settings
from lsy_drone_racing.control.mpcc_controller import MPCCBaselineController
from lsy_drone_racing.control.mpcc_reliable_sprint_controller import MPCCReliableSprintController

#: Level 3 controller settings. Each applies only when ``env.track.randomize`` is true and the
#: ``[controller]`` section does not name the key itself.
LEVEL3_CONTROLLER: dict[str, object] = {"a_lat_max": 14.0}
#: Level 3 overrides of the level 2 planner configuration (`PathConfig` fields).
LEVEL3_PATH: dict[str, object] = {}
#: Level 3 only planner settings (`Level3Settings` fields).
LEVEL3_PLANNER: dict[str, object] = {
    "bounds_margin": 0.25,
    "floor": 0.15,
    "backside_lateral": 0.7,
    "block_openings": True,
    "detour_flip": True,
}
#: Stall watchdog: re-seed the solver along the path when progress over ``stall_window`` seconds
#: stays below ``stall_progress`` metres. Zero window disables it.
LEVEL3_STALL = {"stall_window": 0.6, "stall_progress": 0.05}
#: Speed cap [m/s] while the next gate in the sequence is unseen and within ``target_radius``
#: metres (XY). Zero disables it; the level 2 unseen-object cap stays active either way.
LEVEL3_TARGET = {"target_speed": 1.8, "target_radius": 1.4}


class MPCCLevel3Controller(MPCCReliableSprintController):
    """The level 2 sprint controller with level 3 planning rules on randomized tracks."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Apply the level 3 settings on a private config copy, then build the level 2 stack."""
        tuned = deepcopy(config)
        self._level3 = bool(tuned.env.track.get("randomize", False))
        if self._level3:
            tuned.env.track.randomize = False
            for key, value in LEVEL3_CONTROLLER.items():
                if key not in tuned.controller:
                    tuned.controller[key] = value
        self._l3_path, self._l3_settings = {}, Level3Settings()
        if self._level3:
            ctrl = tuned.controller
            path_keys = {f.name for f in fields(self._path_config_type())}
            self._l3_path = {
                k: ctrl.get(f"l3_{k}", v) for k, v in LEVEL3_PATH.items()
            } | {k[3:]: v for k, v in ctrl.items() if k.startswith("l3_") and k[3:] in path_keys}
            settings = {k: ctrl.get(f"l3_{k}", v) for k, v in LEVEL3_PLANNER.items()}
            settings |= {
                k[3:]: v
                for k, v in ctrl.items()
                if k.startswith("l3_") and k[3:] in Level3Settings.__dataclass_fields__
            }
            self._l3_settings = Level3Settings(**settings)
        self._stall = {k: float(tuned.controller.get(f"l3_{k}", v)) for k, v in LEVEL3_STALL.items()}
        self.reseeds = 0
        self._target = {k: float(tuned.controller.get(f"l3_{k}", v)) for k, v in LEVEL3_TARGET.items()}
        super().__init__(obs, info, tuned)
        self._theta_hist: deque[float] = deque(maxlen=max(1, int(self._stall["stall_window"] * self._freq)))

    @staticmethod
    def _path_config_type() -> type:
        from lsy_drone_racing.control.path_planning import PathConfig

        return PathConfig

    def _init_episode(self, obs: dict, reason: str = "episode start") -> None:
        """Use the level 3 planner on a randomized track, the level 2 path otherwise."""
        if not (self._level3 and self._level2):
            super()._init_episode(obs, reason)
            return
        self._current_obs = obs
        config = replace(self._planner.config, **self._l3_path)
        limits = self._config.env.track.get("safety_limits") or {}
        self._planner = Level3Planner(
            config,
            self._policy["l2_post_clearance"],
            self._l3_settings,
            limits.get("pos_limit_low"),
            limits.get("pos_limit_high"),
        )
        MPCCBaselineController._init_episode(self, obs, reason)
        if self._obstacle_solver_installed:
            self._seed_routed_solver(obs)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Run the level 2 step; on level 3 re-seed a solver that stopped making progress."""
        action = super().compute_control(obs, info)
        if self._level3 and self._stall["stall_window"] > 0 and self._obstacle_solver_installed:
            hist = self._theta_hist
            hist.append(float(self._theta))
            near_end = self._path is not None and self._theta > self._path.length - 0.3
            if (
                len(hist) == hist.maxlen
                and hist[-1] - hist[0] < self._stall["stall_progress"]
                and not near_end
            ):
                self._reseed(obs)
                self.reseeds += 1
                hist.clear()
        return action

    def risk_speed_limit(self, obs: dict) -> float | None:
        """Add a level 3 cap while the next gate is unseen and close."""
        limit = super().risk_speed_limit(obs)
        if not self._level3 or self._target["target_speed"] <= 0:
            return limit
        sequence = np.asarray(obs["gate_sequence"], dtype=int).ravel()
        passed = int(np.asarray(obs.get("n_gates_passed", 0)).ravel()[0])
        target = int(sequence[passed]) if 0 <= passed < len(sequence) else -1
        visited = np.asarray(obs.get("gates_visited", []), dtype=bool).ravel()
        if 0 <= target < len(visited) and not visited[target]:
            gate = np.asarray(obs["gates_pos"], dtype=float)[target]
            if np.linalg.norm(gate[:2] - np.asarray(obs["pos"], float)[:2]) < self._target[
                "target_radius"
            ]:
                cap = self._target["target_speed"]
                limit = cap if limit is None else min(limit, cap)
        return limit

    def _reseed(self, obs: dict) -> None:
        """Reset the solver and warm-start it on the path ahead of the current progress."""
        self._solver.reset()
        speed = max(float(self._last_v_ref), self._mpcc.v_min)
        for k in range(self._mpcc.N + 1):
            theta = min(self._path.length, self._theta + k * self._mpcc.dt * speed)
            x = self._state_vector(obs, theta)
            x[:3] = self._path.pos(theta)
            x[3:6] = self._path.tangent(theta) * speed
            self._solver.set(k, "x", x)
            if k < self._mpcc.N:
                self._solver.set(k, "u", np.r_[np.zeros(4), speed])

    def diagnostics(self) -> dict[str, float]:
        """Add the number of stall re-seeds to the level 2 diagnostics."""
        return {**super().diagnostics(), "reseeds": float(self.reseeds)}
