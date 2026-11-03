"""Reliability-first Level2 MPCC with observed obstacle constraints and reveal-aware pacing.

Run without changing a provided configuration::

    pixi run python scripts/sim.py --config level2.toml \
        --controller mpcc_level2_controller.py

All control inputs come from observations and nominal configuration. Level0/1 use the inherited
baseline settings; fully randomized Level3 is rejected until separately validated.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from lsy_drone_racing.control.level2_solver import ObservedObstacleSolver, get_obstacle_solver
from lsy_drone_racing.control.mpcc_controller import MPCCBaselineController
from lsy_drone_racing.control.path_planning import ArcPath, PathInfo, PathPlanner

LEVEL2_POLICY = {
    "l2_unseen_speed": 1.4,
    "l2_unseen_radius": 1.0,
    "l2_pole_speed": 0.0,  # Optional additional known-pole cap; zero disables it.
    "l2_pole_radius": 0.65,
    "l2_post_clearance": 0.18,
    "l2_constraint_radius": 0.18,
    "l2_uncertainty_buffer": 0.10,
    "l2_projection_window": 1.8,
}


def enforce_pole_clearance(path: ArcPath, obstacles: np.ndarray, radius: float) -> ArcPath:
    """Repair a near-clear reference after gate-frame adjustments, then check the final spline.

    This local projection is not a general routing algorithm. A central crossing or another
    unrepaired geometry raises instead of silently committing a known-invalid clearance claim.
    """
    if not len(obstacles) or radius <= 0:
        return path
    poles = np.asarray(obstacles, dtype=float)
    for _ in range(3):
        points = np.array(path.pos(np.linspace(0, path.length, max(32, int(path.length / 0.01)))))
        delta = points[:, None, :2] - poles[None, :, :2]
        distance = np.linalg.norm(delta, axis=-1)
        nearest = np.argmin(distance, axis=1)
        close = distance[np.arange(len(points)), nearest] < radius
        if not close.any():
            break
        direction = delta[np.arange(len(points)), nearest]
        norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        points[close, :2] = (
            poles[nearest[close], :2] + (radius + 0.005) * direction[close] / norm[close]
        )
        path = ArcPath(points)
    samples = path.pos(np.linspace(0, path.length, max(64, int(path.length / 0.005))))
    if PathPlanner.clearance(samples, poles) < radius - 0.003:
        raise ValueError("local pole clearance repair failed; a different route is required")
    return path


class ClearancePlanner(PathPlanner):
    """Retain existing routing and apply a final local pole-clearance check."""

    def __init__(self, config: object, radius: float):
        """Configure the original planner and the final reference clearance."""
        super().__init__(config)
        self.radius = radius

    def plan(self, obs: dict, start_pos: np.ndarray) -> tuple[ArcPath, PathInfo]:
        """Repair only the finished reference, preserving observed signed gate metadata."""
        path, info = super().plan(obs, start_pos)
        repaired = enforce_pole_clearance(path, info.obstacles, self.radius)
        if repaired is not path:
            info = PathInfo(repaired.points.copy(), info.gate_centers, info.obstacles)
        return repaired, info


class MPCCLevel2Controller(MPCCBaselineController):
    """Add observed collision constraints while preserving the original lifecycle and physics."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Copy requested settings, initialize the framework, then install the dedicated solver."""
        if config.env.track.get("randomize", False):
            raise ValueError("Level 3 full-track randomization needs separate validation")
        randomizations = config.env.get("randomizations") or {}
        self._level2 = any(k in randomizations for k in ("gate_pos", "gate_rpy", "obstacle_pos"))
        tuned = deepcopy(config)
        self._policy = {
            key: tuned.controller.get(key, value) for key, value in LEVEL2_POLICY.items()
        }
        self._current_obs = obs
        self._risk_limit = 0.0
        self._projection_correction = 0.0
        self._obstacle_solver_installed = False
        if self._level2 and "name" not in tuned.controller:
            tuned.controller.name = "mpcc_level2"
        # Let the unchanged base class initialize parameters, references and episode memory.
        super().__init__(obs, info, tuned)
        if self._level2 and len(obs["obstacles_pos"]):
            backend, _ = get_obstacle_solver(self._mpcc, self._params, len(obs["obstacles_pos"]))
            self._solver = ObservedObstacleSolver(
                backend,
                lambda: self._current_obs,
                self._policy["l2_constraint_radius"],
                self._policy["l2_uncertainty_buffer"],
            )
            self._obstacle_solver_installed = True
            self._solver.reset()
            self._apply_progress_bound()
            self._seed_routed_solver(obs)

    def _seed_routed_solver(self, obs: dict) -> None:
        """Initialize the nonconvex obstacle linearization on the routed side of each pole."""
        for k in range(self._mpcc.N + 1):
            theta = min(self._path.length, k * self._mpcc.dt)
            x = self._state_vector(obs, theta)
            x[:3] = self._path.pos(theta)
            x[3:6] = self._path.tangent(theta)
            self._solver.set(k, "x", x)
            if k < self._mpcc.N:
                self._solver.set(k, "u", np.r_[np.zeros(4), 1.0])

    def _init_episode(self, obs: dict, reason: str = "episode start") -> None:
        """Use the local reference repair only for randomized gate/obstacle layouts."""
        self._current_obs = obs
        if self._level2:
            self._planner = ClearancePlanner(
                self._planner.config, self._policy["l2_post_clearance"]
            )
        super()._init_episode(obs, reason)
        if self._obstacle_solver_installed:
            self._seed_routed_solver(obs)

    def _seed_visited_flags(self, obs: dict) -> None:
        """Remember orientation as well as position for future reveal detection."""
        super()._seed_visited_flags(obs)
        self._observed_gate_quat = np.asarray(obs["gates_quat"], dtype=float).copy()

    def _track_info_changed(self, obs: dict) -> list[str]:
        """Detect quaternion-only updates without treating q/-q as a physical change."""
        changes = super()._track_info_changed(obs)
        current = np.asarray(obs["gates_quat"], dtype=float)
        previous = self._observed_gate_quat
        self._observed_gate_quat = current.copy()
        if self._level2 and current.shape == previous.shape:
            difference = np.minimum(
                np.linalg.norm(current - previous, axis=1),
                np.linalg.norm(current + previous, axis=1),
            )
            changes.extend(
                f"gate {i} orientation changed" for i in np.flatnonzero(difference > 1e-6)
            )
        return changes

    def _record_plan(self, reason: str, obs: dict) -> None:
        """Reproject in the new path's coordinates and align replans to observation time."""
        replan = self._level2 and self._reproject_wide
        if replan:
            old = self._theta
            window = self._policy["l2_projection_window"]
            self._theta = float(self._path.closest(obs["pos"], old, window=(-window, window)))
            self._projection_correction = self._theta - old
            self._reproject_wide = False
        super()._record_plan(reason, obs)
        if replan:
            self.plan_log[-1]["tick"] = max(0, self._tick - 1)
            self.plan_log[-1]["t"] = max(0, self._tick - 1) / self._freq

    def risk_speed_limit(self, obs: dict) -> float | None:
        """Return an observation-only cap near unseen geometry or optional known-pole hazards."""
        if not self._level2:
            return None
        limits = []
        pos = np.asarray(obs["pos"], dtype=float)
        for key, flags in (("gates_pos", "gates_visited"), ("obstacles_pos", "obstacles_visited")):
            points = np.asarray(obs[key], dtype=float).reshape(-1, 3)
            visited = np.asarray(obs.get(flags, np.zeros(len(points), dtype=bool)), dtype=bool)
            if len(points) and np.any(~visited):
                distance = np.linalg.norm(points[~visited, :2] - pos[:2], axis=1)
                if np.any(distance < self._policy["l2_unseen_radius"]):
                    limits.append(self._policy["l2_unseen_speed"])
        poles = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
        if self._policy["l2_pole_speed"] > 0 and len(poles):
            if np.any(
                np.linalg.norm(poles[:, :2] - pos[:2], axis=1) < self._policy["l2_pole_radius"]
            ):
                limits.append(self._policy["l2_pole_speed"])
        return float(min(limits)) if limits else None

    def _reference_speed(self, theta: float, speed_actual: float) -> float:
        """Combine the inherited curvature/proximity speed with the reveal-aware cap."""
        speed = super()._reference_speed(theta, speed_actual)
        limit = self.risk_speed_limit(self._current_obs)
        self._risk_limit = limit if limit is not None else 0.0
        return min(speed, limit) if limit is not None else speed

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Refresh live observations before inherited planning and solver parameter writes."""
        self._current_obs = obs
        self._projection_correction = 0.0
        return super().compute_control(obs, info)

    def settings_snapshot(self) -> dict[str, Any]:
        """Export effective policy as well as the inherited solver and planner settings."""
        settings = super().settings_snapshot()
        settings["level2"] = {"enabled": self._level2, **self._policy}
        return settings

    def diagnostics(self) -> dict[str, float]:
        """Expose risk pacing and coordinate repair alongside native solver diagnostics."""
        return {
            **super().diagnostics(),
            "risk_speed_cap": self._risk_limit,
            "projection_correction": self._projection_correction,
        }
