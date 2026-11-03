"""Experimental sprint MPCC with explicit observed gate entry/exit corridors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.corridor_solver import CorridorSolver, get_corridor_solver
from lsy_drone_racing.control.mpcc_level2_controller import MPCCLevel2Controller
from lsy_drone_racing.control.my_controller import MPCCConfig

CORRIDOR_PACE = {
    "v_max": 5.0,
    "a_lat_max": 22.0,
    "w_speed_gate": 2.0,
    "v_lead": 1.2,
    "q_contour_peak": 10000.0,
    "l2_unseen_speed": 2.5,
    "replan_slowdown_time": 0.06,
}


def euler_rates(quat: np.ndarray, body_rates: np.ndarray) -> np.ndarray:
    """Convert measured body angular velocity into XYZ Euler-angle rates."""
    roll, pitch, _ = Rotation.from_quat(quat).as_euler("xyz")
    cosine = np.cos(pitch)
    cosine = np.copysign(max(abs(cosine), 1e-3), cosine)
    sine, cosine_roll = np.sin(roll), np.cos(roll)
    tangent = np.sin(pitch) / cosine
    transform = np.array(
        [
            [1, sine * tangent, cosine_roll * tangent],
            [0, cosine_roll, -sine],
            [0, sine / cosine, cosine_roll / cosine],
        ]
    )
    return transform @ np.asarray(body_rates, dtype=float)


class MPCCCorridorController(MPCCLevel2Controller):
    """Use gate corridors without changing old entries, sensors or physical limits."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Install the dedicated solver after the inherited lifecycle initializes."""
        tuned = deepcopy(config)
        self._second_order_active = False
        self._velocity_weight = float(config.controller.get("corridor_velocity_weight", 0.0))
        for key, value in CORRIDOR_PACE.items():
            if key not in tuned.controller:
                tuned.controller[key] = value
        randomizations = config.env.get("randomizations") or {}
        randomized = any(key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos"))
        bootstrap = deepcopy(tuned)
        requested = {
            key: tuned.controller[key]
            for key in MPCCConfig.__dataclass_fields__
            if key in tuned.controller
        }
        if randomized:
            # The inherited initialization solver writes one shared build directory.
            # Give it one fixed shape/name; only the isolated solver below executes control.
            for key in MPCCConfig.__dataclass_fields__:
                if key in bootstrap.controller:
                    del bootstrap.controller[key]
            bootstrap.controller.name = "corridor_initialization"
        super().__init__(obs, info, bootstrap)
        if self._level2:
            requested.setdefault("name", "mpcc_corridor")
            self._mpcc = replace(self._mpcc, **requested)
            self._config = tuned
            self._reset_mass_history()
            self._build_speed_limit()
            self._second_order_active = bool(config.controller.get("corridor_second_order", False))
            self._solver = self._make_observed_solver(obs)
            self._obstacle_solver_installed = True
            self._solver.reset()
            self._apply_progress_bound()
            self._seed_routed_solver(obs)

    def _make_observed_solver(self, obs: dict) -> CorridorSolver:
        """Allow independent geometry entries to reuse the safe initialization lifecycle."""
        backend, _ = get_corridor_solver(
            self._mpcc,
            self._params,
            len(obs["obstacles_pos"]),
            len(obs["gates_pos"]),
            self._second_order_active,
            self._velocity_weight,
        )
        return CorridorSolver(
            backend,
            lambda: self._current_obs,
            self._policy["l2_constraint_radius"],
            self._policy["l2_uncertainty_buffer"],
            lambda: self._last_v_ref,
        )

    def _state_vector(self, obs: dict, theta: float) -> np.ndarray:
        """Initialize the added momentum states from observation, not command history."""
        state = super()._state_vector(obs, theta)
        if self._second_order_active:
            state = np.r_[state, euler_rates(obs["quat"], obs["ang_vel"])]
        return state

    def settings_snapshot(self) -> dict[str, Any]:
        """Expose prediction and corridor formulation for reproducible experiments."""
        settings = super().settings_snapshot()
        settings["corridor"] = {
            "formulation": 8,
            "second_order_attitude": self._second_order_active,
            "body_half_box": [0.07, 0.07, 0.02],
            "frame_margin": 0.015,
            "normal_release_power": 4,
            "margin_source": "max observed and warm-start forecast body extents",
            "velocity_weight": self._velocity_weight,
            "activation_source": "reference or forecast body; observed body at first three stages",
        }
        return settings

    def diagnostics(self) -> dict[str, float]:
        """Keep nonfinite recoveries visible in addition to native failure statuses."""
        return {
            **super().diagnostics(),
            "nonfinite_recoveries": float(getattr(self._solver, "nonfinite_recoveries", 0)),
        }
