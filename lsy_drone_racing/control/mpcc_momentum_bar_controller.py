"""Sprint MPCC with observed solid gate bars and measured angular momentum."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from lsy_drone_racing.control.gate_bar_solver import (
    BODY_HALF_BOX,
    FRAME_MARGIN,
    GATE_BARS,
    GateBarSolver,
)
from lsy_drone_racing.control.momentum_bar_solver import get_momentum_bar_solver
from lsy_drone_racing.control.mpcc_corridor_controller import MPCCCorridorController


class MPCCMomentumBarController(MPCCCorridorController):
    """Predict the remaining angular motion when planning gate entry and exit."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Keep the shared sprint pace and activate measured-rate prediction on Level2."""
        tuned = deepcopy(config)
        if "name" not in tuned.controller:
            tuned.controller.name = "mpcc_momentum_bar"
        if not tuned.controller.get("corridor_second_order", True):
            raise ValueError("Momentum gate-bar prediction requires second-order attitude")
        if tuned.controller.get("corridor_velocity_weight", 0.0) != 0.0:
            raise ValueError("Momentum gate-bar prediction uses the inherited MPCC cost")
        tuned.controller.corridor_second_order = True
        super().__init__(obs, info, tuned)

    def _make_observed_solver(self, obs: dict) -> GateBarSolver:
        """Use an isolated eighteen-state library and the shared two-refinement adapter."""
        backend, _ = get_momentum_bar_solver(
            self._mpcc, self._params, len(obs["obstacles_pos"]), len(obs["gates_pos"])
        )
        return GateBarSolver(
            backend,
            lambda: self._current_obs,
            self._policy["l2_constraint_radius"],
            self._policy["l2_uncertainty_buffer"],
            lambda: self._last_v_ref,
        )

    def settings_snapshot(self) -> dict[str, Any]:
        """Record the physical frame, body inflation and measured-rate dynamics."""
        settings = super().settings_snapshot()
        settings.pop("corridor", None)
        settings["gate_bars"] = {
            "formulation": "momentum_v1",
            "enabled": self._level2,
            "second_order_attitude": self._second_order_active,
            "body_half_box": BODY_HALF_BOX.tolist(),
            "margin": FRAME_MARGIN,
            "primitive_gate_bars": GATE_BARS,
            "rti_iterations": 2,
            "inflation_source": "max observed and warm-start forecast body extents",
        }
        return settings
