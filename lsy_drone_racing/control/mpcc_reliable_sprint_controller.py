"""Reliable Level2 pace with measured angular momentum and observed solid gate bars."""

from __future__ import annotations

from copy import deepcopy

from lsy_drone_racing.control.mpcc_momentum_bar_controller import MPCCMomentumBarController


class MPCCReliableSprintController(MPCCMomentumBarController):
    """Keep the nominal sprint; reduce randomized reference lead to improve completion."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Apply the training-selected Level2 lead on a private configuration copy."""
        tuned = deepcopy(config)
        randomizations = tuned.env.get("randomizations") or {}
        randomized = any(key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos"))
        if randomized and "v_lead" not in tuned.controller:
            tuned.controller.v_lead = 1.0
        super().__init__(obs, info, tuned)
