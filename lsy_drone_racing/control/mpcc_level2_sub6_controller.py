"""Fixed-seed Level2 speed profile; randomized reliability is not established.

Run without changing the provided track::

    python -m tests.mpcc_evaluation --config config/level2.toml --seed 0 \
        --controller mpcc_level2_sub6_controller.py

The observed-geometry planner, solver, physical limits and lifecycle are inherited.
Only pace defaults change. This profile was selected on seed 0, not on a holdout set.
"""

from __future__ import annotations

from copy import deepcopy

from lsy_drone_racing.control.mpcc_level2_controller import MPCCLevel2Controller

SUB6_MPCC = {
    "v_max": 5.0,
    "a_lat_max": 14.0,
    "w_speed_gate": 2.5,
    "v_lead": 1.0,
    "l2_unseen_speed": 1.8,
    "replan_slowdown_time": 0.15,
}


class MPCCLevel2Sub6Controller(MPCCLevel2Controller):
    """Apply opt-in Level2 pace defaults while honoring explicit caller overrides."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Fill a private configuration copy; nominal levels retain inherited defaults."""
        tuned = deepcopy(config)
        randomizations = config.env.get("randomizations") or {}
        if any(key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos")):
            for key, value in SUB6_MPCC.items():
                if key not in tuned.controller:
                    tuned.controller[key] = value
        super().__init__(obs, info, tuned)
