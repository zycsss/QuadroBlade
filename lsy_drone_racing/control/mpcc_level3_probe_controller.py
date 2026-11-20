"""Probe: fly the level 2 sprint controller unchanged on a level 3 (fully randomized) track.

``MPCCLevel2Controller`` refuses configs with ``env.track.randomize = true``. This probe hides only
that flag from the controller so its level 2 behaviour can be measured on level 3. Everything the
controller plans with still comes from the observation, which reports the generated layout (gates
and obstacles at their nominal generated poses until they come into sensor range). The ``[env.track]``
entries in ``level3.toml`` are placeholders at the origin; the planner does not read them.
"""

from __future__ import annotations

from copy import deepcopy

from lsy_drone_racing.control.mpcc_reliable_sprint_controller import MPCCReliableSprintController


class MPCCLevel3ProbeController(MPCCReliableSprintController):
    """The level 2 sprint controller with the level 3 rejection bypassed (diagnostic only)."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Pass a private config copy with the track randomization flag cleared."""
        probe = deepcopy(config)
        probe.env.track.randomize = False
        super().__init__(obs, info, probe)
