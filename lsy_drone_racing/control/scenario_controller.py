"""Controller that follows an analytic scenario path instead of a planned track.

[MPCCController][lsy_drone_racing.control.my_controller.MPCCController] is built around following
a path that a planner produced from the track. For characterising the drone that is in the way: to
measure what the airframe can do, the reference has to be a line, a circle or a corner of known
geometry, not whatever the planner decides.

This controller keeps every part of the racing controller — the optimal control problem, the
solver, the progress projection, the speed limits, the logged diagnostics — and replaces only the
path and the reference weighting. It is a measuring instrument, not a racing controller, and it is
never referenced from a race config.

The scenario is chosen in the config:

```toml
[controller]
file = "scenario_controller.py"
scenario = "circle radius=1.0 speed=1.8"
```

``speed`` asks for a path speed that the curvature limit still protects; ``overload`` asks for a
speed that ignores the curvature limit, which is how the real lateral acceleration limit is found.
"Scenario" parameters are documented in [scenario_path.build][
lsy_drone_racing.control.scenario_path.build].
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.my_controller import MPCCController
from lsy_drone_racing.control.path_planning import PathInfo
from lsy_drone_racing.control.scenario_path import build, controller_overrides, parse_spec

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Default wall of time after which a scenario is cut short, in seconds.
DEFAULT_TIMEOUT = 20.0

#: Arc length before the end of the path at which the scenario counts as flown, in meters.
FINISH_MARGIN = 0.02


class ScenarioController(MPCCController):
    """Follow an analytic path and report when the manoeuvre is finished."""

    def _init_episode(self, obs: dict[str, NDArray[np.floating]], reason: str = ""):
        """Install the scenario path and reset all controller memory.

        Args:
            obs: The initial observation.
            reason: Ignored; the log records the scenario specification instead.
        """
        controller_cfg = self._config.get("controller") or {}
        spec = str(controller_cfg.get("scenario", "line"))
        path = build(spec)
        self._target_speed = self._spec_value(spec, "speed")
        self._overload_speed = self._spec_value(spec, "overload")
        # A scenario sets its own speed; the MPCC still protects the drone unless this is an
        # overload test, so `v_max` has to allow the request for the clip not to swallow it.
        requested = max(
            value for value in (self._target_speed, self._overload_speed, 0.0) if value is not None
        )
        overrides = controller_overrides(spec)
        if requested > 0.0:
            overrides["v_max"] = max(float(self._mpcc.v_max), requested)
        self._mpcc = replace(self._mpcc, **overrides)
        self._spec = spec
        self.completed = False
        self._timeout_ticks = int(
            round(float(controller_cfg.get("scenario_timeout", DEFAULT_TIMEOUT)) * self._freq)
        )
        info = PathInfo(
            waypoints=path.pos(np.linspace(0.0, path.length, 60)),
            gate_centers=np.zeros((0, 3)),
            gate_ids=np.zeros(0, dtype=int),
            obstacles=np.zeros((0, 3)),
        )
        self._install_path(path, info)
        logger.info("Scenario %r: %.2f m from %s", spec, path.length, np.round(path.pos(0.0), 2))
        self._reset_state(obs, f"scenario: {spec}")

    @staticmethod
    def _spec_value(spec: str, key: str) -> float | None:
        """Return a numeric scenario parameter, or ``None`` when the spec does not set it."""
        return parse_spec(spec)[1].get(key)

    def _reference(
        self, thetas: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate the reference along the scenario path with uniform contouring pressure.

        A scenario has no gates to centre on, so the Gaussian weight allocation of the racing
        controller would leave the contouring weight at zero between them. Full weight everywhere
        is what makes the tracking error a property of the drone rather than of the weighting.

        Args:
            thetas: Arc length parameters, shape ``(n, )``.

        Returns:
            Reference positions ``(n, 3)``, unit tangents ``(n, 3)`` and weights ``(n, )``.
        """
        assert self._path is not None
        theta = np.atleast_1d(np.asarray(thetas, dtype=float))
        return self._path.pos(theta), self._path.tangent(theta), np.ones(len(theta))

    def _reference_speed(self, theta: float, speed_actual: float) -> float:
        """Return the scenario's requested path speed, or the racing law when none is set.

        Args:
            theta: Path progress in meters.
            speed_actual: Current drone speed in m/s.

        Returns:
            The reference path speed in m/s.
        """
        cfg = self._mpcc
        reachable = speed_actual + cfg.v_lead
        if self._overload_speed is not None:
            # Ignore the curvature limit on purpose: that is what measures the real limit.
            speed = min(self._overload_speed, reachable)
        elif self._target_speed is not None:
            limit = float(np.interp(theta, self._speed_grid, self._speed_limit))
            speed = min(self._target_speed, limit, reachable)
        else:
            return super()._reference_speed(theta, speed_actual)
        return float(np.clip(speed, cfg.v_min, cfg.v_max))

    def _replan(self, obs: dict[str, NDArray[np.floating]], reason: str = "sensor update"):
        """Ignore sensor-triggered re-planning: a scenario reference never changes."""
        return

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return whether the manoeuvre is finished or has run out of time.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            ``True`` once the whole path is flown or the timeout elapses.
        """
        assert self._path is not None
        if self._theta >= self._path.length - FINISH_MARGIN:
            self.completed = True
            return True
        return self._tick >= self._timeout_ticks
