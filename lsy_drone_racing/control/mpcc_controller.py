"""A selectable MPCC baseline with reproducible planning and parameter exports.

Run with the existing simulation entry point::

    pixi run python scripts/sim.py --config level0.toml --controller mpcc_controller.py

The inherited controller executes these stages:

1. Read the signed gate sequence and current gate/obstacle observations.
2. Route with A*, smooth the route and parameterize the path by arc length.
3. Project the measured position onto the path; limit reference speed using curvature.
4. Solve the contour/lag/progress MPCC with acados SQP_RTI and warm starting.
5. Apply the first command as [roll, pitch, yaw, collective_thrust].

Observed geometry changes trigger the existing replanning mechanism. This entry point keeps the
existing tuning, fallback, mass adaptation and lifecycle behavior. It exports copies for evaluation
without giving plot consumers access to mutable controller memory.

For later optimization, MPCCConfig exposes N/dt (horizon), q_lag/q_contour (tracking), mu_speed
(progress), v_max/a_lat_max (pace) and r_rates (command smoothing). PathConfig controls the A* grid
and clearances. Override these in an in-memory configuration; preserve the supplied race files.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from lsy_drone_racing.control.my_controller import MPCCController


class MPCCBaselineController(MPCCController):
    """Use the established racing MPCC and expose independent evaluation snapshots."""

    def _record_plan(self, reason: str, obs: dict) -> None:
        """Keep the geometry associated with each inherited plan installation."""
        super()._record_plan(reason, obs)
        self._reference_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._reference_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._reference_sequence = np.asarray(obs["gate_sequence"], dtype=int).copy()
        self._reference_directions = np.asarray(obs["gate_sequence_direction"], dtype=int).copy()

    def reference_snapshot(self) -> dict[str, Any]:
        """Return the active path and its observed geometry, with independent arrays.

        Positions are in metres, gate IDs are zero based and progress is arc length in metres.
        Repeated gate passes retain separate entries in ``gate_centers`` and ``gate_sequence``.
        """
        assert self._path is not None
        arc = np.linspace(0.0, self._path.length, 500)
        return {
            "arc": arc,
            "points": np.asarray(self._path.pos(arc), dtype=float).copy(),
            "waypoints": self._path_info.waypoints.copy(),
            "gate_centers": self._gate_centers.copy(),
            "gates_pos": self._reference_gates_pos.copy(),
            "gates_quat": self._reference_gates_quat.copy(),
            "gate_sequence": self._reference_sequence.copy(),
            "gate_sequence_direction": self._reference_directions.copy(),
            "obstacles": self._obstacles.copy(),
            "length": float(self._path.length),
        }

    def settings_snapshot(self) -> dict[str, Any]:
        """Return effective MPCC/planner settings, independently of mutable configuration."""
        return {
            "mpcc": asdict(self._mpcc),
            "planner": asdict(self._planner.config),
            "env_freq_hz": self._freq,
            "action_order": ["roll", "pitch", "yaw", "collective_thrust"],
        }
