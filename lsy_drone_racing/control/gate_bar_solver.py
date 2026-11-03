"""Observed solid gate bars inflated by the drone's oriented body box."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.corridor_solver import CorridorSolver, build_corridor_ocp
from lsy_drone_racing.control.my_controller import CODE_EXPORT_DIR, _signature

if TYPE_CHECKING:
    from collections.abc import Callable

    from acados_template import AcadosOcp

    from lsy_drone_racing.control.my_controller import MPCCConfig

# Local centres and half sizes from envs/assets/gate.xml collision geoms.
GATE_BARS = (
    ((0.0, 0.0, 0.28), (0.01, 0.36, 0.08)),
    ((0.0, 0.0, -0.28), (0.01, 0.36, 0.08)),
    ((0.0, -0.28, 0.0), (0.01, 0.08, 0.36)),
    ((0.0, 0.28, 0.0), (0.01, 0.08, 0.36)),
)
BODY_HALF_BOX = np.array([0.07, 0.07, 0.02])
FRAME_MARGIN = 0.015
_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


def build_gate_bar_ocp(
    config: MPCCConfig, params: dict, n_obstacles: int, n_gates: int
) -> AcadosOcp:
    """Retain the MPCC dynamics/cost and poles; replace corridors with four solid bars.

    Each gate uses centre3, frame rows9, and inflated body half extents3.
    A positive maximum normalized separation means the body clears a bar on at
    least one axis. Soft bounds allow recovery after a late geometry observation.
    """
    ocp = build_corridor_ocp(config, params, n_obstacles, n_gates)
    x, p = ocp.model.x, ocp.model.p
    guards = [ocp.model.con_h_expr[:n_obstacles]] if n_obstacles else []
    offset = 7 + 3 * n_obstacles
    for gate in range(n_gates):
        start = offset + 15 * gate
        delta = x[:3] - p[start : start + 3]
        local = cs.vertcat(*[cs.dot(p[start + j : start + j + 3], delta) for j in (3, 6, 9)])
        extent = p[start + 12 : start + 15]
        for centre, half in GATE_BARS:
            ratio = cs.sqrt((local - cs.DM(centre)) ** 2 + 1e-10) / (cs.DM(half) + extent)
            guards.append(0.1 * (cs.fmax(cs.fmax(ratio[0], ratio[1]), ratio[2]) - 1.0))
        # Positive denominators also during native precomputation before observations.
        ocp.parameter_values[start + 12 : start + 15] = BODY_HALF_BOX + FRAME_MARGIN
    size = n_obstacles + 4 * n_gates
    ocp.model.con_h_expr = cs.vertcat(*guards)
    ocp.constraints.lh = np.zeros(size)
    ocp.constraints.uh = np.full(size, 1e6)
    ocp.constraints.idxsh = np.arange(size)
    ocp.cost.zl = np.full(size, 50000.0)
    ocp.cost.zu = np.zeros(size)
    ocp.cost.Zl = np.full(size, 100000.0)
    ocp.cost.Zu = np.zeros(size)
    return ocp


def get_gate_bar_solver(
    config: MPCCConfig, params: dict, n_obstacles: int, n_gates: int
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Keep dimensions and model symbols isolated from all other native libraries."""
    key = f"{_signature(config, params)}_p{n_obstacles}_g{n_gates}_v1"
    if key in _CACHE:
        return _CACHE[key]
    ocp = build_gate_bar_ocp(replace(config, name=f"gate_bar_{key}"), params, n_obstacles, n_gates)
    directory = CODE_EXPORT_DIR / f"gate_bar_{key}"
    ocp.code_export_directory = str(directory)
    description = directory / "ocp.json"
    library = directory / (
        f"{get_shared_lib_prefix()}acados_ocp_solver_{ocp.model.name}{get_shared_lib_ext()}"
    )
    reuse = description.is_file() and library.is_file()
    builder = None
    if os.name == "nt":
        builder = ocp_get_default_cmake_builder()
        builder.generator = "MinGW Makefiles"
        builder.build_dir = "build-mingw-gate-bar"
    solver = AcadosOcpSolver(
        ocp,
        json_file=str(description),
        generate=not reuse,
        build=not reuse,
        cmake_builder=builder,
        verbose=False,
    )
    _CACHE[key] = solver, ocp
    return solver, ocp


class GateBarSolver(CorridorSolver):
    """Encode observed/forecast body inflation and run two native RTI refinements."""

    def __init__(
        self,
        backend: AcadosOcpSolver,
        observation: Callable[[], dict],
        radius: float,
        uncertainty: float,
        speed_reference: Callable[[], float] | None = None,
    ):
        """Share recovery and pole encoding with the corridor adapter."""
        super().__init__(backend, observation, radius, uncertainty, speed_reference, iterations=2)

    def set(self, stage: int, field: str, value: np.ndarray) -> None:
        """Replace only the three gate margin slots with inflated body half extents."""
        super().set(stage, field, value)
        if field != "p":
            return
        obs = self.observation()
        frames = Rotation.from_quat(obs["gates_quat"]).as_matrix().transpose(0, 2, 1)
        body = Rotation.from_quat(obs["quat"]).as_matrix()
        warm = np.asarray(self.backend.get(stage, "x"))
        forecast = (
            Rotation.from_euler("xyz", warm[6:9]).as_matrix()
            if np.isfinite(warm[6:9]).all()
            else body
        )
        extent = (
            np.maximum(
                np.abs(frames @ body) @ BODY_HALF_BOX, np.abs(frames @ forecast) @ BODY_HALF_BOX
            )
            + FRAME_MARGIN
        )
        encoded = self._parameters[stage].copy()
        offset = 7 + 3 * len(obs["obstacles_pos"])
        for gate in range(len(frames)):
            encoded[offset + 15 * gate + 12 : offset + 15 * gate + 15] = extent[gate]
        self._parameters[stage] = encoded
        self.backend.set(stage, field, encoded)
