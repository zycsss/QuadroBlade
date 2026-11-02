"""Observation-only soft obstacle constraints for the existing MPCC dynamics and cost."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix

from lsy_drone_racing.control.my_controller import CODE_EXPORT_DIR, _signature, build_ocp

if TYPE_CHECKING:
    from collections.abc import Callable

    from acados_template import AcadosOcp

    from lsy_drone_racing.control.my_controller import MPCCConfig

_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


def build_obstacle_ocp(config: MPCCConfig, params: dict, n_obstacles: int) -> AcadosOcp:
    """Add squared XY-distance constraints; retain original dynamics and actuator bounds.

    Parameters are the original seven path parameters, all observed pole XY pairs, then radii.
    Soft lower bounds avoid making a sudden pose reveal an impossible initial-state constraint.
    Their linear/quadratic penalties strongly discourage predicted incursions; physical clearance
    still needs actual-flight validation because this is a simplified prediction model.
    """
    if n_obstacles <= 0:
        raise ValueError("n_obstacles must be positive")
    ocp = build_ocp(replace(config, name=f"{config.name}_avoid{n_obstacles}"), params)
    original = ocp.model.p
    p = cs.SX.sym("p", 7 + 3 * n_obstacles)
    ocp.model.cost_expr_ext_cost = cs.substitute(ocp.model.cost_expr_ext_cost, original, p[:7])
    ocp.model.p = p
    x = ocp.model.x
    ocp.model.con_h_expr = cs.vertcat(
        *[
            (x[0] - p[7 + 2 * j]) ** 2
            + (x[1] - p[8 + 2 * j]) ** 2
            - p[7 + 2 * n_obstacles + j] ** 2
            for j in range(n_obstacles)
        ]
    )
    ocp.constraints.lh = np.zeros(n_obstacles)
    ocp.constraints.uh = np.full(n_obstacles, 1e6)
    ocp.constraints.idxsh = np.arange(n_obstacles)
    ocp.cost.zl = np.full(n_obstacles, 50000.0)
    ocp.cost.zu = np.zeros(n_obstacles)
    ocp.cost.Zl = np.full(n_obstacles, 100000.0)
    ocp.cost.Zu = np.zeros(n_obstacles)
    ocp.parameter_values = np.zeros(7 + 3 * n_obstacles)
    return ocp


def get_obstacle_solver(
    config: MPCCConfig, params: dict, n_obstacles: int
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Cache the dedicated solver without overwriting any baseline solver library.

    The model-version suffix must change if this module's static formulation or penalties change.
    """
    key = f"{_signature(config, params)}_n{n_obstacles}_v1"
    if key in _CACHE:
        return _CACHE[key]
    ocp = build_obstacle_ocp(config, params, n_obstacles)
    directory = CODE_EXPORT_DIR / f"level2_{key}"
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
        builder.build_dir = "build-mingw-level2"
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


class ObservedObstacleSolver:
    """Adapt the inherited seven-parameter interface using only the current observation."""

    def __init__(
        self,
        backend: AcadosOcpSolver,
        observation: Callable[[], dict],
        radius: float,
        uncertainty: float,
    ):
        """Keep a live observation getter, never a simulator or hidden track data reference."""
        self.backend = backend
        self.observation = observation
        self.radius = radius
        self.uncertainty = uncertainty

    def set(self, stage: int, field: str, value: np.ndarray) -> None:
        """Append observed positions and uncertainty radii only to path-parameter writes."""
        if field == "p":
            obs = self.observation()
            points = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
            visited = np.asarray(
                obs.get("obstacles_visited", np.zeros(len(points), dtype=bool)), dtype=bool
            ).reshape(-1)
            radii = self.radius + self.uncertainty * (~visited)
            value = np.concatenate((value, points[:, :2].ravel(), radii))
        self.backend.set(stage, field, np.ascontiguousarray(value, dtype=float))

    def reset(self) -> None:
        """Reset the native solver's iteration state."""
        self.backend.reset()

    def solve(self) -> int:
        """Return the native status so inherited fallback handling remains active."""
        return self.backend.solve()

    def get(self, stage: int, field: str) -> np.ndarray:
        """Read the unchanged physical/progress state or input layout."""
        return self.backend.get(stage, field)
