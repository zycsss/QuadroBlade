"""Observation-only gate entry/exit corridors added to the existing pole MPCC."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_solver import ObservedObstacleSolver, build_obstacle_ocp
from lsy_drone_racing.control.my_controller import CODE_EXPORT_DIR, _signature, build_ocp

if TYPE_CHECKING:
    from collections.abc import Callable

    from acados_template import AcadosOcp

    from lsy_drone_racing.control.my_controller import MPCCConfig

_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


def build_corridor_ocp(
    config: MPCCConfig,
    params: dict,
    n_obstacles: int,
    n_gates: int,
    second_order: bool = False,
    velocity_weight: float = 0.0,
) -> AcadosOcp:
    """Keep pole constraints and add soft lateral/vertical bounds through each gate.

    Freeze observed/forecast body-box margins as per-stage parameters, keeping
    rotation nonlinearities out of the gate constraints. A quartic normal release
    permits the turn after body clearance. Reference activation allows remote crossings.
    """
    tuned = replace(config, name=f"{config.name}_corridor")
    ocp = (
        build_obstacle_ocp(tuned, params, n_obstacles) if n_obstacles else build_ocp(tuned, params)
    )
    if second_order:
        old_x = ocp.model.x
        x = cs.SX.sym("x", 18)
        dynamics = cs.substitute(ocp.model.f_expl_expr, old_x, x[:15])
        dynamics[6:9] = x[15:18]
        acceleration = (
            cs.DM(params["rpy_coef"]) * x[6:9]
            + cs.DM(params["rpy_rates_coef"]) * x[15:18]
            + cs.DM(params["cmd_rpy_coef"]) * x[11:14]
        )
        ocp.model.cost_expr_ext_cost = cs.substitute(ocp.model.cost_expr_ext_cost, old_x, x[:15])
        if n_obstacles:
            ocp.model.con_h_expr = cs.substitute(ocp.model.con_h_expr, old_x, x[:15])
        ocp.model.x = x
        ocp.model.f_expl_expr = cs.vertcat(dynamics, acceleration)
        ocp.constraints.x0 = np.zeros(18)
    old = ocp.model.p
    offset = 7 + 3 * n_obstacles
    p = cs.SX.sym("p", offset + 15 * n_gates + 3)
    ocp.model.cost_expr_ext_cost = cs.substitute(ocp.model.cost_expr_ext_cost, old, p[:offset])
    velocity_error = ocp.model.x[3:6] - p[-3:]
    ocp.model.cost_expr_ext_cost += velocity_weight * cs.dot(velocity_error, velocity_error)
    constraints = []
    if n_obstacles:
        constraints.append(cs.substitute(ocp.model.con_h_expr, old, p[:offset]))
    for gate in range(n_gates):
        start = offset + 15 * gate
        delta = ocp.model.x[:3] - p[start : start + 3]
        axes = [p[start + j : start + j + 3] for j in (3, 6, 9)]
        normal, lateral, vertical = [cs.dot(axis, delta) for axis in axes]
        active = p[start + 12]
        release = 64.0 * normal**4
        lateral_clearance = p[start + 13] ** 2 + release
        vertical_clearance = p[start + 14] ** 2 + release
        constraints.extend(
            [
                1 - active + active * (lateral_clearance - lateral**2),
                1 - active + active * (vertical_clearance - vertical**2),
            ]
        )
    ocp.model.p = p
    ocp.model.con_h_expr = cs.vertcat(*constraints)
    size = n_obstacles + 2 * n_gates
    ocp.constraints.lh = np.zeros(size)
    ocp.constraints.uh = np.full(size, 1e6)
    ocp.constraints.idxsh = np.arange(size)
    ocp.cost.zl = np.r_[np.full(n_obstacles, 50000.0), np.full(2 * n_gates, 200000.0)]
    ocp.cost.zu = np.zeros(size)
    ocp.cost.Zl = np.r_[np.full(n_obstacles, 100000.0), np.full(2 * n_gates, 1000000.0)]
    ocp.cost.Zu = np.zeros(size)
    ocp.parameter_values = np.zeros(offset + 15 * n_gates + 3)
    return ocp


def get_corridor_solver(
    config: MPCCConfig,
    params: dict,
    n_obstacles: int,
    n_gates: int,
    second_order: bool = False,
    velocity_weight: float = 0.0,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Cache this formulation separately from every pre-existing solver library."""
    weight_key = hashlib.sha256(str(float(velocity_weight)).encode()).hexdigest()[:8]
    key = (
        f"{_signature(config, params)}_p{n_obstacles}_g{n_gates}"
        f"_s{int(second_order)}_w{weight_key}_v8"
    )
    if key in _CACHE:
        return _CACHE[key]
    named = replace(config, name=f"corridor_{key}")
    ocp = build_corridor_ocp(named, params, n_obstacles, n_gates, second_order, velocity_weight)
    directory = CODE_EXPORT_DIR / f"corridor_{key}"
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
        builder.build_dir = "build-mingw-corridor"
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


class CorridorSolver(ObservedObstacleSolver):
    """Adapt the inherited seven path parameters using only current observations."""

    def __init__(
        self,
        backend: AcadosOcpSolver,
        observation: Callable[[], dict],
        radius: float,
        uncertainty: float,
        speed_reference: Callable[[], float] | None = None,
        iterations: int = 1,
    ):
        """Retain valid observation/reference inputs for numerical recovery only."""
        super().__init__(backend, observation, radius, uncertainty)
        self._initial_state: np.ndarray | None = None
        self._parameters: dict[int, np.ndarray] = {}
        self.nonfinite_recoveries = 0
        self.speed_reference = speed_reference if speed_reference is not None else lambda: 0.0
        self.iterations = iterations

    def set(self, stage: int, field: str, value: np.ndarray) -> None:
        """Append observed pole and gate geometry; never alter caller-owned arrays."""
        if stage == 0 and field == "lbx":
            self._initial_state = np.asarray(value, dtype=float).copy()
        if field == "p":
            obs = self.observation()
            poles = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
            visited = np.asarray(
                obs.get("obstacles_visited", np.zeros(len(poles), dtype=bool)), dtype=bool
            )
            radii = self.radius + self.uncertainty * (~visited)
            centres = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
            frames = Rotation.from_quat(np.asarray(obs["gates_quat"])).as_matrix()
            local = np.einsum("gji,gj->gi", frames, value[:3] - centres)
            reference_near = (np.abs(local[:, 0]) < 0.8) & (
                np.max(np.abs(local[:, 1:]), axis=1) < 0.3
            )
            body = Rotation.from_quat(np.asarray(obs["quat"])).as_matrix()
            forecast = np.asarray(self.backend.get(stage, "x"))
            predicted_body = (
                Rotation.from_euler("xyz", forecast[6:9]).as_matrix()
                if np.isfinite(forecast[6:9]).all()
                else body
            )
            half_box = np.array([0.07, 0.07, 0.02])
            frame_rows = frames.transpose(0, 2, 1)
            observed_extent = np.abs(frame_rows @ body) @ half_box
            predicted_extent = np.abs(frame_rows @ predicted_body) @ half_box
            extent = np.maximum(observed_extent, predicted_extent)
            margins = 0.2 - 0.015 - extent
            predicted_position = (
                forecast[:3] if np.isfinite(forecast[:3]).all() else np.asarray(obs["pos"])
            )
            predicted_local = np.einsum("gij,gj->gi", frame_rows, predicted_position - centres)
            outer = 0.36 + extent[:, 1:] + 0.015
            predicted_near = (np.abs(predicted_local[:, 0]) < 0.8) & np.all(
                np.abs(predicted_local[:, 1:]) < outer, axis=1
            )
            active = reference_near | predicted_near
            if stage <= 2:
                observed_local = np.einsum(
                    "gij,gj->gi", frame_rows, np.asarray(obs["pos"]) - centres
                )
                active |= (np.abs(observed_local[:, 0]) < 0.8) & np.all(
                    np.abs(observed_local[:, 1:]) < outer, axis=1
                )
            gates = np.column_stack((centres, frame_rows.reshape(-1, 9), active, margins[:, 1:]))
            tangent = np.asarray(value[3:6], dtype=float)
            target_velocity = (
                float(self.speed_reference()) * tangent / max(np.linalg.norm(tangent), 1e-9)
            )
            value = np.concatenate(
                (value, poles[:, :2].ravel(), radii, gates.ravel(), target_velocity)
            )
            self._parameters[stage] = value.copy()
        self.backend.set(stage, field, np.ascontiguousarray(value, dtype=float))

    def reset(self) -> None:
        """Clear episode-local recovery inputs and native iteration memory."""
        super().reset()
        self._initial_state = None
        self._parameters.clear()
        self.nonfinite_recoveries = 0

    def solve(self) -> int:
        """Report nonfinite output as failure and repair warm starts for the next step."""
        for _ in range(self.iterations):
            status = self.backend.solve()
            if not np.isfinite(self.backend.get(1, "x")).all():
                break
            if status:
                return status
        else:
            return status
        self.nonfinite_recoveries += 1
        self.backend.reset()
        if self._initial_state is not None and self._parameters:
            terminal = max(self._parameters)
            for stage, params in sorted(self._parameters.items()):
                state = self._initial_state.copy()
                if stage:
                    state[:3] = params[:3]
                self.backend.set(stage, "p", params)
                self.backend.set(stage, "x", state)
                if stage < terminal:
                    self.backend.set(stage, "u", np.r_[np.zeros(4), 1.0])
            self.backend.set(0, "lbx", self._initial_state)
            self.backend.set(0, "ubx", self._initial_state)
        return status if status else 100
