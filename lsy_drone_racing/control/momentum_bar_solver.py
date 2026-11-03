"""Gate-bar MPCC with measured angular momentum and second-order attitude lag."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

import casadi as cs
from acados_template import AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix

from lsy_drone_racing.control.corridor_solver import build_corridor_ocp
from lsy_drone_racing.control.gate_bar_solver import build_gate_bar_ocp
from lsy_drone_racing.control.my_controller import CODE_EXPORT_DIR, _signature

if TYPE_CHECKING:
    from acados_template import AcadosOcp

    from lsy_drone_racing.control.my_controller import MPCCConfig

_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


def build_momentum_bar_ocp(
    config: MPCCConfig, params: dict, n_obstacles: int, n_gates: int
) -> AcadosOcp:
    """Use the tested bar geometry with eighteen-state attitude prediction.

    Keep physical position/velocity and actuator state indices unchanged. Add
    Euler-angle rates initialized from the measured body angular velocity. The
    bar inequalities have no dependence on the added angular-rate states.
    """
    bars = build_gate_bar_ocp(config, params, n_obstacles, n_gates)
    ocp = build_corridor_ocp(config, params, n_obstacles, n_gates, second_order=True)
    guards = cs.substitute(bars.model.con_h_expr, bars.model.x, ocp.model.x[:15])
    ocp.model.con_h_expr = cs.substitute(guards, bars.model.p, ocp.model.p)
    for key in ("lh", "uh", "idxsh"):
        setattr(ocp.constraints, key, getattr(bars.constraints, key).copy())
    for key in ("zl", "zu", "Zl", "Zu"):
        setattr(ocp.cost, key, getattr(bars.cost, key).copy())
    ocp.parameter_values = bars.parameter_values.copy()
    return ocp


def get_momentum_bar_solver(
    config: MPCCConfig, params: dict, n_obstacles: int, n_gates: int
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Keep dimensions and model symbols isolated from all other native libraries."""
    key = f"{_signature(config, params)}_p{n_obstacles}_g{n_gates}_v1"
    if key in _CACHE:
        return _CACHE[key]
    ocp = build_momentum_bar_ocp(
        replace(config, name=f"momentum_bar_{key}"), params, n_obstacles, n_gates
    )
    directory = CODE_EXPORT_DIR / f"momentum_bar_{key}"
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
        builder.build_dir = "build-mingw-momentum-bar"
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
