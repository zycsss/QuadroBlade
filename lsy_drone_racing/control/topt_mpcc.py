"""Time-optimal plan, bent to what the sensor reveals, flown by an attitude-mode MPCC.

The controller has three parts:

1. **Plan.** `topt_planner` solves the fastest point-mass trajectory through the nominal track once
   and caches it on disk; every episode on the same track reuses it.
2. **Bend.** `path_bend` keeps that plan as beads every 5 cm and, whenever a gate or pole is seen
   for the first time, bends it to the revealed poses - rigid gate windows, the path under the drone
   held, soft pushes off poles and frames. It never re-plans in flight.
3. **Fly.** A model predictive contouring controller (acados, ``SQP_RTI``) maximizes progress along
   the bent path inside a tunnel whose radius is the live clearance to the true geometry, under a
   speed limit taken from the plan's own speed profile.

The MPCC commands the environment's ``attitude`` interface, ``[roll, pitch, yaw, thrust]``. Behind
it sits the onboard Mellinger attitude law with a zero rate setpoint, identified by
`scripts/identify_attitude.py`: per tilt axis

    torque = K1 (cmd - tilt) - K2 rate - KD ang_acc,   |torque| <= AL_MAX,
    d ang_acc / dt = (torque - ang_acc) / TAU_AL,

so the MPCC plans the torque directly and sends the angle that makes the onboard law produce it,
``cmd = tilt + (torque + K2 rate + KD ang_acc) / K1``. That over-commands the angle by the damping
term, which is what lets the tilt follow the plan at the rate the torque limit allows rather than at
the 7 rad/s the damping alone would leave. The collective thrust follows its command with a
first-order lag ``TAU_F``; the airframe has linear drag.

A disturbance observer closes the loop on the model: the error of the predicted velocity is split
into a thrust factor (along the body z axis - the randomized mass) and an additive acceleration.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.path_bend import ArcPath, BendConfig, PathBender
from lsy_drone_racing.control.topt_planner import PlannerConfig, Track, plan
from lsy_drone_racing.control.track_geometry import DRONE_RADIUS, frame_clearance, pole_clearance

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAVITY = np.array([0.0, 0.0, 9.81])

# Identified by scripts/identify_attitude.py on the cf21B_500 first-principles model.
K1 = 623.2
K2 = 93.77
KD = 0.519
TAU_AL = 0.0323
AL_MAX = 157.5
TAU_F = 0.0575
DRAG = np.array([0.50, 0.50, 0.57])
MASS = 0.0434
#: Force each motor adds or sheds per rad/s^2 of tilt acceleration: ``J / (4 L)``, J = 2.5e-5 /
#: 2.8e-5 kg m^2 (roll / pitch), L = 0.0354 m.
MOTOR_PER_ACC = 1.9e-4
MOTOR_MIN = 0.0214
MOTOR_MAX = 0.2

# MPCC layout. x = [p (3), v (3), tilt (2), rate (2), ang_acc (2), f, f_cmd, s, ds],
# u = [torque (2), df_cmd, dds], p = [disturbance (3), path point (3), tangent (3), s_lin, tunnel
# radius, speed limit, thrust factor].
NX, NU, NP = 16, 4, 13
NY = 4 + 1 + 2 + NU + 1  # contour (3), lag, progress, tilt rates, inputs, vertical velocity error
IX_TILT, IX_RATE, IX_ACC, IX_F, IX_FCMD, IX_S, IX_DS = 6, 8, 10, 12, 13, 14, 15


@dataclass(frozen=True)
class MpccConfig:
    """Tuning of the controller.

    Every field can be overridden from the ``[controller]`` section of a race config with the same
    name; `PlannerConfig` and `BendConfig` fields with the prefixes ``plan_`` and ``bend_``.

    Attributes:
        horizon: Prediction horizon in steps of the control period.
        v_max: Ceiling of the path speed [m/s].
        speed_scale: The speed limit is ``speed_scale * planned speed + speed_add``.
        speed_add: See ``speed_scale`` [m/s].
        progress_add: The progress term asks for this much more than the speed limit [m/s].
        unseen_speed: Speed limit in front of a gate not seen yet [m/s]; randomized tracks only.
        unseen_length: Distance before such a gate over which ``unseen_speed`` applies [m].
        tunnel_min: Smallest tunnel radius [m].
        tunnel_max: Largest tunnel radius [m].
        tunnel_margin: Clearance kept on top of the drone's reach inside the tunnel [m].
        q_contour: Weight of the lateral contour error.
        q_contour_z: Weight of the vertical contour error.
        q_lag: Weight of the lag error.
        q_progress: Weight of the progress term.
        q_rate: Weight of the tilt rates.
        q_torque: Weight of the torque input.
        q_vz: Weight of the vertical velocity error against the rate the path asks for.
        az_max: Soft bound on the vertical acceleration [m/s^2].
        tilt_max: Bound on the total tilt [rad].
        tilt_lead: The tilt bound applies to ``tilt + tilt_lead * rate`` [s].
        cmd_max: Bound on the attitude command sent [rad].
        rescue_tilt: Above this total tilt the MPCC is suspended and the drone levelled [rad].
        thrust_min: Lower bound of the thrust command [N].
        thrust_max: Upper bound of the thrust command [N].
        z_min: Soft floor [m].
        dds_max: Bound on the path acceleration of the model [m/s^2].
        k_dist: Gain of the disturbance observer per step.
        k_thrust: Gain of the thrust-factor estimate per step.
        thrust_factor_min_thrust: The thrust factor only learns above this thrust [N].
        thrust_factor_range: Bounds of the thrust factor; the randomized mass is 43.4 g +- 5 g.
        dist_share: Share of the error along the thrust axis left to the additive disturbance.
        dist_max: Bound on the additive disturbance [m/s^2].
        takeoff_dz: Climb under fixed thrust before the MPCC takes over [m].
        takeoff_vz: Vertical speed assumed for the plan's first node [m/s].
        takeoff_thrust: Thrust of the fixed take-off climb [N].
        rebuild_iterations: Extra SQP iterations after the path changed.
    """

    horizon: int = 40
    v_max: float = 7.0
    # The final hairpin (gate 3 -> gate 1) is where the pace binds: faster, the drone enters it with
    # the thrust at its cap and past the tilt bound. Level 2, seeds 264-391, with the height terms
    # below: 0.80 121/128 at 5.19 s, 0.85 110/128 at 5.06 s, 0.90 101/128 at 4.98 s. Ranking takes
    # the mean time of the finishers, so 0.85 trades some of the spare success for lap time.
    speed_scale: float = 0.85
    speed_add: float = 0.3
    progress_add: float = 0.3
    unseen_speed: float = 2.0
    unseen_length: float = 0.7
    tunnel_min: float = 0.03
    tunnel_max: float = 0.15
    tunnel_margin: float = 0.02
    q_contour: float = 3000.0
    q_contour_z: float = 3000.0
    q_lag: float = 1000.0
    q_progress: float = 60.0
    q_rate: float = 0.5
    q_torque: float = 0.0005
    # Keep the height steady: without these the MPCC gives up height in a hard turn, because it
    # prices it like lateral error and only reacts once it is lost. Level 2,
    # seeds 264-391, off -> on:
    # speed_scale 0.80 116 -> 121/128 (5.14 -> 5.19 s), 0.85 98 -> 110 (5.02 -> 5.06 s),
    # 0.90 85 -> 101 (4.95 -> 4.98 s, 27 gained, 11 lost, p = 0.014). A heavier contour z weight
    # does not do this.
    q_vz: float = 600.0
    az_max: float = 4.0
    tilt_max: float = 1.0
    tilt_lead: float = 0.08
    cmd_max: float = 1.3
    rescue_tilt: float = 1.3
    thrust_min: float = 0.15
    thrust_max: float = 0.74
    z_min: float = 0.08
    dds_max: float = 30.0
    k_dist: float = 0.15
    k_thrust: float = 0.1
    thrust_factor_min_thrust: float = 0.3
    thrust_factor_range: tuple[float, float] = (0.88, 1.13)
    dist_share: float = 0.3
    dist_max: tuple[float, float, float] = (1.5, 1.5, 2.0)
    takeoff_dz: float = 0.04
    takeoff_vz: float = 0.5
    takeoff_thrust: float = 0.7
    rebuild_iterations: int = 2


#: Fields baked into the generated solver; a change triggers a rebuild.
_OCP_FIELDS = (
    "horizon",
    "q_contour",
    "q_contour_z",
    "q_lag",
    "q_progress",
    "q_rate",
    "q_torque",
    "q_vz",
    "az_max",
    "tilt_max",
    "tilt_lead",
    "cmd_max",
    "z_min",
    "dds_max",
)


# region model


def _tilt_axis(tilt: NDArray[np.floating]) -> NDArray[np.floating]:
    """Body z axis for ``[roll, pitch]`` at zero yaw (scipy ``xyz`` Euler angles)."""
    r, p = tilt[..., 0], tilt[..., 1]
    return np.stack((np.sin(p) * np.cos(r), -np.sin(r), np.cos(r) * np.cos(p)), axis=-1)


def _tilt_state(
    quat: NDArray[np.floating], ang_vel: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return roll/pitch and their Euler rates from the attitude and the body rates."""
    roll, pitch, _ = R.from_quat(quat).as_euler("xyz")
    wx, wy, wz = ang_vel
    cr, sr, tp = np.cos(roll), np.sin(roll), np.tan(np.clip(pitch, -1.5, 1.5))
    rates = np.array([wx + sr * tp * wy + cr * tp * wz, cr * wy - sr * wz])
    return np.array([roll, pitch]), rates


def _model(name: str, cfg: MpccConfig) -> AcadosModel:
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    par = ca.SX.sym("p", NP)
    p, v = x[0:3], x[3:6]
    tilt, rate, acc = x[IX_TILT : IX_TILT + 2], x[IX_RATE : IX_RATE + 2], x[IX_ACC : IX_ACC + 2]
    f, f_cmd, s, ds = x[IX_F], x[IX_FCMD], x[IX_S], x[IX_DS]
    dist, p_lin, t_lin, s_lin = par[0:3], par[3:6], par[6:9], par[9]
    r_tunnel, v_limit, thrust_factor = par[10], par[11], par[12]

    cr, sr = ca.cos(tilt[0]), ca.sin(tilt[0])
    cp, sp = ca.cos(tilt[1]), ca.sin(tilt[1])
    z_body = ca.vertcat(sp * cr, -sr, cr * cp)
    accel = z_body * thrust_factor * f / MASS - GRAVITY - DRAG * v + dist

    error = p - (p_lin + t_lin * (s - s_lin))
    e_lag = ca.dot(t_lin, error)
    e_con = error - e_lag * t_lin

    model = AcadosModel()
    model.name = name
    model.x, model.u, model.p = x, u, par
    model.f_expl_expr = ca.vertcat(
        v, accel, rate, acc, (u[0:2] - acc) / TAU_AL, (f_cmd - f) / TAU_F, u[2], ds, u[3]
    )
    # Height is what the drone cannot get back in a hard turn, so its rate is held to the rate the
    # path asks for: the contour error alone only reacts once the height is already lost.
    e_vz = v[2] - ds * t_lin[2]
    model.cost_y_expr = ca.vertcat(e_con, e_lag, ds, rate, u, e_vz)
    model.cost_y_expr_e = ca.vertcat(e_con, e_lag, e_vz)
    lead = tilt + cfg.tilt_lead * rate
    path_terms = ca.vertcat(
        ca.sqrt(ca.dot(e_con, e_con) + 1e-6) - r_tunnel,
        ca.dot(t_lin, v) - v_limit,
        np.cos(cfg.tilt_max) - ca.cos(lead[0]) * ca.cos(lead[1]),
        (accel[2] / cfg.az_max) ** 2 - 1.0,
    )
    # Each motor carries a quarter of the thrust plus the torque share of both tilt axes, and must
    # stay inside its range or the torque is lost and the drone keeps tipping. Normalized to 0.05 N.
    share = MOTOR_PER_ACC * (ca.sqrt(u[0] ** 2 + 1.0) + ca.sqrt(u[1] ** 2 + 1.0))
    # The angle that makes the onboard law produce the planned torque must fit the command range.
    command = tilt + (u[0:2] + K2 * rate + KD * acc) / K1
    model.con_h_expr = ca.vertcat(
        path_terms,
        (f_cmd / 4.0 + share - MOTOR_MAX) / 0.05,
        (MOTOR_MIN - f_cmd / 4.0 + share) / 0.05,
        command**2 - cfg.cmd_max**2,
    )
    model.con_h_expr_e = path_terms
    return model


def _ocp(name: str, cfg: MpccConfig, dt: float, export_dir: Path) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = _model(name, cfg)
    ocp.code_export_directory = str(export_dir)
    ocp.solver_options.N_horizon = cfg.horizon
    ocp.solver_options.tf = cfg.horizon * dt
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    contour = [cfg.q_contour, cfg.q_contour, cfg.q_contour_z, cfg.q_lag]
    ocp.cost.W = np.diag(
        contour
        + [cfg.q_progress, cfg.q_rate, cfg.q_rate, cfg.q_torque, cfg.q_torque, 0.1, 0.01, cfg.q_vz]
    )
    ocp.cost.W_e = np.diag(contour + [cfg.q_vz])
    ocp.cost.yref = np.zeros(NY)
    ocp.cost.yref_e = np.zeros(5)
    ocp.parameter_values = np.zeros(NP)
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array([-AL_MAX, -AL_MAX, -15.0, -cfg.dds_max])
    ocp.constraints.ubu = np.array([AL_MAX, AL_MAX, 15.0, cfg.dds_max])
    # States: height (soft), tilt per axis (soft), thrust command, path speed (soft).
    ocp.constraints.idxbx = np.array([2, IX_TILT, IX_TILT + 1, IX_FCMD, IX_DS])
    ocp.constraints.lbx = np.array([cfg.z_min, -cfg.tilt_max, -cfg.tilt_max, 0.0, 0.0])
    ocp.constraints.ubx = np.array([2.4, cfg.tilt_max, cfg.tilt_max, 0.8, 10.0])
    ocp.constraints.idxsbx = np.array([0, 1, 2, 4])
    n_h = 8
    ocp.constraints.lh = np.full(n_h, -100.0)
    ocp.constraints.uh = np.zeros(n_h)
    ocp.constraints.idxsh = np.arange(n_h)
    # Penalties of the soft bounds: [floor, tilt, tilt, path speed]
    # + [tunnel, speed, total tilt, vertical acceleration]
    # + [motor high, motor low, command roll, command pitch].
    lin_x = np.array([2000.0, 100.0, 100.0, 100.0])
    quad_x = np.array([2e5, 1000.0, 1000.0, 1000.0])
    lin_h = np.array([2000.0, 1000.0, 3000.0, 300.0])
    quad_h = np.array([1e5, 1e4, 3e5, 3000.0])
    lin_in = np.array([300.0, 300.0, 300.0, 300.0])
    quad_in = 10.0 * lin_in
    ocp.cost.zl = ocp.cost.zu = np.r_[lin_x, lin_h, lin_in]
    ocp.cost.Zl = ocp.cost.Zu = np.r_[quad_x, quad_h, quad_in]
    ocp.constraints.idxbx_e = ocp.constraints.idxbx.copy()
    ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
    ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
    ocp.constraints.idxsbx_e = ocp.constraints.idxsbx.copy()
    ocp.constraints.lh_e = np.full(4, -100.0)
    ocp.constraints.uh_e = np.zeros(4)
    ocp.constraints.idxsh_e = np.arange(4)
    ocp.cost.zl_e = ocp.cost.zu_e = np.r_[lin_x, lin_h]
    ocp.cost.Zl_e = ocp.cost.Zu_e = np.r_[quad_x, quad_h]
    opts = ocp.solver_options
    opts.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "ERK"
    opts.sim_method_num_stages = 4
    opts.sim_method_num_steps = 1
    opts.nlp_solver_type = "SQP_RTI"
    opts.qp_solver_iter_max = 50
    opts.qp_solver_warm_start = 1
    return ocp


_SOLVERS: dict[str, AcadosOcpSolver] = {}


def get_solver(cfg: MpccConfig, dt: float) -> AcadosOcpSolver:
    """Return the MPCC solver, generated and compiled only when its signature is new.

    The code lives in ``c_generated_code/`` (ignored by git), one folder per signature, so a solver
    built by one process is loaded by the next instead of compiled again.
    """
    signature = repr(
        (
            {k: v for k, v in asdict(cfg).items() if k in _OCP_FIELDS},
            dt,
            K1,
            K2,
            KD,
            TAU_AL,
            AL_MAX,
            TAU_F,
            DRAG.tolist(),
            MASS,
            MOTOR_PER_ACC,
            "topt-mpcc-2",
        )
    )
    key = hashlib.sha1(signature.encode()).hexdigest()[:10]
    if key in _SOLVERS:
        return _SOLVERS[key]
    name = f"topt_mpcc_{key}"
    root = REPO_ROOT / "c_generated_code" / name
    json_file = root / "ocp.json"
    libs = list(root.glob(f"*acados_ocp_solver_{name}.*")) if root.exists() else []
    if json_file.exists() and libs:
        solver = AcadosOcpSolver(None, json_file=str(json_file), build=False, generate=False)
    else:
        ocp = _ocp(name, cfg, dt, root)
        builder = None
        if os.name == "nt":  # the Windows toolchain needs the CMake builder with MinGW
            from acados_template.builders import ocp_get_default_cmake_builder

            builder = ocp_get_default_cmake_builder()
            builder.generator = "MinGW Makefiles"
            builder.build_dir = str(root / "build")
        root.mkdir(parents=True, exist_ok=True)
        solver = AcadosOcpSolver(
            ocp, json_file=str(json_file), cmake_builder=builder, verbose=False
        )
    _SOLVERS[key] = solver
    return solver


# endregion


def _overrides(cls: type, section: dict, prefix: str = "") -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    out = {}
    for key, value in section.items():
        if key.startswith(prefix) and key[len(prefix) :] in names:
            name = key[len(prefix) :]
            out[name] = tuple(value) if isinstance(value, list) else value
    return out


class ToptMPCC(Controller):
    """Cached time-optimal plan + reveal bending + attitude-mode MPCC."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Plan (or load the plan of) the nominal track, bend it to what is known, build the MPCC.

        Args:
            obs: The initial observation.
            info: The initial info dict.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        section = dict(config.get("controller") or {})
        self.cfg = replace(MpccConfig(), **_overrides(MpccConfig, section))
        plan_cfg = replace(PlannerConfig(), **_overrides(PlannerConfig, section, "plan_"))
        bend_cfg = replace(BendConfig(), **_overrides(BendConfig, section, "bend_"))
        self._freq = int(config.env.freq)
        self._dt = 1.0 / self._freq
        randomizations = config.env.get("randomizations") or {}
        self._randomized = any(k in randomizations for k in ("gate_pos", "gate_rpy"))

        # The plan is made against the nominal track and the nominal start, which are the same in
        # every episode, so it is solved once and cached; the bend then takes the real poses.
        track_cfg = config.env.track
        nominal_gates = np.array([g["pos"] for g in track_cfg.gates], dtype=float)
        nominal_rot = R.from_euler("xyz", [g["rpy"] for g in track_cfg.gates]).as_matrix()
        nominal_poles = np.array([o["pos"] for o in track_cfg.obstacles], dtype=float)
        self._sequence = np.asarray(obs["gate_sequence"], dtype=int)
        directions = np.asarray(obs["gate_sequence_direction"], dtype=float)
        seen, unseen = np.zeros(len(nominal_gates), dtype=bool), []
        for g in self._sequence:
            unseen.append(bool(self._randomized and not seen[g]))
            seen[g] = True
        self._unseen_pass = np.array(unseen)
        track = Track(
            nominal_gates, nominal_rot, nominal_poles, self._sequence, directions, tuple(unseen)
        )
        lift = np.array([0.0, 0.0, self.cfg.takeoff_dz])
        nominal_start = np.asarray(track_cfg.drones[0]["pos"], dtype=float) + lift
        t0 = time.perf_counter()
        trajectory = plan(track, nominal_start, np.array([0.0, 0.0, self.cfg.takeoff_vz]), plan_cfg)
        self.plan_time = time.perf_counter() - t0
        self.plan_duration = trajectory.duration
        self._bender = PathBender(trajectory, nominal_gates, nominal_rot, self._sequence, bend_cfg)

        self._solver = get_solver(self.cfg, self._dt)
        self._solver.reset()
        self._start = np.asarray(obs["pos"], dtype=float).copy()
        self._known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        self._tick = 0
        self._airborne = False
        self._s = 0.0
        self._s_remap: tuple[NDArray, NDArray] | None = None
        self._extra_iterations = 0
        self._warm = False
        self._ds_pred = np.zeros(self.cfg.horizon + 1)
        self._v_pred: NDArray[np.floating] | None = None
        self._dist = np.zeros(3)
        self._thrust_factor = 1.0
        self._f_est = 0.0
        self._f_cmd = self.cfg.takeoff_thrust
        self._acc = np.zeros(2)
        self._failures = 0
        self._last_status = 0
        self._last_ms = 0.0
        self._last_cmd = np.zeros(4)
        self.plan_log: list[dict[str, Any]] = []
        self._path: ArcPath | None = None
        self._build_path(obs, "episode start")

    # region path

    def _build_path(self, obs: dict[str, NDArray[np.floating]], reason: str):
        """Bend the plan to the poses known now, and rebuild the tunnel and the speed limit."""
        t0 = time.perf_counter()
        old = self._path
        drone_bead = None
        if old is not None and self._airborne:
            drone_bead = int(np.clip(np.searchsorted(old.knot_s, self._s), 0, len(old.knot_s) - 2))
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_rot = R.from_quat(obs["gates_quat"]).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        start = self._start + np.array([0.0, 0.0, self.cfg.takeoff_dz])
        beads = self._bender.bend(gates_pos, gates_rot, poles, start, drone_bead)
        self._path = ArcPath(beads)
        if old is not None:
            # Bead k of the old path is bead k of the new one: carry the progress across.
            self._s_remap = (old.knot_s.copy(), self._path.knot_s.copy())
            self._s = float(np.interp(self._s, *self._s_remap))
            self._extra_iterations = self.cfg.rebuild_iterations

        cfg, path = self.cfg, self._path
        clearance = np.minimum(
            frame_clearance(path.pts, gates_pos, gates_rot), pole_clearance(path.pts, poles)
        )
        self._tunnel = np.clip(
            clearance - DRONE_RADIUS - cfg.tunnel_margin, cfg.tunnel_min, cfg.tunnel_max
        )
        limit = cfg.speed_scale * self._bender.speed + cfg.speed_add
        visited = np.asarray(obs["gates_visited"], dtype=bool)
        n_cap = int(round((cfg.unseen_length + 0.1) / self._bender.config.spacing))
        for i, (g, k_c) in enumerate(zip(self._sequence, self._bender.crossings, strict=True)):
            if self._unseen_pass[i] and not visited[g]:
                limit[max(k_c - n_cap, 0) : k_c + 1] = np.minimum(
                    limit[max(k_c - n_cap, 0) : k_c + 1], cfg.unseen_speed
                )
        self._vmax = np.minimum(cfg.v_max, np.interp(path.ss, path.knot_s, limit))
        self._crossing_s = path.knot_s[self._bender.crossings]
        self.plan_log.append(
            {
                "reason": reason,
                "t": self._tick * self._dt,
                "tick": self._tick,
                "theta": self._s,
                "length": path.length,
                "n_waypoints": len(beads),
                "ms": 1000.0 * (time.perf_counter() - t0),
                "drone_pos": np.asarray(obs["pos"], dtype=float).copy(),
                "points": path(np.linspace(0.0, path.length, 200)),
            }
        )

    # endregion

    # region MPCC

    def _solve(
        self,
        p: NDArray[np.floating],
        v: NDArray[np.floating],
        tilt: NDArray[np.floating],
        rate: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float] | None:
        """Run one MPCC iteration; return the attitude command and the thrust, or None."""
        cfg, path, solver = self.cfg, self._path, self._solver
        n, dt = cfg.horizon, self._dt
        f_cmd0 = float(np.clip(self._f_cmd, cfg.thrust_min, 0.8))
        s0 = self._s
        ds0 = self._ds_pred[1] if self._warm else float(np.dot(v, path.tangent(s0)))
        ds0 = float(np.clip(ds0, 0.0, cfg.v_max))
        x0 = np.r_[p, v, tilt, rate, self._acc, self._f_est, f_cmd0, s0, ds0]
        if self._warm:
            xs = np.array([solver.get(j, "x") for j in range(n + 1)])
            us = np.array([solver.get(j, "u") for j in range(n)])
            if self._s_remap is not None:
                xs[:, IX_S] = np.interp(xs[:, IX_S], *self._s_remap)
            s_lin = np.r_[xs[1:, IX_S], xs[-1, IX_S] + dt * xs[-1, IX_DS]]
            s_lin += s0 - s_lin[0]
        else:
            ds_guess = max(ds0, 0.5)
            s_lin = s0 + dt * ds_guess * np.arange(n + 1)
        self._s_remap = None
        s_lin = np.clip(s_lin, 0.0, path.length)
        p_lin, t_lin = path(s_lin), path.tangent(s_lin)
        # Take the smaller of the two neighbouring table entries, never an optimistic interpolation.
        idx = np.clip(np.searchsorted(path.ss, s_lin, side="right") - 1, 0, len(path.ss) - 1)
        nxt = np.minimum(idx + 1, len(path.ss) - 1)
        r_tunnel = np.minimum(self._tunnel[idx], self._tunnel[nxt])
        vmax = np.minimum(self._vmax[idx], self._vmax[nxt])

        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        yref = np.zeros(NY)
        for j in range(n + 1):
            solver.set(
                j,
                "p",
                np.r_[self._dist, p_lin[j], t_lin[j], s_lin[j], r_tunnel[j], vmax[j], self._kappa],
            )
            if j < n:
                yref[4] = vmax[j] + cfg.progress_add
                solver.set(j, "yref", yref)
        for j in range(1, n + 1):
            solver.constraints_set(
                j, "ubx", np.array([2.4, cfg.tilt_max, cfg.tilt_max, cfg.thrust_max, vmax[j]])
            )
        if self._warm:
            for j in range(n):
                x = xs[min(j + 1, n)].copy()
                x[IX_S] = s_lin[j]
                solver.set(j, "x", x)
                solver.set(j, "u", us[min(j + 1, n - 1)])
            x = xs[n].copy()
            x[IX_S] = s_lin[n]
            solver.set(n, "x", x)
        else:
            for j in range(n + 1):
                x = np.r_[
                    p_lin[j],
                    t_lin[j] * ds_guess,
                    tilt,
                    np.zeros(4),
                    self._f_est,
                    f_cmd0,
                    s_lin[j],
                    ds_guess,
                ]
                solver.set(j, "x", x)
                if j < n:
                    solver.set(j, "u", np.zeros(NU))
        solver.set(0, "x", x0)
        t0 = time.perf_counter()
        status = solver.solve()
        for _ in range(self._extra_iterations):
            if status != 0:
                break
            status = solver.solve()
        self._extra_iterations = 0
        self._last_ms = 1000.0 * (time.perf_counter() - t0)
        self._last_status = int(status)
        x1 = solver.get(1, "x")
        if status != 0 or not np.all(np.isfinite(x1)):
            self._warm = False
            self._failures += 1
            return None
        self._warm = True
        self._v_pred = x1[3:6].copy()
        self._acc = x1[IX_ACC : IX_ACC + 2].copy()
        self._f_cmd = float(np.clip(x1[IX_FCMD], cfg.thrust_min, 0.8))
        self._ds_pred = np.array([solver.get(j, "x")[IX_DS] for j in range(n + 1)])
        # Invert the onboard law at the middle of the control interval.
        u0 = solver.get(0, "u")
        tilt_mid = 0.5 * (tilt + x1[IX_TILT : IX_TILT + 2])
        rate_mid = 0.5 * (rate + x1[IX_RATE : IX_RATE + 2])
        acc_mid = 0.5 * (x0[IX_ACC : IX_ACC + 2] + x1[IX_ACC : IX_ACC + 2])
        command = tilt_mid + (u0[0:2] + K2 * rate_mid + KD * acc_mid) / K1
        return np.clip(command, -cfg.cmd_max, cfg.cmd_max), self._f_cmd

    @property
    def _kappa(self) -> float:
        return self._thrust_factor

    def _observe(self, v: NDArray[np.floating], quat: NDArray[np.floating]):
        """Update the thrust factor and the additive disturbance from the velocity prediction."""
        if self._v_pred is None:
            return
        cfg = self.cfg
        error = (v - self._v_pred) * self._freq
        z_body = R.from_quat(quat).apply([0.0, 0.0, 1.0])
        along = float(error @ z_body)
        # A thrust factor is a mass, and a mass does not change in flight. At low thrust the error
        # along the body axis is dominated by what the model does not capture (the motors' floor,
        # the thrust lag), and attributing it to the factor ran it into its clip within 0.2 s and
        # then starved the next hard turn of thrust. So it only learns while the thrust is high
        # enough to say something about the mass, and only within the randomized range.
        if self._f_est >= cfg.thrust_factor_min_thrust:
            thrust_acc = self._f_est / MASS
            low, high = cfg.thrust_factor_range
            self._thrust_factor = float(
                np.clip(self._thrust_factor + cfg.k_thrust * along / thrust_acc, low, high)
            )
            error = error - (1.0 - cfg.dist_share) * along * z_body
        bound = np.asarray(cfg.dist_max)
        self._dist = np.clip(self._dist + cfg.k_dist * error, -bound, bound)

    def _track_thrust(self, f_cmd: float):
        self._f_est = f_cmd + (self._f_est - f_cmd) * np.exp(-self._dt / TAU_F)
        self._f_cmd = f_cmd

    # endregion

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, thrust]`` for the current observation."""
        cfg = self.cfg
        p = np.asarray(obs["pos"], dtype=float)
        v = np.asarray(obs["vel"], dtype=float)
        quat = np.asarray(obs["quat"], dtype=float)
        ang_vel = np.asarray(obs["ang_vel"], dtype=float)
        # Progress only searches a little backwards, or a drone falling behind would drag its own
        # reference back with it.
        self._s = self._path.project(p, self._s, back=0.05, ahead=0.8)

        self._airborne |= p[2] > self._start[2] + cfg.takeoff_dz
        if not self._airborne:
            self._v_pred = None
            self._track_thrust(cfg.takeoff_thrust)
            return self._send(np.zeros(2), cfg.takeoff_thrust)

        z_up = float(R.from_quat(quat).apply([0.0, 0.0, 1.0])[2])
        if z_up < np.cos(cfg.rescue_tilt):  # outside the model: level out first
            self._v_pred = None
            self._warm = False
            self._acc = np.zeros(2)
            self._track_thrust(0.6)
            return self._send(np.zeros(2), 0.6)

        self._observe(v, quat)
        tilt, rate = _tilt_state(quat, ang_vel)
        result = self._solve(p, v, tilt, rate)
        if result is None:
            self._v_pred = None
            hover = MASS * GRAVITY[2] / max(self._thrust_factor, 0.5)
            self._track_thrust(hover)
            return self._send(0.5 * tilt, hover)
        command, thrust = result
        self._track_thrust(thrust)
        return self._send(command, thrust)

    def _send(self, tilt_cmd: NDArray[np.floating], thrust: float) -> NDArray[np.floating]:
        self._last_cmd = np.array([tilt_cmd[0], tilt_cmd[1], 0.0, thrust])
        return np.clip(
            self._last_cmd, [-np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0857], [np.pi / 2] * 3 + [0.8]
        ).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Bend the path when a gate or pole is seen for the first time."""
        self._tick += 1
        known = np.r_[obs["gates_visited"], obs["obstacles_visited"]].astype(bool)
        new = known & ~self._known
        if np.any(new):
            self._known = known
            n_gates = len(obs["gates_visited"])
            names = [
                f"gate {i}" if i < n_gates else f"pole {i - n_gates}" for i in np.flatnonzero(new)
            ]
            self._build_path(obs, "reveal: " + ", ".join(names))
        return False

    def diagnostics(self) -> dict[str, float]:
        """Per-step values for `scripts/debug_run.py`."""
        index = int(np.clip(np.searchsorted(self._path.ss, self._s), 0, len(self._path.ss) - 1))
        return {
            "theta": float(self._s),
            "v_ref": float(self._vmax[index]),
            "tunnel": float(self._tunnel[index]),
            "solver_ms": float(self._last_ms),
            "solver_status": float(self._last_status),
            "mass_ratio": float(1.0 / self._thrust_factor),
            "replans": float(len(self.plan_log) - 1),
            "thrust": float(self._last_cmd[3]),
            "roll_cmd": float(self._last_cmd[0]),
            "pitch_cmd": float(self._last_cmd[1]),
        }

    def episode_callback(self):
        """Log the episode summary."""
        logger.info(
            "episode done: %d ticks, %d solver failures, plan %.2f s",
            self._tick,
            self._failures,
            self.plan_duration,
        )

    def episode_reset(self):
        """Reset the per-episode state that a new episode on the same instance would reuse."""
        self._tick = 0
        self._failures = 0
        self._warm = False
        self._v_pred = None
