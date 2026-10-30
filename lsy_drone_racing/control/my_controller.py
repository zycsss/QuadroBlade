"""Model Predictive Contouring Control (MPCC) for time-optimal drone racing.

The controller follows the formulation of *Model Predictive Contouring Control for Time-Optimal
Quadrotor Flight* (Romero et al., IEEE T-RO 2022) and the reference implementation of the successful
`lsy_drone_racing_ws26`_ entry. Instead of tracking a time-parameterized reference, the drone tracks
a 3D path while the optimizer chooses how fast to traverse it:

* The path [p_d(s)][lsy_drone_racing.control.path_planning.ArcPath] is generated once per episode by
  the [PathPlanner][lsy_drone_racing.control.path_planning.PathPlanner].
* The stage cost penalizes the *lag* error along the path tangent and the *contour* error normal to
  it, and rewards progress along the path. Both weights are scaled by a proximity weight that peaks
  at the gates and obstacles (the Gaussian contouring-weight allocation of the paper), so the drone
  hugs the path exactly where it has to pass through a gate.
* The optimal control problem is solved with acados at the environment frequency (50 Hz) using an
  RTI SQP iteration, which is fast enough to run in the control loop.

The predicted arc length of every stage is obtained from the current projection of the drone onto
the path plus a reference speed that is the closed-form minimizer of the progress cost. This mirrors
Algorithm 1 of the paper ("predict theta_k from the previous solution") while keeping the CasADi
cost small: the path reference enters the OCP as a per-stage parameter instead of a lookup table
inside the solver.

Conventions:
    * Gates are crossed along their local x-axis, so a forward pass goes from ``-x`` to ``+x``.
    * The action is ``[roll, pitch, yaw, collective_thrust]`` in SI units, which is the order
      implemented by [build_action_space][lsy_drone_racing.envs.race_core.build_action_space] and
      expected by the simulator's attitude controller.

.. _lsy_drone_racing_ws26: https://github.com/lihengrongfei/lsy_drone_racing_ws26
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.path_planning import (
    GATE_PROXIMITY_WEIGHT,
    OBSTACLE_PROXIMITY_WEIGHT,
    ArcPath,
    GateFrames,
    PathInfo,
    PathPlanner,
    PathSeed,
    path_config_from,
    proximity_weight,
    track_reference,
)

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_EXPORT_DIR = REPO_ROOT / "c_generated_code"
SOLVER_JSON = CODE_EXPORT_DIR / "mpcc_racing.json"
SIGNATURE_FILE = CODE_EXPORT_DIR / "mpcc_racing.signature"

#: State and input dimensions of the OCP.
NX, NU = 15, 5

#: Obstacle centres carried in the OCP parameter vector, for the distance penalty. Fixed at compile
#: time so the generated solver does not depend on the track; unused slots are sent far away, where
#: the penalty is inactive.
MAX_BARRIER_POLE = 6

#: Value written into an unused obstacle slot [m]. Far enough that the penalty is exactly zero.
BARRIER_OFF = 1.0e3

#: Tunnels carried in the OCP parameter vector, for the gate-passage constraint. Two, because a
#: 0.8 s horizon can hold two gate passes on this track; a third logs a warning, not a silent drop.
MAX_TUNNEL = 2

#: Slope of the tunnel barrier. After MPCC++ (Krinner, Romero et al., RSS 2024), which uses this
#: value: the barrier is nearly inert more than 5 cm inside the tunnel and nearly hard just outside.
TUNNEL_ALPHA = 100.0

#: Value written into an unused tunnel slot [m]. With a zero frame it makes every half-space slack
#: ``1e3``, so the barrier is exactly zero and the slot costs nothing but three multiplications.
TUNNEL_OFF = 1.0e3

#: Number of OCP parameters: reference position, tangent, proximity weight, pole centres, the tunnel
#: slots of ``[in-plane axis a, in-plane axis b, centre, half width]``, then the tunable weights.
NP_REFERENCE = 7
#: Cost weights that travel in the parameter vector rather than in the compiled expression, so a
#: tuning sweep does not need a one minute solver rebuild per evaluation. Their order is
#: the order of [WEIGHT_FIELDS][lsy_drone_racing.control.my_controller.WEIGHT_FIELDS].
#: [WEIGHT_FIELDS][lsy_drone_racing.control.my_controller.WEIGHT_FIELDS].
NP_WEIGHTS = 7
NP = NP_REFERENCE + 2 * MAX_BARRIER_POLE + MAX_TUNNEL * 10 + NP_WEIGHTS

#: The ``MPCCConfig`` weights carried in the parameter vector, in the order they are written.
WEIGHT_FIELDS = (
    "q_lag",
    "q_contour",
    "q_lag_peak",
    "q_contour_peak",
    "mu_speed",
    "r_vtheta",
    "w_tunnel",
)

#: Offset of the weight block inside the parameter vector.
NP_WEIGHT_OFFSET = NP_REFERENCE + 2 * MAX_BARRIER_POLE + MAX_TUNNEL * 10

#: Cap on the barrier exponent. ``log(1 + exp(50))`` is finite, so the cost cannot overflow the way
#: ``log(1 + exp(-alpha * h))`` would for a drone deep inside a narrow tunnel.
TUNNEL_BARRIER_CLIP = 50.0


def tunnel_barrier(slack: cs.SX | cs.DM) -> cs.SX | cs.DM:
    """Return the soft tunnel barrier over a vector of half-space slacks.

    ``log(1 + exp(-alpha * slack))`` is the barrier MPCC++ (Krinner, Romero et al., RSS 2024) uses
    in place of a hard constraint: nearly zero well inside the tunnel, ~``alpha/2`` per unit
    violation near a face, and cheap enough for a real-time SQP iteration. It is written against
    CasADi so the same function is what the solver evaluates and what the tests probe.

    Args:
        slack: Half-space slacks, positive inside the tunnel, as a CasADi expression or array.

    Returns:
        The sum of the barrier terms, with the same type as the input.
    """
    return cs.sum1(cs.log(1.0 + cs.exp(cs.fmin(-TUNNEL_ALPHA * slack, TUNNEL_BARRIER_CLIP))))


#: MPCC settings for a randomized track, i.e. for the levels that re-plan on sensor updates.
#:
#: These live here rather than in the race config because the competition checks the line count of
#: every provided config file and reports an edited one as cheating. They were tuned on level 2,
#: seeds 0-19: the MPCC defaults give 12.34 s per lap, this gives 8.17 s with the same success rate
#: (13/20, the failures are pole contacts caused by late reveals, not by the pace). The reference
#: speed is ``mu_speed / (2 * (w_speed_gate * gate_weight + r_vtheta))``, so ``w_speed_gate`` sets
#: the pace at the gates and ``a_lat_max`` the pace in the corners; ``v_max`` alone barely binds.
RANDOMIZED_TRACK_MPCC = {
    "w_speed_gate": 4.0,
    "a_lat_max": 8.0,
    "v_max": 3.5,
    "replan_slowdown_time": 0.3,
    # The distance penalty is left off: measured on level 2 it changed nothing (13/20 either way),
    # because where the pose is unknown it pushes away from the nominal pose rather than the one the
    # plan uses, and where the pose is known the path already keeps 0.187 m, outside the 0.14 m
    # radius. See the planning page for the numbers.
    "w_obstacle": 0.0,
}
_SOLVER_CACHE: dict[str, tuple[AcadosOcpSolver, AcadosOcp]] = {}


@dataclass(frozen=True)
class MPCCConfig:
    """Tunable parameters of the MPCC controller.

    The weights follow the reference implementation, with the contouring weight split into a
    constant part and a proximity-scaled part, see
    [track_reference][lsy_drone_racing.control.path_planning.track_reference].

    Attributes:
        N: Prediction horizon length.
        dt: Discretization step [s]. Set to ``1 / env.freq`` by the controller.
        q_lag: Lag error weight.
        q_contour: Contour error weight.
        q_lag_peak: Additional lag weight at the gates and obstacles.
        q_contour_peak: Additional contour weight at the gates and obstacles.
        w_obstacle: Weight of the one-sided distance penalty that pushes the drone away from an
            obstacle centre inside ``obstacle_radius``. Zero disables it and leaves the avoidance
            entirely to the planned path. It is a soft net, not a guarantee: it acts on the flown
            trajectory, so it can only react to a clearance the path did not provide.
        obstacle_radius: Distance [m] from an obstacle centre below which the penalty is active. Set
            it just above the drone's collision reach (0.085 m face-on, 0.114 m corner-on for the
            0.14 x 0.14 x 0.04 m box) so the term only fights a contact that is about to happen
            rather than reshaping the whole corner.
        w_tunnel: Weight of the gate-passage tunnel barrier. The tunnel is a box of half width
            ``W(arc)`` around the planned path, wide between gates and narrowing to the opening at
            each gate plane; the barrier makes the drone pay exponentially for approaching a face,
            where the contour cost only pays quadratically for being off the path. Zero disables the
            tunnel and leaves gate accuracy to ``q_contour_peak``, which is what the cost-level
            formulation does and what MPCC++ (Krinner, Romero et al., RSS 2024) argues against.
        tunnel_gate_half_width: Half width [m] of the tunnel at a gate plane. The opening is
            0.40 m wide and the drone's contact radius is 0.099 m, so 0.09 leaves about 11 mm.
        tunnel_nominal_half_width: Half width [m] of the tunnel at the edge of
            ``tunnel_window``. Between gates the drone is free to cut the corner, so this is wide.
        tunnel_window: Arc length [m] over which the tunnel narrows from the nominal half width to
            the gate half width. Beyond it the tunnel is inactive, which is what keeps it from
            fighting the pole avoidance between gates.
        contour_z_weight: Extra emphasis on the vertical contour error. The gates are only 0.4 m
            tall, so altitude accuracy, not lateral accuracy, limits the pass margin.
        q_attitude: Penalty on the absolute roll/pitch/yaw.
        r_rates: Penalty on the rate inputs ``[df, dr, dp, dy]``.
        mu_speed: Weight of the progress reward.
        r_vtheta: Quadratic regularization of the path progress rate.
        w_speed_gate: Proximity-dependent penalty on the path progress rate.
        gate_speed_weight: Peak of the speed law's own proximity bump at a gate, independent of the
            contour weight. Keeping it non-zero is what lets a controller lower the gate contour
            weight (or hand the gate to a tunnel) without also losing the approach slowdown that the
            revealed-pose correction depends on.
        v_min: Minimum reference path speed [m/s].
        v_max: Maximum reference path speed [m/s] and input bound on the progress rate.
        v_lead: Maximum speed [m/s] by which the path reference may run ahead of the drone. Without
            this limit the reference of the near stages would demand a physically impossible motion
            from a standing start.
        a_lat_max: Lateral acceleration [m/s^2] the path speed is limited to. The reference speed is
            capped by ``sqrt(a_lat_max / curvature)``, which slows the drone down exactly where the
            path bends instead of everywhere.
        curvature_smoothing: Number of samples of the curvature table that are averaged, so that a
            single noisy sample does not throttle the reference for a whole corner.
        thrust_lag: First-order lag of the collective thrust command.
        qp_solver: acados QP solver used inside the RTI iteration.
        condensing_N: Number of condensed stages. Values below ``N`` enable partial condensing,
            which solves the same problem with a smaller dense QP.
        attitude_limit: Bound [rad] on the roll/pitch/yaw setpoints.
        rate_limit: Bound on the attitude and thrust rate inputs.
        pos_low: Lower bounds ``[x, y, z]`` of the position. The lower z bound is deliberately far
            below the floor: the drone starts on the ground with a downward velocity, so any bound
            near z=0 makes the QP infeasible during take-off.
        pos_high: Upper bounds ``[x, y, z]`` of the position.
        mass_adapt: Whether to estimate the drone's mass online and rescale the commanded thrust.
            Level 0 keeps the nominal mass, levels 1-3 randomize it by up to about 12%.
        mass_adapt_gain: Low-pass gain of the mass-ratio estimate per control step.
        mass_adapt_window: Number of control steps the force balance is integrated over. A window
            rather than a single step is used because the thrust and attitude transients of the
            onboard controller would otherwise bias the estimate.
        mass_adapt_range: Admissible mass ratio ``(min, max)`` relative to the nominal mass.
        mass_adapt_deadband: Mass ratios closer to 1 than this are treated as exactly 1.
        mass_adapt_min_thrust: Minimum commanded thrust [N] for which an estimate is taken.
        mass_adapt_min_balance: Minimum integrated vertical balance [m/s] of the window. The window
            integrates ``(a_z + g) * window * dt``, so this rejects windows in which a strong dive
            almost cancels gravity and the estimate would be ill-conditioned.
        mass_adapt_max_speed: Maximum drone speed [m/s] for which an estimate is taken. At higher
            speeds the aerodynamic drag biases the vertical force balance.
        replan_slowdown_time: Time [s] for which the reference speed is reduced after a re-plan.
            A revealed gate can be up to 0.2 m away from its nominal pose, and the drone needs a
            moment to converge onto the corrected approach line before it reaches the gate.
        replan_slowdown_factor: Factor applied to the reference speed during that time. Measured on
            level 2, seeds 0-39: 0.5 gives 35/40 at 8.37 s, 0.7 gives **32/40 at 7.78 s**, 0.8 gives
            14/20 at 7.49 s and 0.35 with a 0.5 s window gives 14/20 at 9.34 s. The default is 0.7:
            0.59 s a lap for three finishes, taken deliberately because the lap time is the score.
        replan_coalesce_ticks: Quiet period in control ticks before a pending reveal is planned for.
            Reveals arrive in bursts, 36 percent of them within the previous re-plan's slowdown
            window, and three paths in a quarter of a second buy nothing.
        replan_max_defer_ticks: Hard cap on how long a pending reveal may wait, in control ticks. It
            overrides both the quiet period and the passage deferral, so a reveal can never be
            starved.
        replan_defer_in_passage: Whether to hold a re-plan while the drone flies a gate passage.
            Re-planning there puts the first block behind the drone, which folds the path back on
            itself within centimetres of the anchor.
        reveal_speed: Reference speed ceiling [m/s] in front of a gate that has not been seen yet.
            The ceiling is ``sensor_range / 0.46 s``, the time the drone needs to correct the worst
            pose error, so it is an information limit rather than a tuning knob.
        reveal_window: Arc length [m] in front of a gate over which ``reveal_speed`` applies.
        reveal_hold_time: Time [s] the cap stays on after a gate pose was corrected. The reveal is
            when the correction starts, not when it is finished.
        name: Name of the acados model and of the generated code.
    """

    N: int = 40
    dt: float = 0.02
    q_lag: float = 100.0
    q_contour: float = 100.0
    # Measured on level 2, seeds 0-39, at 300 with a 0.16 m radius: 29/40 at 9.20 s against 32/40 at
    # 8.86 s, and the pole failures it targets (8, 9, 33) all still fail, so its lap time is
    # spent for nothing. Left off; the transients that hit a pole are a timing problem, not a
    # controller-weight problem. The code path stays: it is the right place for a barrier once
    # the plan can guarantee a larger margin.
    w_obstacle: float = 0.0
    obstacle_radius: float = 0.14
    # Zero in the plumbing commit: with the tunnel off this file must reproduce the 35/40 measured
    # before it, which is also how the parameter wiring is verified to be inert.
    w_tunnel: float = 0.0
    tunnel_gate_half_width: float = 0.09
    tunnel_nominal_half_width: float = 0.40
    tunnel_window: float = 0.6
    q_lag_peak: float = 2000.0
    q_contour_peak: float = 6000.0
    contour_z_weight: float = 2.0
    q_attitude: float = 1.0
    r_rates: tuple[float, float, float, float] = (0.2, 0.3, 0.3, 0.5)
    mu_speed: float = 10.0
    r_vtheta: float = 0.2
    w_speed_gate: float = 18.0
    gate_speed_weight: float = GATE_PROXIMITY_WEIGHT
    v_min: float = 0.5
    v_max: float = 2.3
    v_lead: float = 0.6
    a_lat_max: float = 4.0
    curvature_smoothing: int = 9
    thrust_lag: float = 10.0
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    condensing_N: int = 10
    attitude_limit: float = 1.57
    rate_limit: float = 10.0
    pos_low: tuple[float, float, float] = (-2.45, -1.45, -10.0)
    pos_high: tuple[float, float, float] = (2.45, 1.45, 2.45)
    mass_adapt: bool = True
    mass_adapt_gain: float = 0.1
    mass_adapt_window: int = 10
    mass_adapt_range: tuple[float, float] = (0.85, 1.15)
    mass_adapt_deadband: float = 0.01
    mass_adapt_min_thrust: float = 0.3
    mass_adapt_min_balance: float = 1.0
    mass_adapt_max_speed: float = 1.5
    replan_slowdown_time: float = 0.8
    # Measured on level 2: any deferral loses the marginal cases. Debouncing by three ticks (0.06 s)
    # dropped the 20-seed screen from 17/20 to 9/20, because the first-leg reveals already have no
    # slack to give. Left at immediate re-plan; see the planning page for the numbers.
    replan_coalesce_ticks: int = 0
    replan_max_defer_ticks: int = 0
    replan_defer_in_passage: bool = False
    replan_slowdown_factor: float = 0.7
    # Measured on level 2: braking to 0.15 of the reference speed for 0.15 s after every re-plan
    # costs 0.31 s of lap time, 8.99 s to 9.30 s, and does not raise the finish count on the 20-seed
    # screen. The hold is therefore off: a factor of 1.0 with a zero window makes it a no-op.
    replan_hold_time: float = 0.0
    replan_hold_factor: float = 1.0
    # A gate that has not been seen yet may be up to 0.21 m from the pose the plan aims at, and the
    # drone needs about 0.46 s of lateral travel to make that up. 0.7 m of sensor range therefore
    # only leaves a safe speed of 0.7 / 0.46 = 1.5 m/s, and the cap is applied over an arc-length
    # window in front of the gate so the drone has already paid for the correction when it gets
    # there. Measured on level 2: see the planning page.
    reveal_speed: float = 1.5
    reveal_window: float = 1.2
    # Measured: keeping this at 0.5 s costs finishes now that the plan is better. The 35/40
    # configuration runs with the hold off, and the correction is paid for by the post-replan
    # slowdown instead - see the speed sweep on the planning page.
    reveal_hold_time: float = 0.0
    name: str = "mpcc_racing"

    @property
    def horizon_time(self) -> float:
        """Length of the prediction horizon in seconds."""
        return self.N * self.dt


def _signature(config: MPCCConfig, params: dict) -> str:
    """Compute a signature of everything that defines the generated solver.

    Only the fields [build_ocp][lsy_drone_racing.control.my_controller.build_ocp] reads are hashed.
    Hashing the whole dataclass meant every controller-only tweak - a replan slowdown factor, a
    reveal window, a gate-correction hold - threw away the compiled solver and paid a one minute
    rebuild, which is how a run ended in a killed cmake build. The opposite mistake is worse: a
    field that does reach the OCP but is missing here would silently reuse a stale solver, so the
    list below is checked against `build_ocp` and the dimensions are hashed alongside it.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.

    Returns:
        A short hexadecimal signature.
    """
    payload = {
        "config": {name: getattr(config, name) for name in _OCP_FIELDS},
        # The trap the planning page records: a changed NP reuses the stale generated code silently.
        "dimensions": [NX, NU, NP],
        "mass": float(params["mass"]),
        "thrust": [float(params["thrust_min"]), float(params["thrust_max"])],
        "lag": [np.asarray(params[k], dtype=float).tolist() for k in _LAG_KEYS],
        "gravity": np.asarray(params["gravity_vec"], dtype=float).tolist(),
    }
    digest = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(digest).hexdigest()[:16]


#: The MPCCConfig fields that `build_ocp` reads *and* bakes into the compiled expression. Everything
#: else only affects the running controller, or travels in the parameter vector (``WEIGHT_FIELDS``),
#: so changing it must not invalidate the compiled solver.
_OCP_FIELDS = (
    "N",
    "dt",
    "w_obstacle",
    "obstacle_radius",
    "contour_z_weight",
    "q_attitude",
    "r_rates",
    "v_max",
    "thrust_lag",
    "qp_solver",
    "condensing_N",
    "attitude_limit",
    "rate_limit",
    "pos_low",
    "pos_high",
    "name",
)

_LAG_KEYS = ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef")


def _lag_coefficients(params: dict) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Derive the first-order attitude lag from the identified `so_rpy` model.

    Args:
        params: Drone parameters of the `so_rpy` dynamics.

    Returns:
        The decay coefficients ``a`` and the command gains ``b``, both shape ``(3,)``.
    """
    decay = -np.asarray(params["rpy_coef"], dtype=float)
    rates = np.asarray(params["rpy_rates_coef"], dtype=float)
    gain = -np.asarray(params["cmd_rpy_coef"], dtype=float)
    return decay / rates, gain / rates


def build_ocp(config: MPCCConfig, params: dict) -> AcadosOcp:
    """Build the acados optimal control problem of the MPCC.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.

    Returns:
        The fully specified (but not yet generated/solved) OCP.
    """
    mass = float(params["mass"])
    gravity = -float(np.asarray(params["gravity_vec"], dtype=float)[-1])
    lag_a, lag_b = _lag_coefficients(params)
    thrust_min = 4.0 * float(params["thrust_min"])
    thrust_max = 4.0 * float(params["thrust_max"])

    x = cs.SX.sym("x", NX)
    u = cs.SX.sym("u", NU)
    p = cs.SX.sym("p", NP)
    roll, pitch, yaw = x[6], x[7], x[8]
    thrust = x[9]
    thrust_cmd = x[10]
    roll_cmd, pitch_cmd, yaw_cmd = x[11], x[12], x[13]
    v_theta = u[4]

    # Translational and first-order command dynamics, see Eqs. 12-16 of the T-RO22 paper. The
    # attitude and thrust setpoints are states whose derivatives are the inputs, which gives the
    # command smoothing that keeps the solver output usable.
    dynamics = cs.vertcat(
        x[3],
        x[4],
        x[5],
        thrust / mass * (cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw)),
        thrust / mass * (cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw)),
        thrust / mass * cs.cos(roll) * cs.cos(pitch) - gravity,
        lag_a[0] * roll + lag_b[0] * roll_cmd,
        lag_a[1] * pitch + lag_b[1] * pitch_cmd,
        lag_a[2] * yaw + lag_b[2] * yaw_cmd,
        config.thrust_lag * (thrust_cmd - thrust),
        u[0],
        u[1],
        u[2],
        u[3],
        v_theta,
    )

    model = AcadosModel()
    model.name = config.name
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = dynamics
    model.f_impl_expr = None

    # Contouring cost: lag and contour error of the position with respect to the path reference of
    # this stage, weighted by the proximity-aware contouring weights.
    error = x[0:3] - p[0:3]
    tangent = p[3:6] / (cs.norm_2(p[3:6]) + 1e-6)
    lag = cs.dot(tangent, error)
    contour = error - lag * tangent
    # Emphasize the vertical contour error: the drone tends to arrive high at the shortest gates,
    # and the gates are only 0.4 m tall, so altitude accuracy is what limits the pass margin.
    contour = cs.vertcat(contour[0], contour[1], config.contour_z_weight * contour[2])
    # The weights that a tuning sweep touches travel in the parameter vector (WEIGHT_FIELDS), so a
    # sweep costs episodes, not one-minute solver rebuilds per evaluation.
    w = p[NP_WEIGHT_OFFSET : NP_WEIGHT_OFFSET + NP_WEIGHTS]
    q_lag = w[0] + w[2] * p[6]
    q_contour = w[1] + w[3] * p[6]
    cost = q_lag * cs.dot(lag, lag) + q_contour * cs.dot(contour, contour)
    cost += config.q_attitude * cs.dot(x[6:9], x[6:9])
    rates = u[0:4]
    cost += cs.dot(rates, cs.mtimes(cs.DM(np.diag(config.r_rates)), rates))
    cost += w[5] * v_theta * v_theta - w[4] * v_theta
    # One-sided distance penalty: zero while the drone keeps ``obstacle_radius`` from a pole centre,
    # quadratic inside it. ``max(0, r - d)**2`` is continuously differentiable at the boundary,
    # so it adds no kink to the cost, and unused slots sit at BARRIER_OFF where it is zero. It is
    # a soft net over a planned path, not a substitute for one: it can only react to a clearance
    # path did not provide, and it trades against the progress reward.
    if config.w_obstacle > 0.0:
        for slot in range(MAX_BARRIER_POLE):
            offset = 1 + 2 * slot
            gap_x = x[0] - p[offset]
            gap_y = x[1] - p[offset + 1]
            distance = cs.sqrt(gap_x * gap_x + gap_y * gap_y + 1e-9)
            violation = cs.fmax(0.0, config.obstacle_radius - distance)
            cost += config.w_obstacle * violation * violation
    # Gate-passage tunnel: a box of half width W(arc) around the planned path, wide between gates
    # and narrowing to the opening at each gate plane. The paper's four half-spaces around the
    # corner ``p0 = c - W n - W b`` collapse to ``|(x - c).n| <= W`` and ``|(x - c).b| <= W``,
    # i.e. a square cross-section, which is the shape of the gates. The barrier
    # ``log(1 + exp(-alpha h))`` is nearly inert well inside the tunnel and grows exponentially as a
    # face is approached, so the drone cannot buy progress with a gate clip the way it can against a
    # quadratic contour cost. Unused slots carry a zero frame and ``W = TUNNEL_OFF``, which makes
    # every slack huge and the barrier exactly zero.
    for slot in range(MAX_TUNNEL):
        offset = NP_REFERENCE + 2 * MAX_BARRIER_POLE + 10 * slot
        # The two axes span the gate opening; the opening's normal is the corridor axis and is
        # deliberately not constrained, which is what makes the tunnel a corridor and not a box.
        axis_a = p[offset : offset + 3]
        axis_b = p[offset + 3 : offset + 6]
        delta = x[0:3] - p[offset + 6 : offset + 9]
        half_width = p[offset + 9]
        along_a = cs.dot(delta, axis_a)
        along_b = cs.dot(delta, axis_b)
        cost += w[6] * tunnel_barrier(
            cs.vertcat(
                half_width - along_a,
                half_width + along_a,
                half_width - along_b,
                half_width + along_b,
            )
        )
    model.cost_expr_ext_cost = cost
    model.cost_expr_ext_cost_e = cs.SX(0)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = str(CODE_EXPORT_DIR)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.parameter_values = np.full(NP, BARRIER_OFF)
    ocp.parameter_values[:7] = 0.0
    ocp.parameter_values[9:19] = TUNNEL_OFF
    for index, name in enumerate(WEIGHT_FIELDS):
        ocp.parameter_values[NP_WEIGHT_OFFSET + index] = getattr(config, name)

    # Box constraints on position, thrust setpoints, attitude setpoints and path progress.
    ocp.constraints.idxbx = np.array([0, 1, 2, 9, 10, 11, 12, 13, 14])
    ocp.constraints.lbx = np.array(
        [
            *config.pos_low,
            thrust_min,
            thrust_min,
            -config.attitude_limit,
            -config.attitude_limit,
            -config.attitude_limit,
            0.0,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            *config.pos_high,
            thrust_max,
            thrust_max,
            config.attitude_limit,
            config.attitude_limit,
            config.attitude_limit,
            1.0,
        ]
    )
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array(
        [-config.rate_limit, -config.rate_limit, -config.rate_limit, -config.rate_limit, 0.0]
    )
    ocp.constraints.ubu = np.array(
        [config.rate_limit, config.rate_limit, config.rate_limit, config.rate_limit, config.v_max]
    )
    # The initial state is fixed by a full-state box equality that is overwritten every control
    # step. acados requires the stage-0 box to span the full state when an initial state is given.
    ocp.constraints.idxbx_0 = np.arange(NX)
    ocp.constraints.lbx_0 = np.zeros(NX)
    ocp.constraints.ubx_0 = np.zeros(NX)
    ocp.constraints.x0 = np.zeros(NX)

    ocp.solver_options.N_horizon = config.N
    ocp.solver_options.tf = config.horizon_time
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = config.qp_solver
    ocp.solver_options.qp_solver_cond_N = config.condensing_N
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.print_level = 0
    return ocp


def _solver_artifacts_exist() -> bool:
    """Check whether a generated and compiled solver is on disk."""
    return SOLVER_JSON.exists() and (CODE_EXPORT_DIR / "acados_ocp_solver_mpcc_racing.dll").exists()


def get_solver(
    config: MPCCConfig, params: dict, force_build: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Return the acados solver, reusing cached or previously generated artifacts when possible.

    Generating and compiling the solver takes up to a minute on Windows, while the evaluation
    scripts construct a new controller for every episode. The solver is therefore cached per
    process and the generated artifacts are reused across processes as long as the signature of the
    OCP is unchanged.

    Args:
        config: MPCC configuration.
        params: Drone parameters used by the model.
        force_build: Regenerate and rebuild the solver even if artifacts are present.

    Returns:
        The solver and the OCP it was built from.
    """
    signature = _signature(config, params)
    if not force_build and signature in _SOLVER_CACHE:
        return _SOLVER_CACHE[signature]

    ocp = build_ocp(config, params)
    reuse = (
        not force_build
        and _solver_artifacts_exist()
        and SIGNATURE_FILE.exists()
        and SIGNATURE_FILE.read_text().strip() == signature
    )
    if reuse:
        logger.info("Loading cached MPCC solver from %s", SOLVER_JSON)
        solver = AcadosOcpSolver(
            ocp, json_file=str(SOLVER_JSON), generate=False, build=False, verbose=False
        )
    else:
        logger.info("Generating and building the MPCC solver (this takes a while)...")
        cmake_builder = None
        if os.name == "nt":  # The Windows toolchain requires the CMake builder with MinGW.
            cmake_builder = ocp_get_default_cmake_builder()
            cmake_builder.generator = "MinGW Makefiles"
            cmake_builder.build_dir = "build-mingw-mpcc"
        solver = AcadosOcpSolver(
            ocp,
            json_file=str(SOLVER_JSON),
            generate=True,
            build=True,
            cmake_builder=cmake_builder,
            verbose=False,
        )
        CODE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        SIGNATURE_FILE.write_text(signature)
        logger.info("MPCC solver built and stored in %s", CODE_EXPORT_DIR)
    _SOLVER_CACHE[signature] = (solver, ocp)
    return solver, ocp


def build_default_solver(force: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the default level-0 solver.

    Args:
        force: Regenerate and rebuild the solver even if artifacts are present.

    Returns:
        The solver and the OCP it was built from.
    """
    params = load_params(Dynamics.so_rpy, "cf21B_500")
    return get_solver(MPCCConfig(), params, force_build=force)


class MPCCController(Controller):
    """Time-optimal drone racing controller based on Model Predictive Contouring Control."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller, plan the path and build the solver.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._config = config
        self._freq = float(config.env.freq)
        # The strict gate handling (straight passages, gate-frame avoidance, slower gate approach,
        # strong gate centring) only pays off when the gate poses are randomized: it costs lap time,
        # and on a perfectly known track the path already passes the gates cleanly. Enable it only
        # for configs that randomize the track.
        randomizations = config.env.get("randomizations") or {}
        randomized_track = any(
            key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos")
        )
        mpcc_overrides = (
            dict(RANDOMIZED_TRACK_MPCC)
            if randomized_track
            else {"w_speed_gate": 9.0, "q_contour_peak": 2000.0}
        )
        # A `[controller]` entry that names an MPCCConfig field still overrides the defaults, so
        # the aggressiveness knobs can be swept from a scratch config. The race configs must stay
        # byte-identical to the official ones: the competition checks their line count, and editing
        # them is reported as cheating.
        controller_cfg = config.get("controller") or {}
        for key in MPCCConfig.__dataclass_fields__:
            if key in controller_cfg:
                mpcc_overrides[key] = controller_cfg[key]
        self._mpcc = replace(MPCCConfig(dt=1.0 / self._freq), **mpcc_overrides)
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._hover_thrust = float(self._params["mass"]) * (-float(self._params["gravity_vec"][-1]))
        self._thrust_limits = (
            4.0 * float(self._params["thrust_min"]),
            4.0 * float(self._params["thrust_max"]),
        )
        # The strict gate handling (straight passages and gate-frame avoidance) only pays off when
        # the gate poses are randomized: it costs lap time, and on a perfectly known track the path
        # already passes the gates cleanly. `path_config_from` applies that rule and the config
        # overrides of the race config in one place.
        self._planner = PathPlanner(path_config_from(config))
        # The reveal-aware approach cap is information, not tuning: it buys the time to correct a
        # pose that is still unknown. A track that does not randomize its poses has nothing to
        # correct, so the cap would be pure lap time there.
        randomizations = config.env.get("randomizations") or {}
        self._randomized_track = any(
            key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos")
        )
        self._solver, _ = get_solver(self._mpcc, self._params)
        self._nominal_mass = float(self._params["mass"])
        self._mass_ratio = 1.0
        self._applied_thrust = self._hover_thrust
        self._prev_vel: NDArray[np.floating] | None = None
        self._accel = np.zeros(3)
        self._reset_mass_history()
        self._path: ArcPath | None = None
        self._gate_centers = np.zeros((0, 3))
        self._obstacles = np.zeros((0, 3))
        self._draw_points = np.zeros((0, 3))
        self._horizon_points = np.zeros((0, 3))
        self._needs_plan = True
        self._tick = 0
        self._failures = 0
        self._stall_time = 0.0
        self._stall_reported = False
        self.status = 0
        self.solve_time = 0.0
        # Set before the first plan so the opening plan is timed like every re-plan.
        self._last_plan_ms = 0.0
        self._init_episode(obs)

    # region lifecycle

    def _init_episode(self, obs: dict[str, NDArray[np.floating]], reason: str = "episode start"):
        """Plan the path and reset all controller memory for a new episode.

        Args:
            obs: The current observation.
            reason: What to record as the reason for the path in the plan log.
        """
        self._plan_origin = np.asarray(obs["pos"], dtype=float).copy()
        started = time.perf_counter()
        path, info = self._planner.plan(obs, self._plan_origin)
        self._last_plan_ms = (time.perf_counter() - started) * 1e3
        self._install_path(path, info)
        self._reset_state(obs, reason)

    def _reset_state(self, obs: dict[str, NDArray[np.floating]], reason: str):
        """Reset all per-episode controller memory once a path is installed.

        Split out of [MPCCController._init_episode][] so a controller that supplies its own path
        (the scenario controller) resets exactly the same state in exactly the same order.

        Args:
            obs: The current observation.
            reason: What to record as the reason for the path in the plan log.
        """
        self._seed_visited_flags(obs)
        self._reproject_wide = False
        self._projection_window = (-0.15, 0.6)
        self._slowdown_until = 0
        self._hold_until = 0
        self.replans = 0
        self.plan_log: list[dict[str, object]] = []
        # Reveals waiting to be planned for, and when the first and latest of them arrived.
        self._pending: list[str] = []
        self._pending_since = 0
        self._pending_tick = 0
        self._last_v_ref = 0.0
        self._last_f_col = self._hover_thrust
        self._last_f_cmd = self._hover_thrust
        self._last_rpy_cmd = np.zeros(3)
        self._mass_ratio = 1.0
        self._applied_thrust = self._hover_thrust
        self._prev_vel = None
        self._accel = np.zeros(3)
        self._reset_mass_history()
        self._theta = 0.0
        self.min_slack = TUNNEL_OFF
        self.gate_gap = float("nan")
        self.plane_gap = float("nan")
        self._tunnel_warned = False
        self._tick = 0
        self._failures = 0
        self._stall_time = 0.0
        self._stall_reported = False
        self._next_gate_seen = True
        self._gate_corrected_tick = -(10**6)
        self._needs_plan = False
        self._solver.reset()
        # Last, so the entry sees the reset progress and tick, not the previous episode's.
        self._record_plan(reason, obs)

    def _install_path(self, path: ArcPath, info: PathInfo):
        """Make a path the active reference and rebuild everything that depends on it.

        Shared by the race planner and by a scenario controller that injects an analytic path
        instead of planning one, so both go through exactly the same reference machinery.

        Args:
            path: The path to follow.
            info: Its metadata, holding the gate centres and obstacles the reference weights use.
        """
        self._path = path
        self._path_info = info
        self._gate_centers = info.gate_centers
        self._obstacles = info.obstacles
        self._draw_points = path.pos(np.linspace(0.0, path.length, 200))
        # Arc length of every remaining gate pass on this path, for the tunnel funnel and the
        # approach speed cap. The planner computes these as plane *crossings*: a nearest-point
        # lookup put the next gate 8.36 m away on a 9.39 m path and left both mechanisms inert.
        arcs = np.asarray(getattr(info, "pass_arcs", np.zeros(0)), dtype=float).reshape(-1)
        if len(arcs) == len(info.gate_centers):
            self._gate_arc = arcs
        else:
            # A path handed in without crossing arcs (the scenario controller) falls back to the
            # nearest point, which is only used for its gates.
            self._gate_arc = np.array(
                [
                    path.closest(centre, 0.0, window=(-0.15, path.length))
                    for centre in info.gate_centers
                ]
            )
        self._build_speed_limit()
        self._apply_progress_bound()

    def _barrier_params(self) -> NDArray[np.floating]:
        """Return the obstacle centres for the distance penalty, padded with the off value.

        The centres are whatever the controller currently believes: the nominal pose for an obstacle
        it has not seen yet, the true pose once revealed. That is the information the plan uses,
        so the penalty inherits the plan's uncertainty rather than adding a new failure mode.

        Returns:
            The flattened XY centres, shape ``(2 * MAX_BARRIER_POLE,)``.
        """
        values = np.full(2 * MAX_BARRIER_POLE, BARRIER_OFF, dtype=float)
        obstacles = np.asarray(self._obstacles, dtype=float).reshape(-1, 3)
        count = min(len(obstacles), MAX_BARRIER_POLE)
        if count:
            values[: 2 * count] = obstacles[:count, :2].ravel()
        return values

    def _pass_frames(
        self, obs: dict[str, NDArray[np.floating]]
    ) -> tuple[
        NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
    ]:
        """Return the gate frames of the remaining passes, in passing order.

        The tunnel is the gate opening carried along the path, so it is expressed in the gate frame
        rather than in the path's Frenet frame: a Frenet normal is undefined on a straight segment
        and rotated relative to the opening elsewhere, while the gate's own axes are the edges of
        the hole the drone has to fly through.

        Args:
            obs: The current observation.

        Returns:
            Centres, normals, lateral axes and vertical axes, each ``(n_passes, 3)``.
        """
        centres = np.asarray(self._gate_centers, dtype=float).reshape(-1, 3)
        empty = np.zeros((0, 3))
        if not len(centres):
            return centres, empty, empty, empty
        frames = GateFrames.from_quaternions(np.asarray(obs["gates_quat"], dtype=float))
        ids = np.asarray(getattr(self._path_info, "gate_ids", np.zeros(0, dtype=int)), dtype=int)
        if len(ids) == len(centres):
            index = np.clip(ids, 0, len(frames.normals) - 1)
        else:
            # A path handed in by the scenario controller carries no gate ids: resolve each centre
            # against the observed poses, the same fallback the planner uses.
            observed = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
            distance = np.linalg.norm(centres[:, None, :] - observed[None, :, :], axis=-1)
            index = np.argmin(distance, axis=1)
        return centres, frames.normals[index], frames.y_axes[index], frames.z_axes[index]

    def _tunnel_params(
        self,
        theta: float,
        position: NDArray[np.floating],
        frames: tuple[
            NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
        ]
        | None,
    ) -> NDArray[np.floating]:
        """Return the tunnel slots of one stage, shape ``(MAX_TUNNEL * 10, )``.

        The half width is a funnel of the *progress along the path*: it is
        ``tunnel_gate_half_width`` at the arc length of a gate passage, widens smoothly to
        ``tunnel_nominal_half_width`` at the edge of ``tunnel_window``, and is inactive beyond it.
        Keying it on progress rather than on the distance to the gate plane is what keeps a gate the
        drone merely flies *past* from being constrained: the later gates of this track sit within a
        metre of each other in space, so a plane-distance trigger punishes the drone for passing a
        gate it is not crossing yet, which measured 0/20 before this was fixed.

        The slot is centred on the *path point* of the stage, not on the gate centre, so the
        corridor is a tube that follows the path. Centring it on the gate tests only the in-plane
        offset and demands a straight approach: measured, a 0.10 m corridor over a 1.2 m window was
        0/20 as a slab at the gate and is a flyable tube around the path.

        A stage can sit inside the window of two passages at once; the two slots take the narrower
        pair.

        Args:
            theta: Arc length of the stage on the active path [m].
            position: Reference position of the stage on the path, shape ``(3,)``, which is the
                centre of the tube.
            frames: The pass frames from [MPCCController._pass_frames][].

        Returns:
            The slot parameters ``[axis_a, axis_b, centre, half width]`` per slot; the two axes
            are the gate's in-plane axes, not its normal.
        """
        cfg = self._mpcc
        values = np.zeros(MAX_TUNNEL * 10)
        values[9::10] = TUNNEL_OFF
        if cfg.w_tunnel <= 0.0 or frames is None:
            return values
        centres, _, lateral_axes, vertical_axes = frames
        if not len(centres) or not len(self._gate_arc):
            return values
        gaps = np.abs(float(theta) - np.asarray(self._gate_arc, dtype=float)[: len(centres)])
        ratio = np.clip(gaps / max(cfg.tunnel_window, 1e-6), 0.0, 1.0)
        blend = ratio * ratio * (3.0 - 2.0 * ratio)
        half_width = cfg.tunnel_gate_half_width + blend * (
            cfg.tunnel_nominal_half_width - cfg.tunnel_gate_half_width
        )
        # The tolerance keeps a stage exactly on the window edge at the nominal width instead of
        # flipping to inactive on a rounding error: ``1.0 + 0.6 - 1.0`` is not ``0.6``.
        edge = cfg.tunnel_window * (1.0 + 1e-9)
        half_width = np.where(gaps > edge, TUNNEL_OFF, half_width)
        active = int(np.sum(half_width < TUNNEL_OFF))
        if active > MAX_TUNNEL and not self._tunnel_warned:
            logger.warning("%d passages in one stage, only %d tunnel slots", active, MAX_TUNNEL)
            self._tunnel_warned = True
        for slot, gate in enumerate(np.argsort(half_width)[:MAX_TUNNEL]):
            if half_width[gate] >= TUNNEL_OFF:
                continue
            offset = 10 * slot
            values[offset : offset + 3] = lateral_axes[gate]
            values[offset + 3 : offset + 6] = vertical_axes[gate]
            values[offset + 6 : offset + 9] = np.asarray(position, dtype=float)
            values[offset + 9] = float(half_width[gate])
        return values

    def _current_slack(
        self,
        position: NDArray[np.floating],
        theta: float,
        frames: tuple[
            NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
        ]
        | None,
    ) -> float:
        """Return the worst tunnel slack at the drone, in metres, positive inside the opening.

        Args:
            position: The drone's position, shape ``(3,)``.
            theta: The drone's progress on the active path [m].
            frames: The pass frames from [MPCCController._pass_frames][], or ``None`` when the
                tunnel is off.

        Returns:
            The smallest half-space slack of the active tunnel slots, or ``TUNNEL_OFF`` when no gate
            plane is inside ``tunnel_window``.
        """
        assert self._path is not None
        centre = self._path.pos(np.array([float(theta)]))[0]
        tunnel = self._tunnel_params(float(theta), centre, frames)
        slack = []
        for slot in range(MAX_TUNNEL):
            offset = 10 * slot
            half_width = tunnel[offset + 9]
            if half_width >= TUNNEL_OFF:
                continue
            delta = np.asarray(position, dtype=float) - tunnel[offset + 6 : offset + 9]
            along_normal = float(delta @ tunnel[offset : offset + 3])
            along_lateral = float(delta @ tunnel[offset + 3 : offset + 6])
            slack += [
                half_width - along_normal,
                half_width + along_normal,
                half_width - along_lateral,
                half_width + along_lateral,
            ]
        return float(min(slack)) if slack else float(TUNNEL_OFF)

    def _reference(
        self, thetas: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate the per-stage reference for a set of path progress values.

        Args:
            thetas: Arc length parameters, shape ``(n, )``.

        Returns:
            The reference positions ``(n, 3)``, unit tangents ``(n, 3)`` and contouring weights
            ``(n, )``, all as used by the MPCC stage parameters.
        """
        assert self._path is not None
        # The gate Gaussian and the tunnel are alternatives, not companions: with the tunnel on, the
        # gate is a constraint on the state and the contour cost no longer has to hold the drone on
        # the path there. Keeping both measured one finish worse per twenty seeds.
        gate_weight = 0.0 if self._mpcc.w_tunnel > 0.0 else GATE_PROXIMITY_WEIGHT
        return track_reference(self._path, thetas, self._gate_centers, self._obstacles, gate_weight)

    def _record_plan(self, reason: str, obs: dict[str, NDArray[np.floating]]):
        """Append the plan that is now active to the plan log.

        The log is what makes a run analysable after the fact: it holds every path the controller
        committed to, when it did so, where the drone was, and what triggered it.

        Args:
            reason: Why the path was planned, e.g. ``"episode start"`` or the objects that moved.
            obs: The observation the plan was made from.
        """
        assert self._path is not None
        self.plan_log.append(
            {
                "tick": self._tick,
                "t": self._tick / self._freq,
                "reason": reason,
                "drone_pos": np.asarray(obs["pos"], dtype=float).copy(),
                "theta": float(self._theta),
                "length": float(self._path.length),
                "n_waypoints": int(len(self._path_info.waypoints)),
                # Wall clock the planner took for this path. It is the cost the control loop pays
                # inside one tick, so it is worth recording next to the path it produced.
                "ms": float(self._last_plan_ms),
                "points": np.asarray(self._draw_points, dtype=float).copy(),
            }
        )

    def _gate_gaps(self, obs: dict[str, NDArray[np.floating]]) -> tuple[float, float]:
        """Return the progress gap and the plane gap to the nearest remaining pass.

        The tunnel narrows where the *path* crosses a gate, at ``_gate_arc``. If the drone crosses
        the plane at a different progress - it cuts the corner, or the schedule is ahead of it - the
        narrowest section of the funnel is never where the drone is, and the constraint is inert
        exactly when it should bind. These two numbers say which of those happened.

        Args:
            obs: The current observation.

        Returns:
            ``min |theta - gate_arc|`` and ``min |(p - centre) . normal|`` over the remaining
            passes, both in metres.
        """
        centres = np.asarray(self._gate_centers, dtype=float).reshape(-1, 3)
        arcs = np.asarray(self._gate_arc, dtype=float).reshape(-1)
        if not len(centres) or not len(arcs) or self._path is None:
            return float("nan"), float("nan")
        position = np.asarray(obs["pos"], dtype=float)
        frames = GateFrames.from_quaternions(np.asarray(obs["gates_quat"], dtype=float))
        ids = np.asarray(getattr(self._path_info, "gate_ids", np.zeros(0, dtype=int)), dtype=int)
        if len(ids) == len(centres):
            normals = frames.normals[np.clip(ids, 0, len(frames.normals) - 1)]
        else:
            observed = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
            distance = np.linalg.norm(centres[:, None, :] - observed[None, :, :], axis=-1)
            normals = frames.normals[np.argmin(distance, axis=1)]
        return float(np.min(np.abs(self._theta - arcs[: len(centres)]))), float(
            np.min(np.abs(np.einsum("pc,pc->p", position[None, :] - centres, normals)))
        )

    def diagnostics(self) -> dict[str, float]:
        """Return the controller's per-step state for logging.

        Returns:
            One entry per logged quantity: path progress ``theta``, the reference speed the
            controller asked for, the solve time in milliseconds, the solver status, the online mass
            estimate as a ratio of the nominal mass, the number of re-plans so far, the worst tunnel
            slack at the drone (negative means outside the opening), and the applied command.
        """
        return {
            "theta": float(self._theta),
            "v_ref": float(self._last_v_ref),
            "solver_ms": float(self.solve_time) * 1e3,
            "solver_status": float(self.status),
            "mass_ratio": float(self._mass_ratio),
            "replans": float(self.replans),
            "tunnel_slack": float(self.min_slack),
            "gate_gap": float(self.gate_gap),
            "plane_gap": float(self.plane_gap),
            "thrust": float(self._last_f_cmd),
            "roll_cmd": float(self._last_rpy_cmd[0]),
            "pitch_cmd": float(self._last_rpy_cmd[1]),
            "yaw_cmd": float(self._last_rpy_cmd[2]),
        }

    def _seed_visited_flags(self, obs: dict[str, NDArray[np.floating]]):
        """Initialize the reveal latch from the current observation.

        Args:
            obs: The current observation.
        """
        self._last_gate_flags = np.asarray(
            obs.get("gates_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        self._last_obstacle_flags = np.asarray(
            obs.get("obstacles_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

    def _track_info_changed(self, obs: dict[str, NDArray[np.floating]]) -> list[str]:
        """Describe how the observed gate or obstacle poses changed since the previous step.

        The environment reports nominal poses until the drone comes within the sensor range and the
        true pose afterwards, so a change of the observed pose is exactly the moment at which more
        accurate track information becomes available. Comparing the poses rather than only the
        `visited` flags avoids replanning when a flag rises but the reported pose is the same, which
        is the case on tracks without pose randomisation, and it also reacts to a pose that changes
        more than once.

        Args:
            obs: The current observation.

        Returns:
            One human readable entry per changed object, empty if nothing changed. A non-empty list
            is truthy, so the caller can still use it as a condition.
        """
        gates = np.asarray(obs["gates_pos"], dtype=float)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=float)
        gates_visited = np.asarray(
            obs.get("gates_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        obstacles_visited = np.asarray(
            obs.get("obstacles_visited", np.zeros(0, dtype=bool)), dtype=bool
        ).copy()
        previous_gates, previous_obstacles = self._last_gates_pos, self._last_obstacles_pos
        self._last_gates_pos = gates.copy()
        self._last_obstacles_pos = obstacles.copy()
        self._last_gate_flags, self._last_obstacle_flags = gates_visited, obstacles_visited
        if gates.shape != previous_gates.shape or obstacles.shape != previous_obstacles.shape:
            return []
        changes = []
        for gate_id, shift in enumerate(np.linalg.norm(gates - previous_gates, axis=1)):
            if shift > 1e-6:
                changes.append(f"gate {gate_id} moved {shift * 100:.1f} cm")
        for pole_id, shift in enumerate(np.linalg.norm(obstacles - previous_obstacles, axis=1)):
            if shift > 1e-6:
                changes.append(f"pole {pole_id} moved {shift * 100:.1f} cm")
        return changes

    def _replan_due(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return whether the pending track information should be turned into a new path now.

        Three things hold a re-plan back, and a hard cap overrides all of them:

        * a quiet period, so a burst of reveals near the same place produces one path, not three;
        * being inside a gate passage, where re-planning puts the first block behind the drone and
          folds the path back on itself — that cusp is what the smoothing then has to eat;
        * nothing pending at all.

        Args:
            obs: The current observation.

        Returns:
            ``True`` if the pending changes should be planned for now.
        """
        if not self._pending:
            return False
        waited = self._tick - self._pending_since
        if waited >= self._mpcc.replan_max_defer_ticks:
            return True
        if self._tick - self._pending_tick < self._mpcc.replan_coalesce_ticks:
            return False
        if self._mpcc.replan_defer_in_passage and self._in_passage(obs):
            return False
        return True

    def _in_passage(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return whether the drone is close enough to the next gate to be flying its passage.

        Args:
            obs: The current observation.

        Returns:
            ``True`` while the drone is within twice the straight passage length of the next gate.
        """
        if len(self._gate_centers) == 0:
            return False
        here = np.asarray(obs["pos"], dtype=float)[:2]
        next_gate = np.asarray(self._gate_centers, dtype=float)[0, :2]
        radius = 2.0 * float(self._planner.config.gate_straight_length)
        return bool(np.linalg.norm(here - next_gate) < radius)

    def _replan(self, obs: dict[str, NDArray[np.floating]], reason: str = "sensor update"):
        """Re-plan the path from where the drone is, through the passes that are still to come.

        Only the path reference changes: the optimal control problem and its solver are untouched,
        because the path enters the cost as per-stage parameters. The speed limit table and the
        progress bound are rebuilt for the new path.

        The new path starts on the *old* path, at the projection of the pose the drone will have
        when the new path takes effect, and it leaves that point along the old path's tangent. Two
        things follow from anchoring on the path rather than on the measured pose: the new path
        begins on the trajectory the drone is already flying, so the reference at progress zero is a
        point the drone is heading for and not a point it has to reach sideways; and the tangent is
        the direction it is already moving, so the first metres of the plan cannot ask it to turn
        around. Planning from the take-off point, or from the measured pose toward the next gate,
        did both of those wrong, and the folds that the smoothing then had to eat came out of it.

        Progress restarts at zero because the new path begins at the drone.

        The anchor is the drone's pose at the *end* of this control interval, not its pose now: the
        command computed this tick still applies for one interval, so by the time the new path takes
        effect the drone has moved ``velocity * dt`` further on. Anchoring at the current position
        put the start of the path behind the drone, which turns the first segment into a catch-up
        and makes the spline curve sharply right where the curvature limit then throttles the drone.

        Args:
            obs: The current observation.
            reason: Why the path is re-planned, recorded in the plan log.
        """
        here = np.asarray(obs["pos"], dtype=float)
        velocity = np.asarray(obs["vel"], dtype=float)
        trimmed = self._remaining_obs(obs)
        if trimmed is None:
            return
        assert self._path is not None
        predicted = here + velocity / self._freq
        progress = self._path.closest(predicted, self._theta)
        seed = PathSeed(
            position=self._path.pos(progress),
            tangent=self._path.tangent(progress),
            length=self._planner.config.seed_length,
            velocity=velocity,
            acceleration=self._accel,
        )
        started = time.perf_counter()
        path, info = self._planner.plan(trimmed, seed.position, climb=False, seed=seed)
        self._last_plan_ms = (time.perf_counter() - started) * 1e3
        self._install_path(path, info)
        self._theta = 0.0
        self._projection_window = (-0.15, 1.0)
        self._reproject_wide = True
        self._slowdown_until = self._tick + int(round(self._mpcc.replan_slowdown_time * self._freq))
        self._hold_until = self._tick + int(round(self._mpcc.replan_hold_time * self._freq))
        self.replans += 1
        self._record_plan(f"re-plan: {reason}", obs)
        logger.info(
            "Re-planned from progress %.2f m: %.2f m, %d passes left, %.1f ms (replan %d)",
            progress,
            path.length,
            len(np.asarray(trimmed["gate_sequence"]).ravel()),
            self._last_plan_ms,
            self.replans,
        )

    @staticmethod
    def _remaining_obs(
        obs: dict[str, NDArray[np.floating]],
    ) -> dict[str, NDArray[np.floating]] | None:
        """Return the observation with the gate sequence cut down to the passes still to come.

        Args:
            obs: The current observation.

        Returns:
            A shallow copy whose ``gate_sequence`` and ``gate_sequence_direction`` start at the next
            pass, or ``None`` when every pass is already done and there is nothing left to plan.
        """
        passed = int(np.asarray(obs["n_gates_passed"]))
        sequence = np.asarray(obs["gate_sequence"], dtype=int).ravel()
        direction = np.asarray(obs["gate_sequence_direction"], dtype=float).ravel()
        if passed >= len(sequence):
            return None
        trimmed = dict(obs)
        trimmed["gate_sequence"] = sequence[passed:]
        trimmed["gate_sequence_direction"] = direction[passed:]
        return trimmed

    def _apply_progress_bound(self):
        """Push the path length into the per-stage upper bound of the theta state.

        Stage 0 is the fixed initial state and the terminal stage carries no box constraint, so only
        the intermediate stages are updated.
        """
        assert self._path is not None
        ubx = np.array(self._stage_ubx())
        ubx[-1] = self._path.length
        for k in range(1, self._mpcc.N):
            self._solver.set(k, "ubx", ubx)

    def _reset_mass_history(self):
        """Clear the rolling window used by the mass estimate."""
        window = self._mpcc.mass_adapt_window
        self._force_hist: deque[float] = deque(maxlen=window)
        self._speed_hist: deque[float] = deque(maxlen=window)
        self._dvz_hist: deque[float] = deque(maxlen=window)

    def _build_speed_limit(self):
        """Precompute the curvature-limited path speed along the arc length.

        The path speed is capped by the lateral acceleration the drone can produce while still
        tracking the path: ``v <= sqrt(a_lat_max / curvature)``. This throttles the reference
        exactly in the tight corners instead of everywhere, which is what keeps the gate passes
        accurate.
        """
        assert self._path is not None
        length = self._path.length
        self._speed_grid = np.linspace(0.0, length, max(32, int(length / 0.02)))
        curvature = self._path.curvature(self._speed_grid)
        window = max(1, self._mpcc.curvature_smoothing)
        if window > 1:
            kernel = np.ones(window) / window
            curvature = np.convolve(np.pad(curvature, window // 2, mode="edge"), kernel, "valid")
        limit = np.sqrt(self._mpcc.a_lat_max / np.maximum(curvature, 1e-3))
        self._speed_limit = np.minimum(limit, self._mpcc.v_max)

    def episode_callback(self):
        """Log the flight statistics of the finished episode."""
        logger.info("MPCC episode finished after %d steps, theta=%.2f m", self._tick, self._theta)

    def episode_reset(self):
        """Mark the controller for a fresh path plan at the next control step."""
        self._needs_plan = True

    def reset(self):
        """Mark the controller for a fresh path plan at the next control step."""
        self._needs_plan = True

    def render_callback(self, sim: Sim):
        """Draw the planned path and the current horizon reference.

        Args:
            sim: The simulator instance used for rendering.
        """
        from crazyflow.sim.visualize import draw_line, draw_points

        if len(self._draw_points):
            draw_line(sim, self._draw_points, rgba=(0.0, 1.0, 0.0, 0.6))
        if len(self._horizon_points):
            draw_points(sim, self._horizon_points, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

    # endregion

    # region helpers

    def _stage_ubx(self) -> list[float]:
        """Return the compact upper state bounds of a regular stage (path length excluded)."""
        cfg = self._mpcc
        _, thrust_max = self._thrust_limits
        return [
            *cfg.pos_high,
            thrust_max,
            thrust_max,
            cfg.attitude_limit,
            cfg.attitude_limit,
            cfg.attitude_limit,
            1.0,
        ]

    def _state_vector(self, obs: dict[str, NDArray[np.floating]], theta: float) -> NDArray:
        """Assemble the full OCP state from an observation.

        Args:
            obs: The current observation.
            theta: Current path progress in meters.

        Returns:
            The state vector of length ``NX = 15``.
        """
        euler = R.from_quat(np.asarray(obs["quat"], dtype=float)).as_euler("xyz")
        return np.concatenate(
            [
                np.asarray(obs["pos"], dtype=float),
                np.asarray(obs["vel"], dtype=float),
                euler,
                [self._last_f_col, self._last_f_cmd],
                self._last_rpy_cmd,
                [theta],
            ]
        )

    def _gate_seen(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return whether the gate of the next pass has been observed at its true pose.

        The environment reports the nominal pose of an object until the drone comes within the
        sensor range of it, and the true one afterwards, and it says which by the ``visited`` flags.
        A track without pose randomisation never changes a pose and its flags stay false, so there
        "seen" is the right answer: the nominal pose is the true one.

        Args:
            obs: The current observation.

        Returns:
            ``True`` when the next gate's pose is trustworthy, or when there is no next gate.
        """
        gate_ids = np.asarray(self._path_info.gate_ids).ravel()
        if len(gate_ids) == 0:
            return True
        flags = np.asarray(obs.get("gates_visited", np.zeros(0)), dtype=bool).ravel()
        gate = int(gate_ids[0])
        if len(flags) <= gate:
            return True
        return bool(flags[gate])

    def _approach_capped(self, theta: float) -> bool:
        """Return whether the approach to the next gate has to be flown at ``reveal_speed``.

        Two cases ask for it. A gate that has not been seen yet is only nominal, and the correction
        the reveal will demand has to be flown before the drone gets there. A gate that *was* just
        corrected keeps the cap for ``reveal_hold_time``, because the reveal is when the correction
        starts, not when it is finished: measured on level 2, the reveal lands at 0.7 m and the
        0.46 s of lateral travel needed is the whole distance that is left.

        Args:
            theta: Path progress in meters.

        Returns:
            ``True`` if the reference speed should be capped.
        """
        if not self._randomized_track or len(self._gate_arc) == 0:
            return False
        gap = float(self._gate_arc[0]) - theta
        if not -0.2 <= gap < self._mpcc.reveal_window:
            return False
        if not self._next_gate_seen:
            return True
        hold = int(round(self._mpcc.reveal_hold_time * self._freq))
        return self._tick - self._gate_corrected_tick <= hold

    def _reference_speed(self, theta: float, speed_actual: float) -> float:
        """Return the reference path speed at a given path progress.

        This is the closed-form minimizer of the progress stage cost ``-mu * v + w * qc * v^2``,
        clipped to the admissible range and to the current drone speed plus a lead. Clipping to the
        measured speed is what makes the per-stage reference realizable: the near stages of the
        horizon are only ever asked to move slightly faster than the drone already does.

        Args:
            theta: Path progress in meters.
            speed_actual: Current drone speed in m/s.

        Returns:
            The reference path speed in m/s.
        """
        cfg = self._mpcc
        assert self._path is not None
        # The speed bump is its own knob: it slows the drone where a corrected gate pose has to be
        # paid for, a different job from keeping the contour error small at the gate. Sharing one
        # weight with the contour cost made the throttle vanish as soon as the gate contour weight
        # was off, so no experiment could separate gate accuracy from approach speed. The pole
        # bump still contributes, exactly as it did when the two shared one array.
        position = self._path.pos(np.array([theta]))
        speed_weight = max(
            float(proximity_weight(position, self._gate_centers, cfg.gate_speed_weight)[0]),
            float(proximity_weight(position, self._obstacles, OBSTACLE_PROXIMITY_WEIGHT)[0]),
        )
        speed = cfg.mu_speed / (2.0 * (cfg.w_speed_gate * speed_weight + cfg.r_vtheta))
        speed = min(speed, speed_actual + cfg.v_lead)
        speed = min(speed, float(np.interp(theta, self._speed_grid, self._speed_limit)))
        if self._approach_capped(theta):
            speed = min(speed, cfg.reveal_speed)
        if self._tick < self._hold_until:
            # Brake hard right after a re-plan: this is the only place where the controller spends
            # time to buy geometry. The revealed pose can be 0.21 m from the nominal one, and the
            # drone needs about 0.46 s of lateral travel to make that up, against the 0.41 s the
            # 0.7 m sensor range gives it at race speed. Slowing to a crawl for a moment shortens
            # the distance still to fly, so the correction lands before the obstacle instead of
            # after it.
            speed *= cfg.replan_hold_factor
        elif self._tick < self._slowdown_until:
            speed *= cfg.replan_slowdown_factor
        return float(np.clip(speed, cfg.v_min, cfg.v_max))

    def _progress(self, pos: NDArray[np.floating]) -> float:
        """Project the current position onto the path with a monotone guard.

        Args:
            pos: Current drone position, shape ``(3,)``.

        Returns:
            The arc length parameter of the projection.
        """
        assert self._path is not None
        window = getattr(self, "_projection_window", (-0.15, 0.6))
        theta = self._path.closest(np.asarray(pos, dtype=float), self._theta, window=window)
        return float(np.clip(theta, max(0.0, self._theta - 0.05), self._path.length))

    def _command(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Clip an attitude command into the environment's action space.

        Args:
            action: Raw ``[roll, pitch, yaw, thrust]`` command.

        Returns:
            The clipped command as float32.
        """
        low = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, self._thrust_limits[0]])
        high = np.array([np.pi / 2, np.pi / 2, np.pi / 2, self._thrust_limits[1]])
        return np.clip(action, low, high).astype(np.float32)

    # endregion

    def _output(self) -> NDArray[np.floating]:
        """Build the mass-compensated command and remember the thrust that was applied.

        Returns:
            The commanded ``[roll, pitch, yaw, collective_thrust]``.
        """
        thrust = self._last_f_cmd * self._mass_ratio
        command = self._command(np.concatenate((self._last_rpy_cmd, [thrust])))
        self._applied_thrust = float(command[3])
        return command

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude and thrust command.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary. Not used.

        Returns:
            The commanded ``[roll, pitch, yaw, collective_thrust]``.
        """
        if self._needs_plan:
            self._init_episode(obs)
        assert self._path is not None
        self._tick += 1

        pos = np.asarray(obs["pos"], dtype=float)
        # The acceleration a B-spline plan starts with. It has to be taken before the mass estimate
        # overwrites the previous velocity with the current one.
        velocity = np.asarray(obs["vel"], dtype=float)
        self._accel = (
            np.clip((velocity - self._prev_vel) * self._freq, -25.0, 25.0)
            if self._prev_vel is not None
            else np.zeros(3)
        )
        self._update_mass_estimate(obs)
        changes = self._track_info_changed(obs)
        if changes:
            # Collect rather than act immediately: a reveal often arrives in a burst, and a re-plan
            # inside a gate passage folds the path back towards the approach the drone has already
            # flown through. Both are handled by waiting for a quiet tick outside a passage.
            if not self._pending:
                self._pending_since = self._tick
            self._pending.extend(changes)
            self._pending_tick = self._tick
            if any("gate" in change for change in changes):
                self._gate_corrected_tick = self._tick
        if self._replan_due(obs):
            reason = ", ".join(dict.fromkeys(self._pending))
            self._pending = []
            self._replan(obs, reason)
        theta = self._progress(pos)
        self._next_gate_seen = self._gate_seen(obs)
        speed = self._reference_speed(theta, float(np.linalg.norm(np.asarray(obs["vel"], float))))
        self._last_v_ref = float(speed)
        thetas = np.clip(
            theta + np.arange(self._mpcc.N + 1) * self._mpcc.dt * speed, 0.0, self._path.length
        )
        positions, tangents, weights = self._reference(thetas)
        self._horizon_points = positions
        barrier = self._barrier_params()
        # The cost weights travel in the parameter vector, so a tuning sweep costs episodes and not
        # a solver rebuild per evaluation.
        weight_block = np.array([getattr(self._mpcc, name) for name in WEIGHT_FIELDS], dtype=float)
        # The gate frames cost a quaternion conversion per tick, so they are only resolved when the
        # tunnel is on. With it off every slot is inactive and the cost term is exactly zero.
        frames = self._pass_frames(obs) if self._mpcc.w_tunnel > 0.0 else None
        for k in range(self._mpcc.N + 1):
            tunnel = self._tunnel_params(thetas[k], positions[k], frames)
            params = np.concatenate(
                (positions[k], tangents[k], [weights[k]], barrier, tunnel, weight_block)
            )
            self._solver.set(k, "p", np.ascontiguousarray(params, dtype=np.float64))
        # Worst tunnel slack at the drone itself, positive inside the opening. That is what the
        # audit of a gate clip checks: a clip with a positive slack means the tunnel was not the
        # constraint that failed.
        self.min_slack = (
            TUNNEL_OFF if frames is None else self._current_slack(pos, self._theta, frames)
        )
        self.gate_gap, self.plane_gap = self._gate_gaps(obs)

        x0 = self._state_vector(obs, theta)
        self._solver.set(0, "lbx", np.ascontiguousarray(x0))
        self._solver.set(0, "ubx", np.ascontiguousarray(x0))

        start = time.perf_counter()
        status = self._solver.solve()
        self.solve_time = time.perf_counter() - start
        self.status = status
        if status != 0:
            self._failures += 1
            if self._failures == 1 or self._failures % 50 == 0:
                logger.warning(
                    "MPCC solver failed with status %d (%d times)", status, self._failures
                )
            if self._failures > 5:
                self._last_f_col = self._hover_thrust
                self._last_f_cmd = self._hover_thrust
                self._last_rpy_cmd = np.zeros(3)
            return self._output()
        self._failures = 0

        state = np.asarray(self._solver.get(1, "x"), dtype=float)
        self._last_f_col = float(state[9])
        self._last_f_cmd = float(state[10])
        self._last_rpy_cmd = np.asarray(state[11:14], dtype=float)
        self._update_watchdog(theta)
        return self._output()

    def _update_mass_estimate(self, obs: dict[str, NDArray[np.floating]]):
        """Estimate the mass ratio from the integrated vertical force balance.

        The OCP always models the nominal drone, but levels 1-3 randomize the mass by up to about
        12%. Over each control interval the vertical momentum balance of the drone reads

        ``m * dv_z = T * cos(roll) * cos(pitch) * dt - m * g * dt``,

        so integrating it over a short window yields a direct estimate of ``m``. Integrating rather
        than evaluating a single step removes the bias that the thrust and attitude transients of
        the onboard controller would otherwise introduce. The estimate is only taken at low speed
        and moderate thrust, where aerodynamic drag and the vertical component of the horizontal
        acceleration are negligible, and it is filtered and clipped to a plausible range.

        The commanded thrust is rescaled by the estimated ratio at the very end of the control step,
        so the drone accelerates like the nominal model the OCP planned with. The OCP state itself
        keeps using nominal-model units.

        Args:
            obs: The current observation.
        """
        cfg = self._mpcc
        velocity = np.asarray(obs["vel"], dtype=float)
        if not cfg.mass_adapt:
            self._prev_vel = velocity
            return
        euler = R.from_quat(np.asarray(obs["quat"], dtype=float)).as_euler("xyz")
        up = float(np.cos(euler[0]) * np.cos(euler[1]))
        if self._prev_vel is not None:
            self._force_hist.append(self._applied_thrust * up)
            self._speed_hist.append(float(np.linalg.norm(velocity)))
            self._dvz_hist.append(float(velocity[2] - self._prev_vel[2]))
        self._prev_vel = velocity
        window = cfg.mass_adapt_window
        if len(self._force_hist) < window:
            return
        impulsion = cfg.dt * float(sum(self._force_hist))
        gravity = float(-np.asarray(self._params["gravity_vec"])[2])
        denominator = float(sum(self._dvz_hist)) + gravity * window * cfg.dt
        if (
            float(np.mean(self._force_hist)) < cfg.mass_adapt_min_thrust
            or float(np.mean(self._speed_hist)) > cfg.mass_adapt_max_speed
            or denominator < cfg.mass_adapt_min_balance
        ):
            return
        ratio = impulsion / (self._nominal_mass * denominator)
        low, high = cfg.mass_adapt_range
        if abs(ratio - 1.0) < cfg.mass_adapt_deadband:
            ratio = 1.0
        ratio = float(np.clip(ratio, low, high))
        self._mass_ratio += cfg.mass_adapt_gain * (ratio - self._mass_ratio)
        self._mass_ratio = float(np.clip(self._mass_ratio, low, high))

    def _update_watchdog(self, theta: float):
        """Warn once if the drone stops making progress along the path.

        Args:
            theta: Current path progress in meters.
        """
        progress = theta - self._theta
        self._theta = theta
        if progress < 0.002:
            self._stall_time += self._mpcc.dt
        else:
            self._stall_time = 0.0
        if self._stall_time > 3.0 and not self._stall_reported:
            self._stall_reported = True
            logger.warning("Drone is not progressing along the path (theta=%.2f m)", theta)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return whether the controller is finished.

        The environment terminates the episode once the last gate in the sequence is passed, so the
        controller never has to finish the episode itself.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            Always ``False``.
        """
        return False
