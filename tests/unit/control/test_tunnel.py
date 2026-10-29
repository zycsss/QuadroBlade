"""Unit tests for the gate-passage tunnel: the barrier, the funnel, the frame and the schedule.

A wrong sign or a wrong frame makes the barrier push the drone *out* of the opening instead of in,
a wrong arc length puts the funnel somewhere the drone is not, and either way the run still looks
like a normal flight until it clips a gate. These tests pin the formula, the funnel, the frame, and
the arc lengths that place it.
"""

from __future__ import annotations

import casadi as cs
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.bspline_path import BsplinePath, fit_control_points
from lsy_drone_racing.control.my_controller import (
    MAX_TUNNEL,
    TUNNEL_ALPHA,
    TUNNEL_OFF,
    MPCCConfig,
    MPCCController,
    tunnel_barrier,
)
from lsy_drone_racing.control.path_planning import _pass_arcs


def barrier(slack: np.ndarray | list) -> float:
    """Evaluate the deployed barrier function numerically, through CasADi."""
    values = np.atleast_1d(np.asarray(slack, dtype=float))
    symbol = cs.SX.sym("h", len(values))
    function = cs.Function("barrier", [symbol], [tunnel_barrier(symbol)])
    return float(function(cs.DM(values)))


class Harness:
    """The controller's tunnel helpers, bound to a stub instead of the acados solver.

    Only the attributes the helpers actually read are provided: the config, the reveal warning
    latch, the remaining gate centres and the gate ids of the active path.
    """

    def __init__(self, config: MPCCConfig, gate_ids: tuple[int, ...] = (0,)):
        """Bind the helpers and set up a single gate at the origin, one metre up."""
        self._mpcc = config
        self._tunnel_warned = False
        self._gate_centers = np.array([[0.0, 0.0, 1.0]])
        # The passage sits at arc length 1 m, so a stage's theta selects the funnel width.
        self._gate_arc = np.array([1.0])
        self._path_info = type("Info", (), {"gate_ids": np.asarray(gate_ids, dtype=int)})()
        self._tunnel_params = MPCCController._tunnel_params.__get__(self)
        self._current_slack = MPCCController._current_slack.__get__(self)
        self._pass_frames = MPCCController._pass_frames.__get__(self)

    def frames(
        self, centre: np.ndarray, normal: np.ndarray, lateral: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a one-pass frame tuple with the given axes."""
        centre = np.asarray(centre, dtype=float).reshape(1, 3)
        normal = np.asarray(normal, dtype=float).reshape(1, 3)
        lateral = np.asarray(lateral, dtype=float).reshape(1, 3)
        self._gate_centers = centre
        return centre, normal, lateral, np.cross(normal, lateral)

    def tunnel(self, theta: float, frames: tuple[np.ndarray, ...]) -> np.ndarray:
        """Call the controller's per-stage tunnel helper at a decoded arc length."""
        return self._tunnel_params(float(theta), frames)

    def slack(self, position: np.ndarray, theta: float, frames: tuple[np.ndarray, ...]) -> float:
        """Call the controller's slack helper for a position at a given arc length."""
        return self._current_slack(np.asarray(position, dtype=float), float(theta), frames)


def active_config(**overrides: float) -> MPCCConfig:
    """Return a config with the tunnel switched on and a 0.4 m window."""
    base = {
        "w_tunnel": 1.0,
        "tunnel_gate_half_width": 0.09,
        "tunnel_nominal_half_width": 0.40,
        "tunnel_window": 0.6,
    }
    base.update(overrides)
    return MPCCConfig(**base)


IDENTITY_YAW = R.from_euler("z", 0.0)


@pytest.mark.unit
def test_barrier_is_inert_inside_and_steep_outside():
    """Zero well inside the tunnel, log 2 on the face, and alpha per unit violation outside."""
    np.testing.assert_allclose(barrier([1.0, 1.0]), 0.0, atol=1e-9)
    np.testing.assert_allclose(barrier([0.0]), np.log(2.0), rtol=1e-6)
    np.testing.assert_allclose(barrier([-0.1]), TUNNEL_ALPHA * 0.1, rtol=1e-3)
    np.testing.assert_allclose(barrier([-0.2]), TUNNEL_ALPHA * 0.2, rtol=1e-3)
    # Deeper inside is always cheaper than a violation, and getting closer to a face always costs
    # more, which is what makes it a barrier rather than a penalty.
    costs = [barrier([value]) for value in (0.2, 0.05, 0.0, -0.02, -0.1)]
    assert costs == sorted(costs)


@pytest.mark.unit
def test_barrier_does_not_overflow_deep_inside():
    """A large positive slack must not overflow the exponential."""
    assert barrier([1e3, 1e3]) == 0.0
    assert np.isfinite(barrier([5.0]))


@pytest.mark.unit
def test_inactive_slot_costs_nothing():
    """An unused slot has a zero frame and ``W = TUNNEL_OFF``, so every slack is huge."""
    harness = Harness(MPCCConfig(w_tunnel=0.0))
    values = harness.tunnel(1.0, harness.frames([0, 0, 1], [1, 0, 0], [0, 1, 0]))
    assert values.shape == (MAX_TUNNEL * 10,)
    assert np.all(values[9::10] == TUNNEL_OFF)
    assert np.all(values[:9] == 0.0)


@pytest.mark.unit
def test_funnel_narrows_to_the_opening_on_the_plane():
    """Half width is the gate width on the plane, nominal at the window edge, off beyond it."""
    config = active_config()
    harness = Harness(config)
    frames = harness.frames([0.0, 0.0, 1.0], [1, 0, 0], [0, 1, 0])
    for offset, expected in (
        (0.0, config.tunnel_gate_half_width),
        (config.tunnel_window, config.tunnel_nominal_half_width),
        (config.tunnel_window + 0.1, TUNNEL_OFF),
    ):
        theta = 1.0 + offset  # the passage sits at arc length 1 m
        np.testing.assert_allclose(harness.tunnel(theta, frames)[9], expected)


@pytest.mark.unit
def test_slack_is_measured_in_the_gate_frame():
    """Inside the opening the slack is positive, outside it negative, along the gate's own axes."""
    config = active_config()
    harness = Harness(config)
    yaw = np.pi / 2
    rotation = R.from_euler("z", yaw)
    normal = rotation.apply([1.0, 0.0, 0.0])
    lateral = rotation.apply([0.0, 1.0, 0.0])
    centre = np.array([0.0, 0.0, 1.0])
    frames = harness.frames(centre, normal, lateral)
    width = config.tunnel_gate_half_width
    theta = 1.0
    np.testing.assert_allclose(
        harness.slack(centre + lateral * (width - 0.01), theta, frames), 0.01
    )
    np.testing.assert_allclose(
        harness.slack(centre + lateral * (width + 0.01), theta, frames), -0.01
    )
    # An offset along the gate normal is along the corridor, not across it: it does not tighten the
    # box, so the slack is at least the tunnel width (slightly more, because the funnel opens as the
    # drone moves away from the plane).
    assert harness.slack(centre + normal * 0.02, theta, frames) >= width


@pytest.mark.unit
def test_frames_use_gate_ids_and_fall_back_to_the_nearest_pose():
    """With gate ids the pass frame is the gate's; without them it is resolved by position."""
    config = active_config()
    harness = Harness(config, gate_ids=(0,))
    quat = R.from_euler("z", 0.5).as_quat()[None, :]
    obs = {"gates_quat": quat, "gates_pos": np.array([[0.0, 0.0, 1.0]])}
    centres, normals, lateral_axes, _ = harness._pass_frames(obs)
    expected = R.from_quat(quat[0]).as_matrix()
    np.testing.assert_allclose(normals[0], expected[:, 0], atol=1e-9)
    np.testing.assert_allclose(lateral_axes[0], expected[:, 1], atol=1e-9)
    # A path without gate ids finds the same gate by position.
    fallback = Harness(config, gate_ids=())
    fallback._gate_centers = centres
    _, normals_fallback, lateral_fallback, _ = fallback._pass_frames(obs)
    np.testing.assert_allclose(normals_fallback[0], expected[:, 0], atol=1e-9)
    np.testing.assert_allclose(lateral_fallback[0], expected[:, 1], atol=1e-9)


@pytest.mark.unit
def test_pass_arcs_pick_the_crossing_inside_the_opening():
    """A plane crossed twice must be reported at the crossing that goes through the opening.

    This is the bug that made the tunnel inert: the arc length of the next pass was a nearest-point
    lookup, and on this track the nearest point to a gate can be metres from the passage.
    Here the path crosses ``x = 1`` twice, once through the opening and once a metre to the side.
    """
    polyline = np.array(
        [[0.0, 0.0, 1.0], [1.5, 0.0, 1.0], [1.5, 1.2, 1.0], [0.4, 1.2, 1.0], [0.4, 2.0, 1.0]]
    )
    dense = np.concatenate(
        [np.linspace(polyline[i], polyline[i + 1], 60) for i in range(len(polyline) - 1)]
    )
    path = BsplinePath(fit_control_points(dense, 40, 0.05), 0.05)
    centre = np.array([[1.0, 0.0, 1.0]])
    normal = np.array([[1.0, 0.0, 0.0]])
    arcs = _pass_arcs(path, centre, normal)
    crossing = path.pos(np.array([arcs[0]]))[0]
    np.testing.assert_allclose(crossing[0], 1.0, atol=0.02)
    # The first crossing is the one through the opening, at y ~ 0; the second is a metre out.
    np.testing.assert_allclose(crossing[1], 0.0, atol=0.15)


@pytest.mark.unit
def test_pass_arcs_are_monotone():
    """Passes are visited in order, so their crossing arc lengths must be non-decreasing."""
    polyline = np.array([[0.0, 0.0, 1.0], [1.0, 0.2, 1.0], [2.0, -0.2, 1.0], [3.0, 0.1, 1.0]])
    dense = np.concatenate(
        [np.linspace(polyline[i], polyline[i + 1], 60) for i in range(len(polyline) - 1)]
    )
    path = BsplinePath(fit_control_points(dense, 40, 0.05), 0.05)
    centres = np.array([[1.0, 0.2, 1.0], [2.0, -0.2, 1.0]])
    normals = np.tile([1.0, 0.0, 0.0], (2, 1))
    arcs = _pass_arcs(path, centres, normals)
    assert arcs[0] < arcs[1]
    assert arcs[0] >= 0.0


@pytest.mark.unit
def test_speed_weight_is_independent_of_the_contour_weight():
    """With the gate contour weight off, the speed law must still see a bump at the gate.

    This coupling is what made the earlier tunnel experiments meaningless: the throttle was
    multiplied by the same proximity weight the contour cost used, so turning the gate contour term
    off also disabled the slowdown.
    """
    from lsy_drone_racing.control.path_planning import proximity_weight

    centre = np.array([[0.0, 0.0, 1.0]])
    at_gate = proximity_weight(centre, centre, 0.4)
    np.testing.assert_allclose(at_gate[0], 0.4)
    assert proximity_weight(centre, None, 0.4)[0] == 0.0
    np.testing.assert_allclose(proximity_weight(centre, centre, 0.0)[0], 0.0)


@pytest.mark.unit
def test_tunnel_geometry_does_not_invalidate_the_solver():
    """Tunnel half width and window only reach the controller, so they are not OCP fields."""
    from lsy_drone_racing.control.my_controller import _OCP_FIELDS

    for name in ("tunnel_gate_half_width", "tunnel_nominal_half_width", "tunnel_window"):
        assert name not in _OCP_FIELDS
