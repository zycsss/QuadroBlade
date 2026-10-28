"""Unit tests for the cubic B-spline trajectory used as the planned path."""

import numpy as np
import pytest

from lsy_drone_racing.control.bspline_path import (
    BsplinePath,
    fit_control_points,
    pin_initial_state,
    retime,
    uniform_knots,
)


def straight(points: int = 200, length: float = 4.0) -> np.ndarray:
    """Return a dense polyline along +x at z = 1 m."""
    return np.stack(
        (np.linspace(0.0, length, points), np.zeros(points), np.full(points, 1.0)), axis=1
    )


@pytest.mark.unit
def test_knot_vector_is_clamped():
    """The knot vector repeats the end knots, so the spline starts and ends on control points."""
    knots = uniform_knots(10, 0.1)
    assert len(knots) == 10 + 3 + 1
    assert np.allclose(knots[:4], knots[0])
    assert np.allclose(knots[-4:], knots[-1])
    assert np.all(np.diff(knots) >= 0.0)


@pytest.mark.unit
def test_bspline_path_reproduces_a_straight_line():
    """A fitted straight line is a straight path of the right length, with no curvature."""
    control = fit_control_points(straight(), 30, 0.05)
    path = BsplinePath(control, 0.05)
    assert abs(path.length - 4.0) < 0.02
    np.testing.assert_allclose(path.pos(0.0), [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(path.tangent(2.0), [1.0, 0.0, 0.0], atol=1e-6)
    assert float(np.max(path.curvature(np.linspace(0.1, 3.9, 50)))) < 1e-3


@pytest.mark.unit
def test_bspline_is_arc_length_parameterized():
    """Equal steps in the parameter are equal steps in metres."""
    control = fit_control_points(straight(), 30, 0.05)
    path = BsplinePath(control, 0.05)
    steps = np.linalg.norm(np.diff(path.pos(np.linspace(0.0, path.length, 400)), axis=0), axis=1)
    assert abs(steps.mean() - path.length / 399) < 1e-6
    assert steps.std() / steps.mean() < 1e-2


@pytest.mark.unit
def test_initial_state_is_pinned():
    """The first three control points reproduce the measured position, velocity and acceleration."""
    control = fit_control_points(straight(), 30, 0.05)
    velocity = np.array([1.5, 0.0, 0.0])
    acceleration = np.array([0.0, 3.0, 0.0])
    pinned = pin_initial_state(control, control[0], velocity, acceleration, 0.05)
    path = BsplinePath(pinned, 0.05)
    np.testing.assert_allclose(path.pos(0.0), control[0], atol=1e-6)
    np.testing.assert_allclose(path.tangent(0.0), velocity / np.linalg.norm(velocity), atol=1e-3)
    # The derivative control points are what the pin sets: V_0 = v0 / 3 and A_0 = a0 / 6.
    np.testing.assert_allclose(path.velocity_control()[0], velocity / 3.0, atol=1e-9)
    np.testing.assert_allclose(path.acceleration_control()[0], acceleration / 6.0, atol=1e-9)


@pytest.mark.unit
def test_velocity_and_acceleration_control_points():
    """The derivative control points are the finite differences the module documents."""
    control = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.25, 0.1, 1.0], [0.45, 0.3, 1.0], [0.7, 0.6, 1.0]]
    )
    path = BsplinePath(control, 0.1)
    np.testing.assert_allclose(path.velocity_control(), np.diff(control, axis=0) / 0.1)
    np.testing.assert_allclose(path.acceleration_control(), np.diff(control, n=2, axis=0) / 0.01)


@pytest.mark.unit
def test_retime_stretches_to_meet_the_limits():
    """Stretching the knot spacing lowers the implied speed and acceleration of the polygon."""
    control = np.array(
        [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [0.5, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]]
    )
    stretched = retime(control, 0.1, speed_max=1.0, accel_max=2.0)
    assert stretched > 0.1
    path = BsplinePath(control, stretched)
    assert float(np.max(np.linalg.norm(path.velocity_control(), axis=1))) <= 1.0 + 1e-9
    assert float(np.max(np.linalg.norm(path.acceleration_control(), axis=1))) <= 2.0 + 1e-9


@pytest.mark.unit
def test_closest_projection():
    """The projection recovers the arc length of points on the path and never runs backwards."""
    control = fit_control_points(straight(), 30, 0.05)
    path = BsplinePath(control, 0.05)
    for tau in (0.5, 1.5, 3.0):
        assert abs(path.closest(path.pos(tau), tau) - tau) < 0.05
    projected = path.closest(path.pos(0.2), 3.0)
    assert 3.0 - 0.15 - 1e-6 <= projected <= 3.6
