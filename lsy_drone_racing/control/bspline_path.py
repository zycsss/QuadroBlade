"""Cubic B-spline trajectory for the race path.

The planner used to produce a geometric path only: a natural cubic spline through waypoints, arc
length parameterized, with everything about the drone's motion left to the MPCC. What that costs is
in the 40-seed screen. The spline is free to contain curvature no speed can fly - the level 2 track
produced a 0.19 m radius hairpin around a pole, which needs 12 m/s^2 at 1.5 m/s, so the controller
cut the corner and flew into the pole - and it starts at a position with no relation to the velocity
or acceleration the drone actually has.

This module represents the plan as a **uniform cubic B-spline in time**, with the relations that
make the motion controllable at plan time (after Zhou et al., Fast-Planner, RA-L 2019, Sec. IV):

    p(u) = sum_i B_i(u) Q_i
    v(u) = sum_i B_i(u) (Q_{i+1} - Q_i) / dt
    a(u) = sum_i B_i(u) (Q_{i+2} - 2 Q_{i+1} + Q_i) / dt^2

so velocity and acceleration are B-splines of the same control points, and the convex hull property
means bounding the *control points* of the derivatives bounds the whole continuous curve. The first
three control points are the drone's state:

    Q_0 = p0
    Q_1 = p0 + v0 dt / 3
    Q_2 = p0 + 2 v0 dt / 3 + a0 dt^2 / 6

which is the exact initial state of a clamped cubic with knot spacing dt. The path is handed to the
controller through the same duck type as [ArcPath][lsy_drone_racing.control.path_planning.ArcPath],
so the MPCC does not have to know that the reference changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Degree of the B-spline. Cubic is the smallest degree with continuous acceleration.
DEGREE = 3
#: Number of samples used to tabulate arc length against the spline parameter.
ARC_SAMPLES = 2000


def uniform_knots(n_control: int, time_step: float = 1.0) -> NDArray[np.floating]:
    """Return the clamped uniform knot vector of a spline with ``n_control`` control points.

    Args:
        n_control: Number of control points, at least ``DEGREE + 1``.
        time_step: Spacing [s] of the interior knots.

    Returns:
        The knot vector, shape ``(n_control + DEGREE + 1, )``.
    """
    interior = max(n_control - DEGREE, 1)
    span = float(interior) * float(time_step)
    return np.concatenate(
        (
            np.zeros(DEGREE + 1),
            np.arange(1, interior, dtype=float) * time_step,
            np.full(DEGREE + 1, span),
        )
    )


def basis_matrix(
    knots: NDArray[np.floating], n_control: int, params: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Evaluate every basis function at ``params``, shape ``(len(params), n_control)``."""
    return np.asarray(BSpline(knots, np.eye(n_control), DEGREE)(params), dtype=float)


@dataclass
class BsplinePath:
    """A clamped cubic B-spline in time, exposed as an arc length parameterized path.

    Attributes:
        control: Control points, shape ``(n_control, 3)``.
        time_step: Knot spacing [s] of the interior knots.
    """

    control: NDArray[np.floating]
    time_step: float

    def __post_init__(self):
        """Build the spline and the arc length table."""
        control = np.asarray(self.control, dtype=float).reshape(-1, 3)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "time_step", float(self.time_step))
        self._knots = uniform_knots(len(control), self.time_step)
        self._spline = BSpline(self._knots, control, DEGREE, extrapolate=False)
        self._velocity = self._spline.derivative(1)
        self._start = float(self._knots[DEGREE])
        self._end = float(self._knots[-DEGREE - 1])
        # Arc length against the parameter, so the controller's theta stays a length in meters.
        self._params = np.linspace(self._start, self._end, ARC_SAMPLES)
        points = np.asarray(self._spline(self._params), dtype=float)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._arc = np.concatenate(([0.0], np.cumsum(steps)))
        self.length = float(self._arc[-1])
        self.points = points
        self._start_point = points[0]
        self._end_point = points[-1]
        self._start_tangent = self._unit(self._velocity(self._start))
        self._end_tangent = self._unit(self._velocity(self._end))

    @staticmethod
    def _unit(vector: NDArray[np.floating]) -> NDArray[np.floating]:
        """Normalize a vector, with a safe fallback."""
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _to_param(self, tau: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map arc length to the spline parameter."""
        return np.interp(tau, self._arc, self._params)

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the path position at an arc length.

        Outside the path the evaluation continues linearly along the terminal tangents, as the
        controller expects.
        """
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._spline(self._to_param(clipped)), dtype=float).reshape(-1, 3)
        high = flat > self.length
        if np.any(high):
            out[high] = self._end_point + np.outer(flat[high] - self.length, self._end_tangent)
        low = flat < 0.0
        if np.any(low):
            out[low] = self._start_point + np.outer(flat[low], self._start_tangent)
        return out.reshape(*tau_arr.shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent at an arc length."""
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._velocity(self._to_param(clipped)), dtype=float).reshape(-1, 3)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, np.maximum(norm, 1e-9))
        high = flat > self.length
        if np.any(high):
            out[high] = self._end_tangent
        low = flat < 0.0
        if np.any(low):
            out[low] = self._start_tangent
        return out.reshape(*tau_arr.shape, 3)

    def curvature(self, tau: NDArray[np.floating] | float, step: float = 0.05) -> NDArray:
        """Evaluate the curvature, over a finite arc length like `ArcPath` does."""
        tau_arr = np.asarray(tau, dtype=float)
        forward = self.tangent(np.clip(tau_arr + step, 0.0, self.length))
        backward = self.tangent(np.clip(tau_arr - step, 0.0, self.length))
        return np.linalg.norm(forward - backward, axis=-1) / (2.0 * step)

    def closest(
        self,
        point: NDArray[np.floating],
        tau_prev: float,
        window: tuple[float, float] = (-0.15, 0.6),
        step: float = 0.02,
    ) -> float:
        """Project a position onto the path, searching around the previous progress."""
        point = np.asarray(point, dtype=float).reshape(3)
        low = max(0.0, tau_prev + window[0])
        high = min(self.length, tau_prev + window[1])
        if high <= low:
            high = min(self.length, low + step)
        grid = np.arange(low, high + 1e-9, step)
        if len(grid) < 2:
            grid = np.array([low, high])
        distance = np.linalg.norm(self.pos(grid) - point, axis=1)
        best = int(np.argmin(distance))
        if 0 < best < len(grid) - 1:
            y0, y1, y2 = distance[best - 1], distance[best], distance[best + 1]
            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-12:
                offset = 0.5 * (y0 - y2) / denominator
                return float(np.clip(grid[best] + offset * step, low, high))
        return float(grid[best])

    def velocity_control(self) -> NDArray[np.floating]:
        """Control points of the velocity spline, shape ``(n_control - 1, 3)``, in m/s."""
        return np.diff(self.control, axis=0) / self.time_step

    def acceleration_control(self) -> NDArray[np.floating]:
        """Control points of the acceleration spline, shape ``(n_control - 2, 3)``, in m/s^2."""
        return np.diff(self.control, n=2, axis=0) / self.time_step**2


def fit_control_points(
    points: NDArray[np.floating], n_control: int, time_step: float
) -> NDArray[np.floating]:
    """Fit a clamped cubic B-spline through a dense polyline by least squares.

    Args:
        points: The polyline to follow, shape ``(n_points, 3)``.
        n_control: Number of control points to solve for.
        time_step: Knot spacing [s]; sets how the spline is parameterized.

    Returns:
        The control points, shape ``(n_control, 3)``.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    knots = uniform_knots(n_control, time_step)
    params = float(knots[DEGREE]) + arc / max(float(arc[-1]), 1e-9) * (
        float(knots[-DEGREE - 1]) - float(knots[DEGREE])
    )
    basis = basis_matrix(knots, n_control, params)
    control, *_ = np.linalg.lstsq(basis, pts, rcond=None)
    return np.asarray(control, dtype=float)


def pin_initial_state(
    control: NDArray[np.floating],
    position: NDArray[np.floating],
    velocity: NDArray[np.floating],
    acceleration: NDArray[np.floating],
    time_step: float,
) -> NDArray[np.floating]:
    """Rewrite the first three control points so the spline starts in the given state.

    For a clamped cubic with knot spacing ``dt`` the start of the curve is exactly

        p(0) = Q_0,  dp/dt(0) = 3 (Q_1 - Q_0) / dt,
        d2p/dt2(0) = 6 (Q_2 - 2 Q_1 + Q_0) / dt^2,

    so the three control points below reproduce the measured state and nothing else has to be
    constrained for the plan to be velocity- and acceleration-continuous at the anchor.

    Args:
        control: Control points, shape ``(n_control, 3)``.
        position: Drone position at the anchor, shape ``(3,)``.
        velocity: Drone velocity at the anchor, shape ``(3,)``, m/s.
        acceleration: Drone acceleration at the anchor, shape ``(3,)``, m/s^2.
        time_step: Knot spacing [s].

    Returns:
        The control points with the first three replaced.
    """
    out = np.array(control, dtype=float)
    p0 = np.asarray(position, dtype=float).reshape(3)
    v0 = np.asarray(velocity, dtype=float).reshape(3)
    a0 = np.asarray(acceleration, dtype=float).reshape(3)
    dt = float(time_step)
    out[0] = p0
    out[1] = p0 + v0 * dt / 3.0
    out[2] = p0 + 2.0 * v0 * dt / 3.0 + a0 * dt**2 / 6.0
    return out


def retime(
    control: NDArray[np.floating], time_step: float, speed_max: float, accel_max: float
) -> float:
    """Return the knot spacing at which the control points respect the speed and accel limits.

    Velocity and acceleration are finite differences of the control points (see the module
    docstring), so the limits become bounds on those differences. Stretching the time step - the
    simplest form of the time adjustment in Fast-Planner Sec. V - makes the same geometry feasible
    at a lower speed instead of leaving the reference inconsistent with what the drone can fly.

    Args:
        control: Control points, shape ``(n_control, 3)``.
        time_step: The knot spacing the control points were fitted with.
        speed_max: Speed limit [m/s].
        accel_max: Acceleration limit [m/s^2].

    Returns:
        The knot spacing that satisfies both limits, never below ``time_step``.
    """
    control = np.asarray(control, dtype=float).reshape(-1, 3)
    if len(control) < 4:
        return float(time_step)
    speed = np.linalg.norm(np.diff(control, axis=0), axis=1) / float(time_step)
    accel = np.linalg.norm(np.diff(control, n=2, axis=0), axis=1) / float(time_step) ** 2
    needed = max(
        float(speed.max()) / max(speed_max, 1e-6),
        float(np.sqrt(accel.max() / max(accel_max, 1e-6))),
        1.0,
    )
    return float(time_step) * needed


def relax_control_points(
    control: NDArray[np.floating],
    poles: NDArray[np.floating],
    pole_radius: float,
    locked: NDArray[np.bool_],
    fidelity: float = 2.0,
    iterations: int = 0,
    step: float = 0.5,
) -> NDArray[np.floating]:
    """Straighten the control polygon where it is too tight, keeping the passages and the clearance.

    This is the part of the B-spline formulation that changes the *geometry* rather than only its
    representation. The objective is the curvature energy of the control polygon,

        E = sum_i |Q_{i+2} - 2 Q_{i+1} + Q_i|^2 ,

    which is proportional to the squared acceleration of the spline (see the finite-difference
    relation in the module docstring), plus a fidelity term that holds the curve onto the polyline
    it was fitted to. Minimising E rounds the tight arcs that make the path unflyable - the 0.19 m
    hairpin around a pole in seed 8 - while the fidelity term keeps the passages where they are.

    Control points within ``pole_radius`` of a pole are projected back onto that sphere after every
    step, so straightening can never buy smoothness with a collision. Because the spline lies inside
    the convex hull of four consecutive control points, that is the same sufficient condition
    Fast-Planner uses for safety.

    Args:
        control: Control points, shape ``(n_control, 3)``.
        poles: Pole centres, shape ``(n_poles, 3)``.
        pole_radius: Distance [m] the control polygon has to keep from every pole.
        locked: Control points that must not move, shape ``(n_control, )``. The planner locks the
            pinned initial state and the points that define a gate passage.
        fidelity: Weight of the pull towards the fitted control points.
        iterations: Number of gradient steps.
        step: Gradient step scale, relative to the Lipschitz constant of the gradient.

    Returns:
        The relaxed control points, shape ``(n_control, 3)``.
    """
    control = np.asarray(control, dtype=float).reshape(-1, 3)
    locked = np.asarray(locked, dtype=bool).reshape(-1)
    reference = control.copy()
    free = ~locked
    free[:3] = False
    if iterations <= 0 or len(control) < 6 or not free.any():
        return control
    # Work in units of the control polygon's spacing, so the geometry relaxation does not depend on
    # the time parameterisation, and normalise the step by the Lipschitz constant of the gradient.
    spacing = float(np.mean(np.linalg.norm(np.diff(control, axis=0), axis=1)))
    scale = 1.0 / max(spacing, 1e-3) ** 2
    alpha = step / (16.0 * scale + 2.0 * fidelity)
    poles = np.asarray(poles, dtype=float).reshape(-1, 3)
    for _ in range(int(iterations)):
        second = control[2:] - 2.0 * control[1:-1] + control[:-2]
        gradient = np.zeros_like(control)
        gradient[:-2] += second
        gradient[1:-1] -= 2.0 * second
        gradient[2:] += second
        gradient *= 4.0 * scale
        gradient += 2.0 * fidelity * (control - reference)
        gradient[~free] = 0.0
        control = control - alpha * gradient
        for pole in poles:
            delta = control[:, :2] - pole[:2]
            distance = np.linalg.norm(delta, axis=1)
            inside = (distance < pole_radius) & free
            if inside.any():
                grow = pole_radius / np.maximum(distance[inside], 1e-9)
                control[inside, :2] = pole[:2] + delta[inside] * grow[:, None]
        control[~free] = reference[~free]
    return control
