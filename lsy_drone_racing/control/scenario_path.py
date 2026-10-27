"""Analytic reference paths for characterising the drone.

The race planner decides the route through a track, which makes it useless as a measuring
instrument: a "circle" routed through gates is smoothed by the planner and speed-shaped by the
gate proximity weight, so what it measures is the planner, not the drone.

These paths are exact instead. They satisfy the same interface the controller already uses
(`length`, `pos`, `tangent`, `curvature`, `closest`), so they can be installed with
`MPCCController._install_path` in place of a planned path and the whole control stack runs
unchanged. Curvature is exactly ``0`` on a straight and exactly ``1 / radius`` on an arc, which is
what makes the measured limits meaningful.

The manoeuvres cover the questions that decide the controller's parameters: how fast the drone
tracks a straight, how much it cuts a circle (the radial bias that eats the obstacle margin), what
lateral acceleration it can actually sustain, how it takes a curvature step, and how long it needs
to move sideways by a given distance (the reveal-correction question).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _flat(tau: NDArray[np.floating] | float) -> tuple[NDArray[np.floating], tuple[int, ...]]:
    """Flatten a parameter argument and keep its shape for the caller to restore."""
    array = np.asarray(tau, dtype=float)
    return np.atleast_1d(array).ravel(), array.shape


def _vector(direction: NDArray[np.floating], horizontal: bool = True) -> NDArray[np.floating]:
    """Return a unit direction, optionally flattened into the horizontal plane."""
    vector = np.asarray(direction, dtype=float).reshape(3).copy()
    if horizontal:
        vector[2] = 0.0
    return vector / np.linalg.norm(vector)


class ScenarioPath:
    """Base class of an arc-length parameterised analytic reference path.

    Subclasses implement `pos`, `tangent` and `curvature`; `closest` is shared and searches a
    window around the previous progress, mirroring
    [ArcPath.closest][lsy_drone_racing.control.path_planning.ArcPath.closest] so the controller's
    monotone progress guard behaves the same way as on a planned path.

    Every evaluator accepts a scalar or any array shape and returns ``tau.shape + (3,)`` for
    positions and tangents, and ``tau.shape`` for curvature.
    """

    #: Total arc length of the path in meters.
    length: float = 0.0

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the path position.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The position, shape ``tau.shape + (3,)``.
        """
        raise NotImplementedError

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        raise NotImplementedError

    def curvature(
        self, tau: NDArray[np.floating] | float, step: float = 0.05
    ) -> NDArray[np.floating]:
        """Evaluate the curvature.

        Args:
            tau: Arc length parameter, any shape.
            step: Unused; these paths know their curvature exactly. Kept for interface
                compatibility with [ArcPath.curvature][
                lsy_drone_racing.control.path_planning.ArcPath.curvature].

        Returns:
            The curvature in 1/m, shape ``tau.shape``.
        """
        raise NotImplementedError

    def closest(
        self,
        point: NDArray[np.floating],
        tau_prev: float,
        window: tuple[float, float] = (-0.15, 0.6),
        step: float = 0.02,
    ) -> float:
        """Project a position onto the path, searching around the previous progress.

        Args:
            point: Position in world coordinates, shape ``(3,)``.
            tau_prev: Previous arc length parameter, used to center the search window.
            window: Search range relative to ``tau_prev``, in meters.
            step: Sampling distance of the coarse search, in meters.

        Returns:
            The arc length parameter closest to ``point``.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        low = max(0.0, tau_prev + window[0])
        high = min(self.length, tau_prev + window[1])
        if high <= low:
            return float(np.clip(tau_prev, 0.0, self.length))
        grid = np.arange(low, high + step, step)
        if len(grid) < 2:
            return float(low)
        distance = np.linalg.norm(self.pos(grid) - point[None, :], axis=1)
        best = float(grid[int(np.argmin(distance))])
        # Refine around the coarse minimum so the projection is smooth in time.
        fine = np.linspace(max(low, best - step), min(high, best + step), 21)
        fine_distance = np.linalg.norm(self.pos(fine) - point[None, :], axis=1)
        return float(fine[int(np.argmin(fine_distance))])


class Line(ScenarioPath):
    """A straight line, curvature zero everywhere."""

    def __init__(self, start: NDArray[np.floating], direction: NDArray[np.floating], length: float):
        """Build a line.

        Args:
            start: Start point, shape ``(3,)``.
            direction: Direction of travel, any non-zero length.
            length: Length of the line in meters.
        """
        self.start = np.asarray(start, dtype=float).reshape(3)
        self.direction = _vector(direction)
        self.length = float(length)

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the position on the line.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The position, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        return (self.start + flat[:, None] * self.direction).reshape(*shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent, which is constant.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        return np.broadcast_to(self.direction, (len(flat), 3)).reshape(*shape, 3).copy()

    def curvature(
        self, tau: NDArray[np.floating] | float, step: float = 0.05
    ) -> NDArray[np.floating]:
        """Evaluate the curvature, which is zero.

        Args:
            tau: Arc length parameter, any shape.
            step: Unused.

        Returns:
            Zeros with the shape of ``tau``.
        """
        _, shape = _flat(tau)
        return np.zeros(shape)


class Circle(ScenarioPath):
    """A horizontal circle of exact curvature ``1 / radius``, flown at constant altitude.

    ``pos(0)`` is the point at ``start_angle`` on the circle, so the drone starts on the path.
    """

    def __init__(
        self,
        centre: NDArray[np.floating],
        radius: float,
        start_angle: float = 0.0,
        sweep: float = 2.0 * np.pi,
    ):
        """Build a circular arc.

        Args:
            centre: Circle centre, shape ``(3,)``; the altitude is taken from it.
            radius: Radius in meters.
            start_angle: Angle in radians of the start point, measured from ``+x``.
            sweep: Swept angle in radians; ``2 pi`` closes the loop.
        """
        self.centre = np.asarray(centre, dtype=float).reshape(3)
        self.radius = float(radius)
        self.start_angle = float(start_angle)
        self.sweep = float(sweep)
        self.length = abs(self.radius * self.sweep)

    def _angles(self, flat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the polar angle at a set of arc length parameters."""
        return self.start_angle + np.sign(self.sweep) * flat / self.radius

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the position on the arc.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The position, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        angle = self._angles(flat)
        local = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=-1)
        return (self.centre[None, :] + self.radius * local).reshape(*shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent on the arc.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        angle = self._angles(flat)
        sign = np.sign(self.sweep)
        local = np.stack((-sign * np.sin(angle), sign * np.cos(angle), np.zeros_like(angle)), -1)
        return local.reshape(*shape, 3)

    def curvature(
        self, tau: NDArray[np.floating] | float, step: float = 0.05
    ) -> NDArray[np.floating]:
        """Evaluate the curvature, which is exactly ``1 / radius``.

        Args:
            tau: Arc length parameter, any shape.
            step: Unused.

        Returns:
            The curvature, shape ``tau.shape``.
        """
        _, shape = _flat(tau)
        return np.full(shape, 1.0 / self.radius)


class Corner(ScenarioPath):
    """A straight, an arc of a given radius and sweep, then a straight: a sharp turn."""

    def __init__(
        self,
        start: NDArray[np.floating],
        direction: NDArray[np.floating],
        straight_in: float,
        radius: float,
        sweep: float,
        straight_out: float,
        turn_left: bool = True,
    ):
        """Build the turn.

        Args:
            start: Start point, shape ``(3,)``.
            direction: Direction of travel before the turn.
            straight_in: Length of the leading straight in meters.
            radius: Radius of the turn in meters.
            sweep: Turned angle in radians, positive.
            straight_out: Length of the trailing straight in meters.
            turn_left: Whether the turn is to the left of the direction of travel.
        """
        self.start = np.asarray(start, dtype=float).reshape(3)
        self.direction = _vector(direction)
        self.straight_in = float(straight_in)
        self.radius = float(radius)
        self.sweep = float(abs(sweep))
        self.straight_out = float(straight_out)
        self.turn = 1.0 if turn_left else -1.0
        self.length = self.straight_in + self.radius * self.sweep + self.straight_out

        self._arc_entry = self.start + self.straight_in * self.direction
        self._left = np.array((-self.direction[1], self.direction[0], 0.0))
        self._centre = self._arc_entry + self.turn * self.radius * self._left
        self._arc_length = self.radius * self.sweep
        self._radial = self._arc_entry - self._centre
        self._arc_exit, self._exit_direction = self._arc_state(self._arc_length)

    def _arc_state(
        self, arc_length: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return position and heading at a distance along the arc."""
        angle = self.turn * arc_length / self.radius
        cosine, sine = np.cos(angle)[..., None], np.sin(angle)[..., None]
        radial = self._radial * cosine + np.cross([0.0, 0.0, self.turn], self._radial) * sine
        heading = self.direction * cosine + np.cross([0.0, 0.0, self.turn], self.direction) * sine
        return self._centre + radial, heading

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the position along the turn.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The position, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        out = np.empty((len(flat), 3))
        before = flat <= self.straight_in
        after = flat >= self.straight_in + self._arc_length
        during = ~before & ~after
        out[before] = self.start + flat[before, None] * self.direction
        if np.any(during):
            out[during] = self._arc_state(flat[during] - self.straight_in)[0]
        if np.any(after):
            excess = (flat[after] - self.straight_in - self._arc_length)[:, None]
            out[after] = self._arc_exit + excess * self._exit_direction
        return out.reshape(*shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent along the turn.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        out = np.empty((len(flat), 3))
        before = flat <= self.straight_in
        after = flat >= self.straight_in + self._arc_length
        during = ~before & ~after
        out[before] = self.direction
        if np.any(during):
            out[during] = self._arc_state(flat[during] - self.straight_in)[1]
        out[after] = self._exit_direction
        return out.reshape(*shape, 3)

    def curvature(
        self, tau: NDArray[np.floating] | float, step: float = 0.05
    ) -> NDArray[np.floating]:
        """Evaluate the curvature: zero on the straights, ``1 / radius`` on the arc.

        Args:
            tau: Arc length parameter, any shape.
            step: Unused.

        Returns:
            The curvature, shape ``tau.shape``.
        """
        flat, shape = _flat(tau)
        inside = (flat > self.straight_in) & (flat < self.straight_in + self._arc_length)
        return np.where(inside, 1.0 / self.radius, 0.0).reshape(shape)


class ParallelShift(ScenarioPath):
    """A straight whose lateral offset steps by a fixed amount part-way along.

    This reproduces what a re-plan does when a newly revealed obstacle shifts the path sideways:
    the reference jumps, and the drone's task is to get back onto it. The convergence time of that
    jump is the number that decides whether a reveal can be corrected in the time available.
    """

    def __init__(
        self,
        start: NDArray[np.floating],
        direction: NDArray[np.floating],
        length: float,
        offset: float,
        at: float,
    ):
        """Build the shift.

        Args:
            start: Start point, shape ``(3,)``.
            direction: Direction of travel.
            length: Total length in meters.
            offset: Lateral step in meters, positive to the left of the direction of travel.
            at: Arc length at which the reference steps, in meters.
        """
        self.start = np.asarray(start, dtype=float).reshape(3)
        self.direction = _vector(direction)
        self.left = np.array((-self.direction[1], self.direction[0], 0.0))
        self.length = float(length)
        self.offset = float(offset)
        self.at = float(np.clip(at, 0.0, length))

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the position, which jumps sideways at ``at``.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The position, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        stepped = (flat >= self.at)[:, None] * self.offset
        return (self.start + flat[:, None] * self.direction + stepped * self.left).reshape(
            *shape, 3
        )

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent, which does not change across the step.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        flat, shape = _flat(tau)
        return np.broadcast_to(self.direction, (len(flat), 3)).reshape(*shape, 3).copy()

    def curvature(
        self, tau: NDArray[np.floating] | float, step: float = 0.05
    ) -> NDArray[np.floating]:
        """Evaluate the curvature, which is zero on both sides of the step.

        Args:
            tau: Arc length parameter, any shape.
            step: Unused.

        Returns:
            Zeros with the shape of ``tau``.
        """
        _, shape = _flat(tau)
        return np.zeros(shape)


#: Manoeuvre names understood by [build][lsy_drone_racing.control.scenario_path.build].
MANOEUVRES = ("line", "circle", "corner", "shift")


def parse_spec(spec: str) -> tuple[str, dict[str, float]]:
    """Split a scenario specification into its manoeuvre and its numeric parameters.

    Args:
        spec: A specification such as ``"circle radius=1.0 speed=1.8"``.

    Returns:
        The manoeuvre name and the parsed parameters.

    Raises:
        ValueError: If the manoeuvre is unknown or a parameter is not ``key=value``.
    """
    tokens = spec.replace(",", " ").split()
    if not tokens:
        raise ValueError("empty scenario specification")
    name = tokens[0].lower()
    if name not in MANOEUVRES:
        raise ValueError(f"unknown manoeuvre {name!r}; expected one of {MANOEUVRES}")
    options: dict[str, float] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"scenario parameter {token!r} must be key=value")
        key, value = token.split("=", 1)
        options[key.strip().lower()] = float(value)
    return name, options


def build(spec: str) -> ScenarioPath:
    """Build a scenario path from a one-line specification.

    Defaults keep every manoeuvre fully defined and inside the arena (``+-2.45 x +-1.45 m``): a line
    runs 3 m along ``+x`` from ``(-1.5, 0)``, a circle of radius 1 m starts at ``(-1.0, -1.0)`` and
    sweeps 180 degrees to the right of its centre ``(-1.0, 0)``, a corner turns 90 degrees with a
    0.6 m radius after a 0.8 m straight, and a shift moves 0.21 m sideways at the midpoint of a 3 m
    line. ``x``, ``y`` and ``z`` override the start point and ``angle`` is always in degrees.

    Radii above about 1.4 m do not fit a half circle inside the arena, so large-radius tests use a
    smaller ``angle``; the curvature is exact either way and the steady state is measured after the
    entry transient.

    Args:
        spec: The scenario specification.

    Returns:
        The scenario path.

    Raises:
        ValueError: If the manoeuvre is unknown or a parameter is malformed.
    """
    name, options = parse_spec(spec)
    x = options.pop("x", None)
    y = options.pop("y", None)
    z = options.pop("z", 1.0)
    if x is None:
        x = -1.0 if name == "circle" else -1.5
    if y is None:
        y = -1.0 if name == "circle" else 0.0
    start = np.array([x, y, z], dtype=float)
    heading = np.array([1.0, 0.0, 0.0])
    if name == "line":
        return Line(start, heading, options.pop("length", 3.0))
    if name == "circle":
        radius = options.pop("radius", 1.0)
        # A circle starts at the bottom of its own arc, so the drone begins on the path.
        centre = start + np.array([0.0, radius, 0.0])
        angle = np.radians(options.pop("angle", 180.0))
        return Circle(centre, radius, start_angle=-0.5 * np.pi, sweep=angle)
    if name == "corner":
        return Corner(
            start,
            heading,
            straight_in=options.pop("straight_in", 0.8),
            radius=options.pop("radius", 0.6),
            sweep=np.radians(options.pop("angle", 90.0)),
            straight_out=options.pop("straight_out", 0.5),
            turn_left=bool(options.pop("turn_left", 1.0)),
        )
    length = options.pop("length", 3.0)
    return ParallelShift(
        start,
        heading,
        length=length,
        offset=options.pop("offset", 0.21),
        at=options.pop("at", 0.5 * length),
    )


def controller_overrides(spec: str) -> dict[str, float]:
    """Return the MPCC overrides a scenario specification asks for.

    A scenario sets its lateral acceleration budget through ``alat``, which maps onto
    ``a_lat_max``. The path speed is handled by the scenario controller itself, because the safe
    (``speed``) and deliberately unsafe (``overload``) cases differ in whether the curvature limit
    applies.

    Args:
        spec: The scenario specification.

    Returns:
        A mapping of MPCC parameter names to values; empty when the spec asks for neither.
    """
    _, options = parse_spec(spec)
    return {"a_lat_max": options["alat"]} if "alat" in options else {}


def format_spec(name: str, **parameters: float) -> str:
    """Render a scenario specification, for logging and for table rows.

    Args:
        name: The manoeuvre.
        **parameters: Key/value parameters.

    Returns:
        The specification string.
    """
    body = " ".join(f"{key}={value:g}" for key, value in parameters.items())
    return f"{name} {body}".strip()
