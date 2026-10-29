"""Path planning for the drone racing track.

The planner turns the race track (gates in a signed passing order plus cylindrical obstacles) into a
continuously differentiable, arc-length parameterized 3D path that the
[MPCC controller][lsy_drone_racing.control.my_controller.MPCCController] tracks.

The implementation follows the reference solution ``path_planning.py`` of the successful
`lsy_drone_racing_ws26`_ entry:

* Each gate contributes ``num_intermediate_points`` waypoints along the signed gate normal, so the
  drone enters the gate on one side and leaves on the other.
* A detour waypoint is inserted whenever the chord between two consecutive gate passes points more
  than ``angle_threshold_deg`` away from the gate normal, which would otherwise require a sharp
  turn around a gate post.
* Obstacle crossings are detected on a dense spline resampling and replaced by a single waypoint
  that pushes the path out of the obstacle's safety radius.
* The resulting polyline is fitted with a cubic spline and reparameterized by arc length.

Deviations from the reference implementation (all required for this environment):

* The gate order is taken from ``obs["gate_sequence"]`` and ``obs["gate_sequence_direction"]``
  instead of the raw gate array order, so repeated gates and reverse passes are handled.
* The route between two gate passages is searched with a two dimensional A* over a rasterized top
  view of the track, see [astar][lsy_drone_racing.control.astar]. The heuristic push-out of the
  reference implementation is kept as a safety net behind it, and ``PathConfig.use_astar`` selects
  which of the two decides the route.
* The blocked interval's entry sample is kept and a degenerate (dead-centre) obstacle crossing
  falls back to the chord perpendicular instead of placing the waypoint inside the obstacle.
* A re-plan is *seeded*: it starts on the old path at the drone's projection and leaves along the
  old path's tangent, and the part of the first gate passage that lies behind the drone is dropped,
  so a new path never asks the drone to turn around at its first waypoint. See
  [PathSeed][lsy_drone_racing.control.path_planning.PathSeed].
* The path is extended past the last gate and defines a linear continuation outside its parameter
  range, so the controller may evaluate it slightly ahead or behind without extrapolating a cubic.

.. _lsy_drone_racing_ws26: https://github.com/lihengrongfei/lsy_drone_racing_ws26
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.astar import ArenaGrid, AstarConfig
from lsy_drone_racing.control.bspline_path import (
    DEGREE,
    BsplinePath,
    fit_control_points,
    pin_initial_state,
    relax_control_points,
    retime,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Weight of the proximity bump around a gate centre. Mirrors the reference `qc_dyn` allocation and
#: the Gaussian contouring-weight allocation of the T-RO22 paper (Sec. IV-C).
GATE_PROXIMITY_WEIGHT = 0.4
#: Weight of the (slightly weaker) proximity bump around an obstacle.
#:
#: It used to be half the gate weight, on the argument that the drone may cut corners between gates.
#: Measured on level 2, that is exactly what it does at a pole: the planned clearance is 0.15 m, the
#: body touches at 0.114 m, so only 3.6 cm are available for lateral error, and the drone spent 6 to
#: 9 cm of it on the four pole failures of the 40-seed screen. A pole is a hard obstacle, not a
#: corner, so it now asks for as much contouring as a gate does.
OBSTACLE_PROXIMITY_WEIGHT = 0.4
#: Width of the proximity bumps, `exp(-_PROXIMITY_DECAY * d^2)`.
_PROXIMITY_DECAY = 8.0


@dataclass(frozen=True)
class PathConfig:
    r"""Configuration of the path planner.

    Attributes:
        approach_distance: Half width [m] of the waypoint spread along each gate normal.
        num_intermediate_points: Number of waypoints generated per gate pass.
        angle_threshold_deg: Chord angle [deg] above which a detour waypoint is inserted.
        detour_distance: Offset [m] of a detour waypoint from the gate centre.
        min_offset: Smallest approach offset [m] the obstacle-aware shortening may produce.
        safety_distance: Required XY clearance [m] between the path and an obstacle centre.
        push_scale: Factor applied to ``safety_distance`` when placing a detour waypoint. The
            spline between the old and the new waypoint cuts inside the detour point, so the
            waypoint is placed farther out than the required clearance.
        obstacle_iter_max: Maximum number of obstacle avoidance passes. The spline is refitted
            after each pass and the loop stops as soon as the required clearance is achieved.
        length_growth_max: Maximum factor by which obstacle avoidance may lengthen the path before
            it is rejected. A rejected pass keeps the previous, shorter path.
        use_astar: Whether to decide the route around the obstacles with a grid A* search. When
            enabled, every chord between two gate passages that violates the clearance is replaced
            by the shortest clearance-keeping route found on a rasterized top view of the track, so
            the side on which a pole is passed is searched instead of guessed. When disabled, the
            heuristic detour waypoints of ``_avoid_obstacles`` are used instead. Enabled by default:
            on the nominal level 2 track the route keeps 0.168 m to the nearest pole against 0.150 m
            for the heuristic, and on the four randomized seeds tested both finish 4 of 4 episodes
            (12.83 s against 12.74 s), so the extra margin is worth the 0.09 s.
        astar_resolution: Edge length [m] of one A* grid cell. 0.05 m rasterizes the arena into
            5684 cells and costs about 0.2 ms more than 0.10 m, which fits the control period, so
            the finer grid is used.
        astar_frame_margin: Additional clearance [m] on top of the drone radius that an A* route
            keeps from the gate posts.
        astar_bounds: Lower and upper corner of the arena in XY [m] that the A* grid covers.
        smooth_iterations: Number of Laplacian smoothing sweeps applied to the polyline after
            obstacle avoidance, at the full point spacing. The push-out inserts waypoints that
            create near-cusps, which would otherwise cap the flyable path speed.
        smoothing_margin: Factor applied to ``safety_distance`` when the smoothing projects a point
            out of an obstacle. It used to be 0.92, i.e. the smoothing deliberately stopped inside
            the requirement; it is now above 1, so the rounded path sits outside the requirement and
            that margin absorbs the tracking drift measured against it.
        enforce_clearance: Whether to run a final projection pass so the committed path really keeps
            ``safety_distance``. Every earlier stage can move the path inward.
        smooth_stride: Decimation factor applied before smoothing. One sweep only spreads the
            rounding by one point, so the rounded length scales as the product of the spacing
            and ``sqrt(sweeps)``; the two knobs are interchangeable through that product.
            for ``smooth_iterations / stride ** 2`` sweeps rounds the same distance for roughly
            ``stride ** 3`` times less work. The result is resampled back onto the original spacing,
            so the rest of the pipeline sees an unchanged array.
        anchor_window: Arc length [m] around every gate centre that is kept fixed during smoothing,
            so that the path cannot drift sideways inside a gate plane.
        gate_straight_length: Half length [m] of the straight passage forced through each gate.
        gate_straight_blend: Length [m] over which the straight passage blends into the rest of the
            path.
        gate_frame_clearance: In-plane distance [m] from a gate centre that the path must keep when
            it crosses the gate plane but does not pass through the gate. The solid frame ends at
            0.36 m and the drone is a 0.09 m sphere, so anything closer would hit the frame.
        gate_frame_window: Arc length [m] over which an unintended frame crossing is pushed away.
        gate_frame_slab: Half thickness [m] of the slab around a gate plane that counts as close
            enough to touch the frame. A body of 0.10 m radius beside a 0.03 m bar touches it from
            0.13 m away, and this is the distance at which the path is pushed out. It replaces an
            exact crossing test: a path that comes close to the plane and turns away without
            crossing it can hit the ring just as well.
        gate_frame_step: Largest in-plane displacement [m] a single frame-avoidance pass applies.
            Clearing a 0.5 m violation in one pass leaves a kink where its taper ends; the same
            correction spread over several passes keeps the path flyable.
        gate_opening_limit: Largest off-centre distance [m] at which a body still fits through the
            0.40 m wide opening, i.e. the half opening minus the body radius. A passage further out
            than this is not a passage: it is logged, because moving it would make the drone miss a
            gate it has to fly through.
        trim_seeded_passage: Whether a seeded re-plan drops the part of the first gate passage that
            lies behind its anchor. That part is the anchor fold, so it is on by default; the flag
            exists to measure it against the straight approach it removes, which is what centres the
            drone through the 0.40 m opening.
        seed_length: Length [m] of the straight seed step that a re-plan puts after its anchor. It
            sets the direction the new path leaves the old one with, so it has to be long enough for
            the waypoint spline to feel it and short enough not to delay the correction.
        bspline_spacing: Target arc length [m] per control point of the B-spline. Fewer, further
            apart control points give a smoother and less faithful path; more of them follow the
            waypoints more closely.
        bspline_speed: Nominal speed [m/s] the knot spacing is derived from, i.e. how long the
            initial state's velocity vector is in the control polygon.
        bspline_speed_limit: Speed limit [m/s] the retiming stage enforces on the derivative control
            points before the plan is committed.
        bspline_accel_limit: Acceleration limit [m/s^2] the retiming stage enforces. The knot
            spacing is stretched until both limits hold, which is the cheapest form of the time
            adjustment in Fast-Planner (RA-L 2019) Sec. V.
        bspline_relax_iters: Gradient steps spent straightening the control polygon. Zero leaves the
            geometry as fitted; a positive value trades fidelity for curvature, which is what turns
            an unflyable hairpin into a flyable arc.
        climb_height: Height [m] added above the start position before heading to the first gate.
        runout: Extra length [m] the path continues behind the last gate.
        duration: Duration [s] of the auxiliary time-parameterized spline.
        sampling_freq: Sampling frequency [Hz] of the obstacle collision scan.
    """

    approach_distance: float = 0.6
    num_intermediate_points: int = 5
    angle_threshold_deg: float = 120.0
    detour_distance: float = 0.65
    min_offset: float = 0.15
    safety_distance: float = 0.15
    push_scale: float = 1.4
    obstacle_iter_max: int = 2
    length_growth_max: float = 1.25
    use_astar: bool = True
    astar_resolution: float = 0.05
    astar_frame_margin: float = 0.02
    astar_bounds: tuple[tuple[float, float], tuple[float, float]] = ((-2.45, -1.45), (2.45, 1.45))
    smooth_iterations: int = 600
    smooth_stride: int = 4
    smoothing_margin: float = 1.25
    enforce_clearance: bool = True
    # Measured: trimming the first passage removes the anchor fold, 85 of 574 plans down to 2,
    # but costs three successes out of 20, because the straight approach is what centres the drone
    # through the opening. Left off; the fold is cheaper to avoid by not re-planning in a passage.
    trim_seeded_passage: bool = True
    gate_straight_length: float = 0.275
    gate_straight_blend: float = 0.25
    # Measured: 0.55 was requested and 0.447 delivered, because the push runs on the waypoints and
    # the fit through them cuts the corner. Asking for 0.65 to cover that loss costs 0.3 s a lap and
    # does not save seed 7, so the request stays at 0.55 and the gap is recorded here instead: a
    # tangent pass over a bar can deliver ~0.1 m less than requested.
    gate_frame_clearance: float = 0.55
    gate_frame_window: float = 0.3
    gate_frame_slab: float = 0.13
    gate_frame_step: float = 0.15
    gate_opening_limit: float = 0.101
    seed_length: float = 0.15
    bspline_spacing: float = 0.08
    bspline_speed: float = 2.0
    bspline_speed_limit: float = 2.3
    bspline_accel_limit: float = 8.0
    bspline_relax_iters: int = 150
    climb_height: float = 0.25
    runout: float = 0.4
    duration: float = 30.0
    sampling_freq: float = 50.0


@dataclass(frozen=True)
class GateFrames:
    """Body frames of all gates in the track.

    Attributes:
        normals: Passing direction of each gate, shape ``(n_gates, 3)``.
        y_axes: In-plane lateral axis of each gate, shape ``(n_gates, 3)``.
        z_axes: In-plane vertical axis of each gate, shape ``(n_gates, 3)``.
    """

    normals: NDArray[np.floating]
    y_axes: NDArray[np.floating]
    z_axes: NDArray[np.floating]

    @staticmethod
    def from_quaternions(gates_quat: NDArray[np.floating]) -> GateFrames:
        """Extract the gate body frames from their xyzw quaternions.

        Gates have to be crossed along their local x-axis, see
        [gate_passed][lsy_drone_racing.envs.utils.gate_passed].

        Args:
            gates_quat: Gate orientations as (x, y, z, w) quaternions, shape ``(n_gates, 4)``.

        Returns:
            The gate frames.
        """
        rot = R.from_quat(np.atleast_2d(np.asarray(gates_quat, dtype=float))).as_matrix()
        return GateFrames(normals=rot[:, :, 0], y_axes=rot[:, :, 1], z_axes=rot[:, :, 2])


@dataclass(frozen=True)
class PathInfo:
    """Metadata of a planned path.

    Attributes:
        waypoints: The fitted waypoints, shape ``(n_waypoints, 3)``.
        gate_centers: Gate centre of every gate pass in passing order, shape ``(n_passes, 3)``.
        gate_ids: Index of the gate of every pass in passing order, shape ``(n_passes, )``. The
            controller needs it to look up whether that gate has been seen yet, which is what its
            approach speed cap is keyed on.
        obstacles: Obstacle positions used for collision avoidance, shape ``(n_obstacles, 3)``.
    """

    waypoints: NDArray[np.floating]
    gate_centers: NDArray[np.floating]
    gate_ids: NDArray[np.integer]
    obstacles: NDArray[np.floating]


@dataclass(frozen=True)
class PathSeed:
    """Where and in which direction a re-planned path has to start.

    A re-plan is not a fresh plan: the drone is already flying, so the new path has to begin on the
    one it is following and leave it along the direction the drone is already moving. This is the
    whole content of the seed.

    Attributes:
        position: The anchor, a point *on the old path* rather than the drone's measured pose, so
            the new path starts from the old one, not from wherever tracking error left the drone.
        tangent: Unit direction of the old path at the anchor.
        length: Length [m] of the straight seed step, see ``PathConfig.seed_length``.
        velocity: The drone's measured velocity at the anchor, shape ``(3,)``, in m/s. A B-spline
            plan uses it as a boundary condition, so the new path leaves the anchor with the speed
            and direction the drone actually has rather than with whatever the old path was doing.
        acceleration: The drone's measured acceleration at the anchor, shape ``(3,)``, in m/s^2,
            used the same way. It is a finite difference of the observed velocity, so it is coarse;
            the plan only needs it to be continuous, not accurate.
    """

    position: NDArray[np.floating]
    tangent: NDArray[np.floating]
    length: float
    velocity: NDArray[np.floating] | None = None
    acceleration: NDArray[np.floating] | None = None

    def __post_init__(self):
        """Normalize the tangent and reject a degenerate seed."""
        position = np.asarray(self.position, dtype=float).reshape(3)
        tangent = np.asarray(self.tangent, dtype=float).reshape(3)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            raise ValueError("A path seed requires a non-zero tangent.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "tangent", tangent / norm)
        object.__setattr__(self, "length", float(self.length))
        for name in ("velocity", "acceleration"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, None if value is None else np.asarray(value, dtype=float).reshape(3)
            )


def path_config_from(config: object = None, **overrides: object) -> PathConfig:
    """Build the planner configuration for a race configuration.

    Two things depend on the race configuration rather than on the planner: a track that does not
    randomize its gates and obstacles does not need the strict gate handling, which costs lap time
    and buys nothing when the poses are exact, and a config may override individual planner fields
    in its [controller] section.

    The clearance margin does too. A randomized track hands the controller poses that are up to
    0.212 m wrong until the drone is 0.7 m away, and the measured tracking drift adds up to 5 cm, so
    that track plans with an outward margin. A track with exact poses needs none of that: there the
    margin is 1.0, which delivers the required clearance at no lap time cost.

    Args:
        config: A loaded race configuration, or ``None`` to get the plain defaults.
        **overrides: Fields that win over both the defaults and the config file, e.g.
            ``use_astar=False``.

    Returns:
        The planner configuration.
    """
    settings: dict[str, object] = {}
    randomized_track = True
    if config is not None:
        randomizations = config.env.get("randomizations") or {}
        randomized_track = any(
            key in randomizations for key in ("gate_pos", "gate_rpy", "obstacle_pos")
        )
        controller = config.get("controller") or {}
        for key in ("use_astar", "astar_resolution", "safety_distance", "smoothing_margin"):
            if key in controller:
                settings[key] = controller[key]
    if not randomized_track:
        settings["gate_straight_length"] = 0.0
        settings["gate_frame_clearance"] = 0.0
        settings["smoothing_margin"] = 1.0
    settings.update(overrides)
    return PathConfig(**settings)


class ArcPath:
    """Arc-length parameterized, continuously differentiable 3D path.

    Inside its parameter range ``[0, length]`` the path is a natural cubic spline through the
    waypoints with knots equal to the cumulative chord length. Outside the range it continues
    linearly along the terminal tangents, so the controller can evaluate it slightly ahead of the
    end of the track without extrapolating a cubic.
    """

    def __init__(self, points: NDArray[np.floating]):
        """Fit the path.

        Args:
            points: Path waypoints in world coordinates, shape ``(n_points, 3)``.

        Raises:
            ValueError: If fewer than two distinct waypoints are provided.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-9))
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("A path requires at least two distinct waypoints.")
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        self._spline = CubicSpline(s, pts, bc_type="natural", extrapolate=False)
        self.points = pts
        self.length = float(s[-1])
        self._pos_start = pts[0]
        self._pos_end = pts[-1]
        self._tan_start = self._unit_tangent(0.0)
        self._tan_end = self._unit_tangent(self.length)

    def _unit_tangent(self, tau: float) -> NDArray[np.floating]:
        """Evaluate the unit tangent at a single parameter value."""
        tangent = np.asarray(self._spline(min(max(tau, 0.0), self.length), 1), dtype=float)
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _linear_tangent(self, tau: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the linear-extension tangent for parameters outside the path range."""
        out = np.empty((len(tau), 3))
        high = tau > self.length
        out[high] = self._tan_end
        out[~high] = self._tan_start
        return out

    def pos(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the path position.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The path position, shape ``tau.shape + (3,)``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._spline(clipped), dtype=float).reshape(-1, 3)
        high = flat > self.length
        if np.any(high):
            out[high] = self._pos_end + np.outer(flat[high] - self.length, self._tan_end)
        low = flat < 0.0
        if np.any(low):
            out[low] = self._pos_start + np.outer(flat[low], self._tan_start)
        return out.reshape(*tau_arr.shape, 3)

    def tangent(self, tau: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the unit tangent of the path.

        Args:
            tau: Arc length parameter, any shape.

        Returns:
            The unit tangent, shape ``tau.shape + (3,)``.
        """
        tau_arr = np.asarray(tau, dtype=float)
        flat = np.atleast_1d(tau_arr).ravel()
        clipped = np.clip(flat, 0.0, self.length)
        out = np.asarray(self._spline(clipped, 1), dtype=float).reshape(-1, 3)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, np.maximum(norm, 1e-9))
        outside = (flat < 0.0) | (flat > self.length)
        if np.any(outside):
            out[outside] = self._linear_tangent(flat[outside])
        return out.reshape(*tau_arr.shape, 3)

    def curvature(self, tau: NDArray[np.floating] | float, step: float = 0.05) -> NDArray:
        """Evaluate the curvature of the path.

        The curvature is measured over a finite arc length rather than from the analytic second
        derivative, because the analytic value of the fitted spline is very sensitive to the dense
        waypoint sample of the obstacle avoidance.

        Args:
            tau: Arc length parameter, any shape.
            step: Half width [m] over which the tangent change is measured.

        Returns:
            The curvature in 1/m, shape ``tau.shape``.
        """
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
        """Project a position onto the path, searching around the previous progress.

        The search is limited to a window around ``tau_prev`` and refined locally, which keeps the
        projection robust for paths that revisit the same region (a loop through a gate twice).

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
            high = min(self.length, low + step)
        grid = np.arange(low, high + 1e-9, step)
        if len(grid) < 2:
            grid = np.array([low, high])
        dists = np.linalg.norm(self.pos(grid) - point, axis=1)
        best = int(np.argmin(dists))
        # Parabolic refinement around the best sample for sub-step accuracy.
        if 0 < best < len(grid) - 1:
            y0, y1, y2 = dists[best - 1], dists[best], dists[best + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                offset = 0.5 * (y0 - y2) / denom
                return float(np.clip(grid[best] + offset * step, low, high))
        return float(grid[best])


class PathPlanner:
    """Generate an obstacle-avoiding, arc-length parameterized path through the track."""

    def __init__(self, config: PathConfig | None = None):
        """Initialize the planner.

        Args:
            config: Planner configuration. Defaults to [PathConfig][lsy_drone_racing.control.
                path_planning.PathConfig].
        """
        self.config = config or PathConfig()

    def plan(
        self,
        obs: dict[str, NDArray[np.floating]],
        start_pos: NDArray[np.floating],
        climb: bool = True,
        seed: PathSeed | None = None,
    ) -> tuple[ArcPath, PathInfo]:
        """Plan the path from the start position through all gates in the configured order.

        Args:
            obs: Environment observation. Requires ``gates_pos``, ``gates_quat``,
                ``gate_sequence``, ``gate_sequence_direction`` and ``obstacles_pos``.
            start_pos: Position the path starts at, shape ``(3,)``. A re-plan passes the drone's
                current position and the sequence of the passes still to come, so the planned path
                begins where the drone actually is rather than at the take-off point it left behind.
                A seed, when given, replaces this with the seed's position.
            climb: Whether the path rises ``climb_height`` above ``start_pos`` before the gates.
                first gate. True for the first plan of an episode, which starts on the ground.
            seed: Where and in which direction a re-plan has to start, see
                [PathSeed][lsy_drone_racing.control.path_planning.PathSeed]. ``None`` for the
                plan, which starts from rest and has no path to continue.

        Returns:
            The arc-length parameterized path and its metadata.
        """
        gate_pos = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
        frames = GateFrames.from_quaternions(np.asarray(obs["gates_quat"], dtype=float))
        obstacles = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
        sequence = np.asarray(obs["gate_sequence"], dtype=int).ravel()
        direction = np.sign(np.asarray(obs["gate_sequence_direction"], dtype=float)).ravel()
        direction = np.where(direction == 0, 1.0, direction)

        pass_pos = gate_pos[sequence]
        pass_normals = frames.normals[sequence] * direction[:, None]
        pass_y_axes = frames.y_axes[sequence]
        pass_z_axes = frames.z_axes[sequence]

        start = np.asarray(start_pos, dtype=float).reshape(3)
        completed = False
        if seed is not None:
            start = np.asarray(seed.position, dtype=float).reshape(3)
            completed = self._completed_pass(
                start, seed.tangent, pass_pos, pass_normals, pass_y_axes, pass_z_axes
            )
        blocks = self._gate_blocks(start, pass_pos, pass_normals, obstacles, climb=climb, seed=seed)
        # A re-plan starts inside the first passage often enough that the part of it behind the
        # has to go before anything else looks at the waypoints: it is a fold within centimetres of
        # the anchor, and the smoothing can only turn it into a corner.
        if seed is not None and self.config.trim_seeded_passage:
            blocks = self._trim_first_passage(
                blocks,
                pass_pos,
                pass_normals,
                direction=None if seed is None else seed.tangent,
                # Measured: trimming from the end of the seed step rather than from the anchor is
                # what removes the last fold. The seed step is 0.15 m long, so a drone that is
                # closer than that to the passage start ends up with the passage *behind* the step
                # and folds it back anyway.
                origin=None if seed is None else blocks[1][0],
                # A pass the drone has already flown through needs no crossing: forcing one is what
                # asked the drone to come back through the gate it had just left.
                crossing_required=not completed,
            )
        blocks = self._insert_detours(blocks, pass_pos, pass_normals, pass_y_axes, pass_z_axes)
        runout = self._runout(blocks, pass_pos, pass_normals)
        if self.config.use_astar and len(obstacles):
            waypoints = self._connect_blocks(blocks, runout, obstacles, gate_pos, frames.y_axes)
        else:
            waypoints = np.concatenate([np.concatenate(blocks, axis=0), runout])
        waypoints = self._avoid_obstacles(waypoints, obstacles)
        waypoints = self._smooth_path(waypoints, obstacles)
        waypoints = self._straighten_gate_passages(waypoints, pass_pos, pass_normals)
        # Every gate of the track, not only the remaining passes: a gate the drone has already flown
        # through is still a solid ring, and on a track that doubles back the route to the next gate
        # crosses its plane.
        crossing_count = np.zeros(len(gate_pos), dtype=int)
        for gate_id in sequence:
            crossing_count[gate_id] += 1
        waypoints = self._avoid_gate_frames(
            waypoints, gate_pos, frames.normals, frames.y_axes, frames.z_axes, crossing_count
        )
        if self.config.enforce_clearance:
            # Last of the waypoint stages, because every stage above can move the path inward: the
            # smoothing rounds corners and the gate handling pushes sideways. This is the pass that
            # makes the promise true.
            waypoints = self._enforce_clearance(waypoints, obstacles)
        path = self._to_bspline_path(waypoints, obstacles, pass_pos, seed)
        info = PathInfo(
            waypoints=waypoints,
            gate_centers=pass_pos,
            gate_ids=np.asarray(sequence, dtype=int),
            obstacles=obstacles if len(obstacles) else np.zeros((0, 3)),
        )
        return path, info

    @staticmethod
    def _completed_pass(
        start: NDArray[np.floating],
        tangent: NDArray[np.floating],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        pass_y_axes: NDArray[np.floating],
        pass_z_axes: NDArray[np.floating],
    ) -> bool:
        """Return whether the drone is through the first pass of the sequence, or about to be.

        A re-plan can be made on the same tick as a gate pass: the controller plans before the
        environment steps, so the sequence still lists the gate the drone is about to fly through,
        and by the time the new path is active it has flown through it. Planning a passage the drone
        is already past is what produced a second, off-centre crossing of that gate's opening in the
        measurements on level 2 - the drone was 5 cm past the plane, the trimmed passage pulled it
        back through it, and the route then left that gate 0.16 m off centre and into the frame.

        The test mirrors the environment's own pass test - at or just before the plane, inside the
        opening, and heading through it - because only then is the pass either registered or certain
        to register on the next step. The plane tolerance is loose on the near side for exactly that
        reason: the anchor is the pose at the *end* of the interval, so a drone 1 cm before the
        at 1.3 m/s has already crossed it by the time the new path is used.

        Args:
            start: The anchor of the re-plan, shape ``(3,)``.
            tangent: Direction the drone is flying, shape ``(3,)``.
            pass_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            pass_normals: Signed traversal normal of every pass, shape ``(n_passes, 3)``.
            pass_y_axes: In-plane lateral axis of every pass, shape ``(n_passes, 3)``.
            pass_z_axes: In-plane vertical axis of every pass, shape ``(n_passes, 3)``.

        Returns:
            ``True`` if the first pass of the sequence is being completed.
        """
        if len(pass_pos) == 0:
            return False
        normal = np.asarray(pass_normals[0], dtype=float)
        offset = np.asarray(start, dtype=float) - np.asarray(pass_pos[0], dtype=float)
        along = float(offset @ normal)
        lateral = float(offset @ np.asarray(pass_y_axes[0], dtype=float))
        vertical = float(offset @ np.asarray(pass_z_axes[0], dtype=float))
        heading = float(np.asarray(tangent, dtype=float) @ normal)
        return along > -0.05 and max(abs(lateral), abs(vertical)) < 0.20 and heading > 0.3

    def _gate_blocks(
        self,
        start: NDArray[np.floating],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        obstacles: NDArray[np.floating],
        climb: bool = True,
        seed: PathSeed | None = None,
    ) -> list[NDArray[np.floating]]:
        """Build one waypoint block per gate pass, preceded by start and climb blocks.

        Args:
            start: Where the path begins.
            pass_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            pass_normals: Gate normals in passing order, shape ``(n_passes, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.
            climb: Whether the second block rises ``climb_height`` above the start. Only the first
                of an episode starts on the ground; a re-plan starts wherever the drone is, and a
                climb there would pull the path above it. The block itself is kept either way: it is
                the first segment of the polyline, and without it the routing degenerates and the
                path comes out about 45 percent longer. A re-plan replaces the climb with a short
                level step toward the next gate.
            seed: The direction a re-plan leaves the old path with. When given, the second block
                steps along the *old* path's tangent instead of toward the next gate, which is what
                re-plan from turning the drone around at the anchor.

        Returns:
            One block per gate pass, preceded by the start block.
        """
        cfg = self.config
        passages = []
        for pos, normal in zip(pass_pos, pass_normals, strict=True):
            d_in = self._side_offset(pos, normal, -1.0, obstacles)
            d_out = self._side_offset(pos, normal, +1.0, obstacles)
            offsets = np.linspace(-d_in, d_out, cfg.num_intermediate_points)
            passages.append(pos.reshape(1, 3) + offsets[:, None] * normal.reshape(1, 3))
        blocks = [start.reshape(1, 3)]
        if seed is not None:
            # The seed step decides the direction the spline leaves the anchor with, and it must not
            # reach past the start of the first passage: a step that overshoots it makes the passage
            # begin behind a point that is already on the path, which is a fold the trim would then
            # have to eat along with the approach the gate needs for centring.
            gap = float((passages[0][0] - start) @ seed.tangent) if passages else 0.0
            ahead = np.asarray(seed.tangent, dtype=float) * max(0.02, min(seed.length, gap))
        elif climb:
            ahead = np.array([0.0, 0.0, cfg.climb_height])
        elif len(pass_pos):
            towards = np.asarray(pass_pos[0], dtype=float) - start
            length = float(np.linalg.norm(towards))
            ahead = towards / max(length, 1e-9) * max(0.05, min(0.3, 0.4 * length))
        else:
            ahead = np.zeros(3)
        blocks.append((start + ahead).reshape(1, 3))
        blocks.extend(passages)
        return blocks

    def _side_offset(
        self,
        gate_pos: NDArray[np.floating],
        normal: NDArray[np.floating],
        sign: float,
        obstacles: NDArray[np.floating],
    ) -> float:
        """Return the largest approach offset on one side of a gate that clears all obstacles.

        On this track the nominal approach points of three gate passes are within a few
        centimeters of an obstacle pole, so a fixed offset would make the path graze a pole. The
        offset is shortened (down to ``min_offset``) until the waypoint clears every obstacle by the
        same distance a detour waypoint is placed at.

        Args:
            gate_pos: Gate centre, shape ``(3,)``.
            normal: Signed traversal normal of the gate, shape ``(3,)``.
            sign: ``-1`` for the approach side, ``+1`` for the exit side.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The approach offset in meters.
        """
        cfg = self.config
        if obstacles is None or len(obstacles) == 0:
            return cfg.approach_distance
        required = cfg.safety_distance * cfg.push_scale
        offset = cfg.approach_distance
        while offset > cfg.min_offset:
            point = gate_pos + sign * offset * normal
            clearance = float(np.min(np.linalg.norm(obstacles[:, :2] - point[:2], axis=1)))
            if clearance >= required:
                break
            offset -= 0.05
        return offset

    def _runout(
        self,
        blocks: list[NDArray[np.floating]],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return the final waypoint behind the last gate."""
        distance = self.config.approach_distance + self.config.runout
        return (pass_pos[-1] + distance * pass_normals[-1]).reshape(1, 3)

    def _insert_detours(
        self,
        blocks: list[NDArray[np.floating]],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        pass_y_axes: NDArray[np.floating],
        pass_z_axes: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Insert a lateral detour waypoint where the path would double back on itself."""
        cfg = self.config
        threshold = np.deg2rad(cfg.angle_threshold_deg)
        out: list[NDArray[np.floating]] = list(blocks[:2])
        for i, block in enumerate(blocks[2:]):
            out.append(block)
            if i + 1 >= len(blocks) - 2:
                continue
            p1 = block[-1]
            p2 = blocks[2 + i + 1][0]
            chord = p2 - p1
            chord_norm = np.linalg.norm(chord)
            if chord_norm < 1e-6:
                continue
            normal = pass_normals[i]
            cos_angle = float(np.clip(np.dot(chord, normal) / chord_norm, -1.0, 1.0))
            if np.arccos(cos_angle) <= threshold:
                continue
            in_plane = chord - np.dot(chord, normal) * normal
            if np.linalg.norm(in_plane) < 1e-6:
                detour_dir = pass_y_axes[i]
            else:
                proj_angle = np.degrees(
                    np.arctan2(np.dot(in_plane, pass_z_axes[i]), np.dot(in_plane, pass_y_axes[i]))
                )
                if -90 <= proj_angle < 45:
                    detour_dir = pass_y_axes[i]
                elif 45 <= proj_angle < 135:
                    detour_dir = pass_z_axes[i]
                else:
                    detour_dir = -pass_y_axes[i]
            out.append((pass_pos[i] + cfg.detour_distance * detour_dir).reshape(1, 3))
        return out

    def _connect_blocks(
        self,
        blocks: list[NDArray[np.floating]],
        runout: NDArray[np.floating],
        obstacles: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_y_axes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Join the waypoint blocks, routing around the obstacles where a chord is blocked.

        The blocks are kept verbatim, because their points define the gate passages and a path that
        leaves the gate normal inside the opening clips the frame. Only the straight chords between
        two consecutive blocks are negotiable: a chord that already keeps the required clearance is
        kept as it is, and a blocked one is replaced by the route found by
        [ArenaGrid.search][lsy_drone_racing.control.astar.ArenaGrid.search]. The height is
        interpolated along the leg because the search is two dimensional.

        Args:
            blocks: Waypoint blocks of the track, each shape ``(n_block, 3)``.
            runout: Final waypoint behind the last gate, shape ``(1, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.
            gates_pos: Gate centres, shape ``(n_gates, 3)``. Every gate contributes a pair of posts
                that a route has to avoid.
            gates_y_axes: In-plane lateral axis of every gate, shape ``(n_gates, 3)``.

        Returns:
            The joined waypoints, shape ``(m_points, 3)``.
        """
        cfg = self.config
        groups = [np.asarray(block, dtype=float).reshape(-1, 3) for block in blocks]
        groups.append(np.asarray(runout, dtype=float).reshape(-1, 3))
        grid = ArenaGrid(
            obstacles,
            gates_pos,
            gates_y_axes,
            AstarConfig(
                resolution=cfg.astar_resolution,
                pole_clearance=cfg.safety_distance,
                frame_margin=cfg.astar_frame_margin,
                # The smoothing pass below cuts inside a corner exactly like it cuts inside a
                # heuristic detour waypoint, so the search overshoots by the same factor.
                overshoot=cfg.push_scale,
                bounds_low=cfg.astar_bounds[0],
                bounds_high=cfg.astar_bounds[1],
            ),
        )
        joined = [groups[0]]
        for previous, current in zip(groups[:-1], groups[1:], strict=True):
            start, goal = previous[-1], current[0]
            span = float(np.linalg.norm((goal - start)[:2]))
            # Only a chord that misses the clearance itself needs a route. Testing against the
            # routed margin here would reroute legs that already comply, which bends the path
            # around a pole it clears and leaves a kink where the detour rejoins the gate block.
            if span < cfg.astar_resolution or grid.line_meets_clearance(start[:2], goal[:2]):
                joined.append(current)
                continue
            leg = grid.search(start[:2], goal[:2])
            if leg is None or len(leg) < 2:
                logger.warning(
                    "No A* route between %s and %s, keeping the straight chord", start, goal
                )
                joined.append(current)
                continue
            joined.append(self._lift_leg(leg, start, goal)[1:-1])
            joined.append(current)
        return np.concatenate(joined, axis=0)

    @staticmethod
    def _lift_leg(
        leg: NDArray[np.floating], start: NDArray[np.floating], goal: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Attach a height to a two dimensional route by interpolating along its length.

        Args:
            leg: The route in the XY plane, shape ``(n, 2)``.
            start: The anchor the route starts at, shape ``(3,)``.
            goal: The anchor the route ends at, shape ``(3,)``.

        Returns:
            The three dimensional route, shape ``(n, 3)``, starting at ``start`` and ending at
            ``goal``.
        """
        chords = np.linalg.norm(np.diff(leg, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(chords)))
        fraction = cumulative / max(float(cumulative[-1]), 1e-9)
        heights = start[2] + fraction * (goal[2] - start[2])
        lifted = np.column_stack([np.asarray(leg, dtype=float), heights])
        lifted[0] = start
        lifted[-1] = goal
        return lifted

    def _avoid_obstacles(
        self, waypoints: NDArray[np.floating], obstacles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Push the path out of every obstacle's safety radius.

        A single push-out pass is not enough in general, because the spline that is refitted through
        the modified waypoints cuts inside the inserted detour point. The pass is therefore repeated
        with the refitted spline until the required clearance is met or the iteration limit is hit.

        Args:
            waypoints: Path waypoints, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The obstacle-free waypoints, shape ``(n_points, 3)``.
        """
        if len(obstacles) == 0:
            return waypoints
        cfg = self.config
        points = self._sample(self._create_spline(waypoints, cfg.duration), cfg)
        length_0 = self._path_length(points)
        for _ in range(cfg.obstacle_iter_max):
            for obstacle in obstacles:
                points = self._push_out(points, obstacle)
            if len(points) < 2:
                raise RuntimeError("Obstacle avoidance removed all path points.")
            samples = self._sample(self._create_spline(points, cfg.duration), cfg)
            if self._path_length(samples) > cfg.length_growth_max * length_0:
                # The spline zig-zags around the obstacles instead of going around them. Reject
                # the pass and keep the shorter path.
                break
            points = samples
            if self.clearance(samples, obstacles) >= cfg.safety_distance:
                return samples
        return points

    @staticmethod
    def _path_length(points: NDArray[np.floating]) -> float:
        """Return the polyline length of a sampled path."""
        deltas = np.diff(np.asarray(points, dtype=float), axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def _smooth_path(
        self, points: NDArray[np.floating], obstacles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Round the corners of the polyline while keeping the obstacle clearance.

        The waypoint insertion of the obstacle avoidance creates near-cusps: the polyline jumps from
        the entry sample to the detour waypoint and back. Those cusps have a curvature of tens per
        meter and would limit the flyable speed to a fraction of a meter per second. Laplacian
        smoothing rounds them, and every sweep re-projects points that came closer to an obstacle
        than the required clearance onto the clearance circle.

        Smoothing also rounds the corners at the gates, which pushes the path sideways inside the
        gate plane; the drone then follows it into the gate frame. The gate passages are restored
        afterwards, see [PathPlanner._straighten_gate_passages][].

        Args:
            points: Densely sampled path points, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The smoothed path points, shape ``(n_points, 3)``.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if len(pts) < 3:
            return pts
        # Smooth a decimated copy and resample the result back onto the input spacing. A sweep only
        # spreads the rounding by one point, so rounding a distance sigma needs
        # spacing * sqrt(sweeps) == sigma: taking every stride-th point allows stride ** 2
        # fewer sweeps for the same rounding, which is stride ** 3 less work. The array comes
        # back at the input resolution, so nothing downstream can tell the difference.
        stride = max(1, int(self.config.smooth_stride))
        index = (
            np.arange(len(pts))
            if stride == 1
            else np.unique(np.append(np.arange(0, len(pts), stride), len(pts) - 1))
        )
        coarse = pts[index].copy()
        sweeps = max(1, int(round(self.config.smooth_iterations / stride**2)))

        # Project onto the clearance circle scaled by ``smoothing_margin``. This used to be 0.92:
        # the smoothing deliberately stopped inside the required clearance, which is why committed
        # paths came out at 0.10-0.14 m instead of 0.15 m. It is now an outward margin, so the
        # rounded path ends up above the requirement and leaves room for the tracking drift.
        radius = self.config.safety_distance * self.config.smoothing_margin
        radius_squared = radius * radius
        anchors = (
            None
            if obstacles is None or len(obstacles) == 0
            else np.asarray(obstacles, dtype=float)[:, :2]
        )
        rows = np.arange(len(coarse))
        for _ in range(sweeps):
            coarse[1:-1] = 0.25 * coarse[:-2] + 0.5 * coarse[1:-1] + 0.25 * coarse[2:]
            if anchors is None:
                continue
            delta = coarse[:, None, :2] - anchors[None, :, :2]
            # Squared distances: the proximity test only needs a comparison, and the square
            # root is a measurable share of the loop on its own.
            squared = np.einsum("ijk,ijk->ij", delta, delta)
            nearest = np.argmin(squared, axis=1)
            proximity = squared[rows, nearest] < radius_squared
            if not proximity.any():
                continue
            direction = delta[rows, nearest][proximity]
            span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            coarse[proximity, :2] = anchors[nearest[proximity]] + radius * direction / span
        if stride == 1:
            return coarse

        # Back to the original sampling: interpolate by arc length through the smoothed polyline.
        knots = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(coarse, axis=0), axis=1))))
        targets = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        keep = np.concatenate(([True], np.diff(knots) > 1e-12))
        knots, coarse = knots[keep], coarse[keep]
        if len(knots) < 2:
            return pts
        spline = CubicSpline(knots, coarse, bc_type="natural", extrapolate=False)
        return np.asarray(spline(np.clip(targets, knots[0], knots[-1])), dtype=float).reshape(-1, 3)

    def _avoid_gate_frames(
        self,
        points: NDArray[np.floating],
        gate_pos: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        passes: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """Keep the path out of the solid ring of every gate it is not passing through.

        A gate is a solid frame: its collision geometry is a ring covering 0.2 m to 0.36 m from the
        gate centre in the gate plane, while the drone is a body of about 0.10 m radius. A path that
        comes within ``gate_frame_slab`` of a gate plane between roughly 0.10 m and 0.45 m from the
        centre therefore flies through the frame. Every such neighbourhood is pushed outwards, with
        cosine-tapered displacement so that the correction stays local.

        Two things this used to get wrong, both of them measured on level 2:

        * It only ever saw the passes that were still to come. A re-plan gives the planner the
          remaining gate sequence, so a gate the drone had already flown through was not in this
          at all, and the route to the next gate - which on this track doubles back - crossed that
          gate's plane through its own opening, 0.00-0.12 m off centre. The drone followed it into
          the frame. It now receives every gate of the track, with the number of legitimate
          per gate, so a passed gate still counts as a frame to avoid.
        * It only reacted to plane *crossings*. A path that looped close to a gate plane and turned
          away without crossing it came within 0.02 m of the plane at 0.43 m from the centre, and
          approximation "no sign change, no conflict" let it through. The test is now a slab around
          the plane, which catches both crossings and near misses.

        Args:
            points: Path points, shape ``(n_points, 3)``.
            gate_pos: Gate centres of the whole track, shape ``(n_gates, 3)``.
            gate_normals: Gate normals of the whole track, shape ``(n_gates, 3)``. The sign is
                irrelevant here: the plane is what matters.
            gate_y_axes: In-plane lateral axis of every gate, shape ``(n_gates, 3)``.
            gate_z_axes: In-plane vertical axis of every gate, shape ``(n_gates, 3)``.
            passes: Number of legitimate crossings per gate, shape ``(n_gates, )``. Zero for a gate
                that is not in the remaining sequence.

        Returns:
            The corrected path points, shape ``(n_points, 3)``.
        """
        cfg = self.config
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        window = max(2, int(round(cfg.gate_frame_window / max(spacing, 1e-3))))
        # Group by gate plane: a gate can appear several times in the gate order, and all of its
        # crossings through the opening are legitimate while every other approach is not.
        planes: dict[tuple, dict] = {}
        for centre, normal, y_axis, z_axis, count in zip(
            gate_pos, gate_normals, gate_y_axes, gate_z_axes, passes, strict=True
        ):
            centre = np.asarray(centre, dtype=float)
            normal = np.asarray(normal, dtype=float)
            key = tuple(np.round(np.concatenate((centre, normal)), 6))
            plane = planes.setdefault(
                key,
                {
                    "centre": centre,
                    "normal": normal,
                    "y": np.asarray(y_axis, dtype=float),
                    "z": np.asarray(z_axis, dtype=float),
                    "passes": 0,
                },
            )
            plane["passes"] += int(count)
        for plane in planes.values():
            protected = self._passage_mask(pts, plane, window)
            # A single push moves the samples around it too and can clear a whole run of violations,
            # so iterate a few times instead of solving for the exact displacement. The push is also
            # capped per pass: moving a sample 0.5 m in one go leaves a kink where the taper ends,
            # while spreading the same correction over several passes keeps the path flyable.
            for _ in range(5):
                violation = self._frame_violation(pts, plane, protected)
                if violation is None:
                    break
                index, direction, distance = violation
                indices = np.arange(max(0, index - window), min(len(pts), index + window + 1))
                weight = 0.5 * (1.0 + np.cos(np.pi * (indices - index) / window))
                weight[protected[indices]] = 0.0
                step = min(distance, self.config.gate_frame_step)
                pts[indices] += weight[:, None] * step * direction[None, :]
        return pts

    def _passage_mask(
        self, pts: NDArray[np.floating], plane: dict, window: int
    ) -> NDArray[np.bool_]:
        """Mask the samples that belong to a gate's own passage, which must not be pushed.

        The legitimate crossings are the ones closest to the gate centre, and the number of passes
        of the plane says how many there are. A passage that is not centred is logged instead of
        pushed: moving it outwards would make the drone miss the gate it has to fly through.

        Args:
            pts: Path points, shape ``(n_points, 3)``.
            plane: The plane dictionary built in [_avoid_gate_frames][...].
            window: Half width, in samples, of the neighbourhood a push affects.

        Returns:
            A boolean mask over the samples.
        """
        protected = np.zeros(len(pts), dtype=bool)
        # The first samples are the drone itself: a re-plan made as it flies through a gate starts
        # inside that gate, so pushing them out of the opening would be pushing the drone. The pin
        # re-anchors those samples anyway, and the rest of the frame is still avoided.
        protected[: window + 2] = True
        offset = pts - plane["centre"]
        along = offset @ plane["normal"]
        crossing = np.flatnonzero(np.diff(np.sign(along)) != 0)
        if len(crossing) == 0:
            return protected
        extents = []
        for index in crossing:
            fraction = -along[index] / (along[index + 1] - along[index])
            local = offset[index] + fraction * (offset[index + 1] - offset[index])
            # The frame is a square ring, so the relevant measure is the Chebyshev distance in the
            # gate plane: a point at (0.31, 0.40) is 0.51 m from the centre diagonally, but only
            # 0.04 m below the bottom bar.
            extents.append(max(abs(float(local @ plane["y"])), abs(float(local @ plane["z"]))))
        for position in np.argsort(extents)[: plane["passes"]]:
            index = int(crossing[position])
            if extents[position] > self.config.gate_opening_limit:
                logger.debug(
                    "Gate passage %.3f m off the centre: the body does not fit through the opening",
                    extents[position],
                )
            protected[max(0, index - window) : index + window + 2] = True
        return protected

    def _frame_violation(
        self, pts: NDArray[np.floating], plane: dict, protected: NDArray[np.bool_]
    ) -> tuple[int, NDArray[np.floating], float] | None:
        """Return the worst sample inside a gate's frame slab and the shift that clears it.

        Args:
            pts: Path points, shape ``(n_points, 3)``.
            plane: The plane dictionary built in [_avoid_gate_frames][...].
            protected: Samples of the gate's own passage, which are left alone.

        Returns:
            ``(index, unit direction, distance)`` of the sample to move furthest inwards, or
            ``None`` when the path keeps the frame clearance.
        """
        clearance = self.config.gate_frame_clearance
        offset = pts - plane["centre"]
        along = offset @ plane["normal"]
        lateral = offset @ plane["y"]
        vertical = offset @ plane["z"]
        extent = np.maximum(np.abs(lateral), np.abs(vertical))
        inside = (np.abs(along) < self.config.gate_frame_slab) & (extent < clearance) & ~protected
        if not inside.any():
            return None
        index = int(np.argmax(np.where(inside, clearance - extent, -np.inf)))
        # Push along the dominant in-plane axis, which grows the square clearance fastest.
        if abs(lateral[index]) >= abs(vertical[index]):
            direction = plane["y"] if lateral[index] >= 0 else -plane["y"]
        else:
            direction = plane["z"] if vertical[index] >= 0 else -plane["z"]
        return index, np.asarray(direction, dtype=float), float(clearance - extent[index])

    def _straighten_gate_passages(
        self,
        points: NDArray[np.floating],
        gate_pos: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Make every gate passage straight and perpendicular to the gate plane.

        Passing through the gate centre is not sufficient: if the path approaches the gate at an
        angle, the drone crosses the plane in the middle but its body sweeps through the frame's
        side bar, and after the gate it can clip the frame while turning away. The gate frame is a
        solid ring with a 0.4 m opening, so the path has to run along the gate normal through the
        opening.

        The samples within ``gate_straight_length`` of the gate plane (and within the gate's local
        neighbourhood) are moved onto the normal line through the gate centre, with a smooth blend
        at the ends so that the transition does not become a corner.

        Args:
            points: Smoothed path points, shape ``(n_points, 3)``.
            gate_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            gate_normals: Signed traversal normal of every pass, shape ``(n_passes, 3)``.

        Returns:
            The corrected path points, shape ``(n_points, 3)``.
        """
        length = self.config.gate_straight_length
        blend = max(self.config.gate_straight_blend, 1e-3)
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        half_window = max(2, int(round(length / max(spacing, 1e-3))))
        for centre, normal in zip(gate_pos, gate_normals, strict=True):
            centre = np.asarray(centre, dtype=float)
            normal = np.asarray(normal, dtype=float)
            # Straighten only the crossing window of this gate, so a path that legitimately passes
            # by a gate at some distance is not pulled through it.
            crossing = int(np.argmin(np.linalg.norm(pts - centre, axis=1)))
            low = max(1, crossing - half_window)
            high = min(len(pts) - 1, crossing + half_window + 1)
            indices = np.arange(low, high)
            along = (pts[indices] - centre) @ normal
            target = centre + along[:, None] * normal
            distance = np.abs(indices - crossing) * spacing
            ramp = np.clip((half_window * spacing - distance) / blend, 0.0, 1.0)
            weight = 0.5 * (1.0 - np.cos(np.pi * ramp))
            pts[indices] = (1.0 - weight)[:, None] * pts[indices] + weight[:, None] * target
        return pts

    @staticmethod
    def clearance(points: NDArray[np.floating], obstacles: NDArray[np.floating]) -> float:
        """Return the minimum XY distance between a sampled path and the obstacle centres.

        Args:
            points: Sampled path points, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The minimum XY clearance in meters, or ``inf`` if there are no obstacles.
        """
        if len(obstacles) == 0:
            return float("inf")
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        delta = points[:, None, :2] - np.asarray(obstacles, dtype=float)[None, :, :2]
        return float(np.min(np.linalg.norm(delta, axis=-1)))

    def _sample(self, spline: CubicSpline, cfg: PathConfig) -> NDArray[np.floating]:
        """Densely sample a time-parameterized spline."""
        n_samples = max(2, int(cfg.sampling_freq * cfg.duration))
        return np.asarray(spline(np.linspace(0.0, cfg.duration, n_samples)), dtype=float)

    def _push_out(
        self, points: NDArray[np.floating], obstacle: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Replace one blocked interval of the path by a waypoint next to the obstacle."""
        safety = self.config.safety_distance
        dist = np.linalg.norm(points[:, :2] - obstacle[:2], axis=1)
        keep: list[NDArray[np.floating]] = []
        i = 0
        n = len(points)
        while i < n:
            if dist[i] >= safety:
                keep.append(points[i])
                i += 1
                continue
            # Find the first free sample after the blocked interval. The entry sample is kept so
            # the path still approaches the obstacle before turning away.
            j = i
            while j < n and dist[j] < safety:
                j += 1
            keep.append(points[i])
            if j < n:
                keep.append(self._detour_point(points[i], points[j], obstacle))
            i = j
        return np.asarray(keep, dtype=float).reshape(-1, 3)

    def _detour_point(
        self,
        entry: NDArray[np.floating],
        exit_: NDArray[np.floating],
        obstacle: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute the waypoint that leads the path around an obstacle."""
        distance = self.config.safety_distance * self.config.push_scale
        dir_in = entry[:2] - obstacle[:2]
        dir_out = exit_[:2] - obstacle[:2]
        dir_in = dir_in / (np.linalg.norm(dir_in) + 1e-9)
        dir_out = dir_out / (np.linalg.norm(dir_out) + 1e-9)
        direction = dir_in + dir_out
        if np.linalg.norm(direction) < 1e-6:
            # Dead-centre crossing: use the chord perpendicular instead of the (zero) bisector.
            chord = exit_[:2] - entry[:2]
            direction = np.array([-chord[1], chord[0]])
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([0.0, 1.0])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        xy = obstacle[:2] + distance * direction
        return np.array([xy[0], xy[1], 0.5 * (entry[2] + exit_[2])])

    def _create_spline(self, waypoints: NDArray[np.floating], duration: float) -> CubicSpline:
        """Fit a cubic spline through the waypoints, parameterized by chord length."""
        pts = np.asarray(waypoints, dtype=float).reshape(-1, 3)
        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        keep = np.concatenate(([True], segment_lengths > 1e-9))
        pts, cumulative = pts[keep], cumulative[keep]
        knots = cumulative / (cumulative[-1] + 1e-9) * duration
        return CubicSpline(knots, pts, bc_type="natural")

    def _enforce_clearance(
        self, points: NDArray[np.floating], obstacles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Push the finished path back out to the required clearance from every obstacle.

        Every stage before this one can move the path inward: the smoothing rounds corners, the gate
        straightening shifts points inside the gate plane, and the frame avoidance pushes sideways.
        Without a final pass the committed path is whatever those stages left behind, which measured
        between 0.10 m and 0.15 m on level 2 while the planner claimed 0.15 m. This is the pass that
        makes the claim true, and it is cheap because it only touches the points that are too close.

        The projection is iterated because moving one point clear of a pole can bring its neighbour
        closer to another one.

        Args:
            points: The finished waypoints, shape ``(n_points, 3)``.
            obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.

        Returns:
            The waypoints with every point at least ``safety_distance`` from every obstacle centre.
        """
        if obstacles is None or len(obstacles) == 0:
            return points
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        anchors = np.asarray(obstacles, dtype=float)[:, :2]
        radius = float(self.config.safety_distance)
        rows = np.arange(len(pts))
        for _ in range(3):
            delta = pts[:, None, :2] - anchors[None, :, :2]
            squared = np.einsum("ijk,ijk->ij", delta, delta)
            nearest = np.argmin(squared, axis=1)
            violation = squared[rows, nearest] < radius * radius
            if not violation.any():
                break
            direction = delta[rows, nearest][violation]
            span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
            pts[violation, :2] = anchors[nearest[violation]] + radius * direction / span
        return pts

    def _trim_first_passage(
        self,
        blocks: list[NDArray[np.floating]],
        pass_pos: NDArray[np.floating],
        pass_normals: NDArray[np.floating],
        direction: NDArray[np.floating] | None = None,
        origin: NDArray[np.floating] | None = None,
        crossing_required: bool = True,
    ) -> list[NDArray[np.floating]]:
        """Drop the part of the first gate passage that lies behind the drone.

        A passage block runs from ``approach_distance`` before the gate to beyond it. On a re-plan
        the drone is often already *inside* that approach, so the block starts behind it and the
        polyline turns 180 degrees within a few centimetres of the anchor: a cusp that the smoothing
        then has to eat, and the fold this whole change exists to remove. Keeping only the samples
        that are still ahead removes the fold without changing where the path crosses the gate.

        Args:
            blocks: The waypoint blocks, ``blocks[0]`` the start and ``blocks[1]`` its anchor step.
            pass_pos: Gate centres in passing order, shape ``(n_passes, 3)``.
            pass_normals: Gate normals in passing order, shape ``(n_passes, 3)``.
            direction: Direction that counts as "ahead" along which the samples are tested. Defaults
                to the next gate's normal, which is right for a drone flying at the gate. A re-plan
                passes the old path's tangent instead, because a drone that is turning into the gate
                can have passage samples behind it that are still ahead of the gate plane.
            origin: Point the "ahead" test is made from. Defaults to the first waypoint; a seeded
                re-plan passes the end of its seed step, so that the passage does not start behind a
                point that is already on the path.
            crossing_required: Whether the trimmed block still has to cross the gate plane. It does
                for a pass that is still to come, because the environment only counts a pass at a
                plane crossing. A pass the drone has just flown through needs none, and forcing one
                asks the drone to come back through the gate it has just left.

        Returns:
            The blocks with the first passage trimmed to the part ahead of the start.
        """
        if len(blocks) < 3 or len(pass_normals) == 0:
            return blocks
        start = np.asarray(blocks[0][0] if origin is None else origin, dtype=float)
        normal = np.asarray(pass_normals[0], dtype=float)
        ahead_direction = normal if direction is None else np.asarray(direction, dtype=float)
        passage = blocks[2]
        ahead = (passage - start[None, :]) @ ahead_direction
        keep = ahead >= -0.02
        # The pass itself is not negotiable: if the drone is already past the gate plane, the kept
        # samples would never cross it and the gate would never count. Keep the last approach sample
        # in that case, even though it is behind the anchor and therefore a (small) fold.
        side = (passage - np.asarray(pass_pos[0], dtype=float)[None, :]) @ normal
        if crossing_required and not np.any(side[keep] <= 0.0):
            approach = np.flatnonzero(side <= 0.0)
            if len(approach):
                keep[approach[-1]] = True
        if keep.sum() >= 2:
            trimmed = list(blocks)
            trimmed[2] = passage[keep]
            return trimmed
        return blocks

    def _to_bspline_path(
        self,
        waypoints: NDArray[np.floating],
        obstacles: NDArray[np.floating],
        pass_pos: NDArray[np.floating],
        seed: PathSeed | None = None,
    ) -> BsplinePath:
        """Fit the plan as a cubic B-spline trajectory, pinned to the drone's current state.

        The B-spline replaces the natural cubic spline and its arc-length reparameterization for
        three reasons, all of them measured on this track:

        * the first three control points *are* the initial state, so a re-plan leaves the anchor
          with the drone's measured velocity and acceleration instead of an approximation of the old
          path's tangent (`pin_initial_state`);
        * moving one control point only changes ``DEGREE + 1`` spans, which is what makes a local
          repair well defined rather than approximate;
        * velocity and acceleration are finite differences of the control points, so the dynamic
          limits are visible at plan time (`retime`) rather than surviving only as a speed table in
          the controller.

        Args:
            waypoints: The finished waypoint polyline, shape ``(n_points, 3)``.
            obstacles: Pole positions, shape ``(n_obstacles, 3)``, for the clearance term of the
                relaxation.
            pass_pos: Gate centres in passing order, shape ``(n_passes, 3)``. The control points
                that define those passages are locked, so the relaxation cannot pull a gate pass
                off its opening.
            seed: The re-plan's seed, which carries the state to start from.

        Returns:
            The trajectory, exposed as an arc length parameterized path.
        """
        cfg = self.config
        points = np.asarray(waypoints, dtype=float).reshape(-1, 3)
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        n_control = max(DEGREE + 4, int(round(length / max(cfg.bspline_spacing, 1e-3))) + 4)
        time_step = max(cfg.bspline_spacing / max(cfg.bspline_speed, 1e-3), 1e-3)
        control = fit_control_points(points, n_control, time_step)
        if seed is None:
            control[0] = np.asarray(points[0], dtype=float)
        else:
            speed = 0.0 if seed.velocity is None else float(np.linalg.norm(seed.velocity))
            if speed > 0.2:
                direction = np.asarray(seed.velocity, dtype=float) / speed
            else:
                # No usable velocity reading (opening plan, or a test): keep the direction the
                # old path asked for and the nominal speed, which is what the seed's tangent means.
                direction, speed = np.asarray(seed.tangent, dtype=float), cfg.bspline_speed
            control = pin_initial_state(
                control,
                np.asarray(seed.position, dtype=float),
                direction * speed,
                np.zeros(3) if seed.acceleration is None else seed.acceleration,
                time_step,
            )
        if cfg.bspline_relax_iters > 0 and len(obstacles):
            locked = np.zeros(len(control), dtype=bool)
            for centre in pass_pos:
                locked |= np.linalg.norm(control - centre, axis=1) < 0.3
            control = relax_control_points(
                control, obstacles, cfg.safety_distance, locked, iterations=cfg.bspline_relax_iters
            )
        time_step = retime(control, time_step, cfg.bspline_speed_limit, cfg.bspline_accel_limit)
        logger.debug(
            "B-spline plan: %d control points, dt %.3f s, %.2f m", len(control), time_step, length
        )
        return BsplinePath(control, time_step)


def track_reference(
    path: ArcPath,
    thetas: NDArray[np.floating],
    gate_centers: NDArray[np.floating],
    obstacles: NDArray[np.floating],
    gate_weight: float = GATE_PROXIMITY_WEIGHT,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Evaluate the path reference used by the MPCC stage cost.

    The proximity weight ``qc`` follows the Gaussian contouring-weight allocation of the T-RO22
    paper: it peaks at the gates and obstacles and decays to zero elsewhere. The controller scales
    its contouring and lag weights with it, so the drone hugs the path exactly where it has to pass
    through a gate and is free to cut corners in between.

    ``gate_weight`` exists because the gate Gaussian and the tunnel
    ([docs/planning/tunnel_constraint.md](../../docs/planning/tunnel_constraint.md)) are two answers
    to the same problem, and they fight when both are on: the Gaussian pulls the drone onto the path
    at the gate while the tunnel allows the whole opening, so both add cost without safety.
    A controller that runs the tunnel passes ``gate_weight = 0`` and leaves gate accuracy to the
    constraint, which is the formulation of MPCC++ (Krinner, Romero et al., RSS 2024).

    Args:
        path: The planned path.
        thetas: Arc length parameters at which to evaluate, shape ``(n, )``.
        gate_centers: Gate centres in passing order, shape ``(n_passes, 3)``.
        obstacles: Obstacle positions, shape ``(n_obstacles, 3)``.
        gate_weight: Peak contour weight at a gate. Zero disables the gate term.

    Returns:
        The reference positions ``(n, 3)``, unit tangents ``(n, 3)`` and proximity weights
        ``(n, )`` in ``[0, 1]``.
    """
    thetas = np.asarray(thetas, dtype=float).ravel()
    positions = path.pos(thetas)
    tangents = path.tangent(thetas)
    weights = np.zeros(len(thetas))
    if gate_weight > 0.0 and gate_centers is not None and len(gate_centers):
        dist_sq = np.sum((positions[:, None, :] - np.asarray(gate_centers)[None, :, :]) ** 2, -1)
        weights = np.maximum(
            weights, (gate_weight * np.exp(-_PROXIMITY_DECAY * dist_sq)).max(axis=1)
        )
    if obstacles is not None and len(obstacles):
        delta = positions[:, None, :2] - np.asarray(obstacles)[None, :, :2]
        dist_sq = np.sum(delta**2, -1)
        weights = np.maximum(
            weights, (OBSTACLE_PROXIMITY_WEIGHT * np.exp(-_PROXIMITY_DECAY * dist_sq)).max(axis=1)
        )
    return positions, tangents, weights
