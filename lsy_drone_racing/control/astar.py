"""Grid based A* route search over the race track.

The waypoint construction in [path_planning][lsy_drone_racing.control.path_planning] decides on
which side of a pole the path passes with a local heuristic: it shortens the gate approach offset
until the waypoint clears every pole, and pushes the path out of any obstacle it crosses. That is
cheap and works, but the side of the obstacle is a consequence of where the waypoint happened to
land, and on a track where several legs graze a pole the heuristic has to be tuned.

This module decides the same question by search. It rasterizes the track into an occupancy grid and
runs A* between consecutive gate passages, so the side of every pole is found rather than guessed.

Why two dimensions
------------------
The obstacles are vertical cylinders that span the whole altitude band the drone flies in, and the
altitude of every gate passage is prescribed by the gate order. There is therefore nothing to
discover in ``z``: the search only has to decide the route in the ``xy`` plane, and the height is
interpolated along the resulting legs. On the level 2 track this is about 19 times faster than the
equivalent 3D search (20 ms instead of 4.4 s for the whole track at 2.5 cm resolution), which is
what makes the search affordable inside the control loop.

The grid models two kinds of obstacle:

* Poles, as circles of the required clearance around the pole centre. That clearance is the
  planner's ``safety_distance`` and already contains the drone and pole radii, so they are not added
  again.
* Gate frames, as the two side posts of every gate, inflated by the drone radius and a small
  margin. The frames have to be in the grid: two gates of this track are less than a metre apart,
  so a route that ignores the posts threads straight through a frame it is not passing. Only the
  side posts are modelled, because the drone always crosses a gate plane at the opening height; the
  top and bottom bars are outside its body envelope there.

The raw grid path is a staircase, so it is simplified with a line-of-sight shortcut before it leaves
this module. The shortcut is validated against the exact obstacle geometry rather than against the
grid, so simplifying the path can never reintroduce a collision.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Radius of the sphere that encloses the drone [m].
DRONE_RADIUS = 0.09
#: Half width of the gate opening [m]. The frame is square with a 0.4 m wide opening.
GATE_OPENING_HALF = 0.20
#: Half width of the outer edge of the gate frame [m]. The frame is 0.72 m wide overall.
GATE_FRAME_HALF = 0.36
#: Radius of the cylindrical obstacles [m]. The obstacles are 0.03 m in diameter.
POLE_RADIUS = 0.015

#: The eight neighbours of a cell in a two dimensional 8-connected grid.
_NEIGHBOURS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=int
)


@dataclass(frozen=True)
class AstarConfig:
    """Configuration of the A* route search.

    Attributes:
        resolution: Edge length [m] of one grid cell. Finer grids find shorter routes but cost
            quadratically more time.
        pole_clearance: Required clearance [m] between the path and a pole centre. This is the
            planner's `safety_distance`, in the same centre-to-centre convention as
            [PathPlanner.clearance][lsy_drone_racing.control.path_planning.PathPlanner.clearance],
            so the search and the smooth path agree on the requirement. The drone and pole radii
            are part of that number, not added on top of it.
        frame_margin: Additional XY clearance [m] on top of the drone radius that the route keeps
            from a gate post. Only here are the radii added explicitly, because the posts are
            modelled as bare centrelines rather than as solids.
        overshoot: Factor applied to ``pole_clearance``. The planner smooths the route afterwards
            and the Laplacian relaxation cuts inside a corner, so a route that only just satisfies
            the clearance ends up below it. This mirrors ``PathConfig.push_scale``, which places the
            heuristic detour waypoints farther out for the same reason.
        bounds_low: Lower corner of the arena in XY [m].
        bounds_high: Upper corner of the arena in XY [m].
        shortcut: Whether to simplify the staircase path with a line-of-sight shortcut.
    """

    resolution: float = 0.10
    pole_clearance: float = 0.15
    frame_margin: float = 0.02
    overshoot: float = 1.0
    bounds_low: tuple[float, float] = (-2.45, -1.45)
    bounds_high: tuple[float, float] = (2.45, 1.45)
    shortcut: bool = True


def points_to_segment_distance(
    points: NDArray[np.floating], start: NDArray[np.floating], end: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute the distance between many points and one line segment.

    Args:
        points: Query points, shape ``(n, 2)``.
        start: First end of the segment, shape ``(2,)``.
        end: Second end of the segment, shape ``(2,)``.

    Returns:
        The Euclidean distances, shape ``(n, )``.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    chord = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    denom = float(chord @ chord)
    if denom < 1e-12:
        return np.linalg.norm(pts - np.asarray(start, dtype=float), axis=1)
    projection = (pts - np.asarray(start, dtype=float)) @ chord / denom
    closest = np.asarray(start, dtype=float) + np.clip(projection, 0.0, 1.0)[:, None] * chord
    return np.linalg.norm(pts - closest, axis=1)


def point_to_segments_distance(
    point: NDArray[np.floating], starts: NDArray[np.floating], ends: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Compute the distance between one point and many line segments.

    Args:
        point: The query point, shape ``(2,)``.
        starts: First ends of the segments, shape ``(n, 2)``.
        ends: Second ends of the segments, shape ``(n, 2)``.

    Returns:
        The Euclidean distances, shape ``(n, )``.
    """
    origin = np.asarray(starts, dtype=float).reshape(-1, 2)
    chords = np.asarray(ends, dtype=float).reshape(-1, 2) - origin
    denom = np.maximum(np.sum(chords * chords, axis=1), 1e-12)
    projection = ((np.asarray(point, dtype=float) - origin) * chords).sum(axis=1) / denom
    closest = origin + np.clip(projection, 0.0, 1.0)[:, None] * chords
    return np.linalg.norm(np.asarray(point, dtype=float) - closest, axis=1)


def segment_distances(
    start: NDArray[np.floating],
    goal: NDArray[np.floating],
    starts: NDArray[np.floating],
    ends: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the distance between one line segment and many line segments.

    Args:
        start: First end of the query segment, shape ``(2,)``.
        goal: Second end of the query segment, shape ``(2,)``.
        starts: First ends of the other segments, shape ``(n, 2)``.
        ends: Second ends of the other segments, shape ``(n, 2)``.

    Returns:
        The distances, shape ``(n, )``, zero where a segment intersects the query.
    """
    a0 = np.asarray(start, dtype=float)
    a1 = np.asarray(goal, dtype=float)
    origin = np.asarray(starts, dtype=float).reshape(-1, 2)
    target = np.asarray(ends, dtype=float).reshape(-1, 2)
    chords = target - origin
    query = a1 - a0
    offset_origin = origin - a0
    offset_target = target - a0
    relative_origin = a0 - origin
    relative_target = a1 - origin
    d1 = query[0] * offset_origin[:, 1] - query[1] * offset_origin[:, 0]
    d2 = query[0] * offset_target[:, 1] - query[1] * offset_target[:, 0]
    d3 = chords[:, 0] * relative_origin[:, 1] - chords[:, 1] * relative_origin[:, 0]
    d4 = chords[:, 0] * relative_target[:, 1] - chords[:, 1] * relative_target[:, 0]
    intersects = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
    candidates = np.minimum(
        np.minimum(
            points_to_segment_distance(origin, a0, a1), points_to_segment_distance(target, a0, a1)
        ),
        np.minimum(
            point_to_segments_distance(a0, origin, target),
            point_to_segments_distance(a1, origin, target),
        ),
    )
    return np.where(intersects, 0.0, candidates)


def segment_distance(
    start_a: NDArray[np.floating],
    end_a: NDArray[np.floating],
    start_b: NDArray[np.floating],
    end_b: NDArray[np.floating],
) -> float:
    """Compute the distance between two line segments in the plane.

    Args:
        start_a: First end of the first segment, shape ``(2,)``.
        end_a: Second end of the first segment, shape ``(2,)``.
        start_b: First end of the second segment, shape ``(2,)``.
        end_b: Second end of the second segment, shape ``(2,)``.

    Returns:
        The smallest distance between the two segments, zero if they intersect.
    """
    return float(
        segment_distances(
            start_a,
            end_a,
            np.asarray(start_b, dtype=float)[None],
            np.asarray(end_b, dtype=float)[None],
        )[0]
    )


def _cross(lhs: NDArray[np.floating], rhs: NDArray[np.floating]) -> float:
    """Return the two dimensional cross product ``lhs x rhs``."""
    return float(lhs[0] * rhs[1] - lhs[1] * rhs[0])


class ArenaGrid:
    """Rasterized top view of the race track with an A* search on top of it.

    The grid is built once per plan and then queried once per leg, so the cost of rasterizing the
    obstacles is amortized over the whole track.
    """

    def __init__(
        self,
        obstacles_pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_y_axes: NDArray[np.floating],
        config: AstarConfig | None = None,
    ):
        """Rasterize the obstacles and the gate frames.

        Args:
            obstacles_pos: Cylinder centres, shape ``(n_obstacles, 3)``. Only XY and the radius
                matter: the obstacles are treated as vertical and infinitely tall in the search.
            gates_pos: Gate centres, shape ``(n_gates, 3)``.
            gates_y_axes: In-plane lateral axis of every gate, shape ``(n_gates, 3)``. Only its XY
                projection is used, so a strongly pitched gate is approximated.
            config: Search configuration. Defaults to [AstarConfig][...AstarConfig].
        """
        self.config = config or AstarConfig()
        self._poles = np.asarray(obstacles_pos, dtype=float).reshape(-1, 3)[:, :2]
        self.pole_inflate = self.config.pole_clearance * self.config.overshoot
        self.frame_pad = DRONE_RADIUS + self.config.frame_margin
        self._frame_segments = _gate_posts(
            np.asarray(gates_pos, dtype=float).reshape(-1, 3),
            np.asarray(gates_y_axes, dtype=float).reshape(-1, 3),
        )
        self._posts_start = np.asarray([start for start, _ in self._frame_segments], dtype=float)
        self._posts_end = np.asarray([end for _, end in self._frame_segments], dtype=float)
        self._posts_start = self._posts_start.reshape(-1, 2)
        self._posts_end = self._posts_end.reshape(-1, 2)
        self._build()
        # Plain Python tuples rather than numpy rows: the search indexes these once per neighbour
        # of every expanded cell, where a numpy scalar would cost more than the arithmetic.
        self._step_offsets = [(int(row[0]), int(row[1])) for row in _NEIGHBOURS]
        self._step_cost = [
            float(np.hypot(offset_i, offset_j)) * self.config.resolution
            for offset_i, offset_j in self._step_offsets
        ]

    def _build(self) -> None:
        """Rasterize the poles and the gate posts into the occupancy array."""
        cfg = self.config
        low = np.asarray(cfg.bounds_low, dtype=float)
        high = np.asarray(cfg.bounds_high, dtype=float)
        self.shape = tuple(int(np.ceil(value)) for value in (high - low) / cfg.resolution)
        xs = low[0] + (np.arange(self.shape[0]) + 0.5) * cfg.resolution
        ys = low[1] + (np.arange(self.shape[1]) + 0.5) * cfg.resolution
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        occupied = np.zeros(self.shape, dtype=bool)
        if len(self._poles):
            distances = np.hypot(
                grid_x[:, :, None] - self._poles[None, None, :, 0],
                grid_y[:, :, None] - self._poles[None, None, :, 1],
            )
            occupied |= np.min(distances, axis=2) < self.pole_inflate
        for start, end in self._frame_segments:
            occupied |= (
                points_to_segment_distance(
                    np.stack([grid_x.ravel(), grid_y.ravel()], axis=1), start, end
                ).reshape(self.shape)
                < self.frame_pad
            )
        self.occupied = occupied

    def cell_of(self, point: NDArray[np.floating]) -> tuple[int, int]:
        """Return the grid cell that contains a point, clipped to the arena.

        Args:
            point: A point in the XY plane, shape ``(2,)``.

        Returns:
            The ``(i, j)`` index of the containing cell.
        """
        low = np.asarray(self.config.bounds_low, dtype=float)
        index = np.floor((np.asarray(point, dtype=float)[:2] - low) / self.config.resolution)
        return (
            int(np.clip(index[0], 0, self.shape[0] - 1)),
            int(np.clip(index[1], 0, self.shape[1] - 1)),
        )

    def cell_center(self, cell: tuple[int, int]) -> NDArray[np.floating]:
        """Return the XY position of a grid cell centre.

        Args:
            cell: The ``(i, j)`` index of the cell.

        Returns:
            The centre of the cell, shape ``(2,)``.
        """
        low = np.asarray(self.config.bounds_low, dtype=float)
        return low + (np.asarray(cell, dtype=float) + 0.5) * self.config.resolution

    def line_is_clear(self, start_xy: NDArray[np.floating], goal_xy: NDArray[np.floating]) -> bool:
        """Return whether the straight segment between two points clears every obstacle.

        This is the exact geometric test, not the rasterized one, so it is safe to use both as the
        trigger for a search and as the acceptance test of the path shortcut.

        Args:
            start_xy: First end of the segment, shape ``(2,)``.
            goal_xy: Second end of the segment, shape ``(2,)``.

        Returns:
            ``True`` if the segment keeps the required clearance everywhere.
        """
        start = np.asarray(start_xy, dtype=float)[:2]
        goal = np.asarray(goal_xy, dtype=float)[:2]
        if len(self._poles):
            minimum = points_to_segment_distance(self._poles, start, goal).min()
            if float(minimum) < self.pole_inflate:
                return False
        if len(self._posts_start):
            distances = segment_distances(start, goal, self._posts_start, self._posts_end)
            if float(distances.min()) < self.frame_pad:
                return False
        return True

    def search(
        self, start_xy: NDArray[np.floating], goal_xy: NDArray[np.floating]
    ) -> NDArray[np.floating] | None:
        """Find a clearance-keeping route between two points.

        Args:
            start_xy: Route start, shape ``(2,)``.
            goal_xy: Route goal, shape ``(2,)``.

        Returns:
            The simplified route as ``(n, 2)`` points from start to goal, or ``None`` if the goal is
            not reachable.
        """
        start = np.asarray(start_xy, dtype=float)[:2]
        goal = np.asarray(goal_xy, dtype=float)[:2]
        start_cell = self._nearest_free(start)
        goal_cell = self._nearest_free(goal)
        if start_cell is None or goal_cell is None:
            return None
        cells = self._astar(start_cell, goal_cell)
        if cells is None:
            return None
        points = np.array([self.cell_center(cell) for cell in cells], dtype=float)
        # Keep the requested endpoints so the caller's anchors stay exact, unless they are blocked.
        if self.line_is_clear(start, points[0]) and self._point_is_clear(start):
            points[0] = start
        if self.line_is_clear(points[-1], goal) and self._point_is_clear(goal):
            points[-1] = goal
        corners = _drop_collinear(points)
        return self._shortcut(corners) if self.config.shortcut else corners

    def _point_is_clear(self, point: NDArray[np.floating]) -> bool:
        """Return whether a single point keeps the required clearance."""
        return self.line_is_clear(point, point)

    def _nearest_free(self, point: NDArray[np.floating]) -> tuple[int, int] | None:
        """Return the closest free cell to a point, or ``None`` if the arena is exhausted."""
        cell = self.cell_of(point)
        if not self.occupied[cell]:
            return cell
        for radius in range(1, max(self.shape) + 1):
            for offset_i in range(-radius, radius + 1):
                for offset_j in range(-radius, radius + 1):
                    if max(abs(offset_i), abs(offset_j)) != radius:
                        continue
                    candidate = (cell[0] + offset_i, cell[1] + offset_j)
                    if not (0 <= candidate[0] < self.shape[0]):
                        continue
                    if not (0 <= candidate[1] < self.shape[1]):
                        continue
                    if not self.occupied[candidate]:
                        return candidate
        return None

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """Run A* between two free cells and return the cell path."""
        cost = np.full(self.shape, np.inf, dtype=np.float32)
        parent = np.full(self.shape + (2,), -1, dtype=np.int16)
        closed = np.zeros(self.shape, dtype=bool)
        occupied = self.occupied
        rows, columns = self.shape
        resolution = self.config.resolution
        goal_i, goal_j = float(goal[0]), float(goal[1])
        cost[start] = 0.0
        start_estimate = float(np.hypot(start[0] - goal_i, start[1] - goal_j) * resolution)
        queue: list[tuple[float, float, tuple[int, int]]] = [(start_estimate, 0.0, start)]
        while queue:
            _, accumulated, current = heapq.heappop(queue)
            if closed[current]:
                continue
            if current == goal:
                return self._trace(parent, start, goal)
            closed[current] = True
            for (offset_i, offset_j), length in zip(
                self._step_offsets, self._step_cost, strict=True
            ):
                neighbour = (current[0] + offset_i, current[1] + offset_j)
                if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < columns):
                    continue
                if occupied[neighbour] or closed[neighbour]:
                    continue
                candidate = accumulated + float(length)
                if candidate >= cost[neighbour]:
                    continue
                cost[neighbour] = candidate
                parent[neighbour] = current
                estimate = float(
                    np.hypot(neighbour[0] - goal_i, neighbour[1] - goal_j) * resolution
                )
                heapq.heappush(queue, (candidate + estimate, candidate, neighbour))
        return None

    @staticmethod
    def _trace(
        parent: NDArray[np.integer], start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Walk the parent pointers back and return the path from start to goal."""
        path = [goal]
        current = goal
        while current != start:
            current = (int(parent[current][0]), int(parent[current][1]))
            path.append(current)
        return path[::-1]

    def _shortcut(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Drop every point that can be skipped without losing the clearance.

        Args:
            points: The staircase path, shape ``(n, 2)``.

        Returns:
            The simplified path, always starting at ``points[0]`` and ending at ``points[-1]``.
        """
        if len(points) < 3:
            return points
        kept = [points[0]]
        index = 0
        while index < len(points) - 1:
            candidate = len(points) - 1
            while candidate > index + 1 and not self.line_is_clear(
                points[index], points[candidate]
            ):
                candidate -= 1
            kept.append(points[candidate])
            index = candidate
        return np.asarray(kept, dtype=float)


def _drop_collinear(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Drop the points that lie on the straight line through their neighbours.

    An 8-connected grid path is a staircase whose long runs are exactly collinear, so this removes
    most of it without changing its shape. Running the (quadratic) line-of-sight shortcut on the
    remaining corners instead of on the raw path is what keeps the simplification cheap.

    Args:
        points: The path, shape ``(n, 2)``.

    Returns:
        The path with only the corners and both endpoints kept, shape ``(m, 2)``.
    """
    if len(points) < 3:
        return points
    kept = [points[0]]
    for index in range(1, len(points) - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        turns = abs(_cross(incoming, outgoing)) > 1e-9
        reverses = float(incoming @ outgoing) <= 0.0
        if turns or reverses:
            kept.append(points[index])
    kept.append(points[-1])
    return np.asarray(kept, dtype=float)


def _gate_posts(
    gates_pos: NDArray[np.floating], gates_y_axes: NDArray[np.floating]
) -> list[tuple[NDArray[np.floating], NDArray[np.floating]]]:
    """Return the two side posts of every gate as XY segments.

    Args:
        gates_pos: Gate centres, shape ``(n_gates, 3)``.
        gates_y_axes: In-plane lateral axis of every gate, shape ``(n_gates, 3)``.

    Returns:
        One ``(start, end)`` segment pair per gate post.
    """
    segments: list[tuple[NDArray[np.floating], NDArray[np.floating]]] = []
    for centre, axis in zip(gates_pos, gates_y_axes, strict=True):
        lateral = np.asarray(axis, dtype=float)[:2]
        norm = float(np.linalg.norm(lateral))
        if norm < 1e-6:
            continue
        lateral = lateral / norm
        origin = np.asarray(centre, dtype=float)[:2]
        for sign in (1.0, -1.0):
            segments.append(
                (
                    origin + sign * GATE_OPENING_HALF * lateral,
                    origin + sign * GATE_FRAME_HALF * lateral,
                )
            )
    return segments
