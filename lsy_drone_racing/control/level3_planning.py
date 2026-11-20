"""Level 3 extensions of the level 2 A* path planner.

Level 3 generates a new track layout per episode: gates can face any direction, gates passed twice
can be approached from behind, and the route between two passes often has to turn around next to a
gate it does not pass. The level 2 planner was tuned on one fixed layout and keeps only its own
pass windows clear of the frames. This module adds the level 3 rules on top of it without touching
the level 2 code path; only [MPCCLevel3Controller][...mpcc_level3_controller] instantiates it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.astar import GATE_FRAME_HALF, GATE_OPENING_HALF
from lsy_drone_racing.control.mpcc_level2_controller import ClearancePlanner
from lsy_drone_racing.control.path_planning import ArcPath, GateFrames, PathInfo

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Level3Settings:
    """Level 3 only planner settings.

    Attributes:
        post_clearance: XY distance [m] the smoothed path keeps from every gate post segment. Zero
            disables the post push-out.
    """

    post_clearance: float = 0.0
    bounds_margin: float = 0.0
    floor: float = 0.0
    ceiling_margin: float = 0.2
    cusp_angle_deg: float = 0.0
    cusp_chord: float = 0.15
    relax_iterations: int = 200
    gate_protect: float = 0.25


def gate_post_segments(gates_pos: NDArray, gates_quat: NDArray) -> tuple[NDArray, NDArray]:
    """Return the XY start and end points of the two side posts of every gate."""
    frames = GateFrames.from_quaternions(np.asarray(gates_quat, dtype=float))
    starts, ends = [], []
    for centre, axis in zip(np.asarray(gates_pos, dtype=float), frames.y_axes, strict=True):
        lateral = axis[:2] / max(float(np.linalg.norm(axis[:2])), 1e-9)
        for sign in (1.0, -1.0):
            starts.append(centre[:2] + sign * GATE_OPENING_HALF * lateral)
            ends.append(centre[:2] + sign * GATE_FRAME_HALF * lateral)
    return np.asarray(starts).reshape(-1, 2), np.asarray(ends).reshape(-1, 2)


def _closest_on_segments(points: NDArray, starts: NDArray, ends: NDArray) -> tuple[NDArray, NDArray]:
    """Return, per point, the nearest point on any segment and its distance."""
    seg = ends - starts
    length_sq = np.maximum(np.sum(seg**2, axis=1), 1e-12)
    rel = points[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(rel * seg[None], axis=-1) / length_sq[None], 0.0, 1.0)
    nearest = starts[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(points[:, None, :] - nearest, axis=-1)
    index = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    return nearest[rows, index], dist[rows, index]


class Level3Planner(ClearancePlanner):
    """The level 2 planner plus the level 3 frame and turn-around rules."""

    def __init__(
        self,
        config: object,
        radius: float,
        settings: Level3Settings,
        safety_low: NDArray | None = None,
        safety_high: NDArray | None = None,
    ):
        """Configure the level 2 planner and the level 3 settings."""
        super().__init__(config, radius)
        self.settings = settings
        self._posts: tuple[NDArray, NDArray] | None = None
        self._passes: tuple[NDArray, NDArray] | None = None
        self._poles = np.zeros((0, 3))
        self._low = np.asarray(safety_low if safety_low is not None else (-2.5, -1.5, 0), float)
        self._high = np.asarray(safety_high if safety_high is not None else (2.5, 1.5, 2), float)

    def plan(self, obs: dict, start_pos: NDArray) -> tuple[ArcPath, PathInfo]:
        """Plan like level 2; a failed final pole repair keeps the unrepaired path."""
        self._posts = gate_post_segments(obs["gates_pos"], obs["gates_quat"])
        frames = GateFrames.from_quaternions(np.asarray(obs["gates_quat"], dtype=float))
        sequence = np.asarray(obs["gate_sequence"], dtype=int).ravel()
        direction = np.sign(np.asarray(obs["gate_sequence_direction"], dtype=float)).ravel()
        direction = np.where(direction == 0, 1.0, direction)
        gates = np.asarray(obs["gates_pos"], dtype=float).reshape(-1, 3)
        self._passes = (gates[sequence], frames.normals[sequence] * direction[:, None])
        self._poles = np.asarray(obs["obstacles_pos"], dtype=float).reshape(-1, 3)
        try:
            return super().plan(obs, start_pos)
        except ValueError as exc:
            logger.warning("Level 3 pole repair failed (%s); keeping the A* path", exc)
            radius, self.radius = self.radius, 0.0
            try:
                return super().plan(obs, start_pos)
            finally:
                self.radius = radius

    def _smooth_path(self, points: NDArray, obstacles: NDArray) -> NDArray:
        """Smooth like level 2, additionally keeping the gate posts at ``post_clearance``."""
        clearance = self.settings.post_clearance
        if clearance <= 0 or self._posts is None or not len(self._posts[0]):
            return super()._smooth_path(points, obstacles)
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if len(pts) < 3:
            return pts
        starts, ends = self._posts
        radius = self.config.safety_distance * 0.92
        poles = np.asarray(obstacles, dtype=float).reshape(-1, 3)
        for _ in range(self.config.smooth_iterations):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
            if len(poles):
                delta = pts[:, None, :2] - poles[None, :, :2]
                dist = np.linalg.norm(delta, axis=-1)
                nearest = np.argmin(dist, axis=1)
                rows = np.arange(len(pts))
                close = dist[rows, nearest] < radius
                if close.any():
                    direction = delta[rows, nearest]
                    span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
                    pts[close, :2] = (
                        poles[nearest[close], :2] + radius * direction[close] / span[close]
                    )
            nearest_xy, dist = _closest_on_segments(pts[:, :2], starts, ends)
            close = dist < clearance
            close[[0, -1]] = False
            if close.any():
                direction = pts[close, :2] - nearest_xy[close]
                span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
                pts[close, :2] = nearest_xy[close] + clearance * direction / span
        return pts

    # region level 3 final pass

    def _to_arc_path(self, waypoints: NDArray) -> ArcPath:
        """Keep the finished path inside the arena and round its cusps, then fit it."""
        pts = np.asarray(waypoints, dtype=float).reshape(-1, 3).copy()
        st = self.settings
        if st.bounds_margin > 0 or st.floor > 0 or st.cusp_angle_deg > 0:
            pts = self._final_pass(pts)
        return super()._to_arc_path(pts)

    def _protected(self, pts: NDArray) -> NDArray:
        """Mark the samples of every intended gate passage, which must not move."""
        mask = np.zeros(len(pts), dtype=bool)
        mask[[0, -1]] = True
        if self._passes is None:
            return mask
        half = self.settings.gate_protect
        for centre, normal in zip(*self._passes, strict=True):
            rel = pts - centre
            along = rel @ normal
            lateral = np.linalg.norm(rel - along[:, None] * normal, axis=1)
            mask |= (np.abs(along) < half) & (lateral < 0.3)
        return mask

    def _clamp(self, pts: NDArray) -> NDArray:
        st = self.settings
        low = self._low.copy()
        high = self._high.copy()
        if st.bounds_margin > 0:
            low[:2] += st.bounds_margin
            high[:2] -= st.bounds_margin
        else:
            low[:2], high[:2] = -np.inf, np.inf
        low[2] = st.floor if st.floor > 0 else -np.inf
        high[2] = high[2] - st.ceiling_margin if st.bounds_margin > 0 else np.inf
        return np.clip(pts, low, high)

    def _cusps(self, pts: NDArray) -> NDArray:
        """Mark samples where the heading over ``cusp_chord`` turns more than the cusp angle."""
        st = self.settings
        mask = np.zeros(len(pts), dtype=bool)
        if st.cusp_angle_deg <= 0 or len(pts) < 5:
            return mask
        spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        k = max(1, int(round(st.cusp_chord / max(spacing, 1e-4))))
        if len(pts) <= 2 * k:
            return mask
        a = pts[k:-k] - pts[: -2 * k]
        b = pts[2 * k :] - pts[k:-k]
        cos = np.sum(a * b, axis=1) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
        )
        sharp = cos < np.cos(np.deg2rad(st.cusp_angle_deg))
        idx = np.flatnonzero(sharp) + k
        for i in idx:
            mask[max(0, i - 2 * k) : i + 2 * k + 1] = True
        return mask

    def _final_pass(self, pts: NDArray) -> NDArray:
        """Clamp to the arena and relax cusps locally, keeping gate passages and pole clearance."""
        st = self.settings
        protected = self._protected(pts)
        radius = self.config.safety_distance
        for _ in range(st.relax_iterations):
            clamped = self._clamp(pts)
            moved = np.any(np.abs(clamped - pts) > 1e-9, axis=1)
            pts = np.where(protected[:, None], pts, clamped)
            active = (self._cusps(pts) | moved) & ~protected
            if not active.any():
                break
            # Grow the active set a little so the relaxation blends into the untouched path.
            grown = np.convolve(active.astype(float), np.ones(9), "same") > 0
            grown &= ~protected
            new = pts.copy()
            new[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
            pts = np.where(grown[:, None], new, pts)
            if len(self._poles):
                delta = pts[:, None, :2] - self._poles[None, :, :2]
                dist = np.linalg.norm(delta, axis=-1)
                nearest = np.argmin(dist, axis=1)
                rows = np.arange(len(pts))
                close = (dist[rows, nearest] < radius) & grown
                if close.any():
                    direction = delta[rows, nearest][close]
                    span = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
                    pts[close, :2] = self._poles[nearest[close], :2] + radius * direction / span
        clamped = self._clamp(pts)
        return np.where(protected[:, None], pts, clamped)

    # endregion
