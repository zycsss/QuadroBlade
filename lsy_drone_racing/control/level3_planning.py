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
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.astar import (
    GATE_FRAME_HALF,
    GATE_OPENING_HALF,
    ArenaGrid,
    AstarConfig,
)
from lsy_drone_racing.control.mpcc_level2_controller import ClearancePlanner
from lsy_drone_racing.control.path_planning import ArcPath, GateFrames, PathInfo
from lsy_drone_racing.control.track_geometry import frame_clearance

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
    backside_lateral: float = 0.0
    backside_depth: float = 0.35
    block_openings: bool = False
    detour_flip: bool = False


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


class FullFrameGrid(ArenaGrid):
    """A* grid whose gates are solid across their whole width, opening included.

    The level 2 grid models only the side posts, on the grounds that the drone crosses every gate
    plane at the opening height. On level 3 consecutive passes differ in height by up to 0.5 m, so a
    leg that threads the opening of a gate it does not pass crosses that plane at an intermediate
    height and hits the top or bottom bar or the stand below. The gates in ``open_gates`` keep
    the level 2 posts-only model: a leg may still turn back through the gate it has just left or
    is about to enter, which it crosses near that gate's own height.
    """

    def __init__(
        self,
        obstacles_pos: NDArray,
        gates_pos: NDArray,
        gates_y_axes: NDArray,
        config: AstarConfig | None = None,
        open_gates: frozenset[int] = frozenset(),
    ):
        """Rasterize poles and full-width gate frames."""
        self._open_gates = open_gates
        self._full_gates = (
            np.asarray(gates_pos, dtype=float).reshape(-1, 3),
            np.asarray(gates_y_axes, dtype=float).reshape(-1, 3),
        )
        super().__init__(obstacles_pos, gates_pos, gates_y_axes, config)

    def _build(self) -> None:
        """Replace the post segments by one segment across each frame, then rasterize."""
        segments = []
        for index, (centre, axis) in enumerate(zip(*self._full_gates, strict=True)):
            lateral = axis[:2]
            norm = float(np.linalg.norm(lateral))
            if norm < 1e-6:
                continue
            lateral = lateral / norm
            if index in self._open_gates:
                for sign in (1.0, -1.0):
                    segments.append(
                        (
                            centre[:2] + sign * GATE_OPENING_HALF * lateral,
                            centre[:2] + sign * GATE_FRAME_HALF * lateral,
                        )
                    )
                continue
            segments.append(
                (centre[:2] + GATE_FRAME_HALF * lateral, centre[:2] - GATE_FRAME_HALF * lateral)
            )
        self._frame_segments = segments
        self._posts_start = np.asarray([a for a, _ in segments], dtype=float).reshape(-1, 2)
        self._posts_end = np.asarray([b for _, b in segments], dtype=float).reshape(-1, 2)
        super()._build()


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
        self._flip = False
        self._flipped = False
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
        self._flip, self._flipped = False, False
        plain = self._plan_once(obs, start_pos)
        if not (self.settings.detour_flip and self._flipped):
            return plain
        # A turn-around waypoint outside the arena could go round the other side of its gate.
        # Which side is better depends on the rest of the track, so both are planned and the one
        # that keeps more distance from the gate frames is flown; ties keep the level 2 side.
        self._flip = True
        try:
            flipped = self._plan_once(obs, start_pos)
        finally:
            self._flip = False
        rot = R.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_matrix()

        def score(path: ArcPath) -> float:
            points = path.pos(np.linspace(0.0, path.length, max(2, int(path.length / 0.02))))
            return min(float(frame_clearance(points, gates, rot).min()), self.FLIP_SCORE_CAP)

        if score(flipped[0]) > score(plain[0]) + 0.02:
            return flipped
        return plain

    #: Frame clearance [m] above which a path counts as clear when choosing a turn-around side.
    FLIP_SCORE_CAP = 0.15

    def _plan_once(self, obs: dict, start_pos: NDArray) -> tuple[ArcPath, PathInfo]:
        """Plan like level 2; a failed final pole repair keeps the unrepaired path."""
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

    def _connect_blocks(
        self,
        blocks: list[NDArray],
        runout: NDArray,
        obstacles: NDArray,
        gates_pos: NDArray,
        gates_y_axes: NDArray,
    ) -> NDArray:
        """Join the blocks like level 2, on a grid that also blocks the gate openings."""
        if not self.settings.block_openings:
            return super()._connect_blocks(blocks, runout, obstacles, gates_pos, gates_y_axes)
        cfg = self.config
        groups = [np.asarray(block, dtype=float).reshape(-1, 3) for block in blocks]
        groups.append(np.asarray(runout, dtype=float).reshape(-1, 3))
        astar = AstarConfig(
            resolution=cfg.astar_resolution,
            pole_clearance=cfg.safety_distance,
            frame_margin=cfg.astar_frame_margin,
            overshoot=cfg.push_scale,
            bounds_low=cfg.astar_bounds[0],
            bounds_high=cfg.astar_bounds[1],
        )
        # Gate of every group: the pass blocks in order, -1 for the other waypoints.
        centres = np.asarray(gates_pos, dtype=float).reshape(-1, 3)
        pass_gates = []
        if self._passes is not None:
            for centre in self._passes[0]:
                pass_gates.append(int(np.argmin(np.linalg.norm(centres - centre, axis=1))))
        owner, k = [], 0
        for i, group in enumerate(groups):
            is_pass = i >= 2 and len(group) == cfg.num_intermediate_points and k < len(pass_gates)
            owner.append(pass_gates[k] if is_pass else -1)
            k += int(is_pass)
        grids: dict[frozenset[int], FullFrameGrid] = {}
        joined = [groups[0]]
        for i, (previous, current) in enumerate(zip(groups[:-1], groups[1:], strict=True)):
            start, goal = previous[-1], current[0]
            # The gates of the pass blocks this leg leaves and joins stay open (posts only).
            left = next((g for g in reversed(owner[: i + 1]) if g >= 0), -1)
            joins = next((g for g in owner[i + 1 :] if g >= 0), -1)
            key = frozenset(g for g in (left, joins) if g >= 0)
            if key not in grids:
                grids[key] = FullFrameGrid(obstacles, gates_pos, gates_y_axes, astar, key)
            grid = grids[key]
            span = float(np.linalg.norm((goal - start)[:2]))
            if span < cfg.astar_resolution or grid.line_meets_clearance(start[:2], goal[:2]):
                joined.append(current)
                continue
            leg = grid.search(start[:2], goal[:2])
            if leg is None or len(leg) < 2:
                logger.warning("No level 3 A* route between %s and %s", start, goal)
                joined.append(current)
                continue
            joined.append(self._lift_leg(leg, start, goal)[1:-1])
            joined.append(current)
        return np.concatenate(joined, axis=0)

    def _insert_detours(self, blocks, pass_pos, pass_normals, pass_y_axes, pass_z_axes):
        """Add the level 2 detours, then route around the frame of a gate entered from behind.

        When the point before a pass lies on the exit side of that gate (a re-pass, or a gate
        that faces away from the previous one), the route has to swing around the frame. Two
        side waypoints ``backside_lateral`` beside the centre, ``backside_depth`` in front of
        and behind the plane, on the side the route comes from, make the turn explicit instead
        of leaving a hairpin next to a post.
        """
        out = super()._insert_detours(blocks, pass_pos, pass_normals, pass_y_axes, pass_z_axes)
        if self.settings.detour_flip:
            out = self._flip_outside_detours(out, pass_pos, apply=self._flip)
        lateral = self.settings.backside_lateral
        if lateral <= 0:
            return out
        n_points = self.config.num_intermediate_points
        result = []
        pass_index = 0
        for i, block in enumerate(out):
            if i >= 2 and len(block) == n_points and pass_index < len(pass_pos):
                centre, normal = pass_pos[pass_index], pass_normals[pass_index]
                y_axis = pass_y_axes[pass_index]
                previous = result[-1][-1]
                rel = previous - centre
                if rel @ normal > -0.05:
                    side = 1.0 if rel @ y_axis >= 0 else -1.0
                    depth = self.settings.backside_depth
                    exit_side = centre + side * lateral * y_axis + depth * normal
                    entry_side = centre + side * lateral * y_axis - depth * normal
                    result.append(np.vstack([exit_side, entry_side]))
                pass_index += 1
            result.append(block)
        return result

    def _flip_outside_detours(self, out: list, pass_pos: NDArray, apply: bool) -> list:
        """Mirror a level 2 turn-around waypoint that lies outside the arena to the other side.

        The level 2 detour after a sharp turn goes to the side the chord leans to. Next to a wall
        that point can be outside the arena; the final clamp then flattens the loop onto the gate
        frame it was meant to clear. Mirroring the lateral offset through the gate centre turns the
        other way round the gate when the mirrored point lies deeper inside the arena. Without
        ``apply`` the blocks are returned unchanged and only ``_flipped`` records whether a
        candidate exists; [plan][...Level3Planner.plan] then compares both sides.
        """
        n_points = self.config.num_intermediate_points
        low = self._low[:2] + self.settings.bounds_margin
        high = self._high[:2] - self.settings.bounds_margin

        def room(point: NDArray) -> float:
            return float(min(np.min(point[:2] - low), np.min(high - point[:2])))

        result, pass_index = [], -1
        for i, block in enumerate(out):
            if i >= 2 and len(block) == n_points:
                pass_index += 1
            elif i >= 2 and len(block) == 1 and 0 <= pass_index < len(pass_pos):
                centre = pass_pos[pass_index]
                offset = block[0] - centre
                here, mirrored = room(block[0]), room(centre - offset)
                if abs(offset[2]) < 0.1 and here < 0 and mirrored > here:
                    self._flipped = True
                    if apply:
                        block = (centre - offset).reshape(1, 3)
            result.append(block)
        return result

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
