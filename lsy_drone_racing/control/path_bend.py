"""The path the controller follows: the nominal plan, bent to the poses the drone has seen.

The time-optimal plan is made once against the nominal track (`topt_planner`). In flight the sensor
reveals each gate and pole at its true pose, and instead of re-planning, the plan is *bent*: every
5 cm the nominal path carries a bead ``q[k]``, and a reveal computes one displacement ``d[k]`` per
bead, from scratch, from everything known so far. The displacement is the field with the least
bending energy that honours a set of pins,

    minimise  ||L d||^2 + sum_k w[k] * (d[k] - t[k])^2,

with ``L`` the second-difference operator - one symmetric five-banded system per axis, solved
directly. The pins are:

* the start bead, at the drone's real start;
* the beads within ``gate_window`` of every crossing, moved *rigidly* with their gate - fully
  rotated, so the passage stays square to the new plane, but translated only as far as needed to
  bring the crossing within ``gate_tolerance`` of the revealed centre: every centimetre the bend
  does not move the aim is lateral correction the drone does not have to fly;
* the beads under the drone, held where they are, so the path never moves beneath it;
* soft pushes away from every pole and every gate capsule closer than the clearance, the worst bead
  per obstacle per round, until nothing is violated. Gate windows and the beads under the drone are
  never pushed, so a pass always wins over an obstacle.

Because bead ``k`` of one bend is bead ``k`` of the next, the path parameter carries over by
interpolating the two arc-length tables, and the controller never loses its place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.linalg import solveh_banded
from scipy.optimize import minimize_scalar

from lsy_drone_racing.control.track_geometry import gate_capsules, segment_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.topt_planner import Trajectory


class ArcPath:
    """A cubic spline through waypoints, parameterized by arc length."""

    def __init__(self, waypoints: NDArray[np.floating], grid: float = 0.01):
        """Fit the spline and tabulate it.

        Args:
            waypoints: Points ``(m, 3)``, consecutive ones distinct.
            grid: Spacing of the dense table used for projection [m].
        """
        waypoints = np.asarray(waypoints, dtype=float)
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        if len(seg) == 0 or np.any(seg < 1e-6):
            raise ValueError("A path needs distinct consecutive waypoints")
        knots = np.r_[0.0, np.cumsum(seg)]
        raw = CubicSpline(knots, waypoints)
        u = np.unique(
            np.r_[np.linspace(0.0, knots[-1], max(1001, int(knots[-1] / 0.0025) + 1)), knots]
        )
        speed = np.linalg.norm(raw.derivative()(u), axis=1)
        if np.min(speed) < 1e-7:
            raise ValueError("The path has a stationary point")
        arc = np.r_[0.0, cumulative_trapezoid(speed, u)]
        self._spline = CubicSpline(arc, raw(u))
        self._d1 = self._spline.derivative()
        self.length = float(arc[-1])
        #: Arc length of every waypoint.
        self.knot_s = np.interp(knots, u, arc)
        #: Dense arc-length table and its points.
        self.ss = np.r_[np.arange(0.0, self.length, grid), self.length]
        self.pts = self._spline(self.ss)

    def __call__(self, s: float | NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the point(s) at arc length ``s``."""
        return self._spline(np.clip(s, 0.0, self.length))

    def tangent(self, s: float | NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the unit tangent at arc length ``s`` (one per value)."""
        d = self._d1(np.clip(s, 0.0, self.length))
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)

    def project(
        self, point: NDArray[np.floating], s_guess: float, back: float, ahead: float
    ) -> float:
        """Project a point onto the path, searching only ``[s_guess - back, s_guess + ahead]``.

        The window keeps the projection local: where the path passes close to itself (a hairpin
        or a revisited gate) a global search would jump to the other branch.
        """
        lo = float(np.clip(s_guess - back, 0.0, self.length))
        hi = float(np.clip(s_guess + ahead, lo, self.length))
        if hi - lo < 1e-10:
            return lo
        i0, i1 = np.searchsorted(self.ss, lo), np.searchsorted(self.ss, hi, side="right")
        grid = np.r_[lo, self.ss[i0:i1], hi]
        sq = np.sum((self(grid) - point) ** 2, axis=1)
        k = int(np.argmin(sq))
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        if right - left < 1e-10:
            return float(grid[k])
        res = minimize_scalar(
            lambda s: float(np.sum((self(s) - point) ** 2)),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5},
        )
        return float(res.x if res.fun < sq[k] else grid[k])


@dataclass(frozen=True)
class BendConfig:
    """Options of the path bending.

    Attributes:
        spacing: Distance between beads [m].
        run_out: Straight extension behind the last gate [m].
        gate_window: Beads within this arc length of a crossing move rigidly with the gate [m].
        gate_tolerance: A revealed crossing is moved only once it is more than this far from the
            gate centre, and then only to this distance [m].
        crossing_height: Height of the aimed crossing above the gate centre [m].
        pole_clearance: Horizontal distance from a pole axis the bend restores [m].
        frame_clearance: Distance from a gate capsule's surface the bend restores [m].
        iterations: Rounds of obstacle pushes.
    """

    spacing: float = 0.05
    run_out: float = 1.0
    gate_window: float = 0.10
    gate_tolerance: float = 0.04
    crossing_height: float = 0.01
    pole_clearance: float = 0.17
    frame_clearance: float = 0.11
    iterations: int = 8


class PathBender:
    """Holds the nominal plan as beads and bends it to the poses revealed so far."""

    HARD = 1e8
    SOFT = 1e4

    def __init__(
        self,
        trajectory: Trajectory,
        gates_pos: NDArray[np.floating],
        gates_rot: NDArray[np.floating],
        sequence: NDArray[np.integer],
        config: BendConfig | None = None,
    ):
        """Resample the plan into beads.

        Args:
            trajectory: The time-optimal plan.
            gates_pos: The gate centres the plan was made against.
            gates_rot: The gate rotations the plan was made against.
            sequence: Gate index of every pass.
            config: Bending options.
        """
        self.config = config or BendConfig()
        cfg = self.config
        pos, vel = trajectory.pos, trajectory.vel
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-6]
        index = np.cumsum(keep) - 1
        pos, vel, arc = pos[keep], vel[keep], arc[keep]
        crossing_s = arc[index[trajectory.crossings]]
        spline = CubicSpline(arc, pos)
        s_nom = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / cfg.spacing)) + 1)
        beads = spline(s_nom)
        speed = np.interp(s_nom, arc, np.linalg.norm(vel, axis=1))
        tangent = spline(arc[-1], 1)
        tangent /= np.linalg.norm(tangent)
        extra = cfg.spacing * np.arange(1, int(np.ceil(cfg.run_out / cfg.spacing)) + 1)
        beads = np.r_[beads, beads[-1] + extra[:, None] * tangent]
        beads[0] = pos[0]
        #: Nominal beads ``(m, 3)``.
        self.beads = beads
        #: Planned speed at every bead ``(m,)``.
        self.speed = np.r_[speed, np.full(len(extra), speed[-1])]
        #: Bead of every crossing.
        step = s_nom[1] - s_nom[0]
        self.crossings = np.clip(np.round(crossing_s / step).astype(int), 0, len(beads) - 1)
        self.sequence = np.asarray(sequence, dtype=int)
        self._plan_pos = np.asarray(gates_pos, dtype=float).copy()
        self._plan_rot = np.asarray(gates_rot, dtype=float).copy()
        #: Current displacement of every bead.
        self.displacement = np.zeros_like(beads)
        self._banded = self._bending_matrix(len(beads))

    @staticmethod
    def _bending_matrix(m: int) -> NDArray[np.floating]:
        """Return ``L^T L`` (plus a tiny ridge) in the upper banded form of `solveh_banded`."""
        diag = np.full(m, 6.0)
        diag[[0, -1]], diag[[1, -2]] = 1.0, 5.0
        off1 = np.full(m - 1, -4.0)
        off1[[0, -1]] = -2.0
        off2 = np.ones(m - 2)
        banded = np.zeros((3, m))
        banded[0, 2:], banded[1, 1:], banded[2] = off2, off1, diag + 1e-6
        return banded

    def _solve(
        self, weight: NDArray[np.floating], target: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        out = np.empty_like(target)
        for axis in range(3):
            ab = self._banded.copy()
            ab[2] += weight[:, axis]
            out[:, axis] = solveh_banded(ab, weight[:, axis] * target[:, axis])
        return out

    def bend(
        self,
        gates_pos: NDArray[np.floating],
        gates_rot: NDArray[np.floating],
        poles_pos: NDArray[np.floating],
        start: NDArray[np.floating],
        drone_bead: int | None,
    ) -> NDArray[np.floating]:
        """Bend the nominal beads to the given poses.

        Args:
            gates_pos: Gate centres as observed now.
            gates_rot: Gate rotations as observed now.
            poles_pos: Pole tops as observed now.
            start: The drone's real start position.
            drone_bead: Bead the drone is at, or None before take-off; the beads around it are held
                and everything up to it is exempt from pushes.

        Returns:
            The bent beads ``(m, 3)``.
        """
        cfg, q = self.config, self.beads
        m = len(q)
        weight, target = np.zeros((m, 3)), np.zeros((m, 3))
        locked = np.zeros(m, dtype=bool)

        def pin(
            k: int,
            value: NDArray[np.floating],
            w: float,
            mask: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ):
            weight[k] = w * np.asarray(mask)
            target[k] = value

        pin(0, np.asarray(start, dtype=float) - q[0], self.HARD)
        locked[0] = True
        half = int(round(cfg.gate_window / cfg.spacing))
        for g, k_c in zip(self.sequence, self.crossings, strict=True):
            rot_rel = gates_rot[g] @ self._plan_rot[g].T
            moved = rot_rel @ (q[k_c] - self._plan_pos[g]) + gates_pos[g]
            unmoved_local = gates_rot[g].T @ (q[k_c] - gates_pos[g])
            moved_local = gates_rot[g].T @ (moved - gates_pos[g])
            tol = cfg.gate_tolerance
            wanted_local = np.array(
                [
                    moved_local[0],
                    np.clip(unmoved_local[1], -tol, tol),
                    np.clip(unmoved_local[2], cfg.crossing_height - tol, cfg.crossing_height + tol),
                ]
            )
            shift = gates_rot[g] @ (wanted_local - moved_local)
            for k in range(max(k_c - half, 1), min(k_c + half, m - 1) + 1):
                pin(
                    k, rot_rel @ (q[k] - self._plan_pos[g]) + gates_pos[g] + shift - q[k], self.HARD
                )
                locked[k] = True
        if drone_bead is not None:
            for k in range(max(drone_bead - 1, 1), min(drone_bead + 2, m - 1) + 1):
                if not locked[k]:
                    pin(k, self.displacement[k], self.HARD)
            locked[: drone_bead + 3] = True

        capsules = gate_capsules(gates_pos, gates_rot)
        disp = self._solve(weight, target)
        for _ in range(cfg.iterations):
            pts = q + disp
            changed = False
            for pole in poles_pos:
                d = np.linalg.norm(pts[:, :2] - pole[:2], axis=1)
                violation = np.where(locked, -1.0, cfg.pole_clearance - d)
                k = int(np.argmax(violation))
                if violation[k] > 0.005:
                    away = (pts[k, :2] - pole[:2]) / max(d[k], 1e-6)
                    goal = np.r_[pole[:2] + (cfg.pole_clearance + 0.01) * away, 0.0]
                    pin(k, goal - q[k], self.SOFT, (1.0, 1.0, 0.0))
                    changed = True
            for _, a, b, radius in capsules:
                d, closest = segment_distance(pts, a, b)
                need = radius + cfg.frame_clearance
                violation = np.where(locked, -1.0, need - d)
                k = int(np.argmax(violation))
                if violation[k] > 0.005:
                    away = (pts[k] - closest[k]) / max(d[k], 1e-6)
                    pin(k, closest[k] + (need + 0.01) * away - q[k], self.SOFT)
                    changed = True
            if not changed:
                break
            disp = self._solve(weight, target)
        self.displacement = disp
        return q + disp
