"""Collision geometry of the race track.

The gate frames, the openings and the poles are fixed by the rules of the challenge, and the
planner, the path bending, the controller and the figures all need their dimensions. They live here
so that the numbers keep one home and the drawing code does not have to import a planner.

The collision shapes are those of the simulator. `envs/assets/gate.xml` gives a gate four collision
boxes - top, bottom, left and right bar, each 0.16 m deep in the plane and 0.02 m thick - plus a
stand below the bottom bar, and `envs/assets/obstacle.xml` makes a pole a capsule of radius 0.015 m
from the reported top position down to the floor. The planner treats every bar as a capsule along
its centre line, whose radius covers the bar's half depth.

The drone's own size is not a parameter of the simulator model: `crazyflow/drones/cf21B_500.xml`
gives it a box with half extents 0.07 x 0.07 x 0.02, so its reach is ``hypot(0.07, 0.07) = 0.099 m``
when a corner is presented, plus up to ``0.02 * sin(tilt)`` from its half height. `DRONE_RADIUS` is
that worst case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Reach of the drone's collision geometry [m].
DRONE_RADIUS = 0.107
#: Half width of the gate opening [m]. The frame is square with a 0.4 m wide opening.
GATE_OPENING_HALF = 0.20
#: Half width of the outer edge of the gate frame [m]. The frame is 0.72 m wide overall.
GATE_FRAME_HALF = 0.36
#: Radius of the cylindrical obstacles [m]. The obstacles are 0.03 m in diameter.
POLE_RADIUS = 0.015

#: Collision boxes of a gate in its own frame (x along the passing direction): centre, half extents.
GATE_BOXES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.28), (0.01, 0.36, 0.08)),  # top bar
    ((0.0, 0.0, -0.28), (0.01, 0.36, 0.08)),  # bottom bar
    ((0.0, -0.28, 0.0), (0.01, 0.08, 0.36)),  # left bar
    ((0.0, 0.28, 0.0), (0.01, 0.08, 0.36)),  # right bar
    ((0.0, 0.0, -0.86), (0.05, 0.05, 0.50)),  # stand
)
#: The same parts as capsules for the planner and the path bending: end points and radius. A bar's
#: capsule runs along its centre line between the corners, with the bar's half depth as radius; the
#: corners of the frame are then covered by the overlap of two capsules.
GATE_CAPSULES: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...] = (
    ((0.0, -0.28, 0.28), (0.0, 0.28, 0.28), 0.08),  # top bar
    ((0.0, -0.28, -0.28), (0.0, 0.28, -0.28), 0.08),  # bottom bar
    ((0.0, -0.28, -0.28), (0.0, -0.28, 0.28), 0.08),  # left bar
    ((0.0, 0.28, -0.28), (0.0, 0.28, 0.28), 0.08),  # right bar
    ((0.0, 0.0, -0.36), (0.0, 0.0, -1.36), 0.07),  # stand
)


def gate_capsules(
    gates_pos: NDArray[np.floating], gates_rot: NDArray[np.floating]
) -> list[tuple[int, NDArray[np.floating], NDArray[np.floating], float]]:
    """Return every gate's frame and stand as world-frame capsules.

    Args:
        gates_pos: Gate centres, shape ``(n_gates, 3)``.
        gates_rot: Gate rotation matrices, shape ``(n_gates, 3, 3)``; column 0 is the passing axis.

    Returns:
        ``(gate index, end a, end b, radius)`` for every part of every gate.
    """
    capsules = []
    for g, (centre, rot) in enumerate(zip(gates_pos, gates_rot, strict=True)):
        for a, b, radius in GATE_CAPSULES:
            capsules.append((g, centre + rot @ np.asarray(a), centre + rot @ np.asarray(b), radius))
    return capsules


def segment_distance(
    points: NDArray[np.floating], a: NDArray[np.floating], b: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return the distance of points to the segment ``a``-``b`` and the closest segment points.

    Args:
        points: Query points, shape ``(n, 3)``.
        a: Start of the segment, shape ``(3,)``.
        b: End of the segment, shape ``(3,)``.

    Returns:
        The distances ``(n,)`` and the closest points on the segment ``(n, 3)``.
    """
    d = b - a
    t = np.clip((points - a) @ d / float(d @ d), 0.0, 1.0)
    closest = a + t[:, None] * d
    return np.linalg.norm(points - closest, axis=1), closest


def frame_clearance(
    points: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_rot: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Return the distance of points to the nearest gate collision box (frame or stand).

    Args:
        points: Query points, shape ``(n, 3)``.
        gates_pos: Gate centres, shape ``(n_gates, 3)``.
        gates_rot: Gate rotation matrices, shape ``(n_gates, 3, 3)``.

    Returns:
        The distance ``(n,)``; zero inside a box.
    """
    best = np.full(len(points), np.inf)
    for centre, rot in zip(gates_pos, gates_rot, strict=True):
        local = (points - centre) @ rot
        for c, h in GATE_BOXES:
            outside = np.maximum(np.abs(local - np.asarray(c)) - np.asarray(h), 0.0)
            best = np.minimum(best, np.linalg.norm(outside, axis=1))
    return best


def pole_clearance(
    points: NDArray[np.floating], poles_pos: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Return the horizontal distance of points to the nearest pole surface.

    The poles reach from the floor to their reported top, which is above the flight envelope, so the
    horizontal distance to the axis is the whole story.

    Args:
        points: Query points, shape ``(n, 3)``.
        poles_pos: Pole tops, shape ``(n_poles, 3)``.

    Returns:
        The distance ``(n,)`` to the nearest pole surface.
    """
    if len(poles_pos) == 0:
        return np.full(len(points), np.inf)
    d = np.linalg.norm(points[:, None, :2] - np.asarray(poles_pos)[None, :, :2], axis=2)
    return d.min(axis=1) - POLE_RADIUS
