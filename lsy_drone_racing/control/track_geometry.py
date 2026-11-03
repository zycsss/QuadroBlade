"""Collision geometry of the race track.

The gate frames, the openings and the poles are fixed by the rules of the challenge, and both the
planner and the figures need their dimensions. They live here rather than in a planner module so
that the drawing code does not have to import a planner, and so that the numbers keep one home.

The drone's own size is not a parameter of the simulator model:
`crazyflow/drones/cf21B_500.xml`
gives it a box with half extents 0.07 x 0.07 x 0.02, so its horizontal reach is 0.070 m when a face
is presented and ``hypot(0.07, 0.07) = 0.099 m`` when a corner is, plus up to ``0.02 * sin(tilt)``
from the half height leaning into the plane, about 0.008 m at the 24 degrees the drone holds in a
turn. [DRONE_RADIUS][lsy_drone_racing.control.track_geometry.DRONE_RADIUS] is the worst case of
that, so anything derived from it is conservative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Horizontal reach of the drone's collision geometry [m].
DRONE_RADIUS = 0.107
#: Half width of the gate opening [m]. The frame is square with a 0.4 m wide opening.
GATE_OPENING_HALF = 0.20
#: Half width of the outer edge of the gate frame [m]. The frame is 0.72 m wide overall.
GATE_FRAME_HALF = 0.36
#: Radius of the cylindrical obstacles [m]. The obstacles are 0.03 m in diameter.
POLE_RADIUS = 0.015


def gate_posts(
    gates_pos: NDArray[np.floating], gates_y_axes: NDArray[np.floating]
) -> list[tuple[NDArray[np.floating], NDArray[np.floating]]]:
    """Return the two side posts of every gate as XY segments.

    The frame of a gate is solid from the edge of the opening to the outer edge, so the two side
    bars are the only part of it a route can hit at the altitude of a passage. The point-mass
    planner uses these segments to keep away from a frame the drone is not passing.

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
