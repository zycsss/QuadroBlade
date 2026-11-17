"""Build random tracks out of random local scenes, to train a policy that has never seen a level.

Every leg is sampled on its own: how long it is, how sharply it turns away from the last gate, how
sharply it enters the next one and which poles stand near it. The result has the race's format, so
that `topt_planner` solves it unchanged, and the repeated gate of the race's order gives the scenes
that approach a gate from the side or from behind by itself.

The ranges span levels 2 and 3 together, because neither contains the other: level 3 keeps poles at
least 0.8 m from gates, and level 2 always has one closer than that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ARENA_LOW = np.array([-2.5, -1.5])
ARENA_HIGH = np.array([2.5, 1.5])
#: How far gates, poles and the start stay from the arena's walls [m].
GATE_BORDER = 0.3
START_BORDER = 0.5
GATES = 4
POLES = 4
POLE_TOP = 1.55
GATE_HEIGHTS = (0.7, 1.2)
GATE_HEIGHT_NOISE = 0.1
#: Lengths of the first leg, which level 2 makes short, and of the others [m].
FIRST_LEG = (0.6, 1.5, 3.0)
LEG = (0.9, 1.8, 4.4)
#: Turn into a gate [rad]: most are moderate, some sharp.
INTO_COMMON = np.radians(60.0)
INTO_SHARP = np.radians(130.0)
SHARP_SHARE = 0.15
#: Least distance between gates and from the start to a pole [m].
GATE_GAP = 0.9
START_POLE_GAP = 0.4
POLE_GAP = 0.3
#: Half width of a gate's frame, and how far a pole must stay from it [m].
FRAME_HALF = 0.36
POLE_FRAME_GAP = 0.12
#: Where a pole may stand relative to a gate and to a route [m].
POLE_TO_GATE = (0.3, 0.9)
POLE_TO_ROUTE = 0.6
#: Chance that a pole is placed near a gate, or near a route; otherwise anywhere.
NEAR_GATE_SHARE = 0.45
NEAR_ROUTE_SHARE = 0.35
#: How often one piece is sampled again before the whole track starts over.
TRIES = 50


class Layout(NamedTuple):
    """A generated track.

    Attributes:
        start: Drone start on the floor ``(3,)``.
        gates_pos: Gate centres ``(4, 3)``.
        gates_yaw: Gate headings ``(4,)``; the race's first gate axis points this way.
        poles: Pole tops ``(4, 3)``.
    """

    start: NDArray
    gates_pos: NDArray
    gates_yaw: NDArray
    poles: NDArray


def _inside(xy: NDArray, border: float) -> bool:
    """Return whether a point keeps its distance from the arena's walls."""
    return bool(np.all(xy >= ARENA_LOW + border) and np.all(xy <= ARENA_HIGH - border))


def _unit(angle: float) -> NDArray:
    return np.array([np.cos(angle), np.sin(angle)])


def _sample_gates(rng: np.random.Generator, start: NDArray) -> tuple[NDArray, NDArray] | None:
    """Return gate positions and headings in x and y, or None if no room was found."""
    gates, yaws = [], []
    previous, axis = start, rng.uniform(-np.pi, np.pi)
    for i in range(GATES):
        for _ in range(TRIES):
            length = rng.triangular(*(FIRST_LEG if i == 0 else LEG))
            away = axis + rng.choice((-1.0, 1.0)) * rng.uniform(0.0, np.pi)
            xy = previous + length * _unit(away)
            clear = all(np.linalg.norm(xy - g) >= GATE_GAP for g in gates)
            if _inside(xy, GATE_BORDER) and clear:
                break
        else:
            return None
        limit = INTO_SHARP if rng.random() < SHARP_SHARE else INTO_COMMON
        axis = away + rng.choice((-1.0, 1.0)) * rng.uniform(0.0, limit)
        gates.append(xy)
        yaws.append(axis)
        previous = xy
    return np.array(gates), np.array(yaws)


def _frame_distance(xy: NDArray, gate: NDArray, yaw: float) -> float:
    """Return the distance in x and y from a point to a gate's frame, seen as a segment."""
    side = _unit(yaw + np.pi / 2)
    along = np.clip((xy - gate) @ side, -FRAME_HALF, FRAME_HALF)
    return float(np.linalg.norm(xy - gate - along * side))


def _sample_pole(
    rng: np.random.Generator, start: NDArray, gates: NDArray, stops: NDArray
) -> NDArray:
    """Return a candidate pole in x and y: near a gate, near a route, or anywhere."""
    kind = rng.random()
    if kind < NEAR_GATE_SHARE:
        gate = gates[rng.integers(GATES)]
        return gate + rng.uniform(*POLE_TO_GATE) * _unit(rng.uniform(-np.pi, np.pi))
    if kind < NEAR_GATE_SHARE + NEAR_ROUTE_SHARE:
        leg = rng.integers(len(stops) - 1)
        a, b = stops[leg], stops[leg + 1]
        direction = (b - a) / np.linalg.norm(b - a)
        side = np.array([-direction[1], direction[0]])
        offset = rng.uniform(-POLE_TO_ROUTE, POLE_TO_ROUTE)
        return a + rng.uniform(0.0, 1.0) * (b - a) + offset * side
    return rng.uniform(ARENA_LOW + GATE_BORDER, ARENA_HIGH - GATE_BORDER)


def _sample_poles(
    rng: np.random.Generator, start: NDArray, gates: NDArray, yaws: NDArray
) -> NDArray | None:
    """Return pole positions in x and y that leave every frame free, or None if none fit."""
    stops = np.vstack([start, gates, gates[1]])
    poles: list[NDArray] = []
    for _ in range(POLES):
        for _ in range(TRIES):
            xy = _sample_pole(rng, start, gates, stops)
            framed = all(
                _frame_distance(xy, g, y) >= POLE_FRAME_GAP
                for g, y in zip(gates, yaws, strict=True)
            )
            near = np.linalg.norm(gates - xy, axis=1).min() >= POLE_TO_GATE[0]
            apart = all(np.linalg.norm(xy - p) >= POLE_GAP for p in poles)
            if (
                _inside(xy, GATE_BORDER)
                and framed
                and near
                and apart
                and np.linalg.norm(xy - start) >= START_POLE_GAP
            ):
                poles.append(xy)
                break
        else:
            return None
    return np.array(poles)


def sample_layout(rng: np.random.Generator) -> Layout:
    """Return a random track built from random local scenes."""
    while True:
        start = rng.uniform(ARENA_LOW + START_BORDER, ARENA_HIGH - START_BORDER)
        sampled = _sample_gates(rng, start)
        if sampled is None:
            continue
        gates, yaws = sampled
        poles = _sample_poles(rng, start, gates, yaws)
        if poles is None:
            continue
        heights = rng.choice(GATE_HEIGHTS, GATES)
        heights = heights + rng.uniform(-GATE_HEIGHT_NOISE, GATE_HEIGHT_NOISE, GATES)
        return Layout(
            start=np.r_[start, 0.01],
            gates_pos=np.c_[gates, heights],
            gates_yaw=np.arctan2(np.sin(yaws), np.cos(yaws)),
            poles=np.c_[poles, np.full(POLES, POLE_TOP)],
        )
