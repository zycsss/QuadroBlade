"""Split a track into local scenes: the fewest groups that each span at most T seconds of flight.

A group is a run of consecutive events that one local maneuver has to handle together. An event is a
gate pass, or a pole beside the route. Passes rather than gates count, because a gate flown twice is
two different tasks, and likewise a pole beside two legs of the lap is met twice. What lies between
two groups is a transit, and it has no gate and no pole beside its route.

Flight time is used instead of distance, because how local two objects are depends on how fast they
are reached. It comes from a point mass that flies straight between gates with bounded acceleration
and picks its speed through every gate, which is quick enough for a track nobody has seen before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Speeds a point mass may have through a gate [m/s].
GATE_SPEEDS = (1.0, 2.0, 3.0, 4.0, 5.0)
#: Acceleration per axis of the point mass [m/s^2]: on level 0, the lap comes out at 3.29 s against
#: the 3.21 s of `topt_planner`.
ACCELERATION = 10.0
#: A pole farther than this from the straight route does not need avoiding [m]. Wider makes groups
#: less stable: at 0.5 m they change on most tracks when gates and poles shift as on level 2.
ROUTE_WIDTH = 0.25
#: Two passes of the same pole closer than this are one encounter [s].
SAME_ENCOUNTER = 0.2


def axis_time(distance: NDArray, v0: NDArray, v1: NDArray, accel: float = ACCELERATION) -> NDArray:
    """Return the shortest time to move along one axis between two speeds, at full acceleration.

    Args:
        distance: Signed distance to cover [m].
        v0: Speed at the start [m/s].
        v1: Speed at the end [m/s].
        accel: Largest acceleration [m/s^2].

    Returns:
        The time [s] of the bang-bang motion with one switch.
    """
    direction = np.where(v1 >= v0, 1.0, -1.0)
    straight = (v1**2 - v0**2) / (2.0 * accel * direction)
    sign = np.sign(distance - straight)
    sign = np.where(sign == 0.0, direction, sign)
    root = np.sqrt(np.maximum(0.5 * (v0**2 + v1**2) + sign * accel * distance, 0.0))
    return (2.0 * root - sign * (v0 + v1)) / accel


def pass_times(
    start: NDArray,
    gates: NDArray,
    normals: NDArray,
    speeds: tuple[float, ...] = GATE_SPEEDS,
    accel: float = ACCELERATION,
) -> NDArray:
    """Return when a point mass reaches every pass on its fastest route, from rest.

    The speed through each gate is chosen for the whole lap by dynamic programming, along the gate's
    passing direction. The axes are treated as independent, so that a leg takes as long as its
    slowest axis.

    Args:
        start: Where the flight starts ``(3,)``.
        gates: Gate centre of every pass ``(s, 3)``.
        normals: Passing direction of every pass, unit length ``(s, 3)``.
        speeds: Speeds to choose from.
        accel: Acceleration per axis.

    Returns:
        The time of every pass ``(s,)`` [s].
    """
    grid = np.asarray(speeds, dtype=float)
    # First leg, from rest, to every speed of the first gate.
    arrive = grid[:, None] * normals[0]
    first = axis_time(gates[0] - start, np.zeros(3), arrive, accel).max(axis=-1)
    best = first
    back = []
    legs = []
    for i in range(1, len(gates)):
        leave = grid[:, None] * normals[i - 1]
        arrive = grid[:, None] * normals[i]
        leg = axis_time(gates[i] - gates[i - 1], leave[:, None], arrive[None], accel).max(axis=-1)
        total = best[:, None] + leg
        back.append(total.argmin(axis=0))
        legs.append(leg)
        best = total.min(axis=0)
    # Walk back along the chosen speeds, to know how long each leg takes on the fastest route.
    choice = int(best.argmin())
    times = np.zeros(len(gates))
    times[-1] = best[choice]
    for i in range(len(gates) - 1, 0, -1):
        previous = int(back[i - 1][choice])
        times[i - 1] = times[i] - legs[i - 1][previous, choice]
        choice = previous
    return times


def group_times(times: NDArray, span: float) -> list[list[int]]:
    """Return the fewest runs of consecutive events that each span at most ``span`` seconds.

    Several splits can have the same number of groups. The one with the longest shortest transit is
    kept, so that a cut falls where flying apart costs least, and a small change of the times does
    not move it.

    Args:
        times: Time of every event, not decreasing ``(s,)`` [s].
        span: Longest time between the first and the last event of a group [s].

    Returns:
        The events of every group.
    """
    count = len(times)
    best, best_key = None, None
    for mask in range(2 ** (count - 1)):
        cuts = [i + 1 for i in range(count - 1) if mask >> i & 1]
        edges = [0, *cuts, count]
        groups = [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        if any(times[g[-1]] - times[g[0]] > span for g in groups):
            continue
        shortest = min((times[c] - times[c - 1] for c in cuts), default=np.inf)
        key = (len(groups), -shortest)
        if best_key is None or key < best_key:
            best, best_key = groups, key
    return best


@dataclass(frozen=True)
class Group:
    """A local scene.

    Attributes:
        passes: Gate passes in the group, in flight order.
        poles: Poles beside the route in the group.
        start: When the drone reaches the first event [s].
        end: When the drone reaches the last event [s].
    """

    passes: tuple[int, ...]
    poles: tuple[int, ...]
    start: float
    end: float


@dataclass(frozen=True)
class TrackGroups:
    """A track split into groups, with transits between them.

    Attributes:
        groups: The groups in flight order.
        ignored_poles: Poles beside no part of the route.
        times: Time of every pass [s].
    """

    groups: list[Group]
    ignored_poles: list[int]
    times: NDArray


def route_poles(
    start: NDArray, gates: NDArray, poles: NDArray, times: NDArray, route_width: float = ROUTE_WIDTH
) -> list[tuple[float, int]]:
    """Return when the drone passes each pole beside its route, once per leg it is beside.

    Args:
        start: Where the flight starts ``(3,)``.
        gates: Gate centre of every pass ``(s, 3)``.
        poles: Pole positions ``(p, 3)``; only x and y are used.
        times: Time of every pass ``(s,)`` [s].
        route_width: Distance from the route within which a pole matters [m].

    Returns:
        ``(time, pole)`` for every encounter.
    """
    points = np.vstack([start[None], gates])[:, :2]
    clock = np.r_[0.0, times]
    leg = points[1:] - points[:-1]
    length = np.maximum(np.sum(leg**2, axis=1), 1e-9)
    found = []
    for p, pole in enumerate(poles[:, :2]):
        share = np.clip(np.sum((pole - points[:-1]) * leg, axis=1) / length, 0.0, 1.0)
        gap = np.linalg.norm(pole - (points[:-1] + share[:, None] * leg), axis=1)
        when = clock[:-1] + share * (clock[1:] - clock[:-1])
        kept: list[tuple[float, float]] = []
        for i in np.flatnonzero(gap <= route_width):
            # A pole by a gate is beside the legs on both sides of it, but it is one encounter.
            if kept and when[i] - kept[-1][0] < SAME_ENCOUNTER:
                if gap[i] < kept[-1][1]:
                    kept[-1] = (float(when[i]), float(gap[i]))
            else:
                kept.append((float(when[i]), float(gap[i])))
        found.extend((t, p) for t, _ in kept)
    return found


def split_track(
    start: NDArray,
    gates: NDArray,
    normals: NDArray,
    poles: NDArray,
    span: float,
    route_width: float = ROUTE_WIDTH,
    accel: float = ACCELERATION,
) -> TrackGroups:
    """Split a track into the fewest groups of passes and poles within ``span`` seconds.

    Args:
        start: Where the flight starts ``(3,)``.
        gates: Gate centre of every pass ``(s, 3)``.
        normals: Passing direction of every pass ``(s, 3)``.
        poles: Pole positions ``(p, 3)``; only x and y are used.
        span: Longest time a group may span [s].
        route_width: Distance from the route within which a pole matters [m].
        accel: Acceleration per axis of the point mass.

    Returns:
        The groups.
    """
    times = pass_times(start, gates, normals, accel=accel)
    encounters = route_poles(start, gates, poles, times, route_width)
    events = sorted(
        [(float(t), 0, i) for i, t in enumerate(times)] + [(t, 1, p) for t, p in encounters]
    )
    clock = np.array([e[0] for e in events])
    groups = [
        Group(
            passes=tuple(events[i][2] for i in run if events[i][1] == 0),
            poles=tuple(events[i][2] for i in run if events[i][1] == 1),
            start=events[run[0]][0],
            end=events[run[-1]][0],
        )
        for run in group_times(clock, span)
    ]
    ignored = sorted(set(range(len(poles))) - {p for _, p in encounters})
    return TrackGroups(groups, ignored, times)


# Segmenting the ideal path by difficulty.

#: Sample rate of the plan when it is segmented [Hz].
SEGMENT_RATE = 100.0


@dataclass(frozen=True)
class Segment:
    """A stretch of the ideal path.

    Attributes:
        hard: Whether the path bends sharply here.
        start: When the stretch begins [s].
        end: When the stretch ends [s].
        peak_curvature: The sharpest bend [1/m].
        mean_speed: The plan's mean speed [m/s].
        passes: Gate passes in the stretch.
        poles: Poles the path passes within the pole width in the stretch, once per encounter.
    """

    hard: bool
    start: float
    end: float
    peak_curvature: float
    mean_speed: float
    passes: tuple[int, ...]
    poles: tuple[int, ...]


def path_curvature(vel: NDArray, accel: NDArray) -> NDArray:
    """Return the curvature of a path from its velocity and acceleration [1/m]."""
    speed = np.maximum(np.linalg.norm(vel, axis=1), 1e-3)
    return np.linalg.norm(np.cross(vel, accel), axis=1) / speed**3


def _hard_spans(
    curvature: NDArray, enter: float, release: float, merge_gap: int, shortest: int, longest: int
) -> list[tuple[int, int]]:
    """Return the sample ranges ``[a, b)`` of the hard stretches of a curvature profile."""
    spans: list[list[int]] = []
    above = curvature >= release
    i = 0
    while i < len(curvature):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(curvature) and above[j]:
            j += 1
        # With two thresholds a stretch only counts if it really gets sharp, and does not flicker.
        if curvature[i:j].max() >= enter:
            if spans and i - spans[-1][1] <= merge_gap:
                spans[-1][1] = j
            else:
                spans.append([i, j])
        i = j
    # Long hard stretches are split at their gentlest point, so that each stays one compact scene.
    done: list[tuple[int, int]] = []
    todo = [(a, b) for a, b in spans if b - a >= shortest]
    while todo:
        a, b = todo.pop()
        if b - a <= longest:
            done.append((a, b))
            continue
        margin = (b - a) // 4
        cut = a + margin + int(curvature[a + margin : b - margin].argmin())
        todo += [(a, cut), (cut, b)]
    return sorted(done)


def segment_path(
    pos: NDArray,
    vel: NDArray,
    time: NDArray,
    crossings: NDArray,
    poles: NDArray,
    curvature_hard: float = 2.0,
    release: float = 0.6,
    merge_gap: float = 0.15,
    shortest: float = 0.1,
    longest: float = 0.6,
    min_speed: float = 1.0,
    pole_width: float = ROUTE_WIDTH,
) -> list[Segment]:
    """Split an ideal path into hard and easy stretches by how sharply it bends.

    Args:
        pos: Path positions ``(m, 3)``.
        vel: Path velocities ``(m, 3)``.
        time: Time stamps ``(m,)``, not decreasing [s].
        crossings: Node of every gate pass ``(s,)``.
        poles: Pole positions ``(p, 3)``; only x and y are used.
        curvature_hard: Curvature a stretch must reach to be hard [1/m].
        release: A hard stretch lasts until the curvature falls below this share of the above.
        merge_gap: Hard stretches closer than this become one [s].
        shortest: Hard stretches shorter than this are dropped [s].
        longest: Hard stretches longer than this are split [s].
        min_speed: Below this speed, as in the take-off, curvature says nothing [m/s].
        pole_width: A pole this close to the path is one the stretch has to handle [m].

    Returns:
        The stretches in flight order, covering the path up to its last gate pass.
    """
    end_time = float(time[crossings[-1]])
    keep = np.r_[True, np.diff(time) > 1e-6]
    clock = np.arange(0.0, end_time, 1.0 / SEGMENT_RATE)
    sample = lambda values: np.stack(  # noqa: E731
        [np.interp(clock, time[keep], values[keep, i]) for i in range(values.shape[1])], axis=1
    )
    p, v = sample(pos), sample(vel)
    kernel = np.ones(5) / 5.0
    a = np.stack(
        [np.convolve(np.gradient(v[:, i], clock), kernel, mode="same") for i in range(3)], axis=1
    )
    speed = np.linalg.norm(v, axis=1)
    curvature = path_curvature(v, a)
    # Curvature means nothing before the drone is moving.
    moving = int(np.argmax(speed >= min_speed))
    curvature[:moving] = 0.0
    curvature[speed < 0.5 * min_speed] = 0.0

    rate = SEGMENT_RATE
    spans = _hard_spans(
        curvature,
        curvature_hard,
        release * curvature_hard,
        int(merge_gap * rate),
        int(shortest * rate),
        int(longest * rate),
    )
    edges: list[tuple[int, int, bool]] = []
    cursor = 0
    for a_, b_ in spans:
        if a_ > cursor:
            edges.append((cursor, a_, False))
        edges.append((a_, b_, True))
        cursor = b_
    if cursor < len(clock):
        edges.append((cursor, len(clock), False))

    pass_times_ = time[crossings]
    encounters: list[tuple[float, int]] = []
    for j, pole in enumerate(poles[:, :2]):
        gap = np.linalg.norm(p[:, :2] - pole, axis=1)
        near = np.flatnonzero(gap <= pole_width)
        if len(near):
            for run in np.split(near, np.flatnonzero(np.diff(near) > 1) + 1):
                encounters.append((float(clock[run[gap[run].argmin()]]), j))

    segments = []
    for i, (a_, b_, hard) in enumerate(edges):
        t0, t1 = float(clock[a_]), float(clock[b_ - 1])
        last = i == len(edges) - 1
        inside = lambda t, t0=t0, t1=t1, last=last: t0 <= t and (t <= t1 + 0.5 / rate or last)  # noqa: E731
        segments.append(
            Segment(
                hard=hard,
                start=t0,
                end=t1,
                peak_curvature=float(curvature[a_:b_].max()),
                mean_speed=float(speed[a_:b_].mean()),
                passes=tuple(int(k) for k, t in enumerate(pass_times_) if inside(t)),
                poles=tuple(q for t, q in sorted(encounters) if inside(t)),
            )
        )
    return segments
