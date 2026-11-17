"""The grouping must be the fewest groups within the span, and its flight times must be right."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lsy_drone_racing.utils.track_groups import (
    axis_time,
    group_times,
    pass_times,
    path_curvature,
    segment_path,
    split_track,
)


@pytest.mark.parametrize(
    ("distance", "v0", "v1"),
    [(1.0, 0.0, 0.0), (0.5, 0.0, 1.0), (2.0, 3.0, 1.0), (-1.0, 2.0, 2.0), (0.1, 3.0, 3.0)],
)
def test_axis_time_reaches_the_end_state(distance: float, v0: float, v1: float):
    """Integrating the bang-bang motion gives the distance and the speed that were asked for."""
    accel = 12.0
    total = float(axis_time(np.array(distance), np.array(v0), np.array(v1), accel))
    # The switch time follows from reaching v1 at the end, for either order of the two phases.
    results = []
    for sign in (1.0, -1.0):
        first = (v1 - v0 + sign * accel * total) / (2.0 * sign * accel)
        if not -1e-9 <= first <= total + 1e-9:
            continue
        mid = v0 + sign * accel * first
        covered = (v0 + mid) / 2.0 * first + (mid + v1) / 2.0 * (total - first)
        results.append(covered)
    assert any(c == pytest.approx(distance, abs=1e-6) for c in results)


def test_axis_time_of_a_rest_to_rest_move():
    """Half way accelerating, half way braking."""
    assert float(axis_time(np.array(1.0), np.array(0.0), np.array(0.0), 1.0)) == pytest.approx(2.0)


def test_pass_times_increase_along_the_track():
    """Later passes are reached later, and a longer leg takes longer."""
    gates = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [6.0, 0.0, 1.0]])
    normals = np.tile([1.0, 0.0, 0.0], (3, 1))
    times = pass_times(np.array([0.0, 0.0, 1.0]), gates, normals)
    assert np.all(np.diff(times) > 0.0)
    assert times[2] - times[1] > times[1] - times[0]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("span", [0.2, 0.5, 1.0])
def test_groups_are_the_fewest_possible(seed: int, span: float):
    """No other split into consecutive runs within the span has fewer groups."""
    times = np.cumsum(np.random.default_rng(seed).uniform(0.05, 0.9, 6))
    groups = group_times(times, span)
    assert [p for g in groups for p in g] == list(range(6))
    assert all(times[g[-1]] - times[g[0]] <= span for g in groups)
    for cuts in range(len(groups) - 1):
        for chosen in itertools.combinations(range(1, 6), cuts):
            edges = [0, *chosen, 6]
            spans = [times[b - 1] - times[a] for a, b in zip(edges[:-1], edges[1:], strict=True)]
            assert max(spans) > span


def test_equal_splits_cut_at_the_longest_transit():
    """Of two splits with as many groups, the one with the longer transit wins."""
    assert group_times(np.array([0.0, 0.2, 0.9, 1.0]), 0.3) == [[0, 1], [2, 3]]
    assert group_times(np.array([0.0, 0.4, 0.5]), 0.2) == [[0], [1, 2]]


def test_a_gate_flown_twice_is_in_two_groups():
    """Passes are grouped, so the same gate can appear in separate groups."""
    gate = np.array([1.0, 0.0, 1.0])
    other = np.array([4.0, 0.0, 1.0])
    gates = np.stack([gate, other, gate])
    normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    split = split_track(np.array([0.0, 0.0, 1.0]), gates, normals, np.zeros((0, 3)), span=0.3)
    assert [g.passes for g in split.groups] == [(0,), (1,), (2,)]


def long_track() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a start and two gates far apart along x, with their passing directions."""
    gates = np.array([[1.0, 0.0, 1.0], [11.0, 0.0, 1.0]])
    return np.array([0.0, 0.0, 1.0]), gates, np.tile([1.0, 0.0, 0.0], (2, 1))


def test_a_pole_by_a_gate_joins_its_group():
    """A pole reached just after a gate is handled with it."""
    start, gates, normals = long_track()
    split = split_track(start, gates, normals, np.array([[1.3, 0.2, 1.5]]), span=0.4)
    assert split.groups[0].passes == (0,)
    assert split.groups[0].poles == (0,)
    assert split.groups[1].poles == ()


def test_a_lone_pole_on_a_long_leg_is_its_own_group():
    """A transit has nothing beside its route: the pole becomes a scene of its own."""
    start, gates, normals = long_track()
    split = split_track(start, gates, normals, np.array([[6.0, 0.2, 1.5]]), span=0.4)
    assert [g.passes for g in split.groups] == [(0,), (), (1,)]
    assert split.groups[1].poles == (0,)


def test_a_pole_away_from_the_route_is_ignored():
    """A pole beside no leg is in no group."""
    start, gates, normals = long_track()
    split = split_track(start, gates, normals, np.array([[6.0, 3.0, 1.5]]), span=0.4)
    assert split.ignored_poles == [0]
    assert all(g.poles == () for g in split.groups)


def test_a_pole_beside_two_legs_is_met_twice():
    """Flying past a pole twice means handling it twice."""
    gates = np.array([[1.0, 0.0, 1.0], [11.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    split = split_track(
        np.array([0.0, 0.0, 1.0]), gates, normals, np.array([[6.0, 0.2, 1.5]]), span=0.3
    )
    assert sum(g.poles.count(0) for g in split.groups) == 2


def test_groups_cover_every_pass_once():
    """Every pass is in exactly one group, in flight order."""
    start, gates, normals = long_track()
    poles = np.array([[1.3, 0.2, 1.5], [6.0, 0.2, 1.5], [10.8, -0.2, 1.5]])
    split = split_track(start, gates, normals, poles, span=0.5)
    assert [p for g in split.groups for p in g.passes] == [0, 1]
    assert all(a.end <= b.start for a, b in zip(split.groups, split.groups[1:], strict=False))


def flown(radius_at: list[tuple[float, float]], speed: float = 3.0, rate: float = 200.0) -> tuple:
    """Return a plan that holds a curvature for given durations: ``(radius or 0, seconds)``."""
    pos, vel, time = [np.zeros(3)], [np.array([speed, 0.0, 0.0])], [0.0]
    heading = 0.0
    for radius, seconds in radius_at:
        for _ in range(int(seconds * rate)):
            heading += (speed / radius if radius else 0.0) / rate
            v = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
            pos.append(pos[-1] + v / rate)
            vel.append(v)
            time.append(time[-1] + 1.0 / rate)
    return np.array(pos), np.array(vel), np.array(time)


def test_curvature_of_a_circle_is_one_over_its_radius():
    """A circle of radius 0.5 m has a curvature of 2 / m at any speed."""
    v = np.array([[3.0, 0.0, 0.0]])
    a = np.array([[0.0, 3.0**2 / 0.5, 0.0]])
    assert path_curvature(v, a)[0] == pytest.approx(2.0)


def test_a_straight_path_is_all_easy():
    """Nothing bends, so nothing is hard."""
    pos, vel, time = flown([(0.0, 1.0)])
    segments = segment_path(pos, vel, time, np.array([len(pos) - 1]), np.zeros((0, 3)))
    assert [s.hard for s in segments] == [False]


def test_a_sharp_turn_between_straights_is_hard():
    """The turn is hard, the straights on both sides of it easy."""
    pos, vel, time = flown([(0.0, 0.5), (0.3, 0.3), (0.0, 0.5)])
    segments = segment_path(pos, vel, time, np.array([len(pos) - 1]), np.zeros((0, 3)))
    assert [s.hard for s in segments] == [False, True, False]
    hard = segments[1]
    assert hard.start == pytest.approx(0.5, abs=0.05)
    assert hard.end == pytest.approx(0.8, abs=0.05)
    assert hard.peak_curvature == pytest.approx(1.0 / 0.3, rel=0.1)


def test_a_gentle_turn_stays_easy():
    """A wide turn at the same speed does not reach the hard curvature."""
    pos, vel, time = flown([(0.0, 0.3), (2.0, 0.5), (0.0, 0.3)])
    segments = segment_path(pos, vel, time, np.array([len(pos) - 1]), np.zeros((0, 3)))
    assert not any(s.hard for s in segments)


def test_close_hard_turns_merge_and_distant_ones_do_not():
    """Hard stretches close together are one scene."""
    near = flown([(0.0, 0.3), (0.3, 0.2), (0.0, 0.05), (-0.3, 0.2), (0.0, 0.3)])
    far = flown([(0.0, 0.3), (0.3, 0.2), (0.0, 0.5), (-0.3, 0.2), (0.0, 0.3)])
    for path, count in ((near, 1), (far, 2)):
        pos, vel, time = path
        segments = segment_path(pos, vel, time, np.array([len(pos) - 1]), np.zeros((0, 3)))
        assert sum(s.hard for s in segments) == count


def test_a_long_hard_stretch_is_split_to_stay_compact():
    """No hard stretch is longer than asked for."""
    pos, vel, time = flown([(0.0, 0.2), (0.3, 1.5), (0.0, 0.2)])
    segments = segment_path(pos, vel, time, np.array([len(pos) - 1]), np.zeros((0, 3)), longest=0.6)
    hard = [s for s in segments if s.hard]
    assert len(hard) >= 3
    assert all(s.end - s.start <= 0.6 for s in hard)


def test_segments_hold_their_passes_and_poles_and_cover_the_path():
    """Each gate pass is in one stretch, and each pole beside the path is in the right stretch."""
    pos, vel, time = flown([(0.0, 0.5), (0.3, 0.3), (0.0, 0.5)])
    crossings = np.array([20, 130, len(pos) - 1])
    pole_in_turn = pos[130] + [0.0, 0.1, 0.0]
    pole_far = np.array([0.0, 5.0, 1.0])
    poles = np.array([pole_in_turn, pole_far])
    segments = segment_path(pos, vel, time, crossings, poles)
    assert [p for s in segments for p in s.passes] == [0, 1, 2]
    assert 1 in segments[1].passes
    assert segments[1].poles == (0,)
    assert all(1 not in s.poles for s in segments)
    assert segments[0].start == 0.0
    assert all(a.end <= b.start for a, b in zip(segments, segments[1:], strict=False))
