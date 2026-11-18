"""What a policy trained on local scenes sees: nothing that tells it which track it is on.

`race_policy` gives its policy the position in the arena and the share of the lap done, which are
enough to recognize one track and useless on any other. Here those are gone. What remains is the
scene around the drone, in a frame that turns with the guide, so that a scene looks the same
whichever way it faces. The drone's yaw stays at zero, and the output is turned back into the world.

Every function takes NumPy or JAX arrays, so that training compiles the same code that flight runs.
A `GuideLibrary` holds any number of solved tracks, and every drone flies the one its row names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from lsy_drone_racing.utils.path_policy import (
    ACT_DIM,
    LOOK_AHEAD,
    SPACING,
    SPEED_AHEAD,
    THRUST_MAX,
    THRUST_MIN,
    TILT_MAX,
    _xp,
    nearest,
    rotation_matrix,
)
from lsy_drone_racing.utils.race_policy import (
    RUN_OUT_BEADS,
    RaceTrack,
    body_center,
    clearances,
    guide_arrays,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

#: Beads ahead of the drone that set the frame. A long chord does not jump, and it still has a
#: direction during the climb from the floor.
HEADING_CHORD = 20
#: Walls further than this all look the same [m].
WALL_RANGE = 3.0
#: `path_policy.features`, the two next gates, the two nearest poles, the nearest gate with its axis
#: and its closest point, the walls in four directions and the height.
OBS_DIM = 53 + 12 + 6 + 4 + 6 + 4 + 1


class GuideLibrary(NamedTuple):
    """Solved tracks with a shared order of gates. Row ``l`` is one track.

    Attributes:
        beads: Guide paths, all as long as the longest ``(l, m, 3)``.
        speeds: The plan's speed at every bead ``(l, m)``.
        velocities: The plan's velocity at every bead ``(l, m, 3)``.
        quats: The plan's attitude at every bead at zero yaw ``(l, m, 4)``.
        crossings: Bead of every pass ``(l, s)``.
        starts: Where the drone starts ``(l, 3)``.
        gates_pos: Nominal gate centres ``(l, g, 3)``.
        gates_rpy: Nominal gate angles ``(l, g, 3)``.
        poles: Nominal pole tops ``(l, p, 3)``.
        sequence: Gate of every pass ``(s,)``.
        directions: Direction of every pass ``(s,)``.
        boxes_center: Centres of a gate's collision boxes in its frame ``(b, 3)``.
        boxes_half: Half extents of those boxes ``(b, 3)``.
        pos_low: Lower position limit ``(3,)``.
        pos_high: Upper position limit ``(3,)``.
    """

    beads: NDArray
    speeds: NDArray
    velocities: NDArray
    quats: NDArray
    crossings: NDArray
    starts: NDArray
    gates_pos: NDArray
    gates_rpy: NDArray
    poles: NDArray
    sequence: NDArray
    directions: NDArray
    boxes_center: NDArray
    boxes_half: NDArray
    pos_low: NDArray
    pos_high: NDArray


class _Plan(NamedTuple):
    pos: NDArray
    vel: NDArray
    time: NDArray
    crossings: NDArray


def _pad(guide: dict[str, NDArray], length: int, spacing: float = 0.05) -> dict[str, NDArray]:
    """Return a guide lengthened along its run-out to ``length`` beads."""
    short = length - len(guide["beads"])
    tangent = guide["beads"][-1] - guide["beads"][-2]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9) * spacing
    return {
        "beads": np.r_[
            guide["beads"], guide["beads"][-1] + np.outer(np.arange(1, short + 1), tangent)
        ],
        "speeds": np.r_[guide["speeds"], np.full(short, guide["speeds"][-1])],
        "velocities": np.r_[guide["velocities"], np.tile(guide["velocities"][-1], (short, 1))],
        "quats": np.r_[guide["quats"], np.tile(guide["quats"][-1], (short, 1))],
        "crossings": guide["crossings"],
    }


def load_library(files: str | Sequence[str], reference: RaceTrack) -> GuideLibrary:
    """Return the tracks of files written by `scripts/solve_scene_library.py`, as one library.

    Args:
        files: One file, several separated by commas, or a list of them.
        reference: A track that gives the order of the gates, the boxes and the limits.

    Returns:
        The library.
    """
    names = files.split(",") if isinstance(files, str) else list(files)
    guides, layouts = [], []
    for name in names:
        data = np.load(name)
        for i in range(int(data["count"])):
            plan = _Plan(
                data[f"pos_{i}"], data[f"vel_{i}"], data[f"time_{i}"], data[f"crossings_{i}"]
            )
            guides.append(guide_arrays(plan, smooth=True))
            layouts.append(
                {k: data[f"{k}_{i}"] for k in ("start", "gates_pos", "gates_rpy", "poles")}
            )
    length = max(len(g["beads"]) for g in guides)
    guides = [_pad(g, length) for g in guides]
    stack = {k: np.stack([g[k] for g in guides]) for k in guides[0]}
    return GuideLibrary(
        **stack,
        starts=np.stack([x["start"] for x in layouts]),
        gates_pos=np.stack([x["gates_pos"] for x in layouts]),
        gates_rpy=np.stack([x["gates_rpy"] for x in layouts]),
        poles=np.stack([x["poles"] for x in layouts]),
        sequence=reference.sequence,
        directions=reference.directions,
        boxes_center=reference.boxes_center,
        boxes_half=reference.boxes_half,
        pos_low=reference.pos_low,
        pos_high=reference.pos_high,
    )


class StagedScenarios(NamedTuple):
    """Shifted versions of a library's tracks, with a guide for every stage of reveals.

    Scenario ``s`` is one track with one draw of shifts. Its guide at stage ``k`` is row
    ``s * stages + k`` of the library: planned through the layout with the first ``k`` objects of
    ``order`` at their true poses and the rest as reported.

    Attributes:
        gates_pos: True gate centres ``(s, g, 3)``.
        gates_rpy: True gate angles ``(s, g, 3)``.
        poles: True pole tops ``(s, p, 3)``.
        order: Objects in the order the stage-0 guide reaches them, gates ``0..g-1`` then poles
            ``g..g+p-1`` ``(s, g + p)``.
        entry: Bead of the stage-0 guide where each object, in that order, comes into sensor range
            ``(s, g + p)``.
        stages: Guides per scenario, ``g + p + 1``.
    """

    gates_pos: NDArray
    gates_rpy: NDArray
    poles: NDArray
    order: NDArray
    entry: NDArray
    stages: int


def reveal_order(
    beads: NDArray, gates_pos: NDArray, poles: NDArray, sensor_range: float
) -> tuple[NDArray, NDArray]:
    """Return the order in which a guide reaches gates and poles, as the race's sensor does.

    The sensor reveals an object once the drone is within ``sensor_range`` of it in x and y.
    Training and flight both order by the reported poses, which flight knows from the start.

    Args:
        beads: The guide ``(m, 3)``.
        gates_pos: Gate centres ``(g, 3)``.
        poles: Pole tops ``(p, 3)``.
        sensor_range: The race's sensor range [m].

    Returns:
        Objects in order, gates ``0..g-1`` then poles, and the bead at which each, in that order,
        comes into range. An object never in range comes last, by its closest approach, with a bead
        past the guide's end.
    """
    objects = np.r_[gates_pos[:, :2], poles[:, :2]]
    distance = np.linalg.norm(beads[None, :, :2] - objects[:, None], axis=2)
    inside = distance < sensor_range
    never = len(beads) + distance.argmin(axis=1)
    first = np.where(inside.any(axis=1), inside.argmax(axis=1), never)
    order = np.argsort(first, kind="stable")
    return order, first[order]


def staged_layout(
    nominal: tuple[NDArray, NDArray, NDArray],
    true: tuple[NDArray, NDArray, NDArray],
    order: NDArray,
    stage: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return the layout known at a stage: the first ``stage`` objects of ``order`` true.

    Args:
        nominal: Reported gate centres ``(g, 3)``, gate angles ``(g, 3)`` and pole tops ``(p, 3)``.
        true: The same, true.
        order: Objects in reveal order, gates first then poles, as `reveal_order` numbers them.
        stage: How many objects of ``order`` are known.

    Returns:
        Gate centres, gate angles and pole tops.
    """
    gates_pos, gates_rpy, poles = (np.array(x, dtype=float) for x in nominal)
    count = len(gates_pos)
    for i in order[:stage]:
        if i < count:
            gates_pos[i], gates_rpy[i] = true[0][i], true[1][i]
        else:
            poles[i - count] = true[2][i - count]
    return gates_pos, gates_rpy, poles


def load_staged_library(
    files: str | Sequence[str], reference: RaceTrack
) -> tuple[GuideLibrary, StagedScenarios]:
    """Return the guides and scenarios of files written by `scripts/solve_staged_library.py`.

    Args:
        files: One file, several separated by commas, or a list of them.
        reference: A track that gives the order of the gates, the boxes and the limits.

    Returns:
        The library, with ``stages`` rows per scenario, and the scenarios.
    """
    names = files.split(",") if isinstance(files, str) else list(files)
    guides, layouts, scenarios, stages = [], [], [], None
    for name in names:
        data = np.load(name)
        count = int(data["count"])
        stages = int(data["stages"]) if stages is None else stages
        if int(data["stages"]) != stages:
            raise ValueError(f"{name} has {int(data['stages'])} stages, not {stages}")
        for i in range(count):
            layout = {k: data[f"{k}_{i}"] for k in ("start", "gates_pos", "gates_rpy", "poles")}
            for k in range(stages):
                plan = _Plan(
                    data[f"pos_{i}_{k}"],
                    data[f"vel_{i}_{k}"],
                    data[f"time_{i}_{k}"],
                    data[f"crossings_{i}_{k}"],
                )
                guides.append(guide_arrays(plan, smooth=True))
                layouts.append(layout)
            scenarios.append(
                {k: data[f"{k}_{i}"] for k in ("true_gates_pos", "true_gates_rpy", "true_poles")}
                | {"order": data[f"order_{i}"], "entry": data[f"entry_{i}"]}
            )
    length = max(len(g["beads"]) for g in guides)
    guides = [_pad(g, length) for g in guides]
    stack = {k: np.stack([g[k] for g in guides]) for k in guides[0]}
    library = GuideLibrary(
        **stack,
        starts=np.stack([x["start"] for x in layouts]),
        gates_pos=np.stack([x["gates_pos"] for x in layouts]),
        gates_rpy=np.stack([x["gates_rpy"] for x in layouts]),
        poles=np.stack([x["poles"] for x in layouts]),
        sequence=reference.sequence,
        directions=reference.directions,
        boxes_center=reference.boxes_center,
        boxes_half=reference.boxes_half,
        pos_low=reference.pos_low,
        pos_high=reference.pos_high,
    )
    return library, StagedScenarios(
        gates_pos=np.stack([x["true_gates_pos"] for x in scenarios]),
        gates_rpy=np.stack([x["true_gates_rpy"] for x in scenarios]),
        poles=np.stack([x["true_poles"] for x in scenarios]),
        order=np.stack([x["order"] for x in scenarios]),
        entry=np.stack([x["entry"] for x in scenarios]),
        stages=stages,
    )


def track_library(track: RaceTrack, gates_rpy: NDArray) -> GuideLibrary:
    """Return one track as a library of one, as flight and the level checks use it."""
    return GuideLibrary(
        beads=track.beads[None],
        speeds=track.speeds[None],
        velocities=track.velocities[None],
        quats=track.quats[None],
        crossings=track.crossings[None],
        starts=track.start[None],
        gates_pos=track.gates_pos[None],
        gates_rpy=gates_rpy[None],
        poles=track.poles[None],
        sequence=track.sequence,
        directions=track.directions,
        boxes_center=track.boxes_center,
        boxes_half=track.boxes_half,
        pos_low=track.pos_low,
        pos_high=track.pos_high,
    )


def scene_clearances(
    center: NDArray, gates_pos: NDArray, gates_rot: NDArray, poles: NDArray, lib: GuideLibrary
) -> tuple[NDArray, NDArray]:
    """Return the free distance from collision spheres to every pole and gate of every world."""
    return clearances(
        center, _Layout(gates_pos, gates_rot, poles, lib.boxes_center, lib.boxes_half)
    )


class _Layout(NamedTuple):
    gates_pos: NDArray
    gates_rot: NDArray
    poles: NDArray
    boxes_center: NDArray
    boxes_half: NDArray


def closest_frame_point(
    center: NDArray, gate_pos: NDArray, gate_rot: NDArray, lib: GuideLibrary
) -> NDArray:
    """Return the vector from collision spheres to the closest point of one gate each.

    A gate's centre says nothing about where its bars are, and a track is full of gates the drone is
    not heading for. On one track that can be memorized, but on an unseen one it has to be seen.

    Args:
        center: Sphere centres ``(n, 3)``.
        gate_pos: Centre of every drone's gate ``(n, 3)``.
        gate_rot: Rotation of every drone's gate ``(n, 3, 3)``.
        lib: The library, for the boxes that make up a gate.

    Returns:
        Vectors in the world frame ``(n, 3)``.
    """
    xp = _xp(center)
    local = xp.einsum("nj,nji->ni", center - gate_pos, gate_rot)[:, None]
    low = lib.boxes_center[None] - lib.boxes_half[None]
    high = lib.boxes_center[None] + lib.boxes_half[None]
    closest = xp.clip(local, low, high)
    distance = xp.linalg.norm(closest - local, axis=-1)
    box = distance.argmin(axis=1)
    offset = closest[xp.arange(center.shape[0]), box] - local[:, 0]
    return xp.einsum("nij,nj->ni", gate_rot, offset)


def advance_scene_guide(
    pos: NDArray, index: NDArray, passed: NDArray, rows: NDArray, lib: GuideLibrary
) -> tuple[NDArray, NDArray]:
    """Move every drone's place on its guide, which stays between its last and its next gate.

    Args:
        pos: Positions ``(n, 3)``.
        index: Current beads ``(n,)``.
        passed: Passes so far ``(n,)``.
        rows: Row of every drone in the library ``(n,)``.
        lib: The library.

    Returns:
        The new beads and the distance to them.
    """
    xp = _xp(pos)
    count = lib.sequence.shape[0]
    new, _ = nearest(pos, lib.beads, index, rows)
    low = xp.where(passed > 0, lib.crossings[rows, xp.maximum(passed - 1, 0)], 0)
    high = lib.crossings[rows, xp.minimum(passed, count - 1)]
    new = xp.clip(new, low, high)
    return new, xp.linalg.norm(lib.beads[rows, new] - pos, axis=-1)


def scene_heading(index: NDArray, rows: NDArray, lib: GuideLibrary) -> tuple[NDArray, NDArray]:
    """Return the cosine and sine of the angle of the frame: the way the guide leads from here."""
    xp = _xp(index)
    last = lib.beads.shape[1] - 1
    chord = lib.beads[rows, xp.minimum(index + HEADING_CHORD, last)] - lib.beads[rows, index]
    # On a guide that goes straight up the frame is arbitrary; pointing along x keeps it defined.
    norm = xp.sqrt(chord[:, 0] ** 2 + chord[:, 1] ** 2 + 1e-12)
    flat = norm < 1e-5
    return xp.where(flat, 1.0, chord[:, 0] / norm), xp.where(flat, 0.0, chord[:, 1] / norm)


def _turn(v: NDArray, c: NDArray, s: NDArray) -> NDArray:
    """Return vectors ``(n, 3)`` or ``(n, k, 3)`` in the frame; x and y turn, z does not."""
    xp = _xp(v)
    if v.ndim == 3:
        c, s = c[:, None], s[:, None]
    x, y = v[..., 0], v[..., 1]
    return xp.stack((c * x + s * y, -s * x + c * y, v[..., 2]), axis=-1)


def _walls(pos: NDArray, c: NDArray, s: NDArray, lib: GuideLibrary) -> NDArray:
    """Return the distance to the arena's walls ahead, behind, left and right in the frame."""
    xp = _xp(pos)
    room_high = lib.pos_high[:2] - pos[:, :2]
    room_low = pos[:, :2] - lib.pos_low[:2]
    distances = []
    for dx, dy in ((c, s), (-c, -s), (-s, c), (s, -c)):
        rays = []
        for d, high, low in (
            (dx, room_high[:, 0], room_low[:, 0]),
            (dy, room_high[:, 1], room_low[:, 1]),
        ):
            forward = d > 1e-6
            backward = d < -1e-6
            safe = xp.where(xp.abs(d) > 1e-6, xp.abs(d), 1.0)
            rays.append(xp.where(forward, high / safe, xp.where(backward, low / safe, WALL_RANGE)))
        distances.append(xp.clip(xp.minimum(rays[0], rays[1]), -WALL_RANGE, WALL_RANGE))
    return xp.stack(distances, axis=-1)


def scene_features(
    pos: NDArray,
    vel: NDArray,
    quat: NDArray,
    ang_vel: NDArray,
    previous: NDArray,
    index: NDArray,
    passed: NDArray,
    rows: NDArray,
    gates_pos: NDArray,
    gates_rot: NDArray,
    poles: NDArray,
    lib: GuideLibrary,
    bend: Bend | None = None,
) -> NDArray:
    """Return the policy's input, ``(n, OBS_DIM)``.

    Args:
        pos: Positions ``(n, 3)``.
        vel: Velocities ``(n, 3)``.
        quat: Attitudes, ``xyzw`` ``(n, 4)``.
        ang_vel: Angular velocities ``(n, 3)``.
        previous: The previous output, which is already in the frame ``(n, ACT_DIM)``.
        index: Current beads ``(n,)``.
        passed: Passes so far ``(n,)``.
        rows: Row of every drone in the library ``(n,)``.
        gates_pos: Gates as the sensor shows them ``(n, g, 3)``.
        gates_rot: Rotations of those gates ``(n, g, 3, 3)``.
        poles: Poles as the sensor shows them ``(n, p, 3)``.
        lib: The library.
        bend: A bend of the guide, from `make_bend`: the policy then sees the bent guide ahead,
            and its speeds scaled. Its offset from the guide and the frame stay those of the guide
            as planned.
    """
    xp = _xp(pos)
    n = pos.shape[0]
    worlds = xp.arange(n)
    c, s = scene_heading(index, rows, lib)
    m = lib.beads.shape[1]
    matrix = rotation_matrix(quat)
    # The frame turns the world, which is the same as multiplying the attitude from the left.
    attitude = xp.concatenate(
        (
            c[:, None] * matrix[:, :3] + s[:, None] * matrix[:, 3:6],
            -s[:, None] * matrix[:, :3] + c[:, None] * matrix[:, 3:6],
            matrix[:, 6:],
        ),
        axis=1,
    )

    def bead(j: NDArray) -> NDArray:
        point = lib.beads[rows, j]
        return point if bend is None else point + bend_offset(bend, j)

    # The offset from the guide as planned, also under a bend: a bend starts at the drone, so its
    # own offset there is always zero and would hide how far the drone has drifted.
    here = _turn(pos - lib.beads[rows, index], c, s)
    ahead = [_turn(bead(xp.minimum(index + k, m - 1)) - pos, c, s) for k in LOOK_AHEAD]
    scale = 1.0 if bend is None else bend.scale[:, None]
    wanted = [lib.speeds[rows, xp.minimum(index + k, m - 1)][:, None] * scale for k in SPEED_AHEAD]

    count = lib.sequence.shape[0]
    gates = []
    for i in (xp.minimum(passed, count - 1), xp.minimum(passed + 1, count - 1)):
        gate = lib.sequence[i]
        gates.append(_turn(gates_pos[worlds, gate] - pos, c, s))
        gates.append(_turn(gates_rot[worlds, gate][:, :, 0] * lib.directions[i][:, None], c, s))

    center = body_center(pos, quat)
    pole_gaps, frames = scene_clearances(center, gates_pos, gates_rot, poles, lib)
    order = xp.argsort(pole_gaps, axis=1)[:, :2]
    pole_rel = _turn(poles[worlds[:, None], order] - pos[:, None], c, s)[..., :2].reshape(n, 4)
    pole_gap = xp.take_along_axis(pole_gaps, order, axis=1)
    near = frames.argmin(axis=1)
    return xp.concatenate(
        [
            _turn(vel, c, s),
            _turn(ang_vel, c, s),
            attitude,
            previous,
            here,
            *ahead,
            *wanted,
            *gates,
            pole_rel,
            pole_gap,
            _turn(gates_pos[worlds, near] - pos, c, s),
            frames[worlds, near][:, None],
            _turn(gates_rot[worlds, near][:, :, 0], c, s),
            _turn(
                closest_frame_point(center, gates_pos[worlds, near], gates_rot[worlds, near], lib),
                c,
                s,
            ),
            _walls(pos, c, s, lib),
            pos[:, 2:3],
        ],
        axis=1,
    ).astype(np.float32)


def scene_action(u: NDArray, c: NDArray, s: NDArray) -> NDArray:
    """Map the policy's ``[-1, 1]`` output in the frame to the attitude controller's command.

    The outputs are the first two columns of the wanted rotation in the frame, as an offset from
    level, and the thrust. The thrust axis is what moves the drone, so it is turned back into the
    world and flown at zero yaw. Only its direction matters: a drone with this rotor layout flies
    the same at any yaw.

    Args:
        u: The policy's output ``(n, ACT_DIM)``.
        c: Cosine of the angle of the frame ``(n,)``.
        s: Sine of the angle of the frame ``(n,)``.

    Returns:
        ``[roll, pitch, 0, thrust]`` ``(n, 4)``.
    """
    xp = _xp(u)
    u = xp.clip(u, -1.0, 1.0)
    first = u[..., :3] + xp.asarray([1.0, 0.0, 0.0], dtype=u.dtype)
    second = u[..., 3:6] + xp.asarray([0.0, 1.0, 0.0], dtype=u.dtype)
    x = first / xp.maximum(xp.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    y = second - xp.sum(x * second, axis=-1, keepdims=True) * x
    y = y / xp.maximum(xp.linalg.norm(y, axis=-1, keepdims=True), 1e-6)
    z = xp.cross(x, y)
    zx = c * z[..., 0] - s * z[..., 1]
    zy = s * z[..., 0] + c * z[..., 1]
    roll = -xp.arcsin(xp.clip(zy, -1.0, 1.0))
    pitch = xp.arctan2(zx, z[..., 2])
    thrust = THRUST_MIN + 0.5 * (u[..., 6] + 1.0) * (THRUST_MAX - THRUST_MIN)
    return xp.stack(
        (
            xp.clip(roll, -TILT_MAX, TILT_MAX),
            xp.clip(pitch, -TILT_MAX, TILT_MAX),
            xp.zeros_like(roll),
            thrust,
        ),
        -1,
    )


# region bend

#: Beads ahead of where a bend is made at which it sets an offset: 0.5, 1, 2 and 3 m.
BEND_POINTS = (10, 20, 40, 60)
#: Beads ahead at which a bend has faded back to the guide.
BEND_FADE = 80
#: Beads over which a bend's start at the drone blends into its control points.
BEND_START = 10
#: A bender's output: a sideways and an upward offset per control point, then the speed scale.
BEND_DIM = 2 * len(BEND_POINTS) + 1
#: Largest change of the guide's speeds, as a share.
SPEED_SCALE = 0.3
#: Sideways acceleration a bend may ask for when it limits its offsets to what is reachable [m/s^2].
SIDE_ACCEL = 6.0
#: Points of the guide that a bender sees beyond the tracker's, in beads ahead (up to 4 m).
BENDER_AHEAD = (6, 10, 25, 35, 50, 70, 80)
BENDER_DIM = OBS_DIM + 3 * len(BENDER_AHEAD) + BEND_DIM + 4


class Bend(NamedTuple):
    """A bend of every drone's guide, tied to the beads after where it was made.

    The offset at bead ``j`` is that of ``k = j - anchor`` beads past the anchor. It starts at the
    drone (``start``), leaves along the drone's velocity (``drift``, sideways metres per metre
    along),
    blends into the control points by `BEND_START`, and fades back to the guide by `BEND_FADE`.

    Attributes:
        anchor: Bead at which the bend was made ``(n,)``.
        start: Where the drone was, relative to that bead ``(n, 3)``.
        drift: The drone's velocity across the guide, per unit of its speed ``(n, 3)``.
        points: Offsets at `BEND_POINTS` ``(n, 4, 3)``, in the world.
        scale: Factor on the guide's speeds ``(n,)``.
    """

    anchor: NDArray
    start: NDArray
    drift: NDArray
    points: NDArray
    scale: NDArray


def zero_bend(index: NDArray) -> Bend:
    """Return no bend at all: the guide as planned."""
    xp = _xp(index)
    n = index.shape[0]
    zeros = xp.zeros((n, 3))
    return Bend(index, zeros, zeros, xp.zeros((n, len(BEND_POINTS), 3)), xp.ones(n))


def make_bend(
    u: NDArray,
    limit: float | NDArray,
    pos: NDArray,
    vel: NDArray,
    index: NDArray,
    rows: NDArray,
    lib: GuideLibrary,
    from_drone: bool = False,
) -> Bend:
    """Return the bend a bender's output asks for, given the drone's state.

    Args:
        u: The bender's output ``(n, BEND_DIM)``, clipped to [-1, 1]: sideways offsets in the frame,
            upward offsets, and the speed scale.
        limit: Largest offset [m].
        pos: Positions ``(n, 3)``.
        vel: Velocities ``(n, 3)``.
        index: Current beads ``(n,)``.
        rows: Rows in the library ``(n,)``.
        lib: The library.
        from_drone: Start the bend at the drone, leaving along its velocity, instead of on the
            guide. A zero bend is then not the guide, which a tracker trained on guides doesn't
            expect: it cost a started `safety_first` tracker 6 to 16 points of its validation
            finishes.

    Returns:
        The bend. No offset exceeds what the drone can reach by then at `SIDE_ACCEL`.
    """
    xp = _xp(pos)
    u = xp.clip(u, -1.0, 1.0)
    c, s = scene_heading(index, rows, lib)
    count = len(BEND_POINTS)
    side = xp.stack((-s, c, xp.zeros_like(c)), axis=-1)
    up = xp.asarray([0.0, 0.0, 1.0])
    speed = xp.maximum(xp.linalg.norm(vel, axis=-1), 0.5)
    seconds = xp.asarray(BEND_POINTS, dtype=float)[None] * SPACING / speed[:, None]
    cap = xp.minimum(limit, 0.5 * SIDE_ACCEL * seconds**2)
    points = (u[:, :count] * cap)[..., None] * side[:, None] + (u[:, count : 2 * count] * cap)[
        ..., None
    ] * up
    last = lib.beads.shape[1] - 1
    tangent = lib.beads[rows, xp.minimum(index + 1, last)] - lib.beads[rows, index]
    tangent = tangent / xp.maximum(xp.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    across = vel - xp.sum(vel * tangent, axis=-1, keepdims=True) * tangent
    on = 1.0 if from_drone else 0.0
    return Bend(
        index,
        on * (pos - lib.beads[rows, index]),
        on * across / speed[:, None],
        points,
        1.0 + SPEED_SCALE * u[:, 2 * count],
    )


def bend_offset(bend: Bend, j: NDArray) -> NDArray:
    """Return the offsets of beads ``j`` ``(n,)`` or ``(n, k)`` under a bend, in the world."""
    xp = _xp(bend.start)
    flat = j.ndim == 1
    j = j[:, None] if flat else j
    k = xp.maximum(j - bend.anchor[:, None], 0).astype(float)
    blend = xp.clip(k / BEND_START, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    lead = bend.start[:, None] + bend.drift[:, None] * (k * SPACING)[..., None]
    lead = lead * (1.0 - blend)[..., None]
    knots = (0, *BEND_POINTS, BEND_FADE)
    weights = []
    for i in range(1, len(knots) - 1):
        rise = (k - knots[i - 1]) / (knots[i] - knots[i - 1])
        fall = (knots[i + 1] - k) / (knots[i + 1] - knots[i])
        weights.append(xp.clip(xp.minimum(rise, fall), 0.0, 1.0))
    curve = xp.einsum("nkp,npd->nkd", xp.stack(weights, axis=-1), bend.points)
    out = lead + curve
    return out[:, 0] if flat else out


def bender_features(
    feats: NDArray,
    pos: NDArray,
    index: NDArray,
    passed: NDArray,
    rows: NDArray,
    lib: GuideLibrary,
    previous: NDArray,
    seen_gates: NDArray,
    seen_poles: NDArray,
    poles: NDArray,
) -> NDArray:
    """Return a bender's input, ``(n, BENDER_DIM)``.

    Args:
        feats: The tracker's features on the guide as planned ``(n, OBS_DIM)``.
        pos: Positions ``(n, 3)``.
        index: Current beads ``(n,)``.
        passed: Passes so far ``(n,)``.
        rows: Rows in the library ``(n,)``.
        lib: The library.
        previous: The bender's previous output ``(n, BEND_DIM)``.
        seen_gates: Whether the sensor has reached each gate ``(n, g)``.
        seen_poles: Whether it has reached each pole ``(n, p)``.
        poles: Poles as the sensor shows them ``(n, p, 3)``.
    """
    xp = _xp(pos)
    n = pos.shape[0]
    worlds = xp.arange(n)
    c, s = scene_heading(index, rows, lib)
    m = lib.beads.shape[1]
    ahead = [_turn(lib.beads[rows, xp.minimum(index + k, m - 1)] - pos, c, s) for k in BENDER_AHEAD]
    count = lib.sequence.shape[0]
    flags = [
        seen_gates[worlds, lib.sequence[xp.minimum(passed + i, count - 1)]][:, None] for i in (0, 1)
    ]
    near = xp.argsort(xp.linalg.norm(poles[..., :2] - pos[:, None, :2], axis=-1), axis=1)[:, :2]
    flags.append(xp.take_along_axis(seen_poles, near, axis=1))
    return xp.concatenate(
        [feats, *ahead, xp.clip(previous, -1.0, 1.0), *(f.astype(feats.dtype) for f in flags)],
        axis=1,
    ).astype(np.float32)


# endregion


#: The three styles every training plan trains, in the order the selector numbers them.
STYLE_NAMES = ("speed_first", "middle_ground", "safety_first")
#: The style flown from the start until the selector's first decision.
FIRST_STYLE = STYLE_NAMES.index("middle_ground")


def selector_features(feats: NDArray, style: NDArray, proposals: NDArray) -> NDArray:
    """Return the inputs of the style selector.

    Args:
        feats: Scene features, ``(n, OBS_DIM)``.
        style: The style flown now, ``(n,)``, numbered as in `STYLE_NAMES`.
        proposals: What every style would command now, ``(n, 3, ACT_DIM)``, clipped to [-1, 1].

    Returns:
        ``(n, OBS_DIM + 3 + 3 * ACT_DIM)``: the features, the style as one-hot, and the proposals.
    """
    xp = _xp(feats)
    one_hot = (style[:, None] == xp.arange(len(STYLE_NAMES))).astype(feats.dtype)
    flat = proposals.reshape(proposals.shape[0], -1).astype(feats.dtype)
    return xp.concatenate((feats, one_hot, flat), axis=1)


__all__ = [
    "ACT_DIM",
    "BEND_DIM",
    "BENDER_DIM",
    "Bend",
    "FIRST_STYLE",
    "GuideLibrary",
    "StagedScenarios",
    "OBS_DIM",
    "RUN_OUT_BEADS",
    "STYLE_NAMES",
    "advance_scene_guide",
    "bend_offset",
    "bender_features",
    "load_library",
    "load_staged_library",
    "make_bend",
    "reveal_order",
    "scene_action",
    "scene_clearances",
    "scene_features",
    "scene_heading",
    "selector_features",
    "staged_layout",
    "track_library",
    "zero_bend",
]
