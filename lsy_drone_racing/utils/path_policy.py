"""A learned closed-loop controller that flies a path: state and path go in, attitude comes out.

The same `features` serve training and flight. Attitude goes in and out as a rotation matrix, which
has no jumps, unlike Euler angles near a pitch of 90 degrees; `to_action` turns the output into the
angles the attitude controller takes. Everything is in the world frame and relative to the drone,
so the policy never sees where on the track it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.control.path_bend import ArcPath

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    from jax import Array
    from numpy.typing import NDArray

SPACING = 0.05
#: Beads ahead of the drone the policy sees: 0.2 m to 3 m.
LOOK_AHEAD = (4, 8, 12, 16, 20, 30, 40, 60)
#: Beads ahead at which it sees the plan's speed.
SPEED_AHEAD = (0, 8, 20, 40)
#: How far ahead the nearest bead is searched per step.
SEARCH = 24
#: Two columns of a rotation matrix and the thrust.
ACT_DIM = 7
OBS_DIM = 3 + 3 + 9 + ACT_DIM + 3 + 3 * len(LOOK_AHEAD) + len(SPEED_AHEAD)
TILT_MAX = 1.4
#: The race's action space ends here.
YAW_MAX = np.pi / 2
THRUST_MIN = 0.0857
THRUST_MAX = 0.8


def resample(points: NDArray[np.floating], spacing: float = SPACING) -> NDArray[np.floating]:
    """Return a polyline resampled at an exact spacing."""
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    grid = np.arange(0.0, arc[-1], spacing)
    return np.stack([np.interp(grid, arc, points[:, i]) for i in range(3)], axis=1)


def _xp(array: Array | NDArray) -> ModuleType:
    """Return the array module of an array, so training can compile the same code flight runs."""
    return jnp if isinstance(array, jax.Array) else np


def rotation_matrix(quat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the rotation matrices of ``xyzw`` quaternions, flattened row by row, ``(..., 9)``."""
    xp = _xp(quat)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return xp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )


def _columns_to_rpy(first: NDArray[np.floating], second: NDArray[np.floating]) -> tuple:
    """Return the roll, pitch and yaw of the rotation nearest to two columns of a matrix.

    Gram-Schmidt makes the columns orthonormal and their cross product gives the third. The angles
    are the attitude controller's: ``Rotation.from_euler("xyz", [roll, pitch, yaw])``, ``Rz Ry Rx``.
    """
    xp = _xp(first)
    x = first / xp.maximum(xp.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    y = second - xp.sum(x * second, axis=-1, keepdims=True) * x
    y = y / xp.maximum(xp.linalg.norm(y, axis=-1, keepdims=True), 1e-6)
    z = xp.cross(x, y)
    roll = xp.arctan2(y[..., 2], z[..., 2])
    pitch = xp.arcsin(xp.clip(-x[..., 2], -1.0, 1.0))
    yaw = xp.arctan2(x[..., 1], x[..., 0])
    return roll, pitch, yaw


def nearest(
    pos: NDArray[np.floating],
    paths: NDArray[np.floating],
    index: NDArray[np.integer],
    rows: NDArray[np.integer] | None = None,
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """Advance every world's bead index to the nearest bead within the search window.

    Args:
        pos: Drone positions ``(n, 3)``.
        paths: Paths ``(p, m, 3)``.
        index: Current bead of every world ``(n,)``.
        rows: The path of every world ``(n,)``. Defaults to one path per world, in order.

    Returns:
        The new index and the distance to that bead.
    """
    xp = _xp(pos)
    n, m = pos.shape[0], paths.shape[1]
    worlds = xp.arange(n)
    rows = worlds if rows is None else rows
    window = xp.minimum(index[:, None] + xp.arange(SEARCH), m - 1)
    points = paths[rows[:, None], window]
    d = xp.linalg.norm(points - pos[:, None], axis=2)
    k = d.argmin(axis=1)
    return window[worlds, k], d[worlds, k]


def features(
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    quat: NDArray[np.floating],
    ang_vel: NDArray[np.floating],
    previous: NDArray[np.floating],
    paths: NDArray[np.floating],
    speeds: NDArray[np.floating],
    index: NDArray[np.integer],
    rows: NDArray[np.integer] | None = None,
) -> NDArray[np.floating]:
    """Return the policy's input, ``(n, OBS_DIM)``.

    `paths`, `speeds` and `rows` are as in `nearest`; `rows` lets many worlds share a pool of paths
    without copying a whole path per world.
    """
    xp = _xp(pos)
    m = paths.shape[1]
    rows = xp.arange(pos.shape[0]) if rows is None else rows
    here = pos - paths[rows, index]
    ahead = [paths[rows, xp.minimum(index + k, m - 1)] - pos for k in LOOK_AHEAD]
    wanted = [speeds[rows, xp.minimum(index + k, m - 1)][:, None] for k in SPEED_AHEAD]
    return xp.concatenate(
        [vel, ang_vel, rotation_matrix(quat), previous, here, *ahead, *wanted], axis=1
    ).astype(np.float32)


def to_action(u: NDArray[np.floating], yaw_max: float = YAW_MAX) -> NDArray[np.floating]:
    """Map the policy's ``[-1, 1]`` output to the attitude controller's command.

    The first six outputs are the first two columns of the wanted rotation matrix, as an offset
    from level, so a network that outputs zero asks for level flight. The seventh is the thrust.

    Args:
        u: The policy's output, ``(..., ACT_DIM)``.
        yaw_max: Limit on the yaw command [rad]; yaw does not help the drone follow a path.

    Returns:
        ``[roll, pitch, yaw, thrust]``.
    """
    xp = _xp(u)
    u = xp.clip(u, -1.0, 1.0)
    first = u[..., :3] + xp.asarray([1.0, 0.0, 0.0], dtype=u.dtype)
    second = u[..., 3:6] + xp.asarray([0.0, 1.0, 0.0], dtype=u.dtype)
    roll, pitch, yaw = _columns_to_rpy(first, second)
    thrust = THRUST_MIN + 0.5 * (u[..., 6] + 1.0) * (THRUST_MAX - THRUST_MIN)
    return xp.stack(
        (
            xp.clip(roll, -TILT_MAX, TILT_MAX),
            xp.clip(pitch, -TILT_MAX, TILT_MAX),
            xp.clip(yaw, -yaw_max, yaw_max),
            thrust,
        ),
        -1,
    )


class PathPolicy:
    """The trained network, in plain NumPy."""

    def __init__(self, file: str | Path):
        """Load a policy written by `scripts/train_path_policy.py`."""
        data = np.load(file)
        self.mean, self.std = data["obs_mean"], data["obs_std"]
        count = int(data["n_layers"])
        self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(count)]
        self.yaw_max = float(data["yaw_max"]) if "yaw_max" in data.files else YAW_MAX

    def __call__(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the mean action for a batch of features."""
        x = np.clip((x - self.mean) / self.std, -10.0, 10.0)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x


class PathPolicyTracker:
    """Flies one path with a `PathPolicy`; the interface `mpcc_bench.fly` expects of a tracker."""

    def __init__(
        self, policy: PathPolicy, beads: NDArray[np.floating], limit: NDArray[np.floating]
    ):
        """Prepare a flight.

        Args:
            policy: The network.
            beads: The path.
            limit: The plan's speed at every bead.
        """
        run_out = beads[-1] + np.outer(
            np.arange(1, 81) * SPACING,
            (beads[-1] - beads[-2]) / np.linalg.norm(beads[-1] - beads[-2]),
        )
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(beads, axis=0), axis=1))]
        self._arc = np.r_[arc, arc[-1] + np.arange(1, 81) * SPACING]
        self._beads = np.r_[beads, run_out][None]
        self._limit = np.r_[limit, np.full(80, limit[-1])][None]
        self._policy = policy
        self._index = np.zeros(1, dtype=int)
        self._previous = np.zeros((1, ACT_DIM))
        self._path = ArcPath(self._beads[0])
        self._s = 0.0
        self._failures = 0

    def compute_control(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, thrust]`` for the current state."""
        keys = ("pos", "vel", "quat", "ang_vel")
        pos, vel, quat, w = (np.asarray(obs[k], dtype=float)[None] for k in keys)
        self._index, _ = nearest(pos, self._beads, self._index)
        self._s = float(self._arc[self._index[0]])
        x = features(pos, vel, quat, w, self._previous, self._beads, self._limit, self._index)
        u = np.clip(self._policy(x), -1.0, 1.0)
        self._previous = u
        return to_action(u, self._policy.yaw_max)[0].astype(np.float32)
