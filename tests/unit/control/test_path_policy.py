"""Training compiles `path_policy` in JAX and flight runs it in NumPy: they must agree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pathlib import Path

from lsy_drone_racing.utils.path_policy import (
    ACT_DIM,
    OBS_DIM,
    TILT_MAX,
    PathPolicy,
    features,
    nearest,
    rotation_matrix,
    to_action,
)


@pytest.fixture
def scene() -> dict:
    """Return four worlds that share a pool of two paths."""
    rng = np.random.default_rng(0)
    beads = np.arange(100)[:, None] * np.array([0.05, 0.0, 0.0]) + np.array([0.0, 0.0, 1.0])
    paths = np.stack([beads, beads[:, [1, 0, 2]]]).astype(np.float32)
    quat = rng.normal(size=(4, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {
        "pos": rng.normal(1.0, 0.1, (4, 3)).astype(np.float32),
        "vel": rng.normal(size=(4, 3)).astype(np.float32),
        "quat": quat.astype(np.float32),
        "ang_vel": rng.normal(size=(4, 3)).astype(np.float32),
        "previous": rng.uniform(-1, 1, (4, ACT_DIM)).astype(np.float32),
        "paths": paths,
        "speeds": rng.uniform(0, 8, (2, 100)).astype(np.float32),
        "index": np.array([0, 10, 90, 99]),
        "rows": np.array([1, 0, 0, 1]),
    }


def test_features_agree_between_numpy_and_jax(scene: dict):
    """The compiled features are the ones the policy sees in flight."""
    in_jax = {k: jnp.asarray(v) for k, v in scene.items()}
    a, b = features(**scene), features(**in_jax)
    assert a.shape == (4, OBS_DIM)
    assert np.allclose(a, np.asarray(b), atol=1e-5)


def test_nearest_agrees_between_numpy_and_jax(scene: dict):
    """Both search the same window of the same path."""
    args = ("pos", "paths", "index", "rows")
    i_np, d_np = nearest(*(scene[k] for k in args))
    i_jax, d_jax = nearest(*(jnp.asarray(scene[k]) for k in args))
    assert np.array_equal(i_np, np.asarray(i_jax))
    assert np.allclose(d_np, np.asarray(d_jax), atol=1e-5)


def test_pool_gives_the_same_features_as_a_path_per_world(scene: dict):
    """Sharing a pool through `rows` only saves copies."""
    per_world = dict(scene)
    per_world["paths"] = scene["paths"][scene["rows"]]
    per_world["speeds"] = scene["speeds"][scene["rows"]]
    del per_world["rows"]
    assert np.array_equal(features(**scene), features(**per_world))


def test_action_agrees_between_numpy_and_jax():
    """The mapping to attitude and thrust is the same."""
    u = np.random.default_rng(0).uniform(-1.5, 1.5, (64, ACT_DIM)).astype(np.float32)
    assert np.allclose(to_action(u), np.asarray(to_action(jnp.asarray(u))), atol=1e-5)


def test_rotation_matrix_matches_scipy():
    """The features hold the true rotation matrix, row by row."""
    rot = Rotation.random(32, random_state=0)
    assert np.allclose(rotation_matrix(rot.as_quat()), rot.as_matrix().reshape(-1, 9))


def test_zero_output_asks_for_level_flight_at_about_hover():
    """An untrained network must not ask for anything drastic."""
    roll, pitch, yaw, thrust = to_action(np.zeros(ACT_DIM))
    assert (roll, pitch, yaw) == (0.0, 0.0, 0.0)
    assert 0.0434 * 9.81 < thrust < 0.0434 * 11.0


def test_action_gives_the_controllers_angles_for_a_rotation_matrix():
    """A matrix's two columns come back as the angles the controller turns into that matrix."""
    rng = np.random.default_rng(0)
    rpy = rng.uniform([-1.3, -1.3, -1.3], [1.3, 1.3, 1.3], (256, 3))
    matrix = Rotation.from_euler("xyz", rpy).as_matrix()
    # Stay where an output in [-1, 1] can reach: the offsets from level must fit.
    u = np.c_[matrix[..., 0] - [1, 0, 0], matrix[..., 1] - [0, 1, 0], np.zeros(len(rpy))]
    fits = np.abs(u).max(axis=1) <= 1.0
    assert fits.sum() > 50
    assert np.allclose(to_action(u[fits])[:, :3], rpy[fits], atol=1e-6)


def test_yaw_limit_holds_yaw_and_leaves_the_rest_finite():
    """A limit of zero always commands zero yaw."""
    u = np.random.default_rng(2).uniform(-1.0, 1.0, (256, ACT_DIM))
    action = to_action(u, 0.0)
    assert np.all(action[:, 2] == 0.0)
    assert np.all(np.isfinite(action))


def test_action_keeps_angles_inside_the_limits():
    """Whatever the network outputs, the command stays where the race accepts it."""
    u = np.random.default_rng(1).uniform(-1.0, 1.0, (4096, ACT_DIM))
    action = to_action(u)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action[:, :2]) <= TILT_MAX)
    assert np.all(np.abs(action[:, 2]) <= np.pi / 2)


def test_an_input_that_never_varied_in_training_is_ignored(tmp_path: Path):
    rng = np.random.default_rng(0)
    w0, w1 = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    file = tmp_path / "policy.npz"
    np.savez(
        file,
        obs_mean=np.zeros(3),
        obs_std=np.array([1.0, 1.24e-6, 0.5]),
        n_layers=2,
        W0=w0,
        b0=np.zeros(4),
        W1=w1,
        b1=np.zeros(2),
    )
    policy = PathPolicy(file)
    flat = policy(np.array([[0.3, 0.0, -0.2]]))
    # A gate tilted by 0.1 rad: the input had no spread in training and must not change the action.
    tilted = policy(np.array([[0.3, 0.1, -0.2]]))
    assert np.allclose(flat, tilted)
    assert not np.allclose(flat, policy(np.array([[0.3, 0.0, 0.2]])))
