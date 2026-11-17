"""Integration test for the final controller, `RLScene`, on level 0.

On a new track the controller first solves a time-optimal guide, which takes several seconds, so
this test is opt-in:

    RUN_FINAL_CONTROLLER=1 pytest tests/integration/test_final_controller.py
"""

import os
from pathlib import Path

import gymnasium
import pytest
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FINAL_CONTROLLER") != "1",
    reason="set RUN_FINAL_CONTROLLER=1 to plan a guide and fly the final controller",
)

REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.integration
def test_final_controller_finishes_level0():
    """The controller that the level configurations select has to complete the level-0 track."""
    config = load_config(REPO_ROOT / "config/level0.toml")
    config.sim.render = False
    ctrl_cls = load_controller(REPO_ROOT / "lsy_drone_racing/control" / config.controller.file)
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=0,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=0)
    ctrl = ctrl_cls(obs, info, config)
    steps = 0
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    n_passes = int(obs["gate_sequence"].shape[0])
    assert int(obs["n_gates_passed"]) == n_passes, "The final controller did not finish level 0"
    assert steps / config.env.freq < 20.0
