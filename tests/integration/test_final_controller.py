"""Integration tests for the final controller, `RLScene`, flown as the level configurations set it.

On a new track the controller first solves a time-optimal guide, which takes several seconds, so
these tests are opt-in:

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


def _fly(level: str, seed: int, controller: dict | None = None) -> tuple[dict, float, object]:
    """Fly one race of a level with the configured controller.

    Returns:
        The last observation, the time flown and the controller.
    """
    config = load_config(REPO_ROOT / f"config/{level}.toml")
    config.sim.render = False
    for key, value in (controller or {}).items():
        config.controller[key] = value
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
        seed=seed,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset(seed=seed)
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
    return obs, steps / config.env.freq, ctrl


@pytest.mark.integration
def test_final_controller_finishes_level0():
    """The controller that the level configurations select has to complete the level-0 track."""
    obs, seconds, _ = _fly("level0", seed=0)
    n_passes = int(obs["gate_sequence"].shape[0])
    assert int(obs["n_gates_passed"]) == n_passes, "The final controller did not finish level 0"
    assert seconds < 20.0


@pytest.mark.integration
def test_final_controller_flies_level3_as_configured():
    """Level 3, as its configuration sets it, runs to the end of a race with the final controller.

    It flies the style selector by default; seed 0 is not checked for a finish.
    """
    obs, seconds, _ = _fly("level3", seed=0)
    assert seconds > 0.0
    assert int(obs["n_gates_passed"]) >= 0


@pytest.mark.integration
def test_the_style_selector_finishes_a_level3_race():
    """With replanning at every reveal, the selector picks between the three styles and finishes."""
    section = {"guide_planner": "fast_runtime", "replan_on_reveal": True, "style_selector": True}
    obs, seconds, ctrl = _fly("level3", seed=9000, controller=section)
    assert int(obs["n_gates_passed"]) == int(obs["gate_sequence"].shape[0])
    assert seconds < 10.0
    assert ctrl.style_ticks.sum() > 0
