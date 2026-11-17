"""The MPCC on simple paths: does it use the drone, and does it stay on the path?

These fly the real solver and the real dynamics, so the first run compiles a solver per setting.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsy_drone_racing.utils.mpcc_bench import (
    HIGH_THRUST,
    PRIMITIVES,
    THRUST_ACC_MAX,
    fly,
    speed_profile,
    with_settings,
)

#: What `scripts/tune_mpcc_primitives.py` found on these paths, with the limit at full thrust. It
#: uses more of the drone here but is too slow in the race, where the limit comes from the plan.
TUNED = {
    "q_contour": 25.0,
    "q_contour_z": 25.0,
    "q_progress": 15.0,
    "q_rate": 2.0,
    "q_dthrust": 0.025,
    "q_dds": 0.005,
    "k_dist": 0.3,
    "w_tunnel": 16.0,
    "w_speed": 0.0625,
    "progress_add": 1.0,
    "tilt_max": 1.25,
    "tilt_lead": 0.12,
    "thrust_max": 0.77,
    "tunnel_max": 0.1,
    "horizon": 50,
    "dds_max": 15.0,
    "dist_share": 0.0,
}


@pytest.mark.unit
@pytest.mark.parametrize("name", list(PRIMITIVES))
def test_speed_profile_starts_at_rest_and_stays_positive(name: str):
    speed, ideal = speed_profile(PRIMITIVES[name])
    assert speed[0] == 0.0
    assert np.all(speed[1:] > 0.0)
    assert ideal > 0.0


@pytest.mark.unit
def test_speed_profile_respects_the_turn():
    speed, _ = speed_profile(PRIMITIVES["hairpin"])
    level = np.sqrt(THRUST_ACC_MAX**2 - 9.81**2)
    assert speed.min(initial=9.0, where=speed > 0.0) <= np.sqrt(level * 0.35) + 0.05
    assert speed.max() > np.sqrt(level * 0.35)


@pytest.mark.unit
@pytest.mark.parametrize("name", list(PRIMITIVES))
def test_default_settings_fly_every_path(name: str):
    flight = fly(name)
    assert flight.finished
    assert flight.failures == 0
    assert flight.time_ratio < 1.8
    assert flight.error_max < 0.3


@pytest.mark.unit
@pytest.mark.parametrize("name", list(PRIMITIVES))
def test_tuned_settings_are_fast_and_tight(name: str):
    flight = fly(name, with_settings(**TUNED))
    assert flight.finished
    assert flight.failures == 0
    assert flight.time_ratio < 1.5
    assert flight.error_max < 0.12


@pytest.mark.unit
def test_tuned_settings_use_more_thrust_than_the_defaults():
    default = [fly(name) for name in PRIMITIVES]
    tuned = [fly(name, with_settings(**TUNED)) for name in PRIMITIVES]
    assert np.mean([f.thrust_high for f in tuned]) > np.mean([f.thrust_high for f in default])
    assert np.mean([f.time_ratio for f in tuned]) < np.mean([f.time_ratio for f in default])
    assert np.mean([f.error_max for f in tuned]) < np.mean([f.error_max for f in default])
    assert HIGH_THRUST < THRUST_ACC_MAX
