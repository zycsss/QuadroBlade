"""Level2 safety parameters, geometry changes and new-path progress coordinates."""

import importlib
import sys
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
from crazyflow.dynamics import Dynamics, load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import my_controller
from lsy_drone_racing.control.my_controller import MPCCConfig
from lsy_drone_racing.control.path_planning import ArcPath, PathInfo
from lsy_drone_racing.utils import load_config, load_controller
from tests.unit.control.test_mpcc_baseline import ROOT, observation

pytestmark = pytest.mark.unit


@pytest.fixture
def level2(monkeypatch: pytest.MonkeyPatch) -> object:
    """Keep planning real and replace only native solver compilation/execution."""
    solver = SimpleNamespace(reset=lambda: None, set=lambda *args: None)
    monkeypatch.setattr(my_controller, "get_solver", lambda *args: (solver, None))
    cls = load_controller(ROOT / "lsy_drone_racing/control/mpcc_level2_controller.py")
    module = sys.modules[cls.__module__]
    monkeypatch.setattr(module, "get_obstacle_solver", lambda *args: (solver, None))
    cfg = load_config(ROOT / "config/level2.toml")
    cfg.controller.l2_unseen_speed = 1.1
    cfg.controller.l2_pole_speed = 0.8
    cfg.controller.l2_pole_radius = 0.5
    return cls(observation(-1), {}, cfg)


def test_risk_limit_uses_observed_flags_and_known_pole_proximity(level2: object):
    """Unrevealed gate/pole poses and known close poles constrain speed independently."""
    obs = observation(-0.9)
    obs["gates_visited"] = np.array([False])
    assert level2.risk_speed_limit(obs) == 1.1
    obs["gates_visited"][:] = True
    assert level2.risk_speed_limit(obs) is None
    obs["obstacles_pos"] = np.array([[-0.6, 0, 1.55]])
    obs["obstacles_visited"] = np.array([True])
    assert level2.risk_speed_limit(obs) == 0.8
    obs["gates_visited"][:] = False
    assert level2.risk_speed_limit(obs) == 0.8


def test_reprojection_accepts_new_arc_coordinate_and_logs_observation_time(level2: object):
    """Old-path monotonicity must not prevent a backwards coordinate change after replanning."""
    obs = observation(1.5)
    pts = np.array([[0.0, 0, 1], [10.0, 0, 1]])
    path = ArcPath(pts)
    level2._install_path(path, PathInfo(pts, obs["gates_pos"], obs["obstacles_pos"]))
    level2._theta = 3.0
    level2._tick = 10
    level2._reproject_wide = True
    level2._record_plan("re-plan: test pose update", obs)
    assert level2._theta == pytest.approx(1.5, abs=0.02)
    assert not level2._reproject_wide
    assert level2.plan_log[-1]["theta"] == pytest.approx(1.5, abs=0.02)
    assert level2.plan_log[-1]["t"] == pytest.approx(0.18)


def test_quaternion_only_reveal_is_detected_but_sign_flip_is_not(level2: object):
    """A gate orientation can change without a position update; q and -q are equivalent."""
    obs = observation(-1)
    obs["gates_quat"] = -obs["gates_quat"]
    assert level2._track_info_changed(obs) == []
    obs["gates_quat"] = Rotation.from_euler("z", 0.15).as_quat()[None, :]
    assert level2._track_info_changed(obs)


def test_local_clearance_repairs_curve_and_rejects_unrepairable_crossing():
    """Final spline interpolation must not silently reinstate a pole intersection."""
    module = importlib.import_module("lsy_drone_racing.control.mpcc_level2_controller")
    original = ArcPath(np.array([[-1.0, 0.12, 1], [1.0, 0.12, 1]]))
    fixed = module.enforce_pole_clearance(original, np.array([[0.0, 0, 1.55]]), 0.18)
    p = fixed.pos(np.linspace(0, fixed.length, 4000))
    assert np.linalg.norm(p[:, :2], axis=1).min() >= 0.177
    np.testing.assert_allclose(original.pos(0), [-1, 0.12, 1])
    central = ArcPath(np.array([[-1.0, 0, 1], [1.0, 0, 1]]))
    with pytest.raises(ValueError, match="clearance"):
        module.enforce_pole_clearance(central, np.array([[0.0, 0, 1.55]]), 0.18)


def test_obstacle_parameters_follow_live_observations_without_mutating_reference():
    """The solver sees reported poses and per-object uncertainty, never hidden track truth."""
    module = importlib.import_module("lsy_drone_racing.control.level2_solver")
    written = []
    backend = SimpleNamespace(set=lambda k, field, value: written.append(value.copy()))
    obs = {
        "obstacles_pos": np.array([[1.0, 2, 1.55], [3.0, 4, 1.55]]),
        "obstacles_visited": np.array([False, True]),
    }
    adapter = module.ObservedObstacleSolver(backend, lambda: obs, 0.18, 0.10)
    reference = np.arange(7, dtype=float)
    adapter.set(0, "p", reference)
    np.testing.assert_allclose(written[-1], [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 0.28, 0.18])
    np.testing.assert_allclose(reference, np.arange(7))
    obs["obstacles_pos"][0, 0] = 1.2
    obs["obstacles_visited"][0] = True
    adapter.set(0, "p", reference)
    assert written[-1][7] == 1.2
    np.testing.assert_allclose(written[-1][-2:], [0.18, 0.18])


def test_ocp_has_distance_constraints_and_preserves_actuator_limits():
    """Constraint signs/radii must keep outside positive and retain the original thrust bounds."""
    module = importlib.import_module("lsy_drone_racing.control.level2_solver")
    params = load_params(Dynamics.so_rpy, "cf21B_500")
    cfg = MPCCConfig()
    ocp = module.build_obstacle_ocp(cfg, params, 1)
    h = cs.Function("distance_test", [ocp.model.x, ocp.model.p], [ocp.model.con_h_expr])
    x = np.zeros(15)
    x[0] = 1.0
    p = np.zeros(10)
    p[-1] = 0.2
    assert float(h(x, p)) == pytest.approx(0.96)
    x[0] = 0.1
    assert float(h(x, p)) == pytest.approx(-0.03)
    assert len(ocp.constraints.idxsh) == 1
    assert ocp.constraints.ubx[3] == pytest.approx(0.8)
    assert ocp.solver_options.N_horizon == 40


def test_level0_falls_back_without_changing_shared_configuration(monkeypatch: pytest.MonkeyPatch):
    """The new entry cannot silently apply randomized-track pacing to existing nominal races."""
    solver = SimpleNamespace(reset=lambda: None, set=lambda *args: None)
    monkeypatch.setattr(my_controller, "get_solver", lambda *args: (solver, None))
    cls = load_controller(ROOT / "lsy_drone_racing/control/mpcc_level2_controller.py")
    cfg = load_config(ROOT / "config/level0.toml")
    before = cfg.to_dict()
    ctrl = cls(observation(-1), {}, cfg)
    assert cfg.to_dict() == before
    assert ctrl.settings_snapshot()["mpcc"]["v_max"] == 2.3
    assert ctrl.risk_speed_limit(observation(0)) is None


def test_reused_controller_restores_routed_solver_seed(monkeypatch: pytest.MonkeyPatch):
    """Every new episode must seed the nonconvex solver after its iterates are reset."""
    written = []
    solver = SimpleNamespace(
        reset=lambda: written.clear(),
        set=lambda k, field, value: written.append((k, field, value.copy())),
    )
    monkeypatch.setattr(my_controller, "get_solver", lambda *args: (solver, None))
    cls = load_controller(ROOT / "lsy_drone_racing/control/mpcc_level2_controller.py")
    monkeypatch.setattr(
        sys.modules[cls.__module__], "get_obstacle_solver", lambda *a: (solver, None)
    )
    obs = observation(-1)
    obs["obstacles_pos"] = np.array([[3.0, 3, 1.55]])
    obs["obstacles_visited"] = np.array([True])
    ctrl = cls(obs, {}, load_config(ROOT / "config/level2.toml"))
    assert len([v for _, field, v in written if field == "x"]) == ctrl._mpcc.N + 1
    ctrl.episode_reset()
    new_obs = {**obs, "pos": np.array([-1.2, 0, 1.0])}
    ctrl._init_episode(new_obs)
    states = [v for _, field, v in written if field == "x"]
    assert len(states) == ctrl._mpcc.N + 1
    np.testing.assert_allclose(states[0][:3], ctrl._path.pos(0))
    np.testing.assert_allclose(states[0][:3], new_obs["pos"], atol=0.03)


def test_fully_randomized_level3_is_rejected():
    """Level2 validation must not be promoted to a fully randomized course without evidence."""
    cls = load_controller(ROOT / "lsy_drone_racing/control/mpcc_level2_controller.py")
    with pytest.raises(ValueError, match="Level 3"):
        cls(observation(-1), {}, load_config(ROOT / "config/level3.toml"))
