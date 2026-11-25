"""A racing controller trained only on random local scenes, never on the track it flies.

The policy is trained by `scripts/train_scene_policy.py` and sees nothing that identifies a track,
see `lsy_drone_racing.utils.scene_policy`. The gates and poles come from the race's sensor: nominal
at first, and true once they are in range.

The guide comes from one of two planners, chosen with ``controller.guide_planner``:

* ``"topt"`` (default): `topt_planner`, solved once on the nominal track before the race, about 15 s
  on a new track. A policy trained on faster guides needs ``controller.fast_guides = true`` as well.
* ``"fast_runtime"``: `utils.fast_runtime_planner`, about 20 ms per plan. With
  ``controller.replan_on_reveal = true`` the guide is planned again whenever the sensor reveals a
  gate or pole, through the layout as now known. Its policy is ``rl_scene_runtime_guides.npz``,
  trained on guides from the same planner. With ``controller.replan_in_order = true`` as well, it
  replans only when the next object in reveal order is revealed, through a layout where exactly the
  objects revealed in order are true: the rule `scripts/train_scene_policy.py --staged` trains with.

With ``controller.replan_from_state = true`` as well, a replan starts where the drone is, at its
velocity, and goes only through the next ``controller.replan_gates`` gates (1 by default; 0 for the
rest of the lap). It also replans at every pass, so the guide never ends before the race does.

Point ``controller.policy`` in the configuration to another file to fly a different policy.

**Defaults.** A race that shifts gates or poles (levels 2 and 3: ``env.randomizations`` holds
``gate_pos``, ``gate_rpy`` or ``obstacle_pos``) flies the runtime planner, replanned at every
reveal, with the style selector below (`SHIFTED_RACE_DEFAULTS`). Any other race flies ``"topt"``
with ``rl_scene.npz``. Setting ``guide_planner``, ``policy``, ``replan_on_reveal`` or
``style_selector`` in ``[controller]`` replaces these defaults with the settings given.

With ``controller.style_selector = true`` (runtime planner only) the controller flies three
policies, ``speed_first``, ``middle_ground`` and ``safety_first``, and a selector trained by
`scripts/train_style_selector.py` picks which one flies, every ``hold`` control steps, from the
scene and what each style would command. ``controller.style_set`` chooses the styles:
``"stable"`` (default), trained with poles that pop up on the guide when revealed, or ``"fast"``,
the earlier set, which finishes fewer races but laps faster. Both fly under the same selector.
``controller.selector`` and ``controller.style_policies`` point to other files.

With ``controller.bender = true`` (runtime planner only) each policy flies with its own bender,
trained with it by `scripts/train_bender.py`: every few steps the bender bends the guide ahead,
starting at the drone, and the policy flies the bent guide. ``controller.benders`` lists one bender
file per policy; without the selector, ``controller.policy`` and ``controller.bender_policy`` name
the pair.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.topt_planner import FAST_GUIDE_CONFIG, RANDOM_TRACK_CONFIG
from lsy_drone_racing.utils.path_policy import ACT_DIM, PathPolicy
from lsy_drone_racing.utils.race_policy import RUN_OUT_BEADS, build_track, guide_arrays
from lsy_drone_racing.utils.scene_policy import (
    BEND_DIM,
    FIRST_STYLE,
    STYLE_NAMES,
    Bend,
    advance_scene_guide,
    bender_features,
    make_bend,
    reveal_order,
    scene_action,
    scene_features,
    scene_heading,
    selector_features,
    staged_layout,
    track_library,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_collections import ConfigDict
    from numpy.typing import NDArray

    from lsy_drone_racing.utils.scene_policy import GuideLibrary

DEFAULT_POLICY = Path(__file__).parent / "rl_scene.npz"
#: The policy trained on guides of the runtime planner.
RUNTIME_POLICY = Path(__file__).parent / "rl_scene_runtime_guides.npz"
#: The selector, which chooses between the three styles of either set.
SELECTOR_POLICY = Path(__file__).parent / "rl_scene_style_selector.npz"
#: The two sets of three styles, each in the order of `STYLE_NAMES`, chosen with
#: ``controller.style_set``. ``stable`` was trained with half the poles popping up on the guide in a
#: third of its scenarios, and finishes more races; ``fast`` was not, and laps are quicker.
STYLE_SETS = {
    style_set: tuple(
        Path(__file__).parent / f"rl_scene_{style_set}_{name}.npz" for name in STYLE_NAMES
    )
    for style_set in ("stable", "fast")
}
DEFAULT_STYLE_SET = "stable"
#: The three styles trained together with a bender, and those benders.
BENT_POLICIES = tuple(
    Path(__file__).parent / f"rl_scene_{name}_bent_tracker.npz" for name in STYLE_NAMES
)
BENDERS = tuple(Path(__file__).parent / f"rl_scene_{name}_bender.npz" for name in STYLE_NAMES)
#: The selector over the three pairs of a bender and its tracker.
BENT_SELECTOR_POLICY = Path(__file__).parent / "rl_scene_bent_style_selector.npz"
GUIDE_PLANNERS = ("topt", "fast_runtime")
#: Settings of a race whose gates or poles are shifted, unless the configuration chooses its own.
SHIFTED_RACE_DEFAULTS = {
    "guide_planner": "fast_runtime",
    "replan_on_reveal": True,
    "style_selector": True,
}
#: Settings that, given in ``[controller]``, replace `SHIFTED_RACE_DEFAULTS`.
_CHOSEN_KEYS = ("guide_planner", "policy", "replan_on_reveal", "style_selector")
#: Randomizations that move gates or poles away from where they are reported.
_SHIFTS = ("gate_pos", "gate_rpy", "obstacle_pos")


class _Pilot:
    """A policy, and the bender that bends its guide, if it has one."""

    def __init__(self, policy: str | Path, bender: str | Path | None = None):
        self.policy = PathPolicy(policy)
        self.bender = None
        self.bend: Bend | None = None
        self.u = np.zeros((1, BEND_DIM))
        if bender is not None:
            data = np.load(bender)
            self.hold, self.limit = int(data["hold"]), float(data["limit"])
            self.from_drone = bool(data.get("from_drone", False))
            self.bender = PathPolicy(bender)


def _controller_settings(config: ConfigDict) -> dict:
    """Return the ``[controller]`` settings, with a shifted race's defaults if none are chosen."""
    section = dict(config.get("controller") or {})
    shifts = config.get("env", {}).get("randomizations") or {}
    shifted = any(k in shifts for k in _SHIFTS)
    if shifted and not any(k in section for k in _CHOSEN_KEYS):
        section = SHIFTED_RACE_DEFAULTS | section
    return section


def _reported_track(
    obs: dict[str, NDArray[np.floating]],
    track: ConfigDict,
    start: NDArray[np.floating] | None = None,
    observed: bool = False,
) -> dict:
    """Return the track to plan on: the configured one, or the one the race reports.

    On a generated track the configuration only holds placeholders. What an observation shows is the
    layout, true where the sensor has reached and nominal elsewhere.

    Args:
        obs: An observation.
        track: The ``env.track`` section of the configuration.
        start: Where the drone started; its position in ``obs`` if None.
        observed: Use the observed layout also on a fixed track.

    Returns:
        A track section for `build_track`.
    """
    if not observed and not track.get("randomize", False):
        return track
    rpy = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_euler("xyz")
    gates = np.asarray(obs["gates_pos"], dtype=float)
    start = np.asarray(obs["pos"] if start is None else start, dtype=float)
    return {
        "gates": [{"pos": p, "rpy": r} for p, r in zip(gates, rpy, strict=True)],
        "obstacles": [{"pos": p} for p in np.asarray(obs["obstacles_pos"], dtype=float)],
        "drones": [{"pos": start}],
        "gate_order": track["gate_order"],
        "safety_limits": track["safety_limits"],
    }


def _revealed(obs: dict[str, NDArray]) -> tuple[bool, ...]:
    """Return which gates and poles the sensor has reached so far."""
    return tuple(np.asarray(obs["gates_visited"], dtype=bool).ravel()) + tuple(
        np.asarray(obs["obstacles_visited"], dtype=bool).ravel()
    )


def _index_between_gates(lib: GuideLibrary, pos: NDArray, passed: int) -> NDArray[np.integer]:
    """Return the bead nearest to the drone between its last and its next pass, on a new guide."""
    crossings = lib.crossings[0]
    low = int(crossings[passed - 1]) if passed > 0 else 0
    high = int(crossings[min(passed, len(crossings) - 1)])
    beads = lib.beads[0, low : high + 1]
    return np.array([low + int(np.linalg.norm(beads - pos, axis=1).argmin())])


class RLScene(Controller):
    """Flies the track with a policy trained on other tracks."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict):
        """Load the policy and plan the guide.

        Args:
            obs: The initial observation.
            info: The initial info dict.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        section = _controller_settings(config)
        self._planner = section.get("guide_planner", "topt")
        if self._planner not in GUIDE_PLANNERS:
            raise ValueError(f"guide_planner must be one of {GUIDE_PLANNERS}, not {self._planner}")
        self._replan = bool(section.get("replan_on_reveal", False))
        if self._replan and self._planner != "fast_runtime":
            raise ValueError("replan_on_reveal needs guide_planner = 'fast_runtime'")
        self._in_order = bool(section.get("replan_in_order", False))
        if self._in_order and not self._replan:
            raise ValueError("replan_in_order needs replan_on_reveal = true")
        self._from_state = bool(section.get("replan_from_state", False))
        if self._from_state and (not self._replan or self._in_order):
            raise ValueError("replan_from_state needs replan_on_reveal and not replan_in_order")
        self._replan_gates = int(section.get("replan_gates", 1))
        runtime = self._planner == "fast_runtime"
        self._selector = None
        bent = bool(section.get("bender", False))
        if bent and not runtime:
            raise ValueError("bender needs guide_planner = 'fast_runtime'")
        if section.get("style_selector", False):
            if not runtime:
                raise ValueError("style_selector needs guide_planner = 'fast_runtime'")
            if "policy" in section:
                raise ValueError("style_selector flies style_policies, not policy")
            file = section.get("selector", BENT_SELECTOR_POLICY if bent else SELECTOR_POLICY)
            self._hold = int(np.load(file)["hold"])
            self._selector = PathPolicy(file)
            style_set = section.get("style_set", DEFAULT_STYLE_SET)
            if style_set not in STYLE_SETS:
                raise ValueError(f"style_set must be one of {tuple(STYLE_SETS)}")
            styles = BENT_POLICIES if bent else STYLE_SETS[style_set]
            files = section.get("style_policies", styles)
            benders = section.get("benders", BENDERS) if bent else (None,) * len(files)
            if len(files) != len(STYLE_NAMES) or len(benders) != len(STYLE_NAMES):
                raise ValueError(f"style_policies needs one file per style of {STYLE_NAMES}")
            self._pilots = [_Pilot(f, b) for f, b in zip(files, benders, strict=True)]
        else:
            policy = section.get("policy", RUNTIME_POLICY if runtime else DEFAULT_POLICY)
            if bent and "bender_policy" not in section:
                raise ValueError("bender without style_selector needs bender_policy")
            self._pilots = [_Pilot(policy, section.get("bender_policy") if bent else None)]
        #: Control steps flown in each style, in the order of `STYLE_NAMES`.
        self.style_ticks = np.zeros(len(STYLE_NAMES), dtype=int)
        self._track_cfg = config.env.track
        self._start = np.asarray(obs["pos"], dtype=float)
        #: Every plan made in flight: its milliseconds, whether it passed its check, and how many
        #: objects were revealed by then.
        self.replans: list[dict] = []
        if runtime:
            self._library, self.first_plan_clear = self._runtime_library(obs)
        else:
            limits = FAST_GUIDE_CONFIG if section.get("fast_guides", False) else RANDOM_TRACK_CONFIG
            generated = self._track_cfg.get("randomize", False)
            track = build_track(
                _reported_track(obs, self._track_cfg), config=limits if generated else None
            )
            rpy = Rotation.from_matrix(track.gates_rot).as_euler("xyz")
            self._library = track_library(track, rpy)
            self.first_plan_clear = True
        self._known = _revealed(obs)
        self._planned_at_pass = 0
        if self._in_order:
            # The order, and the layout it starts from, are fixed by what was reported at the start.
            rpy = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_euler("xyz")
            gates_pos = np.asarray(obs["gates_pos"], dtype=float)
            poles = np.asarray(obs["obstacles_pos"], dtype=float)
            self._nominal = (gates_pos, rpy, poles)
            self._order, _ = reveal_order(
                self._library.beads[0], gates_pos, poles, float(config.env.sensor_range)
            )
            self._stage = 0
        self._rows = np.zeros(1, dtype=int)
        self.episode_reset()

    def _runtime_library(self, obs: dict[str, NDArray]) -> tuple[GuideLibrary, bool]:
        """Plan a runtime guide through the layout as observed, from where the drone started."""
        from lsy_drone_racing.utils.fast_runtime_planner import runtime_plan

        made = {}

        def plan(track: object, start: NDArray, start_vel: NDArray, _: object) -> object:
            trajectory, planned, _tries = runtime_plan(track, start, start_vel)
            made["clear"] = planned.clear
            return trajectory

        track = build_track(
            _reported_track(obs, self._track_cfg, self._start, observed=True),
            plan_fn=plan,
            smooth=True,
        )
        rpy = Rotation.from_matrix(track.gates_rot).as_euler("xyz")
        return track_library(track, rpy), made["clear"]

    def _state_library(
        self, obs: dict[str, NDArray], pos: NDArray, vel: NDArray, passed: int
    ) -> tuple[GuideLibrary, bool]:
        """Plan from the drone's position and velocity through its next ``replan_gates`` gates.

        The guide starts at the drone. Passes already made cross at its first bead, and passes
        beyond the plan at its last, so `advance_scene_guide` keeps the drone between the right
        gates.
        """
        from lsy_drone_racing.control.topt_planner import Track
        from lsy_drone_racing.utils.fast_runtime_planner import runtime_plan

        lib = self._library
        count = len(lib.sequence)
        end = count if self._replan_gates <= 0 else min(passed + self._replan_gates, count)
        rot = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float))
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        poles = np.asarray(obs["obstacles_pos"], dtype=float)
        sequence = np.asarray(lib.sequence[passed:end])
        track = Track(
            gates_pos,
            rot.as_matrix(),
            poles,
            sequence,
            np.asarray(lib.directions[passed:end], dtype=float),
            tuple(False for _ in sequence),
        )
        trajectory, planned, _ = runtime_plan(track, pos, vel)
        guide = guide_arrays(trajectory, smooth=True)
        last = len(guide["beads"]) - RUN_OUT_BEADS - 1
        crossings = np.full(count, last, dtype=int)
        crossings[:passed] = 0
        crossings[passed:end] = guide["crossings"]
        library = lib._replace(
            beads=guide["beads"][None],
            speeds=guide["speeds"][None],
            velocities=guide["velocities"][None],
            quats=guide["quats"][None],
            crossings=crossings[None],
            gates_pos=gates_pos[None],
            gates_rpy=rot.as_euler("xyz")[None],
            poles=poles[None],
        )
        return library, planned.clear

    def episode_reset(self):
        """Start again at the beginning of the guide."""
        self._index = np.zeros(1, dtype=int)
        self._previous = np.zeros((1, ACT_DIM), dtype=np.float32)
        self._style = FIRST_STYLE if self._selector is not None else 0
        self._tick = 0
        for pilot in getattr(self, "_pilots", ()):
            pilot.bend, pilot.u = None, np.zeros((1, BEND_DIM))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Keep flying; the environment ends the race."""
        return False

    def _replan_if_revealed(self, obs: dict[str, NDArray], pos: NDArray, passed: int):
        """Plan the guide again when the sensor has reached something new.

        A plan that fails its check is not flown: the previous guide stays.
        """
        known = _revealed(obs)
        # A plan through the next gates only must be renewed once they are passed.
        passed_planned = (
            self._from_state and self._replan_gates > 0 and passed > self._planned_at_pass
        )
        if known == self._known and not passed_planned:
            return
        if passed >= len(self._library.sequence):
            return
        self._known = known
        layout = obs
        if self._in_order:
            stage = int(np.cumprod(np.asarray(known)[self._order]).sum())
            if stage <= self._stage:
                return
            self._stage = stage
            rpy = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_euler("xyz")
            true = (
                np.asarray(obs["gates_pos"], float),
                rpy,
                np.asarray(obs["obstacles_pos"], float),
            )
            gates_pos, gates_rpy, poles = staged_layout(self._nominal, true, self._order, stage)
            layout = {
                "gates_pos": gates_pos,
                "gates_quat": Rotation.from_euler("xyz", gates_rpy).as_quat(),
                "obstacles_pos": poles,
            }
        t0 = time.perf_counter()
        if self._from_state:
            vel = np.asarray(obs["vel"], dtype=float)
            library, clear = self._state_library(layout, pos[0], vel, passed)
            # A plan that fails its check is tried again at the next step.
            self._planned_at_pass = passed if clear else self._planned_at_pass
        else:
            library, clear = self._runtime_library(layout)
        self.replans.append(
            {"ms": (time.perf_counter() - t0) * 1e3, "clear": clear, "revealed": sum(known)}
        )
        if clear:
            # A guide planned from the drone starts at it.
            self._index = (
                np.zeros(1, dtype=int)
                if self._from_state
                else _index_between_gates(library, pos[0], passed)
            )
            self._library = library

    def _select(self, x: NDArray, proposals: NDArray) -> NDArray:
        """Return the action of the style flying now, deciding again every ``hold`` steps."""
        if self._tick % self._hold == 0:
            inputs = selector_features(x, np.array([self._style]), proposals)
            self._style = int(self._selector(inputs).argmax(axis=1)[0])
        return proposals[:, self._style]

    def _propose(
        self, x: NDArray, features: Callable, make: Callable, bender_input: Callable
    ) -> NDArray:
        """Return what every pilot would command now, ``(1, pilots, ACT_DIM)``.

        A pilot with a bender bends its guide anew every ``hold`` steps, from the drone's state.

        Args:
            x: The features on the guide as planned.
            features: Returns the features on a bent guide, given its bend.
            make: Returns the bend that a bender's output asks for, given the output, its limit and
                whether it starts at the drone.
            bender_input: Returns a bender's input, given its previous output.
        """
        out = []
        for pilot in self._pilots:
            if pilot.bender is not None and self._tick % pilot.hold == 0:
                u = np.clip(pilot.bender(bender_input(pilot.u)), -1.0, 1.0)
                pilot.u, pilot.bend = u, make(u, pilot.limit, pilot.from_drone)
            features_now = x if pilot.bend is None else features(pilot.bend)
            out.append(np.clip(pilot.policy(features_now), -1.0, 1.0))
        return np.stack(out, axis=1)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return ``[roll, pitch, yaw, thrust]`` for the current observation."""
        keys = ("pos", "vel", "quat", "ang_vel")
        pos, vel, quat, ang_vel = (np.asarray(obs[k], dtype=float)[None] for k in keys)
        passed = np.array([max(int(obs["n_gates_passed"]), 0)])
        passed = np.minimum(passed, len(self._library.sequence) - 1)
        if self._replan:
            library = self._library
            self._replan_if_revealed(obs, pos, int(passed[0]))
            if self._library is not library:
                # A bend belongs to the guide it was made on.
                for pilot in self._pilots:
                    pilot.bend = None
        lib = self._library
        self._index, _ = advance_scene_guide(pos, self._index, passed, self._rows, lib)
        gates_rot = Rotation.from_quat(np.asarray(obs["gates_quat"], dtype=float)).as_matrix()
        poles = np.asarray(obs["obstacles_pos"], dtype=float)[None]

        def features(bend: Bend | None = None) -> NDArray:
            return scene_features(
                pos,
                vel,
                quat,
                ang_vel,
                self._previous,
                self._index,
                passed,
                self._rows,
                np.asarray(obs["gates_pos"], dtype=float)[None],
                gates_rot[None],
                poles,
                lib,
                bend,
            )

        x = features()

        def make(u: NDArray, limit: float, from_drone: bool) -> Bend:
            return make_bend(u, limit, pos, vel, self._index, self._rows, lib, from_drone)

        def bender_input(previous: NDArray) -> NDArray:
            return bender_features(
                x,
                pos,
                self._index,
                passed,
                self._rows,
                lib,
                previous,
                np.asarray(obs["gates_visited"], dtype=bool)[None],
                np.asarray(obs["obstacles_visited"], dtype=bool)[None],
                poles,
            )

        proposals = self._propose(x, features, make, bender_input)
        u = proposals[:, 0] if self._selector is None else self._select(x, proposals)
        self._tick += 1
        self.style_ticks[self._style] += 1
        # The command is in the frame of the guide its pilot flies, bent or not.
        pilot = self._pilots[0 if self._selector is None else self._style]
        c, s = scene_heading(self._index, self._rows, lib, pilot.bend)
        self._previous = u.astype(np.float32)
        return scene_action(u, c, s)[0].astype(np.float32)
