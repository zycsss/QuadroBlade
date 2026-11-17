"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

The final controller is [RLScene][lsy_drone_racing.control.rl_scene.RLScene], which every level
configuration selects. The other modules are:

* [Controller][lsy_drone_racing.control.controller.Controller]: The abstract base class defining the
  interface for all controllers.
* [RLRace][lsy_drone_racing.control.rl_race.RLRace]: An RL policy trained on the track it flies.
* [ToptMPCC][lsy_drone_racing.control.topt_mpcc.ToptMPCC]: A contouring MPC that flies the
  time-optimal plan of [topt_planner][lsy_drone_racing.control.topt_planner].
* [StateController][lsy_drone_racing.control.state_controller.StateController] and
  [AttitudeController][lsy_drone_racing.control.attitude_controller.AttitudeController]: The
  course's example controllers, which the environment tests fly.
"""

from lsy_drone_racing.control.controller import Controller

__all__ = ["Controller"]
