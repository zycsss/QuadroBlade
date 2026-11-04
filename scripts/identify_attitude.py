"""Identify the attitude-mode response of the simulated drone.

The race environment's ``attitude`` interface hands ``[roll, pitch, yaw, thrust]`` to the onboard
Mellinger attitude controller with a zero rate setpoint, so per tilt axis the torque it asks for is

    T = K1 * (cmd - tilt) - K2 * rate - KD * ang_acc,      |T| <= AL_MAX

and the rotors deliver it with a first-order lag ``TAU_AL``. The collective thrust follows its
command affinely, ``F_OFFSET + KF * cmd``, with a first-order lag ``TAU_F``. This script excites
both in the raw Crazyflow simulator (high above the floor, so nothing is hit), records the response
at the simulator rate, and fits the constants by simulating the model against the recording.

Run as:

    $ pixi run python scripts/identify_attitude.py

The fitted constants are printed in the form `lsy_drone_racing.control` uses them.
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import fire
import numpy as np
from crazyflow.sim import Sim
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

SIM_FREQ = 500
CTRL_FREQ = 50
GRAVITY = 9.81


def record(
    seconds: float = 12.0, seed: int = 0, mode: str = "mix", drone: str = "cf21B_500"
) -> dict[str, np.ndarray]:
    """Fly random tilt and thrust steps and record the response at the simulator rate.

    Args:
        seconds: Length of the experiment.
        seed: Seed of the random command sequence.
        mode: ``"roll"`` or ``"pitch"`` to excite one tilt axis at hover thrust, ``"thrust"`` to
            step the thrust level, ``"mix"`` for everything at once.
        drone: Drone model.

    Returns:
        Commands and states, one row per simulator step.
    """
    rng = np.random.default_rng(seed)
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=drone,
        control="attitude",
        dynamics="first_principles",
        freq=SIM_FREQ,
        attitude_freq=SIM_FREQ,
    )
    sim.reset()
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(200.0))
    )
    mass = float(np.asarray(sim.data.params.mass).ravel()[0])
    hover = mass * GRAVITY
    sub = SIM_FREQ // CTRL_FREQ
    n_ticks = int(seconds * CTRL_FREQ)
    cmd = np.zeros(4)
    hold = 0
    rows = {"cmd": [], "quat": [], "ang_vel": [], "vel": []}
    for _ in range(n_ticks):
        if hold <= 0:
            tilt = rng.uniform(-0.9, 0.9, size=2) * (rng.random(2) < 0.8)
            thrust_scale = rng.uniform(0.6, 1.5)
            if mode in ("roll", "pitch"):
                tilt[1 if mode == "roll" else 0] = 0.0
                thrust_scale = 1.0 / np.cos(tilt).prod()
            elif mode == "thrust":
                tilt[:] = 0.0
            cmd = np.array([tilt[0], tilt[1], 0.0, hover * thrust_scale])
            hold = int(rng.integers(5, 20))
        hold -= 1
        sim.attitude_control(cmd.reshape(1, 1, 4).astype(np.float32))
        for _ in range(sub):
            sim.step(1)
            rows["cmd"].append(cmd.copy())
            rows["quat"].append(np.asarray(sim.data.states.quat).reshape(4))
            rows["ang_vel"].append(np.asarray(sim.data.states.ang_vel).reshape(3))
            rows["vel"].append(np.asarray(sim.data.states.vel).reshape(3))
    out = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    out["mass"] = np.array(mass)
    return out


def simulate_tilt(params: np.ndarray, cmd: np.ndarray, x0: np.ndarray, dt: float) -> np.ndarray:
    """Simulate the per-axis tilt model and return the tilt trajectory."""
    k1, k2, kd, tau, al_max = params
    tilt, rate, acc = x0
    out = np.empty(len(cmd))
    for i, c in enumerate(cmd):
        out[i] = tilt
        torque = np.clip(k1 * (c - tilt) - k2 * rate - kd * acc, -al_max, al_max)
        acc += dt * (torque - acc) / tau
        rate += dt * acc
        tilt += dt * rate
    return out


def simulate_thrust(params: np.ndarray, cmd: np.ndarray, f0: float, dt: float) -> np.ndarray:
    """Simulate the first-order collective thrust model ``f -> offset + kf * cmd``."""
    kf, offset, tau = params
    f = f0
    out = np.empty(len(cmd))
    for i, c in enumerate(cmd):
        out[i] = f
        f += dt * (offset + kf * c - f) / tau
    return out


def fit_tilt(data: dict[str, np.ndarray], axis: int, start: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the tilt model of one axis to a recording; return the constants and the rms error."""
    dt = 1.0 / SIM_FREQ
    euler = R.from_quat(data["quat"]).as_euler("xyz")
    skip = SIM_FREQ // 2
    measured = euler[skip:, axis]
    cmd = data["cmd"][skip:, axis]
    rate0 = (euler[skip + 1, axis] - euler[skip - 1, axis]) / (2 * dt)
    x0 = np.array([measured[0], rate0, 0.0])
    res = least_squares(
        lambda p: simulate_tilt(p, cmd, x0, dt) - measured,
        x0=start,
        bounds=([10.0, 1.0, 0.0, 0.005, 10.0], [5000.0, 1000.0, 20.0, 0.5, 2000.0]),
        x_scale="jac",
    )
    return res.x, float(np.sqrt(np.mean(res.fun**2)))


def main(seconds: float = 12.0, seed: int = 0):
    """Record one excitation run per axis, fit the constants, and validate them on a mixed run."""
    dt = 1.0 / SIM_FREQ
    fits = []
    for axis, name in ((0, "roll"), (1, "pitch")):
        data = record(seconds, seed, mode=name)
        best = None
        for start in ([800.0, 117.0, 1.3, 0.05, 170.0], [300.0, 50.0, 0.5, 0.03, 120.0]):
            params, rms = fit_tilt(data, axis, np.array(start))
            if best is None or rms < best[1]:
                best = (params, rms)
        fits.append(best[0])
        k1, k2, kd, tau, al_max = best[0]
        print(
            f"{name:<5}  K1 {k1:8.1f}  K2 {k2:7.2f}  KD {kd:6.3f}  TAU_AL {tau:6.4f} s  "
            f"AL_MAX {al_max:7.1f} rad/s^2   rms tilt error {np.degrees(best[1]):.2f} deg"
        )
    mean = np.mean(fits, axis=0)
    mixed = record(seconds, seed + 1, mode="mix")
    euler = R.from_quat(mixed["quat"]).as_euler("xyz")
    skip = SIM_FREQ // 2
    for axis, name in ((0, "roll"), (1, "pitch")):
        rate0 = (euler[skip + 1, axis] - euler[skip - 1, axis]) / (2 * dt)
        x0 = np.array([euler[skip, axis], rate0, 0.0])
        err = simulate_tilt(mean, mixed["cmd"][skip:, axis], x0, dt) - euler[skip:, axis]
        print(f"validate {name} on a mixed run: rms {np.degrees(np.sqrt(np.mean(err**2))):.2f} deg")

    data = record(seconds, seed, mode="thrust")
    # Collective thrust from the vertical momentum balance along the body z axis.
    mass = float(data["mass"])
    # The airframe has linear drag, ``m * a = f * z_body - m * g - m * DRAG * v``, and a thrust step
    # builds vertical speed, so the drag coefficient is fitted alongside the thrust map; leaving it
    # out folds the drag into a too-small thrust gain.
    acc = np.gradient(data["vel"], dt, axis=0)[skip:] + np.array([0.0, 0.0, GRAVITY])
    vel = data["vel"][skip:]
    z_body = R.from_quat(data["quat"][skip:]).apply([0.0, 0.0, 1.0])
    cmd_f = data["cmd"][skip:, 3]

    def residual_f(p: np.ndarray) -> np.ndarray:
        meas = mass * np.sum((acc + p[3] * vel) * z_body, axis=1)
        return simulate_thrust(p[:3], cmd_f, meas[0], dt) - meas

    res_f = least_squares(
        residual_f,
        x0=np.array([1.0, 0.0, 0.05, 0.3]),
        bounds=([0.5, -0.2, 0.002, 0.0], [1.5, 0.2, 0.5, 2.0]),
    )
    kf, f_offset, tau_f, drag = res_f.x
    rms_f = float(np.sqrt(np.mean(res_f.fun**2)))
    print(
        f"thrust KF {kf:.4f}  F_OFFSET {f_offset:.4f} N  TAU_F {tau_f:.4f} s  DRAG {drag:.3f} 1/s"
        f"   rms {rms_f:.4f} N   (mass {mass:.4f} kg)"
    )
    print(
        f"\nMODEL  K1={mean[0]:.1f}  K2={mean[1]:.2f}  KD={mean[2]:.3f}  TAU_AL={mean[3]:.4f}  "
        f"AL_MAX={mean[4]:.1f}  KF={kf:.4f}  F_OFFSET={f_offset:.4f}  TAU_F={tau_f:.4f}  "
        f"DRAG={drag:.3f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
