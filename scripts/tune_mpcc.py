"""Joint tuning of the MPCC weights with a trust-region Bayesian optimiser.

The controller has eight weights that interact: the contour and lag weights, the progress reward and
its regularisation, the gate proximity weight, the lateral acceleration cap, the top speed and the
tunnel weight. Tuning them one at a time - which is what the planning page records - needs a fresh
40-seed screen per value and cannot see the interactions.

This implements TuRBO-1 (Eriksson et al., NeurIPS 2019) over eight parallel trust regions, the setup
MPCC++ (Krinner, Romero et al., RSS 2024) uses for the same controller, with two changes that suit a
simulator rather than hardware:

* the reward is ``-mean lap - 100 * solver failure rate - 20 * (1 - success rate)``, so a crash is
  worse than a slow lap - the same shape as the paper's lap-time reward with a failure penalty;
* the seeds are drawn from a rotating pool, so the tuner cannot specialise to ten fixed tracks.

Every evaluation runs whole episodes through ``scripts/debug_run.run_episode``, so an episode is
~10 s on CPU and a 600-episode budget is about 100 minutes - worth running in the background:

    pixi run python scripts/tune_mpcc.py --budget 600
    pixi run python scripts/tune_mpcc.py --budget 40 --apply   # a quick check

The winner is printed as a ready-to-paste ``[controller]`` block, and only written into the code
defaults by hand: the tuning result has to be validated on unseen seeds first.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402
from scripts.debug_run import run_episode  # noqa: E402

#: The parameters to tune: (name, lower bound, upper bound, log scale). Bounds bracket the values
#: on the planning page rather than physical limits, because a trust region that starts far from
#: the working point spends its budget rediscovering it.
SPACE: tuple[tuple[str, float, float, bool], ...] = (
    ("q_contour", 10.0, 1.0e3, True),
    ("q_lag", 10.0, 1.0e3, True),
    ("q_contour_peak", 10.0, 5.0e3, True),
    ("mu_speed", 1.0, 60.0, True),
    ("r_vtheta", 0.01, 10.0, True),
    ("w_speed_gate", 0.0, 40.0, False),
    ("a_lat_max", 1.0, 12.0, False),
    ("w_tunnel", 0.0, 1.0e3, True),
)

CONFIG_FILE = "level2.toml"
SOLVER_PENALTY = 100.0
CRASH_PENALTY = 20.0


@dataclass
class Trial:
    """One evaluated parameter set and its reward."""

    unit: np.ndarray
    reward: float
    lap: float
    success: float
    failures: float


def to_unit(values: np.ndarray) -> np.ndarray:
    """Map parameters into the unit cube the optimiser works in."""
    unit = np.empty_like(values, dtype=float)
    for index, (_, low, high, log) in enumerate(SPACE):
        if log and low > 0.0:
            span = math.log(high) - math.log(low)
            unit[index] = (math.log(values[index]) - math.log(low)) / span
        else:
            unit[index] = (values[index] - low) / (high - low)
    return np.clip(unit, 0.0, 1.0)


def from_unit(unit: np.ndarray) -> dict[str, float]:
    """Map a point of the unit cube back into controller parameters."""
    values = {}
    for index, (name, low, high, log) in enumerate(SPACE):
        fraction = float(np.clip(unit[index], 0.0, 1.0))
        if log and low > 0.0:
            span = math.log(high) - math.log(low)
            values[name] = float(math.exp(math.log(low) + fraction * span))
        else:
            values[name] = float(low + fraction * (high - low))
    return values


class Surrogate:
    """A tiny Gaussian process over the unit cube, with an RBF kernel and fitted noise.

    BoTorch is not a dependency of this repository and the search space is eight dimensional with
    fewer than a few hundred points, so a self-contained GP is cheaper than adding a dependency.
    """

    def __init__(self, length: float = 0.3):
        """Store the fixed kernel length scale."""
        self.length = float(length)
        self.x = np.zeros((0, len(SPACE)))
        self.y = np.zeros(0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Remember the observations."""
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the RBF covariance between two sets of points."""
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * squared / self.length**2)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the posterior mean and standard deviation at ``x``."""
        if len(self.y) < 2:
            return np.zeros(len(x)), np.ones(len(x))
        noise = 1e-4 + self.y.var() * 1e-3
        k = self._kernel(self.x, self.x) + noise * np.eye(len(self.x))
        k_s = self._kernel(self.x, x)
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return np.full(len(x), float(self.y.mean())), np.ones(len(x))
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, self.y - self.y.mean()))
        mean = self.y.mean() + k_s.T @ alpha
        v = np.linalg.solve(chol, k_s)
        variance = np.maximum(1.0 - np.sum(v**2, axis=0), 1e-9)
        return mean, np.sqrt(variance)


class TrustRegion:
    """One local optimiser: a GP, a box that shrinks and grows, and an acquisition function."""

    def __init__(self, centre: np.ndarray, length: float, rng: np.random.Generator):
        """Start a region around ``centre`` with a box half side ``length``."""
        self.centre = np.asarray(centre, dtype=float)
        self.length = float(length)
        self.surrogate = Surrogate()
        self.rng = rng
        self.successes = 0
        self.failures = 0

    def propose(self, observed: int, budget: int) -> np.ndarray:
        """Return the next point: random until there is data, then the UCB maximiser."""
        low = np.clip(self.centre - self.length, 0.0, 1.0)
        high = np.clip(self.centre + self.length, 0.0, 1.0)
        if observed < 2:
            return self.rng.uniform(low, high)
        candidates = self.rng.uniform(low, high, size=(512, len(SPACE)))
        mean, std = self.surrogate.predict(candidates)
        beta = 2.0 * (1.0 - observed / max(budget, 1)) + 0.5
        return candidates[int(np.argmax(mean + beta * std))]

    def update(self, better: bool) -> None:
        """Resize the region: grow after three good points, shrink after ten bad ones."""
        self.successes = self.successes + 1 if better else 0
        self.failures = 0 if better else self.failures + 1
        if self.successes >= 3:
            self.length = min(self.length * 2.0, 1.6)
            self.successes = 0
        elif self.failures >= 10:
            self.length = max(self.length / 2.0, 0.02)
            self.failures = 0


def reward(lap: float, success: float, failure_rate: float) -> float:
    """Return the scalar reward: lap time, solver failures and crashes, in that order."""
    return -lap - SOLVER_PENALTY * failure_rate - CRASH_PENALTY * (1.0 - success)


def evaluate(values: dict[str, float], seeds: list[int], device: str = "cpu") -> tuple[float, dict]:
    """Run whole episodes with the given controller parameters and score them.

    Args:
        values: Parameters to override, by ``MPCCConfig`` field name.
        seeds: Episode seeds to run.
        device: Simulation device.

    Returns:
        The reward and the episode summary it came from.
    """
    config = load_config(REPO_ROOT / "config" / CONFIG_FILE)
    config.sim.render = False
    merged = dict(config.get("controller") or {})
    merged.update(values)
    config["controller"] = merged
    controller = load_controller(REPO_ROOT / "lsy_drone_racing/control/my_controller.py")
    laps, successes, solver_failures = [], [], []
    for seed in seeds:
        episode = run_episode(config, controller, seed, verbose=False, device=device)
        record = episode["record"]
        status = np.asarray(record.get("solver_status", [0]))
        solver_failures.append(float(np.mean(status != 0)))
        successes.append(float(episode["verdict"] == "FINISHED"))
        laps.append(float(episode["sim_time"]))
    lap = float(np.mean(laps))
    success = float(np.mean(successes))
    failure_rate = float(np.mean(solver_failures))
    return reward(lap, success, failure_rate), {
        "lap": lap,
        "success": success,
        "solver_failure": failure_rate,
    }


def main(
    budget: int = 600,
    episodes: int = 10,
    region_count: int = 8,
    start: int = 0,
    device: str = "cpu",
    apply: bool = False,
) -> None:
    """Run the optimiser.

    Args:
        budget: Total episodes to spend, across every evaluation.
        episodes: Episodes per evaluation, i.e. the reward is a mean over this many seeds.
        region_count: Number of parallel trust regions.
        start: First seed of the rotating pool, so a second run can use different tracks.
        device: Simulation device.
        apply: Print the best parameters as a ``[controller]`` block when done.
    """
    rng = np.random.default_rng(0)
    # Start every region at the current defaults, in unit coordinates.
    current = load_config(REPO_ROOT / "config" / CONFIG_FILE)
    defaults = dict(current.get("controller") or {})
    values = np.array(
        [
            np.clip(defaults.get(name, 0.5 * (low + high)), low, high)
            for name, low, high, _ in SPACE
        ],
        dtype=float,
    )
    centre = to_unit(values)
    regions = [
        TrustRegion(centre + rng.normal(0.0, 0.05, len(SPACE)), 0.4, rng)
        for _ in range(region_count)
    ]
    trials: list[Trial] = []
    pool = max(episodes, 20)
    start_time = time.perf_counter()
    evaluations = max(1, budget // max(episodes, 1))
    print(f"{evaluations} evaluations of {episodes} episodes, {region_count} regions")
    for evaluation in range(evaluations):
        region = regions[evaluation % region_count]
        point = region.propose(len(trials), evaluations)
        values = from_unit(point)
        first = start + (evaluation * episodes) % pool
        seeds = [first + k for k in range(episodes)]
        score, summary = evaluate(values, seeds, device)
        trials.append(
            Trial(point, score, summary["lap"], summary["success"], summary["solver_failure"])
        )
        x = np.array([trial.unit for trial in trials])
        y = np.array([trial.reward for trial in trials])
        region.surrogate.fit(x, y)
        best = max(trials, key=lambda trial: trial.reward)
        region.update(score > best.reward or score > np.mean(y[-region_count:]))
        print(
            f"[{evaluation + 1:3d}/{evaluations}] reward {score:8.2f}  lap {summary['lap']:5.2f}  "
            f"success {summary['success']:3.0%}  solver {summary['solver_failure']:4.1%}  "
            f"best {best.reward:8.2f}  ({time.perf_counter() - start_time:.0f}s)"
        )
        if apply:
            print("  " + json.dumps(values, sort_keys=True))
    best = max(trials, key=lambda trial: trial.reward)
    chosen = from_unit(best.unit)
    print("\nbest parameters (validate on unseen seeds before making them defaults):")
    print("[controller]")
    for name, value in chosen.items():
        print(f"{name} = {value:.4g}")
    print(f"\nreward {best.reward:.2f}, lap {best.lap:.2f} s, success {best.success:.0%}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
