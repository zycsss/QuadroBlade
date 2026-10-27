"""Script for measuring the drone's capability envelope on analytic scenarios.

Runs the manoeuvres defined in `lsy_drone_racing.control.scenario_path` through the racing
controller and reports what the airframe can actually do: tracking error, speed, sustainable
lateral acceleration, attitude, lateral step response and solver cost. Each stage is documented in
`docs/control/characterisation.md`.

Run it from the repository root:

    pixi run python scripts/characterize.py
    pixi run python scripts/characterize.py --only circle --sweep radius 0.5 0.8 1.1
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.characterization import (
    ScenarioResult,
    plot_scenario,
    run_scenario,
    scenario_table,
    write_result,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Lateral acceleration budget the curvature-limited scenarios are built around [m/s^2].
A_LAT = 8.0

#: Length of the straight-line scenarios [m], long enough to reach the requested speed.
LINE_LENGTH = 4.0

#: x coordinate the straight-line scenarios start from, so the run ends inside the arena.
LINE_START_X = -2.2


def default_matrix() -> list[str]:
    """Return the full set of scenario specifications.

    Returns:
        One specification per measurement, covering the speed sweep, the curvature sweep, the
        overload cases, two corners and the two reveal-like lateral steps.
    """
    from math import sqrt

    matrix = [
        f"line length={LINE_LENGTH:g} x={LINE_START_X:g} speed={speed:g}"
        for speed in (1.0, 1.5, 2.0, 2.5, 3.0)
    ]
    matrix += [
        f"circle radius={radius:g} speed={sqrt(A_LAT * radius):.2f}" for radius in (0.5, 0.8, 1.1)
    ]
    # Overload: ask for more lateral acceleration than the controller's own budget, which is how
    # the real limit is found rather than assumed.
    matrix += [
        f"circle radius=0.8 overload={sqrt(A_LAT * 0.8) * factor:.2f}" for factor in (1.2, 1.5)
    ]
    matrix += [
        # Corners that fit inside the arena: a 90 degree turn of radius 1 m from y = -0.6 keeps the
        # exit inside +-1.45 m, while the same turn from y = 0 does not.
        "corner radius=0.5 angle=90 speed=2.00",
        "corner radius=1 angle=90 y=-0.6 straight_in=0.5 straight_out=0.3 speed=2.83",
    ]
    matrix += [
        # Two ways to shift sideways by the reveal magnitude: the reference steps ahead of the
        # drone (preview available), and the drone starts beside its path (no preview at all).
        f"shift length={LINE_LENGTH:g} x={LINE_START_X:g} offset=0.21 at=2.0 speed={speed:g}"
        for speed in (1.0, 1.7)
    ]
    matrix += [
        f"line length={LINE_LENGTH:g} x={LINE_START_X:g} start_offset=0.21 speed={speed:g}"
        for speed in (1.0, 1.7)
    ]
    return matrix


def main(
    only: str | None = None,
    speed: str | None = None,
    out: str = "plots/scenarios",
    logs: str = "logs/scenarios",
    dpi: int = 140,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """Run the characterisation matrix and report the measured envelope.

    Args:
        only: Run only the scenarios mentioning this substring, e.g. ``circle`` or ``shift``.
        speed: Restrict the speed sweep to these values, comma separated, e.g. ``1.0,2.0``.
        out: Directory for the figures.
        logs: Directory for the logged arrays and the per-run summaries.
        dpi: Resolution of the saved figures.
        device: ``"cpu"`` or ``"gpu"``.
        verbose: Print each scenario as it finishes.
    """
    config = load_config(REPO_ROOT / "scenarios" / "characterisation.toml")
    matrix = default_matrix()
    if only:
        matrix = [spec for spec in matrix if only in spec]
    if speed:
        # fire parses `--speed 1.0,2.0` into a tuple already, and a single value into a float.
        values = speed if isinstance(speed, (tuple, list)) else str(speed).split(",")
        keep = {float(value) for value in values}
        matrix = [
            spec
            for spec in matrix
            if not (spec.startswith("line") or spec.startswith("shift"))
            or float(spec.split("speed=")[1]) in keep
        ]
    if not matrix:
        raise SystemExit("no scenario matched the filters")

    out_dir, log_dir = REPO_ROOT / out, REPO_ROOT / logs
    results: list[ScenarioResult] = []
    for spec in matrix:
        result = run_scenario(spec, config, device=device, verbose=verbose)
        results.append(result)
        npz_path, _ = write_result(result, log_dir)
        figure = plot_scenario(result, out_dir / f"{npz_path.stem}.png", dpi=dpi)
        if verbose:
            metrics = result.metrics
            print(
                f"{spec:44s} {result.verdict:18s} "
                f"v={metrics['speed_mean']:4.2f} "
                f"err={metrics['tracking_rms'] * 1000:5.1f}mm "
                f"a={metrics['a_lat_mean']:4.1f} "
                f"tilt={metrics['tilt_cruise']:4.1f}deg "
                f"-> {figure.name}"
            )

    table = scenario_table(results)
    (out_dir / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(table + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(
            [{"spec": r.spec, "verdict": r.verdict, **r.metrics} for r in results],
            indent=2,
            default=float,
        )
    )
    print()
    print(table)
    print(f"\ntable -> {out_dir / 'summary.md'}")


if __name__ == "__main__":
    fire.Fire(main)
