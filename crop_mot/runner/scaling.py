"""The scaling study entry point: time per scan against problem size. [B4, roadmap step 4c]

The `python -m crop_mot scaling` subcommand. For each value of one swept scenario
parameter, several seeds are simulated and filtered with the run config's filter (the
Monte-Carlo trial loop, D17), and the update time per scan is summarised against the
problem size measured in each run (`analysis.timing`). The result goes into a folder of
its own under runs/: the config copy, run_meta.json (with the thread settings and CPU),
metrics.json and plots/scaling.png.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from crop_mot.analysis.montecarlo import run_trials
from crop_mot.analysis.plots import plot_scaling
from crop_mot.analysis.timing import (
    SWEEPS,
    ScalingResult,
    loglog_slope,
    scaling_point,
    timing_measure,
)
from crop_mot.config import RunConfig, load_run_config
from crop_mot.io import to_jsonable
from crop_mot.runner.run_dir import RunDir, create_run_dir, thread_settings, write_run_meta


def run_scaling(cfg: RunConfig, sweep: str, values: Sequence[float], n_seeds: int,
                out: RunDir) -> ScalingResult:
    """Time the configured filter over a sweep, into out's metrics.json and plots. [B4]

    Seeds cfg.seed, cfg.seed + 1, ... for every value, so the values differ only in the
    swept parameter. The per-run folders are temporary, inside `out`.

    Args:
        cfg: the run configuration to sweep.
        sweep: a key into `analysis.timing.SWEEPS`.
        values: the swept parameter's values.
        n_seeds: seeds per value.
        out: the folder for the result.

    Returns:
        The result, also written to out.metrics under "scaling" and drawn to
        out.plots / "scaling.png".

    Raises:
        KeyError: if the sweep is unknown.
    """
    if sweep not in SWEEPS:
        raise KeyError(f"unknown sweep {sweep!r}; available: {sorted(SWEEPS)}")
    spec = SWEEPS[sweep]
    points = []
    for value in values:
        trials = run_trials(spec.apply(cfg, value), n_seeds, cfg.seed, out.root,
                            {"timing": timing_measure})
        points.append(scaling_point(value, [trial.values["timing"] for trial in trials],
                                    spec.size))
    result = ScalingResult(sweep=sweep, filter_name=cfg.filter_cfg.kind, points=tuple(points),
                           slope=loglog_slope(points),
                           budget_s=cfg.scenario.path.scan_period, n_seeds=n_seeds,
                           threads=thread_settings())
    record = asdict(result)
    if math.isnan(result.slope):
        record["slope"] = None  # fewer than two distinct sizes; keeps the file strict JSON
    out.metrics.write_text(json.dumps(to_jsonable({"scaling": record}), indent=2) + "\n",
                           encoding="utf-8")
    plot_scaling(result, out.plots / "scaling.png")
    return result


def scaling_from_config(config_path: Path, sweep: str, values: Sequence[float],
                        n_seeds: int, runs_base: Path) -> RunDir:
    """The CLI entry point: a fresh folder named after the config and the sweep. [B4]

    Warns on stderr when the BLAS thread variables are not 1, since multi-threaded times
    are not comparable across problem sizes.
    """
    cfg = load_run_config(config_path)
    threads = thread_settings()
    if threads["OMP_NUM_THREADS"] != "1" or threads["OPENBLAS_NUM_THREADS"] != "1":
        print("warning: set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 before timing; "
              f"got {threads}", file=sys.stderr)
    out = create_run_dir(runs_base, f"scaling_{cfg.name}_{sweep}", cfg.seed, config_path)
    write_run_meta(out, cfg.seed, sys.argv)
    run_scaling(cfg, sweep, values, n_seeds, out)
    return out
