"""The empirical half of B3: does the closed form hold on average? [B3]

`compare_r` checks that the filter and the closed form agree on ONE event sequence. That
catches an implementation bug but not a modelling one: if both the filter and the derivation
assume the wrong p_D, they will agree with each other perfectly and both be wrong.

Averaging r over many seeds answers the other question. The event sequence differs each run
- different misdetections, different clutter - so the mean r trajectory reflects the
detection process itself rather than one realisation of it.

Kept deliberately small in phase 1: tens of runs over tens of scans. Long extended trials
belong to phase 2 on ROS 2.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from crop_mot.analysis.estimates_log import read_estimates, r_trajectory
from crop_mot.config import RunConfig
from crop_mot.filters import build_filter
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregate of N repeated runs of one scenario under different seeds. [B3]

    Attributes:
        r_mean: shape (K,), mean existence probability per scan across runs.
        r_std: shape (K,), standard deviation per scan. The standard ERROR is
            r_std / sqrt(n_runs); the test compares against that, not against r_std.
        n_runs: how many runs were aggregated.
        seeds: the seeds used, recorded so the aggregate is itself reproducible.
    """

    r_mean: np.ndarray
    r_std: np.ndarray
    n_runs: int
    seeds: tuple[int, ...]


def run_monte_carlo(
    cfg: RunConfig, n_runs: int, base_seed: int, runs_base: Path
) -> MonteCarloResult:
    """Re-simulate and re-filter the same scenario over n_runs seeds, aggregating r. [B3]

    Each run derives its seed deterministically from base_seed, so the whole aggregate is
    reproducible from two integers. Each run simulates fresh detections - the point is to
    vary the realisation, so reusing one detection set would defeat the exercise.

    One thing to be careful about when interpreting the result: for the phantom scenario the
    runs are NOT identically distributed unless the birth is seeded consistently. If the
    birth model picks "detection index 3 at scan 0" and that happens to be a real plant in
    some seeds and clutter in others, the mean r mixes two different experiments. The
    scenario should pin down which it is.

    Serves: [B3], and the r_montecarlo plot.

    Args:
        cfg: the run configuration to repeat.
        n_runs: how many seeds to average over.
        base_seed: the seed the per-run seeds are derived from.
        runs_base: the runs/ directory; individual runs may be written to a temporary
            subdirectory rather than cluttering runs/ with N folders.

    Returns:
        The aggregated result.

    Run i uses seed base_seed + i for the whole scenario (all four substreams), and the r
    trajectory of the first track the filter reports - all zeros if nothing was born.
    r_std is the sample standard deviation (ddof = 1), so r_std / sqrt(n_runs) is the usual
    estimate of the standard error of the mean. The per-run folders live in a temporary
    directory under runs_base and are deleted afterwards.
    """
    seeds = tuple(base_seed + i for i in range(n_runs))
    trajectories = []
    runs_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="montecarlo_", dir=runs_base) as tmp:
        for seed in seeds:
            run = RunDir(Path(tmp) / f"seed{seed}")
            run.plots.mkdir(parents=True)
            simulate(replace(cfg.scenario, seed=seed), run)
            flt = build_filter(cfg.filter_cfg)
            run_filter(flt, run)

            records = read_estimates(run.estimates(flt.name))
            track_ids = [est.track_id for record in records for est in record.estimates]
            if track_ids:
                trajectories.append(r_trajectory(records, track_ids[0]))
            else:
                trajectories.append(np.zeros(len(records)))

    r = np.array(trajectories)
    return MonteCarloResult(
        r_mean=r.mean(axis=0),
        r_std=r.std(axis=0, ddof=1) if n_runs > 1 else np.zeros(r.shape[1]),
        n_runs=n_runs,
        seeds=seeds,
    )


def standard_error(result: MonteCarloResult) -> np.ndarray:
    """Standard error of the mean r per scan. [B3]

    Serves: the confidence band in the r_montecarlo plot, and the tolerance in
    `test_monte_carlo_mean_r_brackets_analytic`.

    Args:
        result: an aggregate from `run_monte_carlo`.

    Returns:
        Shape (K,) array of r_std / sqrt(n_runs).
    """
    return result.r_std / np.sqrt(result.n_runs)
