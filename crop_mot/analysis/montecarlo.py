"""Monte Carlo over seeds: the per-seed B3 cross-check, and descriptive mean r. [B3]

`compare_r` checks that the filter and the closed form agree on ONE event sequence. This
module repeats a run over many seeds, so the check covers many realisations - different
misdetections, clutter and births - and with them every branch of the recursion.

Decision D17 sets what the repetition is for. The DELIVERABLE is the per-seed
cross-check (`cross_check_summary`): on every seed, every track's r matches the A2
recursion evaluated on that seed's own events. Comparing the mean r over seeds against
"the closed form" does not hold in general - r depends nonlinearly on the events, so the
mean of r is not the closed form at the average events, and each seed has its own event
sequence - so `MonteCarloResult` (mean r and its standard error) is kept as a descriptive
figure, not a test. Whether the filter's assumed model matches the simulator (the
event-rate model check) is future work.

This is plain Monte Carlo, not MCMC: trial i uses seed base_seed + i and is drawn directly
from the simulator, so the trials are independent - no burn-in, no autocorrelation.

Every seed contributes ONE outcome per quantity: tracks in the same seed share clutter and
neighbours, so they are not independent trials, and pooling them would make intervals too
narrow.

Kept deliberately small in phase 1: tens of runs over tens of scans. Long extended trials
belong to phase 2 on ROS 2.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from crop_mot.analysis.crosscheck import TrackCheck, cross_check_run
from crop_mot.analysis.estimates_log import r_trajectory, read_estimates, track_lifetimes
from crop_mot.analysis.metrics import R_TOLERANCE
from crop_mot.config import RunConfig
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_configured_filter


@dataclass(frozen=True)
class Trial:
    """One seed's run, measured while its run folder still existed. [B3]

    Attributes:
        seed: the seed of the whole run (all substreams).
        values: one entry per measurement function, under the name it was given.
    """

    seed: int
    values: dict[str, object]


def run_trials(
    cfg: RunConfig,
    n_runs: int,
    base_seed: int,
    runs_base: Path,
    measures: Mapping[str, Callable[[RunDir, RunConfig], object]],
) -> list[Trial]:
    """Simulate, filter and measure one run per seed: the general trial loop. [B3]

    For seed i = base_seed + i: simulate the scenario with that seed, run the configured
    filter (and its unpruned companion when it prunes, D14), then apply every measurement
    function to the run folder while it still exists. The folders live in a temporary
    directory under runs_base and are deleted afterwards, so a measurement must return
    everything it needs.

    A measurement is any function (run, cfg) -> value, where cfg is that seed's config.
    The B3 cross-check is one (`cross_check_run`); GOSPA, cardinality, NEES, phantom
    lifetime and timing plug in the same way, which is why this loop is not B3-specific.

    Serves: [B3] the per-seed cross-check and the r_montecarlo figure; later steps' metrics.

    Args:
        cfg: the run configuration to repeat.
        n_runs: how many seeds.
        base_seed: the seed the per-run seeds are derived from.
        runs_base: the directory the temporary run folders are created under.
        measures: the measurement functions, by name.

    Returns:
        One Trial per seed, in seed order.
    """
    trials = []
    runs_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="montecarlo_", dir=runs_base) as tmp:
        for i in range(n_runs):
            seed = base_seed + i
            seed_cfg = replace(cfg, seed=seed, scenario=replace(cfg.scenario, seed=seed))
            run = RunDir(Path(tmp) / f"seed{seed}")
            run.plots.mkdir(parents=True)
            simulate(seed_cfg.scenario, run)
            run_configured_filter(seed_cfg.filter_cfg, run)
            trials.append(Trial(seed=seed, values={name: measure(run, seed_cfg)
                                                   for name, measure in measures.items()}))
    return trials


def first_track_r(run: RunDir, cfg: RunConfig) -> np.ndarray:
    """The r trajectory of the first track the filter reports; zeros if none is born. [B3]

    The measurement behind `MonteCarloResult`. One track per seed, so one outcome per seed.
    """
    records = read_estimates(run.estimates(cfg.filter_cfg.kind))
    track_ids = list(track_lifetimes(records))
    if not track_ids:
        return np.zeros(len(records))
    return r_trajectory(records, track_ids[0])


@dataclass(frozen=True)
class MonteCarloResult:
    """Mean r over seeds, as a descriptive figure. [B3]

    Not a pass/fail test (decision D17): the mean of r is not the closed form evaluated at
    any one event sequence. The per-seed check is `CrossCheckSummary`.

    Attributes:
        r_mean: shape (K,), mean existence probability per scan across runs.
        r_std: shape (K,), standard deviation per scan. The standard ERROR is
            r_std / sqrt(n_runs).
        n_runs: how many runs were aggregated.
        seeds: the seeds used, recorded so the aggregate is itself reproducible.
    """

    r_mean: np.ndarray
    r_std: np.ndarray
    n_runs: int
    seeds: tuple[int, ...]


def mean_r(trials: list[Trial], key: str = "r") -> MonteCarloResult:
    """Aggregate per-seed r trajectories into their mean and standard deviation. [B3]

    Args:
        trials: from `run_trials`, each holding an r trajectory under `key`.
        key: the measurement name, default "r" (`first_track_r`).

    Returns:
        The aggregate. r_std is the sample standard deviation (ddof = 1), so
        r_std / sqrt(n_runs) is the usual estimate of the standard error of the mean.
    """
    r = np.array([trial.values[key] for trial in trials])
    return MonteCarloResult(
        r_mean=r.mean(axis=0),
        r_std=r.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros(r.shape[1]),
        n_runs=len(trials),
        seeds=tuple(trial.seed for trial in trials),
    )


def run_monte_carlo(
    cfg: RunConfig, n_runs: int, base_seed: int, runs_base: Path
) -> MonteCarloResult:
    """Mean r of the first track over n_runs seeds (descriptive, D17). [B3]

    Run i uses seed base_seed + i for the whole scenario (all four substreams), so the
    aggregate is reproducible from two integers.

    One thing to be careful about when interpreting the result: for the phantom scenario the
    runs are NOT identically distributed. If the birth model picks "detection index 3 at
    scan 0" and that happens to be a real plant in some seeds and clutter in others, the
    mean r mixes two different experiments.

    Args:
        cfg: the run configuration to repeat.
        n_runs: how many seeds to average over.
        base_seed: the seed the per-run seeds are derived from.
        runs_base: the directory the temporary run folders are created under.

    Returns:
        The aggregated result.
    """
    return mean_r(run_trials(cfg, n_runs, base_seed, runs_base, {"r": first_track_r}))


@dataclass(frozen=True)
class CrossCheckSummary:
    """The per-seed B3 cross-check over many seeds: the B3 deliverable (D17). [B3]

    A seed passes when every one of its tracks matches the reference within the tolerance.
    Seeds where nothing was born have nothing to check and are counted apart.

    Attributes:
        n_runs: how many seeds were run.
        seeds: the seeds, in order.
        tolerance: the absolute tolerance on r (D18).
        n_without_track: seeds in which no track was born.
        n_passed: seeds with at least one track whose every track matched.
        max_abs_error: the largest |r_sim - r_ref| over every scan, track and seed.
        divergences: (seed, track_id, first_divergence_k) for every track that exceeded
            the tolerance; empty when all passed.
        n_scans_checked: scans from a birth on, summed over tracks and seeds.
        branches: scans per branch of the recursion, summed over tracks and seeds, so a
            pass says which branches it covered.
    """

    n_runs: int
    seeds: tuple[int, ...]
    tolerance: float
    n_without_track: int
    n_passed: int
    max_abs_error: float
    divergences: tuple[tuple[int, int, int], ...]
    n_scans_checked: int
    branches: dict[str, int]


def cross_check_summary(trials: list[Trial], key: str = "cross_check") -> CrossCheckSummary:
    """Aggregate per-seed cross-checks, one outcome per seed. [B3]

    Args:
        trials: from `run_trials`, each holding a list[TrackCheck] under `key`
            (`cross_check_run`).
        key: the measurement name, default "cross_check".

    Returns:
        The summary.
    """
    n_without_track = 0
    n_passed = 0
    max_abs_error = 0.0
    divergences = []
    branches: dict[str, int] = {}
    for trial in trials:
        checks: list[TrackCheck] = trial.values[key]
        if not checks:
            n_without_track += 1
            continue
        if all(check.comparison.first_divergence_k is None for check in checks):
            n_passed += 1
        for check in checks:
            max_abs_error = max(max_abs_error, check.comparison.max_abs_error)
            if check.comparison.first_divergence_k is not None:
                divergences.append((trial.seed, check.track_id,
                                    check.comparison.first_divergence_k))
            for branch, count in check.branches.items():
                branches[branch] = branches.get(branch, 0) + count
    return CrossCheckSummary(
        n_runs=len(trials),
        seeds=tuple(trial.seed for trial in trials),
        tolerance=R_TOLERANCE,
        n_without_track=n_without_track,
        n_passed=n_passed,
        max_abs_error=max_abs_error,
        divergences=tuple(divergences),
        n_scans_checked=sum(count for branch, count in branches.items()
                            if branch != "before_birth"),
        branches=branches,
    )


def standard_error(result: MonteCarloResult) -> np.ndarray:
    """Standard error of the mean r per scan. [B3]

    Serves: the confidence band in the r_montecarlo plot.

    Args:
        result: an aggregate from `run_monte_carlo` or `mean_r`.

    Returns:
        Shape (K,) array of r_std / sqrt(n_runs).
    """
    return result.r_std / np.sqrt(result.n_runs)
