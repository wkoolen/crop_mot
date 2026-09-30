"""The B3 cross-check of one run folder: every track against the analytic reference. [B3]

Shared by `analyse`, which writes one run's result to metrics.json and plots it, and by
the Monte-Carlo trial loop, which repeats it over seeds (decision D17). Both therefore
check exactly the same thing: for each track, the event sequence is built from the
filter's own predicted moments (`build_scan_events`), the reference evaluates the A2
recursion on it, and `compare_r` measures the difference with the absolute tolerance of
decision D18.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.analysis.analytic import REFERENCES
from crop_mot.analysis.estimates_log import r_trajectory, read_estimates, track_lifetimes
from crop_mot.analysis.events import branch_counts, build_scan_events, predicted_track_moments
from crop_mot.analysis.metrics import RComparison, compare_r
from crop_mot.config import FilterConfig, RunConfig
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import unpruned_log_name
from crop_mot.sensor.record import read_detections


@dataclass(frozen=True)
class TrackCheck:
    """One track's r against the reference. [B3]

    Attributes:
        track_id: the track.
        r_sim: shape (K,), the filter's r per scan, 0 before the birth.
        r_ref: shape (K,), the reference's.
        comparison: the error summary.
        branches: scans per branch of the recursion, from `events.branch_counts`.
    """

    track_id: int
    r_sim: np.ndarray
    r_ref: np.ndarray
    comparison: RComparison
    branches: dict[str, int]


def checked_log_name(filter_cfg: FilterConfig) -> str:
    """The estimates log the cross-check reads: the unpruned companion when pruning. [B3]

    A2 has no deletion step, so a pruned track cannot be checked past its deletion. The
    unpruned log (decision D14) is identical up to each deletion and runs on after it.
    """
    if filter_cfg.prune.r_min > 0.0:
        return unpruned_log_name(filter_cfg.kind)
    return filter_cfg.kind


def cross_check_run(run: RunDir, cfg: RunConfig) -> list[TrackCheck]:
    """Check every track of a run folder against the configured reference. [B3]

    Args:
        run: a run folder the configured filter has run on (`run_configured_filter`).
        cfg: its run config; `analysis.b3_reference` names the reference.

    Returns:
        One TrackCheck per track in the checked log, in order of first appearance; empty
        when nothing was born.

    Raises:
        ValueError: if the config names no b3_reference.
    """
    if cfg.analysis.b3_reference is None:
        raise ValueError("the B3 cross-check needs analysis.b3_reference in the run config")
    reference = REFERENCES[cfg.analysis.b3_reference](p_S=cfg.filter_cfg.survival.p_S,
                                                       r_birth=cfg.filter_cfg.birth.r_b)
    records = read_estimates(run.estimates(checked_log_name(cfg.filter_cfg)))
    times = [scan.t for scan in read_detections(run.detections)]

    checks = []
    for track_id in track_lifetimes(records):
        moments = predicted_track_moments(records, track_id, times, cfg.filter_cfg)
        events = build_scan_events(run.detections, run.labels, cfg.filter_cfg, moments)
        r_sim = r_trajectory(records, track_id)
        r_ref = reference.r_sequence(events)
        checks.append(TrackCheck(track_id=track_id, r_sim=r_sim, r_ref=r_ref,
                                 comparison=compare_r(r_sim, r_ref),
                                 branches=branch_counts(events)))
    return checks
