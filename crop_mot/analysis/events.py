"""Building the per-scan event sequence the B3 closed form consumes. [B3]

This module is where the truth files and the filter's assumed model meet. It runs after the
filter, reads detections.jsonl, labels.jsonl and truth.jsonl, and produces the ScanEvent
sequence that `BernoulliExistenceReference` turns into an r trajectory.

The p_D and lambda_FA it records are the FILTER's assumed values, not the simulator's. That
is deliberate and it is easy to get backwards: B3 asks "does the filter's r match the closed
form derived from the filter's own assumptions?", which is a check of the implementation.
Using the simulator's true values instead would be asking "is the filter's model correct?",
which is a different question and is what the Monte-Carlo half answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crop_mot.analysis.analytic import ScanEvent
from crop_mot.analysis.estimates_log import ScanEstimates
from crop_mot.association.gating import chi2_threshold, gate_measurements
from crop_mot.config import FilterConfig
from crop_mot.filters.kalman import kf_predict, predicted_measurement
from crop_mot.motion.models import StaticTarget
from crop_mot.sensor.models import build_measurement_model
from crop_mot.types import Scan


def build_scan_events(
    detections_path: Path,
    labels_path: Path,
    cfg: FilterConfig,
    track_mean_per_scan: list[np.ndarray],
) -> list[ScanEvent]:
    """Assemble the ScanEvent sequence for the B3 cross-check.

    For each scan:
      * evaluate the FILTER's assumed p_D and lambda_FA at the hypothesised track location,
        using the scan's reported pose - so a range-dependent profile produces a different
        value each scan, and a constant profile produces the same one every scan;
      * record whether that location was inside the assumed FOV;
      * count how many detections fell inside the gate, and - from labels.jsonl, which only
        this side of the pipeline may read - how many of those were clutter.

    Takes the filter's per-scan predicted mean as an argument rather than recomputing it,
    because p_D at the hypothesised state is only well defined relative to where the filter
    thought the target was. Recomputing it here would risk silently evaluating a different
    trajectory than the one the filter actually followed, and the resulting mismatch would
    look like a bug in the closed form.

    Serves: [B3].

    Args:
        detections_path: run.detections.
        labels_path: run.labels. EVALUATION ONLY.
        cfg: the filter config, supplying the ASSUMED sensor model and the gate.
        track_mean_per_scan: the filter's predicted mean at each scan, from the estimates log.

    Returns:
        One ScanEvent per scan, in increasing k.
    """
    raise NotImplementedError


def predicted_track_moments(
    records: list[ScanEstimates], track_id: int, times: list[float], cfg: FilterConfig
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """The filter's PREDICTED (mean, cov) of one track at each scan, from its estimates log.

    The log holds the POSTERIOR moments after each scan. The prior the filter used at scan
    k is the posterior of scan k-1 pushed through the configured motion model over
    dt = t_k - t_{k-1} - the same `kf_predict` call the filter makes, so for StaticTarget
    this reproduces the filter's prior exactly. Nothing is re-filtered here.

    Serves: [B2] marking gated scans in the r-vs-k plot; [B3] building ScanEvents.

    Args:
        records: the estimates log, one record per scan.
        track_id: which track to follow.
        times: scan timestamps t_k, one per record.
        cfg: the filter config, supplying the motion model.

    Returns:
        One entry per scan: the predicted (mean, cov), or None where the track was not
        reported at the previous scan (before and at its birth scan).
    """
    if cfg.motion.kind != "static":
        raise ValueError(f"unknown motion kind {cfg.motion.kind!r}")
    motion = StaticTarget(q=cfg.motion.q, dim_x=cfg.birth.init_cov.shape[0])

    moments = [None]
    for k in range(1, len(records)):
        previous = [e for e in records[k - 1].estimates if e.track_id == track_id]
        if not previous:
            moments.append(None)
            continue
        dt = times[k] - times[k - 1]
        moments.append(kf_predict(previous[0].mean, previous[0].cov, motion, dt))
    return moments


def gated_detection_indices(
    scan: Scan, mean: np.ndarray, cov: np.ndarray, cfg: FilterConfig
) -> np.ndarray:
    """Indices of this scan's detections inside the track's gate, as the filter gates them.

    Uses the filter's measurement model and chi-square threshold, and the scan's REPORTED
    pose, so the count matches what the filter saw.

    Serves: [B2] the r-vs-k plot's gated-detection markers; [B3] ScanEvent.n_gated.

    Args:
        scan: the scan.
        mean, cov: the track's predicted moments at this scan.
        cfg: the filter config (measurement model and gate).

    Returns:
        Shape (n_gated,) integer array of detection indices, increasing.
    """
    measurement = build_measurement_model(cfg.measurement)
    z_hat, S = predicted_measurement(mean, cov, measurement, scan.pose)
    Z = np.array([d.z for d in scan.detections]).reshape(-1, measurement.dim_z)
    threshold = chi2_threshold(cfg.gate.chi2_prob, measurement.dim_z)
    return gate_measurements(Z, z_hat, S, threshold)
