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

import math
from pathlib import Path

import numpy as np

from crop_mot.analysis.analytic import ScanEvent
from crop_mot.analysis.estimates_log import ScanEstimates
from crop_mot.association.gating import chi2_threshold, gate_measurements
from crop_mot.config import FilterConfig
from crop_mot.filters.kalman import kf_predict, log_predicted_likelihood, predicted_measurement
from crop_mot.motion.models import StaticTarget
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.models import build_measurement_model
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.sensor.sensor_model import build_sensor_model
from crop_mot.types import Scan


def build_scan_events(
    detections_path: Path,
    labels_path: Path,
    cfg: FilterConfig,
    track_moments_per_scan: list[tuple[np.ndarray, np.ndarray] | None],
) -> list[ScanEvent]:
    """Assemble the ScanEvent sequence for the B3 cross-check.

    For each scan:
      * evaluate the FILTER's assumed p_D and lambda_FA at the hypothesised track location,
        using the scan's reported pose - so a range-dependent profile produces a different
        value each scan, and a constant profile produces the same one every scan;
      * record whether that location was inside the assumed FOV;
      * count how many detections fell inside the gate, and - from labels.jsonl, which only
        this side of the pipeline may read - how many of those were clutter;
      * for each gated detection z_i, the likelihood ratio
        ell_i / kappa_i = p_D N(z_i; z_hat, S) / (lambda_FA c(z_i)), with z_hat and S from
        the shared `predicted_measurement` and N from `log_predicted_likelihood`, so the
        Gaussian algebra is the filter's own and never a suspect;
      * whether the track is born at this scan.

    Takes the filter's per-scan predicted moments as an argument rather than recomputing
    them, because p_D at the hypothesised state is only well defined relative to where the
    filter thought the target was. Recomputing them here would risk silently evaluating a
    different trajectory than the one the filter actually followed, and the resulting
    mismatch would look like a bug in the closed form. The covariance is needed as well as
    the mean: the gate and the likelihood both use S = H P H' + R (decision D25).

    The birth scan follows from the moments: the track is first reported at its birth
    scan, so the first non-None predicted moments are one scan later. Deriving it from the
    log rather than from the birth config makes it work for any birth model. Scans up to
    and including the birth have no moments; they get in_fov False, p_D 0, nothing gated,
    and `born` True at the birth scan only.

    What this makes the B3 check verify, and what it does not: the reference receives the
    filter's own predicted moments and its own ell_i / kappa_i, so it checks the EXISTENCE
    recursion given those. An error in the Gaussian update, in gating or in how p_D is
    evaluated is invisible to it, because both sides see the same numbers.

    Serves: [B3].

    Args:
        detections_path: run.detections.
        labels_path: run.labels. EVALUATION ONLY.
        cfg: the filter config, supplying the ASSUMED sensor model and the gate.
        track_moments_per_scan: the filter's predicted (mean, cov) at each scan, or None
            where the track was not reported at the previous scan - exactly what
            `predicted_track_moments` returns. For a pruned bank track, build it from the
            unpruned companion log (decision D14).

    Returns:
        One ScanEvent per scan, in increasing k.

    Raises:
        ValueError: if the moments and the scans differ in length; if the track is never
            reported before the last scan (no birth to find); or if the moments stop after
            the birth, i.e. the track was deleted - A2 has no deletion step.
    """
    scans = read_detections(detections_path)
    labels = read_labels(labels_path)
    if len(track_moments_per_scan) != len(scans):
        raise ValueError(f"{len(track_moments_per_scan)} predicted moments for {len(scans)} "
                         "scans; were they read from another run?")
    reported = [k for k, moments in enumerate(track_moments_per_scan) if moments is not None]
    if not reported:
        raise ValueError("the track has no predicted moments at any scan, so there is no "
                         "birth to check from; it was never reported before the last scan")
    k_birth = reported[0] - 1

    measurement = build_measurement_model(cfg.measurement)
    assumed = cfg.assumed_sensor
    sensor = build_sensor_model(assumed.fov, assumed.detection, assumed.lambda_FA, measurement)

    events = []
    for scan, scan_labels, moments in zip(scans, labels, track_moments_per_scan):
        dt = 0.0 if scan.k == 0 else scan.t - scans[scan.k - 1].t
        lambda_FA = sensor.lambda_FA(scan.pose)
        if moments is None:
            if scan.k > k_birth:
                raise ValueError(f"the track was deleted after scan {scan.k - 1}; A2 has no "
                                 "deletion step, so check it on the unpruned log (D14)")
            events.append(ScanEvent(k=scan.k, dt=dt, in_fov=False, p_D=0.0,
                                    lambda_FA=lambda_FA, n_gated=0, n_clutter_gated=0,
                                    born=scan.k == k_birth))
            continue

        mean, cov = moments
        p_D = sensor.p_D(mean, scan.pose)
        z_hat, S = predicted_measurement(mean, cov, measurement, scan.pose)
        gated = gated_detection_indices(scan, mean, cov, cfg)
        ratios = []
        for i in gated:
            z = scan.detections[i].z
            ell = p_D * np.exp(log_predicted_likelihood(z, z_hat, S))
            kappa = lambda_FA * sensor.clutter_density(z, scan.pose)
            if ell == 0.0:
                ratios.append(0.0)
            elif kappa == 0.0:
                ratios.append(math.inf)
            else:
                ratios.append(float(ell / kappa))
        n_clutter_gated = sum(1 for i in gated if scan_labels.origin[i] is None)
        events.append(ScanEvent(k=scan.k, dt=dt, in_fov=in_fov(mean, scan.pose, assumed.fov),
                                p_D=p_D, lambda_FA=lambda_FA, n_gated=len(gated),
                                n_clutter_gated=n_clutter_gated,
                                likelihood_ratios=tuple(ratios)))
    return events


def branch_counts(events: list[ScanEvent]) -> dict[str, int]:
    """How many scans of each branch of the A2 recursion an event sequence exercises. [B3]

    A cross-check that passed only on miss scans says nothing about the detection
    branches, so the coverage is reported next to the error. Scans before the birth are
    counted apart: the reference returns r = 0 there without evaluating a branch.

    Args:
        events: one track's sequence, from `build_scan_events`.

    Returns:
        Counts under the keys "before_birth", "birth", "out_of_view", "miss",
        "one_detection" and "several_detections"; they sum to len(events).
    """
    counts = dict.fromkeys(("before_birth", "birth", "out_of_view", "miss", "one_detection",
                            "several_detections"), 0)
    born = False
    for event in events:
        if event.born:
            born = True
            counts["birth"] += 1
        elif not born:
            counts["before_birth"] += 1
        elif not event.in_fov:
            counts["out_of_view"] += 1
        elif event.n_gated == 0:
            counts["miss"] += 1
        elif event.n_gated == 1:
            counts["one_detection"] += 1
        else:
            counts["several_detections"] += 1
    return counts


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
