"""Per-scan evaluation of any filter's estimates against truth. [B4, roadmap step 5]

Reads only the run folder: one filter's estimates log, truth.jsonl, labels.jsonl and the
scenario's FOV from the config copy. Nothing here knows how a filter works or how it is
configured, so every series below - and every figure drawn from it - works for every
method that writes `list[TrackEstimate]` (roadmap §2, rule 1).

"In view" is decided by one function on both sides (decision D21): a true plant is in view
when the simulator listed it in `ScanLabels.visible_ids`, which is `in_fov` at the true
pose and the scenario's FOV, and a track is in view when `in_fov` at the same pose and FOV
holds for its mean. Evaluation code, so it may read truth; no filter ever calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from crop_mot.analysis.estimates_log import ScanEstimates, read_estimates
from crop_mot.analysis.metrics import GospaResult, gospa
from crop_mot.config import RunConfig, load_run_config
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.record import read_labels
from crop_mot.types import TrackEstimate
from crop_mot.world.truth import GroundTruth, read_truth

# A track is confirmed when r > R_CONF (decision D28): it makes the set estimate GOSPA
# scores and picks the tracks NEES checks.
R_CONF = 0.5
# GOSPA cutoff and order (decision D31): 0.5 m is 2.5 measurement sigmas, and a track
# further than that from every plant is a false track, not a poor position.
GOSPA_C = 0.5
GOSPA_P = 2.0


@dataclass(frozen=True)
class ScanView:
    """One scan as the evaluation sees it: what is in view, on both sides. [B4]

    Attributes:
        k: scan index.
        tracks: the reported tracks whose mean is in view.
        plant_positions: shape (n, dim_x), the true positions of the plants in view.
    """

    k: int
    tracks: tuple[TrackEstimate, ...]
    plant_positions: np.ndarray


def scan_views(run: RunDir, filter_name: str) -> list[ScanView]:
    """Each scan's in-view tracks and in-view true plants, for one filter's log. [B4]

    Args:
        run: a run folder the filter has run on.
        filter_name: which estimates log to read (e.g. "bernoulli_bank", or an unpruned
            companion's name).

    Returns:
        One ScanView per scan, in scan order.
    """
    cfg = load_run_config(run.config)
    return _views(read_estimates(run.estimates(filter_name)), read_truth(run.truth),
                  read_labels(run.labels), cfg)


def _views(records: list[ScanEstimates], truth: GroundTruth, labels, cfg: RunConfig
           ) -> list[ScanView]:
    fov = cfg.scenario.sensor.fov
    position_of = dict(zip(truth.field.ids.tolist(), truth.field.positions))
    dim_x = truth.field.positions.shape[1]
    views = []
    for record, scan_labels, sample in zip(records, labels, truth.poses):
        pose = sample.true
        tracks = tuple(e for e in record.estimates if in_fov(e.mean, pose, fov))
        plants = np.array([position_of[i] for i in scan_labels.visible_ids]).reshape(-1, dim_x)
        views.append(ScanView(k=record.k, tracks=tracks, plant_positions=plants))
    return views


@dataclass(frozen=True)
class Cardinality:
    """Expected number of tracks in view against the number of plants in view. [B4]

    Attributes:
        sum_r: shape (K,), sum of r over the tracks whose mean is in view (D21).
        n_plants: shape (K,), the number of true plants in view.
    """

    sum_r: np.ndarray
    n_plants: np.ndarray


def cardinality(views: list[ScanView]) -> Cardinality:
    """Sum of r in view, and plants in view, per scan (decision D21). [B4]

    Sum of r is the expected number of objects the filter believes are in view, so for a
    well-behaved method the two curves agree. For known-N methods they differ only where
    the map and the field differ.
    """
    return Cardinality(
        sum_r=np.array([sum(track.r for track in view.tracks) for view in views]),
        n_plants=np.array([len(view.plant_positions) for view in views]),
    )


def confirmed(view: ScanView, r_conf: float = R_CONF) -> list[TrackEstimate]:
    """The in-view tracks with r > r_conf: the set estimate at one scan (D28)."""
    return [track for track in view.tracks if track.r > r_conf]


def gospa_series(views: list[ScanView], c: float = GOSPA_C, p: float = GOSPA_P,
                 r_conf: float = R_CONF) -> list[GospaResult]:
    """GOSPA per scan, between the confirmed tracks in view and the plants in view. [B4]

    Both sets are restricted to the view (D21): a plant behind the robot is not a missed
    object, and a track behind it is not a false one.
    """
    return [gospa(confirmed(view, r_conf), view.plant_positions, c, p) for view in views]


@dataclass(frozen=True)
class Nees:
    """Normalised estimation error squared, averaged over matched tracks per scan. [B4]

    Attributes:
        mean_nees: shape (K,), the average of e' P^-1 e over the scan's matched pairs,
            NaN where no confirmed track is matched.
        n_pairs: shape (K,), how many pairs were averaged.
        dim: the state dimension, the degrees of freedom of one pair's NEES.
    """

    mean_nees: np.ndarray
    n_pairs: np.ndarray
    dim: int


def nees(views: list[ScanView], results: list[GospaResult], r_conf: float = R_CONF) -> Nees:
    """NEES of every confirmed track matched to a plant by GOSPA's assignment. [B4]

    For a pair, e = m - x_true and NEES = e' P^-1 e; if the reported covariance is honest,
    it is chi-square with dim degrees of freedom, and the average over n pairs is
    chi-square with n dim degrees of freedom, divided by n (`nees_band`). This is the check
    on the Gaussian part that the A2 cross-check cannot give.

    Pairs come from the GOSPA assignment at that scan, so they are within the cutoff c.
    Known-N methods will instead match track i to planned plant i (roadmap step 8a); that
    rule arrives with them.

    Args:
        views: from `scan_views`.
        results: `gospa_series(views, ...)` with the same r_conf.
        r_conf: the confirmation threshold the GOSPA series used.
    """
    mean_nees, n_pairs = [], []
    for view, result in zip(views, results):
        tracks = confirmed(view, r_conf)
        values = []
        for i, j in result.pairs:
            e = tracks[i].mean - view.plant_positions[j]
            values.append(float(e @ np.linalg.solve(tracks[i].cov, e)))
        mean_nees.append(np.mean(values) if values else np.nan)
        n_pairs.append(len(values))
    dim = views[0].plant_positions.shape[1] if views else 2
    return Nees(mean_nees=np.array(mean_nees), n_pairs=np.array(n_pairs), dim=dim)


def nees_band(n_pairs: np.ndarray, dim: int, prob: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """The two-sided `prob` band of the average NEES over n pairs, per scan. [B4]

    The average of n independent chi-square(dim) values is chi-square(n dim) / n. NaN
    where n = 0.

    Returns:
        (lower, upper), each shaped like n_pairs.
    """
    n = np.asarray(n_pairs, dtype=float)
    tail = (1.0 - prob) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lower = np.where(n > 0, chi2.ppf(tail, n * dim) / n, np.nan)
        upper = np.where(n > 0, chi2.ppf(1.0 - tail, n * dim) / n, np.nan)
    return lower, upper
