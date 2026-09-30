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

from crop_mot.analysis.counts import weed_origin
from crop_mot.analysis.estimates_log import ScanEstimates, read_estimates, track_lifetimes
from crop_mot.analysis.events import gated_detection_indices, predicted_track_moments
from crop_mot.analysis.metrics import GospaResult, gospa
from crop_mot.config import RunConfig, load_run_config
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.types import TrackEstimate
from crop_mot.world.truth import GroundTruth, read_truth

# A track is confirmed when r > R_CONF (decision D28): it makes the set estimate GOSPA
# scores and picks the tracks NEES checks.
R_CONF = 0.5
# GOSPA cutoff and order (decision D31): 0.5 m is 2.5 measurement sigmas, and a track
# further than that from every plant is a false track, not a poor position.
GOSPA_C = 0.5
GOSPA_P = 2.0
# A track is on a plant or weed when its mean is within D_MATCH of it, the nearest one
# deciding (decision D28, revised): the GOSPA cutoff, so a fate and GOSPA agree on what
# a false track is. The first value, 0.2 m, labelled plant captures sitting 0.2-0.3 m off
# the plant as "sustained by clutter".
D_MATCH = GOSPA_C


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


def scan_views(run: RunDir, filter_name: str, cfg: RunConfig | None = None
               ) -> list[ScanView]:
    """Each scan's in-view tracks and in-view true plants, for one filter's log. [B4]

    Args:
        run: a run folder the filter has run on.
        filter_name: which estimates log to read (e.g. "bernoulli_bank", or an unpruned
            companion's name).
        cfg: the run config, for the scenario's FOV; None reads the folder's copy. The
            Monte-Carlo trial loop's folders have none and pass it.

    Returns:
        One ScanView per scan, in scan order.
    """
    cfg = load_run_config(run.config) if cfg is None else cfg
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


def existence_density(tracks: list[TrackEstimate] | tuple[TrackEstimate, ...],
                      xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """D(x) = sum_i r_i N(x; m_i, P_i) on a grid: the PHD of any filter's output. [B4, step 5]

    Integrates to the sum of r over the tracks, the expected number of objects, so every
    method can be drawn as the same kind of heatmap; the PHD grid filter (roadmap step 14)
    will draw its own grid on the same axes.

    Args:
        tracks: the tracks to sum, with 2D means.
        xs: shape (nx,), grid x coordinates in metres.
        ys: shape (ny,), grid y coordinates in metres.

    Returns:
        Shape (ny, nx), the density in objects per m^2.
    """
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)            # (ny, nx, 2)
    density = np.zeros(grid.shape[:2])
    for track in tracks:
        d = grid - track.mean[:2]
        P = track.cov[:2, :2]
        mahalanobis_sq = np.einsum("...i,ij,...j->...", d, np.linalg.inv(P), d)
        normaliser = 2.0 * np.pi * np.sqrt(np.linalg.det(P))
        density += track.r * np.exp(-0.5 * mahalanobis_sq) / normaliser
    return density


@dataclass(frozen=True)
class GateContents:
    """What fell inside each track's gate, by true origin, per scan (roadmap step 8a). [B4]

    The failure modes of a known-N map, measured from the labels: a clutter return, a weed
    return or a neighbour plant's detection inside a plant's gate, and the plant pulled
    away. A track's own plant is the true plant nearest its first reported mean - for a
    planting-plan slot, the plant planted there - so this works for any filter. Only
    in-view tracks with predicted moments count, i.e. from scan 1 on (the log does not hold
    the prior of scan 0).

    Attributes:
        own, neighbour, clutter, weed: shape (K,), mean number of gated detections per
            in-view track from its own plant, from another plant, from Poisson clutter and
            from a weed.
        pulled: shape (K,), the share of in-view tracks whose mean, after the update, is
            nearer another plant than their own.
        n_tracks: shape (K,), how many in-view tracks each scan averages over.
        shares: over all track-scans of the run, the share with at least one neighbour,
            clutter and weed detection in the gate, and the share pulled (descriptive for
            one run; across seeds each seed would be one outcome).
    """

    own: np.ndarray
    neighbour: np.ndarray
    clutter: np.ndarray
    weed: np.ndarray
    pulled: np.ndarray
    n_tracks: np.ndarray
    shares: dict[str, float]


def gate_contents(run: RunDir, filter_name: str) -> GateContents:
    """Gate contents by true origin and pulled tracks, per scan, for one filter's log. [B4]

    The gate is the filter's own, rebuilt from its log (`predicted_track_moments`,
    `gated_detection_indices`) with the run config's common `measurement` and `gate`
    blocks - fields every filter shares, not one filter's layout.
    """
    cfg = load_run_config(run.config)
    records = read_estimates(run.estimates(filter_name))
    truth = read_truth(run.truth)
    labels = read_labels(run.labels)
    scans = read_detections(run.detections)
    fov = cfg.scenario.sensor.fov
    plants = truth.field.positions
    plant_ids = truth.field.ids.tolist()
    times = [scan.t for scan in scans]

    n_scans = len(records)
    sums = {key: np.zeros(n_scans) for key in ("own", "neighbour", "clutter", "weed", "pulled")}
    n_tracks = np.zeros(n_scans)
    any_counts = {"neighbour": 0, "clutter": 0, "weed": 0, "pulled": 0}
    n_track_scans = 0
    for track_id, lifetime in track_lifetimes(records).items():
        first = [e for e in records[lifetime.k_birth].estimates if e.track_id == track_id][0]
        own = plant_ids[int(np.argmin(np.linalg.norm(plants - first.mean, axis=1)))]
        moments = predicted_track_moments(records, track_id, times, cfg.filter_cfg)
        for k, moment in enumerate(moments):
            if moment is None or not in_fov(moment[0], truth.poses[k].true, fov):
                continue
            posterior = [e for e in records[k].estimates if e.track_id == track_id]
            if not posterior:
                continue
            origins = labels[k].origin
            weeds = weed_origin(labels[k])
            counts = {"own": 0, "neighbour": 0, "clutter": 0, "weed": 0}
            for i in gated_detection_indices(scans[k], *moment, cfg.filter_cfg):
                if weeds[i] is not None:
                    counts["weed"] += 1
                elif origins[i] is None:
                    counts["clutter"] += 1
                elif origins[i] == own:
                    counts["own"] += 1
                else:
                    counts["neighbour"] += 1
            nearest = plant_ids[int(np.argmin(np.linalg.norm(plants - posterior[0].mean,
                                                             axis=1)))]
            pulled = nearest != own
            for key, value in counts.items():
                sums[key][k] += value
            sums["pulled"][k] += pulled
            n_tracks[k] += 1
            n_track_scans += 1
            for key in ("neighbour", "clutter", "weed"):
                any_counts[key] += counts[key] > 0
            any_counts["pulled"] += pulled

    with np.errstate(invalid="ignore", divide="ignore"):
        means = {key: np.where(n_tracks > 0, value / n_tracks, np.nan)
                 for key, value in sums.items()}
    shares = {f"{key}_share": (value / n_track_scans if n_track_scans else float("nan"))
              for key, value in any_counts.items()}
    return GateContents(own=means["own"], neighbour=means["neighbour"],
                        clutter=means["clutter"], weed=means["weed"], pulled=means["pulled"],
                        n_tracks=n_tracks, shares=shares)


@dataclass(frozen=True)
class MissingPlants:
    """How well a filter's r finds the empty slots of the plan (roadmap step 8c, D23). [B4]

    Every slot of the field - a planted plant, or an empty slot at its nominal position
    (truth) - gets a score per scan: the largest r among the reported tracks whose mean
    is nearer to it than to any other slot and within D_MATCH (D28); 0 when no track is
    there. A low score says "nothing here". Scored from positions, not track ids, so a
    method with births (N unknown) is scored the same way.

    Attributes:
        slot_ids: the slots, increasing.
        empty: shape (n_slots,), whether each slot is truly empty.
        scores: shape (K, n_slots), the score per scan.
        seen: shape (K, n_slots), whether the slot has been in view at least once up to
            and including the scan.
        n_scans_in_view: shape (n_slots,), how many scans each slot was in view in all.
    """

    slot_ids: np.ndarray
    empty: np.ndarray
    scores: np.ndarray
    seen: np.ndarray
    n_scans_in_view: np.ndarray

    def rates(self, threshold: float = R_CONF) -> tuple[float, float]:
        """At the last scan, over seen slots: (empty slots with score < threshold, as a
        share of seen empty slots; planted slots with score < threshold, as a share of
        seen planted slots). NaN where a group is empty."""
        seen, low = self.seen[-1], self.scores[-1] < threshold
        empty, planted = seen & self.empty, seen & ~self.empty
        found = low[empty].mean() if empty.any() else float("nan")
        flagged = low[planted].mean() if planted.any() else float("nan")
        return float(found), float(flagged)

    def decision_scans(self, threshold: float = R_CONF) -> list[int | None]:
        """Per seen empty slot: in-view scans until its score falls below the threshold
        and stays there to the end, or None if it never does."""
        result = []
        for j in np.flatnonzero(self.empty & self.seen[-1]):
            low = self.scores[:, j] < threshold
            if not low[-1]:
                result.append(None)
                continue
            last_high = np.flatnonzero(~low)
            k_decided = int(last_high[-1]) + 1 if len(last_high) else 0
            first_seen = int(np.flatnonzero(self.seen[:, j])[0])
            result.append(max(0, k_decided - first_seen))
        return result


def missing_plants(run: RunDir, filter_name: str) -> MissingPlants | None:
    """Slot scores for finding the empty slots; None when the field has none. [B4, 8c]"""
    cfg = load_run_config(run.config)
    truth = read_truth(run.truth)
    if not len(truth.field.missing_ids):
        return None
    records = read_estimates(run.estimates(filter_name))
    fov = cfg.scenario.sensor.fov
    slot_ids = np.concatenate([truth.field.ids, truth.field.missing_ids])
    positions = np.vstack([truth.field.positions, truth.field.missing_positions])
    order = np.argsort(slot_ids)
    slot_ids, positions = slot_ids[order], positions[order]
    empty = np.isin(slot_ids, truth.field.missing_ids)

    n_scans, n_slots = len(records), len(slot_ids)
    scores = np.zeros((n_scans, n_slots))
    in_view = np.zeros((n_scans, n_slots), dtype=bool)
    for k, (record, sample) in enumerate(zip(records, truth.poses)):
        in_view[k] = [in_fov(p, sample.true, fov) for p in positions]
        for track in record.estimates:
            distances = np.linalg.norm(positions - track.mean[:2], axis=1)
            j = int(np.argmin(distances))
            if distances[j] <= D_MATCH:
                scores[k, j] = max(scores[k, j], track.r)
    return MissingPlants(slot_ids=slot_ids, empty=empty, scores=scores,
                         seen=np.logical_or.accumulate(in_view, axis=0),
                         n_scans_in_view=in_view.sum(axis=0))
