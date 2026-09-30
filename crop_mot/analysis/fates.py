"""The controlled phantom's fate in each seed, and fate proportions over seeds. [B3, step 4a]

Roadmap step 4a's robustness half (decisions D17, D28, D34). The phantom is placed at the
same empty spot in every seed, so each seed is one trial of the same experiment, and the
phantom's fate is that seed's ONE outcome: tracks in a seed share clutter and neighbours,
so only seeds are independent. Proportions over seeds get Wilson intervals.

Fates, classified at the end of the run from the estimates log and truth (D28):
    pruned                 the bank deleted it (r fell below r_min, A2 §5)
    confirmed_on_plant     alive, r > r_conf, nearest plant or weed within d_match is a plant
    confirmed_on_weed      the same, nearest is a weed
    sustained_by_clutter   alive, r > r_conf, no plant or weed within d_match
    alive_unconfirmed      alive, r <= r_conf

What was near the phantom is part of the random scene, and averaging over seeds covers it.
To see its effect the seeds are stratified, from truth, by what fell inside the phantom's
gate over its life: "weed_in_gate" when at least one weed return did. The filter cannot
tell a weed return from clutter (`ScanEvent.n_clutter_gated` lumps them), so this reads
`ScanLabels.weed_origin` and `origin` here, on the analysis side, and adds no filter input.

Evaluation code: it reads truth and labels, and no filter calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.analysis.counts import weed_origin
from crop_mot.analysis.estimates_log import read_estimates, track_lifetimes
from crop_mot.analysis.evaluation import D_MATCH, R_CONF
from crop_mot.analysis.events import gated_detection_indices, predicted_track_moments
from crop_mot.config import RunConfig
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.world.truth import read_truth

FATES = ("pruned", "confirmed_on_plant", "confirmed_on_weed", "sustained_by_clutter",
         "alive_unconfirmed")
STRATA = ("all", "weed_in_gate", "no_weed_in_gate")
# Two-sided 95 % normal quantile, for the Wilson interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PhantomOutcome:
    """What happened to the phantom in one seed. [B3]

    The end state is kept next to the fate, so the fates can be reclassified with other
    thresholds without rerunning.

    Attributes:
        fate: one of FATES.
        k_end: the last scan it was reported: its deletion scan, or the last scan.
        r_end: its r at k_end.
        d_plant: distance from its mean at k_end to the nearest true plant, metres.
        d_weed: the same for the nearest weed; None when the field has no weeds.
        n_clutter_in_gate: Poisson clutter returns inside its gate, summed over its life.
        n_weed_in_gate: weed returns inside its gate, summed over its life.
    """

    fate: str
    k_end: int
    r_end: float
    d_plant: float
    d_weed: float | None
    n_clutter_in_gate: int
    n_weed_in_gate: int


def classify_fate(deleted: bool, r: float, d_plant: float, d_weed: float | None,
                  r_conf: float = R_CONF, d_match: float = D_MATCH) -> str:
    """The fate of a track from its end state (decision D28). [B3]

    Args:
        deleted: whether the track was pruned.
        r: its existence probability at the last scan it was reported.
        d_plant: distance from its mean then to the nearest true plant.
        d_weed: the same for the nearest weed, None when there are no weeds.
        r_conf: confirmation threshold.
        d_match: how close the nearest plant or weed must be for the track to be on it.

    Returns:
        One of FATES.
    """
    if deleted:
        return "pruned"
    if r <= r_conf:
        return "alive_unconfirmed"
    d_weed = np.inf if d_weed is None else d_weed
    if min(d_plant, d_weed) > d_match:
        return "sustained_by_clutter"
    return "confirmed_on_plant" if d_plant <= d_weed else "confirmed_on_weed"


def phantom_outcome(run: RunDir, cfg: RunConfig, track_id: int = 0) -> PhantomOutcome | None:
    """The phantom's fate in one run folder, and what fell in its gate. [B3]

    A measurement function for `montecarlo.run_trials`. Reads the filter's own (pruned)
    estimates log, so a deletion is a deletion; the gate at each scan of its life is
    rebuilt from the log exactly as the filter saw it (`predicted_track_moments`,
    `gated_detection_indices`).

    Args:
        run: a run folder the configured filter has run on.
        cfg: its run config.
        track_id: the phantom's track id; 0 for an injected phantom.

    Returns:
        The outcome, or None when the phantom was never reported.
    """
    records = read_estimates(run.estimates(cfg.filter_cfg.kind))
    lifetime = track_lifetimes(records).get(track_id)
    if lifetime is None:
        return None
    truth = read_truth(run.truth)
    labels = read_labels(run.labels)
    scans = read_detections(run.detections)

    n_clutter = n_weed = 0
    moments = predicted_track_moments(records, track_id, [scan.t for scan in scans],
                                      cfg.filter_cfg)
    for scan, scan_labels, moment in zip(scans, labels, moments):
        if moment is None:
            continue
        weeds = weed_origin(scan_labels)
        for i in gated_detection_indices(scan, *moment, cfg.filter_cfg):
            if weeds[i] is not None:
                n_weed += 1
            elif scan_labels.origin[i] is None:
                n_clutter += 1

    (last,) = [e for e in records[lifetime.k_last].estimates if e.track_id == track_id]
    d_plant = float(np.min(np.linalg.norm(truth.field.positions - last.mean, axis=1)))
    d_weed = (float(np.min(np.linalg.norm(truth.weeds - last.mean, axis=1)))
              if len(truth.weeds) else None)
    return PhantomOutcome(fate=classify_fate(lifetime.deleted, last.r, d_plant, d_weed),
                          k_end=lifetime.k_last, r_end=last.r, d_plant=d_plant, d_weed=d_weed,
                          n_clutter_in_gate=n_clutter, n_weed_in_gate=n_weed)


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for a proportion k / n; (0, 1) when n = 0. [B3]

    Chosen over the normal approximation because the fates include proportions near 0
    and 1 with tens of seeds, where the normal interval leaves [0, 1].
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass(frozen=True)
class FateProportion:
    """One fate's share of one stratum's seeds, with its 95 % Wilson interval. [B3]"""

    count: int
    n: int
    low: float
    high: float


def fate_proportions(outcomes: list[PhantomOutcome]) -> dict[str, dict[str, FateProportion]]:
    """Fate proportions per stratum: all seeds, and with / without a weed in the gate. [B3]

    Args:
        outcomes: one per seed in which the phantom was reported.

    Returns:
        proportions[stratum][fate], for every stratum in STRATA and fate in FATES.
    """
    groups = {
        "all": outcomes,
        "weed_in_gate": [o for o in outcomes if o.n_weed_in_gate > 0],
        "no_weed_in_gate": [o for o in outcomes if o.n_weed_in_gate == 0],
    }
    proportions = {}
    for stratum, group in groups.items():
        n = len(group)
        proportions[stratum] = {}
        for fate in FATES:
            count = sum(o.fate == fate for o in group)
            proportions[stratum][fate] = FateProportion(count, n, *wilson_interval(count, n))
    return proportions
