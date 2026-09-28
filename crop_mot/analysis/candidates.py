"""Clutter detections suitable as phantom seeds for the bank experiment. [B2]

EVALUATION ONLY: reads labels.jsonl and truth.jsonl, which no filter ever reads. This is
the config author's tool for choosing `birth.seeds` for the `bernoulli_bank` filter
(decision D12) - the birth model itself stays truth-blind and is only handed indices.

A good phantom seed is a clutter detection in empty space: far enough from every plant
that its gate cannot reach one (otherwise the phantom is captured, decision D9) and far
enough from the other seeds that two tracks do not share detections (the bank's
independence approximation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.analysis.plots import load_run_folder_config
from crop_mot.config import FieldOfView, RunConfig
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.world.truth import read_truth


@dataclass(frozen=True)
class PhantomCandidate:
    """One clutter detection that could seed a phantom track. [B2]

    Attributes:
        k: scan index of the detection.
        detection_index: its index within the scan - what goes into `birth.seeds`.
        z: shape (dim_z,), the detection.
        plant_distance: distance from z to the nearest plant, in metres.
        n_in_view_after: how many LATER scans have z inside the FOV at the scan's reported
            pose - an upper bound on how often the phantom can be looked at and missed.
            A small count means r will freeze rather than decay (p_D = 0 out of view).
    """

    k: int
    detection_index: int
    z: np.ndarray
    plant_distance: float
    n_in_view_after: int


def phantom_candidates(
    run: RunDir, fov: FieldOfView, min_distance: float, min_separation: float = 0.0
) -> list[PhantomCandidate]:
    """Clutter detections at least `min_distance` from every plant, pairwise separated. [B2]

    Args:
        run: a run folder holding truth, labels and detections.
        fov: the FOV used to count later in-view scans; normally the filter's assumed FOV.
        min_distance: minimum distance from z to the nearest plant, in metres.
        min_separation: minimum distance between any two returned candidates. Applied
            greedily, keeping the candidate farther from the plants first; 0 keeps all.

    Returns:
        The candidates, in (k, detection_index) order.
    """
    plants = read_truth(run.truth).field.positions
    scans = read_detections(run.detections)
    labels = read_labels(run.labels)

    found = []
    for scan, scan_labels in zip(scans, labels):
        for i, (detection, origin) in enumerate(zip(scan.detections, scan_labels.origin)):
            if origin is not None:
                continue
            distance = float(np.min(np.linalg.norm(plants - detection.z, axis=1)))
            if distance < min_distance:
                continue
            n_in_view_after = sum(in_fov(detection.z, later.pose, fov)
                                  for later in scans[scan.k + 1:])
            found.append(PhantomCandidate(k=scan.k, detection_index=i, z=detection.z,
                                          plant_distance=distance,
                                          n_in_view_after=n_in_view_after))

    kept: list[PhantomCandidate] = []
    for candidate in sorted(found, key=lambda c: -c.plant_distance):
        if all(np.linalg.norm(candidate.z - other.z) >= min_separation for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: (c.k, c.detection_index))


def assumed_fov(run: RunDir) -> FieldOfView:
    """The FOV to count in-view scans with: the filter's assumed FOV for a track run, the
    scenario's sensor FOV for a simulate-only run."""
    cfg = load_run_folder_config(run)
    if isinstance(cfg, RunConfig):
        return cfg.filter_cfg.assumed_sensor.fov
    return cfg.sensor.fov


def format_candidates(candidates: list[PhantomCandidate]) -> str:
    """A table for the terminal, ready to copy seeds from. [B2]"""
    lines = ["   k  index            z        to plant  in view after"]
    for c in candidates:
        lines.append(f"{c.k:4d}  {c.detection_index:5d}  ({c.z[0]:5.2f}, {c.z[1]:5.2f})"
                     f"  {c.plant_distance:7.2f} m  {c.n_in_view_after:6d} scans")
    return "\n".join(lines)
