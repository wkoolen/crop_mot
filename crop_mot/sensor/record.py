"""Recording detections to disk, and reading them back. [B1/B2]

This module is what makes the fair-comparison requirement structural instead of a promise.
B1 writes detections.jsonl once; every filter run - Bernoulli, GNN, JPDA, PMBM - reads that
same file. There is no code path in which two filters could see different data, because
there is no code path in which a filter calls the simulator.

detections.jsonl and labels.jsonl are written side by side but are read by different
consumers: the filter runner reads only the former, the analysis reads both.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from crop_mot.io import read_jsonl, write_jsonl
from crop_mot.types import Detection, Pose2D, Scan, ScanLabels


def write_detections(path: Path, scans: list[Scan]) -> None:
    """Write all scans to detections.jsonl - the filter's only input.

    One JSON object per scan, holding k, t, the reported pose, and the list of z vectors.
    Deliberately contains no origin information of any kind.

    Serves: [B1] run-folder output; [B2/B4] the shared input that makes filter comparison fair.

    Args:
        path: destination detections.jsonl.
        scans: the scans to write, in increasing k.
    """
    records = []
    for scan in scans:
        records.append({
            "k": scan.k,
            "t": scan.t,
            "pose": asdict(scan.pose),
            "z": [detection.z for detection in scan.detections],
        })
    write_jsonl(path, records)


def read_detections(path: Path) -> list[Scan]:
    """Read detections.jsonl back into Scan objects.

    Serves: [B2] the filter runner; [B4] every baseline, reading the identical file.

    Args:
        path: source detections.jsonl.

    Returns:
        The scans in file order.
    """
    scans = []
    for record in read_jsonl(path):
        detections = tuple(Detection(z=np.asarray(z, dtype=float)) for z in record["z"])
        scans.append(Scan(k=record["k"], t=record["t"], pose=Pose2D(**record["pose"]),
                          detections=detections))
    return scans


def write_labels(path: Path, labels: list[ScanLabels]) -> None:
    """Write the truth-side detection origins to labels.jsonl. EVALUATION ONLY. [B1/B3]

    Kept in a separate file from detections.jsonl on purpose. A filter is handed a path to
    detections.jsonl and nothing else, so it cannot reach this data even by mistake.

    Args:
        path: destination labels.jsonl.
        labels: per-scan origin records, in increasing k.
    """
    # asdict keeps a clutter origin as None, which JSON writes as null - never as a
    # sentinel id that could be mistaken for plant 0.
    write_jsonl(path, [asdict(label) for label in labels])


def read_labels(path: Path) -> list[ScanLabels]:
    """Read labels.jsonl back. EVALUATION ONLY. [B3]

    Serves: [B3] building the ScanEvent sequence, which needs to know how many of the gated
    detections were clutter.

    Args:
        path: source labels.jsonl.

    Returns:
        The label records in file order.
    """
    labels = []
    for record in read_jsonl(path):
        labels.append(ScanLabels(
            k=record["k"],
            origin=tuple(record["origin"]),
            visible_ids=tuple(record["visible_ids"]),
            detected_ids=tuple(record["detected_ids"]),
            truncated_ids=tuple(record.get("truncated_ids", ())),
            weed_origin=tuple(record.get("weed_origin", ())),
            visible_weed_ids=tuple(record.get("visible_weed_ids", ())),
        ))
    return labels
