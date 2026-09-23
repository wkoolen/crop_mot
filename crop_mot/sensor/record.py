"""Recording detections to disk, and reading them back. [B1/B2]

This module is what makes the fair-comparison requirement structural instead of a promise.
B1 writes detections.jsonl once; every filter run - Bernoulli, GNN, JPDA, PMBM - reads that
same file. There is no code path in which two filters could see different data, because
there is no code path in which a filter calls the simulator.

detections.jsonl and labels.jsonl are written side by side but are read by different
consumers: the filter runner reads only the former, the analysis reads both.
"""

from __future__ import annotations

from pathlib import Path

from crop_mot.types import Scan, ScanLabels


def write_detections(path: Path, scans: list[Scan]) -> None:
    """Write all scans to detections.jsonl - the filter's only input.

    One JSON object per scan, holding k, t, the reported pose, and the list of z vectors.
    Deliberately contains no origin information of any kind.

    Serves: [B1] run-folder output; [B2/B4] the shared input that makes filter comparison fair.

    Args:
        path: destination detections.jsonl.
        scans: the scans to write, in increasing k.
    """
    raise NotImplementedError


def read_detections(path: Path) -> list[Scan]:
    """Read detections.jsonl back into Scan objects.

    Serves: [B2] the filter runner; [B4] every baseline, reading the identical file.

    Args:
        path: source detections.jsonl.

    Returns:
        The scans in file order.
    """
    raise NotImplementedError


def write_labels(path: Path, labels: list[ScanLabels]) -> None:
    """Write the truth-side detection origins to labels.jsonl. EVALUATION ONLY. [B1/B3]

    Kept in a separate file from detections.jsonl on purpose. A filter is handed a path to
    detections.jsonl and nothing else, so it cannot reach this data even by mistake.

    Args:
        path: destination labels.jsonl.
        labels: per-scan origin records, in increasing k.
    """
    raise NotImplementedError


def read_labels(path: Path) -> list[ScanLabels]:
    """Read labels.jsonl back. EVALUATION ONLY. [B3]

    Serves: [B3] building the ScanEvent sequence, which needs to know how many of the gated
    detections were clutter.

    Args:
        path: source labels.jsonl.

    Returns:
        The label records in file order.
    """
    raise NotImplementedError
