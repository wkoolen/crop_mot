"""Reading and writing estimates_<filter>.jsonl. [B2/B3/B4]

One record per scan, holding every track the filter reported at that scan. A scan with no
tracks still gets a record with an empty list, so that "the filter reported nothing" and
"the filter did not run this scan" are distinguishable - which matters when a plot shows a
gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crop_mot.io import read_jsonl, write_jsonl
from crop_mot.types import TrackEstimate


@dataclass(frozen=True)
class ScanEstimates:
    """One scan's worth of filter output. [B2/B4]

    Attributes:
        k: scan index.
        estimates: the tracks reported at this scan; possibly empty.
        diagnostics: optional extra scalars from a filter implementing HasDiagnostics.
    """

    k: int
    estimates: tuple[TrackEstimate, ...]
    diagnostics: dict[str, float] | None = None


def write_estimates(path: Path, records: list[ScanEstimates]) -> None:
    """Write a filter's per-scan output to estimates_<filter>.jsonl.

    Serves: [B2] the Bernoulli run; [B4] every filter, into the same run folder.

    Args:
        path: destination file, from RunDir.estimates(filter_name).
        records: one entry per scan, in increasing k.
    """
    lines = []
    for record in records:
        estimates = []
        for estimate in record.estimates:
            estimates.append({
                "track_id": estimate.track_id,
                "r": estimate.r,
                "mean": estimate.mean,
                "cov": estimate.cov,
            })
        lines.append({"k": record.k, "estimates": estimates, "diagnostics": record.diagnostics})
    write_jsonl(path, lines)


def read_estimates(path: Path) -> list[ScanEstimates]:
    """Read an estimates log back.

    Serves: [B3] the analytic cross-check and the plots.

    Args:
        path: source estimates_<filter>.jsonl.

    Returns:
        The records in file order.
    """
    records = []
    for line in read_jsonl(path):
        estimates = tuple(
            TrackEstimate(
                track_id=estimate["track_id"],
                r=estimate["r"],
                mean=np.asarray(estimate["mean"], dtype=float),
                cov=np.asarray(estimate["cov"], dtype=float),
            )
            for estimate in line["estimates"]
        )
        records.append(ScanEstimates(k=line["k"], estimates=estimates,
                                     diagnostics=line.get("diagnostics")))
    return records


def r_trajectory(records: list[ScanEstimates], track_id: int) -> np.ndarray:
    """Extract the existence probability of one track over all scans. [B2/B3]

    This is the series the B2 plot shows decaying for a phantom, and the series B3 compares
    against the closed form.

    Scans where the track was not reported yield 0.0 rather than NaN, because a Bernoulli
    filter that drops below its reporting threshold has not stopped having an r - it has an
    r the extractor chose not to report. Using NaN here would make the decay curve break
    exactly where it gets interesting.

    Args:
        records: the estimates log.
        track_id: which track to follow.

    Returns:
        Shape (n_scans,) array of r values, in scan order.
    """
    raise NotImplementedError
