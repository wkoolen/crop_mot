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
from crop_mot.config import FilterConfig


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
