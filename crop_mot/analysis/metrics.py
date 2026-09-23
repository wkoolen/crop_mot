"""Comparison metrics: B3's r cross-check and B4's multi-target distance. [B3/B4]"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from crop_mot.types import TrackEstimate


@dataclass(frozen=True)
class RComparison:
    """Result of comparing a simulated r trajectory against the closed form. [B3]

    `first_divergence_k` is the field that actually helps when the test fails. A max error
    of 0.3 tells you something is wrong; knowing it first exceeded tolerance at k = 7 tells
    you which branch of the recursion to look at, because you can read off from the event
    sequence what happened at scan 7.

    Attributes:
        max_abs_error: max over k of |r_sim - r_ref|.
        rms_error: root mean square error over all scans.
        first_divergence_k: the first scan index where the absolute error exceeded the
            tolerance, or None if it never did.
    """

    max_abs_error: float
    rms_error: float
    first_divergence_k: int | None


def compare_r(r_sim: np.ndarray, r_ref: np.ndarray, tol: float = 1e-9) -> RComparison:
    """Compare the filter's r trajectory against the analytic one. [B3]

    The tolerance defaults to near machine precision on purpose: this is not a statistical
    comparison. The filter and the closed form are computing the SAME recursion from the
    same event sequence, so anything beyond floating-point noise is an implementation
    disagreement, not sampling error. The statistical question belongs to
    `montecarlo.run_monte_carlo`.

    Args:
        r_sim: shape (K,), the filter's existence probability per scan.
        r_ref: shape (K,), the closed form's, from AnalyticReference.r_sequence.
        tol: absolute tolerance used for first_divergence_k.

    Returns:
        The comparison summary.

    Raises:
        ValueError: if the two arrays have different lengths - which usually means the
            event sequence and the estimates log came from different runs.
    """
    raise NotImplementedError


def gospa(
    estimates: Sequence[TrackEstimate],
    truth_positions: np.ndarray,
    c: float,
    p: float = 2.0,
    alpha: float = 2.0,
) -> float:
    """Generalised Optimal Sub-Pattern Assignment distance. STUB, phase 2. [B4]

    The standard multi-target metric, and the right one for comparing PMBM against JPDA
    against the GNN baseline, because it decomposes into localisation error, missed targets
    and false tracks. A plain RMSE cannot, since it needs a fixed correspondence.

    Not used in B1-B3, where there is one hypothesised target and the quantity of interest
    is r rather than a set distance.

    Args:
        estimates: the tracks a filter reported at one scan.
        truth_positions: shape (n_true, dim_x), true plant positions in view at that scan.
        c: cutoff distance; errors are capped at c, and c/2 is charged per cardinality error.
        p: the norm order, usually 2.
        alpha: cardinality penalty factor; 2 gives the standard decomposable form.

    Returns:
        The GOSPA distance at this scan.
    """
    raise NotImplementedError
