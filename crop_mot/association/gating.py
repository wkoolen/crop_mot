"""Ellipsoidal gating: which measurements could plausibly belong to a track. [B2/B4]

Gating is an approximation, not part of the exact Bayes recursion. It discards
measurements whose squared Mahalanobis distance exceeds a threshold, which is what makes
JPDA and PMBM tractable - but it also means a gated-out measurement contributes exactly
zero instead of a very small weight.

That matters for B3: the closed form must be derived under the SAME gating convention the
filter uses, or the two will disagree by a small amount that is easy to mistake for a bug.
`ScanEvent.n_gated` records how many measurements survived the gate for this reason.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def chi2_threshold(gate_prob: float, dim_z: int) -> float:
    """Squared-Mahalanobis gate threshold for a given gate probability.

    The inverse chi-square CDF with dim_z degrees of freedom. Computed once at filter
    construction rather than per scan, both for speed and so that every filter in a run
    provably uses the same number.

    Serves: [B2] the Bernoulli gate; [B4] every multi-target filter.

    Args:
        gate_prob: probability that a true measurement falls inside the gate, e.g. 0.99.
        dim_z: measurement dimension.

    Returns:
        The threshold on d^2 = (z - z_hat)' S^-1 (z - z_hat). Under the correct-association
        hypothesis d^2 is chi-square with dim_z degrees of freedom, so gating is
        thresholding the log-evidence [A1 §Normaliser — and what it becomes one level up].
    """
    return float(chi2.ppf(gate_prob, dim_z))


def mahalanobis_sq(z: np.ndarray, z_hat: np.ndarray, S: np.ndarray) -> float:
    """Squared Mahalanobis distance of a measurement from its prediction.

    Serves: [B2/B4] gating, and the cost matrix entries used by GNN and JPDA.

    Args:
        z: shape (dim_z,), the measurement.
        z_hat: shape (dim_z,), the predicted measurement.
        S: shape (dim_z, dim_z), the innovation covariance.

    Returns:
        d^2 >= 0.
    """
    d = z - z_hat
    return float(d @ np.linalg.solve(S, d))


def without_weed_labels(indices: np.ndarray, detections) -> np.ndarray:
    """The gated indices whose detection is not labelled "weed". [B4, roadmap step 8b]

    What a plant track keeps when the filter assumes the perfect classifier (decision
    D22): a weed-labelled detection is never associated with it, even inside its gate.
    Shared by the filter and the analysis that rebuilds its gates, so both drop the same.

    Args:
        indices: gated detection indices, increasing.
        detections: the scan's detections, which `indices` point into.

    Returns:
        The indices that remain, still increasing.
    """
    return np.array([i for i in indices if detections[i].label != "weed"], dtype=int)


def gate_measurements(
    measurements: np.ndarray, z_hat: np.ndarray, S: np.ndarray, threshold: float
) -> np.ndarray:
    """Indices of the measurements that fall inside the gate.

    Serves: [B2] restricting the Bernoulli update to plausible detections; [B3] counting
    `n_gated` for the analytic cross-check; [B4] building sparse cost matrices.

    Args:
        measurements: shape (n_meas, dim_z), this scan's measurements.
        z_hat: shape (dim_z,), the predicted measurement for one track.
        S: shape (dim_z, dim_z), innovation covariance.
        threshold: from `chi2_threshold`.

    Returns:
        Shape (n_gated,) integer array of indices into `measurements`, in increasing order.
    """
    inside = [i for i, z in enumerate(measurements) if mahalanobis_sq(z, z_hat, S) <= threshold]
    return np.array(inside, dtype=int)
