"""Single best assignment: the GNN baseline's core. [B4]

GNN is the baseline the thesis compares the probabilistic methods against. It commits to
ONE association per scan - the one minimising total cost - and then runs a plain Kalman
update as if that association were certain. Its failure mode is exactly what PDA/JPDA/PMBM
are designed to avoid: one wrong hard commitment is never revisited.

Because it satisfies the same TrackingFilter interface and reads the same detections.jsonl,
the comparison against it is fair by construction.
"""

from __future__ import annotations

import numpy as np


def build_cost_matrix(
    log_likelihoods: np.ndarray, log_missed: np.ndarray, log_clutter: np.ndarray
) -> np.ndarray:
    """Assemble the (tracks x (measurements + dummies)) cost matrix.

    Costs are NEGATIVE log likelihoods, so that minimising total cost maximises the joint
    association likelihood. The dummy columns represent "this track was not detected" and
    "this measurement is clutter", which is what lets the assignment leave a track or a
    measurement unassigned instead of forcing a bad pairing.

    Serves: [B4] GNN, and the per-hypothesis cost matrices JPDA and PMBM feed to Murty.

    Args:
        log_likelihoods: shape (n_tracks, n_meas), log of p_D * g(z|x) per pair. Entries
            outside the gate should be -inf.
        log_missed: shape (n_tracks,), log(1 - p_D) per track.
        log_clutter: shape (n_meas,), log of lambda_FA * c(z) per measurement.

    Returns:
        Shape (n_tracks, n_meas + n_tracks) cost matrix ready for linear_sum_assignment.
    """
    raise NotImplementedError


def best_assignment(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum-cost assignment via scipy.optimize.linear_sum_assignment.

    A thin wrapper, so that the -inf / large-finite-cost convention is handled in one place:
    linear_sum_assignment rejects matrices containing infinities, so gated-out pairs must be
    replaced by a large finite cost rather than -inf before it is called. Getting that wrong
    produces an exception on some scans and not others, which is a confusing bug to chase.

    Serves: [B4] GNN.

    Args:
        cost: shape (n_rows, n_cols) cost matrix from `build_cost_matrix`.

    Returns:
        Tuple (assignment, total_cost) where assignment has shape (n_rows,) giving the
        chosen column per row.
    """
    raise NotImplementedError
