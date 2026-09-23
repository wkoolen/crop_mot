"""Murty's algorithm: the k best assignments, not just the best one. [B4]

This is the piece that separates GNN from JPDA and PMBM. GNN takes the single best
assignment; JPDA needs the weights of many association events, and PMBM needs to carry a
set of global hypotheses forward. Enumerating every association is factorial, so both use
the k best and truncate.

Stubbed in phase 1 so that B4 is genuinely "add a filter file". Nothing in B1-B3 calls it.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def k_best_assignments(cost: np.ndarray, k: int) -> Iterator[tuple[np.ndarray, float]]:
    """Enumerate the k lowest-cost assignments in increasing order of cost.

    Murty's partition scheme: take the best assignment, then repeatedly partition the
    solution space by forcing and forbidding individual pairs, solving a constrained
    assignment problem in each partition and keeping the cheapest unexplored one.

    Yields lazily, because a caller that prunes on a cost ratio - "stop once a hypothesis
    is 1e-4 times less likely than the best" - should not pay for hypotheses it will
    discard. That pruning rule is usually what actually bounds the cost in practice, not k.

    Serves: [B4] JPDA marginal association probabilities; PMBM global hypothesis management.

    Args:
        cost: shape (n_rows, n_cols) cost matrix; entries must be finite.
        k: maximum number of assignments to yield.

    Yields:
        Tuples (assignment, total_cost) in increasing cost order, at most k of them. Yields
        fewer if the problem admits fewer feasible assignments.
    """
    raise NotImplementedError


def normalise_hypothesis_weights(log_weights: np.ndarray) -> np.ndarray:
    """Turn unnormalised log hypothesis weights into a probability vector.

    Must use the log-sum-exp trick: with a dozen targets the raw likelihood products
    underflow to zero in double precision, and the resulting 0/0 is silent - the weights
    come back as NaN several scans later, far from the cause.

    Serves: [B4] JPDA and PMBM, wherever a mixture is collapsed or reported.

    Args:
        log_weights: shape (n_hypotheses,), unnormalised log weights.

    Returns:
        Shape (n_hypotheses,) non-negative weights summing to 1.
    """
    raise NotImplementedError
