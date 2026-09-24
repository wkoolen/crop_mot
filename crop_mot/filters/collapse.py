"""Collapsing a Bernoulli's post-update mixture back to one Gaussian. [B2, extension point]

After a measurement update, the Bernoulli density GIVEN EXISTENCE is a mixture: one branch
for "the object was missed" and one per gated detection "the object produced z_i"
[A2 §3.1]. `BernoulliState` carries a single Gaussian, so the mixture has to be collapsed.
How to collapse it is an open modelling choice - A2 §3.1 leaves the moment-matching
question as a TODO, and A2 §7 leaves the non-constant p_D case open - so it is an injected
strategy rather than a line buried in the filter.

The existence probability r is NOT affected by the choice: r+ is computed from the branch
weights before collapsing. The choice does affect later scans, through where the collapsed
Gaussian sits (p_D at its mean, the predicted measurement, the gate).

Phase 1 ships one strategy, KeepBestBranch (decision D1, 2026-09-24: start simple, increase
complexity step by step). Adding another - moment matching, or anything else - is one new
class satisfying CollapseStrategy plus one line in COLLAPSE_STRATEGIES, selected by
`filter.collapse` in the run config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CollapseStrategy(Protocol):
    """Turn a weighted mixture of Gaussian branches into one Gaussian. [B2]"""

    def collapse(
        self, branches: list[tuple[float, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse the branches.

        Args:
            branches: (w, mean, cov) triples, one per branch, with the weights normalised to
                sum to 1 [A3 §Normalizing the mixture]. The first branch is the missed
                branch, the rest are the detection branches in gate order.

        Returns:
            Tuple (mean, cov) of the single Gaussian that replaces the mixture.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class KeepBestBranch(CollapseStrategy):
    """Keep the highest-weight branch's Gaussian and discard the others. [B2]

    SIMPLIFICATION: stands in for the full mixture of [A2 §3.1] (which PMBM keeps) and for
    its moment-matched collapse (the A2 §3.1 TODO). It never places the Gaussian between two
    modes - the failure A2 §3.1 warns moment matching has - but it throws away the
    probability mass of every other branch, so the covariance is overconfident whenever the
    branches disagree.

    Ties go to the earlier branch, i.e. to the missed branch before any detection branch.
    """

    def collapse(
        self, branches: list[tuple[float, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The (mean, cov) of the branch with the largest weight, as new arrays."""
        weights = [w for w, _, _ in branches]
        best = weights.index(max(weights))
        _, mean, cov = branches[best]
        return mean.copy(), cov.copy()


# Available strategies, keyed by the run config's `filter.collapse` value.
# A new strategy is a new class above plus one line here.
COLLAPSE_STRATEGIES: dict[str, Callable[[], CollapseStrategy]] = {
    "best_branch": KeepBestBranch,
}
