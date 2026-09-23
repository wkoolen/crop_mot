"""The ONE filter interface that Bernoulli, PDA, JPDA, GNN, PMB and PMBM all satisfy.

This is the most important file in the repository. Settling it before any filter exists is
what makes B4 additive: adding JPDA or PMBM in phase 2 is a new file plus one line in
`crop_mot.filters.FILTERS`, with no change to the runner, the config loader, the detection
format or the plots.

Two things genuinely differ between these six methods, and the interface absorbs both
without a single branch in the runner:

1. SINGLE VS MULTI TARGET. `extract` always returns `list[TrackEstimate]`. Bernoulli and
   PDA return a list of length <= 1. That is the N = 1 case of the same type, not a special
   case, so the runner writes the same estimates log either way.

2. HYPOTHESES. The filter's state type S is opaque to the runner. A Bernoulli density for
   B2, a Poisson intensity plus Bernoulli components for PMB, a global hypothesis tree for
   PMBM - all of it lives inside S and never crosses this interface. `extract` is the point
   where a filter marginalises its internal structure down to per-track (r, mean, cov).
   Separating `extract` from `update`, rather than having one `step()`, is exactly what
   makes that possible - and it mirrors how the MOT literature separates the Bayes recursion
   from the estimator.

Everything else a filter needs - the motion and measurement models, p_D, lambda_FA, c(z),
the birth model, the gate - is injected at construction. That is why `update(state, scan)`
can have an identical signature for all six.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from crop_mot.types import Scan, TrackEstimate

S = TypeVar("S")


class TrackingFilter(Protocol[S]):
    """The single interface every tracking method implements. [B2/B4]

    The runner is identical for all of them, with no branches:

        state = flt.initial_state()
        for scan in scans:
            state = flt.predict(state, dt)
            state = flt.update(state, scan)
            write_estimates(scan.k, flt.extract(state))

    Implementations should be PURE: `predict` and `update` return a new state rather than
    mutating the one they are given. That is not stylistic - it is what lets a test run a
    single prediction in isolation, and what would let a future experiment rewind to scan k
    without re-running from 0.

    Attributes:
        name: the key this filter is registered under in FILTERS; also the suffix of its
            estimates log, e.g. estimates_bernoulli.jsonl.
    """

    name: str

    def initial_state(self) -> S:
        """The prior, before any scan has been seen.

        For B2 this is a Bernoulli with r = 0 (nothing exists yet) or the configured birth
        prior, depending on the birth model.

        Returns:
            A fresh filter state.
        """
        raise NotImplementedError

    def predict(self, state: S, dt: float) -> S:
        """Time update: propagate the state forward by dt seconds.

        Covers both the kinematic prediction (via the injected MotionModel) and the
        existence prediction (via p_S). For static plants with p_S = 1 this is close to the
        identity, which makes it a good place to assert purity in a test.

        Args:
            state: the current filter state.
            dt: time step in seconds, derived by the runner from consecutive scan timestamps.

        Returns:
            A new predicted state. The input must not be modified.
        """
        raise NotImplementedError

    def update(self, state: S, scan: Scan) -> S:
        """Measurement update for one scan, including births.

        The scan carries the REPORTED pose, which the filter uses to evaluate p_D and the
        clutter density. It carries no ground truth of any kind.

        Args:
            state: the predicted filter state.
            scan: this scan's detections and reported pose.

        Returns:
            A new posterior state. The input must not be modified.
        """
        raise NotImplementedError

    def extract(self, state: S) -> list[TrackEstimate]:
        """Marginalise the posterior down to reported tracks.

        This is where a filter's internal representation collapses: JPDA marginalises over
        association events, PMBM over global hypotheses, GNN reports its hard assignment
        with r in {0, 1}. The runner sees only the result.

        Args:
            state: the current filter state.

        Returns:
            Zero or more track estimates. Length <= 1 for Bernoulli and PDA; arbitrary for
            JPDA, GNN, PMB and PMBM.
        """
        raise NotImplementedError


class HasDiagnostics(Protocol[S]):
    """OPTIONAL extra output for debugging and thesis figures. [B4]

    Deliberately NOT part of TrackingFilter: forcing every B4 filter to implement it would
    make the core contract bigger than it needs to be. The runner checks for it with
    hasattr and writes the result alongside the estimates when present.

    Useful for things like PDA association weights, the number of global hypotheses PMBM is
    carrying, or the size of the gate - quantities that explain a plot but are not part of
    the estimate.
    """

    def diagnostics(self, state: S) -> dict[str, float]:
        """Scalar diagnostics for the current state.

        Args:
            state: the current filter state.

        Returns:
            A flat mapping of name to scalar, written to the estimates log as an extra field.
        """
        raise NotImplementedError


class BirthModel(Protocol):
    """Where new tracks come from. [B2/B4]

    For B2's phantom experiment this seeds a single Bernoulli component from a CLUTTER
    detection, so that r should then decay - that decay is the plot B2 has to produce.
    For PMB and PMBM the same interface supplies the birth intensity components.
    """

    def birth_components(self, scan: Scan) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Components born from this scan.

        Args:
            scan: the current scan, so a measurement-driven birth model can place a
                component at a detection.

        Returns:
            A list of (weight, mean, cov) triples. The weight is the birth existence
            probability r_b for a Bernoulli filter, or the component weight for a Poisson
            birth intensity in PMB/PMBM. Returning an empty list means nothing is born.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SurvivalModel:
    """Probability that an existing target survives to the next scan. [B2]

    ASSUMPTION: plants are permanent, so p_S = 1.0. With p_S = 1 the only thing that can
    drive r downward is repeated misdetection, which is precisely what the B2 phantom plot
    is meant to show - and it means any decay in the plot is attributable to the detector
    model rather than to an assumed death process.

    Attributes:
        p_S: survival probability in [0, 1].
    """

    p_S: float = 1.0
