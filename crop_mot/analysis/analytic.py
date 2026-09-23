"""The B3 hook: closed-form references to cross-check a filter against. [B3]

B3 on the roadmap is "check r vs A2" - compare the Bernoulli filter's existence probability
against the hand-derived recursion, so that the implementation is validated against the
mathematics rather than against a plot that looks plausible.

The design point that makes this work for both p_D profiles: `ScanEvent` carries p_D and
lambda_FA PER SCAN rather than the reference storing a single constant. With ConstantPD
every event holds the same value and the recursion collapses to the textbook constant-p_D
form; with RangeDependentPD the values vary as the robot approaches a plant. One
implementation validates both, with no duplicated algebra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ScanEvent:
    """What happened at scan k, in the terms the hand derivation (A2) uses. [B3]

    Built by `crop_mot.analysis.events.build_scan_events` from the recorded detections,
    labels and truth AFTER the filter has run, so the filter never sees any of this.

    p_D and lambda_FA are the FILTER's assumed values at this scan, evaluated at the
    hypothesised target location. Using the simulator's true values instead would turn B3
    from "is the implementation correct?" into "is the model correct?" - a different
    question, which the Monte-Carlo half answers.

    Attributes:
        k: scan index.
        dt: seconds since the previous scan, for the prediction branch.
        in_fov: whether the hypothesised location was inside the assumed FOV. Distinct from
            "p_D == 0" as a matter of bookkeeping: a target out of view is not a
            misdetection, and conflating the two is the most likely way for the closed form
            and the filter to disagree.
        p_D: the filter's assumed detection probability at this scan.
        lambda_FA: the filter's assumed clutter rate at this scan.
        n_gated: how many detections fell inside the gate.
        n_clutter_gated: how many of those were clutter, from labels.jsonl. Evaluation-side
            information, used to interpret a disagreement rather than to compute r.
    """

    k: int
    dt: float
    in_fov: bool
    p_D: float
    lambda_FA: float
    n_gated: int
    n_clutter_gated: int


class AnalyticReference(Protocol):
    """A closed-form trajectory to cross-check a filter against. [B3]

    A Protocol rather than a single class so that a second reference - for a different
    filter, or a different set of assumptions - is a new file. Phase 1 has exactly one
    implementation.

    Attributes:
        name: key used in the config's `b3_reference` field.
    """

    name: str

    def r_sequence(self, events: Sequence[ScanEvent]) -> np.ndarray:
        """Existence probability r_k for k = 0..K-1, from the closed form alone.

        Args:
            events: the per-scan event sequence, in increasing k.

        Returns:
            Shape (K,) array of r values in [0, 1].
        """
        raise NotImplementedError


@dataclass(frozen=True)
class BernoulliExistenceReference(AnalyticReference):
    """The hand-derived Bernoulli existence recursion - thesis item A2. [B3]

    p_D and lambda_FA are read from each ScanEvent rather than stored here, so this one
    class covers both the constant and the range-dependent profile.

    Attributes:
        p_S: assumed survival probability. 1.0 for permanent plants, in which case the
            prediction branch leaves r unchanged and every change in r comes from the
            measurement update.
        r_birth: the birth existence probability r_b the filter was configured with.
        name: "bernoulli_existence".
    """

    p_S: float
    r_birth: float
    name: str = "bernoulli_existence"

    def r_sequence(self, events: Sequence[ScanEvent]) -> np.ndarray:
        """Evaluate the closed-form existence recursion over an event sequence. [B3]

        TODO(human): document the three branches of this recursion, in terms of the
        ScanEvent fields each one reads.

        Write out, as a docstring extension below this marker:
          * the BIRTH branch     - what r is immediately after a component is born;
          * the PREDICTION branch - how p_S and dt act on r between scans;
          * the MISDETECTION branch - how r changes when the target is in the FOV but no
            gated detection is assigned to it (this is the branch that makes a phantom's r
            decay, so it is the one the B2 plot is really showing);
          * the DETECTION branch  - how r changes when a gated detection is assigned,
            including how the clutter explanation lambda_FA * c(z) competes with it;
          * and what happens when `in_fov` is False, which is NOT the same as a
            misdetection.

        Also note any ScanEvent field your derivation needs that is not already there - the
        current fields are k, dt, in_fov, p_D, lambda_FA, n_gated and n_clutter_gated.

        Args:
            events: the per-scan event sequence, in increasing k.

        Returns:
            Shape (K,) array of r values in [0, 1].
        """
        raise NotImplementedError


# Registry of available references, keyed by the config's `b3_reference` field.
# A second reference is a new class plus one line here.
REFERENCES: dict[str, type[AnalyticReference]] = {
    "bernoulli_existence": BernoulliExistenceReference,
}
