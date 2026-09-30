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

import math
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
        likelihood_ratios: one entry per gated detection, in gate order:
            ell_i / kappa_i = p_D N(z_i; z_hat, S) / (lambda_FA c(z_i)), evaluated with the
            filter's ASSUMED sensor, at the filter's predicted moments and the scan's
            reported pose (decision D25). Dimensionless. math.inf where c(z_i) = 0 and
            ell_i > 0; 0.0 where p_D = 0. Empty when nothing gated.
        born: True at exactly one scan of a track's sequence, the scan it is born on: the
            first scan the track is reported in the estimates log (decision D25).

    Raises:
        ValueError: if len(likelihood_ratios) != n_gated.
    """

    k: int
    dt: float
    in_fov: bool
    p_D: float
    lambda_FA: float
    n_gated: int
    n_clutter_gated: int
    likelihood_ratios: tuple[float, ...] = ()
    born: bool = False

    def __post_init__(self) -> None:
        if len(self.likelihood_ratios) != self.n_gated:
            raise ValueError(f"scan {self.k}: {len(self.likelihood_ratios)} likelihood ratios "
                             f"for {self.n_gated} gated detections")


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

        The branches below come from A2 (and from A0 for two or more detections), never from
        `filters/bernoulli.py`. That independence is what makes B3 a cross-check.

        Order within one scan k (decision D3):
          1. PREDICTION: r_pred = p_S * r_{k-1}. p_S is per scan, so dt does not enter;
             with p_S = 1, r_pred = r_{k-1}. A2 has no prediction section: this is the
             Bernoulli prediction without a birth term.
          2. UPDATE of r_pred: exactly one of the four cases below.
          3. BIRTH: at the birth scan only, and only into r = 0, r_k = r_birth [A2 §4]
             (the configured constant standing in for A2 §4's e / (e + lambda_FA(z))). The
             seeding detection does not update the new component that scan, so r at the
             birth scan is exactly r_birth. Before the birth scan, r_k = 0.
        There is no deletion. Compare a pruned bank track on its unpruned log (D14).

        Notation, per scan:
          p_D      ScanEvent.p_D: the filter's assumed p_D at the predicted mean, the
                   plug-in for p_D_bar = integral p_D(x) p(x) dx [A2 §2.1]. 0 out of view.
          kappa_i  lambda_FA(z_i) = ScanEvent.lambda_FA * c(z_i), with c(z) = 1 / FOV area
                   (D5): the clutter intensity at gated detection z_i [A0 §Measurement model].
          ell_i    ell(z_i) = integral p_D(x) g(z_i|x) p(x) dx [A2 §3.1]; for the Gaussian
                   prediction, p_D * N(z_i; z_hat, S). A likelihood, per m^2 like kappa_i,
                   so ell_i / kappa_i is dimensionless.

        IN VIEW, NO GATED DETECTION - misdetection [A2 §2]:
            r_k = r_pred (1 - p_D) / (1 - r_pred p_D)

        IN VIEW, ONE GATED DETECTION - r_marg of [A2 §3.1], the average over its two rows:
            detected: weight r_pred ell_1,              r = 1
            missed:   weight (1 - r_pred p_D) kappa_1,  r = the misdetection result above
        which gives
            r_k = (r_pred ell_1 + r_pred (1 - p_D) kappa_1)
                  / (r_pred ell_1 + (1 - r_pred p_D) kappa_1)

        IN VIEW, TWO OR MORE GATED DETECTIONS - A0 §Measurement model inside A2 §3.1's
        two-branch existence; not yet its own derived section (decision D2; A2 §7 lists PDA
        as open). Integrating A0's bracket (1 - p_D) + sum_i p_D g(z_i|x) / kappa_i over
        p(x) gives
            L   = (1 - p_D) + sum_i ell_i / kappa_i
            r_k = r_pred L / ((1 - r_pred) + r_pred L)
        A detection outside the gate counts as ell_i = 0.

        OUT OF VIEW (in_fov is False): r_k = r_pred. The object was not looked at, so this
        is not a misdetection; it is what every in-view formula gives at p_D = 0, where all
        ell_i = 0.

        All four cases are one statement in odds: r_k / (1 - r_k) = L * r_pred / (1 - r_pred),
        with L = 1 - p_D for a miss, (1 - p_D) + ell_1 / kappa_1 for one detection, the sum
        above for several, and 1 out of view.

        Checks the recursion must pass, all from A2:
          * r_pred = 1 gives r_k = 1 in every case: certainty is absorbing [A2 §2, §5].
          * r_pred = 0.9, p_D = 0.9, a miss: r_k = 0.09 / 0.19 = 0.47 [A2 §2, numerically].
          * ell_1 -> 0 turns the one-detection case into the misdetection case.
          * kappa_1 -> 0 with a detection gives r_k = 1; r_k < 1 needs lambda_FA > 0
            [A2 §3.1].
          * The one-detection case is the n = 1 case of the several-detection form, and
            the misdetection case is its n = 0 case.

        ScanEvent fields:
          read:      k (to check the order), in_fov, p_D, n_gated, likelihood_ratios
                     (ell_i / kappa_i, so lambda_FA and c(z_i) enter through them), born.
                     The last two were added for this recursion (decision D25).
          not read:  dt (p_S is per scan); lambda_FA (inside the ratios);
                     n_clutter_gated (interpretation only).

        Args:
            events: the per-scan event sequence, in increasing k.

        Returns:
            Shape (K,) array of r values in [0, 1].

        Raises:
            ValueError: if events[k].k != k, or if a birth falls on a scan where r is not 0.
        """
        r_out = np.zeros(len(events))
        r = 0.0
        for position, event in enumerate(events):
            if event.k != position:
                raise ValueError(f"event at position {position} has k = {event.k}; "
                                 "the sequence must hold every scan from 0 in order")
            p_D = event.p_D

            # 1. PREDICTION: the Bernoulli prediction without a birth term.
            r_pred = self.p_S * r

            # 2. UPDATE of r_pred: exactly one of the four cases.
            if not event.in_fov:
                # OUT OF VIEW: not looked at, so not a misdetection.
                r = r_pred
            elif event.n_gated == 0:
                # IN VIEW, NO GATED DETECTION: misdetection [A2 §2].
                r = r_pred * (1.0 - p_D) / (1.0 - r_pred * p_D)
            elif math.inf in event.likelihood_ratios:
                # kappa_i -> 0 at a gated detection gives r_k = 1; r_k < 1 needs
                # lambda_FA > 0 [A2 §3.1].
                r = 1.0 if r_pred > 0.0 else 0.0
            elif event.n_gated == 1:
                # IN VIEW, ONE GATED DETECTION: r_marg [A2 §3.1], the detected row
                # (weight r_pred ell_1, r = 1) against the missed row (weight
                # (1 - r_pred p_D) kappa_1, r = the misdetection result). Numerator and
                # denominator divided by kappa_1 > 0, so ell_1 / kappa_1 is the ratio.
                ratio = event.likelihood_ratios[0]
                r = ((r_pred * ratio + r_pred * (1.0 - p_D))
                     / (r_pred * ratio + (1.0 - r_pred * p_D)))
            else:
                # IN VIEW, TWO OR MORE GATED DETECTIONS: A0 §Measurement model inside
                # A2 §3.1's two-branch existence (decision D2).
                L = (1.0 - p_D) + sum(event.likelihood_ratios)
                r = r_pred * L / ((1.0 - r_pred) + r_pred * L)

            # 3. BIRTH: at the birth scan only, into r = 0 [A2 §4], decision D3.
            if event.born:
                if r != 0.0:
                    raise ValueError(f"birth at scan {event.k} into r = {r}; a birth only "
                                     "goes into an empty Bernoulli (decision D3)")
                r = self.r_birth

            r_out[position] = r
        return r_out


# Registry of available references, keyed by the config's `b3_reference` field.
# A second reference is a new class plus one line here.
REFERENCES: dict[str, type[AnalyticReference]] = {
    "bernoulli_existence": BernoulliExistenceReference,
}
