"""B3: the analytic cross-check. "Check r vs A2" on the roadmap. [B3]

This file is the point of work package B3. It answers two different questions, and keeping
them apart is what makes the validation worth anything:

  1. `test_r_matches_analytic_recursion` - does the IMPLEMENTATION match the DERIVATION?
     Both compute the same recursion from the same event sequence, so they should agree to
     near machine precision. A disagreement is a bug in one of them.

  2. `test_monte_carlo_mean_r_brackets_analytic` - does the DERIVATION match the
     SIMULATOR? Averaged over many realisations, the mean r should sit inside a few
     standard errors of the closed form. A disagreement here means the modelling
     assumptions are wrong, and test 1 would happily pass while both were wrong together.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from crop_mot.analysis.analytic import BernoulliExistenceReference, ScanEvent
from crop_mot.analysis.metrics import R_TOLERANCE
from crop_mot.config import RunConfig


@pytest.mark.xfail(strict=True, reason="waits on step 3: build_scan_events")
@pytest.mark.parametrize("p_D_profile", ["constant", "range_dependent"])
def test_r_matches_analytic_recursion(
    tiny_run_config: RunConfig, p_D_profile: str, tmp_path
) -> None:
    """THE B3 CROSS-CHECK: the filter's r trajectory equals the A2 closed form. [B3]

    Procedure:
      1. Simulate a scenario with the given p_D profile and run the Bernoulli filter.
      2. Build the ScanEvent sequence with `build_scan_events`, using the FILTER's assumed
         p_D and lambda_FA - not the simulator's, since this test is about the
         implementation, not the model.
      3. Evaluate `BernoulliExistenceReference.r_sequence` over those events.
      4. Assert `compare_r(...).max_abs_error` is below a near-machine-precision tolerance.

    Parametrised over both p_D profiles. The range-dependent case is only checkable because
    `ScanEvent` carries p_D PER SCAN rather than the reference storing a constant - with a
    stored constant this parametrisation would need a second derivation.

    On failure, `RComparison.first_divergence_k` says which scan diverged first; read off
    what happened at that scan from the event sequence to find which branch of the
    recursion is wrong.
    """
    raise NotImplementedError


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="waits on step 4: reworked into the per-seed "
                   "cross-check (D17)")
def test_monte_carlo_mean_r_brackets_analytic(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """The mean r over many seeds lies within a few standard errors of the closed form. [B3]

    The empirical half of B3: the check that the modelling assumptions - not just the
    arithmetic - are right.

    Compare against the standard ERROR of the mean (r_std / sqrt(n_runs)), not the standard
    deviation. The latter measures how much individual runs differ from each other, which
    for a phantom track is large, and would give a band wide enough to hide a real error.

    Marked slow: it re-simulates and re-filters the whole scenario n_runs times. Deselect
    with `-m "not slow"` during ordinary development.
    """
    raise NotImplementedError


def test_constant_profile_is_the_special_case_of_the_general_recursion() -> None:
    """A constant-p_D event sequence reproduces the textbook constant-p_D form. [B3]

    Guards the design decision that makes one reference cover both profiles: build an event
    sequence where every ScanEvent holds the SAME p_D and lambda_FA, and check that
    `r_sequence` matches the closed-form constant-p_D expression evaluated directly.

    If this ever fails while `test_r_matches_analytic_recursion` passes, the filter and the
    reference have drifted together - which is exactly the failure mode a single
    implementation covering two profiles is vulnerable to.

    The textbook form: after a birth at r_b and n in-view misses at constant p_D, the odds
    are r_b / (1 - r_b) (1 - p_D)^n [A2 §2.1], so
        r = r_b (1 - p_D)^n / (1 - r_b + r_b (1 - p_D)^n).
    Out-of-view scans in between leave n unchanged. A2's own number is checked too:
    r = 0.9, p_D = 0.9 and one miss give 0.09 / 0.19 [A2 §2].
    """
    p_D, lambda_FA, r_b = 0.85, 2.0, 0.3
    in_view = [False, False, True, True, False, True, True, True, False, True]
    events = [ScanEvent(k=k, dt=0.25, in_fov=seen and k > 1, p_D=p_D if seen and k > 1 else 0.0,
                        lambda_FA=lambda_FA, n_gated=0, n_clutter_gated=0, born=k == 1)
              for k, seen in enumerate(in_view)]

    r = BernoulliExistenceReference(p_S=1.0, r_birth=r_b).r_sequence(events)

    n_misses = np.cumsum([event.in_fov for event in events])
    decay = (1.0 - p_D) ** n_misses
    expected = r_b * decay / (1.0 - r_b + r_b * decay)
    expected[0] = 0.0
    np.testing.assert_allclose(r, expected, rtol=0.0, atol=R_TOLERANCE)

    one_miss = [ScanEvent(k=0, dt=0.0, in_fov=False, p_D=0.0, lambda_FA=lambda_FA,
                          n_gated=0, n_clutter_gated=0, born=True),
                ScanEvent(k=1, dt=0.25, in_fov=True, p_D=0.9, lambda_FA=lambda_FA,
                          n_gated=0, n_clutter_gated=0)]
    r = BernoulliExistenceReference(p_S=1.0, r_birth=0.9).r_sequence(one_miss)
    assert r[1] == pytest.approx(0.09 / 0.19, abs=R_TOLERANCE)


def test_one_detection_case_is_the_several_detection_form_with_n_equal_one() -> None:
    """The docstring's own consistency checks on the detection branches. [B3]

    One gated detection agrees with the two-or-more form at n = 1, a vanishing ratio turns
    it into a miss, and an infinite ratio (kappa = 0) gives r = 1 [A2 §3.1].
    """
    reference = BernoulliExistenceReference(p_S=1.0, r_birth=0.4)
    p_D, ratio = 0.7, 2.5

    def r_after(*ratios: float) -> float:
        events = [ScanEvent(k=0, dt=0.0, in_fov=False, p_D=0.0, lambda_FA=2.0, n_gated=0,
                            n_clutter_gated=0, born=True),
                  ScanEvent(k=1, dt=0.25, in_fov=True, p_D=p_D, lambda_FA=2.0,
                            n_gated=len(ratios), n_clutter_gated=0,
                            likelihood_ratios=tuple(ratios))]
        return float(reference.r_sequence(events)[1])

    L = (1.0 - p_D) + ratio
    assert r_after(ratio) == pytest.approx(0.4 * L / (0.6 + 0.4 * L), abs=R_TOLERANCE)
    assert r_after(0.0) == pytest.approx(r_after(), abs=R_TOLERANCE)
    assert r_after(ratio, 0.0) == pytest.approx(r_after(ratio), abs=R_TOLERANCE)
    assert r_after(math.inf) == 1.0


def test_scan_event_needs_one_likelihood_ratio_per_gated_detection() -> None:
    """The invariant len(likelihood_ratios) == n_gated (D25). [B3]"""
    with pytest.raises(ValueError, match="likelihood ratios"):
        ScanEvent(k=3, dt=0.25, in_fov=True, p_D=0.85, lambda_FA=2.0, n_gated=2,
                  n_clutter_gated=0, likelihood_ratios=(1.0,))
