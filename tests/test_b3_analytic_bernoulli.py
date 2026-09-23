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

import pytest

from crop_mot.config import RunConfig


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
    """
    raise NotImplementedError
