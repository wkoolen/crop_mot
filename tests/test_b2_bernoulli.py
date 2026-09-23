"""B2: does the Bernoulli filter behave like a Bernoulli filter? [B2]

Properties rather than fixed numbers. The exact r trajectory is B3's job, checked against
the closed form; these tests catch the coarser failures that would make B3's comparison
meaningless in the first place.
"""

from __future__ import annotations

from crop_mot.config import RunConfig


def test_r_stays_in_unit_interval(tiny_run_config: RunConfig, tmp_path) -> None:
    """r is a probability at every scan. [B2]

    The cheapest possible check, and it catches the most common Bernoulli bug: a missing
    normalisation in the update, which typically pushes r above 1 only on scans with
    several gated detections.
    """
    raise NotImplementedError


def test_r_decays_under_sustained_misdetection(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """With no detection ever assigned, r decreases monotonically. [B2]

    This is the phantom-track behaviour the B2 plot has to show. With p_S = 1 the
    prediction leaves r untouched, so any decay must come from the misdetection branch -
    which means a flat r here points at the update, not at the survival model.
    """
    raise NotImplementedError


def test_predict_and_update_are_pure(tiny_run_config: RunConfig) -> None:
    """Neither predict nor update mutates the state it was given. [B2/B4]

    Part of the TrackingFilter contract. Purity is what lets the other tests run a single
    prediction in isolation, and what would let a future experiment rewind to scan k
    without re-running from 0.
    """
    raise NotImplementedError


def test_covariance_stays_symmetric_positive_definite(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """P stays symmetric and positive definite over a full run. [B2]

    With a static target and p_S = 1, P shrinks monotonically, so accumulated floating-point
    asymmetry eventually makes it indefinite. That failure surfaces much later as a Cholesky
    error inside a likelihood evaluation, far from its cause - hence the Joseph-form note in
    `crop_mot.filters.kalman.kf_update`.
    """
    raise NotImplementedError


def test_extract_returns_at_most_one_track(tiny_run_config: RunConfig, tmp_path) -> None:
    """A Bernoulli filter is single-target: extract returns a list of length <= 1. [B2]

    Documents the N = 1 case of the shared interface. The runner does not special-case it,
    and neither should the filter.
    """
    raise NotImplementedError
