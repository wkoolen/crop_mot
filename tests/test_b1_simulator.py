"""B1: does the simulator generate what the model says it should? [B1]

These tests are the reason to trust every plot downstream. If the detector does not
actually miss with probability 1 - p_D, then B2's r-decay curve is measuring something
other than what the thesis claims.
"""

from __future__ import annotations

import pytest

from crop_mot.config import ScenarioConfig
from crop_mot.runner.run_dir import RunDir


def test_same_seed_reproduces_detections_exactly(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """Two simulations with the same seed produce byte-identical detections.jsonl. [B1]

    The foundation of every comparison in the thesis: if this fails, no two filter runs are
    comparable and no result is reproducible. Compares bytes rather than parsed values, so
    that a change in float formatting is caught too.
    """
    raise NotImplementedError


def test_changing_clutter_rate_does_not_move_the_plants(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """lambda_FA changes the clutter, not the field. [B1, reproducibility]

    This is what the named RNG substreams in `crop_mot.rng` buy. Without them, drawing more
    clutter would consume more random numbers and shift every later draw, so a parameter
    sweep over lambda_FA would silently be a sweep over different fields as well.
    """
    raise NotImplementedError


def test_no_detection_outside_the_fov(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """Every detection, real or clutter, lies inside the FOV wedge at its scan's pose. [B1]

    Catches sign errors in the body-frame transform, which otherwise show up much later as
    an inexplicably bad tracking result.
    """
    raise NotImplementedError


def test_empirical_detection_rate_matches_p_D(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """Over many scans, detected/visible converges to p_D. [B1]

    A statistical test, so it needs a tolerance and enough samples. Use the labels to count
    visible and detected plants; the point is to catch a detector that is systematically
    over- or under-detecting, not to verify the binomial distribution.
    """
    raise NotImplementedError


def test_clutter_count_matches_poisson_lambda(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """The number of clutter detections per scan has mean lambda_FA. [B1]

    Check the mean, and ideally that the variance is also near lambda_FA - a Poisson
    distribution has both equal, so a mismatch reveals the wrong distribution rather than
    just the wrong parameter.
    """
    raise NotImplementedError


def test_pose_known_means_reported_equals_true(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """With pose_known: true, Scan.pose is exactly the true pose. [B1]

    Guards the extension slot: if the gait wobble is ever accidentally applied while
    pose_known is true, B2 and B3 would be running under a different assumption than the
    A2 derivation, and the cross-check would fail for a reason nobody would guess.
    """
    raise NotImplementedError


@pytest.mark.parametrize("profile", ["constant", "range_dependent"])
def test_p_D_is_zero_outside_the_fov(profile: str) -> None:
    """Both p_D profiles return exactly 0.0 outside the FOV. [B1/B3]

    Not a rounding matter: the Bernoulli update treats "out of view" differently from "in
    view but missed", and B3's ScanEvent records `in_fov` precisely so the closed form can
    too. A small non-zero p_D outside the wedge would blur that distinction.
    """
    raise NotImplementedError
