"""B3: the analytic cross-check. "Check r vs A2" on the roadmap. [B3]

This file is the point of work package B3. It answers two different questions, and keeping
them apart is what makes the validation worth anything:

  1. `test_r_matches_analytic_recursion` - does the IMPLEMENTATION match the DERIVATION?
     Both compute the same recursion from the same event sequence, so they should agree to
     near machine precision. A disagreement is a bug in one of them.

  2. `test_per_seed_cross_check_over_random_seeds` - does test 1 hold on every
     realisation, not just the one it was written against? The same check, repeated over
     many seeds, with the branches it covered counted (decision D17).

The other question - does the DERIVATION's model match the SIMULATOR? - is not a test
here. Comparing the mean r over seeds with "the closed form" does not hold in general: r
is nonlinear in the events, and each seed has its own event sequence. The event-rate model
check that answers it (roadmap step 4b) is future work.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.analytic import BernoulliExistenceReference, ScanEvent
from crop_mot.analysis.estimates_log import r_trajectory, read_estimates, track_lifetimes
from crop_mot.analysis.events import branch_counts, build_scan_events, predicted_track_moments
from crop_mot.analysis.crosscheck import cross_check_run
from crop_mot.analysis.metrics import R_TOLERANCE, compare_r
from crop_mot.analysis.montecarlo import cross_check_summary, run_trials
from crop_mot.config import DetectionConfig, PruneConfig, RunConfig
from crop_mot.filters import build_filter
from crop_mot.runner.analyse import analyse_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter, track_from_config
from crop_mot.sensor.record import read_detections, write_detections, write_labels
from crop_mot.types import Detection, Pose2D, Scan, ScanLabels

from conftest import CONFIGS, REPO_ROOT

# Births checked by the cross-check, as (at_scan, detection_index) in the tiny scenario
# lengthened to 20 scans. Chosen by inspecting that run so that together they reach every
# branch of the recursion; the test asserts the coverage, so a simulator change that
# loses a branch fails loudly instead of weakening the check.
#   (0, 0): clutter behind the robot's path; out of view from scan 2 on.
#   (0, 1): clutter ahead of the row; decays through misses and single gated detections.
#   (3, 1): a plant detection; confirmed through scans with several gated detections.
CROSS_CHECK_BIRTHS = ((0, 0), (0, 1), (3, 1))
BRANCHES = ("birth", "out_of_view", "miss", "one_detection", "several_detections")


def _with_profile(cfg: RunConfig, p_D_profile: str) -> RunConfig:
    """The tiny run over 20 scans, with the given p_D profile on BOTH sides (no mismatch)."""
    scenario = replace(cfg.scenario, path=replace(cfg.scenario.path, n_scans=20))
    cfg = replace(cfg, scenario=scenario)
    if p_D_profile == "constant":
        return cfg
    detection = DetectionConfig(kind="range_dependent", p_D_near=0.95, p_D_far=0.6)
    return replace(
        cfg,
        scenario=replace(scenario, sensor=replace(scenario.sensor, detection=detection)),
        filter_cfg=replace(cfg.filter_cfg, assumed_sensor=replace(
            cfg.filter_cfg.assumed_sensor, detection=detection)),
    )


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

    Runs the filter once per birth in CROSS_CHECK_BIRTHS on the same detections, and
    asserts that the checked sequences together reach every branch, so a pass is never a
    miss-only pass. Tolerance: R_TOLERANCE, absolute (D18).
    """
    cfg = _with_profile(tiny_run_config, p_D_profile)
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    simulate(cfg.scenario, run)
    times = [scan.t for scan in read_detections(run.detections)]
    reference = BernoulliExistenceReference(p_S=cfg.filter_cfg.survival.p_S,
                                            r_birth=cfg.filter_cfg.birth.r_b)

    covered = dict.fromkeys(BRANCHES, 0)
    for at_scan, detection_index in CROSS_CHECK_BIRTHS:
        filter_cfg = replace(cfg.filter_cfg, birth=replace(
            cfg.filter_cfg.birth, at_scan=at_scan, detection_index=detection_index))
        flt = build_filter(filter_cfg)
        run_filter(flt, run)
        records = read_estimates(run.estimates(flt.name))
        (track_id,) = track_lifetimes(records)

        moments = predicted_track_moments(records, track_id, times, filter_cfg)
        events = build_scan_events(run.detections, run.labels, filter_cfg, moments)
        result = compare_r(r_trajectory(records, track_id), reference.r_sequence(events))

        assert result.max_abs_error <= R_TOLERANCE, (
            f"birth {(at_scan, detection_index)}: first divergence at scan "
            f"{result.first_divergence_k}: {events[result.first_divergence_k]}")
        for branch in BRANCHES:
            covered[branch] += branch_counts(events)[branch]

    assert all(covered[branch] > 0 for branch in BRANCHES), covered


@pytest.mark.slow
def test_per_seed_cross_check_over_random_seeds(tiny_run_config: RunConfig, tmp_path) -> None:
    """On every seed, every track's r matches the A2 recursion on its own events. [B3]

    The B3 deliverable of decision D17, replacing the former "mean r brackets the closed
    form" test. Plain Monte Carlo: seed base + i, independent trials, each seed one
    outcome (it passes when all its tracks do). A bank of the three CROSS_CHECK_BIRTHS,
    pruned, so each seed checks several tracks, on the unpruned log (D14).

    The sentence it supports: "the existence recursion matched A2 to 1e-12 on X scans over
    n random trials, covering these branches".

    Marked slow: it re-simulates and re-filters the scenario n_runs times. Deselect with
    `-m "not slow"` during ordinary development.
    """
    cfg = _with_profile(tiny_run_config, "constant")
    birth = replace(cfg.filter_cfg.birth, kind="from_measurements", seeds=CROSS_CHECK_BIRTHS,
                    at_scan=CROSS_CHECK_BIRTHS[0][0],
                    detection_index=CROSS_CHECK_BIRTHS[0][1])
    cfg = replace(cfg, filter_cfg=replace(cfg.filter_cfg, kind="bernoulli_bank", birth=birth,
                                          prune=PruneConfig(r_min=1e-3)))
    n_runs = cfg.analysis.monte_carlo.n_runs

    trials = run_trials(cfg, n_runs, base_seed=1000, runs_base=tmp_path,
                        measures={"cross_check": cross_check_run})
    summary = cross_check_summary(trials)

    assert summary.divergences == (), summary.divergences
    assert summary.n_passed == n_runs - summary.n_without_track
    assert summary.n_passed >= n_runs // 2
    assert summary.max_abs_error <= R_TOLERANCE
    assert all(summary.branches[branch] > 0 for branch in BRANCHES), summary.branches


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


def test_build_scan_events_by_hand(tiny_run_config: RunConfig, tmp_path) -> None:
    """One small scan's likelihood ratio, the gate, the labels and the birth flag. [B3]

    Three scans with the robot at the origin facing +y and a track at (0, 2), born at
    scan 0. Scan 1 holds one detection inside the gate (clutter, by its label) and one far
    outside it; scan 2 holds nothing. The ratio is computed here from the densities
    written out, not from the Kalman helpers events.py uses.
    """
    cfg = tiny_run_config.filter_cfg                 # p_D 0.85, lambda_FA 2, R = 0.04 I
    pose = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
    near, far = np.array([0.1, 2.2]), np.array([1.5, 3.5])
    scans = [Scan(k=0, t=0.0, pose=pose, detections=()),
             Scan(k=1, t=0.25, pose=pose, detections=(Detection(z=near), Detection(z=far))),
             Scan(k=2, t=0.5, pose=pose, detections=())]
    labels = [ScanLabels(k=0, origin=(), visible_ids=(), detected_ids=()),
              ScanLabels(k=1, origin=(None, 7), visible_ids=(7,), detected_ids=(7,)),
              ScanLabels(k=2, origin=(), visible_ids=(), detected_ids=())]
    run = RunDir(tmp_path / "run")
    run.root.mkdir()
    write_detections(run.detections, scans)
    write_labels(run.labels, labels)

    mean, cov = np.array([0.0, 2.0]), 0.05 * np.eye(2)
    events = build_scan_events(run.detections, run.labels, cfg,
                               [None, (mean, cov), (mean, cov)])

    s = 0.05 + 0.04                                  # S = P + R = s I
    d2 = float((near - mean) @ (near - mean)) / s
    g = math.exp(-0.5 * d2) / (2.0 * math.pi * s)    # N(z; z_hat, S) in 2D
    area = 0.6 * (4.0**2 - 0.3**2)                   # half_angle (max^2 - min^2)
    expected_ratio = 0.85 * g / (2.0 / area)

    assert [event.born for event in events] == [True, False, False]
    assert [event.in_fov for event in events] == [False, True, True]
    assert events[1].n_gated == 1 and events[1].n_clutter_gated == 1
    assert events[1].likelihood_ratios[0] == pytest.approx(expected_ratio, rel=1e-12)
    assert events[2].n_gated == 0 and events[2].likelihood_ratios == ()
    assert [event.dt for event in events] == [0.0, 0.25, 0.25]

    with pytest.raises(ValueError, match="deleted"):
        build_scan_events(run.detections, run.labels, cfg, [None, (mean, cov), None])


@pytest.fixture(scope="module")
def bank_run(tmp_path_factory) -> RunDir:
    """The shipped bank config (five phantoms, pruned), tracked into a temporary runs/."""
    runs = tmp_path_factory.mktemp("runs")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        return track_from_config(CONFIGS / "b2_bernoulli_bank_phantoms.yaml", runs)
    finally:
        os.chdir(cwd)


def test_every_bank_track_matches_the_analytic_recursion(bank_run: RunDir) -> None:
    """analyse checks each bank track on the unpruned companion log (D12, D14). [B3]

    Each bank component is advanced by the unchanged Bernoulli filter, so each track is the
    A2 recursion. The pruned log cannot be checked past a deletion (A2 has no deletion
    step); the unpruned log is identical up to it and covers the scans after it,
    recapture included (D9).
    """
    analyse_run(bank_run, plots=["r_vs_analytic"])

    check = json.loads(bank_run.metrics.read_text(encoding="utf-8"))["b3_cross_check"]
    assert check["estimates_log"] == "estimates_bernoulli_bank_unpruned.jsonl"
    assert len(check["tracks"]) == 8
    for track in check["tracks"]:
        assert track["max_abs_error"] <= R_TOLERANCE, track
        assert (bank_run.plots / f"r_vs_analytic_track{track['track_id']}.png").is_file()
    covered = {branch: sum(track["branches"][branch] for track in check["tracks"])
               for branch in BRANCHES}
    assert all(count > 0 for count in covered.values()), covered
