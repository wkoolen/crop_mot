"""B2: does the Bernoulli filter behave like a Bernoulli filter? [B2]

Properties rather than fixed numbers. The exact r trajectory is B3's job, checked against
the closed form; these tests catch the coarser failures that would make B3's comparison
meaningless in the first place.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.config import RunConfig
from crop_mot.filters import build_filter
from crop_mot.filters.bernoulli import BernoulliState
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter
from crop_mot.types import Detection, Pose2D, Scan

# A pose from which the point (0, 2) is well inside the FOV: 2 m straight ahead.
POSE = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
IN_VIEW = np.array([0.0, 2.0])


def _simulate_and_track(cfg: RunConfig, root) -> list:
    """Simulate cfg.scenario into `root`, run the Bernoulli filter, return its estimates log."""
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(cfg.scenario, run)
    flt = build_filter(cfg.filter_cfg)
    run_filter(flt, run)
    return read_estimates(run.estimates(flt.name))


def _walk_past(cfg: RunConfig, n_scans: int = 60, lambda_FA: float | None = None) -> RunConfig:
    """The tiny run with the robot walking past the whole row, optionally more clutter."""
    scenario = cfg.scenario
    scenario = replace(scenario, path=replace(scenario.path, n_scans=n_scans))
    if lambda_FA is not None:
        scenario = replace(scenario, sensor=replace(scenario.sensor, lambda_FA=lambda_FA))
    return replace(cfg, scenario=scenario)


def _scan(k: int, points: list) -> Scan:
    return Scan(k=k, t=0.25 * k, pose=POSE,
                detections=tuple(Detection(z=np.asarray(p, dtype=float)) for p in points))


def test_r_stays_in_unit_interval(tiny_run_config: RunConfig, tmp_path) -> None:
    """r is a probability at every scan. [B2]

    The cheapest possible check, and it catches the most common Bernoulli bug: a missing
    normalisation in the update, which typically pushes r above 1 only on scans with
    several gated detections.
    """
    # A full walk past the row with plenty of clutter, so several scans gate several detections.
    records = _simulate_and_track(_walk_past(tiny_run_config, lambda_FA=6.0), tmp_path / "run")
    r_values = [est.r for rec in records for est in rec.estimates]
    assert r_values, "the phantom was never born"
    assert all(0.0 <= r <= 1.0 for r in r_values)

    # And directly: one update with five detections crowding the predicted mean.
    flt = build_filter(tiny_run_config.filter_cfg)
    state = BernoulliState(r=0.5, mean=IN_VIEW.copy(), cov=0.05 * np.eye(2), track_id=0)
    crowd = [IN_VIEW + offset for offset in 0.05 * np.random.default_rng(0).normal(size=(5, 2))]
    posterior = flt.update(flt.predict(state, 0.25), _scan(1, crowd))
    assert 0.0 <= posterior.r <= 1.0


def test_r_decays_under_sustained_misdetection(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """With no detection ever assigned, r decreases monotonically. [B2]

    This is the phantom-track behaviour the B2 plot has to show. With p_S = 1 the
    prediction leaves r untouched, so any decay must come from the misdetection branch -
    which means a flat r here points at the update, not at the survival model.
    """
    flt = build_filter(tiny_run_config.filter_cfg)  # birth at scan 0 from detection 0

    state = flt.initial_state()
    state = flt.update(flt.predict(state, 0.0), _scan(0, [IN_VIEW]))
    r_history = [state.r]
    assert state.r == tiny_run_config.filter_cfg.birth.r_b

    for k in range(1, 15):
        predicted = flt.predict(state, 0.25)
        assert predicted.r == state.r  # p_S = 1: the prediction leaves r untouched
        state = flt.update(predicted, _scan(k, []))
        r_history.append(state.r)

    assert all(later < earlier for earlier, later in zip(r_history, r_history[1:]))
    assert r_history[-1] > 0.0  # decays, but a Bayesian update never reaches 0 [A2 §5]


def test_predict_and_update_are_pure(tiny_run_config: RunConfig) -> None:
    """Neither predict nor update mutates the state it was given. [B2/B4]

    Part of the TrackingFilter contract. Purity is what lets the other tests run a single
    prediction in isolation, and what would let a future experiment rewind to scan k
    without re-running from 0.
    """
    flt = build_filter(tiny_run_config.filter_cfg)
    mean = IN_VIEW.copy()
    cov = 0.05 * np.eye(2)
    state = BernoulliState(r=0.5, mean=mean, cov=cov, track_id=0)
    mean_before, cov_before = mean.copy(), cov.copy()

    predicted = flt.predict(state, 0.25)
    posterior = flt.update(predicted, _scan(1, [IN_VIEW + 0.05, [1.0, 3.0]]))

    assert state.r == 0.5
    assert np.array_equal(state.mean, mean_before)
    assert np.array_equal(state.cov, cov_before)
    # The outputs do not share storage with the input, so mutating them cannot reach back.
    for output in (predicted, posterior):
        assert not np.shares_memory(output.mean, state.mean)
        assert not np.shares_memory(output.cov, state.cov)


def test_covariance_stays_symmetric_positive_definite(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """P stays symmetric and positive definite over a full run. [B2]

    With a static target and p_S = 1, P shrinks monotonically, so accumulated floating-point
    asymmetry eventually makes it indefinite. That failure surfaces much later as a Cholesky
    error inside a likelihood evaluation, far from its cause - hence the Joseph-form note in
    `crop_mot.filters.kalman.kf_update`.
    """
    records = _simulate_and_track(_walk_past(tiny_run_config), tmp_path / "run")
    covs = [est.cov for rec in records for est in rec.estimates]

    # A long static run: the same target detected on every scan, so P shrinks like 1/k.
    flt = build_filter(tiny_run_config.filter_cfg)
    state = flt.update(flt.predict(flt.initial_state(), 0.0), _scan(0, [IN_VIEW]))
    noise = 0.2 * np.random.default_rng(1).normal(size=(2000, 2))
    for k in range(1, 2001):
        state = flt.update(flt.predict(state, 0.25), _scan(k, [IN_VIEW + noise[k - 1]]))
        covs.append(state.cov)

    for P in covs:
        assert np.array_equal(P, P.T)
        assert np.all(np.linalg.eigvalsh(P) > 0.0)


def test_extract_returns_at_most_one_track(tiny_run_config: RunConfig, tmp_path) -> None:
    """A Bernoulli filter is single-target: extract returns a list of length <= 1. [B2]

    Documents the N = 1 case of the shared interface. The runner does not special-case it,
    and neither should the filter.
    """
    records = _simulate_and_track(_walk_past(tiny_run_config, lambda_FA=6.0), tmp_path / "run")
    assert len(records) == 60
    assert all(len(rec.estimates) <= 1 for rec in records)
    assert any(len(rec.estimates) == 1 for rec in records)
