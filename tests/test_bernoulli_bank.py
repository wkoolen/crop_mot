"""The bank of independent Bernoulli filters, pruning, and the hypotheses figure. [B2]

Added with `crop_mot.filters.bernoulli_bank` (decisions D12-D14). The first test is the
important one: a one-seed bank without pruning IS the single Bernoulli filter, so every
bank track inherits B3's validation of the single filter.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from crop_mot.analysis.candidates import phantom_candidates
from crop_mot.analysis.estimates_log import r_trajectory, read_estimates, track_lifetimes
from crop_mot.analysis.plots import animate_hypotheses
from crop_mot.config import BirthConfig, PruneConfig, RunConfig, load_run_config
from crop_mot.filters import build_filter
from crop_mot.runner.analyse import analyse_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter, track_from_config, unpruned_log_name
from crop_mot.sensor.record import read_labels
from crop_mot.types import Detection, Pose2D, Scan
from crop_mot.world.truth import read_truth

from conftest import CONFIGS, REPO_ROOT

BANK_CONFIG = CONFIGS / "b2_bernoulli_bank_phantoms.yaml"
POSE = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
IN_VIEW = np.array([0.0, 2.0])


def _scan(k: int, points: list) -> Scan:
    return Scan(k=k, t=0.25 * k, pose=POSE,
                detections=tuple(Detection(z=np.asarray(p, dtype=float)) for p in points))


def _bank(cfg: RunConfig, seeds=None, r_min: float = 0.0):
    """The tiny run's filter block as a bank, optionally with seeds and pruning."""
    birth = cfg.filter_cfg.birth
    if seeds is not None:
        birth = replace(birth, kind="from_measurements", seeds=tuple(seeds),
                        at_scan=seeds[0][0], detection_index=seeds[0][1])
    return build_filter(replace(cfg.filter_cfg, kind="bernoulli_bank", birth=birth,
                                prune=PruneConfig(r_min=r_min)))


@pytest.fixture(scope="module")
def bank_run(tmp_path_factory) -> RunDir:
    """The shipped bank config, tracked once into a temporary runs/ directory."""
    runs = tmp_path_factory.mktemp("runs")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        return track_from_config(BANK_CONFIG, runs)
    finally:
        os.chdir(cwd)


def test_one_seed_bank_is_the_single_bernoulli_filter(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """Without pruning, a one-seed bank reproduces the Bernoulli filter's r exactly. [B2]"""
    scenario = replace(tiny_run_config.scenario,
                       path=replace(tiny_run_config.scenario.path, n_scans=40),
                       sensor=replace(tiny_run_config.scenario.sensor, lambda_FA=6.0))
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    simulate(scenario, run)

    single = build_filter(tiny_run_config.filter_cfg)
    bank = _bank(tiny_run_config)
    run_filter(single, run)
    run_filter(bank, run)

    r_single = r_trajectory(read_estimates(run.estimates(single.name)), 0)
    r_bank = r_trajectory(read_estimates(run.estimates(bank.name)), 0)
    assert np.any(r_single > 0.0)
    np.testing.assert_array_equal(r_bank, r_single)


def test_pruned_track_is_logged_below_threshold_then_absent(
    tiny_run_config: RunConfig
) -> None:
    """The r that crossed r_min is still reported; the next scan the track is gone. [B2]"""
    r_min = 1e-2
    flt = _bank(tiny_run_config, r_min=r_min)  # birth at scan 0 from detection 0
    state = flt.update(flt.predict(flt.initial_state(), 0.0), _scan(0, [IN_VIEW]))
    reported = [flt.extract(state)]
    for k in range(1, 10):
        state = flt.update(flt.predict(state, 0.25), _scan(k, []))
        reported.append(flt.extract(state))

    r = [estimates[0].r if estimates else None for estimates in reported]
    k_last = max(k for k, value in enumerate(r) if value is not None)
    assert r[k_last] < r_min
    assert all(value >= r_min for value in r[:k_last])
    assert all(value is None for value in r[k_last + 1:])
    assert k_last < 9, "the phantom should have been deleted within the run"


def test_track_ids_follow_seed_order(tiny_run_config: RunConfig) -> None:
    """Two seeds on one scan get ids 0 and 1; a later seed gets id 2. [B2]"""
    far = IN_VIEW + np.array([1.5, 0.0])
    flt = _bank(tiny_run_config, seeds=[(0, 1), (0, 0), (2, 0)])
    state = flt.initial_state()
    state = flt.update(flt.predict(state, 0.0), _scan(0, [IN_VIEW, far]))
    ids = {e.track_id: e.mean for e in flt.extract(state)}
    assert sorted(ids) == [0, 1]
    np.testing.assert_array_equal(ids[0], far)
    np.testing.assert_array_equal(ids[1], IN_VIEW)

    state = flt.update(flt.predict(state, 0.25), _scan(1, []))
    state = flt.update(flt.predict(state, 0.25), _scan(2, [IN_VIEW]))
    assert sorted(e.track_id for e in flt.extract(state)) == [0, 1, 2]


def test_single_bernoulli_refuses_pruning(tiny_run_config: RunConfig) -> None:
    """Pruning lives in the bank; configuring it on 'bernoulli' is an error. [B2]"""
    with pytest.raises(ValueError, match="bernoulli_bank"):
        build_filter(replace(tiny_run_config.filter_cfg, prune=PruneConfig(r_min=1e-3)))


def test_shipped_bank_config_loads() -> None:
    """The seeds list parses into (at_scan, detection_index) pairs; prune is read. [B2]"""
    cfg = load_run_config(BANK_CONFIG)
    birth: BirthConfig = cfg.filter_cfg.birth
    assert birth.kind == "from_measurements"
    assert birth.seeds[1] == (24, 12)  # D8's phantom
    assert (birth.at_scan, birth.detection_index) == birth.seeds[0]
    assert cfg.filter_cfg.prune.r_min == 1e-3


def test_bank_run_writes_the_unpruned_companion(bank_run: RunDir) -> None:
    """With pruning on, track also writes the unpruned log on the same detections. [B2]"""
    pruned = read_estimates(bank_run.estimates("bernoulli_bank"))
    unpruned = read_estimates(bank_run.estimates(unpruned_log_name("bernoulli_bank")))
    lifetimes = track_lifetimes(pruned)
    assert sorted(lifetimes) == list(range(5))
    assert any(life.deleted for life in lifetimes.values())
    assert not any(life.deleted for life in track_lifetimes(unpruned).values())

    # Up to its deletion, a pruned track is identical to its unpruned twin.
    for track_id, life in lifetimes.items():
        span = slice(life.k_birth, life.k_last + 1)
        np.testing.assert_array_equal(r_trajectory(pruned, track_id)[span],
                                      r_trajectory(unpruned, track_id)[span])


def test_phantom_candidates_are_isolated_clutter(bank_run: RunDir) -> None:
    """Every candidate is clutter, far enough from the plants and from each other. [B2]"""
    fov = load_run_config(bank_run.config).filter_cfg.assumed_sensor.fov
    candidates = phantom_candidates(bank_run, fov, min_distance=1.0, min_separation=1.0)
    assert candidates
    plants = read_truth(bank_run.truth).field.positions
    labels = read_labels(bank_run.labels)
    for c in candidates:
        assert labels[c.k].origin[c.detection_index] is None
        assert np.min(np.linalg.norm(plants - c.z, axis=1)) >= 1.0
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            assert np.linalg.norm(a.z - b.z) >= 1.0


def test_hypotheses_plot_is_written(bank_run: RunDir) -> None:
    """analyse renders the hypotheses figure for a bank run. [B2]"""
    analyse_run(bank_run, plots=["hypotheses"])
    assert (bank_run.plots / "hypotheses.png").stat().st_size > 0


@pytest.mark.slow
def test_hypotheses_animation_is_written(bank_run: RunDir) -> None:
    """One GIF frame per scan. [B2]"""
    out = animate_hypotheses(bank_run, "bernoulli_bank", bank_run.plots / "hypotheses.gif")
    with Image.open(out) as gif:
        assert gif.n_frames == len(read_estimates(bank_run.estimates("bernoulli_bank")))
