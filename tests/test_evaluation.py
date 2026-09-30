"""Per-scan evaluation against truth: the view, cardinality, GOSPA and NEES. [B4, step 5]

The series every step-5 figure is drawn from. Checked by hand on synthetic scans, and on
the shipped bank run for what only a real run folder can show.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.stats import chi2

from crop_mot.analysis.evaluation import (
    ScanView,
    cardinality,
    gospa_series,
    nees,
    nees_band,
    scan_views,
)
from crop_mot.runner.analyse import ANY_FILTER_PLOTS, analyse_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import track_from_config
from crop_mot.sensor.record import read_labels
from crop_mot.types import TrackEstimate

from conftest import CONFIGS, REPO_ROOT


def _track(track_id: int, r: float, mean, var: float) -> TrackEstimate:
    return TrackEstimate(track_id=track_id, r=r, mean=np.asarray(mean, dtype=float),
                         cov=var * np.eye(2))


def test_cardinality_gospa_and_nees_by_hand() -> None:
    """Two plants in view; one confirmed track near plant 0, one unconfirmed track. [B4]"""
    plants = np.array([[0.0, 2.0], [0.0, 2.35]])
    view = ScanView(k=0, plant_positions=plants,
                    tracks=(_track(0, 0.9, [0.1, 2.0], 0.01), _track(1, 0.2, [1.0, 3.0], 0.01)))

    count = cardinality([view])
    assert count.sum_r[0] == pytest.approx(1.1) and count.n_plants[0] == 2

    (result,) = gospa_series([view])
    assert result.pairs == ((0, 0),)              # the r = 0.2 track is not confirmed
    assert (result.n_missed, result.n_false) == (1, 0)

    error = nees([view], [result])
    assert error.n_pairs[0] == 1 and error.dim == 2
    assert error.mean_nees[0] == pytest.approx(0.1**2 / 0.01)   # e' P^-1 e = 1.0

    empty = nees([ScanView(k=1, tracks=(), plant_positions=plants)],
                 gospa_series([ScanView(k=1, tracks=(), plant_positions=plants)]))
    assert np.isnan(empty.mean_nees[0]) and empty.n_pairs[0] == 0


def test_nees_band_is_the_averaged_chi_square() -> None:
    """One pair: the chi2(2) quantiles; n pairs: chi2(2 n) / n; none: NaN. [B4]"""
    lower, upper = nees_band(np.array([1, 10, 0]), dim=2)
    assert lower[0] == pytest.approx(chi2.ppf(0.025, 2))
    assert upper[1] == pytest.approx(chi2.ppf(0.975, 20) / 10)
    assert np.isnan(lower[2]) and np.isnan(upper[2])


@pytest.fixture(scope="module")
def bank_run(tmp_path_factory) -> RunDir:
    """The shipped bank config, tracked into a temporary runs/."""
    runs = tmp_path_factory.mktemp("runs")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        return track_from_config(CONFIGS / "b2_bernoulli_bank_phantoms.yaml", runs)
    finally:
        os.chdir(cwd)


def test_plants_in_view_are_the_simulators_visible_plants(bank_run: RunDir) -> None:
    """Truth side of D21: the plant count per scan is labels' visible_ids. [B4]"""
    views = scan_views(bank_run, "bernoulli_bank")
    labels = read_labels(bank_run.labels)
    assert [len(view.plant_positions) for view in views] == \
        [len(scan.visible_ids) for scan in labels]
    for view in views:
        assert all(track.r > 0.0 for track in view.tracks)


def test_any_filter_figures_render_for_the_bank(bank_run: RunDir) -> None:
    """Every step-5 figure renders from the run folder alone. [B4, step 5]"""
    analyse_run(bank_run, plots=list(ANY_FILTER_PLOTS))
    for name in ANY_FILTER_PLOTS:
        assert (bank_run.plots / f"{name}.png").stat().st_size > 0
