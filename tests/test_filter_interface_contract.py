"""Every registered filter satisfies the shared interface. [B2/B4]

This is the test that keeps B4 additive. A new filter added to `FILTERS` is picked up here
automatically, so "it plugs into the pipeline" is verified rather than assumed - and the
day PMBM is added, this file says whether it really does behave like the others.

Parametrised over FILTERS rather than listing filters explicitly, on purpose: an explicit
list is one more place to forget to update.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.config import RunConfig
from crop_mot.filters import FILTERS
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter, track_all_filters
from crop_mot.types import Pose2D, Scan

FILTER_NAMES = sorted(FILTERS)


def _build(filter_name: str, cfg: RunConfig):
    """Build a registered filter from the tiny run's filter block."""
    return FILTERS[filter_name](replace(cfg.filter_cfg, kind=filter_name))


def _simulated_run(cfg: RunConfig, root) -> RunDir:
    """The tiny scenario, walking past the row, simulated into `root`."""
    scenario = replace(cfg.scenario, path=replace(cfg.scenario.path, n_scans=40))
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(scenario, run)
    return run


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_filter_implements_the_four_methods(
    filter_name: str, tiny_run_config: RunConfig
) -> None:
    """The filter has initial_state, predict, update and extract, and they are callable. [B2/B4]

    A structural check that passes today against the Bernoulli builder and will pass for
    every B4 filter without being edited.
    """
    flt = _build(filter_name, tiny_run_config)
    assert flt.name == filter_name
    for method in ("initial_state", "predict", "update", "extract"):
        assert callable(getattr(flt, method, None)), f"{filter_name} lacks {method}"

    # One pass through the loop on an empty scan runs and returns a list.
    empty = Scan(k=0, t=0.0, pose=Pose2D(x=0.0, y=0.0, theta=0.0), detections=())
    state = flt.update(flt.predict(flt.initial_state(), 0.0), empty)
    assert isinstance(flt.extract(state), list)


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_extract_returns_valid_track_estimates(
    filter_name: str, tiny_run_config: RunConfig, tmp_path
) -> None:
    """Whatever a filter reports is a well-formed TrackEstimate. [B2/B4]

    For every returned estimate: r is in [0, 1], mean has shape (dim_x,), cov has shape
    (dim_x, dim_x) and is symmetric, and track_id is stable across scans for the same track.

    This is the contract the evaluation code relies on. GNN's r of exactly 0.0 or 1.0 is a
    valid value here, not a special case - which is the point of having one interface.
    """
    run = _simulated_run(tiny_run_config, tmp_path / "run")
    flt = _build(filter_name, tiny_run_config)
    run_filter(flt, run)
    records = read_estimates(run.estimates(flt.name))
    dim_x = tiny_run_config.filter_cfg.birth.init_cov.shape[0]

    assert any(rec.estimates for rec in records)
    previous_ids = None
    for rec in records:
        ids = [est.track_id for est in rec.estimates]
        assert len(ids) == len(set(ids)), "track ids must be unique within a scan"
        for est in rec.estimates:
            assert 0.0 <= est.r <= 1.0
            assert est.mean.shape == (dim_x,)
            assert est.cov.shape == (dim_x, dim_x)
            assert np.allclose(est.cov, est.cov.T)
        # A single track that persists from one scan to the next keeps its id.
        if previous_ids is not None and len(ids) == 1 and len(previous_ids) == 1:
            assert ids == previous_ids
        previous_ids = ids


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_filter_never_touches_ground_truth(
    filter_name: str, tiny_run_config: RunConfig, tmp_path
) -> None:
    """Running a filter requires only detections.jsonl. [B2/B4]

    Run the filter against a run folder from which truth.jsonl and labels.jsonl have been
    DELETED, and assert it completes normally. A filter that had quietly started reading
    truth would fail here with a FileNotFoundError.

    Stronger than an inspection of imports, and it keeps holding as filters grow.
    """
    run = _simulated_run(tiny_run_config, tmp_path / "run")
    run.truth.unlink()
    run.labels.unlink()

    flt = _build(filter_name, tiny_run_config)
    run_filter(flt, run)  # a filter that read truth would raise FileNotFoundError here
    assert run.estimates(flt.name).is_file()


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_all_filters_consume_the_same_detections(
    filter_name: str, tiny_run_config: RunConfig, tmp_path
) -> None:
    """Every filter reads the identical detection file in a shared run folder. [B4]

    The fair-comparison guarantee, made testable: simulate once, run every registered
    filter into the same run folder, and assert that detections.jsonl is unchanged
    afterwards and that each filter wrote its own estimates_<name>.jsonl.
    """
    run = _simulated_run(tiny_run_config, tmp_path / "run")
    before = run.detections.read_bytes()

    track_all_filters(tiny_run_config, run, FILTER_NAMES)

    assert run.detections.read_bytes() == before
    for name in FILTER_NAMES:
        assert run.estimates(name).is_file()
    assert filter_name in FILTER_NAMES
