"""Computation time per scan and problem size (roadmap step 4c, decision D29). [B4]

Times are never asserted on: they depend on the machine and its load. These tests check
the shape of what is written, so the scaling study can rely on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.analysis.timing import SWEEPS
from crop_mot.config import RegionConfig, RunConfig, WeedsConfig
from crop_mot.filters import build_filter
from crop_mot.io import read_jsonl
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.scaling import run_scaling
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter
from crop_mot.sensor.record import read_detections

FIELDS = {"k", "predict_s", "update_s", "extract_s", "n_detections", "n_reported"}


def test_run_filter_writes_one_timing_record_per_scan(
    tiny_run_config: RunConfig, tmp_run_dir: RunDir
) -> None:
    simulate(tiny_run_config.scenario, tmp_run_dir)
    flt = build_filter(tiny_run_config.filter_cfg)
    run_filter(flt, tmp_run_dir)

    timing = list(read_jsonl(tmp_run_dir.timing(flt.name)))
    scans = read_detections(tmp_run_dir.detections)
    records = read_estimates(tmp_run_dir.estimates(flt.name))
    assert [row["k"] for row in timing] == [scan.k for scan in scans]
    for row, scan, record in zip(timing, scans, records):
        assert set(row) == FIELDS
        assert min(row["predict_s"], row["update_s"], row["extract_s"]) >= 0.0
        assert row["n_detections"] == len(scan.detections)
        assert row["n_reported"] == len(record.estimates)


def test_run_meta_records_threads_and_cpu(tiny_run_config: RunConfig,
                                          tmp_run_dir: RunDir) -> None:
    simulate(tiny_run_config.scenario, tmp_run_dir)
    meta = json.loads(tmp_run_dir.meta.read_text(encoding="utf-8"))
    assert set(meta["threads"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS"}
    assert "cpu" in meta


def test_sweeps_change_only_their_parameter(tiny_run_config: RunConfig) -> None:
    cfg = tiny_run_config
    longer = SWEEPS["row_length"].apply(cfg, 20.0)
    row, path = longer.scenario.world.rows[0], longer.scenario.path
    assert row.y_end == row.y_start + 20.0
    assert path.n_scans == math.ceil((row.y_end - path.y_start) / (path.speed
                                                                 * path.scan_period)) + 1
    clutter = SWEEPS["lambda_FA"].apply(cfg, 8.0)
    assert clutter.scenario.sensor.lambda_FA == 8.0
    assert clutter.filter_cfg.assumed_sensor.lambda_FA == 8.0
    assert SWEEPS["spacing"].apply(cfg, 0.5).scenario.world.rows[0].spacing == 0.5
    with pytest.raises(ValueError, match="weeds"):
        SWEEPS["weed_density"].apply(cfg, 0.3)
    weeds = WeedsConfig(density=0.1, region=RegionConfig(x_min=-1, x_max=1, y_min=0, y_max=6))
    weedy = replace(cfg, scenario=replace(cfg.scenario, world=replace(
        cfg.scenario.world, weeds=weeds)))
    assert SWEEPS["weed_density"].apply(weedy, 0.3).scenario.world.weeds.density == 0.3


def test_run_scaling_writes_its_points_and_figure(tiny_run_config: RunConfig,
                                                  tmp_run_dir: RunDir) -> None:
    """Two clutter rates, two seeds: one point each, scan 0 kept apart; no time asserted."""
    result = run_scaling(tiny_run_config, "lambda_FA", [1.0, 6.0], n_seeds=2, out=tmp_run_dir)

    n_scans = tiny_run_config.scenario.path.n_scans
    assert [p.value for p in result.points] == [1.0, 6.0]
    assert all(p.n_scans == 2 * (n_scans - 1) for p in result.points)
    assert result.points[1].size > result.points[0].size      # more detections per scan
    assert result.budget_s == tiny_run_config.scenario.path.scan_period
    saved = json.loads(tmp_run_dir.metrics.read_text(encoding="utf-8"))["scaling"]
    assert len(saved["points"]) == 2 and saved["sweep"] == "lambda_FA"
    assert (tmp_run_dir.plots / "scaling.png").stat().st_size > 0
    assert np.isfinite(result.slope)
    assert {p.name for p in tmp_run_dir.root.iterdir()} == {"plots", "metrics.json"}
