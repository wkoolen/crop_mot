"""Computation time per scan and problem size (roadmap step 4c, decision D29). [B4]

Times are never asserted on: they depend on the machine and its load. These tests check
the shape of what is written, so the scaling study can rely on it.
"""

from __future__ import annotations

import json

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.config import RunConfig
from crop_mot.filters import build_filter
from crop_mot.io import read_jsonl
from crop_mot.runner.run_dir import RunDir
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
