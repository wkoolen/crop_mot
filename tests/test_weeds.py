"""Weeds: persistent false targets (decision D15). [B1/B2]

A weed is a static object that is not a plant. The detector reports it as a plant now and
then, and because it does not move, that false alarm recurs at the same place - which the
Poisson clutter every filter assumes cannot do. These tests pin the three things the
comparison with and without weeds rests on: weeds only ADD detections (nothing else in a
scan changes), a weed is reported at its configured rate and at its own position, and a
phantom born on a weed is confirmed rather than decaying.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.counts import format_summary, scan_counts, summarise, weed_origin
from crop_mot.config import (
    DetectionConfig,
    RegionConfig,
    RunConfig,
    ScenarioConfig,
    WeedsConfig,
    load_run_config,
    load_scenario_config,
)
from crop_mot.filters import build_filter
from crop_mot.rng import substreams
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.sensor.detector import sample_scan
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.models import build_measurement_model
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.sensor.sensor_model import build_sensor_model
from crop_mot.world.field import generate_field
from crop_mot.world.path import generate_path
from crop_mot.world.truth import GroundTruth, read_truth, write_truth

from conftest import CONFIGS

# A patch of weeds straight ahead of the robot parked at y = 2.3 (see `_standing_still`):
# 1.5-2.5 m away and at least 0.6 m (3 sigma of the measurement noise) inside the wedge,
# so FOV truncation of a weed detection is negligible. Seed 3 draws 8 weeds here.
PATCH = WeedsConfig(density=10.0, region=RegionConfig(x_min=-0.4, x_max=0.4,
                                                      y_min=3.8, y_max=4.8))
WEED_P_D = DetectionConfig(kind="constant", p_D=0.5)


def _with_weeds(cfg: ScenarioConfig, weeds: WeedsConfig = PATCH,
                detection: DetectionConfig = WEED_P_D) -> ScenarioConfig:
    return replace(cfg, world=replace(cfg.world, weeds=weeds),
                   sensor=replace(cfg.sensor, weed_detection=detection))


def _standing_still(cfg: ScenarioConfig, n_scans: int) -> ScenarioConfig:
    """The tiny scenario parked at y = 2.3, as in test_b1_simulator."""
    return replace(cfg, path=replace(cfg.path, y_start=2.3, speed=0.0, n_scans=n_scans))


def _simulate_into(cfg: ScenarioConfig, root) -> RunDir:
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(cfg, run)
    return run


def test_weeds_only_add_detections(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """With weeds, each scan is the scan without weeds plus the weed detections. [B1, D15]

    The whole with / without comparison depends on this: the plants, every plant
    detection and every Poisson clutter return are identical, so any difference in a
    filter's output is caused by the weeds. It holds because weeds draw only from their
    own two substreams.
    """
    cfg = _standing_still(tiny_scenario, n_scans=60)
    without = _simulate_into(cfg, tmp_path / "without")
    with_weeds = _simulate_into(_with_weeds(cfg), tmp_path / "with")

    truth_without, truth_with = read_truth(without.truth), read_truth(with_weeds.truth)
    assert np.array_equal(truth_without.field.positions, truth_with.field.positions)
    assert len(truth_without.weeds) == 0 and len(truth_with.weeds) == 8

    n_weed = 0
    for scan_n, label_n, scan_w, label_w in zip(
            read_detections(without.detections), read_labels(without.labels),
            read_detections(with_weeds.detections), read_labels(with_weeds.labels)):
        kept = [(d.z, o) for d, o, w in zip(scan_w.detections, label_w.origin,
                                            weed_origin(label_w)) if w is None]
        assert len(kept) == len(scan_n.detections)
        for (z, origin), detection, origin_n in zip(kept, scan_n.detections, label_n.origin):
            assert np.array_equal(z, detection.z) and origin == origin_n
        assert label_w.detected_ids == label_n.detected_ids
        n_weed += len(scan_w.detections) - len(kept)
    assert n_weed > 0


def test_zero_weeds_is_byte_identical_to_no_weeds(tiny_scenario: ScenarioConfig,
                                                  tmp_path) -> None:
    """A weeds block that draws no weed changes no output file at all. [B1, D15]"""
    cfg = replace(tiny_scenario, path=replace(tiny_scenario.path, n_scans=60))
    none = _simulate_into(cfg, tmp_path / "none")
    zero = _simulate_into(_with_weeds(cfg, weeds=replace(PATCH, density=0.0)),
                          tmp_path / "zero")
    for name in ("detections", "labels", "truth"):
        assert getattr(none, name).read_bytes() == getattr(zero, name).read_bytes(), name


def test_weed_report_rate_matches_weed_p_D(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """Over many scans, weed detections / visible weed-scans converges to the weed p_D. [B1]"""
    cfg = _with_weeds(_standing_still(tiny_scenario, n_scans=400))
    summary = summarise(scan_counts(read_labels(_simulate_into(cfg, tmp_path / "run").labels)))

    n = summary["n_visible_weeds"]
    assert n >= 1000
    p = WEED_P_D.p_D
    assert abs(summary["weed_p_D_empirical"] - p) < 4.0 * np.sqrt(p * (1.0 - p) / n)
    # The Poisson clutter is still Poisson(lambda_FA): weeds are counted apart from it.
    lam = cfg.sensor.lambda_FA
    assert abs(summary["transient_per_scan"] - lam) < 4.0 * np.sqrt(lam / cfg.path.n_scans)


def test_weed_detections_recur_at_their_weed(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """A weed detection is clutter, lies near its weed, and only visible weeds report. [B1]"""
    cfg = _with_weeds(replace(tiny_scenario, path=replace(tiny_scenario.path, n_scans=60)))
    run = _simulate_into(cfg, tmp_path / "run")
    truth = read_truth(run.truth)
    sigma = np.sqrt(cfg.sensor.measurement.R[0, 0])

    n_weed = 0
    for scan, label, sample in zip(read_detections(run.detections), read_labels(run.labels),
                                   truth.poses):
        in_view = [i for i, w in enumerate(truth.weeds)
                   if in_fov(w, sample.true, cfg.sensor.fov)]
        assert list(label.visible_weed_ids) == in_view
        assert len(label.weed_origin) == len(label.origin)
        for detection, origin, weed in zip(scan.detections, label.origin, label.weed_origin):
            if weed is None:
                continue
            n_weed += 1
            assert origin is None
            assert weed in label.visible_weed_ids
            assert np.linalg.norm(detection.z - truth.weeds[weed]) < 5.0 * sigma
            assert in_fov(detection.z, sample.true, cfg.sensor.fov)
    assert n_weed > 0


def test_counts_split_the_clutter(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """clutter = weed + transient every scan; without weeds the summary has no weed lines."""
    cfg = _standing_still(tiny_scenario, n_scans=60)
    counts = scan_counts(read_labels(_simulate_into(_with_weeds(cfg), tmp_path / "w").labels))
    assert all(c.n_clutter == c.n_weed + c.n_transient for c in counts)
    assert sum(c.n_weed for c in counts) > 0
    assert "from weeds" in format_summary(counts)

    plain = scan_counts(read_labels(_simulate_into(cfg, tmp_path / "plain").labels))
    assert all(c.n_weed == 0 and c.n_visible_weeds == 0 for c in plain)
    assert "weed" not in format_summary(plain)


def test_truth_roundtrips_weeds_and_old_files_load(tiny_scenario: ScenarioConfig,
                                                   tmp_path) -> None:
    """truth.jsonl keeps the weeds exactly; a file written before weeds has none. [B1]"""
    streams = substreams(tiny_scenario.seed)
    weeds = np.array([[0.1, 1.0 / 3.0], [-2.0, 7.25]])
    truth = GroundTruth(field=generate_field(tiny_scenario.world, streams["field"]),
                        poses=generate_path(tiny_scenario.path, streams["path"]), weeds=weeds)
    path = tmp_path / "truth.jsonl"
    write_truth(path, truth)
    assert np.array_equal(read_truth(path).weeds, weeds)

    lines = path.read_text(encoding="utf-8").splitlines()
    old = tmp_path / "old_truth.jsonl"
    old.write_text("\n".join(line for line in lines if '"weeds"' not in line) + "\n",
                   encoding="utf-8")
    assert read_truth(old).weeds.shape == (0, 2)


def test_weed_blocks_go_together(tmp_path) -> None:
    """world.weeds without sensor.weed_detection is an error, and the reverse too. [D15]"""
    text = (CONFIGS / "b1_two_rows_weeds.yaml").read_text(encoding="utf-8")
    no_detection = tmp_path / "no_detection.yaml"
    no_detection.write_text(text.replace("  weed_detection: {kind: constant, p_D: 0.5}\n", ""),
                            encoding="utf-8")
    with pytest.raises(ValueError, match="go together"):
        load_scenario_config(no_detection)

    base = (CONFIGS / "b1_two_rows.yaml").read_text(encoding="utf-8")
    no_weeds = tmp_path / "no_weeds.yaml"
    no_weeds.write_text(base.replace("  lambda_FA: 2.0",
                                     "  weed_detection: {kind: constant, p_D: 0.5}\n"
                                     "  lambda_FA: 2.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="go together"):
        load_scenario_config(no_weeds)


def test_shipped_weed_configs_load() -> None:
    """The with-weeds scenario is the plain one plus weeds; the bank run points at it."""
    plain = load_scenario_config(CONFIGS / "b1_two_rows.yaml")
    weeds = load_scenario_config(CONFIGS / "b1_two_rows_weeds.yaml")
    assert plain.world.weeds is None and plain.sensor.weed_detection is None
    assert weeds.world.weeds.density > 0.0 and weeds.sensor.weed_detection.p_D == 0.5
    assert weeds.seed == plain.seed and weeds.path == plain.path
    assert weeds.world.rows == plain.world.rows
    assert weeds.world.position_jitter_std == plain.world.position_jitter_std
    for name in ("fov", "detection", "lambda_FA", "multiplicity"):
        assert getattr(weeds.sensor, name) == getattr(plain.sensor, name), name
    assert np.array_equal(weeds.sensor.measurement.R, plain.sensor.measurement.R)

    run = load_run_config(CONFIGS / "b2_bernoulli_bank_weeds.yaml")
    assert run.scenario.name == "b1_two_rows_weeds"
    assert len(run.filter_cfg.birth.seeds) == 5


def test_phantom_born_on_a_weed_is_confirmed(tiny_run_config: RunConfig) -> None:
    """A track born on a weed that keeps being reported is confirmed, not decayed. [B2, D15]

    The modelling point of the whole extension. A phantom born on transient clutter decays
    because nothing is at its location on later scans; a weed IS there, so the filter -
    which assumes all clutter is Poisson - sees repeated detections at the track and drives
    r up. One weed, 2.4 m from the nearest plant (outside the gate), reported with
    probability 0.9, plus the usual Poisson clutter; the robot is parked facing it.
    """
    scenario = replace(tiny_run_config.scenario,
                       path=replace(tiny_run_config.scenario.path, n_scans=20, speed=0.0))
    streams = substreams(scenario.seed)
    weed = np.array([-0.5, 2.0])
    truth = GroundTruth(field=generate_field(scenario.world, streams["field"]),
                        poses=generate_path(scenario.path, streams["path"]),
                        weeds=weed[None, :])
    measurement = build_measurement_model(scenario.sensor.measurement)
    model = build_sensor_model(scenario.sensor.fov, scenario.sensor.detection,
                               scenario.sensor.lambda_FA, measurement)
    weed_model = build_sensor_model(scenario.sensor.fov, DetectionConfig("constant", p_D=0.9),
                                    0.0, measurement)
    scans, labels = zip(*(
        sample_scan(truth, sample.true, k, sample.t, model, streams["detection"],
                    streams["clutter"], weed_model=weed_model,
                    rng_weeds=streams["weed_detection"])
        for k, sample in enumerate(truth.poses)))
    assert np.min(np.linalg.norm(truth.field.positions - weed, axis=1)) > 2.0

    # Seed at the weed's first detection - chosen from the labels, as a config author would.
    k0, index = next((label.k, label.weed_origin.index(0)) for label in labels
                     if 0 in label.weed_origin)
    birth = replace(tiny_run_config.filter_cfg.birth, at_scan=k0, detection_index=index,
                    r_b=0.08)
    flt = build_filter(replace(tiny_run_config.filter_cfg, birth=birth))

    state = flt.initial_state()
    for i, scan in enumerate(scans):
        state = flt.update(flt.predict(state, 0.0 if i == 0 else scan.t - scans[i - 1].t),
                           scan)
    (estimate,) = flt.extract(state)
    assert estimate.r > 0.99
    assert np.linalg.norm(estimate.mean - weed) < 0.2
