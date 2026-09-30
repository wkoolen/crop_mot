"""Class labels on detections: the stubbed classifier of roadmap step 8b (decision D22).

World side: each detection's label is drawn from its true origin's row of a confusion
matrix, from its own random stream, so labels change nothing else a scenario produces.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.counts import origin_kinds
from crop_mot.config import ClassifierConfig, ScenarioConfig, load_scenario_config
from crop_mot.io import write_jsonl
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.sensor.record import read_detections, read_labels

from conftest import CONFIGS
from test_weeds import _standing_still, _with_weeds


def _simulate(cfg: ScenarioConfig, root) -> RunDir:
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(cfg, run, summary=False)
    return run


def _pairs(run: RunDir) -> list[tuple[str, str]]:
    """(true origin kind, class label) of every detection in a run."""
    pairs = []
    for scan, labels in zip(read_detections(run.detections), read_labels(run.labels)):
        pairs += list(zip(origin_kinds(labels), [d.label for d in scan.detections]))
    return pairs


def test_the_default_classifier_is_perfect(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """Plants and weeds are labelled what they are; clutter is labelled "plant"."""
    run = _simulate(_standing_still(_with_weeds(tiny_scenario), n_scans=40), tmp_path / "r")
    pairs = _pairs(run)
    assert {kind for kind, _ in pairs} == {"plant", "weed", "clutter"}
    assert all(label == {"plant": "plant", "weed": "weed", "clutter": "plant"}[kind]
               for kind, label in pairs)


def test_labels_change_nothing_else(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """Another confusion matrix gives the same z and labels.jsonl; only labels differ."""
    base = _standing_still(_with_weeds(tiny_scenario), n_scans=20)
    noisy = replace(base, sensor=replace(base.sensor, classifier=ClassifierConfig(
        weed_given=(("plant", 0.3), ("weed", 0.6), ("clutter", 0.5)))))
    a, b = _simulate(base, tmp_path / "a"), _simulate(noisy, tmp_path / "b")
    assert a.labels.read_bytes() == b.labels.read_bytes()
    for scan_a, scan_b in zip(read_detections(a.detections), read_detections(b.detections)):
        assert [d.z.tolist() for d in scan_a.detections] == \
            [d.z.tolist() for d in scan_b.detections]
    assert [label for _, label in _pairs(a)] != [label for _, label in _pairs(b)]


def test_label_rates_follow_the_confusion_matrix(tiny_scenario: ScenarioConfig,
                                                  tmp_path) -> None:
    """Empirical P(weed label | origin) within 4 binomial sigmas of the matrix."""
    rates = {"plant": 0.2, "weed": 0.7, "clutter": 0.4}
    base = _standing_still(_with_weeds(tiny_scenario), n_scans=300)
    cfg = replace(base, sensor=replace(base.sensor, classifier=ClassifierConfig(
        weed_given=tuple(rates.items()))))
    pairs = _pairs(_simulate(cfg, tmp_path / "r"))
    for kind, p in rates.items():
        labels = [label for k, label in pairs if k == kind]
        n = len(labels)
        assert n > 100
        assert abs(labels.count("weed") / n - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_detections_written_before_labels_read_as_plants(tmp_path) -> None:
    path = tmp_path / "detections.jsonl"
    write_jsonl(path, [{"k": 0, "t": 0.0, "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
                        "z": [[1.0, 2.0], [3.0, 4.0]]}])
    (scan,) = read_detections(path)
    assert [d.label for d in scan.detections] == ["plant", "plant"]


def test_classifier_config(tmp_path) -> None:
    """Absent means perfect; a row that does not sum to 1 is refused."""
    assert load_scenario_config(CONFIGS / "b1_two_rows.yaml").sensor.classifier.perfect
    text = (CONFIGS / "b1_two_rows.yaml").read_text(encoding="utf-8")
    block = ("  classifier:\n"
             "    plant: {plant: 0.9, weed: 0.1}\n"
             "    weed: {plant: 0.3, weed: 0.7}\n"
             "    clutter: {plant: 0.8, weed: 0.3}\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("  measurement:\n", block + "  measurement:\n", 1),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="clutter"):
        load_scenario_config(bad)
    good = tmp_path / "good.yaml"
    good.write_text(text.replace("  measurement:\n",
                                 block.replace("0.8, weed: 0.3", "0.7, weed: 0.3")
                                 + "  measurement:\n", 1), encoding="utf-8")
    classifier = load_scenario_config(good).sensor.classifier
    assert classifier.p_weed("weed") == 0.7 and not classifier.perfect


# ---- Filter side: the assumed classifier (roadmap step 8b, D22) -------------------------

PERFECT = ClassifierConfig()


def _known_n(cfg, scenario: ScenarioConfig, classifier):
    from crop_mot.config import PlanConfig
    return replace(cfg.filter_cfg, kind="bernoulli_bank", birth=None,
                   plan=PlanConfig(rows=scenario.world.rows, prior_std=0.036),
                   assumed_classifier=classifier)


def test_without_weeds_the_labels_change_no_estimate(tiny_run_config, tmp_path) -> None:
    """The reduction test of step 8b: no weeds, perfect labels -> exactly step 8a."""
    from crop_mot.analysis.estimates_log import read_estimates
    from crop_mot.filters import build_filter
    from crop_mot.runner.track import run_filter

    scenario = replace(tiny_run_config.scenario,
                       path=replace(tiny_run_config.scenario.path, n_scans=30))
    run = _simulate(scenario, tmp_path / "r")
    logs = {}
    for name, classifier in (("blind", None), ("labels", PERFECT)):
        run_filter(build_filter(_known_n(tiny_run_config, scenario, classifier)), run,
                   log_name=name)
        logs[name] = read_estimates(run.estimates(name))
    for a, b in zip(logs["blind"], logs["labels"]):
        for ea, eb in zip(a.estimates, b.estimates):
            assert ea.r == eb.r
            assert np.array_equal(ea.mean, eb.mean) and np.array_equal(ea.cov, eb.cov)


def test_a_labelled_weed_never_enters_a_plant_gate(tiny_run_config, tmp_path) -> None:
    """With labels, the rebuilt gates hold no weed-labelled detection; without, some do."""
    from crop_mot.analysis.events import gated_detection_indices

    scenario = _standing_still(_with_weeds(tiny_run_config.scenario), n_scans=10)
    run = _simulate(scenario, tmp_path / "r")
    mean, cov = np.array([0.0, 4.3]), 0.12 * np.eye(2)       # inside the weed patch
    counts = {}
    for name, classifier in (("blind", None), ("labels", PERFECT)):
        cfg = _known_n(tiny_run_config, scenario, classifier)
        counts[name] = sum(scan.detections[i].label == "weed"
                           for scan in read_detections(run.detections)
                           for i in gated_detection_indices(scan, mean, cov, cfg))
    assert counts["blind"] > 0 and counts["labels"] == 0


def test_an_imperfect_assumed_classifier_waits_on_its_derivation(tiny_run_config) -> None:
    from crop_mot.filters import build_filter

    noisy = ClassifierConfig(weed_given=(("plant", 0.05), ("weed", 0.9), ("clutter", 0.1)))
    with pytest.raises(NotImplementedError, match="open question 7"):
        build_filter(replace(tiny_run_config.filter_cfg, assumed_classifier=noisy))


def test_the_labels_config_is_the_weeds_config_plus_the_classifier() -> None:
    import yaml

    from crop_mot.config import load_run_config

    assert load_run_config(CONFIGS / "b4_known_n_bank_labels.yaml").filter_cfg \
        .assumed_classifier.perfect
    labels = yaml.safe_load((CONFIGS / "b4_known_n_bank_labels.yaml").read_text())
    blind = yaml.safe_load((CONFIGS / "b4_known_n_bank_weeds.yaml").read_text())
    del labels["filter"]["assumed_classifier"]
    labels["name"] = blind["name"]
    assert labels == blind
