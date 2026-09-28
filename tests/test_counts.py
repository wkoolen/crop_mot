"""Clairvoyant per-scan counts, and the analyse paths that use them. [B1]

Added with `crop_mot.analysis.counts`. The counts are derived from labels.jsonl; these
tests pin the bookkeeping identities, the edge-loss field they rely on, and the guard that
keeps multi-detection scenarios away from the single-detection B3 closed form.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from crop_mot.analysis.counts import format_summary, scan_counts, summarise
from crop_mot.config import MultiplicityConfig, ScenarioConfig
from crop_mot.runner.analyse import analyse_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate, simulate_from_config
from crop_mot.sensor.record import read_labels
from crop_mot.types import ScanLabels

from conftest import CONFIGS


def _labels(cfg: ScenarioConfig, root) -> list[ScanLabels]:
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(cfg, run)
    return read_labels(run.labels)


@pytest.mark.parametrize("multiplicity", [
    MultiplicityConfig(),
    MultiplicityConfig(kind="duplicate", p_split=0.3, max_extra=3, spread_std=0.1),
    MultiplicityConfig(kind="poisson", gamma=3.0, extent_std=0.08),
], ids=["single", "duplicate", "poisson"])
def test_count_identities_hold_every_scan(
    tiny_scenario: ScenarioConfig, tmp_path, multiplicity: MultiplicityConfig
) -> None:
    """total = from plants + clutter, and visible = detected + missed + lost at the edge."""
    cfg = replace(tiny_scenario,
                  path=replace(tiny_scenario.path, n_scans=60),
                  sensor=replace(tiny_scenario.sensor, multiplicity=multiplicity))
    counts = scan_counts(_labels(cfg, tmp_path / "run"))

    for c in counts:
        assert c.n_total == c.n_object_detections + c.n_clutter
        assert c.n_visible == c.n_detected_objects + c.n_missed + c.n_truncated
        assert c.n_missed >= 0
        assert c.n_object_detections >= c.n_detected_objects
        if multiplicity.kind == "single":
            assert c.n_object_detections == c.n_detected_objects

    summary = summarise(counts)
    assert summary["n_scans"] == 60
    assert summary["n_visible"] > 0
    if multiplicity.kind != "single":
        assert summary["detections_per_detected_object"] > 1.0


def test_edge_losses_are_recorded_and_account_for_the_p_D_gap(tmp_path) -> None:
    """On the shipped scenario, truncated_ids is non-empty and closes the p_D gap. [B1, D11]

    (detected + lost at the edge) / visible is the raw coin-flip rate, so it should sit
    near the configured p_D, while detected / visible alone sits below it.
    """
    run = simulate_from_config(CONFIGS / "b1_two_rows.yaml", tmp_path)
    s = summarise(scan_counts(read_labels(run.labels)))
    assert s["n_truncated"] > 0
    coin_rate = (s["n_detected_objects"] + s["n_truncated"]) / s["n_visible"]
    assert s["p_D_empirical"] < coin_rate
    assert abs(coin_rate - 0.85) < 0.05
    assert "empirical p_D" in format_summary(scan_counts(read_labels(run.labels)))


def test_old_labels_without_truncated_ids_still_load(tmp_path) -> None:
    """labels.jsonl written before truncated_ids existed reads back with an empty tuple."""
    path = tmp_path / "labels.jsonl"
    path.write_text('{"k": 0, "origin": [null, 3], "visible_ids": [3, 4], '
                    '"detected_ids": [3]}\n', encoding="utf-8")
    (label,) = read_labels(path)
    assert label.truncated_ids == ()
    (c,) = scan_counts([label])
    assert (c.n_missed, c.n_truncated, c.n_clutter) == (1, 0, 1)


def test_analyse_draws_counts_for_a_simulate_only_run(tmp_path) -> None:
    """`analyse` works on a folder that holds only a simulation: scene and counts."""
    run = simulate_from_config(CONFIGS / "b1_two_rows.yaml", tmp_path)
    analyse_run(run)
    assert (run.plots / "scene.png").is_file()
    assert (run.plots / "counts.png").is_file()
    with pytest.raises(ValueError, match="r_vs_k"):
        analyse_run(run, plots=["r_vs_k"])


def test_b3_plots_refuse_a_multi_detection_scenario(tmp_path) -> None:
    """The A2 closed form assumes one detection per plant; analyse says so. [B3, D10]"""
    text = (CONFIGS / "b2_bernoulli_phantom.yaml").read_text(encoding="utf-8")
    scenario = CONFIGS / "b1_two_rows_duplicate.yaml"
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    run.config.write_text(text.replace("scenario: configs/b1_two_rows.yaml",
                                       f"scenario: {scenario}"), encoding="utf-8")
    with pytest.raises(ValueError, match="multiplicity"):
        analyse_run(run, plots=["r_vs_analytic"])
