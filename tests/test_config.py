"""The config loader is strict: typos fail loudly instead of silently using a default. [B1/B2]

Added during implementation. `crop_mot.config` promises that writing `p_d` for `p_D` is an
error, because a silently ignored key is invisible in a plot. These tests hold it to that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crop_mot.config import load_run_config, load_scenario_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


def test_shipped_configs_load() -> None:
    """Every config in configs/ parses, and the run config follows its scenario key."""
    scenario = load_scenario_config(CONFIGS / "b1_two_rows.yaml")
    assert scenario.sensor.detection.kind == "constant"

    assert scenario.sensor.multiplicity.kind == "single"
    assert load_scenario_config(CONFIGS / "b1_two_rows_duplicate.yaml").sensor.multiplicity.kind \
        == "duplicate"
    assert load_scenario_config(CONFIGS / "b1_two_rows_extended.yaml").sensor.multiplicity.kind \
        == "poisson"

    run = load_run_config(CONFIGS / "b2_bernoulli_phantom.yaml")
    assert run.scenario.name == scenario.name
    assert run.filter_cfg.kind == "bernoulli"
    assert run.filter_cfg.collapse == "best_branch"


def test_unknown_key_is_an_error(tmp_path) -> None:
    """`p_d` instead of `p_D` must raise, naming the offending key."""
    text = (CONFIGS / "b1_two_rows.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("p_D: 0.85", "p_d: 0.85"), encoding="utf-8")
    with pytest.raises(ValueError, match="p_d"):
        load_scenario_config(bad)


def test_constant_detection_without_p_D_is_an_error(tmp_path) -> None:
    """kind: constant with no p_D is an inconsistent combination."""
    text = (CONFIGS / "b1_two_rows.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("    p_D: 0.85\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="p_D"):
        load_scenario_config(bad)


def test_missing_multiplicity_block_defaults_to_single(tmp_path) -> None:
    """The block is optional; without it a scenario behaves as before multiplicity existed."""
    text = (CONFIGS / "b1_two_rows.yaml").read_text(encoding="utf-8")
    trimmed = tmp_path / "trimmed.yaml"
    trimmed.write_text(text.replace("  multiplicity:\n    kind: single\n", ""), encoding="utf-8")
    assert load_scenario_config(trimmed).sensor.multiplicity.kind == "single"


def test_inconsistent_multiplicity_is_an_error(tmp_path) -> None:
    """poisson without gamma, an unknown key, and poisson with a range-dependent p_D."""
    text = (CONFIGS / "b1_two_rows_extended.yaml").read_text(encoding="utf-8")
    cases = {
        "gamma": text.replace("    gamma: 3.0", "    # gamma: 3.0"),
        "p_split": text.replace("    gamma: 3.0", "    gamma: 3.0\n    p_split: 0.3"),
        "range_dependent": text.replace(
            "    kind: constant\n    p_D: 0.85",
            "    kind: range_dependent\n    p_D_near: 0.95\n    p_D_far: 0.55"),
    }
    for match, bad_text in cases.items():
        assert bad_text != text, match
        bad = tmp_path / f"bad_{match}.yaml"
        bad.write_text(bad_text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_scenario_config(bad)


def test_p_D_evaluation_key_is_optional(tmp_path) -> None:
    """`filter.p_D_evaluation` defaults to at_mean and parses when given (roadmap step 3b)."""
    text = (CONFIGS / "b2_bernoulli_phantom.yaml").read_text(encoding="utf-8")
    assert load_run_config(CONFIGS / "b2_bernoulli_phantom.yaml").filter_cfg.p_D_evaluation \
        == "at_mean"
    given = tmp_path / "given.yaml"
    given.write_text(text.replace("  collapse: best_branch\n",
                                  "  collapse: best_branch\n  p_D_evaluation: expected\n"),
                     encoding="utf-8")
    assert load_run_config(given).filter_cfg.p_D_evaluation == "expected"
