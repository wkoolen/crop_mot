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
    """Both configs in configs/ parse, and the run config follows its scenario key."""
    scenario = load_scenario_config(CONFIGS / "b1_two_rows.yaml")
    assert scenario.sensor.detection.kind == "constant"

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
