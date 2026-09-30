"""The heading-error experiment of roadmap step 8d (decisions D16, D41). [B4]"""

from __future__ import annotations

from crop_mot.config import load_run_config, load_yaw_sensitivity_config

from conftest import CONFIGS


def test_the_yaw_config_parses() -> None:
    cfg = load_yaw_sensitivity_config(CONFIGS / "b4_yaw_sensitivity.yaml")
    assert cfg.yaw_bias_deg[0] == 0.0 and max(cfg.yaw_bias_deg) == 2.0
    assert cfg.seeds >= 1
    assert cfg.run.filter_cfg.plan is not None           # the known-N map
    assert load_run_config(CONFIGS / "b4_known_n_bank.yaml").scenario.path.yaw_bias == 0.0
