"""The animated existence map: one frame per scan, for any filter (decision D42). [B4]"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.runner.analyse import analyse_run
from crop_mot.runner.track import track_from_config

from conftest import CONFIGS, REPO_ROOT


@pytest.mark.slow
@pytest.mark.parametrize("config", ["b4_known_n_bank_weeds.yaml", "b4_bounded_n_bank.yaml",
                                    "b2_bernoulli_bank_weeds.yaml"])
def test_existence_map_animation_is_written(config: str, tmp_path) -> None:
    """One GIF frame per scan, on known N with weeds, bounded N and a phantom bank."""
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        run = track_from_config(CONFIGS / config, tmp_path)
    finally:
        os.chdir(cwd)
    analyse_run(run, plots=["existence_anim"])
    n_scans = len(read_estimates(run.estimates("bernoulli_bank")))
    with Image.open(run.plots / "existence_map.gif") as gif:
        assert gif.n_frames == n_scans
