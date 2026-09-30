"""The controlled phantom of roadmap step 4a and its per-seed fates (decision D28). [B3]

The phantom is placed, not born from a detection, so every seed runs the same experiment;
its fate in each seed is that seed's one outcome.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from crop_mot.analysis.estimates_log import r_trajectory, read_estimates
from crop_mot.config import BirthConfig, PruneConfig, RunConfig, load_run_config
from crop_mot.filters import build_filter
from crop_mot.filters.birth import InjectedPhantom
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter
from crop_mot.types import Pose2D, Scan

from conftest import CONFIGS

POSITION = np.array([0.6, 3.0])


def _injected(cfg: RunConfig, kind: str) -> RunConfig:
    birth = BirthConfig(kind="injected", at_scan=5, detection_index=-1, r_b=0.3,
                        init_cov=0.05 * np.eye(2), position=POSITION)
    return replace(cfg, filter_cfg=replace(cfg.filter_cfg, kind=kind, birth=birth,
                                           prune=PruneConfig()))


def test_injected_phantom_is_placed_at_its_scan_only() -> None:
    phantom = InjectedPhantom(at_scan=2, position=POSITION, r_b=0.08, init_cov=np.eye(2))
    pose = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)
    assert phantom.birth_components(Scan(k=1, t=0.25, pose=pose, detections=())) == []
    ((r_b, mean, cov),) = phantom.birth_components(Scan(k=2, t=0.5, pose=pose, detections=()))
    assert r_b == 0.08 and np.array_equal(mean, POSITION) and np.array_equal(cov, np.eye(2))


def test_the_shipped_fates_config_injects_d28s_phantom() -> None:
    birth = load_run_config(CONFIGS / "b2_phantom_fates.yaml").filter_cfg.birth
    assert birth.kind == "injected" and birth.detection_index == -1
    assert birth.at_scan == 24 and birth.r_b == 0.08
    assert np.array_equal(birth.position, [1.94, 4.82])


def test_an_injected_bank_is_the_injected_single_filter(
    tiny_run_config: RunConfig, tmp_path
) -> None:
    """Without pruning, the bank and the single filter give the same r, as for seeds (D12)."""
    cfg = replace(tiny_run_config, scenario=replace(
        tiny_run_config.scenario, path=replace(tiny_run_config.scenario.path, n_scans=20)))
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    simulate(cfg.scenario, run)
    for kind in ("bernoulli", "bernoulli_bank"):
        run_filter(build_filter(_injected(cfg, kind).filter_cfg), run)

    r_single = r_trajectory(read_estimates(run.estimates("bernoulli")), 0)
    r_bank = r_trajectory(read_estimates(run.estimates("bernoulli_bank")), 0)
    assert r_single[4] == 0.0 and r_single[5] == 0.3
    assert np.array_equal(r_single, r_bank)
