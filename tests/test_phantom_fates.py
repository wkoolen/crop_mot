"""The controlled phantom of roadmap step 4a and its per-seed fates (decision D28). [B3]

The phantom is placed, not born from a detection, so every seed runs the same experiment;
its fate in each seed is that seed's one outcome.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.estimates_log import r_trajectory, read_estimates
from crop_mot.analysis.fates import (
    PhantomOutcome,
    classify_fate,
    fate_proportions,
    phantom_outcome,
    wilson_interval,
)
from crop_mot.config import BirthConfig, PruneConfig, RunConfig, load_run_config
from crop_mot.filters import build_filter
from crop_mot.filters.birth import InjectedPhantom
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter, track_from_config
from crop_mot.types import Pose2D, Scan

from conftest import CONFIGS, REPO_ROOT

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
    assert np.array_equal(birth.position, [1.94, 1.82])


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


def test_fates_by_hand() -> None:
    """D28: pruned first, then r_conf = 0.5, then the nearest object within d_match = 0.5 m."""
    assert classify_fate(True, 0.99, 0.01, None) == "pruned"
    assert classify_fate(False, 0.4, 0.01, None) == "alive_unconfirmed"
    assert classify_fate(False, 0.9, 0.3, 0.45) == "confirmed_on_plant"
    assert classify_fate(False, 0.9, 0.45, 0.3) == "confirmed_on_weed"
    assert classify_fate(False, 0.9, 0.6, None) == "sustained_by_clutter"
    assert classify_fate(False, 0.9, 0.3, 0.45, d_match=0.2) == "sustained_by_clutter"


def test_wilson_interval_by_hand() -> None:
    """Known values: 0 of 10 and 5 of 10; no information for n = 0."""
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_fate_proportions_split_by_a_weed_in_the_gate() -> None:
    def outcome(fate: str, n_weed: int) -> PhantomOutcome:
        return PhantomOutcome(fate=fate, k_end=59, r_end=0.9, d_plant=0.1, d_weed=None,
                              n_clutter_in_gate=3, n_weed_in_gate=n_weed)

    shares = fate_proportions([outcome("confirmed_on_weed", 4), outcome("pruned", 0),
                               outcome("confirmed_on_plant", 0)])
    assert shares["all"]["pruned"].count == 1 and shares["all"]["pruned"].n == 3
    assert shares["weed_in_gate"]["confirmed_on_weed"].count == 1
    assert shares["weed_in_gate"]["confirmed_on_weed"].n == 1
    assert shares["no_weed_in_gate"]["confirmed_on_plant"].n == 2


def test_seed_42_phantom_ends_on_a_plant(tmp_path) -> None:
    """The shipped experiment's own seed: a plant captures the phantom, no weed in its gate.

    In the field before D44 this seed's phantom locked onto a weed (D15); the weeds moved
    with the field.
    """
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        run = track_from_config(CONFIGS / "b2_phantom_fates.yaml", tmp_path)
    finally:
        os.chdir(cwd)
    outcome = phantom_outcome(run, load_run_config(run.config))
    assert outcome.fate == "confirmed_on_plant"
    assert outcome.n_weed_in_gate == 0 and outcome.d_plant < outcome.d_weed
