"""The heading-error experiment of roadmap step 8d (decisions D16, D41). [B4]"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from crop_mot.config import (
    PlanConfig,
    YawSensitivityConfig,
    load_run_config,
    load_yaw_sensitivity_config,
)
from crop_mot.rng import substreams
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.yaw_sensitivity import perturbed, run_yaw_sensitivity
from crop_mot.sensor.detector import reexpress
from crop_mot.sensor.record import read_detections
from crop_mot.types import Pose2D
from crop_mot.world.path import generate_path

from conftest import CONFIGS


def test_the_yaw_config_parses() -> None:
    cfg = load_yaw_sensitivity_config(CONFIGS / "b4_yaw_sensitivity.yaml")
    assert cfg.yaw_bias_deg[0] == 0.0 and max(cfg.yaw_bias_deg) == 2.0
    assert cfg.seeds >= 1
    assert cfg.run.filter_cfg.plan is not None           # the known-N map
    assert load_run_config(CONFIGS / "b4_known_n_bank.yaml").scenario.path.yaw_bias == 0.0


def test_the_reported_pose_carries_the_bias(tiny_scenario) -> None:
    """No noise, bias b: the reported heading is off by exactly b; the position is not."""
    path = replace(tiny_scenario.path, pose_known=False, yaw_bias=0.02, yaw_wobble_std=0.0,
                   xy_noise_std=0.0)
    for sample in generate_path(path, substreams(1)["path"]):
        assert sample.reported.theta == pytest.approx(sample.true.theta + 0.02, abs=1e-15)
        assert (sample.reported.x, sample.reported.y) == (sample.true.x, sample.true.y)
    known = generate_path(tiny_scenario.path, substreams(1)["path"])
    assert all(s.reported == s.true for s in known)


def test_reexpress_is_rigid() -> None:
    true = Pose2D(x=0.0, y=0.0, theta=math.pi / 2)
    reported = Pose2D(x=0.1, y=-0.2, theta=math.pi / 2 + 0.05)
    a, b = np.array([0.5, 3.0]), np.array([-1.0, 2.0])
    ra, rb = reexpress(a, true, reported), reexpress(b, true, reported)
    assert np.linalg.norm(ra - rb) == pytest.approx(np.linalg.norm(a - b))
    assert np.linalg.norm(ra - [0.1, -0.2]) == pytest.approx(np.linalg.norm(a))
    assert np.array_equal(reexpress(a, true, true), a)


def test_a_biased_scan_is_the_known_scan_re_expressed(tiny_scenario, tmp_path) -> None:
    """Same seed, bias only: every z is the pose-known z seen through the reported pose."""
    biased = replace(tiny_scenario, path=replace(tiny_scenario.path, pose_known=False,
                                                 yaw_bias=math.radians(1.0),
                                                 yaw_wobble_std=0.0, xy_noise_std=0.0))
    runs = {}
    for name, scenario in (("known", tiny_scenario), ("biased", biased)):
        runs[name] = RunDir(tmp_path / name)
        runs[name].plots.mkdir(parents=True)
        simulate(scenario, runs[name], summary=False)
    for known, bias in zip(read_detections(runs["known"].detections),
                           read_detections(runs["biased"].detections)):
        assert bias.pose.theta == pytest.approx(known.pose.theta + math.radians(1.0))
        expected = [reexpress(d.z, known.pose, bias.pose) for d in known.detections]
        assert np.allclose([d.z for d in bias.detections], expected, atol=1e-12)
        assert [d.label for d in bias.detections] == [d.label for d in known.detections]


def test_the_sweep_writes_its_points_and_figure(tiny_run_config, tmp_run_dir) -> None:
    """Two biases, one seed, on a known-N tiny map; shapes only."""
    plan = PlanConfig(rows=tiny_run_config.scenario.world.rows, prior_std=0.036)
    run = replace(tiny_run_config, filter_cfg=replace(tiny_run_config.filter_cfg,
                                                      kind="bernoulli_bank", birth=None,
                                                      plan=plan))
    run = replace(run, scenario=replace(run.scenario,
                                        path=replace(run.scenario.path, n_scans=30)))
    cfg = YawSensitivityConfig(name="tiny_yaw", run=run, yaw_bias_deg=(0.0, 2.0),
                               yaw_wobble_std_deg=(0.0,), seeds=1)
    points = run_yaw_sensitivity(cfg, tmp_run_dir)
    assert [p.yaw_bias_deg for p in points] == [0.0, 2.0]
    assert all(len(p.mean_gospa) == 1 for p in points)
    assert (tmp_run_dir.plots / "yaw_sensitivity.png").stat().st_size > 0
    assert perturbed(run, 2.0, 0.0).scenario.path.yaw_bias == pytest.approx(math.radians(2.0))
