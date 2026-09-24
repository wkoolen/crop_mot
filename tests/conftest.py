"""Shared pytest fixtures. [B1-B4]

Deliberately small. Fixtures build tiny scenarios - a handful of plants, a handful of scans
- because a test that takes a second to run does not get run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crop_mot.config import (
    AnalysisConfig,
    AssumedSensorConfig,
    BirthConfig,
    DetectionConfig,
    FilterConfig,
    GateConfig,
    MeasurementConfig,
    MonteCarloConfig,
    MotionConfig,
    PathConfig,
    RowConfig,
    RunConfig,
    ScenarioConfig,
    SensorConfig,
    SurvivalConfig,
    WorldConfig,
)
from crop_mot.runner.run_dir import RunDir
from crop_mot.types import FieldOfView

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def tiny_scenario() -> ScenarioConfig:
    """A deliberately small B1 scenario: one short row, a few scans.

    Small enough that a failing assertion can be reasoned about by hand, which is the whole
    point - a test over 60 scans and 70 plants tells you something is wrong but not what.

    Returns:
        A ScenarioConfig with a single row of a few plants and roughly five scans.

    Geometry: five plants at x = 0.6, y = 4.2 .. 5.6; the robot starts at the origin facing
    +y. At scan 0 the nearest plant is sqrt(0.6^2 + 4.2^2) = 4.24 m away, beyond
    max_range = 4.0, so EVERY scan-0 detection is clutter by construction - the phantom
    birth in `tiny_run_config` needs no peeking at labels. Plants come into view from about
    scan 3. Seed 3 gives three clutter detections at scan 0.
    """
    return ScenarioConfig(
        name="tiny",
        seed=3,
        world=WorldConfig(
            rows=(RowConfig(x=0.6, y_start=4.2, y_end=5.6, spacing=0.35),),
            position_jitter_std=0.03,
        ),
        path=PathConfig(
            kind="straight_lane", x=0.0, y_start=0.0, heading=np.pi / 2, speed=0.4,
            n_scans=8, scan_period=0.25, pose_known=True, yaw_wobble_std=0.02,
            xy_noise_std=0.01,
        ),
        sensor=SensorConfig(
            fov=FieldOfView(min_range=0.3, max_range=4.0, half_angle=0.6),
            detection=DetectionConfig(kind="constant", p_D=0.85),
            lambda_FA=2.0,
            measurement=MeasurementConfig(kind="linear_xy", R=0.04 * np.eye(2)),
        ),
    )


@pytest.fixture
def tiny_run_config(tiny_scenario: ScenarioConfig) -> RunConfig:
    """A B2 run config over `tiny_scenario`, with a Bernoulli filter and a phantom birth.

    Args:
        tiny_scenario: the scenario fixture.

    Returns:
        A RunConfig ready to hand to the filter runner.

    The filter's assumed sensor equals the truth sensor (no model mismatch), and the birth
    seeds from detection 0 of scan 0 - clutter by the geometry of `tiny_scenario`.
    """
    sensor = tiny_scenario.sensor
    return RunConfig(
        name="tiny_phantom",
        seed=tiny_scenario.seed,
        scenario=tiny_scenario,
        filter_cfg=FilterConfig(
            kind="bernoulli",
            motion=MotionConfig(kind="static", q=0.0),
            measurement=sensor.measurement,
            assumed_sensor=AssumedSensorConfig(fov=sensor.fov, detection=sensor.detection,
                                               lambda_FA=sensor.lambda_FA),
            birth=BirthConfig(kind="single_from_measurement", at_scan=0, detection_index=0,
                              r_b=0.5, init_cov=0.05 * np.eye(2)),
            survival=SurvivalConfig(p_S=1.0),
            gate=GateConfig(chi2_prob=0.99),
        ),
        analysis=AnalysisConfig(b3_reference="bernoulli_existence",
                                monte_carlo=MonteCarloConfig(n_runs=50),
                                plots=("r_vs_k",)),
    )


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> RunDir:
    """An empty run folder under pytest's tmp_path.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A RunDir whose subdirectories already exist.
    """
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    return run
