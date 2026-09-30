"""The p_D evaluation strategies of roadmap step 3b (decision D27). [B2]

Option A wraps what the filter has always done, so it is checked against the sensor model
directly; the stubs must refuse to be built, so a config cannot select one silently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.config import RunConfig
from crop_mot.filters import build_filter
from crop_mot.filters.detection_prob import PD_EVALUATIONS, AtMean
from crop_mot.sensor.models import build_measurement_model
from crop_mot.sensor.sensor_model import build_sensor_model
from crop_mot.types import Pose2D

POSE = Pose2D(x=0.0, y=0.0, theta=np.pi / 2)


def test_at_mean_is_p_D_at_the_mean_with_the_moments_unchanged(
    tiny_run_config: RunConfig,
) -> None:
    """Option A: both p_D values are p_D(m), 0 out of view; the missed branch is N(m, P)."""
    cfg = tiny_run_config.filter_cfg
    measurement = build_measurement_model(cfg.measurement)
    assumed = cfg.assumed_sensor
    sensor = build_sensor_model(assumed.fov, assumed.detection, assumed.lambda_FA, measurement)
    evaluation = AtMean()
    cov = 0.05 * np.eye(2)

    in_view, out_of_view = np.array([0.0, 2.0]), np.array([0.0, -2.0])
    assert evaluation.miss_p_D(sensor, in_view, cov, POSE) == 0.85
    assert evaluation.detection_p_D(sensor, in_view, cov, POSE, np.array([0.1, 2.1])) == 0.85
    assert evaluation.miss_p_D(sensor, out_of_view, cov, POSE) == 0.0
    mean, missed_cov = evaluation.missed_moments(sensor, in_view, cov, POSE)
    assert np.array_equal(mean, in_view) and np.array_equal(missed_cov, cov)


def test_the_default_is_at_mean(tiny_run_config: RunConfig) -> None:
    assert tiny_run_config.filter_cfg.p_D_evaluation == "at_mean"
    assert isinstance(build_filter(tiny_run_config.filter_cfg).p_D_evaluation, AtMean)


@pytest.mark.parametrize("name", ["at_branch_mean", "expected", "expected_with_shift"])
def test_options_b_to_d_are_stubs(tiny_run_config: RunConfig, name: str) -> None:
    """B, C and D raise at construction, naming what they wait for."""
    with pytest.raises(NotImplementedError, match="waits on"):
        PD_EVALUATIONS[name]()
    with pytest.raises(NotImplementedError, match="waits on"):
        build_filter(replace(tiny_run_config.filter_cfg, p_D_evaluation=name))


def test_an_unknown_evaluation_is_an_error(tiny_run_config: RunConfig) -> None:
    with pytest.raises(ValueError, match="p_D_evaluation"):
        build_filter(replace(tiny_run_config.filter_cfg, p_D_evaluation="exact"))
