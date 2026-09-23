"""The detection process: p_D, lambda_FA and the clutter density c(z). [B1/B2/B3/B4]

One Protocol serves BOTH sides of the pipeline:
  * the simulator uses it to decide what to generate;
  * the filter uses it to evaluate likelihoods.
The filter's instance is built from `filter.assumed_sensor` and may deliberately differ
from the simulator's - that is how model mismatch becomes a config edit rather than a code
change.

Note the Protocol contains no sampling methods. Drawing detections lives in `detector.py`
as free functions, so a filter physically cannot generate data even by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from crop_mot.sensor.models import MeasurementModel
from crop_mot.types import FieldOfView, Pose2D


class SensorModel(Protocol):
    """Detection probability, clutter rate and clutter density. [B1/B2/B4]

    Attributes:
        fov: the wedge outside which p_D is 0 and clutter never appears.
        measurement: the measurement model, so a filter needs only one injected object.
    """

    fov: FieldOfView
    measurement: MeasurementModel

    def p_D(self, x: np.ndarray, pose: Pose2D) -> float:
        """Detection probability for a target at x seen from pose.

        Must return exactly 0.0 outside the FOV. That is not a detail: the misdetection
        branch of the Bernoulli update behaves differently for "in view but missed" and
        "not in view at all", and B3's ScanEvent records `in_fov` for precisely that reason.

        Args:
            x: shape (dim_x,), target state in world coordinates.
            pose: viewing pose (TRUE for the simulator, REPORTED for a filter).

        Returns:
            p_D in [0, 1].
        """
        raise NotImplementedError

    def lambda_FA(self, pose: Pose2D) -> float:
        """Expected number of clutter detections this scan (the Poisson mean).

        Takes a pose so that a pose-dependent clutter rate (e.g. more weeds in one part of
        the field) is possible later without changing any caller.

        Args:
            pose: viewing pose.

        Returns:
            lambda_FA >= 0.
        """
        raise NotImplementedError

    def clutter_density(self, z: np.ndarray, pose: Pose2D) -> float:
        """Clutter density c(z) at measurement z.

        For clutter distributed uniformly over the FOV this is lambda_FA / fov.area()
        inside the wedge and 0 outside. It appears in the denominator of the Bernoulli and
        PDA association weights, which is why it is a density (per m^2), not a probability.

        Args:
            z: shape (dim_z,), a measurement in world coordinates.
            pose: viewing pose.

        Returns:
            c(z) >= 0, in units of 1/m^2 for a 2D measurement.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantPD(SensorModel):
    """Uniform detection probability inside the FOV. [B1/B2/B3]

    ASSUMPTION: p_D is the same everywhere inside the wedge and 0 outside, and clutter is
    uniform over the wedge. This is the assumption the A2 closed-form existence recursion is
    derived under, so it is the default for B2 and the primary B3 cross-check.

    Attributes:
        fov: the wedge.
        measurement: the measurement model.
        p_D_const: the constant detection probability inside the FOV.
        lambda_FA_const: the constant expected clutter count per scan.
    """

    fov: FieldOfView
    measurement: MeasurementModel
    p_D_const: float
    lambda_FA_const: float


@dataclass(frozen=True)
class RangeDependentPD(SensorModel):
    """Detection probability that falls off with range, with optional occlusion. [B1/B3]

    ASSUMPTION: p_D interpolates between p_D_near at fov.min_range and p_D_far at
    fov.max_range, and is multiplied by occlusion_factor when a nearer plant in the same row
    shadows this one. This models the "occluded / missed" plants in the scenario sketch,
    which a constant p_D folds into an average and therefore cannot reproduce.

    This BREAKS the constant-p_D assumption the textbook existence recursion is written
    under - which is exactly why `crop_mot.analysis.analytic.ScanEvent` carries p_D per scan
    instead of the analytic reference storing a single constant. Both profiles are then
    validated by the same closed form, and the constant case is simply the one where every
    ScanEvent holds the same value.

    Attributes:
        fov: the wedge.
        measurement: the measurement model.
        p_D_near: detection probability at fov.min_range.
        p_D_far: detection probability at fov.max_range.
        occlusion_factor: multiplier applied when shadowed by a nearer plant in the same
            row; 1.0 disables occlusion modelling.
    """

    fov: FieldOfView
    measurement: MeasurementModel
    p_D_near: float
    p_D_far: float
    lambda_FA_const: float
    occlusion_factor: float = 1.0
