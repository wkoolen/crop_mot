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

from crop_mot.config import DetectionConfig
from crop_mot.sensor.fov import in_fov
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
        """Clutter density c(z) at measurement z: the SPATIAL pdf of one clutter return.

        For clutter distributed uniformly over the FOV this is 1 / fov.area() inside the
        wedge and 0 outside, so it integrates to 1 over the FOV. The clutter INTENSITY is
        lambda_FA * c(z) - A0's lambda_c(z) = lambda-bar_c f_c(z), A2's lambda_FA(z)
        [A0 §Measurement model]. The intensity is what appears in the Bernoulli and PDA
        association weights, which is why c(z) is a density (per m^2), not a probability.

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

    def p_D(self, x: np.ndarray, pose: Pose2D) -> float:
        """p_D_const inside the FOV, exactly 0.0 outside."""
        if not in_fov(x, pose, self.fov):
            return 0.0
        return self.p_D_const

    def lambda_FA(self, pose: Pose2D) -> float:
        """The constant expected clutter count per scan."""
        return self.lambda_FA_const

    def clutter_density(self, z: np.ndarray, pose: Pose2D) -> float:
        """c(z) = 1 / fov.area() inside the FOV, 0 outside [A0 §Measurement model]."""
        if not in_fov(z, pose, self.fov):
            return 0.0
        return 1.0 / self.fov.area()


@dataclass(frozen=True)
class RangeDependentPD(SensorModel):
    """Detection probability that falls off with range, with optional occlusion. [B1/B3]

    ASSUMPTION: p_D interpolates LINEARLY in range rho between p_D_near at fov.min_range
    and p_D_far at fov.max_range,

        p_D(x) = p_D_near + (p_D_far - p_D_near) * (rho - min_range) / (max_range - min_range)

    inside the FOV, and is exactly 0 outside. This models the plants that are missed more
    often further away, which a constant p_D folds into an average and cannot reproduce.

    OCCLUSION IS NOT IMPLEMENTED (extension slot): multiplying by occlusion_factor "when a
    nearer plant in the same row shadows this one" needs the other plants' positions, which
    p_D(x, pose) does not receive - and the filter side could not evaluate it without
    truth. occlusion_factor != 1.0 therefore raises NotImplementedError at construction.

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
        lambda_FA_const: the constant expected clutter count per scan.
        occlusion_factor: multiplier applied when shadowed by a nearer plant in the same
            row; 1.0 disables occlusion modelling, and is the only value supported in
            phase 1.
    """

    fov: FieldOfView
    measurement: MeasurementModel
    p_D_near: float
    p_D_far: float
    lambda_FA_const: float
    occlusion_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.occlusion_factor != 1.0:
            raise NotImplementedError(
                "RangeDependentPD occlusion is an extension slot: p_D(x, pose) cannot see "
                "the other plants. Use occlusion_factor: 1.0 in phase 1."
            )

    def p_D(self, x: np.ndarray, pose: Pose2D) -> float:
        """Linear in range from p_D_near to p_D_far inside the FOV, exactly 0.0 outside."""
        if not in_fov(x, pose, self.fov):
            return 0.0
        rho = float(np.hypot(x[0] - pose.x, x[1] - pose.y))
        fraction = (rho - self.fov.min_range) / (self.fov.max_range - self.fov.min_range)
        return self.p_D_near + (self.p_D_far - self.p_D_near) * fraction

    def lambda_FA(self, pose: Pose2D) -> float:
        """The constant expected clutter count per scan."""
        return self.lambda_FA_const

    def clutter_density(self, z: np.ndarray, pose: Pose2D) -> float:
        """c(z) = 1 / fov.area() inside the FOV, 0 outside [A0 §Measurement model]."""
        if not in_fov(z, pose, self.fov):
            return 0.0
        return 1.0 / self.fov.area()


def build_sensor_model(
    fov: FieldOfView, detection: DetectionConfig, lambda_FA: float,
    measurement: MeasurementModel,
) -> SensorModel:
    """Construct the SensorModel selected by detection.kind. [B1/B2/B3]

    The one place a detection config becomes a SensorModel, used for the simulator's TRUTH
    sensor, the filter's ASSUMED sensor and the B3 event builder - so the filter and the
    analysis evaluate p_D and c(z) with identical code.

    Args:
        fov: the wedge.
        detection: the `detection:` config block.
        lambda_FA: expected clutter count per scan.
        measurement: the measurement model.

    Returns:
        A ConstantPD or RangeDependentPD.

    Raises:
        ValueError: if detection.kind is unknown.
    """
    if detection.kind == "constant":
        return ConstantPD(fov=fov, measurement=measurement, p_D_const=detection.p_D,
                          lambda_FA_const=lambda_FA)
    if detection.kind == "range_dependent":
        return RangeDependentPD(fov=fov, measurement=measurement,
                                p_D_near=detection.p_D_near, p_D_far=detection.p_D_far,
                                lambda_FA_const=lambda_FA,
                                occlusion_factor=detection.occlusion_factor)
    raise ValueError(f"unknown detection kind {detection.kind!r}")
