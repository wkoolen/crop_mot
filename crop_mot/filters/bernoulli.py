"""The Bernoulli filter: one target with an existence probability r. [B2]

The only filter implemented in phase 1, and the one B3 validates against the hand-derived
A2 recursion.

A Bernoulli density is a single target that may or may not exist:
    p(X) = 1 - r            if X is empty
    p(X) = r * N(x; m, P)   if X = {x}
so the filter carries a scalar r alongside a Gaussian. The prediction and update of r are
what B3 checks against the closed form; the Gaussian part is a standard Kalman filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.config import FilterConfig
from crop_mot.filters.base import BirthModel, SurvivalModel, TrackingFilter
from crop_mot.motion.models import MotionModel
from crop_mot.sensor.models import MeasurementModel
from crop_mot.sensor.sensor_model import SensorModel
from crop_mot.types import Scan, TrackEstimate


@dataclass(frozen=True)
class BernoulliState:
    """The filter's state: existence probability plus a Gaussian. [B2]

    This is the type parameter S for the Bernoulli filter. The runner never inspects it -
    it only passes it back into predict/update and hands it to extract.

    Attributes:
        r: existence probability in [0, 1].
        mean: shape (dim_x,), the spatial density's mean, meaningful only when r > 0.
        cov: shape (dim_x, dim_x), the spatial density's covariance.
        track_id: identifier carried into the TrackEstimate so a trajectory can be
            reconstructed from the estimates log.
    """

    r: float
    mean: np.ndarray
    cov: np.ndarray
    track_id: int


@dataclass(frozen=True)
class BernoulliFilter(TrackingFilter[BernoulliState]):
    """Single-target Bernoulli filter over recorded detections. [B2]

    Satisfies the common TrackingFilter interface with N = 1: `extract` returns a list of
    length 0 or 1, so the runner treats it exactly like PMBM.

    All model knowledge is injected, and all of it comes from `filter.assumed_sensor` and
    the sibling config blocks - never from the simulator. The filter therefore cannot tell
    whether the detection it is tracking came from a plant or from clutter, which is the
    whole point of the phantom experiment.

    Attributes:
        motion: target dynamics, StaticTarget in phase 1.
        measurement: the measurement model used for g(z|x).
        sensor: the ASSUMED detection process, supplying p_D, lambda_FA and c(z).
        birth: the birth model.
        survival: p_S.
        gate_chi2: squared-Mahalanobis gate threshold, precomputed from the configured gate
            probability and dim_z.
        name: "bernoulli".
    """

    motion: MotionModel
    measurement: MeasurementModel
    sensor: SensorModel
    birth: BirthModel
    survival: SurvivalModel
    gate_chi2: float
    name: str = "bernoulli"

    def initial_state(self) -> BernoulliState:
        """Prior with r = 0: nothing exists until the birth model creates it. [B2]

        Returns:
            A BernoulliState with r = 0.0 and a placeholder Gaussian.
        """
        raise NotImplementedError

    def predict(self, state: BernoulliState, dt: float) -> BernoulliState:
        """Time update. [B2]

        Two independent things happen:
          * existence: r <- p_S * r. With p_S = 1 for permanent plants this leaves r
            unchanged, so ALL decay in the B2 plot comes from the measurement update.
          * spatial: the Gaussian is propagated through the motion model, which is the
            identity for a static target.

        Args:
            state: current state.
            dt: time step in seconds.

        Returns:
            A new predicted state; `state` is not modified.
        """
        raise NotImplementedError

    def update(self, state: BernoulliState, scan: Scan) -> BernoulliState:
        """Measurement update for one scan, including births. [B2]

        The structure the A2 derivation follows, and which B3 checks:
          * MISDETECTION branch: no gated detection is assigned to the target. The
            existence probability shrinks because a target that exists would have been
            detected with probability p_D. This branch is why r decays for a phantom.
          * DETECTION branch: each gated detection contributes a weight proportional to
            r * p_D * g(z|x), competing against the clutter explanation lambda_FA * c(z).
            The Gaussian is updated with the weighted measurement.
          * BIRTH: components from the birth model are merged in.

        p_D is evaluated through the ASSUMED sensor model at the predicted mean, using the
        scan's reported pose. It is 0 outside the FOV, which makes "out of view" behave
        differently from "in view but missed" - a distinction B3's ScanEvent records.

        Args:
            state: the predicted state.
            scan: this scan's detections and reported pose.

        Returns:
            A new posterior state; `state` is not modified.
        """
        raise NotImplementedError

    def extract(self, state: BernoulliState) -> list[TrackEstimate]:
        """Report the track if it is probably there. [B2]

        For a Bernoulli filter this is a thresholding decision on r. The threshold is a
        REPORTING choice, not part of the Bayes recursion - B3 compares the raw r
        trajectory, not the thresholded one, so this must not alter r.

        Args:
            state: the current state.

        Returns:
            A one-element list when the track is reported, empty otherwise.
        """
        raise NotImplementedError


def build_bernoulli(cfg: FilterConfig) -> BernoulliFilter:
    """Construct a BernoulliFilter from its config block. [B2]

    This is the builder registered in `crop_mot.filters.FILTERS`. Every B4 filter gets an
    equivalent function with the same signature, which is what keeps the runner free of
    filter-specific construction logic.

    Args:
        cfg: the `filter:` block of a run config.

    Returns:
        A ready-to-run BernoulliFilter with all models injected.

    Raises:
        ValueError: if cfg.kind is not "bernoulli", or a referenced model kind is unknown.
    """
    raise NotImplementedError
