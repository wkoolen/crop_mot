"""Shared Gaussian algebra used by every filter. [B2/B4]

Factored out so that Bernoulli, PDA, JPDA, GNN, PMB and PMBM all do the SAME linear algebra
rather than six slightly different versions of it. When a B4 filter disagrees with B2 about
a number, this file is the thing that should not be suspected.

All functions are pure and take explicit moments; nothing here knows about existence
probabilities, association or gating.
"""

from __future__ import annotations

import numpy as np

from crop_mot.motion.models import MotionModel
from crop_mot.sensor.models import MeasurementModel
from crop_mot.types import Pose2D


def kf_predict(
    mean: np.ndarray, cov: np.ndarray, motion: MotionModel, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman prediction step.

    A thin wrapper over MotionModel.predict_moments, kept so that every filter's `predict`
    reads the same way.

    Serves: [B2] the Bernoulli time update; [B4] all multi-target filters.

    Args:
        mean: shape (dim_x,), prior mean.
        cov: shape (dim_x, dim_x), prior covariance.
        motion: the injected motion model.
        dt: time step in seconds.

    Returns:
        Tuple (predicted mean, predicted covariance).
    """
    raise NotImplementedError


def predicted_measurement(
    mean: np.ndarray, cov: np.ndarray, model: MeasurementModel, pose: Pose2D
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted measurement and its innovation covariance.

    Returns z_hat = h(mean, pose) and S = H P H' + R. S is needed in three places - the
    likelihood, the Kalman gain and the gate - so computing it once and passing it around
    avoids three chances to disagree.

    Serves: [B2] the Bernoulli update; [B4] gating and association weights.

    Args:
        mean: shape (dim_x,), predicted state mean.
        cov: shape (dim_x, dim_x), predicted state covariance.
        model: the injected measurement model.
        pose: the REPORTED pose from the scan.

    Returns:
        Tuple (z_hat of shape (dim_z,), S of shape (dim_z, dim_z)).
    """
    raise NotImplementedError


def kf_update(
    mean: np.ndarray,
    cov: np.ndarray,
    z: np.ndarray,
    model: MeasurementModel,
    pose: Pose2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman measurement update for a single measurement.

    Should use the Joseph form for the covariance update, or at minimum symmetrise the
    result. With a static target and p_S = 1 the covariance shrinks monotonically over a
    long run, and an asymmetric P accumulating floating-point error is the classic way a
    long tracking run silently stops being positive definite.

    Serves: [B2] the detection branch of the Bernoulli update; [B4] every filter.

    Args:
        mean: shape (dim_x,), predicted mean.
        cov: shape (dim_x, dim_x), predicted covariance.
        z: shape (dim_z,), the measurement.
        model: the injected measurement model.
        pose: the REPORTED pose.

    Returns:
        Tuple (posterior mean, posterior covariance).
    """
    raise NotImplementedError


def log_predicted_likelihood(
    z: np.ndarray, z_hat: np.ndarray, S: np.ndarray
) -> float:
    """Log of the predicted measurement likelihood, log N(z; z_hat, S).

    This is the quantity the thesis writes as g(z|x) marginalised over the predicted state -
    it is what appears in the Bernoulli and PDA association weights, NOT the likelihood at
    a point estimate. Computed in log space because with several targets and several
    measurements the products underflow quickly.

    Serves: [B2] the Bernoulli update weights; [B4] association weights and cost matrices.

    Args:
        z: shape (dim_z,), the measurement.
        z_hat: shape (dim_z,), the predicted measurement.
        S: shape (dim_z, dim_z), the innovation covariance.

    Returns:
        The natural log of the Gaussian density evaluated at z.
    """
    raise NotImplementedError
