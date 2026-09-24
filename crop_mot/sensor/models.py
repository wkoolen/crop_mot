"""Measurement models: given that a detection happened, what does z look like? [B1/B2]

The Protocol exists so that a range-bearing or stereo-3D model is a later FILE rather than
a later refactor. Phase 1 ships only the linear world-frame model, because keeping
g(z|x) linear-Gaussian is what lets the Bernoulli, PDA and JPDA algebra in the thesis be
written without Jacobian approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from crop_mot.config import MeasurementConfig
from crop_mot.types import Pose2D


class MeasurementModel(Protocol):
    """Maps a target state to measurement space. [B1/B2/B4]

    Every implementation must state its modelling ASSUMPTION in its docstring - that is the
    text that ends up justifying a step in the thesis derivation.

    Attributes:
        dim_x: dimension of the target state x.
        dim_z: dimension of a measurement z.
        R: measurement noise covariance, shape (dim_z, dim_z).
    """

    dim_x: int
    dim_z: int
    R: np.ndarray

    def h(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Noise-free predicted measurement of a target at x, seen from pose.

        Args:
            x: shape (dim_x,), target state.
            pose: the REPORTED pose when called by a filter.

        Returns:
            Shape (dim_z,), the expected measurement.
        """
        raise NotImplementedError

    def H(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Jacobian dh/dx evaluated at x.

        Constant for linear models, in which case x is ignored. Filters call this rather
        than assuming H = I, which is what keeps a nonlinear model a drop-in replacement.

        Args:
            x: shape (dim_x,), linearisation point.
            pose: the pose used for the transform.

        Returns:
            Shape (dim_z, dim_x).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class LinearGaussianXY(MeasurementModel):
    """Position measurement in world coordinates. [B1/B2]

    ASSUMPTION: the black-box detector reports a plant's position in WORLD xy, having
    already un-projected the image detection through the known robot pose. Under that
    assumption

        z = H x + v,    H = I_2,    v ~ N(0, R)

    so the likelihood g(z|x) = N(z; x, R) is exactly linear-Gaussian and every filter update
    is a plain Kalman update with no linearisation error.

    What this trades away, worth stating in the thesis: a real camera's noise grows with
    range and is anisotropic (much worse along the optical axis than across it), so a
    constant R is optimistic. The honest version is the range-bearing or stereo model -
    see `crop_mot.sensor.stereo` - which is why this class implements a Protocol rather
    than being hard-coded into the filters.

    Attributes:
        R: shape (2, 2) measurement noise covariance in m^2.
    """

    R: np.ndarray
    dim_x: int = 2
    dim_z: int = 2

    def h(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """z_hat = H x; the pose is not needed because z is already in world xy."""
        return self.H(x, pose) @ x

    def H(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """H = I_2, independent of x and pose."""
        return np.eye(self.dim_z, self.dim_x)


def build_measurement_model(cfg: MeasurementConfig) -> MeasurementModel:
    """Construct the measurement model named by cfg.kind. [B1/B2]

    Used by the simulator (truth `sensor.measurement`), by the filter (`filter.measurement`)
    and by the B3 analysis, so all three build the model the same way.

    Args:
        cfg: a `measurement:` config block.

    Returns:
        The measurement model.

    Raises:
        ValueError: if cfg.kind is unknown.
    """
    if cfg.kind == "linear_xy":
        return LinearGaussianXY(R=cfg.R)
    raise ValueError(f"unknown measurement kind {cfg.kind!r}; phase 1 has only 'linear_xy'")
