"""Motion models for the prediction step. [B2/B4]"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MotionModel(Protocol):
    """Target dynamics, used by every filter's `predict`. [B2/B4]

    Every implementation states its ASSUMPTION in its docstring, because that assumption is
    what a reader of the thesis needs in order to judge the result.

    Attributes:
        dim_x: dimension of the target state x.
    """

    dim_x: int

    def predict_moments(
        self, mean: np.ndarray, cov: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a Gaussian through the dynamics for dt seconds.

        This is the Chapman-Kolmogorov step specialised to a Gaussian: for a linear model
        m <- F m and P <- F P F' + Q.

        Args:
            mean: shape (dim_x,), prior mean.
            cov: shape (dim_x, dim_x), prior covariance.
            dt: time step in seconds.

        Returns:
            Tuple (predicted mean, predicted covariance) with the same shapes.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class StaticTarget(MotionModel):
    """Plants do not move. [B1/B2]

    ASSUMPTION: the target state is a fixed position, so F = I and the prediction is the
    identity. Q = q * dt * I with q >= 0; q is normally 0, and a small positive value is
    used only to stop the covariance collapsing to numerical zero over a long run, which
    would make the Kalman gain vanish and freeze the estimate.

    This is a real simplification worth defending in the thesis: plants grow, lean in wind
    and are re-observed from different angles, so a static point target is an idealisation
    of a rigid object with a stable centroid.

    Attributes:
        q: process noise density. 0.0 means a genuinely static target.
        dim_x: state dimension, 2 for xy in phase 1.
    """

    q: float = 0.0
    dim_x: int = 2
