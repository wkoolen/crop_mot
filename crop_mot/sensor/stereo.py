"""Stereo depth to 3D measurements. NICE-TO-HAVE, phase 2+. STUB ONLY. [C1, deferred]

Nothing in B1-B4 imports this module. It sits at the back of the pipeline so that the 3D
route is written down rather than rediscovered later, and so the reason it is cheap is on
the record: models declare their own dim_x / dim_z, so going 3D means adding a
LinearGaussianXYZ next to LinearGaussianXY and a frustum next to the FOV wedge. The
filters, the runner, the detection format and the analysis do not change, because none of
them hard-code a dimension.

Open decision C1 on the roadmap (2D vs 3D) does not block anything in phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoDepthModel:
    """A stereo pair reporting 3D plant positions with range-dependent noise. [deferred]

    ASSUMPTION, when this is eventually implemented: depth comes from disparity, so depth
    error grows roughly as depth^2 * disparity_std / (baseline * focal). That makes the
    measurement covariance strongly anisotropic - much larger along the optical axis than
    across it - and range-dependent, which is why a constant R stops being defensible in 3D
    and why this cannot simply reuse LinearGaussianXY with a bigger R.

    Attributes:
        baseline: stereo baseline in metres.
        focal_px: focal length in pixels.
        disparity_std_px: standard deviation of the disparity estimate in pixels.
    """

    baseline: float
    focal_px: float
    disparity_std_px: float

    def depth_covariance(self, depth: float) -> np.ndarray:
        """Measurement covariance at a given depth, in the camera frame. [deferred]

        Args:
            depth: distance along the optical axis in metres.

        Returns:
            Shape (3, 3) covariance, with the large eigenvalue along the optical axis.
        """
        raise NotImplementedError
