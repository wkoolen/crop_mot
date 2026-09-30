"""How a Gaussian track's detection probability is evaluated. [B2, roadmap step 3b]

p_D(x) depends on the state - it is exactly 0 outside the FOV - and A2 derives the Bernoulli
update with the exact forms (decision D27):
    miss weight      1 - p_D_bar,  p_D_bar = integral p_D(x) N(x; m, P) dx     [A2 §2, §2.1]
    detection        ell(z) = integral p_D(x) g(z|x) N(x; m, P) dx              [A2 §3.1]
    missed density   (1 - p_D(x)) N(x; m, P) / (1 - p_D_bar)                    [A2 §2]
A filter that carries one Gaussian needs an approximation of each. The options are the rows
of the roadmap's step-3b table, and they differ in exactly those three places, which are
the three methods of `PdEvaluation`:

    option                       detection branch        miss weight    miss density
    A  AtMean (default, D27)     p_D(m)                  1 - p_D(m)     N(m, P) unchanged
    B  AtBranchMean              p_D at the branch's     as A           as A
                                 Kalman-updated mean
    C  Expected                  p_D_bar, sigma points   1 - p_D_bar    as A
    D  ExpectedWithShift         as C                    as C           moment-matched, moved
                                                                        away from the FOV

A is wrong near the FOV edge: a track whose mean is just outside gets no update, even with
most of its mass inside - the region of D11's edge losses. With the 3.6 cm map prior of
roadmap step 8a, P is small and away from the edges all four agree.

Same pattern as `collapse.py`: a Protocol, one class per option, one dict, selected by the
optional `filter.p_D_evaluation` key. Only A is implemented; B to D raise at construction,
each naming what it waits for, so a config cannot select one silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from crop_mot.sensor.sensor_model import SensorModel
from crop_mot.types import Pose2D


class PdEvaluation(Protocol):
    """The p_D values and the missed-branch moments a Bernoulli update uses. [B2]"""

    def miss_p_D(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                 pose: Pose2D) -> float:
        """The p_D in the miss weight r (1 - p_D), standing in for p_D_bar [A2 §2.1].

        Args:
            sensor: the ASSUMED sensor model.
            mean, cov: the track's predicted moments.
            pose: the scan's reported pose.

        Returns:
            p_D in [0, 1].
        """
        raise NotImplementedError

    def detection_p_D(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                      pose: Pose2D, z: np.ndarray) -> float:
        """The p_D in ell(z_i) = p_D N(z_i; z_hat, S) for one gated detection [A2 §3.1].

        Args:
            sensor: the ASSUMED sensor model.
            mean, cov: the track's predicted moments.
            pose: the scan's reported pose.
            z: the gated detection.

        Returns:
            p_D in [0, 1].
        """
        raise NotImplementedError

    def missed_moments(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                       pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
        """The (mean, cov) of the missed branch, approximating (1 - p_D(x)) N(x) [A2 §2].

        Args:
            sensor: the ASSUMED sensor model.
            mean, cov: the track's predicted moments.
            pose: the scan's reported pose.

        Returns:
            The missed branch's (mean, cov).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class AtMean(PdEvaluation):
    """Option A: p_D at the predicted mean everywhere; the missed branch unchanged. [B2]

    The plug-in for p_D_bar that the filter has used since B2 (decision D27): exact when
    p_D is constant over the support of N(m, P) [A2 §2.1], and the missed density
    (1 - p_D(x)) N(x) is then N(m, P) itself [A2 §2]. Wrapping it here changed no output.
    """

    def miss_p_D(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                 pose: Pose2D) -> float:
        """p_D(m); exactly 0 when the mean is outside the assumed FOV."""
        return sensor.p_D(mean, pose)

    def detection_p_D(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                      pose: Pose2D, z: np.ndarray) -> float:
        """p_D(m), the same value as the miss weight: z does not enter."""
        return sensor.p_D(mean, pose)

    def missed_moments(self, sensor: SensorModel, mean: np.ndarray, cov: np.ndarray,
                       pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
        """The predicted (mean, cov), unchanged."""
        return mean, cov


@dataclass(frozen=True)
class AtBranchMean(PdEvaluation):
    """Option B: p_D at each detection branch's Kalman-updated mean. STUB. [B2]

    The detection says where the object is, so evaluating p_D there improves the detection
    branch near the FOV edge; the miss weight and density stay as in A. Waits on the
    author's derivation of the detection update with non-constant p_D, which A2 §7 lists as
    open: which approximation of ell(z) = integral p_D g N dx it stands for.
    """

    def __post_init__(self) -> None:
        raise NotImplementedError("p_D_evaluation 'at_branch_mean' (option B) waits on the "
                                  "detection update with non-constant p_D, open in A2 §7")


@dataclass(frozen=True)
class Expected(PdEvaluation):
    """Option C: p_D_bar from sigma points, in the miss weight and the detections. STUB. [B2]

    p_D_bar = integral p_D(x) N(x; m, P) dx [A2 §2.1] by a sigma-point rule (5 points in
    2D, 5 FOV evaluations per track), so the miss weight 1 - p_D_bar is exact up to the
    rule; the missed density stays N(m, P). The next brick after A (decision D27), once the
    FOV-edge effect is measured to matter. Waits on the choice of the sigma-point rule, and
    on A2 §7's open detection update for ell(z). When it is built, `ScanEvent.in_fov` must
    follow p_D_bar > 0 rather than the mean being in view.
    """

    def __post_init__(self) -> None:
        raise NotImplementedError("p_D_evaluation 'expected' (option C) waits on the "
                                  "sigma-point rule and A2 §7's detection update")


@dataclass(frozen=True)
class ExpectedWithShift(PdEvaluation):
    """Option D: as C, plus the missed density moment-matched and shifted. STUB. [B2]

    The missed density (1 - p_D(x)) N(x; m, P) / (1 - p_D_bar) [A2 §2] is not Gaussian; a
    miss pushes the mass away from where the object would have been seen ("negative
    information", A2 §2.2). Moment-matching it with sigma points is the closest of the four
    to exact. Waits on the author's derivation of those moments.
    """

    def __post_init__(self) -> None:
        raise NotImplementedError("p_D_evaluation 'expected_with_shift' (option D) waits on "
                                  "the derivation of the missed density's moments (A2 §2.2)")


# Available evaluations, keyed by the run config's `filter.p_D_evaluation` value.
PD_EVALUATIONS: dict[str, Callable[[], PdEvaluation]] = {
    "at_mean": AtMean,
    "at_branch_mean": AtBranchMean,
    "expected": Expected,
    "expected_with_shift": ExpectedWithShift,
}
