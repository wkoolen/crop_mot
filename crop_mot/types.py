"""Plain dataclasses that cross every module boundary in the pipeline. [B1-B4]

These are the ONLY types that travel between truth generation, the sensor model, the
filters and the evaluation code. Keeping them plain (frozen dataclasses of floats and numpy
arrays) is what makes the phase-2 ROS 2 adapter a thin conversion layer rather than a
rewrite: a node converts ROS messages into these and calls the same filter.

The truth / filter split is expressed here structurally:
  * `Detection` carries only a measurement vector - no origin label.
  * `Scan` is the filter's input and holds the REPORTED pose.
  * `ScanLabels` is the truth-side twin and is never passed to a filter.
A filter therefore cannot read ground truth by accident, because no function it is given
takes a `ScanLabels`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """A robot pose in the world frame. [B1]

    Used for BOTH the true pose and the reported pose - which one you are holding is a
    property of where it came from, not of the type:
      * `Scan.pose` is always the REPORTED pose, i.e. what the filter is allowed to see.
      * The true pose is written to truth.jsonl and used only by the simulator and the
        evaluation code.
    With `path.pose_known: true` in the config the two are equal, which is the assumption
    B2 and the A2 closed form rely on.

    Attributes:
        x, y: position in metres, world frame.
        theta: heading in radians, counter-clockwise from the +x axis.
    """

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class FieldOfView:
    """The camera footprint, modelled as a wedge in the body frame. [B1]

    Matches the top-down scenario sketch: the quadruped walks up a lane and sees a sector
    ahead of it. A detection can only exist inside this wedge, and clutter is drawn
    uniformly over it.

    Attributes:
        min_range: metres; closer than this the plant is out of frame / too blurred.
        max_range: metres; beyond this the detector does not report.
        half_angle: radians; the wedge is symmetric about the heading, so the full opening
            angle is 2 * half_angle.
    """

    min_range: float
    max_range: float
    half_angle: float

    def area(self) -> float:
        """Area of the wedge in m^2.

        Serves: [B1] as the normaliser of the uniform clutter density c(z) = 1 / area, so that
        the clutter intensity is lambda_FA * c(z) = lambda_FA / area [A0 §Measurement model];
        [B2] the same quantity inside the Bernoulli update.

        Returns:
            Area of the annular sector between min_range and max_range spanning
            2 * half_angle radians: half_angle * (max_range^2 - min_range^2).
        """
        return self.half_angle * (self.max_range**2 - self.min_range**2)


@dataclass(frozen=True)
class Detection:
    """One measurement z reported by the black-box detector. [B1]

    Carries NO origin label: whether this came from a plant or from clutter is recorded
    separately in `ScanLabels` and never reaches a filter. That separation is the whole
    reason this class has a single field.

    Attributes:
        z: shape (dim_z,). In phase 1 this is a world-frame xy position in metres, because
            the detector is assumed to have already un-projected through the known pose -
            see `crop_mot.sensor.models.LinearGaussianXY` for the assumption.
    """

    z: np.ndarray


@dataclass(frozen=True)
class Scan:
    """Everything received at one time step. The filter's only input. [B1/B2]

    Attributes:
        k: scan index, 0-based.
        t: timestamp in seconds. The runner derives dt from consecutive scans rather than
            storing it, so a dropped scan cannot silently desynchronise the prediction.
        pose: the REPORTED robot pose (equal to the true pose when pose_known is true).
        detections: this scan's measurements, in an order that carries no information -
            the simulator shuffles them so that detection index cannot leak origin.
    """

    k: int
    t: float
    pose: Pose2D
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class ScanLabels:
    """Truth-side twin of `Scan`. Evaluation only - never passed to a filter. [B1/B3]

    Written to labels.jsonl alongside detections.jsonl. B3 uses it to build the
    `ScanEvent` sequence that the analytic reference consumes.

    Attributes:
        k: scan index, matching the corresponding `Scan`.
        origin: one entry per detection in the same order as `Scan.detections`; the object
            id the detection came from, or None if it is clutter.
        visible_ids: object ids that were inside the FOV this scan (whether detected or not).
        detected_ids: object ids that were actually detected this scan. The difference
            between visible_ids and detected_ids is exactly the misdetection event that
            drives r downward in B2.
    """

    k: int
    origin: tuple[int | None, ...]
    visible_ids: tuple[int, ...]
    detected_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrackEstimate:
    """One track's marginal posterior. The common output of every filter. [B2/B4]

    This is the type that lets one runner serve Bernoulli, PDA, JPDA, GNN, PMB and PMBM
    without special cases: a single-target filter returns a list of length <= 1, a
    multi-target filter returns a longer list, and a filter with an internal hypothesis
    structure marginalises it away in `extract`.

    Attributes:
        track_id: stable identifier across scans, so a trajectory can be reconstructed from
            the estimates log.
        r: existence probability in [0, 1]. GNN, which makes hard assignments, reports
            exactly 0.0 or 1.0; that is a value, not a special case.
        mean: shape (dim_x,), posterior mean of the target state x.
        cov: shape (dim_x, dim_x), posterior covariance.
    """

    track_id: int
    r: float
    mean: np.ndarray
    cov: np.ndarray
