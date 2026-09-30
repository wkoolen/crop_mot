"""The robot's walk, and the pose-known switch. [B1]

The switch fixes whether `Scan.pose` MEANS the true pose or an estimate of it.

pose_known = True is a simplification for now (decision D16): the robot is assumed to carry
RTK-GPS, so tracking static plants is mapping with known poses. It is revisited if RTK is
not available on the Go2 setup. pose_known = False is implemented for the heading-error
sensitivity experiment only (roadmap step 8d, decision D41): no shipped run config uses
it, and the filters still treat the reported pose as the true one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.config import PathConfig
from crop_mot.types import Pose2D


@dataclass(frozen=True)
class PoseSample:
    """One time step's true and reported pose. [B1]

    Attributes:
        t: timestamp in seconds.
        true: the pose the simulator generates detections from.
        reported: the pose handed to the filter inside a `Scan`. Identical to `true` when
            PathConfig.pose_known is True.
    """

    t: float
    true: Pose2D
    reported: Pose2D


def generate_path(cfg: PathConfig, rng_path: np.random.Generator) -> list[PoseSample]:
    """Build the robot's walk along the lane, as (true, reported) pose pairs.

    With cfg.pose_known = True: the robot advances at `speed` along the heading, one sample
    every `scan_period`, for `n_scans` samples. The reported pose IS the true pose and
    yaw_wobble_std / xy_noise_std are ignored. This is the assumption B2 and the A2 closed
    form rely on.

    Known pose is a simplification for now (decision D16, confirmed by the author on
    2026-09-30): the robot is assumed to carry RTK-GPS. It is revisited if RTK is not
    available on the Go2 setup. With static plants the problem is then mapping with known
    poses - no filter carries a pose state, and the path only decides what is in view,
    through p_D(x, pose). The bound to state alongside it: RTK gives position to about a
    centimetre, but heading comes from another sensor, and one degree of heading error
    moves a detection at 4 m by about 7 cm, against a measurement noise of sigma = 0.2 m.
    A heading bias moves every detection in a scan the same way and does not average out
    over scans, so its effect is measured (with NEES) in a sensitivity experiment rather
    than assumed away.

    cfg.pose_known = False, for that sensitivity experiment only (D16, D41): the
    reported pose is the true pose plus a constant yaw_bias and N(0, yaw_wobble_std^2) in
    heading, and N(0, xy_noise_std^2) in position, drawn per scan from the path substream
    (dx, dy, dtheta). Nothing downstream changes shape: detections are still generated
    from the TRUE pose and re-expressed through the REPORTED one
    (`sensor.detector.reexpress`), so the error appears as a measurement bias exactly as
    it would on the robot - the same for every detection of a scan.

    Serves: [B1] the pose sequence; [B1 extension] the pose-uncertainty experiment.

    Args:
        cfg: path geometry, timing, and the pose_known switch.
        rng_path: the "path" substream generator. Unused while pose_known is True.

    Returns:
        A list of `n_scans` PoseSample entries, ordered by increasing t starting at t = 0.

    Raises:
        ValueError: if cfg.kind is not "straight_lane".
    """
    if cfg.kind != "straight_lane":
        raise ValueError(f"unknown path kind {cfg.kind!r}; phase 1 has only 'straight_lane'")
    samples = []
    for k in range(cfg.n_scans):
        t = k * cfg.scan_period
        distance = cfg.speed * t
        true_pose = Pose2D(
            x=cfg.x + distance * np.cos(cfg.heading),
            y=cfg.y_start + distance * np.sin(cfg.heading),
            theta=cfg.heading,
        )
        if cfg.pose_known:
            # The reported pose IS the true pose; rng_path is not drawn from.
            reported = true_pose
        else:
            dx, dy, dtheta = rng_path.normal(0.0, [cfg.xy_noise_std, cfg.xy_noise_std,
                                                   cfg.yaw_wobble_std])
            reported = Pose2D(x=true_pose.x + dx, y=true_pose.y + dy,
                              theta=true_pose.theta + cfg.yaw_bias + dtheta)
        samples.append(PoseSample(t=t, true=true_pose, reported=reported))
    return samples
