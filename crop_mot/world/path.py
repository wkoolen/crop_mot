"""The robot's walk, and the pose-known switch. [B1]

This file carries one of the three extension slots that are stubbed but unused in B1-B4.
The slot matters more than it looks: it fixes whether `Scan.pose` MEANS the true pose or an
estimate of it. Deciding that after the filters exist would mean revisiting every filter's
un-projection, so the distinction is written down now.

pose_known = True is a simplification for now (decision D16): the robot is assumed to carry
RTK-GPS, so tracking static plants is mapping with known poses. It is revisited if RTK is
not available on the Go2 setup. The slot stays for the heading-error sensitivity experiment.
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

    EXTENSION SLOT, cfg.pose_known = False, kept for that sensitivity experiment only
    (D16): the reported pose is the true pose perturbed by
    N(0, yaw_wobble_std^2) in heading and N(0, xy_noise_std^2) in position - gait-induced
    odometry error on a quadruped. Nothing downstream changes shape: detections are still
    generated from the TRUE pose and the filter still un-projects with the REPORTED one, so
    the mismatch appears as a measurement bias exactly as it would on the robot. This is not
    implemented in phase 1 and no config in configs/ enables it.

    Serves: [B1] the pose sequence; [B1 extension] the pose-uncertainty experiment.

    Args:
        cfg: path geometry, timing, and the pose_known switch.
        rng_path: the "path" substream generator. Unused while pose_known is True, but
            still accepted so that flipping the switch does not change the signature.

    Returns:
        A list of `n_scans` PoseSample entries, ordered by increasing t starting at t = 0.

    Raises:
        ValueError: if cfg.kind is not "straight_lane".
        NotImplementedError: if cfg.pose_known is False (the extension slot above).
    """
    if cfg.kind != "straight_lane":
        raise ValueError(f"unknown path kind {cfg.kind!r}; phase 1 has only 'straight_lane'")
    if not cfg.pose_known:
        raise NotImplementedError(
            "pose_known: false is the gait-wobble EXTENSION SLOT and is not implemented in "
            "phase 1; B2 and the A2 closed form assume the reported pose is the true pose."
        )

    samples = []
    for k in range(cfg.n_scans):
        t = k * cfg.scan_period
        distance = cfg.speed * t
        true_pose = Pose2D(
            x=cfg.x + distance * np.cos(cfg.heading),
            y=cfg.y_start + distance * np.sin(cfg.heading),
            theta=cfg.heading,
        )
        # pose_known: the reported pose IS the true pose; rng_path is not drawn from.
        samples.append(PoseSample(t=t, true=true_pose, reported=true_pose))
    return samples
