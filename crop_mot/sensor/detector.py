"""Sampling the black-box detector. The generative half of B1. [B1]

Free functions rather than methods on SensorModel, so that a filter - which is handed a
SensorModel - has no way to generate data. The Protocol evaluates; this module samples.

This is the single place where truth turns into the filter's view of the world, so it is
also the single place that produces the `ScanLabels` the filter must never see.
"""

from __future__ import annotations

import numpy as np

from crop_mot.sensor.sensor_model import SensorModel
from crop_mot.types import Pose2D, Scan, ScanLabels
from crop_mot.world.truth import GroundTruth


def sample_scan(
    truth: GroundTruth,
    pose: Pose2D,
    k: int,
    t: float,
    model: SensorModel,
    rng_detect: np.random.Generator,
    rng_clutter: np.random.Generator,
) -> tuple[Scan, ScanLabels]:
    """Draw one scan of detections from the truth.

    The generative model, which is the thing the thesis derivation assumes:
      1. For each plant, test visibility against the FOV using the TRUE pose.
      2. Each visible plant is detected independently with probability p_D(x, pose).
      3. A detected plant produces z = h(x, pose) + v with v ~ N(0, R).
      4. Independently, draw n_clutter ~ Poisson(lambda_FA) false alarms, positioned
         uniformly over the FOV area.
      5. SHUFFLE all detections together before returning, so that index order carries no
         information about origin. Without this step a filter could cheat by assuming the
         first detection is the real one, and any resulting performance would be an artefact.

    Two separate generators are taken, not one, so that changing lambda_FA does not perturb
    the detection coin flips - see `crop_mot.rng` for why that matters.

    Serves: [B1] the core of the simulator.

    Args:
        truth: plant positions; the TRUE pose is taken from the `pose` argument, not from
            here, because the caller is iterating over scans.
        pose: the TRUE pose at this scan. Visibility and geometry are physical facts and
            do not depend on what the robot believes.
        k: scan index, copied into both returned objects.
        t: timestamp in seconds.
        model: the TRUTH sensor model (built from the scenario's `sensor` block).
        rng_detect: the "detection" substream, for detection coin flips and measurement noise.
        rng_clutter: the "clutter" substream, for the Poisson count and clutter positions.

    Returns:
        A tuple (scan, labels):
          * scan - what the filter receives, holding the REPORTED pose;
          * labels - the truth-side record of which detection came from which plant, which
            plants were visible, and which were detected.
    """
    raise NotImplementedError
