"""Sampling the black-box detector. The generative half of B1. [B1]

Free functions rather than methods on SensorModel, so that a filter - which is handed a
SensorModel - has no way to generate data. The Protocol evaluates; this module samples.

This is the single place where truth turns into the filter's view of the world, so it is
also the single place that produces the `ScanLabels` the filter must never see.
"""

from __future__ import annotations

import numpy as np

from crop_mot.sensor.fov import in_fov, sample_uniform_in_fov
from crop_mot.sensor.sensor_model import SensorModel
from crop_mot.types import Detection, Pose2D, Scan, ScanLabels
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

    The generative model, which is the thing the thesis derivation assumes
    [A0 §Measurement model]:
      1. For each plant, test visibility against the FOV using the TRUE pose.
      2. Each visible plant is detected independently with probability p_D(x, pose).
      3. A detected plant produces z = h(x, pose) + v with v ~ N(0, R). z is NOT truncated
         to the FOV: near the edge, noise can place a real detection just outside it,
         exactly as A0's model allows.
      4. Independently, draw n_clutter ~ Poisson(lambda_FA) false alarms, positioned
         uniformly over the FOV area.
      5. SORT all detections together by z (lexicographically) before returning, so that
         index order carries no information about origin. Without this step a filter could
         cheat by assuming the first detection is the real one, and any resulting
         performance would be an artefact. Sorting rather than a random shuffle uses no
         random numbers, so neither generator's consumption depends on the other's counts.

    Two separate generators are taken, not one, so that changing lambda_FA does not perturb
    the detection coin flips - see `crop_mot.rng` for why that matters. For the same reason
    every visible plant always consumes one uniform (the coin flip) and one noise draw from
    rng_detect, detected or not: the detection stream's consumption then depends only on
    the geometry, not on p_D.

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
    R = model.measurement.R
    z_list = []
    origin_list = []
    visible_ids = []
    detected_ids = []

    for plant_id, x in zip(truth.field.ids, truth.field.positions):
        if not in_fov(x, pose, model.fov):
            continue
        visible_ids.append(int(plant_id))
        u = rng_detect.random()
        v = rng_detect.multivariate_normal(np.zeros(R.shape[0]), R)
        if u < model.p_D(x, pose):
            detected_ids.append(int(plant_id))
            z_list.append(model.measurement.h(x, pose) + v)
            origin_list.append(int(plant_id))

    n_clutter = rng_clutter.poisson(model.lambda_FA(pose))
    for z in sample_uniform_in_fov(pose, model.fov, n_clutter, rng_clutter):
        z_list.append(z)
        origin_list.append(None)

    order = sorted(range(len(z_list)), key=lambda i: tuple(z_list[i]))
    reported_pose = truth.poses[k].reported
    scan = Scan(
        k=k,
        t=t,
        pose=reported_pose,
        detections=tuple(Detection(z=z_list[i]) for i in order),
    )
    labels = ScanLabels(
        k=k,
        origin=tuple(origin_list[i] for i in order),
        visible_ids=tuple(visible_ids),
        detected_ids=tuple(detected_ids),
    )
    return scan, labels
