"""Sampling the black-box detector. The generative half of B1. [B1]

Free functions rather than methods on SensorModel, so that a filter - which is handed a
SensorModel - has no way to generate data. The Protocol evaluates; this module samples.

This is the single place where truth turns into the filter's view of the world, so it is
also the single place that produces the `ScanLabels` the filter must never see.
"""

from __future__ import annotations

import numpy as np

from crop_mot.config import MultiplicityConfig
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
    multiplicity: MultiplicityConfig = MultiplicityConfig(),
    rng_multi: np.random.Generator | None = None,
    weed_model: SensorModel | None = None,
    rng_weeds: np.random.Generator | None = None,
) -> tuple[Scan, ScanLabels]:
    """Draw one scan of detections from the truth.

    The generative model, which is the thing the thesis derivation assumes
    [A0 §Measurement model]:
      1. For each plant, test visibility against the FOV using the TRUE pose.
      2. Each visible plant is detected independently with probability p_D(x, pose).
      3. A detected plant produces z = h(x, pose) + v with v ~ N(0, R). A z that falls
         OUTSIDE the FOV is not reported: the camera cannot report outside its image, so
         the FOV is the measurement space and c(z) is a proper pdf on it (decision D4,
         revised 2026-09-24). Consequence, deliberately NOT modelled by the filter's p_D:
         a plant within a few sigma of the FOV edge is effectively detected with
         probability p_D(x) * P(x + v in FOV) < p_D(x). Such a plant counts as visible
         but not detected in the labels, and is listed in `truncated_ids`.
      3b. MULTIPLICITY (extension slot, `sensor.multiplicity`). With kind "single" -
         the default, and the A0 point-target assumption - steps 2-3 are the whole story:
         at most one detection per plant. With "duplicate", a detected plant gets extra
         hits z_primary + N(0, spread_std^2 I), each with probability p_split, up to
         max_extra: a detector returning several boxes for one plant. With "poisson"
         (extended objects), steps 2-3 are replaced: each visible plant produces
         n ~ Poisson(gamma) detections z = h(x) + N(0, extent_std^2 I) + v, so it is
         missed with probability exp(-gamma) and p_D is not used. Every z is truncated
         to the FOV on its own; a plant counts as detected if at least one survives.
      4. Independently, draw n_clutter ~ Poisson(lambda_FA) false alarms, positioned
         uniformly over the FOV area.
      4b. WEEDS (persistent false targets, decision D15; only when the truth has weeds).
         Each weed inside the FOV is reported with probability weed_model.p_D(w, pose) at
         z = h(w, pose) + v, v ~ N(0, R), truncated to the FOV like a plant detection.
         Its origin is None - it is a false alarm - and `weed_origin` names the weed.
         What sets it apart from step 4 is that the weed does not move: the same false
         alarm can recur at the same place scan after scan, which the Poisson clutter the
         filter assumes cannot do. The coin flips are independent across scans, so the
         recurrence comes from the fixed position alone.
      5. SORT all detections together by z (lexicographically) before returning, so that
         index order carries no information about origin. Without this step a filter could
         cheat by assuming the first detection is the real one, and any resulting
         performance would be an artefact. Sorting rather than a random shuffle uses no
         random numbers, so neither generator's consumption depends on the other's counts.

    Two separate generators are taken, not one, so that changing lambda_FA does not perturb
    the detection coin flips - see `crop_mot.rng` for why that matters. For the same reason
    every visible plant always consumes one uniform (the coin flip) and one noise draw from
    rng_detect, detected or not: the detection stream's consumption then depends only on
    the geometry, not on p_D. Everything the multiplicity models draw comes from a third
    generator, rng_multi, so "single" output is identical whether or not it is passed,
    and switching to "duplicate" leaves the primary detections and the clutter unchanged.
    The weeds draw from a fourth, rng_weeds, with the same one-uniform-one-noise-draw rule
    per visible weed, so adding weeds leaves every plant detection and clutter return as
    it was: the scan gains the weed detections and nothing else changes.

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
        multiplicity: the scenario's `sensor.multiplicity` block; default "single".
        rng_multi: the "multiplicity" substream. Required unless multiplicity is "single".
        weed_model: the TRUTH sensor model for weeds (built from `sensor.weed_detection`),
            supplying the weeds' p_D. Required when the truth has weeds.
        rng_weeds: the "weed_detection" substream. Required when the truth has weeds.

    Returns:
        A tuple (scan, labels):
          * scan - what the filter receives, holding the REPORTED pose;
          * labels - the truth-side record of which detection came from which plant, which
            plants were visible, which were detected and which were lost at the FOV edge,
            and, with weeds, which detections came from which weed.

    Raises:
        ValueError: if multiplicity needs rng_multi and none is given, or its kind is
            unknown; or if the truth has weeds and weed_model or rng_weeds is missing.
    """
    if multiplicity.kind not in ("single", "duplicate", "poisson"):
        raise ValueError(f"unknown multiplicity kind {multiplicity.kind!r}")
    if multiplicity.kind != "single" and rng_multi is None:
        raise ValueError(f"multiplicity {multiplicity.kind!r} needs the rng_multi substream")
    has_weeds = len(truth.weeds) > 0
    if has_weeds and (weed_model is None or rng_weeds is None):
        raise ValueError("the truth has weeds: sample_scan needs weed_model and rng_weeds")

    R = model.measurement.R
    z_list = []
    origin_list = []
    weed_list = []
    visible_ids = []
    detected_ids = []
    truncated_ids = []

    for plant_id, x in zip(truth.field.ids, truth.field.positions):
        if not in_fov(x, pose, model.fov):
            continue
        visible_ids.append(int(plant_id))
        if multiplicity.kind == "poisson":
            generated = _poisson_hits(x, pose, model, multiplicity, rng_multi)
        else:
            u = rng_detect.random()
            v = rng_detect.multivariate_normal(np.zeros(R.shape[0]), R)
            z = model.measurement.h(x, pose) + v
            generated = [z] if u < model.p_D(x, pose) else []
            if generated and multiplicity.kind == "duplicate":
                generated += _extra_hits(z, multiplicity, rng_multi)

        kept = [z for z in generated if in_fov(z, pose, model.fov)]
        if kept:
            detected_ids.append(int(plant_id))
            z_list.extend(kept)
            origin_list.extend([int(plant_id)] * len(kept))
            weed_list.extend([None] * len(kept))
        elif generated:
            truncated_ids.append(int(plant_id))

    n_clutter = rng_clutter.poisson(model.lambda_FA(pose))
    for z in sample_uniform_in_fov(pose, model.fov, n_clutter, rng_clutter):
        z_list.append(z)
        origin_list.append(None)
        weed_list.append(None)

    visible_weed_ids = []
    for weed_id, w in enumerate(truth.weeds):
        if not in_fov(w, pose, model.fov):
            continue
        visible_weed_ids.append(weed_id)
        u = rng_weeds.random()
        v = rng_weeds.multivariate_normal(np.zeros(R.shape[0]), R)
        z = model.measurement.h(w, pose) + v
        if u < weed_model.p_D(w, pose) and in_fov(z, pose, model.fov):
            z_list.append(z)
            origin_list.append(None)
            weed_list.append(weed_id)

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
        truncated_ids=tuple(truncated_ids),
        weed_origin=tuple(weed_list[i] for i in order) if has_weeds else (),
        visible_weed_ids=tuple(visible_weed_ids),
    )
    return scan, labels


def _extra_hits(
    z_primary: np.ndarray, multiplicity: MultiplicityConfig, rng_multi: np.random.Generator
) -> list[np.ndarray]:
    """Duplicate hits on an already-detected plant. [B1, multiplicity "duplicate"]

    The number of extras is geometric and capped: each further hit is added with
    probability p_split while fewer than max_extra have been added, so the expected count
    is p_split + p_split^2 + ... + p_split^max_extra. Each extra lies at
    z_primary + N(0, spread_std^2 I), i.e. it inherits the primary's measurement noise
    and adds its own offset, as a detector box that is split or doubled would.
    """
    n = 0
    while n < multiplicity.max_extra and rng_multi.random() < multiplicity.p_split:
        n += 1
    offsets = rng_multi.normal(0.0, multiplicity.spread_std, size=(n, z_primary.shape[0]))
    return [z_primary + offset for offset in offsets]


def _poisson_hits(
    x: np.ndarray, pose: Pose2D, model: SensorModel, multiplicity: MultiplicityConfig,
    rng_multi: np.random.Generator,
) -> list[np.ndarray]:
    """Detections of one visible plant as an extended object. [B1, multiplicity "poisson"]

    ASSUMPTION: the standard extended-object measurement model - the number of detections
    is Poisson(gamma) and each one is an independent draw z = h(x) + w + v with
    w ~ N(0, extent_std^2 I) (where on the plant) and v ~ N(0, R) (sensor noise). The sum
    w + v is drawn as one Gaussian with covariance R + extent_std^2 I.
    """
    n = rng_multi.poisson(multiplicity.gamma)
    R = model.measurement.R
    cov = R + multiplicity.extent_std**2 * np.eye(R.shape[0])
    noise = rng_multi.multivariate_normal(np.zeros(R.shape[0]), cov, size=n)
    z_hat = model.measurement.h(x, pose)
    return [z_hat + w for w in noise]
