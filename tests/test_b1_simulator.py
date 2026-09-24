"""B1: does the simulator generate what the model says it should? [B1]

These tests are the reason to trust every plot downstream. If the detector does not
actually miss with probability 1 - p_D, then B2's r-decay curve is measuring something
other than what the thesis claims.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.config import DetectionConfig, ScenarioConfig
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.sensor.fov import in_fov, sample_uniform_in_fov
from crop_mot.sensor.models import LinearGaussianXY
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.sensor.sensor_model import build_sensor_model
from crop_mot.types import FieldOfView, Pose2D
from crop_mot.world.truth import read_truth


def _simulate_into(cfg: ScenarioConfig, root) -> RunDir:
    """Simulate `cfg` into a fresh folder `root` (no config copy needed for these tests)."""
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(cfg, run)
    return run


def _standing_still(cfg: ScenarioConfig, n_scans: int) -> ScenarioConfig:
    """The tiny scenario with the robot parked at y = 2.3, where all five plants are in view.

    Gives thousands of visible-plant samples for the statistical tests without a long row.
    Parked so every plant is about 3 sigma of the measurement noise (0.6 m) inside the
    wedge: the detector drops a z that lands outside the FOV, so a plant near the edge has
    an effective detection rate below p_D, which would bias the p_D test.
    """
    return replace(cfg, path=replace(cfg.path, y_start=2.3, speed=0.0, n_scans=n_scans))


def test_same_seed_reproduces_detections_exactly(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """Two simulations with the same seed produce byte-identical detections.jsonl. [B1]

    The foundation of every comparison in the thesis: if this fails, no two filter runs are
    comparable and no result is reproducible. Compares bytes rather than parsed values, so
    that a change in float formatting is caught too.
    """
    first = _simulate_into(tiny_scenario, tmp_path / "first")
    second = _simulate_into(tiny_scenario, tmp_path / "second")

    assert first.detections.read_bytes() == second.detections.read_bytes()
    assert first.labels.read_bytes() == second.labels.read_bytes()
    assert first.truth.read_bytes() == second.truth.read_bytes()


def test_changing_clutter_rate_does_not_move_the_plants(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """lambda_FA changes the clutter, not the field. [B1, reproducibility]

    This is what the named RNG substreams in `crop_mot.rng` buy. Without them, drawing more
    clutter would consume more random numbers and shift every later draw, so a parameter
    sweep over lambda_FA would silently be a sweep over different fields as well.
    """
    more_clutter = replace(tiny_scenario, sensor=replace(tiny_scenario.sensor, lambda_FA=6.0))
    base = _simulate_into(tiny_scenario, tmp_path / "base")
    other = _simulate_into(more_clutter, tmp_path / "other")

    truth_base = read_truth(base.truth)
    truth_other = read_truth(other.truth)
    assert np.array_equal(truth_base.field.ids, truth_other.field.ids)
    assert np.array_equal(truth_base.field.positions, truth_other.field.positions)

    # The detection coin flips have their own stream too, so the same plants are detected -
    # only the clutter differs.
    labels_base = read_labels(base.labels)
    labels_other = read_labels(other.labels)
    assert [lab.detected_ids for lab in labels_base] == [lab.detected_ids for lab in labels_other]
    n_clutter_base = sum(lab.origin.count(None) for lab in labels_base)
    n_clutter_other = sum(lab.origin.count(None) for lab in labels_other)
    assert n_clutter_base != n_clutter_other


def test_no_detection_outside_the_fov(tiny_scenario: ScenarioConfig, tmp_path) -> None:
    """Every detection, real or clutter, lies inside the FOV wedge at its scan's pose. [B1]

    Catches sign errors in the body-frame transform, which otherwise show up much later as
    an inexplicably bad tracking result.
    """
    walk_past = replace(tiny_scenario, path=replace(tiny_scenario.path, n_scans=60))
    run = _simulate_into(walk_past, tmp_path / "run")
    truth = read_truth(run.truth)
    scans = read_detections(run.detections)
    labels = read_labels(run.labels)
    fov = walk_past.sensor.fov
    position_of = dict(zip(truth.field.ids.tolist(), truth.field.positions))

    n_clutter = 0
    n_real = 0
    for scan, label, sample in zip(scans, labels, truth.poses):
        for detection, origin in zip(scan.detections, label.origin):
            assert in_fov(detection.z, sample.true, fov), f"detection outside FOV, k={scan.k}"
            if origin is None:
                n_clutter += 1
            else:
                n_real += 1
                assert in_fov(position_of[origin], sample.true, fov)
        in_view = {i for i, x in position_of.items() if in_fov(x, sample.true, fov)}
        assert set(label.visible_ids) == in_view
    assert n_clutter > 0 and n_real > 0


def test_empirical_detection_rate_matches_p_D(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """Over many scans, detected/visible converges to p_D. [B1]

    A statistical test, so it needs a tolerance and enough samples. Use the labels to count
    visible and detected plants; the point is to catch a detector that is systematically
    over- or under-detecting, not to verify the binomial distribution.
    """
    cfg = _standing_still(tiny_scenario, n_scans=400)
    run = _simulate_into(cfg, tmp_path / "run")
    labels = read_labels(run.labels)

    n_visible = sum(len(lab.visible_ids) for lab in labels)
    n_detected = sum(len(lab.detected_ids) for lab in labels)
    assert n_visible >= 1000

    p_D = cfg.sensor.detection.p_D
    rate = n_detected / n_visible
    standard_error = np.sqrt(p_D * (1.0 - p_D) / n_visible)
    assert abs(rate - p_D) < 4.0 * standard_error


def test_clutter_count_matches_poisson_lambda(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """The number of clutter detections per scan has mean lambda_FA. [B1]

    Check the mean, and ideally that the variance is also near lambda_FA - a Poisson
    distribution has both equal, so a mismatch reveals the wrong distribution rather than
    just the wrong parameter.
    """
    cfg = _standing_still(tiny_scenario, n_scans=400)
    run = _simulate_into(cfg, tmp_path / "run")
    counts = np.array([lab.origin.count(None) for lab in read_labels(run.labels)], dtype=float)

    lam = cfg.sensor.lambda_FA
    n = len(counts)
    # Standard errors of the sample mean and of the sample variance for Poisson(lam) data.
    se_mean = np.sqrt(lam / n)
    se_var = np.sqrt((lam + 2.0 * lam**2) / n)
    assert abs(counts.mean() - lam) < 4.0 * se_mean
    assert abs(counts.var(ddof=1) - lam) < 4.0 * se_var


def test_pose_known_means_reported_equals_true(
    tiny_scenario: ScenarioConfig, tmp_path
) -> None:
    """With pose_known: true, Scan.pose is exactly the true pose. [B1]

    Guards the extension slot: if the gait wobble is ever accidentally applied while
    pose_known is true, B2 and B3 would be running under a different assumption than the
    A2 derivation, and the cross-check would fail for a reason nobody would guess.
    """
    run = _simulate_into(tiny_scenario, tmp_path / "run")
    truth = read_truth(run.truth)
    scans = read_detections(run.detections)

    assert len(scans) == len(truth.poses) == tiny_scenario.path.n_scans
    for scan, sample in zip(scans, truth.poses):
        assert sample.reported == sample.true
        assert scan.pose == sample.true
        assert scan.t == sample.t


@pytest.mark.parametrize("profile", ["constant", "range_dependent"])
def test_p_D_is_zero_outside_the_fov(profile: str) -> None:
    """Both p_D profiles return exactly 0.0 outside the FOV. [B1/B3]

    Not a rounding matter: the Bernoulli update treats "out of view" differently from "in
    view but missed", and B3's ScanEvent records `in_fov` precisely so the closed form can
    too. A small non-zero p_D outside the wedge would blur that distinction.
    """
    fov = FieldOfView(min_range=0.3, max_range=4.0, half_angle=0.6)
    if profile == "constant":
        detection = DetectionConfig(kind="constant", p_D=0.85)
    else:
        detection = DetectionConfig(kind="range_dependent", p_D_near=0.95, p_D_far=0.55)
    model = build_sensor_model(fov, detection, 2.0, LinearGaussianXY(R=0.04 * np.eye(2)))

    pose = Pose2D(x=1.0, y=-2.0, theta=np.pi / 2)  # facing +y
    outside = [
        np.array([1.0, -3.0]),   # behind the robot
        np.array([1.0, 2.5]),    # beyond max_range
        np.array([1.0, -1.9]),   # closer than min_range
        np.array([3.5, -1.0]),   # in range but outside the half angle
    ]
    for x in outside:
        value = model.p_D(x, pose)
        assert value == 0.0 and isinstance(value, float)
        assert model.clutter_density(x, pose) == 0.0
    assert model.p_D(np.array([1.0, 0.0]), pose) > 0.0


def test_clutter_is_uniform_over_fov_area() -> None:
    """Clutter is uniform over AREA, not over range. [B1, added]

    Guards the sqrt in `sample_uniform_in_fov`: the annulus between min_range and the
    radius that splits the wedge's area in half must receive half of the points. Sampling
    the range uniformly instead would put far more than half there.
    """
    fov = FieldOfView(min_range=0.3, max_range=4.0, half_angle=0.6)
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    n = 20000
    points = sample_uniform_in_fov(pose, fov, n, np.random.default_rng(0))

    rho = np.hypot(points[:, 0], points[:, 1])
    rho_half = np.sqrt(0.5 * (fov.max_range**2 + fov.min_range**2))
    inner_fraction = np.mean(rho <= rho_half)
    assert abs(inner_fraction - 0.5) < 4.0 * np.sqrt(0.25 / n)
    assert all(in_fov(x, pose, fov) for x in points)


def test_range_dependent_p_D_is_linear_between_its_endpoints() -> None:
    """RangeDependentPD gives p_D_near at min_range and p_D_far at max_range. [B1, added]

    Pins decision D6 (linear in range), and checks that occlusion is refused rather than
    silently ignored.
    """
    fov = FieldOfView(min_range=0.3, max_range=4.0, half_angle=0.6)
    detection = DetectionConfig(kind="range_dependent", p_D_near=0.95, p_D_far=0.55)
    model = build_sensor_model(fov, detection, 2.0, LinearGaussianXY(R=0.04 * np.eye(2)))
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)

    assert model.p_D(np.array([0.3, 0.0]), pose) == pytest.approx(0.95)
    assert model.p_D(np.array([4.0, 0.0]), pose) == pytest.approx(0.55)
    assert model.p_D(np.array([2.15, 0.0]), pose) == pytest.approx(0.75)

    with pytest.raises(NotImplementedError):
        build_sensor_model(fov, replace(detection, occlusion_factor=0.6), 2.0,
                           LinearGaussianXY(R=0.04 * np.eye(2)))
