"""B1 entry point: scenario config -> truth + detections on disk. [B1]

This is the only module that connects the world to the sensor. After it has run, the
scenario exists solely as files, and everything downstream - every filter, every baseline,
every re-analysis - works from those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from crop_mot.config import ScenarioConfig, load_scenario_config
from crop_mot.rng import substreams
from crop_mot.runner.run_dir import RunDir, create_run_dir, write_run_meta
from crop_mot.sensor.detector import sample_scan
from crop_mot.sensor.models import build_measurement_model
from crop_mot.sensor.record import write_detections, write_labels
from crop_mot.sensor.sensor_model import build_sensor_model
from crop_mot.world.field import generate_field
from crop_mot.world.path import generate_path
from crop_mot.world.truth import GroundTruth, write_truth


def simulate(cfg: ScenarioConfig, run: RunDir) -> None:
    """Generate a scenario and write it to a run folder. [B1]

    Steps:
      1. Split the seed into named substreams (field / path / detection / clutter), so that
         changing the detector cannot move the plants.
      2. Generate the plant field and the robot path.
      3. For each scan, sample detections from the TRUE pose, producing both the Scan the
         filter will see and the ScanLabels it must not.
      4. Write truth.jsonl, labels.jsonl and detections.jsonl, then run_meta.json.

    Writes detections.jsonl LAST of the three data files, so that its presence is a
    reliable signal that the simulation completed - a half-written detections file consumed
    by a filter is a hard bug to diagnose.

    Serves: [B1]; and indirectly every other work package, since this produces the shared
    input that makes filter comparison fair.

    Args:
        cfg: the parsed scenario configuration.
        run: an existing run folder to write into.
    """
    streams = substreams(cfg.seed)
    field = generate_field(cfg.world, streams["field"])
    poses = generate_path(cfg.path, streams["path"])
    truth = GroundTruth(field=field, poses=poses)

    measurement = build_measurement_model(cfg.sensor.measurement)
    model = build_sensor_model(cfg.sensor.fov, cfg.sensor.detection, cfg.sensor.lambda_FA,
                               measurement)

    scans = []
    labels = []
    for k, sample in enumerate(poses):
        scan, label = sample_scan(truth, sample.true, k, sample.t, model,
                                  streams["detection"], streams["clutter"])
        scans.append(scan)
        labels.append(label)

    write_truth(run.truth, truth)
    write_labels(run.labels, labels)
    write_detections(run.detections, scans)
    write_run_meta(run, cfg.seed, sys.argv)


def simulate_from_config(config_path: Path, runs_base: Path) -> RunDir:
    """Load a scenario config, create a run folder and simulate into it. [B1]

    The `python -m crop_mot simulate` entry point.

    Args:
        config_path: path to a B1 scenario YAML file.
        runs_base: the runs/ directory.

    Returns:
        The run folder that was created and populated.
    """
    cfg = load_scenario_config(config_path)
    run = create_run_dir(runs_base, cfg.name, cfg.seed, config_path)
    simulate(cfg, run)
    return run
