"""B1 entry point: scenario config -> truth + detections on disk. [B1]

This is the only module that connects the world to the sensor. After it has run, the
scenario exists solely as files, and everything downstream - every filter, every baseline,
every re-analysis - works from those files.
"""

from __future__ import annotations

from pathlib import Path

from crop_mot.config import ScenarioConfig
from crop_mot.runner.run_dir import RunDir


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
    raise NotImplementedError


def simulate_from_config(config_path: Path, runs_base: Path) -> RunDir:
    """Load a scenario config, create a run folder and simulate into it. [B1]

    The `python -m crop_mot simulate` entry point.

    Args:
        config_path: path to a B1 scenario YAML file.
        runs_base: the runs/ directory.

    Returns:
        The run folder that was created and populated.
    """
    raise NotImplementedError
