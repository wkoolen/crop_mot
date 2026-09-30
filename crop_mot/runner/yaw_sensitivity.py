"""The heading-error experiment: NEES and GOSPA against a yaw bias. [B4, roadmap step 8d]

The `python -m crop_mot yaw` subcommand. For every combination of a constant yaw bias and
a heading-wobble amplitude, several seeds are simulated with the pose reported wrongly
(`path.pose_known: false`) while the filter keeps assuming the reported pose is the true
pose, and the known-N map's NEES and GOSPA are measured (decisions D16, D41).
"""

from __future__ import annotations

from pathlib import Path

from crop_mot.config import YawSensitivityConfig, load_yaw_sensitivity_config
from crop_mot.runner.run_dir import RunDir, create_run_dir


def run_yaw_sensitivity(cfg: YawSensitivityConfig, out: RunDir) -> None:
    """Sweep the yaw bias and wobble, into out's metrics.json and plots. STUB. [B4]

    Waits on the unknown-pose branch of `world.path.generate_path` (the pose-known switch's
    extension slot) and on re-expressing detections through the reported pose, which the
    next brick adds; NEES (roadmap step 5) exists.
    """
    raise NotImplementedError("the yaw experiment waits on the unknown-pose branch of "
                              "generate_path (roadmap step 8d, next brick)")


def yaw_from_config(config_path: Path, runs_base: Path) -> RunDir:
    """The CLI entry point: a fresh folder named after the experiment. [B4]"""
    cfg = load_yaw_sensitivity_config(config_path)
    out = create_run_dir(runs_base, cfg.name, cfg.run.seed, config_path)
    run_yaw_sensitivity(cfg, out)
    return out
