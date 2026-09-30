"""The heading-error experiment: NEES and GOSPA against a yaw bias. [B4, roadmap step 8d]

The `python -m crop_mot yaw` subcommand. For every combination of a constant yaw bias and
a heading-wobble amplitude, several seeds are simulated with the pose reported wrongly
(`path.pose_known: false`) while the filter keeps assuming the reported pose is the true
pose, and the map's NEES and GOSPA are measured against truth (decisions D16, D41).

The position noise is set to 0: RTK gives the position to about a centimetre, and the
question is the heading, which comes from another sensor. Expected (roadmap step 8d):
NEES leaves its band once the bias times the range exceeds the posterior std - for the
3.6 cm prior of the known-N map, below one degree at 4 m.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from crop_mot.analysis.evaluation import gospa_series, nees, nees_band, scan_views
from crop_mot.analysis.montecarlo import run_trials
from crop_mot.analysis.plots import plot_yaw_sensitivity
from crop_mot.config import RunConfig, YawSensitivityConfig, load_yaw_sensitivity_config
from crop_mot.io import to_jsonable
from crop_mot.runner.run_dir import RunDir, create_run_dir, write_run_meta


@dataclass(frozen=True)
class YawPoint:
    """One (bias, wobble) combination over its seeds, one outcome per seed. [B4]

    Attributes:
        yaw_bias_deg, yaw_wobble_std_deg: the combination.
        mean_nees: per seed, the average over scans of the average NEES of that scan's
            matched confirmed tracks.
        nees_in_band: per seed, the share of paired scans whose average NEES is inside its
            95 % band.
        mean_gospa: per seed, the mean GOSPA over scans, in metres.
    """

    yaw_bias_deg: float
    yaw_wobble_std_deg: float
    mean_nees: tuple[float, ...]
    nees_in_band: tuple[float, ...]
    mean_gospa: tuple[float, ...]


def perturbed(run: RunConfig, bias_deg: float, wobble_deg: float) -> RunConfig:
    """The run with its pose reported with a yaw bias and wobble, position exact."""
    path = replace(run.scenario.path, pose_known=False, yaw_bias=math.radians(bias_deg),
                   yaw_wobble_std=math.radians(wobble_deg), xy_noise_std=0.0)
    return replace(run, scenario=replace(run.scenario, path=path))


def yaw_measure(run: RunDir, cfg: RunConfig) -> dict[str, float]:
    """One run's NEES and GOSPA, for `montecarlo.run_trials`. [B4]"""
    views = scan_views(run, cfg.filter_cfg.kind, cfg)
    results = gospa_series(views)
    error = nees(views, results)
    lower, upper = nees_band(error.n_pairs, error.dim)
    paired = error.n_pairs > 0
    inside = (error.mean_nees >= lower) & (error.mean_nees <= upper)
    return {
        "mean_nees": float(np.mean(error.mean_nees[paired])) if paired.any() else math.nan,
        "nees_in_band": float(inside[paired].mean()) if paired.any() else math.nan,
        "mean_gospa": float(np.mean([result.distance for result in results])),
    }


def run_yaw_sensitivity(cfg: YawSensitivityConfig, out: RunDir) -> list[YawPoint]:
    """Sweep the yaw bias and wobble, into out's metrics.json and plots. [B4]

    Seeds cfg.run.seed, cfg.run.seed + 1, ... for every combination, so combinations differ
    only in the pose error. The per-run folders are temporary, inside `out`.

    Returns:
        One YawPoint per combination, wobble-major; also written to out.metrics under
        "yaw_sensitivity" and drawn to out.plots / "yaw_sensitivity.png".
    """
    points = []
    for wobble in cfg.yaw_wobble_std_deg:
        for bias in cfg.yaw_bias_deg:
            trials = run_trials(perturbed(cfg.run, bias, wobble), cfg.seeds, cfg.run.seed,
                                out.root, {"yaw": yaw_measure})
            values = [trial.values["yaw"] for trial in trials]
            points.append(YawPoint(
                yaw_bias_deg=bias, yaw_wobble_std_deg=wobble,
                mean_nees=tuple(v["mean_nees"] for v in values),
                nees_in_band=tuple(v["nees_in_band"] for v in values),
                mean_gospa=tuple(v["mean_gospa"] for v in values),
            ))
    out.metrics.write_text(json.dumps(to_jsonable({"yaw_sensitivity": {
        "filter": cfg.run.filter_cfg.kind, "seeds": cfg.seeds,
        "points": [asdict(point) for point in points]}}), indent=2) + "\n", encoding="utf-8")
    plot_yaw_sensitivity(points, out.plots / "yaw_sensitivity.png",
                         title=f"{cfg.name}: {cfg.run.filter_cfg.kind}, {cfg.seeds} seeds each")
    return points


def yaw_from_config(config_path: Path, runs_base: Path) -> RunDir:
    """The CLI entry point: a fresh folder named after the experiment. [B4]"""
    cfg = load_yaw_sensitivity_config(config_path)
    out = create_run_dir(runs_base, cfg.name, cfg.run.seed, config_path)
    write_run_meta(out, cfg.run.seed, sys.argv)
    run_yaw_sensitivity(cfg, out)
    return out
