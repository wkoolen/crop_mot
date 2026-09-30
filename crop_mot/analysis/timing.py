"""Computation time per scan against problem size (roadmap step 4c, decision D29). [B4]

The question is part of choosing the phase-2 method: what does a scan cost, and how does
the cost grow with the problem? The scan period, 0.25 s, is the real-time budget.

The hypothesis to test, not assume: cost per scan should follow the number of objects in
view and how many tracks share gates, not the total number N - but the current bank
updates every component on every scan, so its cost grows with the number of components.
Expected costs, to check the measured slopes against (T tracks, M detections):
    bank O(T M); GNN O(n^3) per cluster, n = T + M; exact JPDA exponential in the cluster;
    Murty's k-best O(k n^3); PMBM the number of global hypotheses times Murty; PHD grid
    O(G M) for G cells.

A sweep changes one scenario parameter (`SWEEPS`); each run's problem size is measured
from the run folder afterwards, not assumed from the parameter. Times come from the
timing log `run_filter` writes. Scan 0 includes warm-up and is reported apart. Timings are
only comparable single-threaded: set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 before
Python starts. Tests never assert on times.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from crop_mot.config import RunConfig
from crop_mot.io import read_jsonl
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.record import read_labels
from crop_mot.world.truth import read_truth


def _with_plan_rows(cfg: RunConfig, rows) -> RunConfig:
    """A known-N filter's plan follows the world's rows, so the model stays matched."""
    plan = cfg.filter_cfg.plan
    if plan is None:
        return cfg
    return replace(cfg, filter_cfg=replace(cfg.filter_cfg, plan=replace(plan, rows=rows)))


def _row_length(cfg: RunConfig, length: float) -> RunConfig:
    """Rows of the given length (and the plan's, if any); the path walks all of it."""
    world = cfg.scenario.world
    rows = tuple(replace(row, y_end=row.y_start + length) for row in world.rows)
    weeds = world.weeds
    if weeds is not None:
        weeds = replace(weeds, region=replace(weeds.region, y_max=rows[0].y_end))
    path = cfg.scenario.path
    step = path.speed * path.scan_period
    n_scans = math.ceil((rows[0].y_end - path.y_start) / step) + 1
    scenario = replace(cfg.scenario, world=replace(world, rows=rows, weeds=weeds),
                       path=replace(path, n_scans=n_scans))
    return _with_plan_rows(replace(cfg, scenario=scenario), rows)


def _spacing(cfg: RunConfig, spacing: float) -> RunConfig:
    """Plants this far apart along every row (and in the plan, if any)."""
    world = cfg.scenario.world
    rows = tuple(replace(row, spacing=spacing) for row in world.rows)
    return _with_plan_rows(replace(cfg, scenario=replace(cfg.scenario,
                                                         world=replace(world, rows=rows))), rows)


def _lambda_FA(cfg: RunConfig, lambda_FA: float) -> RunConfig:
    """The clutter rate, on both sides, so the filter's model stays matched."""
    scenario = replace(cfg.scenario, sensor=replace(cfg.scenario.sensor, lambda_FA=lambda_FA))
    assumed = replace(cfg.filter_cfg.assumed_sensor, lambda_FA=lambda_FA)
    return replace(cfg, scenario=scenario,
                   filter_cfg=replace(cfg.filter_cfg, assumed_sensor=assumed))


def _weed_density(cfg: RunConfig, density: float) -> RunConfig:
    """Weeds per m^2; the scenario must already have a weeds block."""
    world = cfg.scenario.world
    if world.weeds is None:
        raise ValueError("the weed_density sweep needs a scenario with a weeds block")
    weeds = replace(world.weeds, density=density)
    return replace(cfg, scenario=replace(cfg.scenario, world=replace(world, weeds=weeds)))


@dataclass(frozen=True)
class Sweep:
    """One scenario parameter to vary, and the problem size it drives. [B4]

    Attributes:
        apply: (config, value) -> the config with that value.
        parameter: the swept parameter, with its unit, for the figure.
        size: which measured problem size the figure's x axis shows, one of the keys of
            `timing_measure`'s result.
        size_label: that size, for the figure.
    """

    apply: Callable[[RunConfig, float], RunConfig]
    parameter: str
    size: str
    size_label: str


# The four sweeps of roadmap step 4c.
SWEEPS: dict[str, Sweep] = {
    "row_length": Sweep(_row_length, "row length [m]", "n_plants", "plants in the field N"),
    "spacing": Sweep(_spacing, "plant spacing [m]", "plants_in_view",
                     "plants in view per scan"),
    "lambda_FA": Sweep(_lambda_FA, "clutter rate lambda_FA", "detections",
                       "detections per scan"),
    "weed_density": Sweep(_weed_density, "weed density [per m²]", "detections",
                          "detections per scan"),
}


def timing_measure(run: RunDir, cfg: RunConfig) -> dict[str, object]:
    """One run's update times and problem sizes, for `montecarlo.run_trials`. [B4]

    Returns:
        "update_s": update times of scans 1.. (scan 0 includes warm-up),
        "scan0_update_s": scan 0's update time,
        "n_plants", "plants_in_view", "detections", "tracks": the problem sizes - total
        plants, and the per-scan means of plants in view, detections and reported tracks.
    """
    timing = list(read_jsonl(run.timing(cfg.filter_cfg.kind)))
    labels = read_labels(run.labels)
    return {
        "update_s": [row["update_s"] for row in timing[1:]],
        "scan0_update_s": timing[0]["update_s"],
        "n_plants": float(len(read_truth(run.truth).field.positions)),
        "plants_in_view": float(np.mean([len(scan.visible_ids) for scan in labels])),
        "detections": float(np.mean([row["n_detections"] for row in timing])),
        "tracks": float(np.mean([row["n_reported"] for row in timing])),
    }


@dataclass(frozen=True)
class ScalingPoint:
    """One swept value, over its seeds. [B4]

    Attributes:
        value: the swept parameter's value.
        size: the problem size the sweep drives, averaged over seeds.
        tracks: reported tracks per scan, averaged over seeds.
        median_s, p95_s: the median and 95th percentile of the update time per scan,
            over every scan but scan 0 of every seed.
        scan0_median_s: the median of scan 0's update time over seeds.
        n_scans: how many scan times the percentiles are taken over.
    """

    value: float
    size: float
    tracks: float
    median_s: float
    p95_s: float
    scan0_median_s: float
    n_scans: int


def scaling_point(value: float, measures: list[dict[str, object]], size: str) -> ScalingPoint:
    """Aggregate one swept value's per-seed measures. [B4]"""
    times = np.concatenate([m["update_s"] for m in measures])
    return ScalingPoint(
        value=value,
        size=float(np.mean([m[size] for m in measures])),
        tracks=float(np.mean([m["tracks"] for m in measures])),
        median_s=float(np.median(times)),
        p95_s=float(np.percentile(times, 95)),
        scan0_median_s=float(np.median([m["scan0_update_s"] for m in measures])),
        n_scans=int(times.size),
    )


def loglog_slope(points: list[ScalingPoint]) -> float:
    """The empirical exponent: the slope of log(median time) against log(size). [B4]

    NaN with fewer than two distinct sizes.
    """
    sizes = np.array([p.size for p in points])
    if len(np.unique(sizes)) < 2 or np.any(sizes <= 0):
        return float("nan")
    slope, _ = np.polyfit(np.log(sizes), np.log([p.median_s for p in points]), 1)
    return float(slope)


@dataclass(frozen=True)
class ScalingResult:
    """A whole sweep: one point per value, the fitted exponent and the budget. [B4]

    Attributes:
        sweep: the key into SWEEPS.
        filter_name: the filter timed.
        points: one per swept value, in the order given.
        slope: `loglog_slope(points)`, the empirical exponent of the median time.
        budget_s: the real-time budget, the scenario's scan period.
        n_seeds: seeds per value.
        threads: the BLAS thread variables the times were taken under.
    """

    sweep: str
    filter_name: str
    points: tuple[ScalingPoint, ...]
    slope: float
    budget_s: float
    n_seeds: int
    threads: dict[str, str | None]
