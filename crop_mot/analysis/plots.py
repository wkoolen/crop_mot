"""Figures, written to the run folder's plots/ directory. [B1/B2/B3]

Every function SAVES a PNG and returns the path; none of them call plt.show(). The
container is headless (MPLBACKEND=Agg in the Dockerfile), so a show() would silently do
nothing there and behave differently on a desktop - and a figure that only exists on screen
cannot be traced back to the run that produced it.

Each figure is written into the run folder next to the data it was made from, so a thesis
figure is always one directory away from its config and its seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from crop_mot.analysis.estimates_log import read_estimates, r_trajectory
from crop_mot.analysis.events import gated_detection_indices, predicted_track_moments
from crop_mot.analysis.montecarlo import MonteCarloResult, standard_error
from crop_mot.config import (
    RunConfig,
    ScenarioConfig,
    load_run_config,
    load_scenario_config,
)
from crop_mot.runner.run_dir import RunDir
from crop_mot.sensor.fov import in_fov
from crop_mot.sensor.record import read_detections, read_labels
from crop_mot.world.truth import read_truth

# Colours: the first three slots of the dataviz reference palette (validated as a set for
# colour-vision deficiency) plus its chrome, light mode, since these figures go to print.
# Every figure with two or more series carries a legend, so colour never works alone.
SERIES_1 = "#2a78d6"   # blue: the primary quantity (r of the filter; real detections)
SERIES_2 = "#eb6834"   # orange: the comparison (analytic reference; clutter; gated scans)
SERIES_3 = "#1baf7a"   # aqua: a third marker (birth)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

# Half-width of the Monte-Carlo confidence band, in standard errors of the mean.
MC_BAND_SE = 3.0

# matplotlib's Figure API is used directly, never pyplot: no global figure state and no
# backend that could try to open a window.


def _new_figure(n_rows: int = 1, height: float = 3.6,
                height_ratios: list[float] | None = None) -> tuple[Figure, list]:
    """A figure with n_rows stacked axes sharing x, styled with recessive chrome."""
    fig = Figure(figsize=(7.0, height), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(n_rows, 1, sharex=True, squeeze=False,
                        gridspec_kw={"height_ratios": height_ratios})[:, 0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, list(axes)


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def _load_config(run: RunDir) -> ScenarioConfig | RunConfig:
    """The run folder's config copy: a RunConfig for a `track` run, a ScenarioConfig for a
    `simulate` run. A run config is recognised by its `scenario:` key."""
    raw = yaml.safe_load(run.config.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scenario" in raw:
        return load_run_config(run.config)
    return load_scenario_config(run.config)


def plot_scene(run: RunDir, out: Path, k: int = 0) -> Path:
    """Top-down view of the field, the robot path and one scan's detections. [B1]

    The sanity-check figure for B1, and the one that catches geometry bugs fastest: plants
    as points, the robot path as a line, the FOV wedge at a chosen scan, and that scan's
    detections. Clutter and true detections are distinguishable here because this is
    evaluation code and may read labels.jsonl.

    Args:
        run: the run folder to read from.
        out: destination PNG path.
        k: the scan whose FOV wedge and detections are drawn (default 0). Added during
            implementation: the docstring asks for "a chosen scan"; `analyse` passes the
            birth scan so the phantom's seed is visible.

    Returns:
        The path written.
    """
    cfg = _load_config(run)
    scenario = cfg.scenario if isinstance(cfg, RunConfig) else cfg
    fov = scenario.sensor.fov
    truth = read_truth(run.truth)
    scan = read_detections(run.detections)[k]
    labels = read_labels(run.labels)[k]
    pose = truth.poses[k].true

    fig, (ax,) = _new_figure(height=6.0)

    wedge = Wedge((pose.x, pose.y), fov.max_range,
                  np.degrees(pose.theta - fov.half_angle), np.degrees(pose.theta + fov.half_angle),
                  width=fov.max_range - fov.min_range, facecolor=to_rgba(SERIES_1, 0.08),
                  edgecolor=MUTED, linewidth=1.0, label=f"FOV at scan {k}")
    ax.add_patch(wedge)

    path = np.array([[sample.true.x, sample.true.y] for sample in truth.poses])
    ax.plot(path[:, 0], path[:, 1], color=INK, linewidth=2.0, label="robot path")
    ax.plot(pose.x, pose.y, marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
            markeredgewidth=2.0, linestyle="none", label=f"robot at scan {k}")

    plants = truth.field.positions
    ax.plot(plants[:, 0], plants[:, 1], marker="o", markersize=6, linestyle="none",
            markerfacecolor="none", markeredgecolor=INK_SECONDARY, label="plants (truth)")

    real = np.array([d.z for d, o in zip(scan.detections, labels.origin) if o is not None])
    clutter = np.array([d.z for d, o in zip(scan.detections, labels.origin) if o is None])
    if len(real):
        ax.plot(real[:, 0], real[:, 1], marker="o", markersize=6, color=SERIES_1,
                linestyle="none", label="detection from a plant")
    if len(clutter):
        ax.plot(clutter[:, 0], clutter[:, 1], marker="X", markersize=9, color=SERIES_2,
                markeredgecolor=SURFACE, linestyle="none", label="clutter detection")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Scene: {scenario.name}, detections at scan {k}", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_r_vs_k(run: RunDir, filter_name: str, track_id: int, out: Path) -> Path:
    """Existence probability against scan index. THE B2 DELIVERABLE. [B2]

    For the phantom scenario this is the r-decay curve: a track born from a clutter
    detection, whose existence probability falls as scan after scan fails to produce a
    detection at that location.

    Should mark which scans produced a gated detection, since a flat stretch in r is only
    interesting once you know whether it was caused by a detection or by the target leaving
    the FOV.

    Both markings are reconstructed from the run folder exactly as the filter saw them:
    the filter's predicted moments come from its estimates log
    (`events.predicted_track_moments`), the gate from `events.gated_detection_indices`,
    and "out of view" means the predicted mean was outside the ASSUMED FOV at the scan's
    reported pose - so p_D was 0 and the scan could not change r. Scans before the birth
    have no predicted moments and are not marked. Reads the run config, detections and
    estimates; no truth.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        track_id: which track to follow.
        out: destination PNG path.

    Returns:
        The path written.
    """
    cfg = load_run_config(run.config)
    records = read_estimates(run.estimates(filter_name))
    scans = read_detections(run.detections)
    r = r_trajectory(records, track_id)
    k = np.array([record.k for record in records])

    moments = predicted_track_moments(records, track_id, [scan.t for scan in scans],
                                      cfg.filter_cfg)
    gated_k = []
    out_of_view_k = []
    for scan, moment in zip(scans, moments):
        if moment is None:
            continue
        mean, cov = moment
        if not in_fov(mean, scan.pose, cfg.filter_cfg.assumed_sensor.fov):
            out_of_view_k.append(scan.k)
        elif len(gated_detection_indices(scan, mean, cov, cfg.filter_cfg)) > 0:
            gated_k.append(scan.k)

    fig, (ax,) = _new_figure()
    for i, scan_k in enumerate(out_of_view_k):
        ax.axvspan(scan_k - 0.5, scan_k + 0.5, color=MUTED, alpha=0.15, linewidth=0,
                   label="track out of view (p_D = 0)" if i == 0 else None)
    ax.plot(k, r, color=SERIES_1, linewidth=2.0, solid_joinstyle="round",
            solid_capstyle="round", label=f"r, track {track_id}")
    if gated_k:
        ax.plot(gated_k, r[np.searchsorted(k, gated_k)], marker="o", markersize=7,
                linestyle="none", color=SERIES_2, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="scan with a gated detection")
    born = np.flatnonzero(r > 0.0)
    if len(born):
        ax.plot(k[born[0]], r[born[0]], marker="D", markersize=7, linestyle="none",
                color=SERIES_3, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"birth, r_b = {r[born[0]]:g}")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    ax.set_xlabel("scan k")
    ax.set_ylabel("existence probability r")
    ax.set_title(f"Existence probability, {filter_name} filter: {cfg.name}", color=INK,
                 fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_r_vs_analytic(
    r_sim: np.ndarray, r_ref: np.ndarray, out: Path, title: str = ""
) -> Path:
    """The B3 cross-check figure: filter r against the closed form. [B3]

    Two curves plus their difference. The difference panel is the useful one - on the same
    axes the two curves are indistinguishable when the check passes, which makes a passing
    result look like a single line and an informative figure look like a bug.

    Args:
        r_sim: shape (K,), the filter's r trajectory.
        r_ref: shape (K,), the analytic reference's.
        out: destination PNG path.
        title: optional title, e.g. the p_D profile being validated.

    Returns:
        The path written.
    """
    k = np.arange(len(r_sim))
    fig, (ax, ax_diff) = _new_figure(n_rows=2, height=5.0, height_ratios=[2.0, 1.0])

    ax.plot(k, r_sim, color=SERIES_1, linewidth=2.0, label="filter r")
    ax.plot(k, r_ref, color=SERIES_2, linewidth=2.0, linestyle=(0, (4, 3)),
            label="closed form (A2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("existence probability r")
    ax.set_title(title or "Filter r against the analytic reference", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    ax_diff.axhline(0.0, color=MUTED, linewidth=1.0)
    ax_diff.plot(k, np.asarray(r_sim) - np.asarray(r_ref), color=SERIES_1, linewidth=2.0)
    ax_diff.set_xlabel("scan k")
    ax_diff.set_ylabel("r filter - r closed form")
    return _save(fig, out)


def plot_r_montecarlo(result: MonteCarloResult, r_ref: np.ndarray, out: Path) -> Path:
    """Mean r over many seeds, with a confidence band, against the closed form. [B3]

    The empirical half of B3. Plots the mean plus or minus a few standard errors - NOT plus
    or minus the standard deviation, which measures the spread of individual runs rather
    than the uncertainty in the mean, and would produce a band wide enough to hide a real
    modelling error.

    Args:
        result: aggregate from `run_monte_carlo`.
        r_ref: shape (K,), the closed-form reference.
        out: destination PNG path.

    Returns:
        The path written.
    """
    k = np.arange(len(result.r_mean))
    band = MC_BAND_SE * standard_error(result)

    fig, (ax,) = _new_figure()
    ax.fill_between(k, result.r_mean - band, result.r_mean + band, color=SERIES_1, alpha=0.12,
                    linewidth=0, label=f"mean \u00b1 {MC_BAND_SE:g} standard errors")
    ax.plot(k, result.r_mean, color=SERIES_1, linewidth=2.0,
            label=f"mean r over {result.n_runs} runs")
    ax.plot(k, r_ref, color=SERIES_2, linewidth=2.0, linestyle=(0, (4, 3)),
            label="closed form (A2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("scan k")
    ax.set_ylabel("existence probability r")
    ax.set_title("Monte-Carlo mean r against the closed form", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)
