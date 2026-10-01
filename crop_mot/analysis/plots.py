"""Figures, written to the run folder's plots/ directory. [B1/B2/B3]

Every function SAVES a PNG and returns the path; none of them call plt.show(). The
container is headless (MPLBACKEND=Agg in the Dockerfile), so a show() would silently do
nothing there and behave differently on a desktop - and a figure that only exists on screen
cannot be traced back to the run that produced it.

Each figure is written into the run folder next to the data it was made from, so a thesis
figure is always one directory away from its config and its seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Wedge
from matplotlib.ticker import FuncFormatter, MaxNLocator

from crop_mot.analysis.counts import origin_kinds, scan_counts, weed_origin
from crop_mot.analysis.estimates_log import (
    TrackLifetime,
    r_trajectory,
    read_estimates,
    track_lifetimes,
)
from crop_mot.analysis.evaluation import (
    GOSPA_C,
    GOSPA_P,
    R_CONF,
    cardinality,
    existence_density,
    gate_contents,
    gospa_series,
    missing_plants,
    nees,
    nees_band,
    scan_views,
)
from crop_mot.analysis.events import gated_detection_indices, predicted_track_moments
from crop_mot.analysis.fates import FATES, STRATA, FateProportion
from crop_mot.analysis.montecarlo import MonteCarloResult, standard_error
from crop_mot.analysis.timing import SWEEPS, ScalingResult
from crop_mot.config import (
    RunConfig,
    ScenarioConfig,
    load_run_config,
    load_scenario_config,
)
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import unpruned_log_name
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

# Weeds (decision D15) get no hue of their own: the scene is a scatter, where only the first
# three slots validate all-pairs. A weed detection is a false alarm, so it keeps the clutter
# orange and is told apart by shape - a triangle, against the transient clutter's x - and a
# weed itself is a hollow neutral triangle, as a plant is a hollow neutral circle. In the
# stacked counts bars the weed segment is the clutter orange, hatched.
WEED_MARKER = "^"
WEED_HATCH = "////"

# Half-width of the Monte-Carlo confidence band, in standard errors of the mean.
MC_BAND_SE = 3.0

# The reserved status colour for a deleted hypothesis (the palette's "critical"). A status
# colour, never a series slot, and it always ships with the x icon and a "deleted" label.
DELETED = "#d03b3b"
# Floor of the log-r axis in the hypotheses figure; r below it is clipped.
R_LOG_FLOOR = 1e-5
# Radius of the uncertainty ellipse drawn at a hypothesis, in standard deviations.
ELLIPSE_SIGMA = 2.0

# matplotlib's Figure API is used directly, never pyplot: no global figure state and no
# backend that could try to open a window.


def _new_figure(n_rows: int = 1, height: float = 3.6,
                height_ratios: list[float] | None = None) -> tuple[Figure, list]:
    """A figure with n_rows stacked axes sharing x, styled with recessive chrome."""
    fig = Figure(figsize=(7.0, height), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(n_rows, 1, sharex=True, squeeze=False,
                        gridspec_kw={"height_ratios": height_ratios})[:, 0]
    for ax in axes:
        _style_axes(ax)
    return fig, list(axes)


def _style_axes(ax) -> None:
    """Recessive chrome: light grid behind the data, no top/right spines, muted ticks."""
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


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def load_run_folder_config(run: RunDir) -> ScenarioConfig | RunConfig:
    """The run folder's config copy: a RunConfig for a `track` run, a ScenarioConfig for a
    `simulate` run. A run config is recognised by its `scenario:` key."""
    raw = yaml.safe_load(run.config.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scenario" in raw:
        return load_run_config(run.config)
    return load_scenario_config(run.config)


def _fov_wedge(pose, fov, **style) -> Wedge:
    """The FOV annulus sector at a pose, as a patch to add to a scene axis."""
    return Wedge((pose.x, pose.y), fov.max_range,
                 np.degrees(pose.theta - fov.half_angle), np.degrees(pose.theta + fov.half_angle),
                 width=fov.max_range - fov.min_range, linewidth=1.0, **style)


def _draw_field(ax, truth, robot_at: int | None = None) -> None:
    """The robot path, the plants, any weeds and optionally the robot at one scan."""
    path = np.array([[sample.true.x, sample.true.y] for sample in truth.poses])
    ax.plot(path[:, 0], path[:, 1], color=INK, linewidth=2.0, label="robot path")
    if robot_at is not None:
        pose = truth.poses[robot_at].true
        ax.plot(pose.x, pose.y, marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=2.0, linestyle="none", label=f"robot at scan {robot_at}")
    plants = truth.field.positions
    ax.plot(plants[:, 0], plants[:, 1], marker="o", markersize=6, linestyle="none",
            markerfacecolor="none", markeredgecolor=INK_SECONDARY, label="plants (truth)")
    if len(truth.weeds):
        ax.plot(truth.weeds[:, 0], truth.weeds[:, 1], marker=WEED_MARKER, markersize=8,
                linestyle="none", markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                markeredgewidth=1.2, label="weeds (truth)")


def plot_scene(run: RunDir, out: Path, k: int = 0) -> Path:
    """Top-down view of the field, the robot path and one scan's detections. [B1]

    The sanity-check figure for B1, and the one that catches geometry bugs fastest: plants
    as points, the robot path as a line, the FOV wedge at a chosen scan, and that scan's
    detections. Clutter and true detections are distinguishable here because this is
    evaluation code and may read labels.jsonl; so are weed detections and transient
    clutter, for a scenario with weeds.

    Args:
        run: the run folder to read from.
        out: destination PNG path.
        k: the scan whose FOV wedge and detections are drawn (default 0). Added during
            implementation: the docstring asks for "a chosen scan"; `analyse` passes the
            birth scan so the phantom's seed is visible.

    Returns:
        The path written.
    """
    cfg = load_run_folder_config(run)
    scenario = cfg.scenario if isinstance(cfg, RunConfig) else cfg
    fov = scenario.sensor.fov
    truth = read_truth(run.truth)
    scan = read_detections(run.detections)[k]
    labels = read_labels(run.labels)[k]
    pose = truth.poses[k].true

    fig, (ax,) = _new_figure(height=6.0)

    ax.add_patch(_fov_wedge(pose, fov, facecolor=to_rgba(SERIES_1, 0.08), edgecolor=MUTED,
                            label=f"FOV at scan {k}"))
    _draw_field(ax, truth, robot_at=k)

    sources = list(zip(scan.detections, labels.origin, weed_origin(labels)))
    real = np.array([d.z for d, o, _ in sources if o is not None])
    clutter = np.array([d.z for d, o, w in sources if o is None and w is None])
    weed = np.array([d.z for d, _, w in sources if w is not None])
    if len(real):
        ax.plot(real[:, 0], real[:, 1], marker="o", markersize=6, color=SERIES_1,
                linestyle="none", label="detection from a plant")
    if len(clutter):
        ax.plot(clutter[:, 0], clutter[:, 1], marker="X", markersize=9, color=SERIES_2,
                markeredgecolor=SURFACE, linestyle="none", label="clutter detection")
    if len(weed):
        ax.plot(weed[:, 0], weed[:, 1], marker=WEED_MARKER, markersize=9, color=SERIES_2,
                markeredgecolor=SURFACE, linestyle="none", label="detection from a weed")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Scene: {scenario.name}, detections at scan {k}", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_counts(run: RunDir, out: Path) -> Path:
    """Clairvoyant per-scan counts: detections received against plants in view. [B1]

    Each scan's detections as a stacked bar, split by true origin (from a plant / clutter,
    and for a scenario with weeds, clutter from a weed as a hatched segment on top), with
    the number of plants inside the FOV drawn as a line on the same count axis. The
    gap between the line and the plant-origin bar is the misses plus the FOV-edge losses;
    once multiplicity is on, the plant-origin bar can rise above the line. Evaluation code,
    so it reads labels.jsonl. Works on a simulate-only run folder.

    Args:
        run: the run folder to read from.
        out: destination PNG path.

    Returns:
        The path written.
    """
    cfg = load_run_folder_config(run)
    scenario = cfg.scenario if isinstance(cfg, RunConfig) else cfg
    counts = scan_counts(read_labels(run.labels))
    k = np.array([c.k for c in counts])
    from_plants = np.array([c.n_object_detections for c in counts])
    clutter = np.array([c.n_transient for c in counts])
    weed = np.array([c.n_weed for c in counts])
    visible = np.array([c.n_visible for c in counts])

    fig, (ax,) = _new_figure()
    # A surface-coloured edge keeps a visible gap between adjacent bars and segments.
    bar = {"width": 0.9, "edgecolor": SURFACE, "linewidth": 1.0}
    ax.bar(k, from_plants, color=SERIES_1, label="detections from plants", **bar)
    weeds = any(c.n_visible_weeds for c in counts)
    ax.bar(k, clutter, bottom=from_plants, color=SERIES_2,
           label="transient clutter" if weeds else "clutter detections", **bar)
    if weeds:
        # The hatch is drawn in the edge colour, so the stripes are surface-coloured.
        ax.bar(k, weed, bottom=from_plants + clutter, color=SERIES_2, hatch=WEED_HATCH,
               label="clutter from weeds", **bar)
    ax.plot(k, visible, color=INK, linewidth=2.0, solid_joinstyle="round",
            solid_capstyle="round", label="plants in FOV (truth)")

    ax.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("scan k")
    ax.set_ylabel("count")
    ax.set_title(f"Detections per scan against plants in view: {scenario.name}", color=INK,
                 fontsize=11)
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

    gated_k, out_of_view_k = _scan_marks(records, track_id, scans, cfg.filter_cfg)

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


def _scan_marks(records, track_id: int, scans, filter_cfg) -> tuple[list[int], list[int]]:
    """Scans where the track had a gated detection, and scans where it was out of view.

    Reconstructed from the estimates log exactly as the filter saw each scan: the predicted
    moments from `events.predicted_track_moments`, the gate from
    `events.gated_detection_indices`, and "out of view" as the predicted mean outside the
    ASSUMED FOV at the reported pose (p_D = 0, so the scan cannot change r). Scans at or
    before the birth, or after a deletion, have no predicted moments and are in neither.

    Returns:
        Tuple (gated_k, out_of_view_k) of scan indices.
    """
    moments = predicted_track_moments(records, track_id, [scan.t for scan in scans],
                                      filter_cfg)
    gated_k = []
    out_of_view_k = []
    for scan, moment in zip(scans, moments):
        if moment is None:
            continue
        mean, cov = moment
        if not in_fov(mean, scan.pose, filter_cfg.assumed_sensor.fov):
            out_of_view_k.append(scan.k)
        elif len(gated_detection_indices(scan, mean, cov, filter_cfg)) > 0:
            gated_k.append(scan.k)
    return gated_k, out_of_view_k


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


def plot_r_montecarlo(result: MonteCarloResult, r_ref: np.ndarray | None, out: Path) -> Path:
    """Mean r over many seeds, with a confidence band. Descriptive (decision D17). [B3]

    Plots the mean plus or minus a few standard errors - NOT plus or minus the standard
    deviation, which measures the spread of individual runs rather than the uncertainty in
    the mean.

    The closed form is drawn only when `r_ref` is given, and it is only meaningful where
    every seed has the same event sequence - a phantom that sees nothing but misses. In
    general the mean of r is not the closed form at any one event sequence, so `analyse`
    draws none; the per-seed check is `montecarlo.cross_check_summary`.

    Args:
        result: aggregate from `run_monte_carlo` or `mean_r`.
        r_ref: shape (K,), a closed-form reference to draw, or None.
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
    if r_ref is not None:
        ax.plot(k, r_ref, color=SERIES_2, linewidth=2.0, linestyle=(0, (4, 3)),
                label="closed form (A2)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("scan k")
    ax.set_ylabel("existence probability r")
    ax.set_title("Monte-Carlo mean r of the first track (descriptive)", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


FATE_LABELS = {
    "pruned": "pruned",
    "confirmed_on_plant": "confirmed on a plant",
    "confirmed_on_weed": "confirmed on a weed",
    "sustained_by_clutter": "sustained by clutter",
    "alive_unconfirmed": "alive, unconfirmed",
}
STRATUM_LABELS = {
    "all": "all seeds",
    "weed_in_gate": "weed return in gate",
    "no_weed_in_gate": "no weed return in gate",
}


def plot_phantom_fates(proportions: dict[str, dict[str, FateProportion]], out: Path,
                       title: str = "") -> Path:
    """The controlled phantom's fate over seeds, per stratum, with Wilson intervals. [B3, 4a]

    One panel per stratum (all seeds, then with and without a weed return in the gate),
    one bar per fate: the share of that stratum's seeds, its 95 % Wilson interval, and
    the count behind it. Fates are told apart by their axis labels, not by colour.

    Args:
        proportions: from `fates.fate_proportions`.
        out: destination PNG path.
        title: optional figure title.

    Returns:
        The path written.
    """
    fig = Figure(figsize=(9.0, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(STRATA), sharey=True, sharex=True)
    rows = np.arange(len(FATES))
    for ax, stratum in zip(axes, STRATA):
        _style_axes(ax)
        shares = proportions[stratum]
        n = shares[FATES[0]].n
        values = np.array([shares[f].count / n if n else 0.0 for f in FATES])
        low = np.array([shares[f].low for f in FATES])
        high = np.array([shares[f].high for f in FATES])
        ax.barh(rows, values, height=0.6, color=SERIES_1, edgecolor=SURFACE, linewidth=1.0)
        ax.errorbar(values, rows, xerr=[values - low, high - values], fmt="none",
                    ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3)
        for row, fate in zip(rows, FATES):
            ax.annotate(f"{shares[fate].count}/{n}", (high[row], row), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK_SECONDARY,
                        fontsize=8)
        ax.set_xlim(0.0, 1.15)
        ax.set_xlabel("share of seeds")
        ax.set_title(f"{STRATUM_LABELS[stratum]} (n = {n})", color=INK, fontsize=10)
    axes[0].set_yticks(rows, [FATE_LABELS[f] for f in FATES])
    axes[0].invert_yaxis()
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    return _save(fig, out)


def plot_scaling(result: ScalingResult, out: Path) -> Path:
    """Update time per scan against problem size, with the real-time budget. [B4, 4c]

    Log-log axes: the median and the 95th percentile of the update time per scan (scan 0
    excluded, it includes warm-up) at each swept value, against the problem size measured
    in those runs, and the scan period as the budget line. The title gives the fitted
    exponent of the median, the number to compare with the method's expected cost.

    Args:
        result: from `runner.scaling.run_scaling`.
        out: destination PNG path.

    Returns:
        The path written.
    """
    spec = SWEEPS[result.sweep]
    size = np.array([p.size for p in result.points])
    fig, (ax,) = _new_figure()
    ax.axhline(result.budget_s, color=INK_SECONDARY, linewidth=1.0,
               label=f"real-time budget: scan period {result.budget_s:g} s")
    ax.plot(size, [p.p95_s for p in result.points], color=SERIES_2, linewidth=2.0,
            marker="o", markersize=6, label="95th percentile")
    ax.plot(size, [p.median_s for p in result.points], color=SERIES_1, linewidth=2.0,
            marker="o", markersize=6, label="median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)
    ax.set_xlabel(spec.size_label)
    ax.set_ylabel("update time per scan [s]")
    slope = "n/a" if np.isnan(result.slope) else f"{result.slope:.2f}"
    ax.set_title(f"Time per scan, {result.filter_name}: slope of the median {slope}\n"
                 f"sweep of {spec.parameter}, {result.n_seeds} seeds per value", color=INK,
                 fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_gate_contents(run: RunDir, filter_name: str, out: Path) -> Path:
    """What fell in the tracks' gates, by true origin, and the tracks pulled away. [B4, 8a]

    Top: the mean number of gated detections per in-view track, stacked by origin - the
    track's own plant, a neighbour plant, Poisson clutter, a weed. Bottom: the share of
    in-view tracks whose mean sits nearer another plant than their own. The failure modes
    of roadmap step 8a, from the labels; the legend gives the run's shares.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    contents = gate_contents(run, filter_name)
    k = np.arange(len(contents.own))
    shares = contents.shares
    fig, (ax, ax_pulled) = _new_figure(n_rows=2, height=5.2, height_ratios=[2.0, 1.0])
    layers = [np.nan_to_num(values) for values in (contents.own, contents.neighbour,
                                                   contents.clutter, contents.weed)]
    polygons = ax.stackplot(
        k, layers, colors=[SERIES_1, SERIES_3, SERIES_2, SERIES_2], edgecolor=SURFACE,
        linewidth=1.0,
        labels=["own plant",
                f"a neighbour plant (in {shares['neighbour_share']:.0%} of gates)",
                f"Poisson clutter (in {shares['clutter_share']:.0%})",
                f"a weed (in {shares['weed_share']:.0%})"])
    polygons[3].set_hatch(WEED_HATCH)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("detections per gate")
    ax.set_title(f"Gate contents by true origin, {filter_name}", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    ax_pulled.plot(k, contents.pulled, color=INK, linewidth=2.0, drawstyle="steps-mid",
                   label=f"pulled: {shares['pulled_share']:.1%} of track-scans")
    ax_pulled.set_ylim(-0.02, 1.02)
    ax_pulled.set_ylabel("share pulled")
    ax_pulled.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    _time_axis(ax_pulled, len(k))
    return _save(fig, out)


def _not_applicable(out: Path, title: str, reason: str) -> Path:
    """A labelled placeholder panel: the figure does not apply to this run (§2 rule 1).

    Roadmap §2 rule 1: a figure may declare a filter or a run "not applicable"; it then
    renders this panel, which the figure contract test accepts, instead of raising.
    """
    fig, (ax,) = _new_figure(height=2.4)
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"not applicable: {reason}", ha="center", va="center",
            color=INK_SECONDARY, fontsize=11, transform=ax.transAxes)
    ax.set_title(title, color=INK, fontsize=11)
    return _save(fig, out)


def plot_missing_plants(run: RunDir, filter_name: str, out: Path) -> Path:
    """Finding the plan's empty slots from r: over time, and over the threshold. [B4, 8c]

    Left: every seen slot's score over time (the r of the track on it, 0 without one) -
    planted slots thin, empty slots bold - with the confirmation threshold r_conf. Right,
    at the last scan over the slots seen: the share of empty slots found (score below a
    threshold) against the share of planted slots wrongly flagged, as the threshold goes
    from 0 to 1, with r_conf marked. "Not applicable" for a field without missing plants.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    title = f"Finding missing plants, {filter_name}"
    result = missing_plants(run, filter_name)
    if result is None:
        return _not_applicable(out, title, "the field has no missing plants (p_missing = 0)")

    found, flagged = result.rates()
    fig = Figure(figsize=(9.0, 3.8), facecolor=SURFACE, layout="constrained")
    ax, ax_rates = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1.0]})
    for axis in (ax, ax_rates):
        _style_axes(axis)
    k = np.arange(result.scores.shape[0])
    seen = result.seen[-1]
    for j in np.flatnonzero(seen & ~result.empty):
        ax.plot(k, result.scores[:, j], color=SERIES_1, linewidth=1.0, alpha=0.35)
    for j in np.flatnonzero(seen & result.empty):
        ax.plot(k, result.scores[:, j], color=SERIES_2, linewidth=2.2)
    ax.axhline(R_CONF, color=INK_SECONDARY, linewidth=1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    ax.set_xlabel("scan k")
    ax.set_ylabel("r of the track on the slot")
    ax.legend(handles=[
        Line2D([], [], color=SERIES_1, linewidth=1.0, alpha=0.6,
               label=f"planted slot ({int((seen & ~result.empty).sum())} seen)"),
        Line2D([], [], color=SERIES_2, linewidth=2.2,
               label=f"empty slot ({int((seen & result.empty).sum())} seen)"),
        Line2D([], [], color=INK_SECONDARY, linewidth=1.0, label=f"r_conf = {R_CONF:g}"),
    ], loc="lower left", frameon=False, fontsize=8)

    thresholds = np.concatenate([[0.0], np.unique(result.scores[-1]), [1.0 + 1e-9]])
    curve = np.array([result.rates(t) for t in thresholds])
    ax_rates.plot(curve[:, 1], curve[:, 0], color=SERIES_2, linewidth=2.0,
                  drawstyle="steps-post")
    ax_rates.plot(flagged, found, marker="o", markersize=8, color=SERIES_2,
                  markeredgecolor=SURFACE, linestyle="none")
    ax_rates.annotate(f"r_conf = {R_CONF:g}", (flagged, found), xytext=(8, 6),
                      textcoords="offset points", color=INK_SECONDARY, fontsize=8)
    ax_rates.set_xlim(-0.02, 1.02)
    ax_rates.set_ylim(-0.02, 1.02)
    ax_rates.set_xlabel("planted slots flagged empty")
    ax_rates.set_ylabel("empty slots found")
    fig.suptitle(f"{title}: {found:.0%} of empty slots found, {flagged:.0%} of plants "
                 f"flagged (last scan, r < {R_CONF:g})", color=INK, fontsize=11)
    return _save(fig, out)


def plot_yaw_sensitivity(points: list, out: Path, title: str = "") -> Path:
    """NEES, its in-band share and GOSPA against a constant yaw bias. [B4, step 8d]

    One series per heading-wobble amplitude; each point is the mean over seeds, with a
    bar from the lowest to the highest seed (one outcome per seed). Left: the average NEES
    on a log axis against its expected value 2 - it leaves the band once the heading error
    moves detections by more than the posterior std. Middle: the share of scans whose
    average NEES is inside the 95 % band. Right: mean GOSPA.

    Args:
        points: the `runner.yaw_sensitivity.YawPoint`s of one sweep.
        out: destination PNG path.
        title: optional figure title.

    Returns:
        The path written.
    """
    fig = Figure(figsize=(10.0, 3.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 3, sharex=True)
    wobbles = sorted({p.yaw_wobble_std_deg for p in points})
    colours = (SERIES_1, SERIES_2, SERIES_3)
    panels = (("mean_nees", "average NEES"), ("nees_in_band", "scans with NEES in band"),
              ("mean_gospa", "mean GOSPA [m]"))
    for ax, (key, label) in zip(axes, panels):
        _style_axes(ax)
        for colour, wobble in zip(colours, wobbles):
            series = [p for p in points if p.yaw_wobble_std_deg == wobble]
            bias = np.array([p.yaw_bias_deg for p in series])
            values = [np.array(getattr(p, key), dtype=float) for p in series]
            mean = np.array([np.nanmean(v) for v in values])
            spread = np.array([[m - np.nanmin(v), np.nanmax(v) - m]
                               for m, v in zip(mean, values)]).T
            ax.errorbar(bias, mean, yerr=spread, color=colour, linewidth=2.0, marker="o",
                        markersize=6, capsize=3, label=f"wobble {wobble:g}°")
        ax.set_xlabel("yaw bias [°]")
        ax.set_ylabel(label)
    axes[0].axhline(2.0, color=INK_SECONDARY, linewidth=1.0, label="expected value 2")
    axes[0].set_yscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    axes[0].yaxis.set_major_formatter(plain)
    axes[0].yaxis.set_minor_formatter(plain)
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    return _save(fig, out)


def _time_axis(ax, n_scans: int) -> None:
    ax.set_xlim(-0.5, n_scans - 0.5)
    ax.set_xlabel("scan k")


def plot_cardinality(run: RunDir, filter_name: str, out: Path) -> Path:
    """Expected number of tracks in view against plants in view (D21). [B4, step 5]

    Sum of r over the tracks whose mean is in view, and the number of true plants in view,
    decided by the same FOV function on both sides. Reads only the run folder, so it works
    for any filter.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    count = cardinality(scan_views(run, filter_name))
    k = np.arange(len(count.sum_r))

    fig, (ax,) = _new_figure()
    ax.plot(k, count.n_plants, color=INK, linewidth=2.0, drawstyle="steps-mid",
            label="plants in view (truth)")
    ax.plot(k, count.sum_r, color=SERIES_1, linewidth=2.0, solid_joinstyle="round",
            solid_capstyle="round", label="sum of r, tracks in view")
    _time_axis(ax, len(k))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("number of objects")
    ax.set_title(f"Cardinality in view, {filter_name}", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_gospa(run: RunDir, filter_name: str, out: Path) -> Path:
    """GOSPA over time, and its split into localisation, missed and false (D31). [B4, step 5]

    Top: the GOSPA distance per scan, between the confirmed tracks in view (r > r_conf)
    and the plants in view. Bottom: GOSPA^p stacked from its three parts - the assigned
    pairs' d^p, c^p / 2 per missed plant and c^p / 2 per false track - which add up to
    the top curve raised to p.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    results = gospa_series(scan_views(run, filter_name))
    k = np.arange(len(results))
    half = GOSPA_C**GOSPA_P / 2.0
    parts = [
        (np.array([r.localisation for r in results]), SERIES_1, "localisation, assigned pairs"),
        (half * np.array([r.n_missed for r in results]), SERIES_2, "missed plants"),
        (half * np.array([r.n_false for r in results]), SERIES_3, "false tracks"),
    ]

    fig, (ax, ax_parts) = _new_figure(n_rows=2, height=5.2, height_ratios=[1.0, 1.3])
    ax.plot(k, [r.distance for r in results], color=INK, linewidth=2.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("GOSPA [m]")
    ax.set_title(f"GOSPA in view, {filter_name} (r > {R_CONF:g}, c = {GOSPA_C:g} m)", color=INK,
                 fontsize=11)

    ax_parts.stackplot(k, [values for values, _, _ in parts],
                       colors=[colour for _, colour, _ in parts],
                       labels=[label for _, _, label in parts],
                       edgecolor=SURFACE, linewidth=1.0)
    ax_parts.set_ylim(bottom=0)
    ax_parts.set_ylabel(f"GOSPA^{GOSPA_P:g} [m^{GOSPA_P:g}]")
    _time_axis(ax_parts, len(k))
    ax_parts.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_nees(run: RunDir, filter_name: str, out: Path) -> Path:
    """Average NEES of the matched confirmed tracks, against its 95 % band. [B4, step 5]

    The check on the Gaussian part that B3's existence check cannot give: an average above
    the band means the reported covariances are too small (overconfident), below it too
    large. Pairs are GOSPA's assignment (D31); scans without a pair leave a gap. The lower
    panel shows how many pairs each scan averages, since the band narrows as that grows.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    views = scan_views(run, filter_name)
    error = nees(views, gospa_series(views))
    lower, upper = nees_band(error.n_pairs, error.dim)
    k = np.arange(len(error.mean_nees))

    fig, (ax, ax_n) = _new_figure(n_rows=2, height=5.2, height_ratios=[2.0, 1.0])
    ax.fill_between(k, lower, upper, color=MUTED, alpha=0.18, linewidth=0, step="mid",
                    label=f"95 % band, chi-square({error.dim} n) / n")
    ax.axhline(error.dim, color=MUTED, linewidth=1.0, label=f"expected value {error.dim}")
    ax.plot(k, error.mean_nees, color=SERIES_1, linewidth=2.0, marker="o", markersize=4,
            label="average NEES")
    if np.any(error.n_pairs > 0):
        ax.set_yscale("log")
        plain = FuncFormatter(lambda value, _: f"{value:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
    ax.set_ylabel("NEES")
    ax.set_title(f"NEES of confirmed tracks matched to plants, {filter_name}", color=INK,
                 fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    ax_n.plot(k, error.n_pairs, color=INK_SECONDARY, linewidth=2.0, drawstyle="steps-mid")
    ax_n.set_ylim(bottom=0)
    ax_n.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_n.set_ylabel("pairs n")
    _time_axis(ax_n, len(k))
    return _save(fig, out)


# The palette's blue ramp from the surface to step 700: one hue, light to dark, for the
# existence map's magnitude. Zero recedes into the surface.
EXISTENCE_CMAP = LinearSegmentedColormap.from_list(
    "existence", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# Grid spacing of the existence map, metres.
EXISTENCE_GRID_STEP = 0.05
# Decades the existence map's log colour scale spans below its top. A slot on a 0.25 m
# prior peaks at 1 / (2 pi 0.25^2) = 2.5 per m^2 and a converged track near 100, so with
# 3 decades the prior sits mid-scale, coloured out to about 2.5 sigma, and each update
# visibly darkens and shrinks it.
EXISTENCE_DECADES = 3


def _existence_norm(vmax: float) -> LogNorm:
    """The existence map's colour scale: logarithmic, EXISTENCE_DECADES decades below vmax.

    Clipped, so a density below the bottom, zero included, takes the ramp's first colour,
    the surface.
    """
    vmax = vmax if vmax > 0.0 else 1.0
    return LogNorm(vmin=vmax * 10.0**-EXISTENCE_DECADES, vmax=vmax, clip=True)


def _existence_colorbar(colorbar) -> None:
    colorbar.set_label("D(x) [objects per m²], log scale", color=INK_SECONDARY)
    # Through the colorbar, not its axis: a redraw (the GIF's update_normal) keeps it.
    colorbar.formatter = FuncFormatter(lambda v, _: f"{v:g}")
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def _map_limits(truth, tracks, extra: np.ndarray | None = None,
                margin: float = 0.6) -> tuple[tuple[float, float], ...]:
    """Scene limits covering the path, plants, weeds, extra points and every track's
    2-sigma ellipse."""
    points = [np.array([[sample.true.x, sample.true.y] for sample in truth.poses]),
              truth.field.positions[:, :2], truth.weeds.reshape(-1, 2)]
    if extra is not None:
        points.append(extra)
    for track in tracks:
        reach = ELLIPSE_SIGMA * np.sqrt(np.diag(track.cov[:2, :2]))
        points.append(np.array([track.mean[:2] - reach, track.mean[:2] + reach]))
    points = np.vstack(points)
    return ((points[:, 0].min() - margin, points[:, 0].max() + margin),
            (points[:, 1].min() - margin, points[:, 1].max() + margin))


def _scan_or_last(records, k: int | None) -> int:
    return len(records) - 1 if k is None else k


def plot_tracks(run: RunDir, filter_name: str, out: Path, k: int | None = None) -> Path:
    """Every track on the scene at one scan, ellipse opacity set by r. [B4, step 5]

    Generalises the hypotheses figure, whose one row per track does not scale to a mapped
    field of about 70 plants: here each track is only its 2-sigma ellipse and mean, as
    opaque as it is likely to exist. Reads only the run folder.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.
        k: the scan to show; None means the last scan, the final map.

    Returns:
        The path written.
    """
    cfg = load_run_config(run.config)
    truth = read_truth(run.truth)
    records = read_estimates(run.estimates(filter_name))
    k = _scan_or_last(records, k)
    tracks = records[k].estimates
    wedge = _fov_wedge(truth.poses[k].true, cfg.scenario.sensor.fov,
                       facecolor=to_rgba(SERIES_1, 0.06), edgecolor=MUTED,
                       label=f"FOV at scan {k}")

    fig, (ax,) = _new_figure(height=6.0)
    ax.add_patch(wedge)
    _draw_field(ax, truth, robot_at=k)
    for track in tracks:
        opacity = float(np.clip(track.r, 0.05, 1.0))
        ax.add_patch(_ellipse(track.mean[:2], track.cov[:2, :2], edgecolor=SERIES_1,
                              alpha=opacity))
        ax.plot(*track.mean[:2], marker="o", markersize=4, color=SERIES_1, alpha=opacity,
                linestyle="none")
    for r in (1.0, 0.25):
        ax.add_patch(Ellipse((np.nan, np.nan), 1, 1, fill=False, linewidth=1.5,
                             edgecolor=SERIES_1, alpha=r,
                             label=f"track, {ELLIPSE_SIGMA:g}σ ellipse, r = {r:g}"))

    pose, fov = truth.poses[k].true, cfg.scenario.sensor.fov
    bearings = pose.theta + np.linspace(-fov.half_angle, fov.half_angle, 9)
    arc = np.column_stack([pose.x + fov.max_range * np.cos(bearings),
                           pose.y + fov.max_range * np.sin(bearings)])
    (x0, x1), (y0, y1) = _map_limits(truth, tracks, extra=arc)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Tracks at scan {k}, {filter_name}: {len(tracks)} reported", color=INK,
                 fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    return _save(fig, out)


def plot_existence_map(run: RunDir, filter_name: str, out: Path, k: int | None = None) -> Path:
    """The existence map D(x) = sum_i r_i N(x; m_i, P_i) at one scan. [B4, step 5]

    The PHD of the filter's reported tracks, as a heatmap on the palette's one-hue ramp,
    with the true plants and the robot path on top. It integrates to the sum of r, given
    in the title, so methods with and without identities are drawn alike.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.
        k: the scan to show; None means the last scan, the final map.

    Returns:
        The path written.
    """
    truth = read_truth(run.truth)
    records = read_estimates(run.estimates(filter_name))
    k = _scan_or_last(records, k)
    tracks = records[k].estimates
    (x0, x1), (y0, y1) = _map_limits(truth, tracks)
    xs = np.arange(x0, x1 + EXISTENCE_GRID_STEP, EXISTENCE_GRID_STEP)
    ys = np.arange(y0, y1 + EXISTENCE_GRID_STEP, EXISTENCE_GRID_STEP)
    density = existence_density(tracks, xs, ys)

    fig = Figure(figsize=(5.4, 6.8), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style_axes(ax)
    ax.grid(False)
    # Log colour scale: a converged track peaks near 100 per m^2 and a slot on a wide prior
    # near 2.5, and on a linear scale the prior would vanish. The colorbar carries the
    # true values.
    image = ax.imshow(density, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                      cmap=EXISTENCE_CMAP, norm=_existence_norm(density.max()),
                      interpolation="nearest")
    _existence_colorbar(fig.colorbar(image, ax=ax, shrink=0.8))
    _draw_field(ax, truth)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    total = sum(track.r for track in tracks)
    ax.set_title(f"Existence map at scan {k}, {filter_name}\nsum of r = {total:.2f}",
                 color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9,
              ncols=3)
    return _save(fig, out)


def plot_lifetimes(run: RunDir, filter_name: str, out: Path) -> Path:
    """Every track's life, from birth to deletion or the last scan. [B4, step 5]

    One row per track: a line from the first to the last scan it is reported, ending in a
    deletion (the status colour with an x, and the number of scans it lived) or in an open
    circle when the track is still alive at the last scan. This is the quantity A2 §5 is
    about: how long a phantom survives before it is deleted. Read from the estimates log
    alone (`track_lifetimes`), so it works for any filter that has births.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.
    """
    records = read_estimates(run.estimates(filter_name))
    lifetimes = track_lifetimes(records)
    n_scans = len(records)

    fig, (ax,) = _new_figure(height=max(2.4, 1.2 + 0.32 * len(lifetimes)))
    for row, (track_id, life) in enumerate(lifetimes.items()):
        ax.plot([life.k_birth, life.k_last], [row, row], color=SERIES_1, linewidth=2.0,
                solid_capstyle="round")
        ax.plot(life.k_birth, row, marker="D", markersize=6, color=SERIES_3,
                markeredgecolor=SURFACE, linestyle="none")
        if life.deleted:
            ax.plot(life.k_last, row, marker="X", markersize=9, color=DELETED,
                    markeredgecolor=SURFACE, linestyle="none")
            ax.annotate(f"{life.k_last - life.k_birth + 1} scans", (life.k_last, row),
                        xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK_SECONDARY, fontsize=8)
        else:
            ax.plot(life.k_last, row, marker="o", markersize=7, markerfacecolor=SURFACE,
                    markeredgecolor=SERIES_1, markeredgewidth=1.5, linestyle="none")

    handles = [
        Line2D([], [], color=SERIES_1, linewidth=2.0, label="reported"),
        Line2D([], [], marker="D", markersize=6, color=SERIES_3, markeredgecolor=SURFACE,
               linestyle="none", label="birth"),
        Line2D([], [], marker="X", markersize=9, color=DELETED, markeredgecolor=SURFACE,
               linestyle="none", label="deleted"),
        Line2D([], [], marker="o", markersize=7, markerfacecolor=SURFACE,
               markeredgecolor=SERIES_1, markeredgewidth=1.5, linestyle="none",
               label="alive at the last scan"),
    ]
    ax.set_yticks(range(len(lifetimes)), [f"track {i}" for i in lifetimes])
    ax.set_ylim(-0.7, max(len(lifetimes), 1) - 0.3)
    ax.invert_yaxis()
    _time_axis(ax, n_scans)
    n_deleted = sum(life.deleted for life in lifetimes.values())
    ax.set_title(f"Track lifetimes, {filter_name}: {len(lifetimes)} born, {n_deleted} deleted",
                 color=INK, fontsize=11)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=9)
    return _save(fig, out)



@dataclass(frozen=True)
class _Hypothesis:
    """One track of a bank run, as the hypotheses figure needs it. Evaluation only."""

    track_id: int
    lifetime: TrackLifetime
    r: np.ndarray            # (K,), NaN where not reported
    means: np.ndarray        # (K, 2), NaN where not reported
    covs: list               # (2, 2) per scan, None where not reported
    r_unpruned: np.ndarray   # (K,), NaN where not reported or no companion log
    means_unpruned: np.ndarray
    gated_k: list[int]
    out_of_view_k: list[int]


@dataclass(frozen=True)
class _HypothesesData:
    """Everything the hypotheses figure and its animation draw, loaded once."""

    cfg: RunConfig
    truth: object
    scans: list
    labels: list
    hypotheses: list[_Hypothesis]
    r_min: float
    limits: tuple[tuple[float, float], tuple[float, float]]


def _track_series(records, track_id: int) -> tuple[np.ndarray, np.ndarray, list]:
    """(r, means, covs) of one track per scan, NaN / None where it was not reported."""
    r = np.full(len(records), np.nan)
    means = np.full((len(records), 2), np.nan)
    covs = [None] * len(records)
    for k, record in enumerate(records):
        for estimate in record.estimates:
            if estimate.track_id == track_id:
                r[k] = estimate.r
                means[k] = estimate.mean[:2]
                covs[k] = estimate.cov[:2, :2]
    return r, means, covs


def _load_hypotheses(run: RunDir, filter_name: str) -> _HypothesesData:
    cfg = load_run_config(run.config)
    truth = read_truth(run.truth)
    scans = read_detections(run.detections)
    records = read_estimates(run.estimates(filter_name))
    unpruned_path = run.estimates(unpruned_log_name(filter_name))
    unpruned = read_estimates(unpruned_path) if unpruned_path.is_file() else None

    hypotheses = []
    for track_id, lifetime in track_lifetimes(records).items():
        r, means, covs = _track_series(records, track_id)
        if unpruned is not None:
            r_unpruned, means_unpruned, _ = _track_series(unpruned, track_id)
        else:
            r_unpruned, means_unpruned = np.full_like(r, np.nan), np.full_like(means, np.nan)
        gated_k, out_of_view_k = _scan_marks(records, track_id, scans, cfg.filter_cfg)
        hypotheses.append(_Hypothesis(track_id, lifetime, r, means, covs, r_unpruned,
                                      means_unpruned, gated_k, out_of_view_k))

    # Fixed scene limits, so animation frames do not jump: the path, every detection and
    # every hypothesis ellipse. Plants the sensor never reaches are left out.
    points = [np.array([[sample.true.x, sample.true.y] for sample in truth.poses])]
    points += [np.array([d.z[:2] for d in scan.detections]) for scan in scans if scan.detections]
    for h in hypotheses:
        for k in range(h.lifetime.k_birth, h.lifetime.k_last + 1):
            reach = ELLIPSE_SIGMA * np.sqrt(np.diag(h.covs[k]))
            points += [np.array([h.means[k] - reach, h.means[k] + reach])]
    points = np.vstack(points)
    margin = 0.4
    limits = ((points[:, 0].min() - margin, points[:, 0].max() + margin),
              (points[:, 1].min() - margin, points[:, 1].max() + margin))
    return _HypothesesData(cfg=cfg, truth=truth, scans=scans, labels=read_labels(run.labels),
                           hypotheses=hypotheses, r_min=cfg.filter_cfg.prune.r_min,
                           limits=limits)


def _ellipse(mean: np.ndarray, cov: np.ndarray, **style) -> Ellipse:
    """The ELLIPSE_SIGMA covariance ellipse of a 2D Gaussian."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2.0 * ELLIPSE_SIGMA * np.sqrt(eigenvalues[::-1])
    return Ellipse(tuple(mean), width, height, angle=angle, fill=False, linewidth=1.5, **style)


def _hypotheses_figure(n_tracks: int) -> tuple[Figure, object, list]:
    """A scene axis on top and one log-r row per track below, the rows sharing x."""
    fig = Figure(figsize=(7.0, 4.6 + 1.15 * n_tracks), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(1 + n_tracks, 1, height_ratios=[4.6] + [1.15] * n_tracks)
    scene = fig.add_subplot(grid[0])
    rows = []
    for i in range(n_tracks):
        rows.append(fig.add_subplot(grid[1 + i], sharex=rows[0] if rows else None))
    return fig, scene, rows


def _hypotheses_legend(ax, r_min: float, animated: bool, weeds: bool) -> None:
    """One legend for the scene and the rows, built from proxies so every frame matches."""
    def marker(label, **style):
        return Line2D([], [], linestyle="none", label=label, **style)

    handles = [
        Line2D([], [], color=INK, linewidth=2.0, label="robot path"),
        marker("plants (truth)", marker="o", markersize=6, markerfacecolor="none",
               markeredgecolor=INK_SECONDARY),
        marker("detection from a plant", marker="o", markersize=3, color=INK_SECONDARY,
               alpha=1.0 if animated else 0.35),
        marker("clutter detection", marker="X", markersize=6, color=SERIES_2,
               alpha=1.0 if animated else 0.55),
    ]
    if weeds:
        handles += [
            marker("weeds (truth)", marker=WEED_MARKER, markersize=8, markerfacecolor="none",
                   markeredgecolor=INK_SECONDARY, markeredgewidth=1.2),
            marker("detection from a weed", marker=WEED_MARKER, markersize=6, color=SERIES_2,
                   alpha=1.0 if animated else 0.55),
        ]
    handles += [
        marker("birth (seed detection)", marker="D", markersize=7, color=SERIES_3,
               markeredgecolor=SURFACE),
        Line2D([], [], color=SERIES_1, linewidth=2.0, marker="o", markersize=6,
               label="live hypothesis: mean, r"),
        Line2D([], [], color=DELETED, linewidth=2.0, marker="X", markersize=9,
               markeredgecolor=SURFACE, label="deleted hypothesis"),
        Line2D([], [], color=INK_SECONDARY, linewidth=1.5,
               label=f"{ELLIPSE_SIGMA:g}σ ellipse"),
        Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (1, 2)),
               label="without pruning"),
        marker("scan with a gated detection", marker="|", markersize=9, color=SERIES_2,
               markeredgewidth=2.0),
        Patch(facecolor=MUTED, alpha=0.15, label="out of view (p_D = 0)"),
    ]
    if r_min > 0.0:
        handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1.0,
                              linestyle=(0, (4, 3)), label=f"pruning threshold r_min = {r_min:g}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=8)


def _draw_hypotheses(data: _HypothesesData, scene, rows, k_now: int, animated: bool) -> None:
    """Draw the scene and the log-r rows as they stand after scan k_now.

    Static figure (animated=False, k_now = last scan): every scan's detections faint, the
    FOV at each birth scan, each hypothesis coloured by its final status. Animation frame:
    the FOV and detections of scan k_now only, the robot at k_now, the r rows drawn up to a
    cursor at k_now, and a hypothesis turns into a deleted ghost once it has been pruned.
    """
    fov = data.cfg.filter_cfg.assumed_sensor.fov
    n_scans = len(data.scans)

    _style_axes(scene)
    if animated:
        scan = data.scans[k_now]
        scene.add_patch(_fov_wedge(scan.pose, fov, facecolor=to_rgba(SERIES_1, 0.08),
                                   edgecolor=MUTED))
        detection_scans = [(scan, data.labels[k_now])]
        detection_alpha = (1.0, 1.0)
    else:
        for h in data.hypotheses:
            birth_pose = data.scans[h.lifetime.k_birth].pose
            scene.add_patch(_fov_wedge(birth_pose, fov, facecolor=to_rgba(SERIES_1, 0.04),
                                       edgecolor=GRID))
        detection_scans = list(zip(data.scans, data.labels))
        detection_alpha = (0.35, 0.55)
    _draw_field(scene, data.truth, robot_at=k_now if animated else None)

    for scan, scan_labels in detection_scans:
        for detection, origin, weed in zip(scan.detections, scan_labels.origin,
                                           weed_origin(scan_labels)):
            if weed is not None:
                scene.plot(*detection.z[:2], marker=WEED_MARKER, markersize=6, color=SERIES_2,
                           markeredgewidth=0, alpha=detection_alpha[1], linestyle="none")
            elif origin is None:
                scene.plot(*detection.z[:2], marker="X", markersize=6, color=SERIES_2,
                           markeredgewidth=0, alpha=detection_alpha[1], linestyle="none")
            else:
                scene.plot(*detection.z[:2], marker="o", markersize=3, color=INK_SECONDARY,
                           markeredgewidth=0, alpha=detection_alpha[0], linestyle="none")

    for h, row in zip(data.hypotheses, rows):
        _draw_hypothesis(data, h, scene, row, k_now, animated, n_scans)

    scene.set_xlim(*data.limits[0])
    scene.set_ylim(*data.limits[1])
    scene.set_aspect("equal")
    scene.set_xlabel("x [m]")
    scene.set_ylabel("y [m]")
    when = f"after scan {k_now}" if animated else f"over {n_scans} scans"
    scene.set_title(f"Phantom hypotheses {when}: {data.cfg.name}", color=INK, fontsize=11)
    _hypotheses_legend(scene, data.r_min, animated, weeds=len(data.truth.weeds) > 0)

    rows[-1].set_xlabel("scan k")
    rows[-1].set_xlim(-0.5, n_scans - 0.5)
    for row in rows[:-1]:
        row.tick_params(labelbottom=False)


def _draw_hypothesis(data, h: _Hypothesis, scene, row, k_now: int, animated: bool,
                     n_scans: int) -> None:
    """One hypothesis: its track in the scene and its log-r row."""
    life = h.lifetime
    _style_axes(row)
    row.set_yscale("log")
    row.set_ylim(R_LOG_FLOOR, 2.0)
    row.set_yticks([1e-4, 1e-2, 1.0])
    row.set_ylabel(f"#{h.track_id}", rotation=0, ha="right", va="center", color=INK,
                   fontsize=10)
    if data.r_min > 0.0:
        row.axhline(data.r_min, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    if animated:
        row.axvline(k_now, color=INK, linewidth=1.0)
    if life.k_birth > k_now:
        return

    k_end = min(k_now, life.k_last)
    deleted = life.deleted and k_now > life.k_last
    colour = DELETED if deleted else SERIES_1
    ghost = 0.45 if deleted and animated else 1.0
    span = slice(life.k_birth, k_end + 1)
    k = np.arange(n_scans)

    # --- the row: r over time, on a log axis ---
    for scan_k in h.out_of_view_k:
        if scan_k <= k_now:
            row.axvspan(scan_k - 0.5, scan_k + 0.5, color=MUTED, alpha=0.15, linewidth=0)
    row.plot(k[span], np.maximum(h.r[span], R_LOG_FLOOR), color=colour, linewidth=2.0,
             solid_joinstyle="round", solid_capstyle="round")
    gated = [g for g in h.gated_k if g <= k_end]
    if gated:
        # A tick strip along the bottom edge, so the markers never hide the r line.
        row.plot(gated, np.full(len(gated), 0.07), marker="|", markersize=8,
                 markeredgewidth=2.0, linestyle="none", color=SERIES_2,
                 transform=row.get_xaxis_transform())
    row.plot(life.k_birth, h.r[life.k_birth], marker="D", markersize=6, linestyle="none",
             color=SERIES_3, markeredgecolor=SURFACE, markeredgewidth=1.0)
    if deleted:
        after = slice(life.k_last, k_now + 1)
        row.plot(k[after], np.maximum(h.r_unpruned[after], R_LOG_FLOOR), color=MUTED,
                 linewidth=1.5, linestyle=(0, (1, 2)))
        row.plot(life.k_last, max(h.r[life.k_last], R_LOG_FLOOR), marker="X", markersize=10,
                 linestyle="none", color=DELETED, markeredgecolor=SURFACE,
                 markeredgewidth=1.0)
        status = (f"r = {h.r[life.k_last]:.1e} < r_min at k = {life.k_last}, "
                  f"deleted before k = {life.k_last + 1}")
    else:
        moved = np.linalg.norm(h.means[k_end] - h.means[life.k_birth])
        status = f"r = {h.r[k_end]:.2g} at k = {k_end}, mean {moved:.2f} m from its seed"
    row.set_title(f"born k = {life.k_birth}; {status}", loc="left", fontsize=8,
                  color=INK_SECONDARY, pad=2)

    # --- the scene: where the hypothesis sat, and how it ended ---
    means = h.means[span]
    scene.plot(means[:, 0], means[:, 1], color=colour, linewidth=1.5, alpha=ghost)
    if deleted and not animated:
        after = h.means_unpruned[life.k_last:]
        scene.plot(after[:, 0], after[:, 1], color=MUTED, linewidth=1.5, linestyle=(0, (1, 2)))
    scene.plot(*h.means[life.k_birth], marker="D", markersize=7, linestyle="none",
               color=SERIES_3, markeredgecolor=SURFACE, markeredgewidth=1.5)
    end = h.means[k_end]
    scene.add_patch(_ellipse(end, h.covs[k_end], edgecolor=colour, alpha=ghost,
                             linestyle=(0, (4, 2)) if deleted else "solid"))
    if deleted:
        scene.plot(*end, marker="X", markersize=11, linestyle="none", color=DELETED,
                   markeredgecolor=SURFACE, markeredgewidth=1.5, alpha=ghost)
        label = f"#{h.track_id} deleted"
    else:
        scene.plot(*end, marker="o", markersize=6, linestyle="none", color=SERIES_1,
                   markeredgecolor=SURFACE, markeredgewidth=1.5)
        label = f"#{h.track_id}"
    scene.annotate(label, end, xytext=(8, 6), textcoords="offset points", fontsize=8,
                   color=INK, alpha=ghost)


def plot_hypotheses(run: RunDir, filter_name: str, out: Path) -> Path:
    """Every hypothesis of a bank run over the whole run: where, how long, how it ended. [B2]

    The figure for the bank experiment (decisions D12-D14). On top, a scene as in
    `plot_scene`: plants, robot path, every scan's detections faint, the FOV at each birth
    scan, and per hypothesis its birth, its posterior-mean history (a jump is a capture by
    a plant) and its uncertainty ellipse at its last estimate. A deleted hypothesis is
    drawn in the reserved deletion colour with a dashed ellipse, an x and a "deleted"
    label; where the companion log exists, a dotted line shows where it would have gone
    without pruning.

    Below, one row per hypothesis: r against scan index on a LOG axis - on a linear axis
    the decay from r_b to r_min is invisible - with the pruning threshold, the deletion,
    the unpruned r after it, gated-detection scans and out-of-view scans (as in
    `plot_r_vs_k`).

    Deletion is read from the estimates log alone (`track_lifetimes`). Detection origins
    come from labels.jsonl: evaluation code.

    Args:
        run: the run folder of a `bernoulli_bank` run.
        filter_name: which estimates log to read.
        out: destination PNG path.

    Returns:
        The path written.

    Raises:
        ValueError: if the log reports no track.
    """
    data = _load_hypotheses(run, filter_name)
    if not data.hypotheses:
        raise ValueError(f"{run.estimates(filter_name)} reports no track")
    fig, scene, rows = _hypotheses_figure(len(data.hypotheses))
    _draw_hypotheses(data, scene, rows, k_now=len(data.scans) - 1, animated=False)
    return _save(fig, out)


def animate_hypotheses(run: RunDir, filter_name: str, out: Path, fps: float = 4.0) -> Path:
    """The hypotheses figure scan by scan, as a GIF. [B2]

    One frame per scan: the FOV and detections of that scan, the live hypotheses, deleted
    ones as faded ghosts, and the log-r rows drawn up to a cursor. Written with
    matplotlib's PillowWriter (Pillow is a matplotlib dependency; nothing new is needed).

    Args:
        run: the run folder of a `bernoulli_bank` run.
        filter_name: which estimates log to read.
        out: destination GIF path.
        fps: frames (scans) per second.

    Returns:
        The path written.
    """
    data = _load_hypotheses(run, filter_name)
    if not data.hypotheses:
        raise ValueError(f"{run.estimates(filter_name)} reports no track")
    fig, scene, rows = _hypotheses_figure(len(data.hypotheses))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out, dpi=100):
        for k_now in range(len(data.scans)):
            for ax in [scene, *rows]:
                ax.clear()
            _draw_hypotheses(data, scene, rows, k_now=k_now, animated=True)
            writer.grab_frame(facecolor=SURFACE)
    return out


# Grid spacing of the animated existence map, metres: fine enough that a converged track
# (posterior std about 3 cm) is more than one cell wide.
MOVIE_GRID_STEP = 0.02
# The empty slots of the plan (step 8c), in truth: a hollow square in the clutter orange -
# a place where a track should find nothing.
EMPTY_SLOT_MARKER = "s"
# The animated map's window, metres behind and ahead of the robot along the lane: the FOV
# reaches 4 m ahead, and the rest shows slots not yet seen at their prior.
MOVIE_WINDOW = (1.5, 5.5)


@dataclass(frozen=True)
class _ExistenceMovie:
    """Everything the existence-map animation draws, loaded and summarised once."""

    filter_name: str
    truth: object
    scans: list
    kinds: list                  # per scan, each detection's true origin kind
    records: list
    fov: object
    xs: np.ndarray
    ys: np.ndarray
    norm: LogNorm                # the colour scale, fixed over frames
    drops_weed_labels: bool      # the filter ignores weed-labelled detections (D22)
    by_origin: dict              # per origin kind, detections per scan, shape (K,)
    gates: object                # evaluation.GateContents
    mean_r: np.ndarray           # (K,) mean r of the tracks in view, NaN without any
    median_std: np.ndarray       # (K,) their median position std, metres
    in_a_gate: list              # per scan, the detections inside at least one track's gate


def _load_existence_movie(run: RunDir, filter_name: str) -> _ExistenceMovie:
    cfg = load_run_config(run.config)
    truth = read_truth(run.truth)
    scans = read_detections(run.detections)
    kinds = [origin_kinds(label) for label in read_labels(run.labels)]
    records = read_estimates(run.estimates(filter_name))

    points = [np.array([[s.true.x, s.true.y] for s in truth.poses]), truth.field.positions,
              truth.weeds.reshape(-1, 2), truth.field.missing_positions.reshape(-1, 2)]
    points = np.vstack(points)
    x0, y0 = points.min(axis=0) - 0.6
    x1, y1 = points.max(axis=0) + 0.6
    # A track's density peaks at r / (2 pi sqrt(det P)); the brightest peak over the run
    # tops the fixed colour scale, so a slot visibly gains certainty frame by frame.
    peaks = [e.r / (2.0 * np.pi * np.sqrt(np.linalg.det(e.cov[:2, :2])))
             for record in records for e in record.estimates]

    # Which detections reached some track's gate, as the filter gated them: the ones that
    # could move a track. With an assumed classifier, weed-labelled ones never do (D22).
    in_a_gate = [set() for _ in scans]
    times = [scan.t for scan in scans]
    for track_id in track_lifetimes(records):
        moments = predicted_track_moments(records, track_id, times, cfg.filter_cfg)
        for k, moment in enumerate(moments):
            if moment is not None:
                in_a_gate[k].update(
                    int(i) for i in gated_detection_indices(scans[k], *moment, cfg.filter_cfg))

    views = scan_views(run, filter_name, cfg)
    mean_r = np.array([np.mean([t.r for t in v.tracks]) if v.tracks else np.nan
                       for v in views])
    median_std = np.array([np.median([np.sqrt(np.trace(t.cov[:2, :2]) / 2.0)
                                      for t in v.tracks]) if v.tracks else np.nan
                           for v in views])
    return _ExistenceMovie(
        filter_name=filter_name, truth=truth, scans=scans, kinds=kinds, records=records,
        fov=cfg.scenario.sensor.fov,
        xs=np.arange(x0, x1 + MOVIE_GRID_STEP, MOVIE_GRID_STEP),
        ys=np.arange(y0, y1 + MOVIE_GRID_STEP, MOVIE_GRID_STEP),
        norm=_existence_norm(max(peaks) if peaks else 1.0),
        drops_weed_labels=cfg.filter_cfg.assumed_classifier is not None,
        by_origin={kind: np.array([ks.count(kind) for ks in kinds])
                   for kind in ("plant", "clutter", "weed")},
        gates=gate_contents(run, filter_name),
        mean_r=mean_r, median_std=median_std, in_a_gate=in_a_gate,
    )


def _existence_movie_figure() -> tuple[Figure, object, list]:
    """The map on the left, four time series stacked on the right, sharing x."""
    fig = Figure(figsize=(12.0, 8.0), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.35])
    scene = fig.add_subplot(grid[:, 0])
    series = []
    for i in range(4):
        series.append(fig.add_subplot(grid[i, 1], sharex=series[0] if series else None))
    return fig, scene, series


def _draw_existence_frame(movie: _ExistenceMovie, scene, series, k: int, colorbar) -> None:
    """One frame: the map after scan k, and every series up to k."""
    truth = movie.truth
    tracks = movie.records[k].estimates
    pose = truth.poses[k].true
    # A window that follows the robot, MOVIE_WINDOW behind and ahead: at the scale of the
    # whole field a converged track (std about 3 cm) would be a single pixel.
    behind, ahead = MOVIE_WINDOW
    ys = movie.ys[(movie.ys >= pose.y - behind) & (movie.ys <= pose.y + ahead)]
    density = existence_density(tracks, movie.xs, ys)
    image = scene.imshow(density, origin="lower", cmap=EXISTENCE_CMAP,
                         norm=movie.norm,
                         extent=(movie.xs[0], movie.xs[-1], ys[0], ys[-1]),
                         interpolation="nearest")
    if colorbar is not None:
        colorbar.update_normal(image)
    scene.add_patch(_fov_wedge(pose, movie.fov, facecolor="none", edgecolor=INK_SECONDARY))
    path = np.array([[s.true.x, s.true.y] for s in truth.poses[:k + 1]])
    scene.plot(path[:, 0], path[:, 1], color=INK, linewidth=2.0)
    scene.plot(pose.x, pose.y, marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
               markeredgewidth=2.0, linestyle="none")
    hollow = {"markerfacecolor": "none", "linestyle": "none"}
    scene.plot(*truth.field.positions.T, marker="o", markersize=5,
               markeredgecolor=INK_SECONDARY, markeredgewidth=0.8, **hollow)
    if len(truth.weeds):
        scene.plot(*truth.weeds.T, marker=WEED_MARKER, markersize=7,
                   markeredgecolor=INK_SECONDARY, markeredgewidth=1.0, **hollow)
    if len(truth.field.missing_positions):
        scene.plot(*truth.field.missing_positions.T, marker=EMPTY_SLOT_MARKER, markersize=7,
                   markeredgecolor=SERIES_2, markeredgewidth=1.2, **hollow)

    style = {"plant": ("o", INK, 4), "clutter": ("X", SERIES_2, 8),
             "weed": (WEED_MARKER, SERIES_2, 8)}
    for i, (detection, kind) in enumerate(zip(movie.scans[k].detections, movie.kinds[k])):
        marker, colour, size = style[kind]
        dropped = movie.drops_weed_labels and detection.label == "weed"
        if kind != "plant" and i in movie.in_a_gate[k]:
            # A false return that reached a track's gate: it can move that track.
            scene.plot(*detection.z[:2], marker="o", markersize=15, linestyle="none",
                       markerfacecolor="none", markeredgecolor=SERIES_2, markeredgewidth=1.5)
        scene.plot(*detection.z[:2], marker=marker, markersize=size, linestyle="none",
                   color=colour, markeredgecolor=colour if dropped else SURFACE,
                   markerfacecolor="none" if dropped else colour, markeredgewidth=1.2)
    scene.set_xlim(movie.xs[0], movie.xs[-1])
    scene.set_ylim(pose.y - behind, pose.y + ahead)
    scene.set_aspect("equal")
    scene.set_xlabel("x [m]")
    scene.set_ylabel("y [m]")
    n_false = sum(1 for i, kind in enumerate(movie.kinds[k])
                  if kind != "plant" and i in movie.in_a_gate[k])
    scene.set_title(f"Existence map after scan {k}: sum of r = "
                    f"{sum(t.r for t in tracks):.1f}\n{n_false} false return(s) in a gate",
                    color=INK, fontsize=11)
    _existence_movie_legend(scene, movie)

    kk = np.arange(k + 1)
    n_scans = len(movie.scans)
    ax_det, ax_gate, ax_r, ax_std = series
    for ax in series:
        _style_axes(ax)
        ax.set_xlim(-0.5, n_scans - 0.5)
        ax.axvline(k, color=INK_SECONDARY, linewidth=1.0)

    bar = {"width": 0.9, "edgecolor": SURFACE, "linewidth": 0.8}
    plant, clutter, weed = (movie.by_origin[kind][:k + 1]
                            for kind in ("plant", "clutter", "weed"))
    ax_det.bar(kk, plant, color=SERIES_1, label="from a plant", **bar)
    ax_det.bar(kk, clutter, bottom=plant, color=SERIES_2, label="clutter", **bar)
    ax_det.bar(kk, weed, bottom=plant + clutter, color=SERIES_2, hatch=WEED_HATCH,
               label="from a weed", **bar)
    top = (movie.by_origin["plant"] + movie.by_origin["clutter"] + movie.by_origin["weed"]).max()
    ax_det.set_ylim(0, top + 1)
    ax_det.set_ylabel("detections")
    ax_det.set_title("Detections per scan, by true origin", color=INK, fontsize=10)
    ax_det.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    gates = movie.gates
    layers = [np.nan_to_num(v[:k + 1]) for v in (gates.neighbour, gates.clutter, gates.weed)]
    polygons = ax_gate.stackplot(kk, layers, colors=[SERIES_3, SERIES_2, SERIES_2],
                                 edgecolor=SURFACE, linewidth=0.8,
                                 labels=["a neighbour plant", "clutter", "a weed"])
    polygons[2].set_hatch(WEED_HATCH)
    full = np.nan_to_num(gates.neighbour + gates.clutter + gates.weed)
    ax_gate.set_ylim(0, max(full.max(), 0.1) * 1.1)
    ax_gate.set_ylabel("per gate")
    ax_gate.set_title("Returns not from the track's own plant, inside its gate", color=INK,
                      fontsize=10)
    ax_gate.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    ax_r.plot(kk, movie.mean_r[:k + 1], color=SERIES_1, linewidth=2.0)
    ax_r.set_ylim(-0.02, 1.02)
    ax_r.set_ylabel("mean r")
    ax_r.set_title("Existence of the tracks in view", color=INK, fontsize=10)

    ax_std.plot(kk, 100.0 * movie.median_std[:k + 1], color=SERIES_1, linewidth=2.0)
    ax_std.set_ylim(0, 100.0 * np.nanmax(movie.median_std) * 1.1)
    ax_std.set_ylabel("std [cm]")
    ax_std.set_xlabel("scan k")
    ax_std.set_title("Median position std of the tracks in view", color=INK, fontsize=10)


def _existence_movie_legend(ax, movie: _ExistenceMovie) -> None:
    def marker(label, **style):
        return Line2D([], [], linestyle="none", label=label, **style)

    handles = [
        Line2D([], [], color=INK, linewidth=2.0, label="robot path"),
        marker("plant (truth)", marker="o", markersize=5, markerfacecolor="none",
               markeredgecolor=INK_SECONDARY),
        marker("detection from a plant", marker="o", markersize=4, color=INK,
               markeredgecolor=SURFACE),
        marker("clutter detection", marker="X", markersize=8, color=SERIES_2,
               markeredgecolor=SURFACE),
    ]
    if len(movie.truth.weeds):
        handles += [
            marker("weed (truth)", marker=WEED_MARKER, markersize=7, markerfacecolor="none",
                   markeredgecolor=INK_SECONDARY),
            marker("detection from a weed", marker=WEED_MARKER, markersize=8, color=SERIES_2,
                   markeredgecolor=SURFACE),
        ]
    if movie.drops_weed_labels:
        handles.append(marker("labelled weed: ignored", marker=WEED_MARKER, markersize=8,
                              markerfacecolor="none", markeredgecolor=SERIES_2))
    handles.append(marker("false return inside a track's gate", marker="o", markersize=12,
                          markerfacecolor="none", markeredgecolor=SERIES_2,
                          markeredgewidth=1.5))
    if len(movie.truth.field.missing_positions):
        handles.append(marker("empty slot (truth)", marker=EMPTY_SLOT_MARKER, markersize=7,
                              markerfacecolor="none", markeredgecolor=SERIES_2))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              frameon=False, fontsize=8, ncols=2)


def animate_existence_map(run: RunDir, filter_name: str, out: Path,
                          fps: float = 4.0) -> Path:
    """The existence map scan by scan, with what each scan's detections did, as a GIF. [B4]

    One frame per scan k. Left: D(x) = sum_i r_i N(x; m_i, P_i) after scan k on a colour
    scale fixed over the whole run - a slot brightens as its r rises and sharpens as its
    covariance shrinks - with the true plants, weeds and empty slots, the robot's path and
    FOV, and scan k's detections by true origin (a weed-labelled detection that the filter
    ignores is drawn hollow, D22). Right, up to a cursor at k: detections per scan by
    origin; the returns inside the tracks' gates that are not from their own plant - a
    neighbour, clutter, a weed - which is how clutter and weeds reach the update; the mean
    r of the tracks in view; and their median position std.

    Reads only the run folder: works for any filter.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        out: destination GIF path.
        fps: frames (scans) per second.

    Returns:
        The path written.
    """
    movie = _load_existence_movie(run, filter_name)
    fig, scene, series = _existence_movie_figure()
    _draw_existence_frame(movie, scene, series, 0, colorbar=None)
    colorbar = fig.colorbar(scene.images[0], ax=scene, shrink=0.6, pad=0.02)
    _existence_colorbar(colorbar)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out, dpi=100):
        for k in range(len(movie.scans)):
            for ax in [scene, *series]:
                ax.clear()
            _draw_existence_frame(movie, scene, series, k, colorbar)
            writer.grab_frame(facecolor=SURFACE)
    return out
