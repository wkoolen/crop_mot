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
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Wedge

from crop_mot.analysis.counts import scan_counts
from crop_mot.analysis.estimates_log import (
    TrackLifetime,
    r_trajectory,
    read_estimates,
    track_lifetimes,
)
from crop_mot.analysis.events import gated_detection_indices, predicted_track_moments
from crop_mot.analysis.montecarlo import MonteCarloResult, standard_error
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
    """The robot path, the plants and optionally the robot at one scan - scene context."""
    path = np.array([[sample.true.x, sample.true.y] for sample in truth.poses])
    ax.plot(path[:, 0], path[:, 1], color=INK, linewidth=2.0, label="robot path")
    if robot_at is not None:
        pose = truth.poses[robot_at].true
        ax.plot(pose.x, pose.y, marker="o", markersize=8, color=INK, markeredgecolor=SURFACE,
                markeredgewidth=2.0, linestyle="none", label=f"robot at scan {robot_at}")
    plants = truth.field.positions
    ax.plot(plants[:, 0], plants[:, 1], marker="o", markersize=6, linestyle="none",
            markerfacecolor="none", markeredgecolor=INK_SECONDARY, label="plants (truth)")


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


def plot_counts(run: RunDir, out: Path) -> Path:
    """Clairvoyant per-scan counts: detections received against plants in view. [B1]

    Each scan's detections as a stacked bar, split by true origin (from a plant / clutter),
    with the number of plants inside the FOV drawn as a line on the same count axis. The
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
    clutter = np.array([c.n_clutter for c in counts])
    visible = np.array([c.n_visible for c in counts])

    fig, (ax,) = _new_figure()
    # A surface-coloured edge keeps a visible gap between adjacent bars and segments.
    bar = {"width": 0.9, "edgecolor": SURFACE, "linewidth": 1.0}
    ax.bar(k, from_plants, color=SERIES_1, label="detections from plants", **bar)
    ax.bar(k, clutter, bottom=from_plants, color=SERIES_2, label="clutter detections", **bar)
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


# The reserved status colour for a deleted hypothesis (the palette's "critical"). A status
# colour, never a series slot, and it always ships with the x icon and a "deleted" label.
DELETED = "#d03b3b"
# Floor of the log-r axis in the hypotheses figure; r below it is clipped.
R_LOG_FLOOR = 1e-5
# Radius of the uncertainty ellipse drawn at a hypothesis, in standard deviations.
ELLIPSE_SIGMA = 2.0


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


def _hypotheses_legend(ax, r_min: float, animated: bool) -> None:
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
        for detection, origin in zip(scan.detections, scan_labels.origin):
            if origin is None:
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
    _hypotheses_legend(scene, data.r_min, animated)

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
