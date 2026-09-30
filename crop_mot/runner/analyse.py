"""B1/B2/B3 entry point: a finished run folder -> plots and metrics. [B1/B2/B3]

The `python -m crop_mot analyse` subcommand. Added during implementation: the skeleton's
CLI named an `analyse` subcommand but had no runner function behind it, and `__main__` is
meant to stay thin. Everything here reads from the run folder only; by the time it runs, the
estimates it judges are already on disk.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from crop_mot.analysis.analytic import REFERENCES
from crop_mot.analysis.estimates_log import (
    ScanEstimates,
    r_trajectory,
    read_estimates,
    track_lifetimes,
)
from crop_mot.analysis.events import branch_counts, build_scan_events, predicted_track_moments
from crop_mot.analysis.metrics import R_TOLERANCE, compare_r
from crop_mot.analysis.plots import (
    animate_hypotheses,
    load_run_folder_config,
    plot_counts,
    plot_hypotheses,
    plot_r_vs_analytic,
    plot_r_vs_k,
    plot_scene,
)
from crop_mot.config import RunConfig, ScenarioConfig
from crop_mot.io import to_jsonable
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import unpruned_log_name
from crop_mot.sensor.record import read_detections

# Every plot `analyse` knows how to draw, as named in the config's `analysis.plots`.
PLOT_NAMES = ("scene", "counts", "r_vs_k", "hypotheses", "hypotheses_anim", "r_vs_analytic",
              "r_montecarlo")
# The ones a simulate-only run folder (no filter, no estimates) can draw.
SCENARIO_PLOTS = ("scene", "counts")
# The ones built on the A2 closed form, which assumes multiplicity "single".
B3_PLOTS = ("r_vs_analytic", "r_montecarlo")


def analyse_run(run: RunDir, plots: Sequence[str] | None = None,
                scene_k: int | None = None) -> None:
    """Render the configured plots and write metrics.json for one run folder. [B1/B2/B3]

    Reads `run.config`. For a `track` run that is a RunConfig: each requested plot is
    rendered into run.plots, and `r_vs_analytic` also writes the B3 cross-check to
    run.metrics (see `_cross_check`). For a simulate-only run it is a ScenarioConfig: only
    the B1 plots (SCENARIO_PLOTS) are available, and `plots` defaults to all of them.

    Per-track plots (`r_vs_k`, `r_vs_analytic`) are drawn for every track in the estimates
    log: `<plot>.png` when there is one track, `<plot>_track<id>.png` when there are
    several. Relative paths inside the config copy resolve against the working directory,
    as for `track`.

    Args:
        run: the run folder to analyse.
        plots: plot names to render, overriding the config's `analysis.plots`; None means
            use the config's list. Lets the B2 r_vs_k plot be produced through the CLI on
            its own.
        scene_k: the scan the `scene` plot shows. None means the first scan any track is
            reported, or 0 if there is none - read from the log, so it works for any birth
            model.

    Raises:
        ValueError: if a plot name is unknown or needs a filter run the folder does not
            hold; if a per-track plot is requested and the filter never reported a track;
            if `r_vs_analytic` is requested without `analysis.b3_reference`; or if a B3
            plot is requested for a scenario whose multiplicity is not "single" (the A2
            closed form assumes at most one detection per plant).
        NotImplementedError: for `r_montecarlo`, which waits on roadmap step 4 (D17).
    """
    cfg = load_run_folder_config(run)
    if isinstance(cfg, ScenarioConfig):
        names = tuple(plots) if plots is not None else SCENARIO_PLOTS
        _check_names(names, allowed=SCENARIO_PLOTS, where="a simulate-only run")
        for name in names:
            out = run.plots / f"{name}.png"
            if name == "scene":
                plot_scene(run, out, k=scene_k if scene_k is not None else 0)
            else:
                plot_counts(run, out)
        return

    names = tuple(plots) if plots is not None else cfg.analysis.plots
    _check_names(names, allowed=PLOT_NAMES, where="analyse")

    multiplicity = cfg.scenario.sensor.multiplicity.kind
    b3 = [name for name in names if name in B3_PLOTS]
    if b3 and multiplicity != "single":
        raise ValueError(
            f"plot(s) {b3} compare against the A2 closed form, which assumes at most one "
            f"detection per plant; this scenario has multiplicity {multiplicity!r}"
        )

    filter_name = cfg.filter_cfg.kind
    records = read_estimates(run.estimates(filter_name))
    for name in names:
        out = run.plots / f"{name}.png"
        if name == "scene":
            plot_scene(run, out, k=scene_k if scene_k is not None else _first_report(records))
        elif name == "counts":
            plot_counts(run, out)
        elif name == "r_vs_k":
            track_ids = _track_ids(records, run.estimates(filter_name))
            for track_id in track_ids:
                plot_r_vs_k(run, filter_name, track_id,
                            _per_track_png(run, name, track_id, track_ids))
        elif name == "hypotheses":
            plot_hypotheses(run, filter_name, out)
        elif name == "hypotheses_anim":
            animate_hypotheses(run, filter_name, run.plots / "hypotheses.gif")
        elif name == "r_vs_analytic":
            _cross_check(run, cfg)
        else:
            raise NotImplementedError(
                f"plot {name!r} waits on roadmap step 4 (D17): run_monte_carlo becomes the "
                "per-seed trial loop first"
            )


def _cross_check(run: RunDir, cfg: RunConfig) -> None:
    """The B3 cross-check of every track, into run.metrics and one r_vs_analytic plot each.

    Each track's r trajectory is compared with the configured analytic reference, evaluated
    on the event sequence `build_scan_events` builds from that track's predicted moments.
    When the filter prunes, the check reads the unpruned companion log instead (D14): it is
    identical up to each deletion and runs on after it, and A2 has no deletion step.

    metrics.json gets a "b3_cross_check" entry: the reference, the log checked, the
    tolerance, and per track the errors, the first divergence and the branch coverage.
    Other entries already in the file are kept.
    """
    if cfg.analysis.b3_reference is None:
        raise ValueError("plot 'r_vs_analytic' needs analysis.b3_reference in the run config")
    reference = REFERENCES[cfg.analysis.b3_reference](p_S=cfg.filter_cfg.survival.p_S,
                                                       r_birth=cfg.filter_cfg.birth.r_b)
    log_name = cfg.filter_cfg.kind
    if cfg.filter_cfg.prune.r_min > 0.0:
        log_name = unpruned_log_name(cfg.filter_cfg.kind)
    records = read_estimates(run.estimates(log_name))
    times = [scan.t for scan in read_detections(run.detections)]

    track_ids = _track_ids(records, run.estimates(log_name))
    tracks = []
    for track_id in track_ids:
        moments = predicted_track_moments(records, track_id, times, cfg.filter_cfg)
        events = build_scan_events(run.detections, run.labels, cfg.filter_cfg, moments)
        r_sim = r_trajectory(records, track_id)
        r_ref = reference.r_sequence(events)
        comparison = compare_r(r_sim, r_ref)
        plot_r_vs_analytic(
            r_sim, r_ref, _per_track_png(run, "r_vs_analytic", track_id, track_ids),
            title=f"{cfg.name}, track {track_id}: filter r against A2 "
                  f"(max |error| {comparison.max_abs_error:.1e})")
        tracks.append({"track_id": track_id,
                       "max_abs_error": comparison.max_abs_error,
                       "rms_error": comparison.rms_error,
                       "first_divergence_k": comparison.first_divergence_k,
                       "branches": branch_counts(events)})

    metrics = json.loads(run.metrics.read_text(encoding="utf-8")) if run.metrics.is_file() else {}
    metrics["b3_cross_check"] = {"reference": cfg.analysis.b3_reference,
                                 "estimates_log": run.estimates(log_name).name,
                                 "tolerance": R_TOLERANCE,
                                 "tracks": tracks}
    run.metrics.write_text(json.dumps(to_jsonable(metrics), indent=2) + "\n",
                           encoding="utf-8")


def _check_names(names: Sequence[str], allowed: Sequence[str], where: str) -> None:
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown} for {where}; available: {list(allowed)}")


def _track_ids(records: list[ScanEstimates], log: Path) -> list[int]:
    track_ids = list(track_lifetimes(records))
    if not track_ids:
        raise ValueError(f"{log} reports no track")
    return track_ids


def _first_report(records: list[ScanEstimates]) -> int:
    """The first scan any track is reported, or 0 if none is."""
    lifetimes = track_lifetimes(records).values()
    return min((lifetime.k_birth for lifetime in lifetimes), default=0)


def _per_track_png(run: RunDir, name: str, track_id: int, track_ids: list[int]) -> Path:
    """<name>.png for a one-track run, <name>_track<id>.png when there are several."""
    if len(track_ids) == 1:
        return run.plots / f"{name}.png"
    return run.plots / f"{name}_track{track_id}.png"
