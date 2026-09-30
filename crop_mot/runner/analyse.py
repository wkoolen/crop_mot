"""B1/B2/B3 entry point: a finished run folder -> plots and metrics. [B1/B2/B3]

The `python -m crop_mot analyse` subcommand. Added during implementation: the skeleton's
CLI named an `analyse` subcommand but had no runner function behind it, and `__main__` is
meant to stay thin. Everything here reads from the run folder only; by the time it runs, the
estimates it judges are already on disk.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from crop_mot.analysis.crosscheck import checked_log_name, cross_check_run
from crop_mot.analysis.estimates_log import ScanEstimates, read_estimates, track_lifetimes
from crop_mot.analysis.evaluation import D_MATCH, R_CONF
from crop_mot.analysis.fates import fate_proportions, phantom_outcome
from crop_mot.analysis.metrics import R_TOLERANCE
from crop_mot.analysis.montecarlo import (
    cross_check_summary,
    first_track_r,
    mean_r,
    run_trials,
)
from crop_mot.analysis.plots import (
    animate_hypotheses,
    load_run_folder_config,
    plot_cardinality,
    plot_counts,
    plot_existence_map,
    plot_gospa,
    plot_hypotheses,
    plot_lifetimes,
    plot_nees,
    plot_phantom_fates,
    plot_r_montecarlo,
    plot_r_vs_analytic,
    plot_r_vs_k,
    plot_scene,
    plot_tracks,
)
from crop_mot.config import RunConfig, ScenarioConfig
from crop_mot.io import to_jsonable
from crop_mot.runner.run_dir import RunDir

# Every plot `analyse` knows how to draw, as named in the config's `analysis.plots`.
PLOT_NAMES = ("scene", "counts", "r_vs_k", "hypotheses", "hypotheses_anim", "r_vs_analytic",
              "r_montecarlo", "phantom_fates", "tracks", "existence_map", "cardinality",
              "gospa", "nees", "lifetimes")
# The step-5 figures that work for any filter: each reads only (run, filter_name).
ANY_FILTER_PLOTS = {
    "tracks": plot_tracks,
    "existence_map": plot_existence_map,
    "cardinality": plot_cardinality,
    "gospa": plot_gospa,
    "nees": plot_nees,
    "lifetimes": plot_lifetimes,
}
# The ones a simulate-only run folder (no filter, no estimates) can draw.
SCENARIO_PLOTS = ("scene", "counts")
# The ones built on the A2 closed form, which assumes multiplicity "single".
B3_PLOTS = ("r_vs_analytic", "r_montecarlo")


def analyse_run(run: RunDir, plots: Sequence[str] | None = None,
                scene_k: int | None = None) -> None:
    """Render the configured plots and write metrics.json for one run folder. [B1/B2/B3]

    Reads `run.config`. For a `track` run that is a RunConfig: each requested plot is
    rendered into run.plots; `r_vs_analytic` also writes the B3 cross-check to run.metrics
    (see `_cross_check`), and `r_montecarlo` the per-seed cross-check over
    `analysis.monte_carlo.n_runs` seeds (see `_monte_carlo`). For a simulate-only run it is
    a ScenarioConfig: only the B1 plots (SCENARIO_PLOTS) are available, and `plots`
    defaults to all of them.

    Per-track plots (`r_vs_k`, `r_vs_analytic`) are drawn for every track in the estimates
    log: `<plot>.png` when there is one track, `<plot>_track<id>.png` when there are
    several. The ANY_FILTER_PLOTS (roadmap step 5, D32) read only the run folder and the
    filter's name. Relative paths inside the config copy resolve against the working
    directory, as for `track`.

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
            if a B3 plot is requested without `analysis.b3_reference`, or `r_montecarlo`
            without `analysis.monte_carlo`; or if a B3 plot is requested for a scenario
            whose multiplicity is not "single" (the A2 closed form assumes at most one
            detection per plant).
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
        elif name == "r_montecarlo":
            _monte_carlo(run, cfg, out)
        elif name == "phantom_fates":
            _phantom_fates(run, cfg, out)
        else:
            ANY_FILTER_PLOTS[name](run, filter_name, out)


def _cross_check(run: RunDir, cfg: RunConfig) -> None:
    """The B3 cross-check of every track, into run.metrics and one r_vs_analytic plot each.

    `crosscheck.cross_check_run` does the work; a pruning filter is checked on its
    unpruned companion log (D14). metrics.json gets a "b3_cross_check" entry: the
    reference, the log checked, the tolerance, and per track the errors, the first
    divergence and the branch coverage (D26).
    """
    checks = cross_check_run(run, cfg)
    log = run.estimates(checked_log_name(cfg.filter_cfg))
    if not checks:
        raise ValueError(f"{log} reports no track")
    track_ids = [check.track_id for check in checks]
    tracks = []
    for check in checks:
        plot_r_vs_analytic(
            check.r_sim, check.r_ref,
            _per_track_png(run, "r_vs_analytic", check.track_id, track_ids),
            title=f"{cfg.name}, track {check.track_id}: filter r against A2 "
                  f"(max |error| {check.comparison.max_abs_error:.1e})")
        tracks.append({"track_id": check.track_id,
                       "max_abs_error": check.comparison.max_abs_error,
                       "rms_error": check.comparison.rms_error,
                       "first_divergence_k": check.comparison.first_divergence_k,
                       "branches": check.branches})
    _write_metrics(run, "b3_cross_check", {"reference": cfg.analysis.b3_reference,
                                           "estimates_log": log.name,
                                           "tolerance": R_TOLERANCE,
                                           "tracks": tracks})


def _monte_carlo(run: RunDir, cfg: RunConfig, out: Path) -> None:
    """The per-seed cross-check over many seeds (D17), and the descriptive mean-r figure.

    Seeds cfg.seed, cfg.seed + 1, ... - the first is this run's own - each simulated,
    filtered and checked in a temporary folder inside the run folder, deleted afterwards.
    metrics.json gets a "b3_monte_carlo" entry: the `CrossCheckSummary`. The figure shows
    the mean r of the first track with its standard-error band, and no closed form: the
    mean of r is not the closed form at any one event sequence.
    """
    if cfg.analysis.b3_reference is None or cfg.analysis.monte_carlo is None:
        raise ValueError("plot 'r_montecarlo' needs analysis.b3_reference and "
                         "analysis.monte_carlo in the run config")
    trials = run_trials(cfg, cfg.analysis.monte_carlo.n_runs, cfg.seed, run.root,
                        {"r": first_track_r, "cross_check": cross_check_run})
    plot_r_montecarlo(mean_r(trials), None, out)
    _write_metrics(run, "b3_monte_carlo", asdict(cross_check_summary(trials)))


def _phantom_fates(run: RunDir, cfg: RunConfig, out: Path) -> None:
    """The controlled phantom's fate over seeds (roadmap step 4a, D28, D34).

    Seeds cfg.seed, cfg.seed + 1, ... - the first is this run's own - each simulated and
    filtered in a temporary folder inside the run folder; each seed's one outcome is the
    phantom's fate and what fell in its gate. metrics.json gets a "phantom_fates" entry: the
    thresholds, the proportions with Wilson intervals per stratum, and the per-seed
    outcomes, so any stratification can be redone without rerunning.
    """
    if cfg.analysis.monte_carlo is None:
        raise ValueError("plot 'phantom_fates' needs analysis.monte_carlo in the run config")
    trials = run_trials(cfg, cfg.analysis.monte_carlo.n_runs, cfg.seed, run.root,
                        {"phantom": phantom_outcome})
    outcomes = [trial.values["phantom"] for trial in trials
                if trial.values["phantom"] is not None]
    proportions = fate_proportions(outcomes)
    plot_phantom_fates(proportions, out,
                       title=f"{cfg.name}: the phantom's fate over {len(outcomes)} seeds")
    _write_metrics(run, "phantom_fates", {
        "r_conf": R_CONF,
        "d_match": D_MATCH,
        "n_runs": len(trials),
        "n_without_phantom": len(trials) - len(outcomes),
        "proportions": {stratum: {fate: asdict(share) for fate, share in shares.items()}
                        for stratum, shares in proportions.items()},
        "seeds": [{"seed": trial.seed, **asdict(trial.values["phantom"])}
                  for trial in trials if trial.values["phantom"] is not None],
    })


def _write_metrics(run: RunDir, key: str, value: object) -> None:
    """Set one entry of run.metrics, keeping the others."""
    metrics = json.loads(run.metrics.read_text(encoding="utf-8")) if run.metrics.is_file() else {}
    metrics[key] = value
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
