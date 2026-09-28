"""B1/B2/B3 entry point: a finished run folder -> plots and metrics. [B1/B2/B3]

The `python -m crop_mot analyse` subcommand. Added during implementation: the skeleton's
CLI named an `analyse` subcommand but had no runner function behind it, and `__main__` is
meant to stay thin. Everything here reads from the run folder only; by the time it runs, the
estimates it judges are already on disk.
"""

from __future__ import annotations

from collections.abc import Sequence

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.analysis.plots import (
    animate_hypotheses,
    load_run_folder_config,
    plot_counts,
    plot_hypotheses,
    plot_r_vs_k,
    plot_scene,
)
from crop_mot.config import ScenarioConfig
from crop_mot.runner.run_dir import RunDir

# Every plot `analyse` knows how to draw, as named in the config's `analysis.plots`.
PLOT_NAMES = ("scene", "counts", "r_vs_k", "hypotheses", "hypotheses_anim", "r_vs_analytic",
              "r_montecarlo")
# The ones a simulate-only run folder (no filter, no estimates) can draw.
SCENARIO_PLOTS = ("scene", "counts")
# The ones built on the A2 closed form, which assumes multiplicity "single".
B3_PLOTS = ("r_vs_analytic", "r_montecarlo")


def analyse_run(run: RunDir, plots: Sequence[str] | None = None) -> None:
    """Render the configured plots and write metrics.json for one run folder. [B1/B2/B3]

    Reads `run.config`. For a `track` run that is a RunConfig: each requested plot is
    rendered into run.plots, and - when the config names a `b3_reference` and the B3
    plots are requested - the r comparison is written to run.metrics. For a
    simulate-only run it is a ScenarioConfig: only the B1 plots (SCENARIO_PLOTS) are
    available, and `plots` defaults to all of them.

    Args:
        run: the run folder to analyse.
        plots: plot names to render, overriding the config's `analysis.plots`; None means
            use the config's list. Lets the B2 r_vs_k plot be produced through the CLI on
            its own.

    The track followed is the first track id in the estimates log (the only one, for the
    Bernoulli filter). Relative paths inside the config copy resolve against the working
    directory, as for `track`.

    Raises:
        ValueError: if a plot name is unknown or needs a filter run the folder does not
            hold; if the filter never reported a track; or if a B3 plot is requested for
            a scenario whose multiplicity is not "single" (the A2 closed form assumes at
            most one detection per plant).
    """
    cfg = load_run_folder_config(run)
    if isinstance(cfg, ScenarioConfig):
        names = tuple(plots) if plots is not None else SCENARIO_PLOTS
        _check_names(names, allowed=SCENARIO_PLOTS, where="a simulate-only run")
        for name in names:
            out = run.plots / f"{name}.png"
            if name == "scene":
                plot_scene(run, out)
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
    for name in names:
        out = run.plots / f"{name}.png"
        if name == "scene":
            plot_scene(run, out, k=cfg.filter_cfg.birth.at_scan)
        elif name == "counts":
            plot_counts(run, out)
        elif name == "r_vs_k":
            plot_r_vs_k(run, filter_name, _first_track_id(run, filter_name), out)
        elif name == "hypotheses":
            plot_hypotheses(run, filter_name, out)
        elif name == "hypotheses_anim":
            animate_hypotheses(run, filter_name, run.plots / "hypotheses.gif")
        else:
            raise NotImplementedError(f"plot {name!r} belongs to B3 (step 7)")


def _check_names(names: Sequence[str], allowed: Sequence[str], where: str) -> None:
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown} for {where}; available: {list(allowed)}")


def _first_track_id(run: RunDir, filter_name: str) -> int:
    records = read_estimates(run.estimates(filter_name))
    track_ids = [est.track_id for record in records for est in record.estimates]
    if not track_ids:
        raise ValueError(f"{run.estimates(filter_name)} reports no track")
    return track_ids[0]
