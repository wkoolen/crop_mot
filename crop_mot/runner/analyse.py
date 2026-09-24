"""B2/B3 entry point: a finished run folder -> plots and metrics. [B2/B3]

The `python -m crop_mot analyse` subcommand. Added during implementation: the skeleton's
CLI named an `analyse` subcommand but had no runner function behind it, and `__main__` is
meant to stay thin. Everything here reads from the run folder only; by the time it runs, the
estimates it judges are already on disk.
"""

from __future__ import annotations

from collections.abc import Sequence

from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.analysis.plots import plot_r_vs_k, plot_scene
from crop_mot.config import load_run_config
from crop_mot.runner.run_dir import RunDir

# Every plot `analyse` knows how to draw, as named in the config's `analysis.plots`.
PLOT_NAMES = ("scene", "r_vs_k", "r_vs_analytic", "r_montecarlo")


def analyse_run(run: RunDir, plots: Sequence[str] | None = None) -> None:
    """Render the configured plots and write metrics.json for one run folder. [B2/B3]

    Reads `run.config` as a RunConfig (the run folder of a `track` run holds a copy of the
    run config). Renders each requested plot into run.plots, and - when the config names a
    `b3_reference` and the B3 plots are requested - writes the r comparison to
    run.metrics.

    Args:
        run: the run folder to analyse.
        plots: plot names to render, overriding the config's `analysis.plots`; None means
            use the config's list. Lets the B2 r_vs_k plot be produced through the CLI on
            its own.

    The track followed is the first track id in the estimates log (the only one, for the
    Bernoulli filter). Relative paths inside the config copy resolve against the working
    directory, as for `track`.

    Raises:
        ValueError: if a plot name is unknown, or the filter never reported a track.
    """
    cfg = load_run_config(run.config)
    names = tuple(plots) if plots is not None else cfg.analysis.plots
    unknown = [name for name in names if name not in PLOT_NAMES]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown}; available: {list(PLOT_NAMES)}")

    filter_name = cfg.filter_cfg.kind
    records = read_estimates(run.estimates(filter_name))
    track_ids = [est.track_id for record in records for est in record.estimates]
    if not track_ids:
        raise ValueError(f"{run.estimates(filter_name)} reports no track")
    track_id = track_ids[0]

    for name in names:
        out = run.plots / f"{name}.png"
        if name == "scene":
            plot_scene(run, out, k=cfg.filter_cfg.birth.at_scan)
        elif name == "r_vs_k":
            plot_r_vs_k(run, filter_name, track_id, out)
        else:
            raise NotImplementedError(f"plot {name!r} belongs to B3 (step 7)")
