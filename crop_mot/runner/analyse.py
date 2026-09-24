"""B2/B3 entry point: a finished run folder -> plots and metrics. [B2/B3]

The `python -m crop_mot analyse` subcommand. Added during implementation: the skeleton's
CLI named an `analyse` subcommand but had no runner function behind it, and `__main__` is
meant to stay thin. Everything here reads from the run folder only; by the time it runs, the
estimates it judges are already on disk.
"""

from __future__ import annotations

from collections.abc import Sequence

from crop_mot.runner.run_dir import RunDir


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

    Raises:
        ValueError: if a plot name is unknown.
    """
    raise NotImplementedError
