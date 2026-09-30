"""The compare entry point: several filters on one run folder, figures side by side. [B4]

The `python -m crop_mot compare` subcommand (roadmap step 6). Every filter runs on the same
detections.jsonl (`track_all_filters`), every figure that works for any filter
(`analyse.ANY_FILTER_PLOTS`, D32) is drawn once per filter, and the drawings are put next
to each other, one composite per figure. The per-filter figures stay in
plots/compare/<filter>/ so each can be used on its own.

metrics.json gets a "compare" entry with one row per filter holding the roadmap's evidence
for choosing a method: accuracy (mean GOSPA), consistency (the share of scans whose
average NEES lies inside its 95 % band) and cost (median update time per scan).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread

from crop_mot.analysis.evaluation import gospa_series, nees, nees_band, scan_views
from crop_mot.analysis.plots import INK, SURFACE
from crop_mot.config import load_run_config
from crop_mot.io import read_jsonl, to_jsonable
from crop_mot.runner.analyse import ANY_FILTER_PLOTS
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import track_all_filters


def filter_summary(run: RunDir, filter_name: str) -> dict[str, float]:
    """The evidence row of one filter: accuracy, consistency and cost. [B4]

    Returns:
        "mean_gospa" in metres; "nees_in_band", the share of scans with at least one NEES
        pair whose average lies inside its 95 % band (NaN when there are none);
        "median_update_s" over scans 1.. of the timing log.
    """
    views = scan_views(run, filter_name)
    results = gospa_series(views)
    error = nees(views, results)
    lower, upper = nees_band(error.n_pairs, error.dim)
    paired = error.n_pairs > 0
    inside = (error.mean_nees >= lower) & (error.mean_nees <= upper)
    timing = list(read_jsonl(run.timing(filter_name)))
    return {
        "mean_gospa": float(np.mean([result.distance for result in results])),
        "nees_in_band": float(inside[paired].mean()) if paired.any() else float("nan"),
        "median_update_s": float(np.median([row["update_s"] for row in timing[1:]])),
    }


def compare_run(run: RunDir, filter_names: Sequence[str],
                plots: Sequence[str] | None = None) -> list[Path]:
    """Run the named filters on one run folder and draw each figure side by side. [B4]

    Args:
        run: a run folder with detections and a run config whose filter block builds every
            named filter (only its `kind` is replaced).
        filter_names: keys into `crop_mot.filters.FILTERS`, in the order to show them.
        plots: names from ANY_FILTER_PLOTS; None means all of them.

    Returns:
        The composite figures written, plots/compare_<plot>.png, one per plot.

    Raises:
        KeyError: if a filter or plot name is unknown.
    """
    names = list(plots) if plots is not None else list(ANY_FILTER_PLOTS)
    unknown = [name for name in names if name not in ANY_FILTER_PLOTS]
    if unknown:
        raise KeyError(f"unknown plot(s) {unknown}; available: {list(ANY_FILTER_PLOTS)}")
    cfg = load_run_config(run.config)
    track_all_filters(cfg, run, list(filter_names))

    written = []
    for name in names:
        drawings = []
        for filter_name in filter_names:
            out = run.plots / "compare" / filter_name / f"{name}.png"
            ANY_FILTER_PLOTS[name](run, filter_name, out)
            drawings.append(out)
        written.append(_side_by_side(drawings, list(filter_names),
                                     run.plots / f"compare_{name}.png"))

    summary = {filter_name: filter_summary(run, filter_name) for filter_name in filter_names}
    metrics = json.loads(run.metrics.read_text(encoding="utf-8")) if run.metrics.is_file() else {}
    metrics["compare"] = {name: {key: (None if np.isnan(value) else value)
                                 for key, value in row.items()}
                          for name, row in summary.items()}
    run.metrics.write_text(json.dumps(to_jsonable(metrics), indent=2) + "\n", encoding="utf-8")
    return written


def _side_by_side(images: list[Path], titles: list[str], out: Path) -> Path:
    """Place saved figures next to each other, each under its filter's name."""
    pictures = [imread(path) for path in images]
    height, width = pictures[0].shape[:2]
    panel_width = 7.0
    fig = Figure(figsize=(panel_width * len(pictures),
                          panel_width * height / width + 0.4),
                 facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(pictures), squeeze=False)[0]
    for ax, picture, title in zip(axes, pictures, titles):
        ax.imshow(picture)
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out
