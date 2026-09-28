"""B2 entry point: recorded detections -> estimates log. [B2/B4]

The runner below is the payoff of the interface design. It contains no filter-specific
logic, no `if isinstance(...)`, and no notion of how many targets there are or whether the
filter maintains hypotheses. The same nine lines run Bernoulli today and PMBM in phase 2.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from crop_mot.analysis.estimates_log import ScanEstimates, write_estimates
from crop_mot.config import PruneConfig, RunConfig, load_run_config
from crop_mot.filters import FILTERS, build_filter
from crop_mot.filters.base import TrackingFilter
from crop_mot.runner.run_dir import RunDir, create_run_dir
from crop_mot.runner.simulate import simulate
from crop_mot.sensor.record import read_detections


def run_filter(flt: TrackingFilter, run: RunDir, log_name: str | None = None) -> None:
    """Run one filter over the recorded detections and write its estimates. [B2/B4]

    The loop, identical for every filter:

        state = flt.initial_state()
        for scan in read_detections(run.detections):
            state = flt.predict(state, dt)      # dt from consecutive scan timestamps
            state = flt.update(state, scan)
            write estimates for scan.k from flt.extract(state)

    dt is derived from the scan timestamps rather than from the config, so a scenario with
    an irregular or dropped scan does not silently desynchronise the prediction.

    If the filter also satisfies `HasDiagnostics`, its diagnostics are written alongside the
    estimates. Checked with hasattr rather than required, so that a B4 filter that does not
    care about diagnostics is not forced to implement an empty method.

    Reads ONLY run.detections. It is never given run.truth or run.labels, which is how
    "the filter never sees ground truth" is enforced at the call site as well as in the types.

    Serves: [B2] the Bernoulli run; [B4] every multi-target filter and the GNN baseline,
    through this identical function.

    Args:
        flt: any object satisfying TrackingFilter.
        run: the run folder holding detections.jsonl; estimates_<flt.name>.jsonl is written
            into the same folder.
        log_name: the estimates log suffix, default flt.name. Added for the unpruned
            companion run (decision D14), which is the same filter under another log name.
    """
    records = []
    state = flt.initial_state()
    previous_t = None
    for scan in read_detections(run.detections):
        dt = 0.0 if previous_t is None else scan.t - previous_t
        previous_t = scan.t
        state = flt.predict(state, dt)
        state = flt.update(state, scan)
        diagnostics = flt.diagnostics(state) if hasattr(flt, "diagnostics") else None
        records.append(ScanEstimates(k=scan.k, estimates=tuple(flt.extract(state)),
                                     diagnostics=diagnostics))
    write_estimates(run.estimates(log_name or flt.name), records)


def track_from_config(config_path: Path, runs_base: Path, run_dir: Path | None = None) -> RunDir:
    """Load a run config, ensure detections exist, and run the configured filter. [B2]

    The `python -m crop_mot track` entry point. If `run_dir` is given, the filter runs
    against that existing run folder's detections - which is how several filters end up
    sharing one detection set. If it is None, the referenced scenario is simulated first
    into a fresh run folder.

    Args:
        config_path: path to a B2 run YAML file.
        runs_base: the runs/ directory.
        run_dir: an existing run folder to reuse, or None to simulate a new one.

    Returns:
        The run folder containing the new estimates log.

    Raises:
        FileNotFoundError: if run_dir is given but contains no detections.jsonl.

    A fresh run folder is named after the RUN config and holds a copy of it; the scenario
    is simulated with its own seed, i.e. exactly as `simulate` on that scenario would.

    When the config prunes (filter.prune.r_min > 0), the same filter is also run with
    pruning off into estimates_<kind>_unpruned.jsonl (decision D14), on the same
    detections. That companion log is what the hypotheses figure draws after a deletion:
    what r would have done had the track been kept.
    """
    cfg = load_run_config(config_path)
    if run_dir is None:
        run = create_run_dir(runs_base, cfg.name, cfg.seed, config_path)
        simulate(cfg.scenario, run)
    else:
        run = RunDir(run_dir)
        if not run.detections.is_file():
            raise FileNotFoundError(f"{run.detections} does not exist; simulate first")
    run_filter(build_filter(cfg.filter_cfg), run)
    if cfg.filter_cfg.prune.r_min > 0.0:
        unpruned = replace(cfg.filter_cfg, prune=PruneConfig())
        run_filter(build_filter(unpruned), run, log_name=unpruned_log_name(cfg.filter_cfg.kind))
    return run


def unpruned_log_name(filter_name: str) -> str:
    """The estimates log suffix of a filter's unpruned companion run (decision D14)."""
    return f"{filter_name}_unpruned"


def track_all_filters(cfg: RunConfig, run: RunDir, filter_names: list[str]) -> None:
    """Run several filters over the SAME recorded detections. [B4]

    The fair-comparison entry point: Bernoulli, GNN, JPDA and PMBM against one detection
    set, one seed, one run folder. Because each writes estimates_<name>.jsonl next to the
    detections they all read, the comparison cannot accidentally be run on mismatched data.

    Serves: [B4] the baseline comparison the thesis reports.

    Args:
        cfg: the run config, supplying each filter's parameters.
        run: the run folder holding detections.jsonl.
        filter_names: keys into `crop_mot.filters.FILTERS`.

    Raises:
        KeyError: if any name is not registered in FILTERS.
    """
    unknown = [name for name in filter_names if name not in FILTERS]
    if unknown:
        raise KeyError(f"unknown filter(s) {unknown}; available: {sorted(FILTERS)}")
    for name in filter_names:
        run_filter(build_filter(replace(cfg.filter_cfg, kind=name)), run)
