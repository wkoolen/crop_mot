"""B2 entry point: recorded detections -> estimates log. [B2/B4]

The runner below is the payoff of the interface design. It contains no filter-specific
logic, no `if isinstance(...)`, and no notion of how many targets there are or whether the
filter maintains hypotheses. The same nine lines run Bernoulli today and PMBM in phase 2.
"""

from __future__ import annotations

from pathlib import Path

from crop_mot.config import RunConfig
from crop_mot.filters.base import TrackingFilter
from crop_mot.runner.run_dir import RunDir


def run_filter(flt: TrackingFilter, run: RunDir) -> None:
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
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
    raise NotImplementedError
