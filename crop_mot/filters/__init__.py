"""Filter registry. Adding a B4 filter is a new file plus one line here. [B2/B4]

A plain dict, deliberately: no entry-point plugin system, no metaclass registry, no
decorator that runs at import time. You can read the whole mechanism in ten seconds and
there is exactly one place to look when a filter name does not resolve.

To add a filter in phase 2:
  1. Write crop_mot/filters/<name>.py with a class satisfying TrackingFilter and a
     build_<name>(cfg: FilterConfig) function.
  2. Add one line to FILTERS below.
  3. Nothing else. test_filter_interface_contract.py picks it up automatically, the runner
     is unchanged, and it reads the same detections.jsonl as every other filter.
"""

from __future__ import annotations

from collections.abc import Callable

from crop_mot.config import FilterConfig
from crop_mot.filters.base import TrackingFilter
from crop_mot.filters.bernoulli import build_bernoulli

FILTERS: dict[str, Callable[[FilterConfig], TrackingFilter]] = {
    "bernoulli": build_bernoulli,
    # Phase 2 (B4) - one line each, in roughly increasing order of difficulty:
    #   "gnn":   build_gnn,    # baseline: hard assignment via linear_sum_assignment
    #   "pda":   build_pda,    # many measurements, one target
    #   "jpda":  build_jpda,   # many measurements, many targets
    #   "pmb":   build_pmb,    # Poisson + multi-Bernoulli
    #   "pmbm":  build_pmbm,   # + a mixture over global hypotheses
}


def build_filter(cfg: FilterConfig) -> TrackingFilter:
    """Construct the filter named by cfg.kind.

    Serves: [B2] the runner; [B4] every baseline, through the same call.

    Args:
        cfg: the `filter:` block of a run config.

    Returns:
        A constructed filter satisfying TrackingFilter.

    Raises:
        KeyError: if cfg.kind is not in FILTERS. The message should list the available
            names, since a typo here is otherwise a confusing failure.
    """
    raise NotImplementedError
