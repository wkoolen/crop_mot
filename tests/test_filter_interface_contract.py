"""Every registered filter satisfies the shared interface. [B2/B4]

This is the test that keeps B4 additive. A new filter added to `FILTERS` is picked up here
automatically, so "it plugs into the pipeline" is verified rather than assumed - and the
day PMBM is added, this file says whether it really does behave like the others.

Parametrised over FILTERS rather than listing filters explicitly, on purpose: an explicit
list is one more place to forget to update.
"""

from __future__ import annotations

import pytest

from crop_mot.config import RunConfig
from crop_mot.filters import FILTERS

FILTER_NAMES = sorted(FILTERS)


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_filter_implements_the_four_methods(filter_name: str) -> None:
    """The filter has initial_state, predict, update and extract, and they are callable. [B2/B4]

    A structural check that passes today against the Bernoulli builder and will pass for
    every B4 filter without being edited.
    """
    raise NotImplementedError


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_extract_returns_valid_track_estimates(
    filter_name: str, tiny_run_config: RunConfig, tmp_path
) -> None:
    """Whatever a filter reports is a well-formed TrackEstimate. [B2/B4]

    For every returned estimate: r is in [0, 1], mean has shape (dim_x,), cov has shape
    (dim_x, dim_x) and is symmetric, and track_id is stable across scans for the same track.

    This is the contract the evaluation code relies on. GNN's r of exactly 0.0 or 1.0 is a
    valid value here, not a special case - which is the point of having one interface.
    """
    raise NotImplementedError


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_filter_never_touches_ground_truth(filter_name: str, tmp_path) -> None:
    """Running a filter requires only detections.jsonl. [B2/B4]

    Run the filter against a run folder from which truth.jsonl and labels.jsonl have been
    DELETED, and assert it completes normally. A filter that had quietly started reading
    truth would fail here with a FileNotFoundError.

    Stronger than an inspection of imports, and it keeps holding as filters grow.
    """
    raise NotImplementedError


@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_all_filters_consume_the_same_detections(
    filter_name: str, tiny_run_config: RunConfig, tmp_path
) -> None:
    """Every filter reads the identical detection file in a shared run folder. [B4]

    The fair-comparison guarantee, made testable: simulate once, run every registered
    filter into the same run folder, and assert that detections.jsonl is unchanged
    afterwards and that each filter wrote its own estimates_<name>.jsonl.
    """
    raise NotImplementedError
