"""Every figure renders for every registered filter (roadmap step 7, §2 rule 1). [B4]

The figure-side twin of `test_filter_interface_contract.py`. Parametrised over FILTERS and
over `analyse.ANY_FILTER_PLOTS` rather than listing either, on purpose: the day a filter
or a figure is added, this file checks the pair without being edited.

A figure may declare a filter "not applicable" (roadmap §2 rule 1): it then draws a
labelled placeholder panel, which counts as rendering. An exception does not. The B2/B3
figures (r_vs_k, hypotheses, r_vs_analytic, r_montecarlo) are outside this contract: they
draw the A2 recursion's quantities and are Bernoulli-specific by design.

Each filter is built from one run config's filter block with only `kind` replaced, as
`track_all_filters` and `test_filter_interface_contract.py` do.
"""

from __future__ import annotations

import os

import pytest

from crop_mot.config import load_run_config
from crop_mot.filters import FILTERS
from crop_mot.runner.analyse import ANY_FILTER_PLOTS
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import track_all_filters, track_from_config

from conftest import CONFIGS, REPO_ROOT

FILTER_NAMES = sorted(FILTERS)
PLOT_NAMES = list(ANY_FILTER_PLOTS)


@pytest.fixture(scope="module")
def contract_run(tmp_path_factory) -> RunDir:
    """The shipped phantom run, with every registered filter run on its detections."""
    runs = tmp_path_factory.mktemp("runs")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        run = track_from_config(CONFIGS / "b2_bernoulli_phantom.yaml", runs)
    finally:
        os.chdir(cwd)
    track_all_filters(load_run_config(run.config), run, FILTER_NAMES)
    return run


@pytest.mark.parametrize("plot_name", PLOT_NAMES)
@pytest.mark.parametrize("filter_name", FILTER_NAMES)
def test_figure_renders_for_filter(contract_run: RunDir, filter_name: str,
                                   plot_name: str) -> None:
    """The figure writes a non-empty PNG from (run, filter_name) alone. [B4]"""
    out = contract_run.plots / "contract" / filter_name / f"{plot_name}.png"
    assert ANY_FILTER_PLOTS[plot_name](contract_run, filter_name, out) == out
    assert out.stat().st_size > 0
