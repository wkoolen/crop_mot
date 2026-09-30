"""The compare command: several filters on one run folder, side by side (roadmap step 6).

Checked on the case whose answer is known: a one-seed bank without pruning is the single
Bernoulli filter (D12), so every accuracy and consistency number must agree between them.
"""

from __future__ import annotations

import json
import os

import pytest

from crop_mot.runner.compare import compare_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.track import track_from_config

from conftest import CONFIGS, REPO_ROOT


@pytest.fixture(scope="module")
def phantom_run(tmp_path_factory) -> RunDir:
    """The shipped single-phantom config, tracked into a temporary runs/."""
    runs = tmp_path_factory.mktemp("runs")
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        return track_from_config(CONFIGS / "b2_bernoulli_phantom.yaml", runs)
    finally:
        os.chdir(cwd)


def test_compare_draws_side_by_side_and_agrees_where_it_must(phantom_run: RunDir) -> None:
    filters = ["bernoulli", "bernoulli_bank"]
    written = compare_run(phantom_run, filters, plots=["gospa", "cardinality"])

    assert written == [phantom_run.plots / "compare_gospa.png",
                       phantom_run.plots / "compare_cardinality.png"]
    for name in filters:
        assert (phantom_run.plots / "compare" / name / "gospa.png").is_file()
        assert phantom_run.estimates(name).is_file()
    rows = json.loads(phantom_run.metrics.read_text(encoding="utf-8"))["compare"]
    assert set(rows) == set(filters)
    assert rows["bernoulli"]["mean_gospa"] == rows["bernoulli_bank"]["mean_gospa"]
    assert rows["bernoulli"]["nees_in_band"] == rows["bernoulli_bank"]["nees_in_band"]


def test_compare_refuses_an_unknown_plot(phantom_run: RunDir) -> None:
    with pytest.raises(KeyError, match="hypotheses"):
        compare_run(phantom_run, ["bernoulli"], plots=["hypotheses"])
