"""N bounded by the planting plan, with missing plants: roadmap step 8c (decision D23).

World side: each planned slot is empty with probability p_missing, from its own stream, so
the plants that are there do not move. Filter side: each slot starts at r_0 < 1 and is the
unchanged Bernoulli recursion - checked against A2 slot by slot - so what the bank does
with an empty slot is the independence approximation's doing, not a bug.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.crosscheck import cross_check_run
from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.analysis.metrics import R_TOLERANCE
from crop_mot.config import (
    AnalysisConfig,
    PlanConfig,
    RunConfig,
    ScenarioConfig,
    load_run_config,
    nominal_positions,
)
from crop_mot.filters import build_filter
from crop_mot.rng import substreams
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter
from crop_mot.world.field import generate_field
from crop_mot.world.truth import read_truth

from conftest import CONFIGS


def _missing(scenario: ScenarioConfig, p_missing: float, n_scans: int = 30) -> ScenarioConfig:
    return replace(scenario, world=replace(scenario.world, p_missing=p_missing),
                   path=replace(scenario.path, n_scans=n_scans))


def _bounded(cfg: RunConfig, scenario: ScenarioConfig, r_0: float) -> RunConfig:
    plan = PlanConfig(rows=scenario.world.rows, prior_std=0.036, r_0=r_0)
    return replace(cfg, scenario=scenario,
                   filter_cfg=replace(cfg.filter_cfg, kind="bernoulli_bank", birth=None,
                                      plan=plan),
                   analysis=AnalysisConfig(b3_reference="bernoulli_existence",
                                           monte_carlo=None, plots=()))


def _simulate(scenario: ScenarioConfig, root) -> RunDir:
    run = RunDir(root)
    run.plots.mkdir(parents=True)
    simulate(scenario, run, summary=False)
    return run


def test_missing_plants_leave_the_others_where_they_were(tiny_scenario) -> None:
    """Present plants keep their id (= slot) and position; empty slots sit on the grid."""
    long_row = replace(tiny_scenario.world, rows=(replace(tiny_scenario.world.rows[0],
                                                          y_end=40.0),))
    full = generate_field(long_row, substreams(7)["field"])
    streams = substreams(7)
    field = generate_field(replace(long_row, p_missing=0.3), streams["field"],
                           streams["missing"])
    nominal, _ = nominal_positions(long_row.rows)
    assert len(field.ids) + len(field.missing_ids) == len(nominal)
    assert set(field.ids) | set(field.missing_ids) == set(range(len(nominal)))
    assert np.array_equal(field.positions, full.positions[field.ids])
    assert np.array_equal(field.missing_positions, nominal[field.missing_ids])
    n = len(nominal)
    assert abs(len(field.missing_ids) / n - 0.3) < 4 * np.sqrt(0.3 * 0.7 / n)


def test_truth_records_the_empty_slots(tiny_scenario, tmp_path) -> None:
    run = _simulate(_missing(tiny_scenario, 0.5), tmp_path / "r")
    truth = read_truth(run.truth)
    assert len(truth.field.missing_ids) > 0
    assert truth.field.missing_positions.shape == (len(truth.field.missing_ids), 2)


def test_r_0_one_is_known_n(tiny_run_config, tmp_path) -> None:
    """The reduction of step 8c: r_0 = 1 gives exactly the step-8a map."""
    scenario = _missing(tiny_run_config.scenario, 0.3)
    run = _simulate(scenario, tmp_path / "r")
    known = replace(_bounded(tiny_run_config, scenario, 1.0).filter_cfg,
                    plan=PlanConfig(rows=scenario.world.rows, prior_std=0.036))
    run_filter(build_filter(known), run, log_name="known")
    run_filter(build_filter(_bounded(tiny_run_config, scenario, 1.0).filter_cfg), run,
               log_name="bounded")
    for a, b in zip(read_estimates(run.estimates("known")),
                    read_estimates(run.estimates("bounded"))):
        assert [(e.r, e.mean.tolist()) for e in a.estimates] == \
            [(e.r, e.mean.tolist()) for e in b.estimates]


def test_every_slot_is_the_a2_recursion(tiny_run_config, tmp_path) -> None:
    """Each slot, from r_0 and its plan prior at scan 0, matches A2 to 1e-12 (D18)."""
    scenario = _missing(replace(tiny_run_config.scenario,
                                path=replace(tiny_run_config.scenario.path, speed=0.2)),
                        0.3, n_scans=40)
    cfg = _bounded(tiny_run_config, scenario, 0.9)
    run = _simulate(scenario, tmp_path / "r")
    run_filter(build_filter(cfg.filter_cfg), run)
    checks = cross_check_run(run, cfg)
    assert len(checks) == len(nominal_positions(scenario.world.rows)[0])
    assert max(check.comparison.max_abs_error for check in checks) <= R_TOLERANCE
    covered = {branch: sum(check.branches[branch] for check in checks)
               for branch in ("out_of_view", "miss", "one_detection", "several_detections")}
    assert all(covered.values()), covered


def test_the_shipped_bounded_config() -> None:
    cfg = load_run_config(CONFIGS / "b4_bounded_n_bank.yaml")
    assert cfg.scenario.world.p_missing == 0.1
    assert cfg.filter_cfg.plan.r_0 == pytest.approx(1.0 - cfg.scenario.world.p_missing)
    assert cfg.filter_cfg.plan.rows == cfg.scenario.world.rows
    assert cfg.filter_cfg.birth is None
