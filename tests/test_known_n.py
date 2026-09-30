"""N known: the planting-plan prior of roadmap step 8a. [B4]

The bank started from the plan's N slots at r = 1, with no births. Checked: the slots are
the nominal plan (never truth), r stays exactly 1 (certainty is absorbing, A2 §5), a
missed slot keeps its prior Gaussian (constant p_D, A2 §2), and each slot evolves exactly
as it would alone - the bank's independence approximation (D12), which step 12's JPDA
will be measured against.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crop_mot.analysis.crosscheck import cross_check_run
from crop_mot.analysis.estimates_log import read_estimates
from crop_mot.config import PlanConfig, RowConfig, RunConfig, load_run_config, nominal_positions
from crop_mot.filters import build_filter
from crop_mot.filters.bernoulli_bank import plan_components
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.simulate import simulate
from crop_mot.runner.track import run_filter
from crop_mot.world.truth import read_truth

from conftest import CONFIGS

PRIOR_STD = 0.036


def _known_n(cfg: RunConfig, plan: PlanConfig) -> RunConfig:
    """The tiny run's filter as a known-N bank over `plan`."""
    return replace(cfg, filter_cfg=replace(cfg.filter_cfg, kind="bernoulli_bank", birth=None,
                                           plan=plan))


def _tiny_plan(cfg: RunConfig) -> PlanConfig:
    return PlanConfig(rows=cfg.scenario.world.rows, prior_std=PRIOR_STD)


@pytest.fixture
def tiny_run(tiny_run_config: RunConfig, tmp_path) -> RunDir:
    scenario = replace(tiny_run_config.scenario,
                       path=replace(tiny_run_config.scenario.path, n_scans=30))
    run = RunDir(tmp_path / "run")
    run.plots.mkdir(parents=True)
    simulate(scenario, run)
    return run


def test_the_shipped_configs_plan_the_scenarios_rows() -> None:
    for name in ("b4_known_n_bank.yaml", "b4_known_n_bank_weeds.yaml"):
        cfg = load_run_config(CONFIGS / name)
        assert cfg.filter_cfg.birth is None
        assert cfg.filter_cfg.plan.rows == cfg.scenario.world.rows
        assert cfg.filter_cfg.plan.prior_std == PRIOR_STD


def test_slots_are_the_nominal_plan_at_r_one(tiny_run_config: RunConfig, tiny_run) -> None:
    """Truth-blind: the slots sit on the nominal grid, not on the jittered true plants."""
    components = plan_components(_tiny_plan(tiny_run_config))
    nominal, _ = nominal_positions(tiny_run_config.scenario.world.rows)
    true_plants = read_truth(tiny_run.truth).field.positions
    assert len(components) == len(nominal) == len(true_plants)
    assert [c.track_id for c in components] == list(range(len(nominal)))
    assert all(c.r == 1.0 for c in components)
    assert np.array_equal(np.array([c.mean for c in components]), nominal)
    assert not np.array_equal(nominal, true_plants)
    assert np.array_equal(components[0].cov, PRIOR_STD**2 * np.eye(2))


def test_every_slot_stays_at_r_one_and_misses_keep_the_prior(
    tiny_run_config: RunConfig, tiny_run
) -> None:
    """r = 1 is absorbing [A2 §5]; with constant p_D a miss leaves the Gaussian as it was."""
    cfg = _known_n(tiny_run_config, _tiny_plan(tiny_run_config))
    flt = build_filter(cfg.filter_cfg)
    run_filter(flt, tiny_run)
    records = read_estimates(tiny_run.estimates(flt.name))

    n_slots = len(plan_components(cfg.filter_cfg.plan))
    for record in records:
        assert [e.track_id for e in record.estimates] == list(range(n_slots))
        assert all(e.r == 1.0 for e in record.estimates)
    # Scan 0: every plant is beyond max_range (conftest), so every slot is out of view.
    prior = plan_components(cfg.filter_cfg.plan)
    for estimate, slot in zip(records[0].estimates, prior):
        assert np.array_equal(estimate.mean, slot.mean)
        assert np.array_equal(estimate.cov, slot.cov)


def test_a_slot_in_the_bank_is_that_slot_alone(tiny_run_config: RunConfig, tiny_run) -> None:
    """Independence (D12): slot i's track equals a one-slot plan at slot i's position."""
    plan = _tiny_plan(tiny_run_config)
    run_filter(build_filter(_known_n(tiny_run_config, plan).filter_cfg), tiny_run)
    full = read_estimates(tiny_run.estimates("bernoulli_bank"))

    nominal, _ = nominal_positions(plan.rows)
    slot = 2
    x, y = nominal[slot]
    alone = PlanConfig(rows=(RowConfig(x=x, y_start=y, y_end=y, spacing=1.0),),
                       prior_std=PRIOR_STD)
    run_filter(build_filter(_known_n(tiny_run_config, alone).filter_cfg), tiny_run,
               log_name="alone")
    single = read_estimates(tiny_run.estimates("alone"))
    for record, record_alone in zip(full, single):
        (estimate,) = [e for e in record.estimates if e.track_id == slot]
        (estimate_alone,) = record_alone.estimates
        assert np.array_equal(estimate.mean, estimate_alone.mean)
        assert np.array_equal(estimate.cov, estimate_alone.cov)


def test_a_plan_is_refused_where_it_does_not_belong(tiny_run_config: RunConfig,
                                                    tiny_run) -> None:
    plan = _tiny_plan(tiny_run_config)
    with pytest.raises(ValueError, match="bernoulli_bank"):
        build_filter(replace(tiny_run_config.filter_cfg, plan=plan))
    with pytest.raises(ValueError, match="step 13"):
        build_filter(replace(tiny_run_config.filter_cfg, kind="bernoulli_bank", plan=plan))
    no_tracks = _known_n(tiny_run_config, plan)
    no_tracks = replace(no_tracks, filter_cfg=replace(no_tracks.filter_cfg, plan=None))
    with pytest.raises(ValueError, match="neither"):
        cross_check_run(tiny_run, no_tracks)


def test_gate_contents_on_the_shipped_known_n_run(tmp_path) -> None:
    """Step 8a's failure modes, from the labels, on the shipped config's own seed (D40).

    Neighbour detections sit in most gates at 0.35 m spacing, yet with a 3.6 cm prior the
    best branch keeps every plant's own detection: no plant is pulled (seed 42's value).
    """
    import os

    from crop_mot.analysis.evaluation import gate_contents
    from crop_mot.runner.track import track_from_config

    from conftest import REPO_ROOT

    cwd = os.getcwd()
    os.chdir(REPO_ROOT)  # the config's scenario path is relative to the repo root
    try:
        run = track_from_config(CONFIGS / "b4_known_n_bank.yaml", tmp_path)
    finally:
        os.chdir(cwd)
    contents = gate_contents(run, "bernoulli_bank")
    assert set(contents.shares) == {"neighbour_share", "clutter_share", "weed_share",
                                    "pulled_share"}
    assert contents.shares["neighbour_share"] > 0.5
    assert contents.shares["weed_share"] == 0.0          # no weeds in this field
    assert contents.shares["pulled_share"] == 0.0
    assert np.nanmin(contents.own) >= 0.0 and contents.n_tracks.max() > 10
