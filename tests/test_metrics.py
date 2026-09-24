"""The B3 comparison and aggregation helpers, independent of the closed form. [B3, added]

Added during implementation so that `compare_r`, the Monte-Carlo aggregation and the B3
figures are checked on their own before the analytic reference exists - a failure in
`test_r_matches_analytic_recursion` should then point at the recursion, not at these.
"""

from __future__ import annotations

import numpy as np
import pytest

from crop_mot.analysis.metrics import compare_r
from crop_mot.analysis.montecarlo import MonteCarloResult, run_monte_carlo, standard_error
from crop_mot.analysis.plots import plot_r_montecarlo, plot_r_vs_analytic
from crop_mot.config import RunConfig


def test_compare_r_reports_the_first_divergence() -> None:
    r_ref = np.array([0.0, 0.5, 0.25, 0.125])
    r_sim = r_ref + np.array([0.0, 1e-12, 0.01, -0.02])

    result = compare_r(r_sim, r_ref, tol=1e-9)
    assert result.first_divergence_k == 2
    assert result.max_abs_error == pytest.approx(0.02)
    assert result.rms_error == pytest.approx(np.sqrt((1e-24 + 1e-4 + 4e-4) / 4))

    assert compare_r(r_ref, r_ref).first_divergence_k is None


def test_compare_r_refuses_different_lengths() -> None:
    with pytest.raises(ValueError):
        compare_r(np.zeros(3), np.zeros(4))


def test_standard_error_is_std_over_sqrt_n() -> None:
    result = MonteCarloResult(r_mean=np.zeros(2), r_std=np.array([0.2, 0.4]), n_runs=16,
                              seeds=tuple(range(16)))
    assert np.allclose(standard_error(result), [0.05, 0.1])


def test_monte_carlo_is_reproducible(tiny_run_config: RunConfig, tmp_path) -> None:
    first = run_monte_carlo(tiny_run_config, n_runs=4, base_seed=100, runs_base=tmp_path)
    second = run_monte_carlo(tiny_run_config, n_runs=4, base_seed=100, runs_base=tmp_path)

    n_scans = tiny_run_config.scenario.path.n_scans
    assert first.r_mean.shape == first.r_std.shape == (n_scans,)
    assert first.seeds == (100, 101, 102, 103)
    assert np.array_equal(first.r_mean, second.r_mean)
    assert np.all((first.r_mean >= 0.0) & (first.r_mean <= 1.0))
    assert list(tmp_path.iterdir()) == []  # per-run folders are cleaned up


def test_b3_figures_are_written(tmp_path) -> None:
    r_ref = 0.5 * 0.8 ** np.arange(10)
    out = plot_r_vs_analytic(r_ref + 1e-3, r_ref, tmp_path / "plots" / "a.png", title="t")
    assert out.is_file()

    result = MonteCarloResult(r_mean=r_ref, r_std=np.full(10, 0.1), n_runs=25,
                              seeds=tuple(range(25)))
    assert plot_r_montecarlo(result, r_ref, tmp_path / "plots" / "b.png").is_file()
