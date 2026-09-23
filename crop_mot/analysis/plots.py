"""Figures, written to the run folder's plots/ directory. [B1/B2/B3]

Every function SAVES a PNG and returns the path; none of them call plt.show(). The
container is headless (MPLBACKEND=Agg in the Dockerfile), so a show() would silently do
nothing there and behave differently on a desktop - and a figure that only exists on screen
cannot be traced back to the run that produced it.

Each figure is written into the run folder next to the data it was made from, so a thesis
figure is always one directory away from its config and its seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crop_mot.analysis.montecarlo import MonteCarloResult
from crop_mot.runner.run_dir import RunDir


def plot_scene(run: RunDir, out: Path) -> Path:
    """Top-down view of the field, the robot path and one scan's detections. [B1]

    The sanity-check figure for B1, and the one that catches geometry bugs fastest: plants
    as points, the robot path as a line, the FOV wedge at a chosen scan, and that scan's
    detections. Clutter and true detections are distinguishable here because this is
    evaluation code and may read labels.jsonl.

    Args:
        run: the run folder to read from.
        out: destination PNG path.

    Returns:
        The path written.
    """
    raise NotImplementedError


def plot_r_vs_k(run: RunDir, filter_name: str, track_id: int, out: Path) -> Path:
    """Existence probability against scan index. THE B2 DELIVERABLE. [B2]

    For the phantom scenario this is the r-decay curve: a track born from a clutter
    detection, whose existence probability falls as scan after scan fails to produce a
    detection at that location.

    Should mark which scans produced a gated detection, since a flat stretch in r is only
    interesting once you know whether it was caused by a detection or by the target leaving
    the FOV.

    Args:
        run: the run folder.
        filter_name: which estimates log to read.
        track_id: which track to follow.
        out: destination PNG path.

    Returns:
        The path written.
    """
    raise NotImplementedError


def plot_r_vs_analytic(
    r_sim: np.ndarray, r_ref: np.ndarray, out: Path, title: str = ""
) -> Path:
    """The B3 cross-check figure: filter r against the closed form. [B3]

    Two curves plus their difference. The difference panel is the useful one - on the same
    axes the two curves are indistinguishable when the check passes, which makes a passing
    result look like a single line and an informative figure look like a bug.

    Args:
        r_sim: shape (K,), the filter's r trajectory.
        r_ref: shape (K,), the analytic reference's.
        out: destination PNG path.
        title: optional title, e.g. the p_D profile being validated.

    Returns:
        The path written.
    """
    raise NotImplementedError


def plot_r_montecarlo(result: MonteCarloResult, r_ref: np.ndarray, out: Path) -> Path:
    """Mean r over many seeds, with a confidence band, against the closed form. [B3]

    The empirical half of B3. Plots the mean plus or minus a few standard errors - NOT plus
    or minus the standard deviation, which measures the spread of individual runs rather
    than the uncertainty in the mean, and would produce a band wide enough to hide a real
    modelling error.

    Args:
        result: aggregate from `run_monte_carlo`.
        r_ref: shape (K,), the closed-form reference.
        out: destination PNG path.

    Returns:
        The path written.
    """
    raise NotImplementedError
