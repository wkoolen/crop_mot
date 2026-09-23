"""Shared pytest fixtures. [B1-B4]

Deliberately small. Fixtures build tiny scenarios - a handful of plants, a handful of scans
- because a test that takes a second to run does not get run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crop_mot.config import RunConfig, ScenarioConfig
from crop_mot.runner.run_dir import RunDir

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def tiny_scenario() -> ScenarioConfig:
    """A deliberately small B1 scenario: one short row, a few scans.

    Small enough that a failing assertion can be reasoned about by hand, which is the whole
    point - a test over 60 scans and 70 plants tells you something is wrong but not what.

    Returns:
        A ScenarioConfig with a single row of a few plants and roughly five scans.
    """
    raise NotImplementedError


@pytest.fixture
def tiny_run_config(tiny_scenario: ScenarioConfig) -> RunConfig:
    """A B2 run config over `tiny_scenario`, with a Bernoulli filter and a phantom birth.

    Args:
        tiny_scenario: the scenario fixture.

    Returns:
        A RunConfig ready to hand to the filter runner.
    """
    raise NotImplementedError


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> RunDir:
    """An empty run folder under pytest's tmp_path.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        A RunDir whose subdirectories already exist.
    """
    raise NotImplementedError
