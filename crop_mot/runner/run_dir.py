"""Run folders: one directory per experiment, self-describing. [reproducibility]

A run folder is the unit of reproducibility. It contains everything needed to explain a
figure six months later:

    runs/<timestamp>_<name>_seed<seed>/
      config.yaml                  verbatim copy of the config used
      run_meta.json                seed, git sha, package versions, argv, timestamp
      truth.jsonl                  plant positions + true and reported poses  (EVAL ONLY)
      labels.jsonl                 per-detection origin ids                   (EVAL ONLY)
      detections.jsonl             the ONLY filter input
      estimates_<filter>.jsonl     one per filter - this is what makes comparison fair
      metrics.json
      plots/

Note that several filters share ONE run folder. Simulate once, then track repeatedly into
the same directory, and the fact that they saw identical data is visible in the file tree.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# The repository root, for asking git which commit produced a run.
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunDir:
    """A run folder and the paths inside it. [reproducibility]

    Attributes:
        root: the run folder itself.
    """

    root: Path

    @property
    def config(self) -> Path:
        """Path to the verbatim config copy."""
        return self.root / "config.yaml"

    @property
    def meta(self) -> Path:
        """Path to run_meta.json."""
        return self.root / "run_meta.json"

    @property
    def truth(self) -> Path:
        """Path to truth.jsonl. EVALUATION ONLY - never handed to a filter."""
        return self.root / "truth.jsonl"

    @property
    def labels(self) -> Path:
        """Path to labels.jsonl. EVALUATION ONLY - never handed to a filter."""
        return self.root / "labels.jsonl"

    @property
    def detections(self) -> Path:
        """Path to detections.jsonl - the only input a filter is given."""
        return self.root / "detections.jsonl"

    @property
    def metrics(self) -> Path:
        """Path to metrics.json."""
        return self.root / "metrics.json"

    @property
    def plots(self) -> Path:
        """Path to the plots/ subdirectory."""
        return self.root / "plots"

    def estimates(self, filter_name: str) -> Path:
        """Path to estimates_<filter_name>.jsonl.

        One file per filter in the same run folder is what makes the fair comparison
        structural: all of them necessarily read the same detections.jsonl sitting next to
        them.

        Args:
            filter_name: the filter's `name` attribute, e.g. "bernoulli".

        Returns:
            The estimates log path for that filter.
        """
        return self.root / f"estimates_{filter_name}.jsonl"


def create_run_dir(base: Path, name: str, seed: int, config_source: Path) -> RunDir:
    """Create a new timestamped run folder and copy the config into it.

    The copy is taken BEFORE anything runs, so that a crashed run still records what it was
    trying to do. The folder name embeds the scenario name and seed so that `ls runs/` is
    already informative.

    Serves: [B1] `simulate`; [B2] `track` when starting a fresh run.

    Args:
        base: the runs/ directory.
        name: scenario or run name from the config.
        seed: the run's seed.
        config_source: the config file to copy in verbatim.

    Returns:
        The created RunDir, with plots/ already created.

    The timestamp is UTC. If a folder with the same name already exists (two runs started
    within the same second, e.g. in a test), a suffix _1, _2, ... is appended rather than
    overwriting it.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    stem = f"{stamp}_{name}_seed{seed}"
    root = base / stem
    suffix = 0
    while root.exists():
        suffix += 1
        root = base / f"{stem}_{suffix}"

    root.mkdir(parents=True)
    run = RunDir(root)
    shutil.copyfile(config_source, run.config)
    run.plots.mkdir()
    return run


def write_run_meta(run: RunDir, seed: int, argv: list[str]) -> None:
    """Record the provenance of this run.

    Captures the seed, the git commit SHA (and whether the tree was dirty), the installed
    numpy/scipy/matplotlib versions, the Python version, the command line, and a UTC
    timestamp.

    The dirty-tree flag earns its place: "the plot came from commit abc123" is worthless if
    there were uncommitted changes, and that is the normal state of a thesis repo.

    Serves: reproducibility across the whole project.

    Args:
        run: the run folder.
        seed: the seed used.
        argv: sys.argv of the invoking process.
    """
    meta = {
        "seed": seed,
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": None,
        "python": platform.python_version(),
        "packages": {name: _package_version(name)
                     for name in ("numpy", "scipy", "matplotlib", "PyYAML")},
        "argv": list(argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    status = _git(["status", "--porcelain"])
    if status is not None:
        meta["git_dirty"] = status != ""

    with run.meta.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _git(args: list[str]) -> str | None:
    """Output of a git command in the repo, or None if git is unavailable."""
    try:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    """Installed version of a distribution, without importing it."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
