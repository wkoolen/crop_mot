"""JSONL read/write helpers, and the one place that knows how numpy meets JSON. [B1-B3]

Format choice: JSONL (one JSON object per line). It is human-readable, diffable, greppable
and needs no dependency beyond the standard library. Phase-1 trials are short - tens of
scans - so size is irrelevant.

Deliberately NOT future-proofed with an .npz backend: long trials arrive in phase 2 on
ROS 2, where the natural recording format is a rosbag anyway, so a second backend built now
would be generality that gets thrown away. If that turns out wrong, this is the one file to
change.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars and arrays into plain JSON-serialisable Python.

    Serves: [B1/B2] every record written to a run folder passes through here, so array
    encoding is defined once rather than in each writer.

    Args:
        obj: an arbitrary nested structure that may contain np.ndarray, np.floating or
            np.integer.

    Returns:
        The same structure with arrays as nested lists and numpy scalars as float/int.
    """
    raise NotImplementedError


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to a JSONL file, one compact JSON object per line.

    Writes with an explicit newline="\\n" so the file is byte-identical on Windows and
    Linux - which matters because `test_b1_simulator` asserts that the same seed reproduces
    detections.jsonl exactly.

    Serves: [B1] detections.jsonl, truth.jsonl, labels.jsonl; [B2] estimates_<filter>.jsonl.

    Args:
        path: destination file; parent directories must already exist.
        records: dicts that `to_jsonable` can handle.
    """
    raise NotImplementedError


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file lazily, yielding one dict per line.

    Serves: [B2] the filter runner reading detections.jsonl; [B3] the analysis reading
    estimates and labels.

    Args:
        path: source file.

    Yields:
        One decoded dict per non-empty line, in file order.
    """
    raise NotImplementedError
