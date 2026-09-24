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

import json
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


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
    # json.dumps writes floats with repr(), the shortest string that parses back to the same
    # float64, so the round trip is exact. allow_nan=False: a NaN in a run file is a bug.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), separators=(",", ":"), allow_nan=False))
            f.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file lazily, yielding one dict per line.

    Serves: [B2] the filter runner reading detections.jsonl; [B3] the analysis reading
    estimates and labels.

    Args:
        path: source file.

    Yields:
        One decoded dict per non-empty line, in file order.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
