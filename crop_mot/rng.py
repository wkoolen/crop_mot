"""Named random substreams derived from one seed. [B1, reproducibility]

Why this file exists at all: if the simulator drew everything from a single generator, then
changing lambda_FA would change how many clutter draws happen, which would shift every
subsequent draw, which would move the plants. Two runs that differ only in clutter rate
would then have different fields, and the comparison would be meaningless.

Splitting the seed into named substreams fixes that: `field` always produces the same plant
positions for a given seed, no matter what the detector is doing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# The substreams the pipeline uses. Adding a name here is cheap; renaming one changes every
# run's output for a given seed, so treat these as part of the reproducibility contract.
STREAM_NAMES: tuple[str, ...] = ("field", "path", "detection", "clutter")


def stream_key(name: str) -> int:
    """Map a substream name to a stable integer for seeding.

    Uses zlib.crc32 rather than the builtin hash(): Python randomises string hashing per
    process (PYTHONHASHSEED), so hash("field") differs between runs and would silently
    destroy reproducibility while looking correct.

    Serves: [B1] reproducibility.

    Args:
        name: substream name, e.g. "field".

    Returns:
        A deterministic non-negative integer, identical across processes and platforms.
    """
    raise NotImplementedError


def substreams(seed: int, names: Sequence[str] = STREAM_NAMES) -> dict[str, np.random.Generator]:
    """Build one independent numpy Generator per named substream.

    Each generator is seeded from np.random.SeedSequence([seed, stream_key(name)]), so the
    streams are independent of each other and independent of the ORDER in which they are
    created - unlike SeedSequence.spawn(), where inserting a new stream shifts the others.

    Serves: [B1] the simulator draws plant jitter from "field", pose noise from "path",
    detection coin flips from "detection" and clutter from "clutter";
    [B3] `run_monte_carlo` varies the top-level seed and reuses this function.

    Args:
        seed: the single integer that, with the config file, defines a run.
        names: substream names to create. Defaults to STREAM_NAMES.

    Returns:
        Mapping from name to a freshly seeded np.random.Generator.
    """
    raise NotImplementedError
