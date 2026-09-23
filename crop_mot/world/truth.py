"""The ground-truth bundle and its on-disk form. [B1/B3]

truth.jsonl is written by the simulator and read only by the evaluation code. No filter is
ever given a path to it, and no filter function takes a GroundTruth argument - that is how
"the filter never sees ground truth" is enforced structurally rather than by discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crop_mot.world.field import PlantField
from crop_mot.world.path import PoseSample


@dataclass(frozen=True)
class GroundTruth:
    """Everything true about a scenario. Evaluation only. [B1/B3]

    Attributes:
        field: the true plant positions.
        poses: the per-scan true and reported poses.
    """

    field: PlantField
    poses: list[PoseSample]


def write_truth(path: Path, truth: GroundTruth) -> None:
    """Write ground truth to truth.jsonl in a run folder.

    Records both the true and the reported pose per scan. With pose_known = True they are
    equal, and `test_b1_simulator` asserts exactly that - which is a cheap way to catch a
    regression where the wobble accidentally gets applied.

    Serves: [B1] run-folder output; [B3] the analysis reads it back to build ScanEvents.

    Args:
        path: destination truth.jsonl.
        truth: the bundle to serialise.
    """
    raise NotImplementedError


def read_truth(path: Path) -> GroundTruth:
    """Read truth.jsonl back into a GroundTruth.

    Serves: [B3] the analysis and plotting step, which runs after the filter and therefore
    cannot rely on the simulator still being in memory.

    Args:
        path: source truth.jsonl.

    Returns:
        The deserialised ground truth.
    """
    raise NotImplementedError
