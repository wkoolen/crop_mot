"""The ground-truth bundle and its on-disk form. [B1/B3]

truth.jsonl is written by the simulator and read only by the evaluation code. No filter is
ever given a path to it, and no filter function takes a GroundTruth argument - that is how
"the filter never sees ground truth" is enforced structurally rather than by discipline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path

import numpy as np

from crop_mot.io import read_jsonl, write_jsonl
from crop_mot.types import Pose2D
from crop_mot.world.field import PlantField
from crop_mot.world.path import PoseSample


@dataclass(frozen=True)
class GroundTruth:
    """Everything true about a scenario. Evaluation only. [B1/B3]

    Attributes:
        field: the true plant positions.
        poses: the per-scan true and reported poses.
        weeds: shape (n_weeds, 2), the true weed positions; a weed's id is its row index.
            Shape (0, 2) for a field without weeds (decision D15).
    """

    field: PlantField
    poses: list[PoseSample]
    weeds: np.ndarray = dataclass_field(default_factory=lambda: np.empty((0, 2)))


def write_truth(path: Path, truth: GroundTruth) -> None:
    """Write ground truth to truth.jsonl in a run folder.

    Records both the true and the reported pose per scan. With pose_known = True they are
    equal, and `test_b1_simulator` asserts exactly that - which is a cheap way to catch a
    regression where the wobble accidentally gets applied.

    Serves: [B1] run-folder output; [B3] the analysis reads it back to build ScanEvents.

    Args:
        path: destination truth.jsonl.
        truth: the bundle to serialise.

    Layout: the first line is the plant field ({"kind": "field", ...}), the second the weeds
    ({"kind": "weeds", "positions"}, an empty list for a field without weeds); every
    following line is one scan's poses ({"kind": "pose", "k", "t", "true", "reported"}).
    """
    field_record = {
        "kind": "field",
        "ids": truth.field.ids,
        "positions": truth.field.positions,
        "row_index": truth.field.row_index,
    }
    if len(truth.field.missing_ids):
        # Only a field with missing plants carries these, so every other truth.jsonl is
        # unchanged (roadmap step 8c, D23).
        field_record["missing_ids"] = truth.field.missing_ids
        field_record["missing_positions"] = truth.field.missing_positions
    records = [field_record, {
        "kind": "weeds",
        "positions": truth.weeds,
    }]
    for k, sample in enumerate(truth.poses):
        records.append({
            "kind": "pose",
            "k": k,
            "t": sample.t,
            "true": asdict(sample.true),
            "reported": asdict(sample.reported),
        })
    write_jsonl(path, records)


def read_truth(path: Path) -> GroundTruth:
    """Read truth.jsonl back into a GroundTruth.

    Serves: [B3] the analysis and plotting step, which runs after the filter and therefore
    cannot rely on the simulator still being in memory.

    Args:
        path: source truth.jsonl.

    Returns:
        The deserialised ground truth. A truth.jsonl written before weeds existed has no
        weeds record and reads back with no weeds.
    """
    field = None
    weeds = np.empty((0, 2))
    poses = []
    for record in read_jsonl(path):
        if record["kind"] == "field":
            field = PlantField(
                ids=np.asarray(record["ids"], dtype=int),
                positions=np.asarray(record["positions"], dtype=float).reshape(-1, 2),
                row_index=np.asarray(record["row_index"], dtype=int),
                missing_ids=np.asarray(record.get("missing_ids", []), dtype=int),
                missing_positions=np.asarray(record.get("missing_positions", []),
                                             dtype=float).reshape(-1, 2),
            )
        elif record["kind"] == "weeds":
            weeds = np.asarray(record["positions"], dtype=float).reshape(-1, 2)
        else:
            poses.append(PoseSample(t=record["t"], true=Pose2D(**record["true"]),
                                    reported=Pose2D(**record["reported"])))
    if field is None:
        raise ValueError(f"{path}: no field record")
    return GroundTruth(field=field, poses=poses, weeds=weeds)
