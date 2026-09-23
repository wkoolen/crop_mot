"""Plant positions: the static targets. [B1]

ASSUMPTION running through the whole thesis: plants are static point targets. They do not
move, do not appear and do not disappear, so p_S = 1 and the motion model is the identity.
Growth, wind and occlusion by leaves are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.config import WorldConfig


@dataclass(frozen=True)
class PlantField:
    """The true positions of every plant in the scenario. [B1]

    Attributes:
        ids: shape (n_plants,), stable integer ids used in `ScanLabels.origin` so a
            detection can be traced back to the plant that produced it.
        positions: shape (n_plants, dim_x); world-frame xy in metres in phase 1.
        row_index: shape (n_plants,), which configured row each plant belongs to. Kept
            because occlusion in RangeDependentPD is defined within a row, and because the
            scene plot colours by row.
    """

    ids: np.ndarray
    positions: np.ndarray
    row_index: np.ndarray


def generate_field(cfg: WorldConfig, rng_field: np.random.Generator) -> PlantField:
    """Place plants along each configured row, with planting irregularity.

    Each row is filled at `spacing` intervals from y_start to y_end, then every position is
    perturbed by N(0, position_jitter_std^2) independently in x and y. The jitter is what
    makes the data association problem non-trivial: on a perfectly regular grid, nearest
    neighbour would be unambiguous.

    Draws only from the "field" substream, so changing p_D or lambda_FA leaves the field
    identical for a given seed.

    Serves: [B1] the truth half of the simulator.

    Args:
        cfg: row specifications and jitter magnitude.
        rng_field: the "field" substream generator from `crop_mot.rng.substreams`.

    Returns:
        A PlantField with ids assigned 0..n_plants-1 in row-major order.
    """
    raise NotImplementedError
