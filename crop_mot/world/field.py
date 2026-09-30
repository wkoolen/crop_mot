"""Plant positions: the static targets. And weed positions: static non-targets. [B1]

ASSUMPTION running through the whole thesis: plants are static point targets. They do not
move, do not appear and do not disappear, so p_S = 1 and the motion model is the identity.
Growth, wind and occlusion by leaves are out of scope.

Weeds are static too, but they are not targets: a detection of a weed is a false alarm that
happens to recur at the same place (decision D15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.config import WeedsConfig, WorldConfig


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
    nominal = []
    row_index = []
    for i, row in enumerate(cfg.rows):
        # Plants at y_start, y_start + spacing, ... up to and including y_end. The small
        # tolerance keeps a plant exactly at y_end despite floating-point division.
        n_in_row = int(np.floor((row.y_end - row.y_start) / row.spacing + 1e-9)) + 1
        for j in range(n_in_row):
            nominal.append([row.x, row.y_start + j * row.spacing])
            row_index.append(i)

    nominal = np.array(nominal, dtype=float)
    jitter = rng_field.normal(0.0, cfg.position_jitter_std, size=nominal.shape)
    return PlantField(
        ids=np.arange(len(nominal)),
        positions=nominal + jitter,
        row_index=np.array(row_index, dtype=int),
    )


def generate_weeds(cfg: WeedsConfig | None, rng_weeds: np.random.Generator) -> np.ndarray:
    """Place the weeds: static non-plant objects the detector can mistake for plants. [B1]

    A homogeneous Poisson point process over the configured rectangle: the count is
    Poisson(density * area) and each weed is uniform over the rectangle (decision D15).
    Nothing keeps a weed away from a plant or the lane, since real weeds grow anywhere;
    whether a given weed can be told apart from the plants is left to the tracking.

    Draws only from the "weeds" substream, so the plants are identical with and without
    weeds for a given seed.

    Serves: [B1] the persistent false targets of a scenario with weeds.

    Args:
        cfg: the `world.weeds` block, or None for a field without weeds.
        rng_weeds: the "weeds" substream generator.

    Returns:
        Shape (n_weeds, 2) world-frame positions in metres; shape (0, 2) when cfg is None.
        A weed's id is its row index here.
    """
    if cfg is None:
        return np.empty((0, 2))
    region = cfg.region
    n = rng_weeds.poisson(cfg.density * region.area())
    x = rng_weeds.uniform(region.x_min, region.x_max, size=n)
    y = rng_weeds.uniform(region.y_min, region.y_max, size=n)
    return np.column_stack([x, y])
