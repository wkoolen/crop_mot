"""Plant positions: the static targets. And weed positions: static non-targets. [B1]

ASSUMPTION running through the whole thesis: plants are static point targets. They do not
move, do not appear and do not disappear, so p_S = 1 and the motion model is the identity.
Growth, wind and occlusion by leaves are out of scope.

Weeds are static too, but they are not targets: a detection of a weed is a false alarm that
happens to recur at the same place (decision D15).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from crop_mot.config import WeedsConfig, WorldConfig, nominal_positions


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
        missing_ids: the slots of the nominal grid that have no plant (roadmap step 8c,
            D23); a plant's id is its slot index, so ids skip these. Empty by default.
        missing_positions: shape (n_missing, dim_x), those slots' nominal positions:
            where the plant would have been.
    """

    ids: np.ndarray
    positions: np.ndarray
    row_index: np.ndarray
    missing_ids: np.ndarray = dataclass_field(default_factory=lambda: np.empty(0, dtype=int))
    missing_positions: np.ndarray = dataclass_field(default_factory=lambda: np.empty((0, 2)))


def generate_field(cfg: WorldConfig, rng_field: np.random.Generator,
                   rng_missing: np.random.Generator | None = None) -> PlantField:
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

        rng_missing: the "missing" substream; needed only when cfg.p_missing > 0.

    Returns:
        A PlantField with ids assigned 0..n_slots-1 in row-major order - one per slot of
        the nominal grid, so a plant's id is its slot - minus the empty slots.
    """
    # Plants at y_start, y_start + spacing, ... up to and including y_end: the grid the
    # filter's planting plan also uses (`nominal_positions`), here with jitter added.
    nominal, row_index = nominal_positions(cfg.rows)
    jitter = rng_field.normal(0.0, cfg.position_jitter_std, size=nominal.shape)
    ids = np.arange(len(nominal))
    if cfg.p_missing == 0.0:
        return PlantField(ids=ids, positions=nominal + jitter, row_index=row_index)
    # Missing plants (roadmap step 8c, D23): each slot is empty independently with
    # probability p_missing, from its own stream, after the jitter of every slot is drawn,
    # so the plants that are there sit exactly where they would without missing ones.
    if rng_missing is None:
        raise ValueError("p_missing > 0 needs the 'missing' substream")
    missing = rng_missing.random(len(nominal)) < cfg.p_missing
    present = ~missing
    return PlantField(ids=ids[present], positions=(nominal + jitter)[present],
                      row_index=row_index[present], missing_ids=ids[missing],
                      missing_positions=nominal[missing])


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
