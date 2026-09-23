"""Birth models: where new tracks come from. [B2/B4]"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.filters.base import BirthModel
from crop_mot.types import Scan


@dataclass(frozen=True)
class SingleFromMeasurement(BirthModel):
    """Seed exactly one component from one detection at one chosen scan. [B2]

    This is the birth model the B2 phantom experiment uses. At scan `at_scan` it picks a
    detection and creates a single Bernoulli component there with existence probability
    r_b. In the phantom scenario that detection is a CLUTTER return, so no plant is
    actually at that location, and the subsequent misdetections should drive r downward -
    which is the r-decay plot B2 has to produce.

    Deliberately simple: birth happens once, at a known scan, at a known measurement. That
    makes the resulting r trajectory hand-derivable, which is what B3 needs. A realistic
    measurement-driven birth model that spawns a component at every unassociated detection
    would produce a more interesting simulation and a much harder closed form.

    Note this class does not know which detection is clutter - it is given an index, and the
    CONFIG author chooses one that happens to be clutter by inspecting labels.jsonl. The
    birth model itself stays truth-blind.

    Attributes:
        at_scan: the scan index at which to seed.
        detection_index: which detection in that scan to seed from.
        r_b: birth existence probability.
        init_cov: shape (dim_x, dim_x), initial covariance for the new component. Should be
            noticeably larger than R, since a single measurement localises the target only
            up to the measurement noise.
    """

    at_scan: int
    detection_index: int
    r_b: float
    init_cov: np.ndarray

    def birth_components(self, scan: Scan) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """One component at the chosen detection, but only on the chosen scan.

        Args:
            scan: the current scan.

        Returns:
            A single (r_b, mean, init_cov) triple when scan.k == at_scan and that detection
            exists; an empty list on every other scan.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NoBirth(BirthModel):
    """Never create a track. [B2, testing]

    Used to isolate the prediction and update behaviour of a filter from its birth process -
    for example to check that r stays exactly 0 when nothing is ever born, which catches a
    whole class of indexing bugs.
    """

    def birth_components(self, scan: Scan) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Always empty.

        Args:
            scan: ignored.

        Returns:
            An empty list.
        """
        raise NotImplementedError
