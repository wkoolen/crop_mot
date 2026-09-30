"""A bank of independent Bernoulli filters, one per chosen seed, with pruning. [B2]

The B2 extension behind the hypothesis-history figure (decision D12). Where the single
Bernoulli filter follows one phantom, the bank follows several at once - each born from a
detection the config author picked, each carrying its own existence probability r - and
deletes any whose r has fallen below a threshold (A2 §5's deletion, decision D13).

Each component is advanced by the UNCHANGED `BernoulliFilter`, so every track follows the
A2 recursion exactly and B3's closed form applies to it track by track. The bank itself
adds only two things: births at the configured seeds, and pruning.

INDEPENDENCE APPROXIMATION: the components do not compete for detections. Each one runs
its own Bernoulli update against the full scan, so a detection inside two gates updates
both tracks. That is the difference from JPDA/PMB (B4), and it is harmless as long as the
seeds are more than a gate apart - which `analysis.candidates.phantom_candidates` checks
when the seeds are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from crop_mot.config import FilterConfig, PlanConfig, PruneConfig, nominal_positions
from crop_mot.filters.base import BirthModel, TrackingFilter
from crop_mot.filters.bernoulli import BernoulliFilter, BernoulliState, build_bernoulli
from crop_mot.filters.birth import SingleFromMeasurement, build_single_birth
from crop_mot.types import Scan, TrackEstimate


@dataclass(frozen=True)
class BernoulliBankState:
    """The bank's state: the live components. [B2]

    Opaque to the runner, like every filter state.

    Attributes:
        components: one BernoulliState per live track, in track-id order. A pruned track
            is removed from the tuple; its id is never reused.
    """

    components: tuple[BernoulliState, ...]


@dataclass(frozen=True)
class BernoulliBankFilter(TrackingFilter[BernoulliBankState]):
    """Independent Bernoulli filters over recorded detections, with pruning. [B2]

    Attributes:
        single: the Bernoulli filter that advances each component, built with NoBirth -
            births are the bank's job, so that each one gets its own component.
        births: one birth model per seed; its position in the tuple is the track id, so
            ids are fixed by the config and match between a pruned and an unpruned run.
        r_min: prune threshold; a component whose predicted r is below it is deleted.
            0.0 never deletes.
        initial: the components that exist before the first scan: the slots of a planting
            plan (roadmap step 8a, `plan_components`), empty without one.
        name: "bernoulli_bank".
    """

    single: BernoulliFilter
    births: tuple[BirthModel, ...]
    r_min: float
    initial: tuple[BernoulliState, ...] = ()
    name: str = "bernoulli_bank"

    def initial_state(self) -> BernoulliBankState:
        """The planting plan's slots, if any; otherwise nothing until a seed's scan. [B2/B4]"""
        return BernoulliBankState(components=self.initial)

    def predict(self, state: BernoulliBankState, dt: float) -> BernoulliBankState:
        """Time update of every component, then pruning. [B2]

        Each component is predicted by the single Bernoulli filter (r <- p_S r, Gaussian
        through the motion model). A component whose predicted r is below r_min is then
        deleted [A2 §5] (decision D13).

        Pruning here rather than after the update is what makes a deletion readable from
        the estimates log: the posterior r that fell below r_min is still reported at its
        scan, and the track is absent from the next scan on. `extract` never alters r
        (decision D7).

        Args:
            state: current state.
            dt: time step in seconds.

        Returns:
            A new predicted state; `state` is not modified.
        """
        predicted = []
        for component in state.components:
            component = self.single.predict(component, dt)
            if component.r >= self.r_min:
                predicted.append(component)
        return BernoulliBankState(components=tuple(predicted))

    def update(self, state: BernoulliBankState, scan: Scan) -> BernoulliBankState:
        """Measurement update of every component, then births. [B2]

        Every component is updated by the single Bernoulli filter against the whole scan
        (the independence approximation in the module docstring). Then each seed whose
        scan this is adds a new component (r_b, z, init_cov) - the D3 convention per
        component: the seeding z does not update its own new track this scan, so that
        track's reported r at its birth scan is exactly r_b.

        Args:
            state: the predicted state.
            scan: this scan's detections and reported pose.

        Returns:
            A new posterior state; `state` is not modified.
        """
        components = [self.single.update(component, scan) for component in state.components]
        for track_id, birth in enumerate(self.births):
            for r_b, mean, cov in birth.birth_components(scan):
                components.append(BernoulliState(r=float(r_b), mean=mean, cov=cov,
                                                 track_id=track_id))
        return BernoulliBankState(components=tuple(components))

    def extract(self, state: BernoulliBankState) -> list[TrackEstimate]:
        """Report every live component with r > 0 (decision D7). [B2]

        Args:
            state: the current state.

        Returns:
            One TrackEstimate per reported component, in track-id order of birth.
        """
        estimates = []
        for component in state.components:
            estimates.extend(self.single.extract(component))
        return estimates


def build_bernoulli_bank(cfg: FilterConfig) -> BernoulliBankFilter:
    """Construct a BernoulliBankFilter from its config block. [B2]

    The registered builder for "bernoulli_bank". Birth kind "from_measurements" seeds one
    track per entry of `birth.seeds`; "single_from_measurement" seeds one track from
    (at_scan, detection_index), which makes a one-seed bank the single Bernoulli filter;
    "injected" places the controlled phantom of roadmap step 4a (D28), which the bank can
    then prune. With `birth: null` and a `plan`, the bank starts from the plan's slots at
    r = 1 and has no births: the known-N map of roadmap step 8a.

    Args:
        cfg: the `filter:` block of a run config.

    Returns:
        A ready-to-run BernoulliBankFilter.

    Raises:
        ValueError: if cfg.kind is not "bernoulli_bank", a referenced model kind is
            unknown, or a plan comes with births.
    """
    if cfg.kind != "bernoulli_bank":
        raise ValueError(f"build_bernoulli_bank got filter kind {cfg.kind!r}")
    if cfg.plan is not None and cfg.birth is not None:
        raise ValueError("a planting plan with births waits on roadmap step 13 (their track "
                         "ids would collide); use birth: null with filter.plan")
    if cfg.birth is None:
        births = ()
    elif cfg.birth.kind == "from_measurements":
        births = tuple(
            SingleFromMeasurement(at_scan=at_scan, detection_index=detection_index,
                                  r_b=cfg.birth.r_b, init_cov=cfg.birth.init_cov)
            for at_scan, detection_index in cfg.birth.seeds
        )
    else:
        births = (build_single_birth(cfg.birth),)

    # The single filter is built by its own builder, so the bank cannot drift from it;
    # only its birth model is swapped out, and pruning stays with the bank.
    single = build_bernoulli(replace(cfg, kind="bernoulli", prune=PruneConfig(), birth=None,
                                     plan=None))
    initial = () if cfg.plan is None else plan_components(cfg.plan)
    return BernoulliBankFilter(single=single, births=births, r_min=cfg.prune.r_min,
                               initial=initial)


def plan_components(plan: PlanConfig) -> tuple[BernoulliState, ...]:
    """One component per planned slot at r_0: the known-N or bounded-N map. [B4, 8a/8c]

    Slot i is the i-th nominal position of the plan's rows, row by row, with covariance
    prior_std^2 I, and its track id is i. With r_0 = 1, N is known (step 8a): certainty is
    absorbing [A2 §5], so every slot stays at r = 1 and the problem is association and
    position only. With r_0 < 1, N is bounded by the plan (step 8c, D23): each slot is a
    Bernoulli that misses can drive down, which is how an empty slot should be found.

    Args:
        plan: the planting plan.

    Returns:
        The components, in slot order.
    """
    positions, _ = nominal_positions(plan.rows)
    cov = plan.prior_std**2 * np.eye(positions.shape[1])
    return tuple(BernoulliState(r=plan.r_0, mean=position.copy(), cov=cov.copy(),
                                track_id=i)
                 for i, position in enumerate(positions))
