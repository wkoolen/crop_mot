"""The Bernoulli filter: one target with an existence probability r. [B2]

The only filter implemented in phase 1, and the one B3 validates against the hand-derived
A2 recursion.

A Bernoulli density is a single target that may or may not exist:
    p(X) = 1 - r            if X is empty
    p(X) = r * N(x; m, P)   if X = {x}
so the filter carries a scalar r alongside a Gaussian. The prediction and update of r are
what B3 checks against the closed form; the Gaussian part is a standard Kalman filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crop_mot.association.gating import chi2_threshold, gate_measurements
from crop_mot.config import FilterConfig
from crop_mot.filters.base import BirthModel, SurvivalModel, TrackingFilter
from crop_mot.filters.birth import build_single_birth
from crop_mot.filters.collapse import COLLAPSE_STRATEGIES, CollapseStrategy
from crop_mot.filters.detection_prob import PD_EVALUATIONS, AtMean, PdEvaluation
from crop_mot.filters.kalman import (
    kf_predict,
    kf_update,
    log_predicted_likelihood,
    predicted_measurement,
)
from crop_mot.motion.models import MotionModel, StaticTarget
from crop_mot.sensor.models import MeasurementModel, build_measurement_model
from crop_mot.sensor.sensor_model import SensorModel, build_sensor_model
from crop_mot.types import Scan, TrackEstimate


@dataclass(frozen=True)
class BernoulliState:
    """The filter's state: existence probability plus a Gaussian. [B2]

    This is the type parameter S for the Bernoulli filter. The runner never inspects it -
    it only passes it back into predict/update and hands it to extract.

    Attributes:
        r: existence probability in [0, 1].
        mean: shape (dim_x,), the spatial density's mean, meaningful only when r > 0.
        cov: shape (dim_x, dim_x), the spatial density's covariance.
        track_id: identifier carried into the TrackEstimate so a trajectory can be
            reconstructed from the estimates log.
    """

    r: float
    mean: np.ndarray
    cov: np.ndarray
    track_id: int


@dataclass(frozen=True)
class BernoulliFilter(TrackingFilter[BernoulliState]):
    """Single-target Bernoulli filter over recorded detections. [B2]

    Satisfies the common TrackingFilter interface with N = 1: `extract` returns a list of
    length 0 or 1, so the runner treats it exactly like PMBM.

    All model knowledge is injected, and all of it comes from `filter.assumed_sensor` and
    the sibling config blocks - never from the simulator. The filter therefore cannot tell
    whether the detection it is tracking came from a plant or from clutter, which is the
    whole point of the phantom experiment.

    Attributes:
        motion: target dynamics, StaticTarget in phase 1.
        measurement: the measurement model used for g(z|x).
        sensor: the ASSUMED detection process, supplying p_D, lambda_FA and c(z).
        birth: the birth model.
        survival: p_S.
        collapse: how the post-update mixture of branch densities becomes one Gaussian
            again (see crop_mot.filters.collapse). Does not affect r at the current scan.
        gate_chi2: squared-Mahalanobis gate threshold, precomputed from the configured gate
            probability and dim_z.
        p_D_evaluation: how p_D enters the miss weight, the detection branches and the
            missed-branch moments (see crop_mot.filters.detection_prob); AtMean by default.
        name: "bernoulli".
    """

    motion: MotionModel
    measurement: MeasurementModel
    sensor: SensorModel
    birth: BirthModel
    survival: SurvivalModel
    collapse: CollapseStrategy
    gate_chi2: float
    p_D_evaluation: PdEvaluation = AtMean()
    name: str = "bernoulli"

    def initial_state(self) -> BernoulliState:
        """Prior with r = 0: nothing exists until the birth model creates it. [B2]

        Returns:
            A BernoulliState with r = 0.0 and a placeholder Gaussian.
        """
        dim_x = self.motion.dim_x
        return BernoulliState(r=0.0, mean=np.zeros(dim_x), cov=np.eye(dim_x), track_id=0)

    def predict(self, state: BernoulliState, dt: float) -> BernoulliState:
        """Time update. [B2]

        Two independent things happen:
          * existence: r <- p_S * r. With p_S = 1 for permanent plants this leaves r
            unchanged, so ALL decay in the B2 plot comes from the measurement update.
          * spatial: the Gaussian is propagated through the motion model, which is the
            identity for a static target.

        The existence line r <- p_S * r is the Bernoulli prediction without a birth term
        (births enter in `update`). A2 has no prediction section, so it is pinned by this
        docstring rather than cited.

        Args:
            state: current state.
            dt: time step in seconds.

        Returns:
            A new predicted state; `state` is not modified.
        """
        mean, cov = kf_predict(state.mean, state.cov, self.motion, dt)
        return BernoulliState(r=self.survival.p_S * state.r, mean=mean, cov=cov,
                              track_id=state.track_id)

    def update(self, state: BernoulliState, scan: Scan) -> BernoulliState:
        """Measurement update for one scan, including births. [B2]

        The structure the A2 derivation follows, and which B3 checks:
          * MISDETECTION branch: no gated detection is assigned to the target. The
            existence probability shrinks because a target that exists would have been
            detected with probability p_D. This branch is why r decays for a phantom.
          * DETECTION branch: each gated detection contributes a weight proportional to
            r * p_D * g(z|x), competing against the clutter explanation lambda_FA * c(z).
            The Gaussian is updated with the weighted measurement.
          * BIRTH: components from the birth model are merged in.

        p_D is evaluated through the ASSUMED sensor model at the predicted mean, using the
        scan's reported pose. It is 0 outside the FOV, which makes "out of view" behave
        differently from "in view but missed" - a distinction B3's ScanEvent records.

        The expressions. With predicted state (r, m, P), gated detections z_1..z_n, the
        clutter intensity kappa_j = lambda_FA * c(z_j) and ell_i = p_D * N(z_i; z_hat, S),
        the unnormalised weights of the hypotheses are the product form of the
        single-object likelihood of [A0 §Measurement model]:
            object absent:          w_absent = (1 - r) * prod_j kappa_j
            object missed:          w_0      = r (1 - p_D) * prod_j kappa_j        [A2 §2]
            z_i from the object:    w_i      = r ell_i * prod_{j != i} kappa_j     [A2 §3.1]
        and  r+ = (w_0 + sum_i w_i) / (w_absent + w_0 + sum_i w_i).
        With no gated detection this is A2 §2's boxed r+ = r(1 - p_D)/(1 - r p_D); with one
        it is A2 §3.1's r_marg. The n >= 2 case combines A0 §Measurement model with A2 §3.1
        and is not yet its own derived section (A2 §7 lists PDA as open). The product form
        needs no division by c(z); with a matched FOV every reported z has c(z) > 0.

        Deliberate simplifications, each standing in for a derived form:
          * p_D(m) at the predicted mean stands in for p_D_bar = integral p_D(x) p(x) dx
            [A2 §2.1]; exact when p_D is constant over the support of p(x).
          * the missed branch keeps the predicted Gaussian unchanged, which is exact only
            for constant p_D [A2 §2]; with range-dependent p_D a miss also reshapes p(x)
            [A2 §2.2], which one Gaussian cannot carry.
          * each detection branch is a plain Kalman update, the constant-p_D form of
            p+ proportional to p_D g p [A2 §3]; for non-constant p_D see A2 §7.
          * the branches are collapsed to one Gaussian by the injected CollapseStrategy
            instead of kept as the [A2 §3.1] mixture.
          * gating: detections outside the gate are treated as clutter, contributing
            exactly zero rather than a very small weight.

        Birth (decision D3): applied AFTER the update above, only into an empty Bernoulli
        (r == 0). The component is (r_b, z, init_cov) and its seeding z is not reused this
        scan, so the reported r at the birth scan is exactly r_b.

        Args:
            state: the predicted state.
            scan: this scan's detections and reported pose.

        Returns:
            A new posterior state; `state` is not modified.

        Raises:
            ValueError: if a birth arrives while a component already exists, or with more
                than one component (no derivation covers merging them); or if every
                hypothesis has zero weight, i.e. the scan is impossible under the assumed
                model.
        """
        posterior = state
        if state.r > 0.0:
            posterior = self._update_existing(state, scan)

        births = self.birth.birth_components(scan)
        if not births:
            return posterior
        if len(births) > 1 or posterior.r > 0.0:
            raise ValueError(
                "the Bernoulli filter holds one component: a birth is only accepted into an "
                f"empty filter (scan {scan.k}: r = {posterior.r}, {len(births)} born)"
            )
        r_b, mean, cov = births[0]
        return BernoulliState(r=float(r_b), mean=mean, cov=cov, track_id=posterior.track_id)

    def _update_existing(self, state: BernoulliState, scan: Scan) -> BernoulliState:
        """The misdetection and detection branches for a component with r > 0. [B2]

        See `update` for the expressions and their citations.
        """
        r, m, P = state.r, state.mean, state.cov
        pose = scan.pose

        # The p_D of the miss weight, standing in for p_D_bar [A2 §2.1]; with the default
        # AtMean the plug-in p_D(m), exactly 0 outside the assumed FOV (decision D27).
        p_D = self.p_D_evaluation.miss_p_D(self.sensor, m, P, pose)
        lambda_FA = self.sensor.lambda_FA(pose)

        z_hat, S = predicted_measurement(m, P, self.measurement, pose)
        Z = np.array([d.z for d in scan.detections]).reshape(-1, self.measurement.dim_z)
        gated = gate_measurements(Z, z_hat, S, self.gate_chi2)

        # Clutter intensity lambda_FA * c(z) at each gated detection [A0 §Measurement model].
        kappa = [lambda_FA * self.sensor.clutter_density(Z[i], pose) for i in gated]
        # ell_i = p_D * N(z_i; z_hat, S): "this component produced z_i" [A2 §3.1].
        ell = [self.p_D_evaluation.detection_p_D(self.sensor, m, P, pose, Z[i])
               * np.exp(log_predicted_likelihood(Z[i], z_hat, S)) for i in gated]

        prod_kappa = float(np.prod(kappa))  # 1.0 when nothing is gated
        w_absent = (1.0 - r) * prod_kappa
        w_missed = r * (1.0 - p_D) * prod_kappa  # [A2 §2]
        w_detected = []
        for i in range(len(gated)):
            others = kappa[:i] + kappa[i + 1:]
            w_detected.append(r * ell[i] * float(np.prod(others)))  # [A2 §3.1]

        w_exists = w_missed + sum(w_detected)
        total = w_absent + w_exists
        if total == 0.0:
            raise ValueError(
                f"scan {scan.k}: every hypothesis has zero weight - the detections are "
                "impossible under the assumed sensor model (e.g. an assumed FOV smaller "
                "than the detector's)"
            )
        r_post = w_exists / total

        if w_exists == 0.0:
            return BernoulliState(r=r_post, mean=m.copy(), cov=P.copy(),
                                  track_id=state.track_id)

        # Branch densities given existence, weights normalised [A3 §Normalizing the mixture].
        m_missed, P_missed = self.p_D_evaluation.missed_moments(self.sensor, m, P, pose)
        branches = [(w_missed / w_exists, m_missed, P_missed)]
        for i, w in zip(gated, w_detected):
            m_i, P_i = kf_update(m, P, Z[i], self.measurement, pose)
            branches.append((w / w_exists, m_i, P_i))
        mean, cov = self.collapse.collapse(branches)

        return BernoulliState(r=r_post, mean=mean, cov=cov, track_id=state.track_id)

    def extract(self, state: BernoulliState) -> list[TrackEstimate]:
        """Report the track if it is probably there. [B2]

        For a Bernoulli filter this is a thresholding decision on r. The threshold is a
        REPORTING choice, not part of the Bayes recursion - B3 compares the raw r
        trajectory, not the thresholded one, so this must not alter r.

        The threshold is r > 0 (decision D7): the track is reported whenever a component
        exists. A phantom born at r_b = 0.08 must appear in the estimates log for the B2
        plot and the B3 comparison; a confirmation threshold, if wanted, is applied in
        analysis.

        Args:
            state: the current state.

        Returns:
            A one-element list when the track is reported, empty otherwise.
        """
        if state.r <= 0.0:
            return []
        return [TrackEstimate(track_id=state.track_id, r=float(state.r), mean=state.mean.copy(),
                              cov=state.cov.copy())]


def build_bernoulli(cfg: FilterConfig) -> BernoulliFilter:
    """Construct a BernoulliFilter from its config block. [B2]

    This is the builder registered in `crop_mot.filters.FILTERS`. Every B4 filter gets an
    equivalent function with the same signature, which is what keeps the runner free of
    filter-specific construction logic.

    Args:
        cfg: the `filter:` block of a run config.

    Returns:
        A ready-to-run BernoulliFilter with all models injected.

    Raises:
        ValueError: if cfg.kind is not "bernoulli", or a referenced model kind is unknown,
            or pruning is configured (it lives in the bank filter; a single Bernoulli would
            have nothing to report after deleting its one component).
        NotImplementedError: if the p_D evaluation is one of the stubbed options B to D.
    """
    if cfg.kind != "bernoulli":
        raise ValueError(f"build_bernoulli got filter kind {cfg.kind!r}")
    if cfg.motion.kind != "static":
        raise ValueError(f"unknown motion kind {cfg.motion.kind!r}; phase 1 has only 'static'")
    if cfg.birth.kind not in ("single_from_measurement", "injected"):
        raise ValueError(f"unknown birth kind {cfg.birth.kind!r}")
    if cfg.prune.r_min > 0.0:
        raise ValueError("filter.prune is implemented by 'bernoulli_bank', not 'bernoulli'")
    if cfg.collapse not in COLLAPSE_STRATEGIES:
        raise ValueError(f"unknown collapse {cfg.collapse!r}; "
                         f"available: {sorted(COLLAPSE_STRATEGIES)}")
    if cfg.p_D_evaluation not in PD_EVALUATIONS:
        raise ValueError(f"unknown p_D_evaluation {cfg.p_D_evaluation!r}; "
                         f"available: {sorted(PD_EVALUATIONS)}")

    measurement = build_measurement_model(cfg.measurement)
    assumed = cfg.assumed_sensor
    return BernoulliFilter(
        motion=StaticTarget(q=cfg.motion.q, dim_x=measurement.dim_x),
        measurement=measurement,
        sensor=build_sensor_model(assumed.fov, assumed.detection, assumed.lambda_FA, measurement),
        birth=build_single_birth(cfg.birth),
        survival=SurvivalModel(p_S=cfg.survival.p_S),
        collapse=COLLAPSE_STRATEGIES[cfg.collapse](),
        gate_chi2=chi2_threshold(cfg.gate.chi2_prob, measurement.dim_z),
        p_D_evaluation=PD_EVALUATIONS[cfg.p_D_evaluation](),
    )
