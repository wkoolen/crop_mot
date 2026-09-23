"""Typed configuration: YAML in, frozen dataclasses out. [B1-B4, reproducibility]

One config file plus one seed defines a run completely. The runner copies the config into
the run folder verbatim, so a result can always be traced back to the exact inputs that
produced it.

Two rules this module enforces:
  * Parsing is STRICT. An unknown key is an error, not a silently ignored typo. Writing
    `p_d: 0.85` instead of `p_D: 0.85` must fail loudly rather than quietly run with a
    default, because that failure mode is invisible in a plot.
  * The simulator's `sensor` block and the filter's `filter.assumed_sensor` block are
    SEPARATE types. They look similar on purpose - the filter's beliefs may deliberately
    differ from the truth - but they can never be confused for one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crop_mot.types import FieldOfView

# --------------------------------------------------------------------------------------
# B1: scenario (world, path, truth sensor)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowConfig:
    """One row of plants, parallel to the +y axis. [B1]

    Attributes:
        x: lateral position of the row in metres.
        y_start, y_end: extent of the row along +y in metres.
        spacing: nominal along-row plant spacing in metres.
    """

    x: float
    y_start: float
    y_end: float
    spacing: float


@dataclass(frozen=True)
class WorldConfig:
    """The static field of plants. [B1]

    Attributes:
        rows: the plant rows; the robot walks the lane between them.
        position_jitter_std: metres, standard deviation of planting irregularity applied to
            each nominal plant position. Drawn from the "field" substream, so it does not
            change when detector parameters change.
    """

    rows: tuple[RowConfig, ...]
    position_jitter_std: float


@dataclass(frozen=True)
class PathConfig:
    """The robot's walk, and how accurately its pose is reported. [B1]

    Attributes:
        kind: path generator to use; "straight_lane" is the only one in phase 1.
        x: lateral position of the lane in metres.
        y_start: where the walk begins along +y.
        heading: radians; for a straight lane up +y this is pi/2.
        speed: metres per second.
        n_scans: number of scans to generate.
        scan_period: seconds between scans; the runner derives dt from this.
        pose_known: if True the reported pose IS the true pose and the two noise parameters
            below are ignored. This is the assumption B2 and the A2 closed form rely on.
            EXTENSION SLOT: setting it False models gait-induced odometry error.
        yaw_wobble_std: radians, heading noise. Ignored while pose_known is True.
        xy_noise_std: metres, position noise. Ignored while pose_known is True.
    """

    kind: str
    x: float
    y_start: float
    heading: float
    speed: float
    n_scans: int
    scan_period: float
    pose_known: bool
    yaw_wobble_std: float
    xy_noise_std: float


@dataclass(frozen=True)
class DetectionConfig:
    """Which p_D profile to use, and its parameters. [B1/B3]

    `kind` selects the SensorModel implementation:
      * "constant"        -> ConstantPD, using p_D. This is what the A2 closed form is
                             derived under and the default for B2/B3.
      * "range_dependent" -> RangeDependentPD, using p_D_near / p_D_far / occlusion_factor.
                             Models the occluded plants in the scenario sketch.

    Both remain validatable in B3 because `ScanEvent` carries p_D per scan rather than the
    analytic reference storing a single constant.

    Attributes:
        kind: "constant" or "range_dependent".
        p_D: used when kind == "constant".
        p_D_near, p_D_far: used when kind == "range_dependent"; detection probability at
            min_range and at max_range respectively.
        occlusion_factor: used when kind == "range_dependent"; multiplier applied when a
            nearer plant in the same row shadows this one.
    """

    kind: str
    p_D: float | None = None
    p_D_near: float | None = None
    p_D_far: float | None = None
    occlusion_factor: float = 1.0


@dataclass(frozen=True)
class MeasurementConfig:
    """Which measurement model to build. [B1/B2]

    Attributes:
        kind: "linear_xy" is the only phase-1 model. A 3D "linear_xyz" would be a new file,
            not a change here, because models declare their own dim_x / dim_z.
        R: measurement noise covariance, shape (dim_z, dim_z), in m^2.
    """

    kind: str
    R: np.ndarray


@dataclass(frozen=True)
class SensorConfig:
    """The TRUTH detector used by the simulator. [B1]

    Distinct from AssumedSensorConfig, which is what the filter believes. Keeping them as
    two types means a model-mismatch experiment is a config edit, and means the filter can
    never accidentally be handed the true parameters.

    Attributes:
        fov: the wedge the camera sees.
        detection: p_D profile and parameters.
        lambda_FA: expected number of clutter detections per scan (Poisson mean).
        measurement: measurement model and noise.
    """

    fov: FieldOfView
    detection: DetectionConfig
    lambda_FA: float
    measurement: MeasurementConfig


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete B1 scenario: everything needed to produce detections.jsonl. [B1]

    Attributes:
        name: used in the run folder name.
        seed: the single integer that, with this config, defines the run.
        world: the plant field.
        path: the robot's walk.
        sensor: the truth detector.
    """

    name: str
    seed: int
    world: WorldConfig
    path: PathConfig
    sensor: SensorConfig


# --------------------------------------------------------------------------------------
# B2: filter
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionConfig:
    """Target dynamics. [B2]

    Attributes:
        kind: "static" is the only phase-1 model - plants do not move.
        q: process noise density. Normally 0.0; a small positive value only to stop the
            covariance collapsing numerically over a long run.
    """

    kind: str
    q: float = 0.0


@dataclass(frozen=True)
class AssumedSensorConfig:
    """What the FILTER believes about the detector. [B2/B4]

    Deliberately a separate type from SensorConfig even though the fields overlap. It has
    no `measurement` field because the filter's measurement model is configured directly
    under `filter.measurement`.

    Attributes:
        fov: the FOV the filter assumes, used to evaluate p_D and the clutter density.
        detection: the p_D profile the filter assumes.
        lambda_FA: the clutter rate the filter assumes.
    """

    fov: FieldOfView
    detection: DetectionConfig
    lambda_FA: float


@dataclass(frozen=True)
class BirthConfig:
    """Where new tracks come from. [B2/B4]

    Attributes:
        kind: "single_from_measurement" seeds one Bernoulli component from a chosen
            detection at a chosen scan. For the B2 phantom experiment that detection is a
            clutter return, so r should then decay.
        at_scan: scan index at which to seed. Only used by "single_from_measurement".
        r_b: birth existence probability r_b assigned to the new component.
        init_cov: initial covariance for the new component, shape (dim_x, dim_x).
    """

    kind: str
    at_scan: int
    r_b: float
    init_cov: np.ndarray


@dataclass(frozen=True)
class SurvivalConfig:
    """Target survival. [B2]

    Attributes:
        p_S: probability a target that exists at scan k still exists at k+1. 1.0 for
            plants, which are permanent - a real simplification worth stating in the thesis.
    """

    p_S: float = 1.0


@dataclass(frozen=True)
class GateConfig:
    """Measurement gating. [B2/B4]

    Attributes:
        chi2_prob: gate probability, e.g. 0.99. Converted to a chi-square threshold on the
            squared Mahalanobis distance given dim_z.
    """

    chi2_prob: float = 0.99


@dataclass(frozen=True)
class FilterConfig:
    """Everything needed to build one filter. [B2/B4]

    This is the single argument every entry in `crop_mot.filters.FILTERS` takes, which is
    what keeps adding a B4 filter to a new file plus one dict line.

    Attributes:
        kind: key into FILTERS, e.g. "bernoulli". Phase 2 adds "pda", "jpda", "gnn",
            "pmb", "pmbm".
        motion: target dynamics.
        measurement: the filter's measurement model.
        assumed_sensor: the filter's beliefs about p_D, lambda_FA and the FOV.
        birth: birth model.
        survival: survival model.
        gate: gating parameters.
    """

    kind: str
    motion: MotionConfig
    measurement: MeasurementConfig
    assumed_sensor: AssumedSensorConfig
    birth: BirthConfig
    survival: SurvivalConfig
    gate: GateConfig


# --------------------------------------------------------------------------------------
# B3: analysis
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloConfig:
    """Repeat-run settings for the empirical half of B3. [B3]

    Attributes:
        n_runs: how many seeds to average over. Deliberately small in phase 1 (tens);
            long trials belong to phase 2 on ROS 2.
    """

    n_runs: int = 50


@dataclass(frozen=True)
class AnalysisConfig:
    """What to compute and plot after a filter run. [B3]

    Attributes:
        b3_reference: key selecting the analytic reference, e.g. "bernoulli_existence" for
            the A2 closed form. None skips the cross-check.
        monte_carlo: repeat-run settings, or None to skip.
        plots: which figures to render into the run folder's plots/ directory.
    """

    b3_reference: str | None
    monte_carlo: MonteCarloConfig | None
    plots: tuple[str, ...]


@dataclass(frozen=True)
class RunConfig:
    """A complete B2/B3 run: a scenario, a filter, and what to analyse. [B2/B3]

    Attributes:
        name: used in the run folder name.
        seed: the run's seed. Note this may differ from scenario.seed when re-filtering a
            previously recorded scenario.
        scenario: the B1 scenario, loaded from the path in the `scenario:` key.
        filter_cfg: the filter to run. Maps from the YAML key `filter` (renamed here to
            avoid shadowing the builtin).
        analysis: post-run analysis settings.
    """

    name: str
    seed: int
    scenario: ScenarioConfig
    filter_cfg: FilterConfig
    analysis: AnalysisConfig


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_scenario_config(path: Path) -> ScenarioConfig:
    """Parse a B1 scenario YAML file into a ScenarioConfig.

    Serves: [B1] `python -m crop_mot simulate --config configs/b1_two_rows.yaml`.

    Args:
        path: path to a YAML file with top-level keys name, seed, world, path, sensor.

    Returns:
        The parsed, validated scenario.

    Raises:
        ValueError: on an unknown key, a missing required key, or an inconsistent
            combination (e.g. detection.kind == "constant" with no p_D).
    """
    raise NotImplementedError


def load_run_config(path: Path) -> RunConfig:
    """Parse a B2 run YAML file, following its `scenario:` key to load the B1 scenario.

    There is no config-merge or inheritance machinery: a run config POINTS AT a scenario
    file and is otherwise self-contained. That keeps "which world did this run use" a
    one-line answer.

    Serves: [B2/B3] `python -m crop_mot track --config configs/b2_bernoulli_phantom.yaml`.

    Args:
        path: path to a YAML file with top-level keys name, seed, scenario, filter, analysis.

    Returns:
        The parsed, validated run configuration, with the referenced scenario already loaded.

    Raises:
        ValueError: on an unknown key, a missing required key, or an unreadable scenario path.
        FileNotFoundError: if the referenced scenario file does not exist.
    """
    raise NotImplementedError
