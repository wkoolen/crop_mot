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
from typing import Any

import numpy as np
import yaml

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
class RegionConfig:
    """An axis-aligned rectangle in the world frame, in metres. [B1]"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def area(self) -> float:
        """Area of the rectangle in m^2."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass(frozen=True)
class WeedsConfig:
    """Weeds: static non-plant objects the detector sometimes reports as plants. [B1]

    PERSISTENT FALSE TARGETS (decision D15). Unlike the Poisson clutter, which is drawn
    fresh every scan, a weed stays where it is, so the detector can report it again at the
    same place whenever it is in view. How often it does is set by the truth sensor's
    `weed_detection` block.

    ASSUMPTION: weeds form a homogeneous Poisson point process over `region`: the count is
    Poisson(density * region area) and each position is uniform over the rectangle. Drawn
    from the "weeds" substream, so adding weeds moves no plant and changes no plant
    detection or clutter return.

    Attributes:
        density: expected weeds per m^2 of the region.
        region: the rectangle weeds are placed in.
    """

    density: float
    region: RegionConfig


@dataclass(frozen=True)
class WorldConfig:
    """The static field of plants, and optionally weeds. [B1]

    Attributes:
        rows: the plant rows; the robot walks the lane between them.
        position_jitter_std: metres, standard deviation of planting irregularity applied to
            each nominal plant position. Drawn from the "field" substream, so it does not
            change when detector parameters change.
        weeds: the weeds, or None for a field without any. Optional in the YAML.
    """

    rows: tuple[RowConfig, ...]
    position_jitter_std: float
    weeds: WeedsConfig | None = None


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
class MultiplicityConfig:
    """How many detections one plant can produce in one scan. [B1, extension slot]

    `kind` selects the generative model in `crop_mot.sensor.detector.sample_scan`:
      * "single"    -> at most one detection per plant per scan: the A0 point-target
                       model every phase-1 filter and the A2 closed form assume. Default.
      * "duplicate" -> a detector artefact: after the ordinary p_D detection, extra hits
                       are added near it, each with probability p_split, up to max_extra,
                       at z_primary + N(0, spread_std^2 I).
      * "poisson"   -> the extended-object (EOT) model: each visible plant produces
                       Poisson(gamma) detections at x + N(0, extent_std^2 I) + v. The
                       detection probability becomes 1 - exp(-gamma); detection.p_D is
                       not used.

    TRUTH ONLY. The filter's assumed sensor has no multiplicity, so any kind other than
    "single" is a model-mismatch experiment for the Bernoulli filter. An EOT filter would
    add an assumed counterpart.

    Attributes:
        kind: "single", "duplicate" or "poisson".
        p_split: "duplicate" only; probability of each further extra hit, in [0, 1).
        max_extra: "duplicate" only; cap on extra hits per detected plant.
        spread_std: "duplicate" only; metres, spread of an extra hit around the primary z.
        gamma: "poisson" only; expected detections per visible plant per scan.
        extent_std: "poisson" only; metres, isotropic extent of the plant.
    """

    kind: str = "single"
    p_split: float | None = None
    max_extra: int = 3
    spread_std: float | None = None
    gamma: float | None = None
    extent_std: float | None = None


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
        multiplicity: detections per plant per scan. Optional in the YAML, default "single".
        weed_detection: how often a visible weed is reported as a plant, in the same format
            as `detection`. Required when the world has weeds and refused otherwise
            (decision D15). The filter's assumed sensor has no counterpart, so a scenario
            with weeds is a model-mismatch experiment.
    """

    fov: FieldOfView
    detection: DetectionConfig
    lambda_FA: float
    measurement: MeasurementConfig
    multiplicity: MultiplicityConfig = MultiplicityConfig()
    weed_detection: DetectionConfig | None = None


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
            clutter return, so r should then decay. "from_measurements" seeds one
            component per entry of `seeds`, for the `bernoulli_bank` filter (decision D12).
            "injected" places one component at `position` at scan `at_scan`, with no
            detection behind it: the controlled phantom of roadmap step 4a (D28), an
            experiment setting rather than a model of where objects come from.
        at_scan: scan index at which to seed. For "from_measurements", the first seed's.
        detection_index: which detection of scan `at_scan` to seed from. Chosen by the
            config author (by inspecting labels.jsonl), so the birth model stays truth-blind.
            For "from_measurements", the first seed's; -1 for "injected", which uses none.
        r_b: birth existence probability r_b assigned to the new component. PHASE-1
            STAND-IN: a configured constant in place of the measurement-driven
            r_b = e / (e + lambda_FA c(z)), e = integral lambda_u(x) p_D(x) g(z|x) dx,
            derived in [A2 §4].
        init_cov: initial covariance for the new component, shape (dim_x, dim_x).
        seeds: (at_scan, detection_index) pairs, one per component, for
            "from_measurements"; the pair's position in the list is the track id. Empty for
            "single_from_measurement". Chosen by the config author, e.g. with
            `python3 -m crop_mot candidates`.
        position: shape (dim_x,), where "injected" places its component; None otherwise.
            Configured, not read from truth, so the filter stays truth-blind.
    """

    kind: str
    at_scan: int
    detection_index: int
    r_b: float
    init_cov: np.ndarray
    seeds: tuple[tuple[int, int], ...] = ()
    position: np.ndarray | None = None


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
class PruneConfig:
    """Deleting a component whose existence probability has become negligible. [B2]

    A2 §5's deletion: a phantom that is looked at and missed scan after scan ratchets r
    down until it falls below a threshold and is removed. Part of the filter state, unlike
    the reporting threshold in `extract` (decision D7), which never alters r.

    Attributes:
        r_min: a component whose predicted r is below r_min is deleted (decision D13).
            0.0, the default, never deletes - which keeps every existing run and the B3
            comparison unchanged.
    """

    r_min: float = 0.0


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
        collapse: how the post-update mixture of branch densities is collapsed back to one
            Gaussian; a key into `crop_mot.filters.collapse.COLLAPSE_STRATEGIES`. Optional in
            the YAML, default "best_branch".
        prune: component deletion. Optional in the YAML, default off (r_min = 0).
        p_D_evaluation: how p_D is evaluated for a Gaussian track; a key into
            `crop_mot.filters.detection_prob.PD_EVALUATIONS`. Optional in the YAML, default
            "at_mean" (roadmap step 3b, decision D27).
    """

    kind: str
    motion: MotionConfig
    measurement: MeasurementConfig
    assumed_sensor: AssumedSensorConfig
    birth: BirthConfig
    survival: SurvivalConfig
    gate: GateConfig
    collapse: str = "best_branch"
    prune: PruneConfig = PruneConfig()
    p_D_evaluation: str = "at_mean"


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
    return _parse_scenario(_read_yaml(path), where=str(path))


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

    A relative `scenario:` path is resolved against the current working directory, i.e. the
    repo root from which `python3 -m crop_mot` is run.
    """
    where = str(path)
    raw = _read_yaml(path)
    _check_keys(raw, where, required={"name", "seed", "scenario", "filter", "analysis"})

    scenario_path = Path(_as_str(raw["scenario"], f"{where}: scenario"))
    if not scenario_path.is_absolute():
        scenario_path = Path.cwd() / scenario_path
    if not scenario_path.is_file():
        raise FileNotFoundError(
            f"{where}: scenario file {raw['scenario']!r} not found (relative paths are "
            f"resolved against the working directory {Path.cwd()})"
        )

    return RunConfig(
        name=_as_str(raw["name"], f"{where}: name"),
        seed=_as_int(raw["seed"], f"{where}: seed"),
        scenario=load_scenario_config(scenario_path),
        filter_cfg=_parse_filter(raw["filter"], f"{where}: filter"),
        analysis=_parse_analysis(raw["analysis"], f"{where}: analysis"),
    )


# --------------------------------------------------------------------------------------
# Parsing helpers. Every block is checked against an explicit set of allowed keys, so a
# typo is an error rather than a silently ignored default.
# --------------------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return raw


def _check_keys(
    block: Any, where: str, required: set[str], optional: set[str] = frozenset()
) -> None:
    """Raise ValueError unless `block` is a mapping with exactly the allowed keys."""
    if not isinstance(block, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(block).__name__}")
    unknown = set(block) - required - optional
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(required | optional)}"
        )
    missing = required - set(block)
    if missing:
        raise ValueError(f"{where}: missing required key(s) {sorted(missing)}")


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: expected an integer, got {value!r}")
    return value


def _as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: expected true or false, got {value!r}")
    return value


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a string, got {value!r}")
    return value


def _as_matrix(value: Any, where: str) -> np.ndarray:
    """A square matrix of numbers, e.g. R or init_cov."""
    matrix = np.array(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{where}: expected a square matrix, got shape {matrix.shape}")
    return matrix


def _parse_fov(raw: Any, where: str) -> FieldOfView:
    _check_keys(raw, where, required={"min_range", "max_range", "half_angle"})
    fov = FieldOfView(
        min_range=_as_float(raw["min_range"], f"{where}.min_range"),
        max_range=_as_float(raw["max_range"], f"{where}.max_range"),
        half_angle=_as_float(raw["half_angle"], f"{where}.half_angle"),
    )
    if not 0.0 <= fov.min_range < fov.max_range or not 0.0 < fov.half_angle <= np.pi:
        raise ValueError(f"{where}: need 0 <= min_range < max_range and 0 < half_angle <= pi")
    return fov


def _parse_detection(raw: Any, where: str) -> DetectionConfig:
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError(f"{where}: expected a mapping with a 'kind' key")
    kind = raw["kind"]
    if kind == "constant":
        _check_keys(raw, where, required={"kind", "p_D"})
        return DetectionConfig(kind=kind, p_D=_as_float(raw["p_D"], f"{where}.p_D"))
    if kind == "range_dependent":
        _check_keys(raw, where, required={"kind", "p_D_near", "p_D_far"},
                    optional={"occlusion_factor"})
        return DetectionConfig(
            kind=kind,
            p_D_near=_as_float(raw["p_D_near"], f"{where}.p_D_near"),
            p_D_far=_as_float(raw["p_D_far"], f"{where}.p_D_far"),
            occlusion_factor=_as_float(raw.get("occlusion_factor", 1.0),
                                       f"{where}.occlusion_factor"),
        )
    raise ValueError(f"{where}.kind: unknown kind {kind!r}; use 'constant' or 'range_dependent'")


def _parse_measurement(raw: Any, where: str) -> MeasurementConfig:
    _check_keys(raw, where, required={"kind", "R"})
    return MeasurementConfig(kind=_as_str(raw["kind"], f"{where}.kind"),
                             R=_as_matrix(raw["R"], f"{where}.R"))


def _parse_region(raw: Any, where: str) -> RegionConfig:
    _check_keys(raw, where, required={"x_min", "x_max", "y_min", "y_max"})
    region = RegionConfig(**{key: _as_float(raw[key], f"{where}.{key}") for key in raw})
    if not (region.x_min < region.x_max and region.y_min < region.y_max):
        raise ValueError(f"{where}: need x_min < x_max and y_min < y_max")
    return region


def _parse_weeds(raw: Any, where: str) -> WeedsConfig:
    _check_keys(raw, where, required={"density", "region"})
    density = _as_float(raw["density"], f"{where}.density")
    if density < 0.0:
        raise ValueError(f"{where}.density: expected >= 0, got {density}")
    return WeedsConfig(density=density, region=_parse_region(raw["region"], f"{where}.region"))


def _parse_world(raw: Any, where: str) -> WorldConfig:
    _check_keys(raw, where, required={"rows", "position_jitter_std"}, optional={"weeds"})
    if not isinstance(raw["rows"], list) or not raw["rows"]:
        raise ValueError(f"{where}.rows: expected a non-empty list")
    rows = []
    for i, row in enumerate(raw["rows"]):
        row_where = f"{where}.rows[{i}]"
        _check_keys(row, row_where, required={"x", "y_start", "y_end", "spacing"})
        rows.append(RowConfig(
            x=_as_float(row["x"], f"{row_where}.x"),
            y_start=_as_float(row["y_start"], f"{row_where}.y_start"),
            y_end=_as_float(row["y_end"], f"{row_where}.y_end"),
            spacing=_as_float(row["spacing"], f"{row_where}.spacing"),
        ))
    return WorldConfig(rows=tuple(rows),
                       position_jitter_std=_as_float(raw["position_jitter_std"],
                                                     f"{where}.position_jitter_std"),
                       weeds=(_parse_weeds(raw["weeds"], f"{where}.weeds")
                              if "weeds" in raw else None))


def _parse_path(raw: Any, where: str) -> PathConfig:
    keys = {"kind", "x", "y_start", "heading", "speed", "n_scans", "scan_period",
            "pose_known", "yaw_wobble_std", "xy_noise_std"}
    _check_keys(raw, where, required=keys)
    return PathConfig(
        kind=_as_str(raw["kind"], f"{where}.kind"),
        x=_as_float(raw["x"], f"{where}.x"),
        y_start=_as_float(raw["y_start"], f"{where}.y_start"),
        heading=_as_float(raw["heading"], f"{where}.heading"),
        speed=_as_float(raw["speed"], f"{where}.speed"),
        n_scans=_as_int(raw["n_scans"], f"{where}.n_scans"),
        scan_period=_as_float(raw["scan_period"], f"{where}.scan_period"),
        pose_known=_as_bool(raw["pose_known"], f"{where}.pose_known"),
        yaw_wobble_std=_as_float(raw["yaw_wobble_std"], f"{where}.yaw_wobble_std"),
        xy_noise_std=_as_float(raw["xy_noise_std"], f"{where}.xy_noise_std"),
    )


def _parse_multiplicity(raw: Any, where: str) -> MultiplicityConfig:
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError(f"{where}: expected a mapping with a 'kind' key")
    kind = raw["kind"]
    if kind == "single":
        _check_keys(raw, where, required={"kind"})
        return MultiplicityConfig()
    if kind == "duplicate":
        _check_keys(raw, where, required={"kind", "p_split", "spread_std"},
                    optional={"max_extra"})
        p_split = _as_float(raw["p_split"], f"{where}.p_split")
        max_extra = _as_int(raw.get("max_extra", 3), f"{where}.max_extra")
        spread_std = _as_float(raw["spread_std"], f"{where}.spread_std")
        if not 0.0 <= p_split < 1.0 or max_extra < 0 or spread_std < 0.0:
            raise ValueError(f"{where}: need 0 <= p_split < 1, max_extra >= 0, spread_std >= 0")
        return MultiplicityConfig(kind=kind, p_split=p_split, max_extra=max_extra,
                                  spread_std=spread_std)
    if kind == "poisson":
        _check_keys(raw, where, required={"kind", "gamma", "extent_std"})
        gamma = _as_float(raw["gamma"], f"{where}.gamma")
        extent_std = _as_float(raw["extent_std"], f"{where}.extent_std")
        if gamma < 0.0 or extent_std < 0.0:
            raise ValueError(f"{where}: need gamma >= 0 and extent_std >= 0")
        return MultiplicityConfig(kind=kind, gamma=gamma, extent_std=extent_std)
    raise ValueError(
        f"{where}.kind: unknown kind {kind!r}; use 'single', 'duplicate' or 'poisson'"
    )


def _parse_sensor(raw: Any, where: str) -> SensorConfig:
    _check_keys(raw, where, required={"fov", "detection", "lambda_FA", "measurement"},
                optional={"multiplicity", "weed_detection"})
    detection = _parse_detection(raw["detection"], f"{where}.detection")
    multiplicity = (_parse_multiplicity(raw["multiplicity"], f"{where}.multiplicity")
                    if "multiplicity" in raw else MultiplicityConfig())
    if multiplicity.kind == "poisson" and detection.kind != "constant":
        raise ValueError(
            f"{where}: multiplicity.kind 'poisson' sets the detection probability through "
            f"gamma, so detection.kind must be 'constant' (its p_D is then unused)"
        )
    return SensorConfig(
        fov=_parse_fov(raw["fov"], f"{where}.fov"),
        detection=detection,
        lambda_FA=_as_float(raw["lambda_FA"], f"{where}.lambda_FA"),
        measurement=_parse_measurement(raw["measurement"], f"{where}.measurement"),
        multiplicity=multiplicity,
        weed_detection=(_parse_detection(raw["weed_detection"], f"{where}.weed_detection")
                        if "weed_detection" in raw else None),
    )


def _parse_scenario(raw: Any, where: str) -> ScenarioConfig:
    _check_keys(raw, where, required={"name", "seed", "world", "path", "sensor"})
    world = _parse_world(raw["world"], f"{where}: world")
    sensor = _parse_sensor(raw["sensor"], f"{where}: sensor")
    # Weeds are split over two blocks - where they are (world) and how the detector sees
    # them (sensor) - so a half-configured pair is caught here rather than half-ignored.
    if (world.weeds is None) != (sensor.weed_detection is None):
        raise ValueError(
            f"{where}: world.weeds and sensor.weed_detection go together; give both for a "
            f"field with weeds, or neither for one without"
        )
    return ScenarioConfig(
        name=_as_str(raw["name"], f"{where}: name"),
        seed=_as_int(raw["seed"], f"{where}: seed"),
        world=world,
        path=_parse_path(raw["path"], f"{where}: path"),
        sensor=sensor,
    )


def _parse_filter(raw: Any, where: str) -> FilterConfig:
    _check_keys(raw, where,
                required={"kind", "motion", "measurement", "assumed_sensor", "birth",
                          "survival", "gate"},
                optional={"collapse", "prune", "p_D_evaluation"})

    motion = raw["motion"]
    _check_keys(motion, f"{where}.motion", required={"kind"}, optional={"q"})

    assumed = raw["assumed_sensor"]
    _check_keys(assumed, f"{where}.assumed_sensor",
                required={"fov", "detection", "lambda_FA"})

    birth = raw["birth"]
    if isinstance(birth, dict) and birth.get("kind") == "from_measurements":
        _check_keys(birth, f"{where}.birth", required={"kind", "seeds", "r_b", "init_cov"})
        seeds = _parse_seeds(birth["seeds"], f"{where}.birth.seeds")
        at_scan, detection_index = seeds[0]
        position = None
    elif isinstance(birth, dict) and birth.get("kind") == "injected":
        _check_keys(birth, f"{where}.birth",
                    required={"kind", "at_scan", "position", "r_b", "init_cov"})
        seeds = ()
        at_scan = _as_int(birth["at_scan"], f"{where}.birth.at_scan")
        detection_index = -1
        if not isinstance(birth["position"], list):
            raise ValueError(f"{where}.birth.position: expected a list of coordinates")
        position = np.array([_as_float(v, f"{where}.birth.position")
                             for v in birth["position"]])
    else:
        _check_keys(birth, f"{where}.birth",
                    required={"kind", "at_scan", "detection_index", "r_b", "init_cov"})
        seeds = ()
        at_scan = _as_int(birth["at_scan"], f"{where}.birth.at_scan")
        detection_index = _as_int(birth["detection_index"], f"{where}.birth.detection_index")
        position = None

    prune = raw.get("prune", {})
    _check_keys(prune, f"{where}.prune", required=set(), optional={"r_min"})
    r_min = _as_float(prune.get("r_min", 0.0), f"{where}.prune.r_min")
    if not 0.0 <= r_min < 1.0:
        raise ValueError(f"{where}.prune.r_min: expected 0 <= r_min < 1, got {r_min}")

    survival = raw["survival"]
    _check_keys(survival, f"{where}.survival", required=set(), optional={"p_S"})

    gate = raw["gate"]
    _check_keys(gate, f"{where}.gate", required=set(), optional={"chi2_prob"})

    return FilterConfig(
        kind=_as_str(raw["kind"], f"{where}.kind"),
        motion=MotionConfig(kind=_as_str(motion["kind"], f"{where}.motion.kind"),
                            q=_as_float(motion.get("q", 0.0), f"{where}.motion.q")),
        measurement=_parse_measurement(raw["measurement"], f"{where}.measurement"),
        assumed_sensor=AssumedSensorConfig(
            fov=_parse_fov(assumed["fov"], f"{where}.assumed_sensor.fov"),
            detection=_parse_detection(assumed["detection"],
                                       f"{where}.assumed_sensor.detection"),
            lambda_FA=_as_float(assumed["lambda_FA"], f"{where}.assumed_sensor.lambda_FA"),
        ),
        birth=BirthConfig(
            kind=_as_str(birth["kind"], f"{where}.birth.kind"),
            at_scan=at_scan,
            detection_index=detection_index,
            r_b=_as_float(birth["r_b"], f"{where}.birth.r_b"),
            init_cov=_as_matrix(birth["init_cov"], f"{where}.birth.init_cov"),
            seeds=seeds,
            position=position,
        ),
        survival=SurvivalConfig(p_S=_as_float(survival.get("p_S", 1.0),
                                              f"{where}.survival.p_S")),
        gate=GateConfig(chi2_prob=_as_float(gate.get("chi2_prob", 0.99),
                                            f"{where}.gate.chi2_prob")),
        collapse=_as_str(raw.get("collapse", "best_branch"), f"{where}.collapse"),
        prune=PruneConfig(r_min=r_min),
        p_D_evaluation=_as_str(raw.get("p_D_evaluation", "at_mean"),
                               f"{where}.p_D_evaluation"),
    )


def _parse_seeds(raw: Any, where: str) -> tuple[tuple[int, int], ...]:
    """A non-empty list of {at_scan, detection_index} mappings, as (int, int) pairs."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: expected a non-empty list of "
                         "{at_scan, detection_index} mappings")
    seeds = []
    for i, seed in enumerate(raw):
        _check_keys(seed, f"{where}[{i}]", required={"at_scan", "detection_index"})
        seeds.append((_as_int(seed["at_scan"], f"{where}[{i}].at_scan"),
                      _as_int(seed["detection_index"], f"{where}[{i}].detection_index")))
    return tuple(seeds)


def _parse_analysis(raw: Any, where: str) -> AnalysisConfig:
    _check_keys(raw, where, required={"b3_reference", "monte_carlo", "plots"})

    b3_reference = raw["b3_reference"]
    if b3_reference is not None:
        b3_reference = _as_str(b3_reference, f"{where}.b3_reference")

    monte_carlo = raw["monte_carlo"]
    if monte_carlo is not None:
        _check_keys(monte_carlo, f"{where}.monte_carlo", required=set(), optional={"n_runs"})
        monte_carlo = MonteCarloConfig(n_runs=_as_int(monte_carlo.get("n_runs", 50),
                                                      f"{where}.monte_carlo.n_runs"))

    if not isinstance(raw["plots"], list):
        raise ValueError(f"{where}.plots: expected a list of plot names")
    plots = tuple(_as_str(name, f"{where}.plots") for name in raw["plots"])

    return AnalysisConfig(b3_reference=b3_reference, monte_carlo=monte_carlo, plots=plots)
