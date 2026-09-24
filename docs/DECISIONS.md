# Implementation decisions (B1–B3)

The modelling and interface decisions taken with the author while implementing B1–B3 from
the skeleton. The table is current. The approved implementation plan it grew from is
appended unchanged below, as the record of what was agreed before coding started; where
the two differ, the table wins.

Each decision is also stated where it takes effect: in the docstring of the code that
implements it, and in the commit message that introduced it (branch
`b1-b3-implementation`).

## Current decisions

| # | Topic | Decision | Where it lands |
|---|---|---|---|
| D1 | Density collapse after an update | Keep the highest-weight branch (`KeepBestBranch`), behind a `CollapseStrategy` Protocol so other strategies (moment matching, ...) are one class plus one dict line. Selected by the optional `filter.collapse` key, default `best_branch`. Start simple, add complexity step by step. | `filters/collapse.py`, `FilterConfig.collapse` |
| D2 | Update with two or more gated detections | A2 derives at most one measurement (A2 §7 lists PDA as open). Implemented as A0 §Measurement model's single-object likelihood inside A2 §3.1's two-branch existence, cited as such and marked as not yet its own derived section. | `BernoulliFilter.update` |
| D3 | Birth convention | The measurement update runs first; the birth is applied after it, only into an empty Bernoulli (r = 0), as (r_b, z, init_cov). The seeding z is not reused that scan, so the reported r at the birth scan is exactly r_b. A birth into r > 0, or of more than one component, raises `ValueError`. The analytic reference must use the same convention. | `BernoulliFilter.update`, `SingleFromMeasurement` |
| D4 | Real detections outside the FOV (**revised**) | The detector reports only z inside the FOV: the FOV is the measurement space and c(z) a proper pdf on it. Consequence, deliberately not modelled by the filter: a plant within a few sigma of the FOV edge is detected with probability below p_D. *Superseded first choice:* no truncation (pure A0). Dropped after measuring that 7.1 % of real detections (41 of 60 scans, seed 42) fell outside the FOV, where c(z) = 0 pins a gating track at r = 1. | `sensor/detector.py::sample_scan` |
| D5 | Meaning of c(z) | c(z) = 1 / fov.area() inside the FOV, 0 outside: the spatial pdf (A0's f_c). The clutter intensity is lambda_FA * c(z) (A2's lambda_FA(z), A0's lambda_c(z)). | `SensorModel.clutter_density`, `FieldOfView.area` |
| D6 | Range-dependent p_D | Linear in range from p_D_near at min_range to p_D_far at max_range, exactly 0 outside the FOV. Occlusion is not implementable through `p_D(x, pose)`; `occlusion_factor != 1.0` raises `NotImplementedError` (extension slot). | `RangeDependentPD` |
| D7 | Reporting threshold in `extract` | Report the track whenever r > 0. A confirmation threshold, if wanted, belongs in analysis. | `BernoulliFilter.extract` |
| D8 | B2 phantom experiment | Seed 42, birth at scan 24 from detection 12: a clutter return at (1.94, 4.82), 1.17 m from the nearest plant. init_cov = 0.12 I (3 R). *Superseded first choice:* scan 0, detection 0, init_cov 0.25 I, which was 0.71 m from a plant and was captured by it (r to 1). | `configs/b2_bernoulli_phantom.yaml` |
| D9 | Phantom recapture | Kept and shown, not engineered away. r decays 0.08 to about 2e-4 over scans 24–31, then the phantom is recaptured (r > 0.5 at k = 37, then 1). Under keep-best-branch the branch weights compare r(1 - p_D) with r p_D N(z; z_hat, S) / (lambda_FA c(z)), so r cancels and the Gaussian follows any detection within about 1 m. Nothing prunes. *Alternatives considered:* init_cov 0.06 I (this seed's phantom then stays dead); pruning below a threshold (A2 §5's deletion). | `configs/b2_bernoulli_phantom.yaml` comment |

Standing decisions from the brief, implemented as stated:

- r_b is a configured constant, a stand-in for A2 §4's measurement-driven
  r_b = e / (e + lambda_FA c(z)). The derived version will be a new `BirthModel` class.
- p_S enters only as r <- p_S * r in `predict`, as the docstring pins it.

Smaller changes approved with the plan:

- `BirthConfig.detection_index`.
- Detections are sorted by z instead of shuffled.
- Shared builders `build_measurement_model` and `build_sensor_model`.
- The `analyse` entry point in `runner/analyse.py`, with `--plots`.
- An optional scan index `k` on `plot_scene`.

## Still open

- **`BernoulliExistenceReference.r_sequence`**: the TODO(human) docstring (the derivation).
  Before B3 can be finished, it has to settle:
  - how the detection branch is carried, since the filter uses
    sum_i p_D N(z_i; z_hat, S) / (lambda_FA c(z_i)) and `ScanEvent` holds only counts;
  - how the birth scan is marked;
  - which event sequence the Monte-Carlo reference is evaluated on, and what happens to
    seeds with no birth;
  - that there is no pruning.
- **`build_scan_events`**: will take the predicted covariance as well as the mean, so it
  can gate. Pending, together with the `ScanEvent` schema.
- **`docs/derivations/README.md`**: the A2 section-to-function rows (TODO(human)).

---

# Appendix: the approved implementation plan (2026-09-24, unchanged)

# crop_mot B1–B3 implementation plan

## Context

The repo at `/workspace` is an approved skeleton: 108 `raise NotImplementedError` bodies (tests
included) whose docstrings are the spec. Scope is B1 (simulator), B2 (Bernoulli filter + phantom
r-decay plot via the CLI), and B3 (r vs the A2 closed form + Monte-Carlo). B4 stays empty, and
`assignment.py` and `murty.py` are not touched. The work follows HANDOVER's seven steps as
vertical slices, with one commit per slice.

Pre-flight checks passed:
- DESIGN.md is current (world/, sensor/, analysis/analytic.py, initial_state/predict/update/extract).
- Python 3.10.12, numpy 1.26.4, scipy 1.13.1, matplotlib 3.10.9, PyYAML 6.0.3, pytest 9.1.1.
  MPLBACKEND=Agg, PYTHONPATH=/workspace.
- The suite gives 3 passed (docs) and 25 NotImplementedError. No ImportError.
- We are on `master`, so the work goes on a branch `b1-b3-implementation`. Each slice gets a commit
  "<WP>: <slice>" ending with the Co-Authored-By trailer.

## Decisions made by the author (2026-09-24)

| # | Topic | Decision |
|---|---|---|
| D1 | Density collapse | **Keep the best branch**, behind an extensible `CollapseStrategy` Protocol (new `filters/collapse.py`, one class `KeepBestBranch`, dict `COLLAPSE_STRATEGIES`). It is selected by an optional `filter.collapse: best_branch` key (`FilterConfig.collapse: str = "best_branch"`). A future moment-match or max-marginal strategy is one new class plus one dict line. |
| D2 | Update with m ≥ 2 gated detections | Implement the A0×A2 combination. Cite [A0 §Measurement model] + [A2 §3.1], with a docstring line saying it is not yet its own derived section. |
| D3 | Birth convention | The update runs first. The birth is applied after it, only into an empty Bernoulli (r == 0), with state (r_b, z, init_cov). The seed z is not reused that scan, so reported r at the birth scan = r_b exactly. A birth into r > 0, or with more than one component, raises `ValueError`. **Your r_sequence birth branch must match this.** |
| D4 | Noisy z outside the FOV | Keep A0's model with no truncation. `test_no_detection_outside_the_fov` asserts clutter z is inside the wedge and each real detection's source plant is inside the wedge (via labels). Its docstring is updated. A gated z with c(z) = 0 gives r⁺ = 1, per the A2 §3.1 limit, and this is documented. |
| D5 | c(z) | c(z) = 1/area inside the FOV, 0 outside (A0's f_c). The clutter intensity is lambda_FA·c(z). Fix the docstrings of `FieldOfView.area`, `sample_uniform_in_fov` and `SensorModel.clutter_density`. |
| D6 | RangeDependentPD | p_D is linear in range from p_D_near at min_range to p_D_far at max_range, and exactly 0 outside the FOV. `occlusion_factor != 1.0` raises `NotImplementedError` in `__post_init__`, naming the extension slot. The docstring is updated. |
| D7 | `extract` | Report the track whenever r > 0. A confirmation threshold, if wanted, belongs in analysis. The docstring says so. |

The per-decision docstring lines implement your three standing decisions:
- r_b is a configured stand-in for [A2 §4] r_b = e/(e + lambda_FA c(z)). This goes in the birth
  docstring and in the b2 YAML comment.
- p_S is pinned by the docstring: r ← p_S·r.
- The collapse (D1).

## Smaller proposals (approved with this plan unless you object)

1. **`BirthConfig.detection_index: int`.** The b2 YAML and `SingleFromMeasurement` both need it, and
   the strict loader would otherwise reject the YAML.
2. **Sort detections by z instead of shuffling.** Order carries no origin information, and no
   randomness is used, so the `detection` and `clutter` streams stay decoupled. Every visible plant
   always consumes one uniform and one noise draw, so stream use depends only on geometry. The
   `sample_scan` docstring step 5 is reworded.
3. **Shared builders, so the simulator, the filter and events.py build identical models.**
   - `sensor/models.py::build_measurement_model(cfg)`
   - `sensor/sensor_model.py::build_sensor_model(fov, detection, lambda_FA, measurement)`
   - A motion builder inside `build_bernoulli`.
4. **`analyse` entry point.** New `runner/analyse.py::analyse_run(run, plots=None)`. It reads
   `run.config` as a RunConfig, renders the configured plots and writes `metrics.json`. The CLI is
   `analyse --run DIR [--plots NAME ...]`. `--plots` lets the B2 plot be produced through the CLI at
   step 6, before the B3 plots exist.
5. **`build_scan_events` needs covariances to gate.** It gets `track_cov_per_scan`. The predicted
   moments at k come from the logged posterior at k−1 through the configured motion model (exact for
   StaticTarget). A shared analysis helper computes them, and both events.py and `plot_r_vs_k` use it.
   The filter is not touched.
6. **Wiring.**
   - Concrete classes get their own method bodies (they currently inherit the Protocol's raising stubs).
   - Relative `scenario:` paths resolve against the CWD.
   - A fresh `track` simulates `cfg.scenario` with its own seed into a folder named by the run config.
   - A run-dir name collision within one second gets a `_1` suffix.
   - The first scan predicts with dt = 0.

## Implementation slices

After every slice, run `pytest -m "not slow"`, the full suite and `pytest -m docs`. Earlier tests
must stay green. Later tests must fail only with NotImplementedError. Stub test bodies are written
from their docstrings and none is weakened.

### Step 1: types, io, rng, config (+ RunDir paths for the fixture) → `test_io_roundtrip.py`
- `FieldOfView.area` = half_angle·(max_range² − min_range²).
- `io`: `to_jsonable` recursion, `json.dumps` (repr floats round-trip float64 exactly), `newline="\n"`.
- `rng`: `zlib.crc32` and `SeedSequence([seed, stream_key(name)])`.
- `config`:
  - Strict parser with an allowed-key set per block and a `ValueError` naming the offending key.
  - kind/field consistency checks.
  - R and init_cov as ndarray.
  - Includes the D1 `collapse` key and `detection_index`.
- Commit: "B1: types, io, rng and strict config".

### Step 2: world/
- `generate_field`: row-major ids, N(0, jitter²) from `field`.
- `generate_path`: straight lane, pose_known=True only; the wobble branch raises `NotImplementedError`.
- `write_truth` / `read_truth`.

### Step 3: sensor/
- `in_fov`: body-frame transform, range and bearing test.
- `sample_uniform_in_fov`: sqrt-area radius [A0 §Measurement model].
- `LinearGaussianXY.h/H`, `ConstantPD`, `RangeDependentPD` (D6).
- `sample_scan`: visibility from the true pose, Bernoulli(p_D) coin flip, z = h(x) + v,
  Poisson(lambda_FA) uniform clutter, sort (proposal 2).
- `record` read/write, with None origins preserved.

### Step 4: run_dir, simulate, `__main__` → `test_b1_simulator.py`, and the simulate CLI produces config.yaml, run_meta.json, truth/labels/detections
- `write_run_meta`: seed, `git rev-parse HEAD` + dirty flag, versions, argv, UTC time.
- `conftest.tiny_scenario`: one short row placed so that no plant is in the FOV at scan 0. That
  guarantees scan-0 detections are clutter without reading labels, and the seed is chosen so scan 0
  has at least one detection.
- Statistical tests use a longer `dataclasses.replace` variant with binomial/Poisson ±4σ bounds.
- Commit: "B1: detector, recording and simulate CLI".

### Step 5: kalman, gating, motion, birth, collapse, bernoulli, filters registry, track runner → `test_b2_bernoulli.py`, `test_filter_interface_contract.py`

Kalman:
- `kf_predict` delegates to `StaticTarget` (F = I, Q = q·dt·I) [A1 §The prediction step is CK, evaluated].
- `predicted_measurement` returns ẑ and S = HPHᵀ + R.
- `kf_update` uses the Joseph form [A1 §Result (Kalman filter update)].
- `log_predicted_likelihood` is log N(z; ẑ, S) [A1 §Normaliser — and what it becomes one level up].

Gating: `chi2.ppf`, Mahalanobis distance, gated indices.

Bernoulli `update`, with predicted state (r, m, P):
- If r == 0, skip to birth.
- p_D = sensor.p_D(m, pose). **Simplification:** this plug-in stands in for p̄_D = ∫p_D p dx [A2 §2.1].
- For gated z_i, with ℓ_i = p_D·N(z_i; ẑ, S) and clutter intensity κ_j = lambda_FA·c(z_j), the
  product form of the [A0 §Measurement model] likelihood gives:
  - object absent: w_∅ = (1−r)·∏κ_j
  - object missed: w_0 = r(1−p_D)·∏κ_j [A2 §2]
  - z_i from the object: w_i = r·ℓ_i·∏_{j≠i}κ_j [A2 §3.1]
- r⁺ = (w_0 + Σw_i)/(w_∅ + w_0 + Σw_i). This reduces to A2 §2 at m=0 and to r_marg at m=1, and
  handles c(z)=0 without dividing by zero.
- Out of the FOV p_D = 0, so r and the density are unchanged. That falls out of the formula.
- Branch weights are normalised [A3 §Normalizing the mixture]. The branches are (w_0, m, P) and
  (w_i, `kf_update`(m, P, z_i)). They go to `self.collapse` (D1). The KeepBestBranch docstring says
  what it stands in for: the full mixture of [A2 §3.1].
- Birth is then applied per D3.

Rest of the filter:
- `predict`: r ← p_S·r, plus `kf_predict`.
- `extract` per D7.
- `build_bernoulli` computes gate_chi2 once.

Runner and registry:
- `build_filter` raises `KeyError` listing the available names.
- `run_filter` is the 9-line loop, with dt from timestamps and diagnostics via hasattr.
- `track_from_config` and `track_all_filters` are plain loops, needed by the contract test. They do
  not implement B4.

Tests:
- `test_r_decays_under_sustained_misdetection` feeds synthetic scans: a birth scan, then empty scans
  with the phantom in view.
- The ground-truth test deletes truth.jsonl and labels.jsonl before running the filter.

Commit: "B2: Kalman, gating and Bernoulli filter".

### Step 6: estimates_log, plot_scene, plot_r_vs_k, analyse runner → B2 deliverable
- `r_trajectory` writes 0.0 for scans where the track is not reported.
- `plot_r_vs_k` marks scans with a gated detection and shades out-of-FOV scans (proposal 5 helper).
- CLI run:
  1. `simulate`.
  2. Inspect `labels.jsonl` and set `detection_index` to the lowest-index scan-0 clutter detection
     (reported to you).
  3. `track --config configs/b2_bernoulli_phantom.yaml`.
  4. `analyse --run <dir> --plots scene r_vs_k`.
- I report what the plot shows. If it does not decay (see heads-up), I explain why and ask before
  touching `init_cov` or the index.
- Commit: "B2: estimates log and r-decay plot".

### Step 7: analytic, events, metrics, montecarlo, B3 plots → `test_b3_analytic_bernoulli.py` including slow
- **Blocked on your r_sequence docstring. I ask for it here.** I transcribe it without opening
  `filters/bernoulli.py`. Any ScanEvent field it needs is raised as a schema change first.
- `compare_r`, `standard_error`, `run_monte_carlo` (seeds base_seed + i, runs in a temp subdir),
  `plot_r_vs_analytic` (curves plus a difference panel) and `plot_r_montecarlo` (mean ± k·SE).
- `analyse_run` writes `metrics.json`.
- Commit: "B3: analytic cross-check and Monte-Carlo".

## Heads-up for your r_sequence docstring (shapes the ScanEvent schema)

- **Detection branch information.** The detection branch needs ℓ(z_i)/(lambda_FA c(z_i)), which
  depends on where z landed. No ScanEvent field carries it; there are only counts. Either the B3
  scenario is built so that nothing gates after birth, or ScanEvent needs a field such as
  Σ_i N(z_i; ẑ, S)/(lambda_FA c(z_i)).
- **Birth scan.** No ScanEvent field marks the birth scan, and the reference has r_birth but no at_scan.
- **Collapse consequence (D1).** With keep-best-branch, the logged means already reflect which
  branch won. events.py evaluates p_D and the gate at those means, so the reference sees the filter's
  actual trajectory.
- **Monte-Carlo reference.** Which event sequence is the single r_ref of the MC test evaluated on?
  Seeds with no scan-0 clutter give no birth (e^−2 ≈ 13.5 % of runs at lambda_FA = 2).
- **The configured phantom may not decay.** With init_cov = 0.25 I, the 99 % gate covers about
  8.4 m² of the 9.5 m² FOV. The phantom will usually gate plant or clutter detections.
- **README row cites a missing section.** The row cites "A0 §clutter intensity", but A0 has no
  such section. The code cites [A0 §Measurement model]. The README is yours, so I leave it alone.

## Verification

- After each slice: `python3 -m pytest tests -q -m "not slow"`, then `python3 -m pytest tests -q`,
  then `python3 -m pytest tests -q -m docs`.
- B1: `python3 -m crop_mot simulate --config configs/b1_two_rows.yaml`, then `ls` the run folder
  (config.yaml, run_meta.json, truth/labels/detections.jsonl).
- B2: `track`, then `analyse --plots scene r_vs_k`. Open `plots/r_vs_k.png` and look at it.
- B3: the full suite including `test_monte_carlo_mean_r_brackets_analytic`, plus `analyse --run <dir>`
  producing all four plots and `metrics.json`.
- Invariants: `grep -rn "rclpy\|plt.show" crop_mot/` is empty. `runner/track.py` has no
  `isinstance` and no filter-name branch. No filter signature accepts truth or labels.
