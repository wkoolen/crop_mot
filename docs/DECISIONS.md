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
| D9 | Phantom recapture | Kept and shown, not engineered away. r decays 0.08 to about 2e-4 over scans 24–31, then the phantom is recaptured (r > 0.5 at k = 37, then 1). Under keep-best-branch the branch weights compare r(1 - p_D) with r p_D N(z; z_hat, S) / (lambda_FA c(z)), so r cancels and the Gaussian follows any detection within about 1 m. Nothing prunes. *Alternatives considered:* init_cov 0.06 I (this seed's phantom then stays dead); pruning below a threshold (A2 §5's deletion). *Update 2026-09-28:* pruning now exists (D13) in the bank filter; the single-track run keeps no pruning, and the bank run shows this recapture only as the unpruned overlay. | `configs/b2_bernoulli_phantom.yaml` comment |
| D10 | Detections per plant per scan (multiplicity) | Optional `sensor.multiplicity` block, truth side only: `single` (default; A0 point target, at most one detection per plant), `duplicate` (detector artefact: extra hits at z_primary + N(0, spread_std² I), capped geometric count with p_split, max_extra) or `poisson` (extended objects: Poisson(gamma) hits at h(x) + N(0, extent_std² I + R); p_D unused, so only with `detection.kind: constant`). All extra draws come from a new `multiplicity` substream, so `single` output is byte-identical to before and `duplicate` keeps the primary detections and clutter of the same seed. Every hit is truncated to the FOV (D4). The filter's `assumed_sensor` has no multiplicity, so Bernoulli on these scenarios is a model-mismatch study; the B3 plots refuse any kind other than `single`. An EOT filter would add an assumed counterpart. | `config.py::MultiplicityConfig`, `sensor/detector.py`, `runner/analyse.py`, `configs/b1_two_rows_{duplicate,extended}.yaml` |
| D11 | Clairvoyant scan counts | Derived on demand from labels.jsonl, never stored, so they cannot drift from it. `simulate` prints a summary; `analyse` draws a `counts` plot, also for a simulate-only run folder. `ScanLabels.truncated_ids` (default empty, so older labels load) separates a D4 edge loss from a p_D miss: on seed 42, 58 of 967 visible plant-scans are edge losses, which is why detected/visible is 0.78 against p_D = 0.85. | `analysis/counts.py`, `types.py::ScanLabels`, `plots.py::plot_counts` |
| D12 | Several phantom hypotheses | A new filter `bernoulli_bank`: independent Bernoulli components, one per author-picked seed in `birth.seeds` (birth kind `from_measurements`); the seed's list position is the track id. Each component is advanced by the unchanged `BernoulliFilter`, so each track is the A2 recursion and a one-seed unpruned bank equals the single filter exactly (tested). Components do not compete for detections: an independence approximation, harmless while seeds are more than a gate apart. *Alternative rejected:* birth at every unassociated detection - effectively PMB without its Poisson part, B4 scope, and outside B3's closed form. | `filters/bernoulli_bank.py`, `BirthConfig.seeds` |
| D13 | Pruning | Optional `filter.prune.r_min`, default 0 (off; every earlier run unchanged). The bank deletes a component in `predict` when its predicted r < r_min [A2 §5], so the posterior r that crossed is still logged and the track is absent from the next scan: deletion is readable from the estimates log alone (`track_lifetimes`). The single `bernoulli` filter refuses a non-zero r_min. | `BernoulliBankFilter.predict`, `PruneConfig` |
| D14 | Unpruned companion log | When r_min > 0, `track` also runs the same filter with pruning off into `estimates_<kind>_unpruned.jsonl`, on the same detections; `run_filter` gained an optional `log_name`. The `hypotheses` figure draws it after each deletion, showing what pruning prevented. | `runner/track.py` |
| D15 | Weeds: persistent false targets | Optional, truth side only, in two blocks that must come together: `world.weeds` (where: a homogeneous Poisson point process, `density` per m² over a rectangular `region`) and `sensor.weed_detection` (how often a visible weed is reported as a plant, in the `detection` block format). Each scan every weed in the FOV is reported independently with that probability at w + N(0, R), truncated to the FOV (D4); recurrence comes from the fixed position alone. A weed detection has `origin` None - to a filter tracking plants it is clutter - and the new `ScanLabels.weed_origin` / `visible_weed_ids` (empty without weeds, so older labels load) tell it apart from the Poisson clutter; truth.jsonl gains a `weeds` record (absent in older files: no weeds). Positions come from a new `weeds` substream and the coin flips and noise from `weed_detection`, so with weeds each scan is exactly the scan without weeds plus the weed detections (tested), and a weeds block that draws no weed is byte-identical to none. The filter's `assumed_sensor` has no counterpart: weeds are a model-mismatch study of the Poisson-clutter assumption. Unlike multiplicity (D10) they do not break the A2 single-detection assumption, so the B3 plots are not refused: the per-run cross-check still checks the implementation, and the Monte-Carlo half is where the mismatch should show. *Alternatives considered:* weed ids in `origin` (every consumer would count weeds as plants unless changed); one combined block (mixes world and detector, the split `SensorConfig` exists to keep apart). *Measured 2026-09-28* (seed 42, `b2_bernoulli_bank_weeds.yaml` against `b2_bernoulli_bank_phantoms.yaml`): phantoms #1 and #4 lock onto weeds and are confirmed (r 0.99, 0.998) where without weeds #1 is pruned and #4 is recaptured by a plant. Over 200 seeds per weed p_D in {0.2, 0.35, 0.5, 0.7, 0.85}, one pruned track per phantom seed at least 1 m from plants, a phantom born on a weed ends confirmed on it in 7, 12, 16, 32, 49 % of cases and is pruned in 6-18 %; one born on Poisson clutter at least 1 m from any weed ends on a weed in about 1 %. In every group the most common end (up to about 76 %) is capture by a plant (D9). | `config.py::WeedsConfig`, `world/field.py::generate_weeds`, `world/truth.py`, `sensor/detector.py::sample_scan`, `types.py::ScanLabels`, `analysis/counts.py`, `analysis/candidates.py`, `analysis/plots.py`, `configs/b1_two_rows_weeds.yaml`, `configs/b2_bernoulli_bank_weeds.yaml` |
| D16 | Known pose (RTK-GPS) (**revised**) | `path.pose_known: true` is a simplification for now, confirmed by the author on 2026-09-30: the robot is assumed to carry RTK-GPS, so its absolute pose is known. It is revisited if RTK is not available on the Go2 setup, not silently kept. With static plants the problem is mapping with known poses: no filter carries a pose state, the motion model is the identity, and all uncertainty is in association, existence and clutter; the path only decides what is in view, through p_D(x, pose). *Bound stated with it:* RTK gives position to about a centimetre, but heading comes from another sensor; one degree of heading error moves a detection at 4 m by about 7 cm, against sigma = 0.2 m measurement noise. A heading bias moves every detection in a scan the same way and does not average out over scans, so heading error is studied in a sensitivity experiment (roadmap step 8d, NEES against yaw bias); the yaw-wobble slot (`pose_known: false`) stays for it. No behaviour change. *Superseded first wording (same day):* known pose as the thesis scope, not a simplification; weakened in the review of 2026-09-30 because RTK on the Go2 is not confirmed. | `world/path.py::generate_path` |
| D17 | Monte Carlo over seeds | Decided by the author on 2026-09-30. The per-seed cross-check is the B3 deliverable: on every seed, every track's r matches the A2 recursion evaluated on that seed's own events (`montecarlo.cross_check_summary`: seeds passed out of those with a birth, max error, divergences as (seed, track, scan), scans checked per branch). The former test "mean r brackets the closed form" is dropped: r is nonlinear in the events and each seed has its own event sequence, so no single closed form describes the mean; it became `test_per_seed_cross_check_over_random_seeds`. `MonteCarloResult` (mean r of the first track, standard-error band) stays as the descriptive `r_montecarlo` figure, with no closed-form line. The event-rate model check (miss rate against p_D, gated clutter against lambda_FA c(z) times the gate area, weeds in gates) is future work (roadmap 4b). `run_monte_carlo` is rebuilt on `run_trials`, a general loop: per seed base_seed + i, simulate, run the configured filter and its unpruned companion (`run_configured_filter`), and apply named measurement functions while the folder exists, so later metrics plug in without a new loop. One outcome per seed, since tracks in a seed share clutter and neighbours. Plain Monte Carlo (independent trials), not MCMC. `analyse --plots r_montecarlo` runs `analysis.monte_carlo.n_runs` seeds from the run's own and writes the summary to metrics.json as `b3_monte_carlo`. *Measured 2026-09-30:* `b2_bernoulli_phantom`, seeds 42-91: 44 seeds with a birth, all pass, 1584 scans checked, max error 1.0e-15, every branch covered; the other 6 have no detection 12 at scan 24. *Not yet:* the controlled phantom injection, fates and strata (roadmap open question 4) and timing (4c). | `analysis/montecarlo.py`, `analysis/crosscheck.py`, `runner/analyse.py`, `runner/track.py::run_configured_filter`, `analysis/plots.py::plot_r_montecarlo` |
| D21 | Cardinality restricted to the view | The cardinality figure compares, per scan, the sum of r over the tracks whose mean is in view with the number of true plants in view. "In view" is one function on both sides: `in_fov` at the true pose with the scenario's FOV, which is what the simulator used for `ScanLabels.visible_ids`. GOSPA and NEES are restricted the same way, so a plant behind the robot is neither a missed object nor a false track. Decided by the author in the review of 2026-09-30. | `analysis/evaluation.py::scan_views`, `cardinality` |
| D22 | Class labels on detections (roadmap step 8b) | Decided by the author in the review of 2026-09-30: detections carry a class label, plant tracks do not associate weed-labelled detections, and weeds may get their own map later. Built: `Detection.label` ("plant" or "weed", default "plant"), what the detector says it saw - an output like z, never its origin - written to detections.jsonl as a list parallel to z (an older file reads back as all "plant"). World side: `sensor.classifier`, a confusion matrix P(label | origin) with rows plant, weed, clutter (`ClassifierConfig`, default the perfect classifier: plants and weeds labelled what they are, clutter labelled "plant"); labels are drawn from a new `classifier` substream, one uniform per detection, so truth, labels.jsonl and every z are unchanged (checked on both shipped scenarios). Filter side: `filter.assumed_classifier`, the same format, what the filter believes; absent means labels are ignored, so every earlier estimate is unchanged (difference 0.0). With the perfect classifier a plant track drops weed-labelled detections from its gate (`gating.without_weed_labels`, shared by the filter, `build_scan_events` and `gate_contents`); an imperfect one raises until the label factor is derived (open question 7, the author's). Reduction test: without weeds the label-aware filter equals step 8a exactly. Weed-labelled detections stay in detections.jsonl for a future weed map. *Measured 2026-09-30* (seed 42, `b4_known_n_bank_labels` against `b4_known_n_bank_weeds`): weed returns in gates 9.2 % -> 0 %, mean GOSPA 0.249 -> 0.244 m, neighbours and clutter unchanged. | `types.py::Detection`, `config.py::ClassifierConfig`, `sensor/detector.py`, `sensor/record.py`, `association/gating.py`, `filters/bernoulli.py`, `analysis/events.py`, `configs/b4_known_n_bank_labels.yaml` |
| D23 | N bounded by the plan, and missing plants (roadmap step 8c) | Decided by the author in the review of 2026-09-30: the thesis scope is N bounded by the plan, bounded N moves forward as step 8c, and N known is its limiting case. Built (open question 8 answered by Claude at the author's request, 2026-09-30): the scenario's `world.p_missing` empties each planned slot independently (the simplest placement; runs of neighbouring gaps are a later brick), from a new `missing` substream drawn after every slot's jitter, so present plants keep their id (= slot) and position, and truth.jsonl records the empty slots' ids and nominal positions (only when there are any). The filter's `plan.r_0` is each slot's prior existence, default 1 (step 8a); `b4_bounded_n_bank.yaml` uses r_0 = 0.9 = 1 - p_missing on `b1_two_rows_missing.yaml` (p_missing 0.1). Reduction test: r_0 = 1 gives step 8a exactly. The A2 cross-check covers every slot: `BernoulliExistenceReference.r_initial` (r before scan 0, an initial condition of the author's recursion, not a new branch) and the plan's prior as the scan-0 moments. `evaluation.missing_plants` scores every slot by position - the largest r of the tracks nearest it within d_match, 0 without one - so methods with births are scored alike; figure `missing_plants` (in `ANY_FILTER_PLOTS`; a labelled "not applicable" panel, the first, when the field has no missing plants) and `compare`'s `empty_found` / `plants_flagged`. *Measured 2026-09-30* (seed 42): every slot matches A2 to 1.1e-15, and the bank finds none of the 4 empty slots it sees - each ends at r = 1.0, above some real plants - as the roadmap expected: two neighbour detections 0.35 m away, each with a likelihood ratio of about 3.5 against clutter, multiply an empty slot's odds by about 7 per scan. Any threshold that finds 75 % of them flags 28 % of the plants. With the empty slots confirmed, mean GOSPA is 0.44 m and NEES is in band on 85 % of scans. This is the main motivation for JIPDA (step 12). | `config.py::WorldConfig.p_missing`, `PlanConfig.r_0`, `world/field.py`, `world/truth.py`, `analysis/analytic.py`, `analysis/crosscheck.py`, `analysis/evaluation.py::missing_plants`, `analysis/plots.py`, `configs/b1_two_rows_missing.yaml`, `configs/b4_bounded_n_bank.yaml` |
| D18 | Tolerance of regression, reduction and B3 comparisons | An absolute tolerance of 1e-12 on every r, mean and covariance entry, instead of byte-identical logs. The B3 cross-check uses the same number on r: `metrics.R_TOLERANCE`, now the default of `compare_r` (was 1e-9). Absolute, because r saturates near 1, where a relative error on 1 - r or a log-odds error would blow up without meaning anything. | `analysis/metrics.py::R_TOLERANCE`, `compare_r` |
| D24 | Tests written before their code | Marked `@pytest.mark.xfail(strict=True, reason="waits on step N")`, N a roadmap step. The suite stays green, and strict mode fails the moment such a test starts passing, so the marker is removed in the commit that makes it pass. First uses: `test_r_matches_analytic_recursion` (step 3) and `test_monte_carlo_mean_r_brackets_analytic` (step 4, D17), reworked there into `test_per_seed_cross_check_over_random_seeds`. | `tests/` |
| D25 | B3 event interface | `ScanEvent` gains `likelihood_ratios`: per gated detection, in gate order, ell_i / kappa_i = p_D N(z_i; z_hat, S) / (lambda_FA c(z_i)), evaluated with the filter's assumed sensor at its predicted moments and the scan's reported pose; `len == n_gated` is checked; `math.inf` where c(z_i) = 0. It also gains `born`: True at the one scan the track is born on, the first scan it is reported in the estimates log, so any birth model works (seeds today, the planting map and measurement births later). `build_scan_events` takes the predicted (mean, cov) per scan, as `predicted_track_moments` returns them, instead of means only: the gate and ell_i both need S. `r_sequence` is transcribed from its docstring, which the author checked line by line against A2 (2026-09-30), without opening `filters/bernoulli.py`. *Scope, stated with it:* the check verifies the existence recursion given the filter's predicted moments and likelihoods. The Gaussian algebra comes from the shared `kalman.py` helpers the filter also uses, and both sides see the same gate and the same p_D value, so B3 does not check those; NEES does (roadmap step 5). | `analysis/analytic.py::ScanEvent`, `BernoulliExistenceReference.r_sequence`, `analysis/events.py::build_scan_events` |
| D26 | B3 cross-check through `analyse` | `analyse --plots r_vs_analytic` checks every track in the estimates log against the configured `b3_reference`, draws one figure per track and writes a `b3_cross_check` entry to metrics.json: the reference, the log checked, the tolerance (D18), and per track the max and rms error, the first divergence and the branch coverage (`events.branch_counts`: before birth, birth, out of view, miss, one and several gated detections), so a pass is never read without knowing which branches it covered. A pruning filter is checked on its unpruned companion log (D14): A2 has no deletion step. Per-track plots (`r_vs_k`, `r_vs_analytic`) are `<plot>.png` for one track and `<plot>_track<id>.png` for several. The scene plot's scan is `--scene-k`, by default the first scan any track is reported (was `birth.at_scan`: the same scan for every shipped config, and absent from a planting map). The two bank configs now name `bernoulli_existence` and draw `r_vs_analytic`; their comment said the bank was outside B3, but each bank track is the A2 recursion (D12). `r_montecarlo` is D17's. *Measured 2026-09-30* (seed 42): every track of `b2_bernoulli_phantom`, `b2_bernoulli_bank_phantoms` and `b2_bernoulli_bank_weeds` matches A2 to 1.2e-15 at most, the six branch kinds all covered. | `runner/analyse.py`, `analysis/events.py::branch_counts`, `__main__.py`, `configs/b2_bernoulli_bank_{phantoms,weeds}.yaml` |
| D27 | p_D evaluation (roadmap open question 3) | A2 claims the exact forms: the existence update uses p_D_bar = integral p_D(x) p(x) dx [A2 §2, §2.1] and the detection branch ell(z) = integral p_D(x) g(z|x) p(x) dx [A2 §3.1]. The filter keeps option A of roadmap step 3b, `AtMean`: p_D at the predicted mean, the plug-in its docstring and the `r_sequence` docstring already state. Step 3b adds the strategy interface with A wrapping the current code (outputs within 1e-12, D18) and B to D as stubs; C (p_D_bar from sigma points) is the next brick, once the FOV-edge effect is measured to matter (D11: detected/visible 0.78 against p_D 0.85). `ScanEvent.p_D` keeps carrying the value the filter used, so B3 keeps testing the combination formula, not the p_D approximation. Decided by Claude at the author's request (2026-09-30: answer the open questions, keep everything as documented, build complexity brick by brick); the author may revisit it. | roadmap step 3b: `filters/detection_prob.py` (D33) |
| D28 | Fate thresholds and the controlled phantom (roadmap open question 4) (**revised**) | r_conf = 0.5: a track is confirmed when it is more likely there than not (the threshold D9 already reads recapture at). The same threshold makes the set estimate for GOSPA and picks the tracks NEES checks (roadmap step 5). d_match = 0.5 m, the GOSPA cutoff c (D31): a track is on a plant or weed when its mean is within 0.5 m of it, the nearest one deciding, so a fate and GOSPA agree on what a false track is. *Superseded first choice (same day):* d_match = 0.2 m, one measurement sigma. Over 200 seeds of `b2_phantom_fates` it labelled 40 phantoms "sustained by clutter" that plants had captured: r about 1, 0.20-0.29 m outboard of the row, posterior std about 0.045 m. The controlled phantom reuses D8's: position (1.94, 4.82) at scan 24, r = 0.08, init_cov 0.12 I, so the per-seed fates extend the documented single-seed experiment; the rows are jittered by 3 cm only, so it stays about 1.17 m from the nearest plant in every seed. Decided by Claude at the author's request (2026-09-30: answer the open questions, keep everything as documented, build complexity brick by brick); the author may revisit it. | roadmap steps 4a (fates, not built yet) and 5 |
| D29 | Timing log format (roadmap open question 5) | A separate `timing_<log_name>.jsonl` per filter run, as the roadmap recommends, with k, predict_s, update_s, extract_s, n_detections and n_reported per scan, so the estimates logs stay deterministic; thread settings and the CPU model go into run_meta.json. Decided by Claude at the author's request (2026-09-30: answer the open questions, keep everything as documented, build complexity brick by brick); the author may revisit it. | `runner/track.py::run_filter`, `RunDir.timing`, `runner/run_dir.py::write_run_meta` (step 4c) |
| D30 | Association accuracy and the association figure (roadmap open question 6) | Deferred, not dropped. With only the Bernoulli filter and the bank there is one association rule (each track against the whole scan, independently), so the metric cannot separate methods yet. The richer diagnostics channel both need is designed at roadmap step 11, when GNN gives a second rule; `HasDiagnostics` stays a flat dict[str, float] until then, and step 5 is built without the metric. Decided by Claude at the author's request (2026-09-30: answer the open questions, keep everything as documented, build complexity brick by brick); the author may revisit it. | roadmap steps 5, 7 and 11 |
| D31 | GOSPA and NEES settings | GOSPA with c = 0.5 m, p = 2 and alpha = 2 (the form that splits into localisation, missed and false): 0.5 m is 2.5 measurement sigmas, and a track further than that from every plant is a false track, not a poor position. The set estimate it scores is the in-view tracks with r > r_conf (D28). `gospa` returns the decomposition and the assigned pairs instead of the stub's float. NEES is averaged over the pairs of GOSPA's assignment, against the 95 % band of chi-square(n dim) / n; the known-N rule (track i against planned plant i) arrives with roadmap step 8a. Decided by Claude at the author's request (2026-09-30: build complexity brick by brick); the roadmap left c open. | `analysis/metrics.py::gospa`, `analysis/evaluation.py` |
| D32 | Figures that work for any filter (roadmap step 5) | Six figures read only (run, filter_name) - the estimates log, truth, labels and the scenario FOV, never a filter's config - and are listed in `analyse.ANY_FILTER_PLOTS`: `tracks` (every track as its 2-sigma ellipse, opacity r), `existence_map` (D(x) = sum r_i N(x; m_i, P_i) on a 5 cm grid, square-root colour scale stated on the colorbar), both at the last scan by default (`analyse --scene-k` picks another); `cardinality` (D21); `gospa` (distance, and GOSPA^2 stacked from localisation, missed and false, D31); `nees` (average over GOSPA pairs against the 95 % band, pair count below); `lifetimes` (birth to deletion per track, A2 §5). Association accuracy is deferred (D30). The "not applicable" panel of roadmap §2 rule 1 arrived with `missing_plants` (D23). *Measured 2026-09-30* (seed 42, `b2_bernoulli_bank_phantoms`): the two phantoms that plants capture sit above the NEES band from scan 44, i.e. their covariances are overconfident. Decided by Claude at the author's request (2026-09-30: build complexity brick by brick). | `analysis/plots.py`, `analysis/evaluation.py`, `runner/analyse.py` |
| D33 | p_D evaluation interface (roadmap step 3b) | `filters/detection_prob.py::PdEvaluation` has three methods, the three columns of the step-3b table: `miss_p_D` (the p_D in r (1 - p_D)), `detection_p_D` (the p_D in each ell_i) and `missed_moments` (the missed branch's Gaussian). Option A, `AtMean`, wraps what the filter always did and is the default (D27); B `AtBranchMean`, C `Expected` and D `ExpectedWithShift` raise `NotImplementedError` at construction, each naming what it waits for. Selected by the optional `filter.p_D_evaluation` key (default `at_mean`), like `collapse`. The Bernoulli filter, and through it the bank, uses it, and so does `build_scan_events`, so `ScanEvent.p_D` and the ratios carry the values the filter used. *Measured 2026-09-30:* the estimates logs of `b2_bernoulli_phantom` and both bank configs (seed 42, pruned and unpruned) are identical before and after, difference 0.0. | `filters/detection_prob.py`, `filters/bernoulli.py`, `analysis/events.py`, `config.py::FilterConfig.p_D_evaluation` |
| D34 | The controlled phantom is a birth kind | Birth kind `injected` (`filters/birth.py::InjectedPhantom`) places one component at `birth.position` at scan `at_scan` with r_b and init_cov, with no detection behind it: the controlled phantom of roadmap step 4a (D28). It uses the filter's birth-model slot because that is the only way in that keeps the runner free of special cases (HANDOVER invariant 2), but it is documented as an experiment setting, not a model of where objects come from. Both `bernoulli` and `bernoulli_bank` accept it (`build_single_birth`); the bank can prune it. `detection_index` is -1 for it. `configs/b2_phantom_fates.yaml` runs D28's phantom in the field with weeds over 200 seeds. *Why not `single_from_measurement`:* a fixed detection index is clutter in one seed, a plant in another and missing in a third (6 of 50 seeds in D17's run), which mixes different experiments. | `filters/birth.py`, `filters/bernoulli.py`, `filters/bernoulli_bank.py`, `config.py::BirthConfig.position`, `configs/b2_phantom_fates.yaml` |
| D35 | Phantom fates over seeds (roadmap step 4a) | `analysis/fates.py`: per seed, the controlled phantom's fate (D28) from the pruned log and truth, and how many Poisson clutter and weed returns fell in its gate over its life, rebuilt from the log as the filter gated (analysis side: `ScanLabels.weed_origin` and `origin`, no filter input). Seeds are stratified by whether a weed return ever fell in the gate. Proportions per stratum get 95 % Wilson intervals (the normal interval leaves [0, 1] near 0 and 1 with tens of seeds). `PhantomOutcome` keeps r, the distances to the nearest plant and weed and the gate counts, so fates can be reclassified without rerunning. `analyse --plots phantom_fates` runs `analysis.monte_carlo.n_runs` seeds, writes `phantom_fates` to metrics.json and draws one panel per stratum. *Measured 2026-09-30* (`b2_phantom_fates`, seeds 42-241): confirmed on a plant 135/200 (61-74 %), pruned 40 (15-26 %), confirmed on a weed 15 (5-12 %), alive and unconfirmed 9, sustained by clutter 1. On a weed only with a weed return in the gate: 15/67 there, 0/133 without. Captured phantoms end a median 0.16 m from their plant, biased outboard under `best_branch` (compare the NEES excess, D32). *Not yet:* the sweep of the injection's distance to the nearest weed, per-seed plant proportions (they need a mapping filter, step 8a) and the confusion-rate comparison (step 8b). | `analysis/fates.py`, `runner/analyse.py`, `analysis/plots.py::plot_phantom_fates` |
| D36 | Scaling study (roadmap step 4c) | `python3 -m crop_mot scaling --config X --sweep S --values ... --seeds n` times the config's filter over one swept scenario parameter, n seeds per value through the trial loop (D17), into a folder of its own with metrics.json (`scaling`) and plots/scaling.png. Four sweeps (`analysis/timing.py::SWEEPS`), each with the problem size it drives, measured in the runs rather than assumed: `row_length` (the path lengthened to walk the row) against total plants N, `spacing` against plants in view per scan, `lambda_FA` (set on both sides, so the model stays matched) and `weed_density` against detections per scan. Median and 95th percentile of the update time per scan over all seeds, scan 0 apart (warm-up), on log-log axes with the scan period as the budget; the slope of log median against log size is the empirical exponent. The command warns when OMP_NUM_THREADS or OPENBLAS_NUM_THREADS is not 1. `simulate` gained `summary=False` so trial loops do not print a counts table per seed. *Measured 2026-09-30* (`b2_bernoulli_bank_weeds`, single-threaded): lambda_FA 1-32, 3 seeds: median 0.3-2.6 ms, slope 1.47 against 15-47 detections per scan (noisy: the bank holds 1-2 tracks); row length 6-48 m, 2 seeds: median 0.6-1.2 ms, slope 0.28 against N = 36-276 plants, as expected, since the bank's track count is set by its seeds, not by N. Every point is two orders under the 0.25 s budget; the study becomes informative with a mapping filter (step 8a). *Measured with the known-N map* (`b4_known_n_bank`, row length 6-48 m, 2 seeds; the `row_length` and `spacing` sweeps now move a filter's plan with the world, keeping it matched): N = 36, 70, 138, 276 slots give medians 13, 26, 44, 88 ms (95th percentiles 30-128 ms), slope 0.92 against N - the bank's cost grows with total N, as the roadmap expected, since every slot is updated every scan. The median would reach the budget near N = 790, the 95th percentile near 540; skipping slots with p_D(mean) = 0 is the fix to test. *Not done:* skipping tracks with p_D(mean) = 0, which needs a 1e-12 test first (roadmap 4c). | `analysis/timing.py`, `runner/scaling.py`, `analysis/plots.py::plot_scaling`, `__main__.py` |
| D37 | The compare command (roadmap step 6) | `python3 -m crop_mot compare --run DIR --filters A B [--plots ...]` runs every named filter on the run folder's detections with `track_all_filters` (each built from the run config's one filter block, only `kind` replaced), draws every figure of `ANY_FILTER_PLOTS` (D32) once per filter into plots/compare/<filter>/, and places them side by side in plots/compare_<plot>.png. metrics.json gets `compare`: per filter the roadmap's evidence - mean GOSPA, the share of paired scans whose average NEES is inside its 95 % band, and the median update time per scan. Composing the saved PNGs, rather than drawing into shared axes, leaves every figure function unchanged. *Measured 2026-09-30* (`b2_bernoulli_phantom`, seed 42): bernoulli and bernoulli_bank agree exactly (mean GOSPA 1.41 m, NEES in band 1.0), as a one-seed unpruned bank must (D12). | `runner/compare.py`, `__main__.py` |
| D38 | Figure contract (roadmap step 7) | `tests/test_figure_contract.py` renders every figure of `analyse.ANY_FILTER_PLOTS` for every entry of `FILTERS`, parametrised over both so a new filter or figure is checked without editing the test; each filter is built from the shipped phantom config's filter block with only `kind` replaced, as in `test_filter_interface_contract.py`. A labelled "not applicable" panel counts as rendering, an exception does not; the first such panel is `missing_plants` on a field without missing plants (D23). The B2/B3 figures (`r_vs_k`, `hypotheses`, `r_vs_analytic`, `r_montecarlo`) are outside the contract: they show the A2 recursion's quantities and are Bernoulli-specific by design. The optional association figure stays deferred with its diagnostics channel (D30). | `tests/test_figure_contract.py` |
| D39 | N known: the planting-plan prior (roadmap step 8a) | An optional `filter.plan` block (`PlanConfig`: the planned rows in the `world.rows` format, and `prior_std`) makes `bernoulli_bank` start from one component per planned slot: the nominal position, covariance prior_std^2 I, r = 1, track id = slot index (`plan_components`). With `birth: null` (now allowed; `FilterConfig.birth` is optional and the single filter then gets `NoBirth`) nothing is ever born. Truth-blind like `assumed_sensor`: the plan is the nominal grid, and `config.nominal_positions` is now the one grid function the simulator (plus jitter; its output is byte-identical) and the plan share, so filters still import nothing from `crop_mot.world`. Kept in the bank rather than a new filter kind: each slot is the unchanged Bernoulli filter, so the step-8a map is "the bank on the map" that roadmap step 12 compares JPDA with. Refused: a plan in the single filter, and a plan together with births (their track ids would collide; step 13). For r = 1 the A2 check is trivial (certainty is absorbing); since step 8c it covers planned slots from their prior at scan 0 (D23). Configs `b4_known_n_bank.yaml` and `b4_known_n_bank_weeds.yaml`, prior_std 0.036 m. *Measured 2026-09-30* (`b4_known_n_bank`, seed 42): 70 slots at r = 1 every scan, mean GOSPA 0.24 m, average NEES inside its band on 98 % of scans, median update 24 ms per scan. | `config.py::PlanConfig`, `nominal_positions`, `filters/bernoulli_bank.py::plan_components`, `filters/bernoulli.py`, `configs/b4_known_n_bank{,_weeds}.yaml` |
| D40 | Step 8a's failure modes, from the labels | `evaluation.gate_contents` (figure `gate_contents`, one of `ANY_FILTER_PLOTS`): for every in-view track from scan 1 on, the gate rebuilt from the log as the filter applied it, its detections sorted by true origin - the track's own plant (the true plant nearest its first report: a plan slot's own plant), a neighbour plant, Poisson clutter, a weed - and whether the track is pulled (its mean nearer another plant than its own). It reads the run config's common `measurement` and `gate` blocks, which every filter shares. `compare` adds `pulled_share` per filter. *Measured 2026-09-30* (seed 42): `b4_known_n_bank`: a neighbour's detection in 91 % of gates (1.6 per gate against 0.76 from the plant itself), clutter in 19 %, no plant pulled; `b4_known_n_bank_weeds`: the same plus a weed return in 9 %, still none pulled. *Reading it:* with a 3.6 cm prior the Kalman gain is about 0.03, so even a wrongly kept neighbour detection 0.35 m away moves a slot by about 1 cm; being pulled needs more than 17.5 cm. For known N the neighbour detections cost small biases (GOSPA localisation, NEES), not swaps, and which detection the best branch kept - the direct association error - is the association-accuracy metric deferred with its channel (D30). The 91 % is what roadmap step 12 (JPDA) must handle. | `analysis/evaluation.py::gate_contents`, `analysis/plots.py::plot_gate_contents`, `runner/compare.py` |
| D41 | The heading-error experiment (roadmap step 8d) | Open question 9 answered by Claude at the author's request (2026-09-30): `path.yaw_bias` (radians, default 0, ignored while `pose_known` is true) is the constant yaw-bias slot next to the existing wobble. `configs/b4_yaw_sensitivity.yaml` (`YawSensitivityConfig`: a `run:` config, `yaw_bias_deg` 0-2, `yaw_wobble_std_deg` 0 and 1, seeds) and `python3 -m crop_mot yaw --config ...` sweep them on the known-N map, the filter still assuming the reported pose is the true one. First brick as the roadmap documents (config, command, stub body, parse test); the body followed at once, since its condition (NEES, step 5) was met. `pose_known: false` is now implemented, for this experiment only: the reported pose is the true pose plus `yaw_bias` and N(0, wobble^2) in heading and N(0, xy_noise^2) in position, drawn per scan from the path substream; detections are generated from the true pose and re-expressed through the reported one (`detector.reexpress`: z_rep = p_rep + R(theta_rep - theta_true)(z - p_true), rigid), so every z of a scan moves the same way; with a known pose nothing changes. The experiment sets the position noise to 0 (RTK) and measures, per seed, the average NEES, its in-band share and mean GOSPA. *Measured 2026-09-30* (`b4_yaw_sensitivity.yaml`, 3 seeds): without wobble, bias 0 / 0.5 / 1 / 2 degrees gives average NEES 2.2 / 2.3 / 2.5 / 2.9, in band on 88 / 84 / 77 / 61 % of scans, mean GOSPA 0.258 / 0.259 / 0.273 / 0.285 m; a 1-degree wobble changes little (correlation, not size, matters, as D16 says). Weaker than the roadmap expected ("out of the band below one degree"): each update moves a slot only about 3 % of the way to a detection (prior 3.6 cm against 20 cm noise), so a slot takes up only part of the heading-induced shift, and that part stays within its about 3 cm posterior std until about 2 degrees. | `config.py::PathConfig.yaw_bias`, `YawSensitivityConfig`, `world/path.py::generate_path`, `sensor/detector.py::reexpress`, `runner/yaw_sensitivity.py`, `analysis/plots.py::plot_yaw_sensitivity`, `configs/b4_yaw_sensitivity.yaml` |
| D42 | The existence map over time, as a GIF | `analyse --plots existence_anim` writes plots/existence_map.gif, one frame per scan, for any filter (it reads only the run folder). Left: D(x) = sum_i r_i N(x; m_i, P_i) after scan k on a 2 cm grid in a window that follows the robot (1.5 m behind to 5.5 m ahead - on the whole field a converged track is one pixel), on a square-root colour scale fixed over the run (its top is the brightest track peak, r / (2 pi sqrt(det P))), so a slot brightens as r rises and sharpens as P shrinks; the true plants, weeds and empty slots; the robot's path and FOV; scan k's detections by true origin (a weed-labelled one the filter ignores is hollow, D22), with a ring around every clutter or weed return that falls inside some track's gate as the filter gated it - the false returns that can move a track. Right, up to a cursor at k: detections per scan by origin; the returns inside the in-view gates not from the track's own plant (neighbour, clutter, weed; `gate_contents`); the mean r of the tracks in view; their median position std. Kept out of `ANY_FILTER_PLOTS` (one frame per scan is slow; a slow test covers known N with weeds, bounded N and the phantom bank). *Measured 2026-09-30* (`b4_known_n_bank_weeds`, seed 42): 60 frames of 1200 x 800 px, 2.8 MB, about 85 s. With the 3.6 cm prior the known-N slots start almost certain (std 3.6 -> 2.9 cm, r = 1), so the map changes little; the phantom bank and the bounded-N run show more. | `analysis/plots.py::animate_existence_map`, `runner/analyse.py` |

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

The B3 → B4 roadmap's open questions are in `docs/ROADMAP.md` §7. From the B1–B3 list:

- **The Monte-Carlo half of B3**: the per-seed cross-check (D17) and the controlled
  phantom's fates (D34, D35) are done; the event-rate model check is future work.

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
