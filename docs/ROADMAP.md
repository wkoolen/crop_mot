# Roadmap B3 → B4: brief for the implementing agent

Written 2026-09-30 in a design session with the author (Wessel), who approved the step
order below. This file is the brief for an agent continuing the work in a fresh session: it
holds the context, the standing assumptions, the plan and its progress. **Update the
progress table (§8) at the end of every step.**

## 0. Starting a session

1. Read, in this order:
   - this file;
   - `docs/HANDOVER.md`: who reads the code, the hard constraints, the four invariants.
     All of it still applies;
   - the decision table in `docs/DECISIONS.md`, which wins over older text;
   - `docs/derivations/README.md` and `docs/derivations/A2.md`;
   - `crop_mot/filters/README.md`.
2. Run `git status`, `git log --oneline -10` and `python3 -m pytest tests -q -m "not slow"`.
3. Report to the author: the state of the tree, which step in §8 is next (§0a first, while
   it is open), and anything blocking it. **Start a step only after the author confirms it.** Several steps depend on
   derivations only the author can write.

**Precedence.** HANDOVER.md says "B4: do not implement". This roadmap lifts that one step at
a time; everything else in HANDOVER.md stands. When this file was written, the weeds work
(D15) was uncommitted on branch `b1-b3-implementation`. Check before assuming.

## 0a. Next iteration (priority): fix the B3 interface

The author made this the next iteration on 2026-09-30. Do it before anything else in §6.
It unblocks steps 2 and 3, and through them the whole validation chain.

**The problem.** The `r_sequence` docstring in `crop_mot/analysis/analytic.py` is now
written: the author drafted it, and it was corrected against A2 §2, §3.1, §4 and A0 on
2026-09-30. It needs two inputs that `ScanEvent` does not carry, and `build_scan_events`
cannot compute them with its current signature:

| Gap | Why the recursion needs it | Today |
|---|---|---|
| ℓ_i / κ_i for each gated detection | the one- and several-detection cases (A2 §3.1, A0) weigh "this object produced z_i" against "z_i is clutter" | only the count `n_gated` |
| which scan is the birth scan | the birth case sets r = r_birth there, and r = 0 before it | nothing marks it |
| the predicted covariance | gating and ℓ_i both need S = H P H^T + R | `build_scan_events` takes means only |

**Proposed changes.** Confirm the names and semantics with the author at the start of the
session, then implement.

1. **`ScanEvent.likelihood_ratios: tuple[float, ...] = ()`.** One entry per gated
   detection, in gate order: ℓ_i / κ_i = p_D N(z_i; ẑ, S) / (λ_FA c(z_i)).
   - Evaluate it with the filter's *assumed* sensor, at the filter's predicted moments and
     the scan's reported pose.
   - Use the shared helpers in `filters/kalman.py` (`predicted_measurement`,
     `log_predicted_likelihood`), as `filters/README.md` asks, so the linear algebra is
     never a suspect.
   - Invariant: `len(likelihood_ratios) == n_gated`, checked in `__post_init__`. Keep
     `n_gated` for readability.
2. **`ScanEvent.born: bool = False`.** True at exactly one scan: the first scan the track
   appears in the estimates log (`track_lifetimes`).
   - Take it from the log, not from the birth config, so it works for any birth model
     (seeds today, the step-8 map and the step-13 birth later).
   - Scans before the birth have `in_fov` False, `p_D` 0, `n_gated` 0, empty ratios and
     `born` False; the reference returns r = 0 there.
3. **`build_scan_events(detections_path, labels_path, cfg, track_moments_per_scan)`.**
   `track_moments_per_scan` is a list of `(mean, cov) | None` per scan, exactly what
   `predicted_track_moments` returns, replacing the means-only `track_mean_per_scan`.
   - The birth scan follows from the moments: it is the scan before the first non-None
     entry, which is the first scan the track is reported. If deriving it that way reads
     less clearly than passing it explicitly, pass it and say why.
   - For a pruned bank track, build the moments from the unpruned companion log (D14).
4. **Record it as a decision** (next free D-number), in the `ScanEvent` and
   `build_scan_events` docstrings, and in the commit message.

**Scope, stated in the decision.** B3 then checks the existence recursion *given* the
likelihoods. The Gaussian likelihood itself comes from the shared Kalman helpers that the
filter also uses, so B3 does not independently check it. That is deliberate: the Kalman
algebra is A1's, and `kalman.py` is its implementation.

**Then, in the same iteration:**

5. **Confirm the author has checked the `r_sequence` docstring against A2, line by line.**
   It was corrected in a session that had read `filters/bernoulli.py`, so the independence
   has to come from the author.
6. **Implement the body of `r_sequence` from its docstring only.** Do not open
   `filters/bernoulli.py` while doing so (§6 step 2).
7. **Continue with step 3:** `analyse` routing and the three B3 tests.

**Done when:**
- `ScanEvent` has both fields, with the invariant checked;
- `build_scan_events` is implemented with the new signature;
- a unit test checks `likelihood_ratios` against a hand-computed value for one small
  scan, and that `born` is True exactly once;
- `test_constant_profile_is_the_special_case_of_the_general_recursion` passes;
- all existing tests still pass.

## 1. Standing assumptions

- **Pose known (RTK-GPS).** The robot's absolute pose is known, so `path.pose_known: true`
  is part of the thesis scope, not a phase-1 shortcut (recorded as D16 in step 1).
  - With static plants, the problem becomes *mapping with known poses*. No filter carries
    a pose state, the motion model is the identity, and all uncertainty is in
    association, existence and clutter.
  - The robot's path only decides what is in view, through p_D(x, pose).
  - A bound to state alongside it: RTK gives position to about a centimetre, but heading
    comes from elsewhere. One degree of heading error moves a detection at 4 m by about
    7 cm, against σ = 0.2 m measurement noise.
  - The yaw-wobble slot in `world/path.py` stays, as a sensitivity experiment only.
- **Plants are static and permanent** (p_S = 1, `StaticTarget`).
- **The thesis scope is "N known".** There are N plants whose positions come from the
  planting plan, known to RTK precision. Weeds and clutter are unknown and come on top; to
  the filter, both are false positives (step 8).
- **"N unknown" is studied as the general case.** The known-N methods are limits of it,
  and it is the case where phantom deletion (A2 §5) becomes visible.

## 2. Three design rules

1. **One output format, so every figure works for every method.** Every filter reports
   `list[TrackEstimate]`, i.e. (r, mean, cov) per track; known-N filters report r = 1.
   Figures read only the run folder: the estimates log, plus truth for evaluation. Every
   new method therefore gets every existing figure, and no figure may depend on one
   filter's config layout.
2. **Every new method reproduces its parent.** Each new filter ships with at least one
   *reduction test*: under a restricting config it gives the same estimates as the simpler
   method, on the same run folder.
   - The precedent is D12: a one-seed unpruned bank equals the single Bernoulli filter
     (`tests/test_bernoulli_bank.py`).
   - The thesis claims the relationships in §3; these tests are the evidence for them.
3. **One branch table, combined in different ways.** For each track, the update builds the
   table of A2 §3.1: "missed", "z_i came from this track", "z_i is clutter". The methods
   differ only in how they combine that table (A2 §3.2):
   - GNN selects one branch;
   - PDA, JPDA, JIPDA and PMB average over the branches;
   - MHT and PMBM keep several hypotheses alive.

   The table is computed in one shared function (step 9) that every filter calls.

## 3. Method map

| | One object | Many objects, hard association | Many objects, averaged association | Many objects, several hypotheses kept |
|---|---|---|---|---|
| **N known** (r ≡ 1, no birth) | PDA | GNN (the baseline) | JPDA | MHT |
| **N unknown** (r, birth) | Bernoulli (= IPDA with moment matching) | GNN + M-of-N track logic | JIPDA ≈ PMB | PMBM |
| **N unknown, no identities** | | | PHD grid (the "heatmap") | |

Reductions to test:

- **Unknown N → known N** (pin r = 1, turn birth off): Bernoulli → PDA, PMB → JPDA,
  PMBM → MHT.
- **Many objects → one object:** JPDA with one track = PDA; PMB with one object = Bernoulli.
- **JPDA with no overlapping gates** = a bank of independent PDAs.
- **GNN with one track** = PDA with `KeepBestBranch`, up to ties. Both compare (1 − p_D)
  against p_D N(z_i; ẑ, S) / κ_i.
- **The PHD grid** is PMB's Poisson part.

"GNN + M-of-N" appears for completeness only; it is not planned.

## 4. Rules for every step

- **Done means:**
  - tests are green;
  - the step's named test or figure exists;
  - every choice made has a new row in the DECISIONS table (next free D-number);
  - docstrings are updated;
  - there is one commit whose message states the decision.

  Existing run outputs stay byte-identical unless a decision says otherwise; if they
  change, the decision says so.
- **Every new filter gets:**
  - a reduction test (§3);
  - a place in every figure (step 7's contract test enforces this);
  - a timing log (step 4c).
- **Math comes from the derivations.** Cite them as `[A2 §3.1]`. If a step needs an
  expression that isn't in a derivation, stop and ask the author; do not derive it
  yourself. This matters most for anything a B3-style check compares against: deriving it
  from the code makes the check a tautology.
- **Interface changes need the author's approval first.** Known candidates:
  - a new `ScanEvent` field (step 2);
  - a timing log written by the runner (step 4c);
  - a richer diagnostics channel for association weights (step 7).
- **Author-only items:** every `TODO(human)` and `TODO(Wessel)`. Never fill them in.

## 5. Context numbers

Back-of-envelope figures from 2026-09-30, for the `b1_two_rows*` scenarios.

- **Clutter intensity.** The FOV area is 9.546 m², so κ = λ_FA c(z) = 2 / 9.546 ≈ 0.21 per m².
- **Gate size.** The gate is χ² at 0.99 in 2D. Its radius is:
  - 0.61 m for a converged track (S ≈ R = 0.04 I);
  - 1.21 m for a newborn track (init_cov 0.12 I).

  Plants are 0.35 m apart within a row, so each detection falls inside three to four plant
  tracks' gates. The bank's independence approximation (D12) does not hold for the full
  field.
- **How existence changes per scan**, for a converged track, in log-odds:
  - about +1.78 for a detection;
  - about −1.90 for an in-view miss (odds × 0.15).

  The break-even true detection rate is about 52 %. Plants are seen about 78 % of the time
  (D11), so they confirm quickly. Weeds at weed p_D 0.5 hover (drift ≈ −0.06 per scan),
  consistent with D15's measurements.
- **Track count.** About 70 plants in the 12 m two-row field, so figures must handle tens
  of tracks, not five.
- **Heatmap lesson** (for step 14). Two ways to build a heatmap, each with a flaw:
  - An *occupancy grid* makes each cell its own Bernoulli. It handles misses well, but one
    detection raises every cell in its gate.
  - A *PHD* handles detections correctly: the denominator spreads one object's worth of
    mass. But a single miss cuts a confirmed object's mass to (1 − p_D) = 0.15, where a
    Bernoulli at r = 0.99 only drops to 0.94.

  PMB uses each where it is strong: the PHD for undetected objects, Bernoullis for
  detected ones.

## 6. Steps

### Phase 1: finish B3

**Step 1. Record the RTK assumption (D16).** Add a decision row and update the docstring of
`world/path.py::generate_path`. No behaviour change.

**Step 2. Turn A2 into `r_sequence`. AUTHOR ONLY.** This is the `TODO(human)` in
`crop_mot/analysis/analytic.py`. Point out two gaps in `ScanEvent` to the author:
- **Detection branch.** A2 §3.1's detection branch needs ℓ(z)/λ_FA(z), i.e.
  p_D·g(z_i)/κ(z_i), for each gated detection. `ScanEvent` carries only the count
  `n_gated`. Without a new field, the cross-check covers only miss, out-of-view and birth
  scans.
- **Birth branch.** No field says which scan the birth is on. The reference receives only
  `r_birth`, while `compare_r` compares from scan 0 and `r_trajectory` reports r = 0 before
  the birth. Either add a marker (e.g. a `born` flag) or say explicitly how the reference
  learns the birth scan.

Whether and how to add these fields is the author's call, because they are part of the
derivation's interface.

Also ask the author for the A2 rows of the derivation map (the `TODO(human)` in
`docs/derivations/README.md`). They make the B3 code's citations traceable.

The body of `r_sequence` is written from the author's branches, not from
`filters/bernoulli.py`. If the agent translates the author's written branches into code,
it must not consult the filter while doing so. Otherwise the cross-check becomes a
tautology.

**Step 3. Wire up the single-run cross-check.**
- Implement `build_scan_events` in `analysis/events.py`. **Fix its signature first:** it
  takes `track_mean_per_scan` (means only), but gating and the likelihood both need the
  predicted covariance too. Pass the (mean, cov) pairs that `predicted_track_moments`
  already returns. This is an interface change; raise it with the author.
- Route `r_vs_analytic` through `runner/analyse.py`, and write the comparison to
  `metrics.json`. While in `analyse`, remove its one-track, seeded-birth assumptions:
  - loop over all tracks instead of `_first_track_id`;
  - make the scene's scan an option instead of `cfg.filter_cfg.birth.at_scan`, which the
    step-8 planting map does not have.
- Fill in `test_r_matches_analytic_recursion` and
  `test_constant_profile_is_the_special_case_of_the_general_recursion`. They can be written
  before step 2 is done and fail until it is.
- Then run the check per track on the bank; each bank track is the A2 recursion.
  - **Compare pruned tracks on the unpruned companion log** (D14,
    `estimates_<kind>_unpruned.jsonl`). After a deletion, `r_trajectory` reports 0, but A2
    has no deletion step.
  - The unpruned log is identical up to the deletion. It also runs longer, so it covers
    more branches, including recapture (D9).

Done when both tests pass for both p_D profiles.

B3 is the root of the validation chain: every later method is checked by reduction tests
that end at the Bernoulli filter B3 validates. So keep `ScanEvent` single-track and
Bernoulli-specific; do not generalise it to many targets. Because events are built from
the filter's own predicted moments, the check does not depend on the collapse strategy.
After steps 9 and 10, rerun it as their regression test.

**Step 4. Monte-Carlo: per-seed cross-check (4a), model check (4b, future work), timing (4c).**

*Why the existing test must change.* `test_monte_carlo_mean_r_brackets_analytic`, as
written, compares the mean r over seeds with "the closed form". That doesn't hold in
general:
- r depends nonlinearly on the events, so the mean of r is not the closed form evaluated
  at the average events;
- each seed has its own event sequence, so there is no single closed form to compare with.

It only holds for a phantom that sees nothing but misses.

The author chose (a) as the B3 deliverable and (b) as future work (2026-09-30). Record
this as a decision and rework the test's name and docstring to match. The model question
that test 2 was meant to answer ("does the derivation match the simulator?") moves to 4b.

*This is plain Monte Carlo, not MCMC.* The trials are independent: trial i uses seed
base + i, and each trial is drawn directly from the simulator.
- Metropolis–Hastings and other MCMC methods build a *correlated* chain to sample a
  distribution you cannot sample directly. Here the simulator samples scenarios directly.
- Independence is what keeps the error bars simple: no burn-in, no autocorrelation.
- MCMC may appear later only inside PMBM, to sample association hypotheses (Gibbs sampling,
  mentioned in A2 §3).

**4a. Per-seed cross-check and robustness (the deliverable).**
- **First, restructure `run_monte_carlo` into a general trial loop.** Today it follows
  only the first track, keeps only r, and deletes each run folder before anything else can
  be measured.
  - The new loop: for each seed, simulate → run the filter(s) → apply a list of per-run
    measurement functions while the run folder still exists → aggregate.
  - 4a plugs in `compare_r`, branch coverage and fates; 4c plugs in timing; steps 5, 6 and
    16 plug in GOSPA, cardinality and phantom lifetime.

  This is the most important choice for keeping the code reusable. A B3-only loop would be
  rewritten three times.
- For each of n seeds: simulate, run the filter, and for every track build its events,
  evaluate the reference and run `compare_r`.
- Report:
  - the pass rate at the tolerance;
  - the maximum |error| over all seeds;
  - the seed and scan of any divergence;
  - **branch coverage**: how many scans of each kind were checked (birth, in-view miss,
    out-of-view, one gated detection, two or more).

  The sentence this produces for the thesis: "the implementation matched A2 to 1e-9 on X
  scans over n random trials, covering these branches".
- **Robustness over trials.** Record the fate of each hypothesis: pruned, confirmed on a
  plant, confirmed on a weed, or still alive at the end. Report each fate as a proportion
  with a binomial (Wilson) confidence interval. This turns the one-off 200-seed study in
  D15 into a command.
  - `ScanEvent.n_clutter_gated` lumps weed returns in with Poisson clutter, so it cannot
    decide "confirmed on a weed".
  - Classify fates with an analysis-side helper that reads `ScanLabels.weed_origin` and
    `origin`. Do not add a filter input for this.
- Keep `MonteCarloResult` (mean r ± standard error) as a descriptive figure, not a
  pass/fail test.
- **Open question.** For the fate statistics, every seed must run the same experiment. But
  `birth.seeds` picks detections by index, and a given index means something different in
  each seed (the `run_monte_carlo` docstring flags this). Propose a truth-blind fix to the
  author, e.g. a birth model at a configured position and scan.

**4b. Event-rate model check (future work: describe it in the thesis, implement later).**
Compare what the filter actually sees with what its assumed model predicts:
- the in-view miss rate of tracked plants against p_D;
- gated clutter per scan against λ_FA c(z) × the gate area;
- weed returns falling inside gates.

This answers "are the assumptions right?", which (a) does not. D11's finding is the kind of
mismatch it would show: detected/visible is 0.78 against p_D 0.85, because of edge
truncation.

**4c. Computation time and problem size.**
- **Timing log.** `run_filter` times `predict`, `update` and `extract` for each scan with
  `time.perf_counter`. It writes `timing_<log_name>.jsonl` with the fields k, predict_s,
  update_s, extract_s, n_detections and n_reported.
  - It is a separate file so the estimates logs stay deterministic and byte-identical.
  - Filter-internal sizes (components, gated pairs, hypotheses, cluster sizes) go through
    the existing `HasDiagnostics`.
- **Scaling command.** Sweep one scenario parameter over several values × seeds. Plot the
  median and 95th-percentile update time per scan against problem size. Draw the real-time
  budget (the scan period, 0.25 s) as a horizontal line. Sweep these separately:
  - row length (total N at fixed density);
  - plant spacing (cluster size);
  - λ_FA and weed density (detections per scan).

  Fit the log-log slope as the empirical exponent.
- **Hypothesis to test, not assume.** Cost per scan should follow the number of objects
  *in view* and the *cluster size* (how many tracks share gates), not total N. The current
  bank updates every component on every scan, so its cost grows with total N.
  - Skipping tracks with p_D(mean) = 0 is exact for the current Bernoulli update: every ℓ_i
    is 0, so r and the Gaussian are unchanged.
  - Add the skip only together with a test that the estimates stay byte-identical.
- **Expected cost per scan**, to check against (T tracks, M detections, n = T + M):

  | Method | Expected cost per scan |
  |---|---|
  | Bank | O(T·M) |
  | GNN (Hungarian) | O(n³), per cluster |
  | Exact JPDA | exponential in the cluster size |
  | Murty's k-best assignments | O(k·n³) |
  | PMBM | number of global hypotheses × the Murty cost |
  | PHD grid | O(G·M), for G grid cells |
- **Timing hygiene:**
  - run with single-threaded BLAS (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) set in
    the environment;
  - record the thread settings and the CPU model in `run_meta.json`;
  - report scan 0 separately, since it includes warm-up;
  - never assert on times in pytest; tests only check the shape of the log.
- Every filter added later gets its own curve on the same figure.

### Phase 2: the figure toolkit (before any new filter)

**Step 5. Figures that work for any filter.** Each figure reads only `(run, filter_name)`:
- **Tracks on the scene**, with ellipse opacity set by r. This generalises the hypotheses
  figure, whose one-row-per-track layout does not scale to about 70 tracks.
- **Existence map**: D(x) = Σ r_i N(x; m_i, P_i) on a grid. This is the PHD of any filter's
  output, so every method can be drawn as a heatmap. The PHD grid filter (step 14) draws
  its own grid in the same figure.
- **Cardinality**: Σ r_i against the true number of plants in view (from labels). It stays
  flat at N for known-N methods.
- **GOSPA over time**, split into localisation error, missed objects and false tracks.
  Implement the stub in `analysis/metrics.py`.
- **Phantom lifetime**: scans from birth to deletion, for methods that have births. This is
  the quantity A2 §5 is about.

**Step 6. A `compare` command** that runs several filters on one run folder and shows each
figure side by side. `track_all_filters` in `runner/track.py` already does the running.

**Step 7. A figure contract test**: every figure must render for every entry in `FILTERS`,
in the style of `test_filter_interface_contract.py`.

Optional, and it needs an interface decision: an association figure showing the track ×
detection weights at one scan. It is the most direct picture of how GNN, JPDA and PMB
differ. It needs a richer channel than `HasDiagnostics`, which only carries a flat
`dict[str, float]`.

### Phase 3: N known (the thesis scope)

**Step 8. The planting-plan prior.**
- **What:** a filter-side map of the N plants. The filter config holds:
  - the rows (x, y_start, y_end, spacing);
  - a prior position std that combines RTK and planting accuracy, e.g.
    √(0.02² + 0.03²) ≈ 0.036 m.
- **How:** `initial_state` holds N components with r = 1, and there is no birth.
- **Why this is truth-blind:** it works the same way as `assumed_sensor`. The map is the
  nominal plan the farmer knows, not the jittered positions in truth.jsonl.
- **Weeds and clutter** are not in the map, so to the filter they are false positives.

Expected behaviour to check:
- The prior (about 4 cm) is much tighter than the measurement noise (20 cm), so the
  problem is almost pure association.
- S is still dominated by R, so the gate is still about 0.6 m wide and neighbouring plants
  share detections. That is why step 12 exists.
- The failure mode is a weed inside a plant's gate. Measure how often it happens from the
  labels.

Future work, one line in the decision: the proposal says N is "bounded a priori", which
means *at most* N, e.g. a planned plant that never grew. That brings back an r per map
slot, and connects this case to the unknown-N half.

**Step 9. The shared branch table.** Move the per-track branch computation out of
`BernoulliFilter._update_existing` into one function (e.g. `filters/branches.py`). This is a
pure refactor: every existing estimates log must stay byte-identical.

**Step 10. Moment-matching collapse.** Add one class in `filters/collapse.py` and one line
in `COLLAPSE_STRATEGIES`. This needs the author's `TODO(Wessel)` in A2 §3.1 (the P_merge
formula) first.
- **Reduction test:** Bernoulli with moment matching and r = 1 equals PDA.
- **Citation note:** textbook PDA has a gate-probability factor P_G, which this filter sets
  to 1.

**Step 11. GNN** (`association/assignment.py`, `filters/gnn.py`), using scipy's
`linear_sum_assignment`. GNN is the baseline of the comparison.
- **Reduction test:** with one track, GNN equals Bernoulli/PDA with `KeepBestBranch`.
- **What that shows:** the current filter already hard-selects the position while still
  averaging r.

**Step 12. JPDA** (`filters/jpda.py`, `association/murty.py`). This needs the author's
derivation first; A2 §7 lists it as open.
- **Reduction tests:** one track equals PDA; tracks whose gates never overlap equal a bank
  of PDAs.
- **What the second test enables:** measuring D12's independence approximation. Run the
  bank and JPDA on the step-8 map and show where they diverge.

### Phase 4: N unknown

**Step 13. Birth from measurements** (A2 §4): r_b = e / (e + λ_FA c(z)), with
e = ∫ λ_u p_D g dx and λ_u taken from the config.
- It is a new `BirthModel` next to `SingleFromMeasurement`, as the latter's docstring
  already plans.
- Link to step 8: the planting plan is one choice of λ_u. N known is the limiting case
  where λ_u is N sharp bumps with r = 1.

**Step 14. PHD grid filter** (`filters/phd_grid.py`): the heatmap as an actual filter. It
has a closed form to check against, in the style of B3: a region that is in view with no
detection nearby loses exactly a factor (1 − p_D) per scan.

**Step 15. PMB** (`filters/pmb.py`). It combines:
- step 12's association step, with existence probabilities;
- step 14's grid as the Poisson part;
- step 13's birth.

This needs A3 first.
- **Reduction test:** r = 1 with no Poisson part equals JPDA.
- **Reduction test:** one object equals Bernoulli with moment matching.

**Step 16. PMBM, only if the data calls for it.** A2 §3.2 leaves open whether keeping
several hypotheses matters under this scope. Decide by comparing PMB with JPDA using steps
4c–7: phantom lifetime, GOSPA, and time per scan.

## 7. Open questions

Carry these until they are decided, then move each answer into DECISIONS.md.

1. The `ScanEvent` fields for the detection branch and the birth scan (step 2). The next
   iteration (§0a) proposes `likelihood_ratios` and `born`; confirm the names with the
   author.
2. Passing (mean, cov) instead of means to `build_scan_events` (step 3). This is part of
   §0a.
3. How to give every seed the same phantom birth, for the fate statistics (step 4a).
4. The timing log format: a separate file (recommended) or a field in the estimates log
   (step 4c).
5. The association figure, and the diagnostics channel it needs (step 7).
6. Map slots with r < 1, for "bounded" N (step 8, future work).
7. Whether PMBM is needed (step 16).

## 8. Progress

| Step | Status | Notes |
|---|---|---|
| **0a** | **next** | Priority: fix the B3 interface (`ScanEvent.likelihood_ratios`, `ScanEvent.born`, `build_scan_events` takes (mean, cov)), then implement `r_sequence` and step 3 |
| 1 | done | 2026-09-30: D16 recorded; `generate_path` and the `world/path.py` module docstring updated. No behaviour change |
| 2 | partial | 2026-09-30: `r_sequence` branches documented. The author drafted them and they were corrected against A2 §2/§3.1/§4 and A0; the author still has to check them line by line. Waiting on: the `ScanEvent` fields (§0a); the A2 rows in `docs/derivations/README.md`. The body still raises `NotImplementedError` |
| 3 | partial | `compare_r`, `plot_r_vs_analytic` and `plot_r_montecarlo` exist. `build_scan_events` and the three B3 test bodies still raise `NotImplementedError`. `build_scan_events` needs (mean, cov); `analyse` still assumes one track |
| 4 | partial | `run_monte_carlo` and `standard_error` exist, but `run_monte_carlo` must become the general trial loop (4a); 4a, 4b and 4c not started |
| 5 | partial | scene, counts, r_vs_k and hypotheses figures exist (shaped for the bank); `gospa` is a stub |
| 6 | not started | `track_all_filters` exists |
| 7 | not started | |
| 8 | not started | |
| 9 | not started | |
| 10 | not started | waits on the A2 §3.1 `TODO(Wessel)` |
| 11 | not started | `assignment.py` is a stub |
| 12 | not started | `murty.py` is a stub; waits on a JPDA derivation |
| 13 | not started | |
| 14 | not started | |
| 15 | not started | waits on A3 |
| 16 | not started | decided by the data from steps 4c–7 |
