# Roadmap B3 → B4: brief for the implementing agent

Written 2026-09-30 in a design session with the author (Wessel), who approved the step
order below. Revised the same day after an independent review; the author's decisions from
that review are reserved as D16 to D24 (§7b). This file is the brief for an agent continuing
the work in a fresh session: it holds the context, the standing assumptions, the plan and
its progress. **Update the progress table (§8) at the end of every step.**

**Purpose.** This roadmap builds the proof of concept for the graduation proposal. Its job
is to help the author choose the approach for phase 2, and for a real quadruped in a real
field. Every step should therefore produce evidence for that choice:

- accuracy: GOSPA, association accuracy, missing plants found;
- consistency: NEES, i.e. whether the reported covariances can be trusted;
- cost per scan, against the scan period of 0.25 s;
- how hard the method is to validate (does it reduce to something already checked?).

A step that feeds none of these is a candidate to drop; raise it with the author.

## 0. Starting a session

1. Read, in this order:
   - this file;
   - `docs/HANDOVER.md`: who reads the code, the hard constraints, the four invariants.
     All of it still applies;
   - the decision table in `docs/DECISIONS.md`, which wins over older text;
   - `docs/derivations/README.md` and `docs/derivations/A2.md`;
   - `crop_mot/filters/README.md`.
2. Run `git status`, `git log --oneline -10` and `python3 -m pytest tests -q -m "not slow"`.
3. Report to the author: the state of the tree, which step in §8 is next, and anything
   blocking it. **Start a step only after the author confirms it.** Several steps depend on
   derivations only the author can write.

**Precedence.** HANDOVER.md says "B4: do not implement". This roadmap lifts that one step at
a time; everything else in HANDOVER.md stands. When this file was written, the weeds work
(D15) was uncommitted on branch `b1-b3-implementation`. Check before assuming.

## 1. Standing assumptions

- **Pose known (RTK-GPS), as a simplification for now** (D16). The robot's absolute pose is
  known, so `path.pose_known: true`. Whether the Go2 setup will have RTK is not yet
  confirmed; if it does not, this assumption is revisited, not silently kept.
  - With static plants, the problem becomes *mapping with known poses*. No filter carries
    a pose state, the motion model is the identity, and all uncertainty is in
    association, existence and clutter.
  - The robot's path only decides what is in view, through p_D(x, pose).
  - A bound to state alongside it: RTK gives position to about a centimetre, but heading
    comes from elsewhere. One degree of heading error moves a detection at 4 m by about
    7 cm, against σ = 0.2 m measurement noise.
  - The size of that error is not the main risk; its correlation is. A heading bias moves
    every detection in a scan the same way, and a slowly varying bias does not average
    out over scans. The 3.6 cm map prior (step 8a) is already tighter than 7 cm, so a
    filter that ignores heading error becomes overconfident. Step 8d stubs the
    experiment that measures this with NEES.
- **Plants are static and permanent** (p_S = 1, `StaticTarget`).
- **The thesis scope is "N bounded by the planting plan".** Each planned slot holds a plant
  with prior probability r_0, at a position known to RTK precision (step 8c). "N known"
  (r ≡ 1) is the limiting case: it is built first (step 8a) and it anchors the reduction
  tests (D23).
- **Weeds are labelled by the vision model** (D22). The detector will report a class label,
  plant or weed. Plant tracks do not associate weed-labelled detections, even inside their
  gate. The detector is stubbed as a confusion matrix (step 8b), so the false positives a
  plant filter sees are Poisson clutter and weeds that were labelled "plant".
  - Weed-labelled detections are kept, not thrown away: weeds may get their own map later.
    That map is an "N unknown" problem (no plan for weeds), so it is the natural use of the
    Phase 4 machinery.
- **"N unknown" is studied as the general case.** The known-N and bounded-N methods are
  limits of it, and it is the case where phantom deletion (A2 §5) becomes visible.

## 2. Three design rules

1. **One output format, so every figure works for every method.** Every filter reports
   `list[TrackEstimate]`, i.e. (r, mean, cov) per track; known-N filters report r = 1.
   Figures read only the run folder: the estimates log, plus truth for evaluation. Every
   new method therefore gets every existing figure, and no figure may depend on one
   filter's config layout.
   - Methods that keep several hypotheses (MHT, PMBM) need a rule for producing this list:
     the best global hypothesis, or a marginal over hypotheses. The rule changes every
     comparison, so it is a decision per method, recorded when the method is added.
   - A figure may declare a filter *not applicable*: phantom lifetime for methods without
     births, per-track figures for the PHD grid (it has no identities). It then renders a
     labelled placeholder panel. The contract test (step 7) accepts a placeholder, not an
     exception.
2. **Every new method reproduces its parent.** Each new filter ships with at least one
   *reduction test*: under a restricting config it gives the same estimates as the simpler
   method, on the same run folder. "The same" means equal within an absolute tolerance of
   1e-12 on every r, mean and covariance entry (D18).
   - The precedent is D12: a one-seed unpruned bank equals the single Bernoulli filter
     (`tests/test_bernoulli_bank.py`).
   - The thesis claims the relationships in §3; these tests are the evidence for them.
3. **One branch table, combined in different ways.** For each track, the update builds the
   table of A2 §3.1: "does not exist", "exists but missed", "z_i came from this track",
   "z_i is clutter", and, for methods with a Poisson part, "z_i is a new object". The
   methods differ only in how they combine that table (A2 §3.2):
   - GNN selects one branch;
   - PDA, JPDA, JIPDA and PMB average over the branches;
   - MHT and PMBM keep several hypotheses alive.

   The table is computed in one shared function (step 9) that every filter calls. Its
   structure is fixed in step 9 *before* any method that needs it is written, so it is not
   refactored again.

## 3. Method map

| | One object | Many objects, hard association | Many objects, averaged association | Many objects, several hypotheses kept |
|---|---|---|---|---|
| **N known** (r ≡ 1, no birth) | PDA | GNN (the baseline) | JPDA | MHT (fixed set of tracks) |
| **N bounded** (r per slot, no birth) | Bernoulli per slot | | JIPDA without birth | |
| **N unknown** (r, birth) | Bernoulli (= IPDA with moment matching) | GNN + M-of-N track logic | JIPDA ≈ PMB | PMBM |
| **N unknown, no identities** | | | PHD grid (the "heatmap") | |

Reductions to test:

- **Unknown N → known N** (pin r = 1, turn birth off): Bernoulli → PDA, PMB → JPDA,
  PMBM → MHT. "MHT" here means MHT over a fixed set of tracks, without track initiation.
- **Bounded N → known N** (r_0 = 1): the slot bank of step 8c → step 8a; JIPDA → JPDA.
- **Many objects → one object:** JPDA with one track = PDA. PMB with one object and zero
  Poisson intensity = Bernoulli with moment matching; the Bernoullis PMB creates from
  detections then have r = 0 and must be dropped, not kept.
- **JPDA with no overlapping gates** = a bank of independent PDAs.
- **GNN with one track** = PDA with `KeepBestBranch`, up to ties. This holds only for the
  GNN of step 11, whose cost matrix carries the same likelihood ratios: detection cost
  −ln(p_D g(z_i)/κ_i), and one missed-detection column per track with cost −ln(1 − p_D).
  A plain distance-based GNN does not reduce.
- **The PHD grid without its detection term is PMB's Poisson part.** PMB updates the
  undetected-object intensity only with the miss factor, D⁺(x) = (1 − p_D(x)) D(x),
  because each detection creates a new Bernoulli instead. The full PHD grid agrees with it
  only on scans with no detections.

**Association marginals in reduction tests** (D19). Every reduction test that involves
association marginals (JPDA, JIPDA, PMB) computes them by exact enumeration, on scenarios
whose clusters are small enough for that. Approximate methods (Murty k-best, loopy belief
propagation) are tested separately against exact enumeration on small clusters, with a
tolerance that is its own decision.

"GNN + M-of-N" appears for completeness only; it is not planned.

## 4. Rules for every step

- **Done means:**
  - tests are green;
  - the step's named test or figure exists;
  - every choice made has a new row in the DECISIONS table (use the reserved number in
    §7b if there is one, otherwise the next free D-number);
  - docstrings are updated;
  - there is one commit whose message states the decision.

  Existing run outputs stay equal within 1e-12 absolute (D18) unless a decision says
  otherwise; if they change, the decision says so.
- **Tests written before their code exists** are marked
  `@pytest.mark.xfail(strict=True, reason="waits on step N")` (D24). The suite stays green,
  and strict mode fails the moment the test starts passing, so the marker is removed in
  the commit that makes it pass.
- **Every new filter gets:**
  - a reduction test (§3);
  - a place in every figure (step 7's contract test enforces this);
  - a timing log (step 4c).
- **Math comes from the derivations.** Cite them as `[A2 §3.1]`. If a step needs an
  expression that isn't in a derivation, stop and ask the author; do not derive it
  yourself. This matters most for anything a B3-style check compares against: deriving it
  from the code makes the check a tautology.
- **Interface changes need the author's approval first.** Known candidates:
  - new `ScanEvent` fields (step 2);
  - (mean, cov) into `build_scan_events` (step 3);
  - the p_D evaluation strategy (step 3b);
  - a timing log written by the runner (step 4c);
  - a richer diagnostics channel for association weights (step 7);
  - a class label on detections, and an assumed confusion matrix (step 8b);
  - a missing-plant rate in the scenario, and r_0 in the filter config (step 8c);
  - a constant yaw-bias slot in `world/path.py` (step 8d).
- **Author-only items:** every `TODO(human)` and `TODO(Wessel)`. Never fill them in.

## 5. Context numbers

Back-of-envelope figures from 2026-09-30, for the `b1_two_rows*` scenarios.

- **Clutter intensity.** The FOV area is 9.546 m², so κ = λ_FA c(z) = 2 / 9.546 ≈ 0.21 per m².
- **Gate size.** The gate is χ² at 0.99 in 2D (threshold 9.21). Its radius is:
  - 0.61 m for a converged track (S ≈ R = 0.04 I);
  - 1.21 m for a newborn track (init_cov 0.12 I, so S = 0.12 I + R = 0.16 I).

  Plants are 0.35 m apart within a row, so each detection falls inside three to four plant
  tracks' gates. The bank's independence approximation (D12) does not hold for the full
  field.
- **How existence changes per scan**, for a converged track, in log-odds. The first two
  lines are the simple estimate; the rest add what falls in the gate. Filter p_D = 0.85
  evaluated at the mean. Monte Carlo over 10⁵ scans per line, in-row neighbours only:

  | Situation | Mean log-odds change per scan |
  |---|---|
  | detection, gate otherwise empty | about +1.8 |
  | in-view miss, gate empty | −1.90 (odds × 0.15) |
  | in-view miss, Poisson clutter in the gate (0.24 returns per gate on average) | about −1.33 |
  | weed at weed p_D 0.5, gate empty | about −0.04 |
  | weed at weed p_D 0.5, with clutter in the gate | about +0.30 |
  | **empty map slot** between plants at ±0.35 m (seen 78 %), with clutter, bank | **about +1.29** |

  Consequences:
  - The break-even true detection rate is about 52 % with an empty gate, and about 42 %
    once clutter is counted. Plants are seen about 78 % of the time (D11), so they confirm
    quickly either way.
  - Weeds hover only if the gate is empty. With clutter they drift up. This disagrees with
    D15's measurement ("hover"). Before relying on either, check whether D15's weed tracks
    had a larger covariance or a different clutter density (open question 12). With class
    labels (D22), this line only applies to weeds labelled "plant".
  - **An empty slot is kept alive by its neighbours.** Under the bank's independence
    approximation, the detections of the two neighbour plants fall in the empty slot's gate
    and raise its r by about 1.3 per scan. The bank therefore cannot find missing plants at
    0.35 m spacing. JIPDA (step 12 with existence) should, because it lets each neighbour's
    detection be explained by the neighbour. This is the key comparison of step 8c.
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
`world/path.py::generate_path`. No behaviour change. The row says: known pose is a
simplification for now, confirmed by the author on 2026-09-30; it is revisited if RTK is not
available on the Go2 setup; heading error is studied in step 8d.

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
- `compare_r` compares with an **absolute** error on r (D18). r saturates near 1, so a
  relative error on 1 − r, or a log-odds error, would blow up without meaning anything.
- Route `r_vs_analytic` through `runner/analyse.py`, and write the comparison to
  `metrics.json`. While in `analyse`, remove its one-track, seeded-birth assumptions:
  - loop over all tracks instead of `_first_track_id`;
  - make the scene's scan an option instead of `cfg.filter_cfg.birth.at_scan`, which the
    step-8a planting map does not have.
- Fill in `test_r_matches_analytic_recursion` and
  `test_constant_profile_is_the_special_case_of_the_general_recursion`. They can be written
  before step 2 is done; mark them `xfail(strict=True)` until it is (D24).
- Then run the check per track on the bank; each bank track is the A2 recursion.
  - **Compare pruned tracks on the unpruned companion log** (D14,
    `estimates_<kind>_unpruned.jsonl`). After a deletion, `r_trajectory` reports 0, but A2
    has no deletion step.
  - The unpruned log is identical up to the deletion. It also runs longer, so it covers
    more branches, including recapture (D9).

Done when both tests pass for both p_D profiles.

**What this check shows, and what it does not.** The events are built from the filter's own
predicted moments and its own ℓ/κ. So the check verifies the *existence recursion*: given
the filter's moments, r follows A2. An error in the mean/cov update, in gating or in how p_D
is evaluated is invisible to it, because both sides see the same numbers. The thesis
sentence is therefore "the existence recursion matches A2, given the filter's predicted
moments". The Gaussian part is checked separately with NEES (step 5).

B3 is the root of the validation chain: every later method is checked by reduction tests
that end at the Bernoulli filter B3 validates. So keep `ScanEvent` single-track and
Bernoulli-specific; do not generalise it to many targets. Because events are built from
the filter's own predicted moments, the check does not depend on the collapse strategy.
After steps 9 and 10, rerun it as their regression test.

**Step 3b. Stub the p_D evaluation strategy.** p_D depends on the state (it drops to 0 at
the FOV edges). The exact branches are then:
- detection: ∫ p_D(x) g(z | x) N(x; m, P) dx;
- missed: weight 1 − p̄_D with p̄_D = ∫ p_D(x) N(x; m, P) dx, and density
  (1 − p_D(x)) N(x; m, P) / (1 − p̄_D), which is not Gaussian.

The filter most likely uses p_D(m). Which form A2 claims is the author's call (open
question 3). The agent adds a small strategy interface, e.g. `filters/detection_prob.py`
with a `PdEvaluation` protocol, and these options:

| Option | Detection branch | Miss weight | Miss density | Cost | Where it is wrong |
|---|---|---|---|---|---|
| A. `AtMean` (current, default) | p_D(m) | 1 − p_D(m) | unchanged N(m, P) | none | near the FOV edge: a track whose mean is just outside gets no update, even with most of its mass inside |
| B. `AtBranchMean` | p_D at the Kalman-updated mean of that branch | as A | as A | small | as A for misses; the detection itself says where the object is, so the detection branch improves |
| C. `Expected` | p̄_D from sigma points (5 points in 2D) | 1 − p̄_D (exact) | unchanged N(m, P) | 5 FOV evaluations per track | only the miss density shape |
| D. `ExpectedWithShift` | as C | as C | (1 − p_D(x)) N(x) moment-matched with sigma points: the mean moves away from the FOV ("negative information") | as C | closest to exact; needs a derivation first |

- Implement A by wrapping the current code; all outputs stay within 1e-12 (D18).
- B, C and D are stubs that raise `NotImplementedError`, each with a docstring that names
  the derivation it waits for.
- Whatever is chosen, `ScanEvent` should carry the p_D value(s) the filter used, so the
  A2 cross-check keeps testing the combination formula and not the p_D approximation.
- Where it matters: D11's edge truncation (detected/visible 0.78 against p_D 0.85) is
  exactly the region where A and C differ. With the 3.6 cm map prior, P is small, so away
  from the edges all four agree.

**Step 4. Monte-Carlo: per-seed cross-check (4a), model check (4b, future work), timing (4c).**

*Why the existing test must change.* `test_monte_carlo_mean_r_brackets_analytic`, as
written, compares the mean r over seeds with "the closed form". That doesn't hold in
general:
- r depends nonlinearly on the events, so the mean of r is not the closed form evaluated
  at the average events;
- each seed has its own event sequence, so there is no single closed form to compare with.

It only holds for a phantom that sees nothing but misses.

The author chose (a) as the B3 deliverable and (b) as future work (2026-09-30, D17). Rework
the test's name and docstring to match. The model question that test 2 was meant to answer
("does the derivation match the simulator?") moves to 4b.

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
    16 plug in GOSPA, cardinality, NEES and phantom lifetime.

  This is the most important choice for keeping the code reusable. A B3-only loop would be
  rewritten three times.
- For each of n seeds: simulate, run the filter, and for every track build its events,
  evaluate the reference and run `compare_r`.
- Report:
  - the pass rate at the tolerance;
  - the maximum absolute error over all seeds;
  - the seed and scan of any divergence;
  - **branch coverage**: how many scans of each kind were checked (birth, in-view miss,
    out-of-view, one gated detection, two or more).

  The sentence this produces for the thesis: "the existence recursion matched A2 to 1e-12
  on X scans over n random trials, covering these branches".
- **Robustness over trials: one outcome per seed.** Tracks in the same seed share clutter
  and neighbours, so they are not independent trials. Pooling them would make the
  confidence intervals too narrow. The rule: every seed contributes one number per
  quantity, and intervals are computed across seeds.
  - **Phantoms.** Each seed gets one *controlled phantom injection*: a hypothesis placed at
    a configured empty position and scan, with a configured r. It is an experiment
    setting, not a `BirthModel`: there is no detection behind it. Its fate is the seed's
    one outcome, reported as a proportion with a Wilson interval.
  - **Clutter or a weed near the phantom** is part of the random scene; averaging over seeds
    is what covers it. To see its effect, *stratify*: at analysis time, from truth, record
    how many clutter returns and weeds fell inside the phantom's gate over its life, and
    report the fates per stratum (e.g. "weed in gate: yes / no"), each with its own
    interval. To *control* it instead, place the injection at a configured distance d from
    the nearest weed and sweep d.
  - **Plants.** Per seed, compute a proportion over its plants, e.g. the fraction of in-view
    plants confirmed by scan k, or, among plants with a weed or clutter return in their
    gate, the fraction pulled more than x cm away. That proportion is the seed's outcome;
    report the mean over seeds with a t-interval.
  - **Fate categories:** pruned; confirmed on a plant; confirmed on a weed; alive and
    sustained by clutter (no plant or weed within d_match, r above r_conf); alive and
    unconfirmed. The thresholds r_conf and d_match are decisions (open question 4).
  - `ScanEvent.n_clutter_gated` lumps weed returns in with Poisson clutter, so it cannot
    decide "confirmed on a weed". Classify fates with an analysis-side helper that reads
    `ScanLabels.weed_origin` and `origin`. Do not add a filter input for this.
  - With class labels (step 8b), "confirmed on a weed" can only happen through a weed
    labelled "plant"; report it against the confusion rate.
- Keep `MonteCarloResult` (mean r ± standard error) as a descriptive figure, not a
  pass/fail test.

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
  - It is a separate file so the estimates logs stay deterministic.
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
  - Skipping tracks with p_D(mean) = 0 is exact in exact arithmetic for the current
    Bernoulli update: every ℓ_i is 0, so r and the Gaussian are unchanged. In floating
    point the current code computes r / ((1 − r) + r), and (1 − r) + r is not always
    exactly 1.0, so the skip may change the last bits.
  - Add the skip only together with a test that the estimates stay within 1e-12 (D18).
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
- **Cardinality, restricted to the view** (D21): Σ r_i over the tracks whose mean is in
  view, against the number of true plants in view (from labels). "In view" is decided by
  the same FOV function on both sides. For known-N methods the two curves differ only where
  the map and the field differ, e.g. a missing plant in step 8c.
- **GOSPA over time**, split into localisation error, missed objects and false tracks.
  Implement the stub in `analysis/metrics.py`.
- **NEES over time.** For each confirmed track matched to a true plant, e = m − x_true and
  NEES = eᵀ P⁻¹ e. Plot the average over tracks with the 95 % band of χ²₂ averaged over
  that many tracks. Known-N and bounded-N methods match track i to planned plant i; the
  other methods use the GOSPA assignment. This is the check on the Gaussian part that the
  A2 cross-check cannot give (step 3), and the measurement step 8d needs.
- **Association accuracy** (proposed; the author confirms or drops it): per scan, the
  fraction of plant-origin detections whose largest association weight goes to their own
  plant's track, and the fraction of clutter and weed returns that go to no track. With a
  4 cm prior, known-N is almost pure association, so this separates methods where GOSPA
  barely can. It needs the association weights, i.e. the richer channel of step 7 (open
  question 6).
- **Phantom lifetime**: scans from birth to deletion, for methods that have births;
  "not applicable" for the rest (§2 rule 1). This is the quantity A2 §5 is about.

**Step 6. A `compare` command** that runs several filters on one run folder and shows each
figure side by side. `track_all_filters` in `runner/track.py` already does the running.

**Step 7. A figure contract test**: every figure must render for every entry in `FILTERS`,
in the style of `test_filter_interface_contract.py`. A labelled "not applicable" panel
counts as rendering; an exception does not.

Optional, and it needs an interface decision: an association figure showing the track ×
detection weights at one scan. It is the most direct picture of how GNN, JPDA and PMB
differ. It needs a richer channel than `HasDiagnostics`, which only carries a flat
`dict[str, float]`. The association-accuracy metric of step 5 needs the same channel.

### Phase 3: N known, then N bounded (the thesis scope)

**Step 8a. The planting-plan prior.**
- **What:** a filter-side map of the N plants. The filter config holds:
  - the rows (x, y_start, y_end, spacing);
  - a prior position std that combines RTK and planting accuracy, e.g.
    √(0.02² + 0.03²) ≈ 0.036 m.
- **How:** `initial_state` holds N components with r = 1, and there is no birth.
- **Why this is truth-blind:** it works the same way as `assumed_sensor`. The map is the
  nominal plan the farmer knows, not the jittered positions in truth.jsonl.
- **Clutter** is not in the map, so to the filter it is a false positive. Weeds are handled
  by their label from step 8b on; before 8b exists, weed returns act as clutter.

Expected behaviour to check:
- The prior (about 4 cm) is much tighter than the measurement noise (20 cm), so the
  problem is almost pure association.
- S is still dominated by R, so the gate is still about 0.6 m wide and neighbouring plants
  share detections. That is why step 12 exists.
- The failure modes are a clutter return or a mislabelled weed inside a plant's gate, and a
  neighbour's detection pulling a plant. Measure how often each happens from the labels.

**Step 8b. Class labels on detections (stubbed classifier).**
- **World side:** each detection gets a `label` in {plant, weed}, drawn from a confusion
  matrix C[origin][label] in the scenario config, with rows for plant, weed and clutter.
  The default is the perfect classifier (identity for plant and weed; clutter labelled
  "plant", so it still reaches plant tracks).
- **Filter side:** the filter sees the label and an *assumed* confusion matrix, the same
  way it sees `assumed_sensor`; never the truth.
  - **Perfect classifier (first version):** plant tracks drop weed-labelled detections
    before the update. No new math.
  - **Imperfect classifier (later):** the label enters the branch table as a factor,
    ℓ_i = p_D g(z_i) P(label_i | plant) and κ_i = λ_FA c(z_i) P(label_i | clutter) plus a
    term for weeds labelled "plant". That weed term is persistent in space, not Poisson,
    so treating it as part of κ is an approximation. Needs the author's derivation first.
- **Weed-labelled detections** are written to the run folder, not used by the plant
  filter. A weed map built from them is future work (Phase 4 machinery, see §1).
- **Reduction test:** with a scenario without weeds, 8b gives the same estimates as 8a
  (D18).
- Interface change (a new detection field, a new config block): the author approves first.

**Step 8c. Bounded N: missing plants** (moved forward from future work, D23).
- **What:** each map slot is a Bernoulli with prior r_0 < 1, e.g. the emergence rate of
  the crop, from the filter config. No birth. The Gaussian comes from the plan as in 8a.
- **World side:** a scenario parameter `p_missing`: the probability that a planned slot has
  no plant. Truth records which slots are empty; the filter does not see it.
- **Output:** r per slot over time. Metric: how well "r < threshold" finds the truly empty
  slots (precision and recall, or a ROC curve over the threshold), and how many scans it
  takes to decide.
- **Reduction test:** r_0 = 1 gives 8a.
- **Expected result, to confirm:** on the bank, an empty slot between two plants at 0.35 m
  is kept alive by its neighbours' detections (about +1.3 log-odds per scan, §5). So the
  bank should fail here, and JIPDA (step 12 with existence) should not. Run the bank first
  and show the failure; this is the main motivation for step 12 and a central result for
  the proposal.
- Open: the value of r_0, and how p_missing is placed (independent per slot, or in runs of
  neighbouring slots). Open question 8.

**Step 8d. Stub the yaw experiment.**
- **What:** a command skeleton, e.g. `experiments/yaw_sensitivity.py`, that sweeps a
  constant yaw bias b ∈ {0, 0.5, 1, 2}° and the existing yaw-wobble amplitude, while the
  filter keeps assuming the pose is known. It reports NEES, GOSPA and (if confirmed)
  association accuracy against b.
- **Expected:** NEES leaves the χ² band once b × range exceeds the posterior std, which for
  the 3.6 cm prior is already below 1° at 4 m.
- **Stub only for now:** the config block and the command exist, the body raises
  `NotImplementedError` with a docstring that says it waits for NEES (step 5). One test
  checks that the config parses.
- A constant yaw-bias slot in `world/path.py` is an interface change (open question 9);
  the wobble slot already exists.

**Step 9. The shared branch table.** Move the per-track branch computation out of
`BernoulliFilter._update_existing` into one function (e.g. `filters/branches.py`). This is a
pure refactor: every existing estimates log must stay within 1e-12 (D18).

Fix the structure now, for every later method, so the table is not refactored again. For
one scan, with tracks j = 1..T and detections i = 1..M, all weights in log space:

| Field | Shape | Meaning |
|---|---|---|
| `log_w_absent` | T | ln(1 − r_j): the track does not exist; −∞ when r ≡ 1 |
| `log_w_missed` | T | ln(r_j (1 − p_D,j)): exists but not detected, with p_D from step 3b |
| `log_w_det` | T × M | ln(r_j p_D,j g_j(z_i)), plus ln P(label_i \| plant) once 8b's imperfect classifier exists; −∞ outside the gate |
| `gated` | T × M | boolean gate mask |
| `log_kappa` | M | ln κ(z_i), plus the label factor from 8b |
| `log_w_new` | M | ln e(z_i), the "new object" weight from the Poisson part (steps 13 and 15); −∞ when there is none |
| `det_moments` | T × M | Kalman-updated (mean, cov) per gated pair |
| `missed_moments` | T | (mean, cov) of the missed branch (unchanged for step 3b options A to C) |

- The Bernoulli update is one way to combine it: the existence weight is
  exp(`log_w_missed`) + Σ_i exp(`log_w_det` − `log_kappa`), against exp(`log_w_absent`).
- Splitting "does not exist" from "exists but missed" matters: the current Bernoulli update
  sums them, but JIPDA and PMB need them apart.
- Only the columns the Bernoulli filter uses are filled in step 9; `log_w_new` is −∞ and
  the label factor is 0 until steps 8b and 13 fill them.

**Step 10. Moment-matching collapse.** Add one class in `filters/collapse.py` and one line
in `COLLAPSE_STRATEGIES`. This needs the author's `TODO(Wessel)` in A2 §3.1 (the P_merge
formula) first.
- **Reduction test:** Bernoulli with moment matching and r = 1 equals PDA.
- **Citation note:** textbook PDA has a gate-probability factor P_G, which this filter sets
  to 1.

**Step 11. GNN** (`association/assignment.py`, `filters/gnn.py`), using scipy's
`linear_sum_assignment`. GNN is the baseline of the comparison.
- **Cost matrix** (D20): T rows; M detection columns with cost −ln(p_D g_j(z_i)/κ_i)
  (from `log_w_det` − `log_kappa`, infinite outside the gate), plus T missed-detection
  columns where entry (j, j) costs −ln(1 − p_D,j) and the other entries are infinite.
  Without the missed columns, GNN cannot leave a track unassigned, and the reduction fails.
- **Reduction test:** with one track and r pinned to 1, GNN equals PDA with
  `KeepBestBranch`, up to ties. (The Bernoulli with `KeepBestBranch` and free r does not
  match: it still averages r.)
- **What that shows:** the current filter already hard-selects the position while still
  averaging r.

**Step 12. JPDA and JIPDA** (`filters/jpda.py`, `association/murty.py`). This needs the
author's derivation first; A2 §7 lists it as open. Ask for it with existence included, so
the same code covers N bounded (JIPDA without birth, for step 8c).
- **Marginals:** exact enumeration for small clusters; Murty k-best for large ones. The
  reduction tests use exact enumeration only (D19). Murty is tested against exact
  enumeration on small clusters.
- **Reduction tests:** one track equals PDA; tracks whose gates never overlap equal a bank
  of PDAs; r ≡ 1 turns JIPDA into JPDA.
- **What the second test enables:** measuring D12's independence approximation. Run the
  bank and JPDA on the step-8a map, and the bank and JIPDA on the step-8c map, and show
  where they diverge.

### Phase 4: N unknown

**Step 13. Birth from measurements** (A2 §4): r_b = e / (e + λ_FA c(z)), with
e = ∫ λ_u p_D g dx and λ_u taken from the config.
- It is a new `BirthModel` next to `SingleFromMeasurement`, as the latter's docstring
  already plans. It fills `log_w_new` in the branch table.
- Link to steps 8a and 8c: the planting plan is one choice of λ_u. N known is the limiting
  case where λ_u is N sharp bumps with r = 1; N bounded is the same bumps with r = r_0.
- It is also what a future weed map (§1) would use, since weeds have no plan.

**Step 14. PHD grid filter** (`filters/phd_grid.py`): the heatmap as an actual filter. It
has a closed form to check against, in the style of B3: a region that is in view with no
detection nearby loses exactly a factor (1 − p_D) per scan. Its per-track figures are "not
applicable" (§2 rule 1).

**Step 15. PMB** (`filters/pmb.py`). It combines:
- step 12's association step, with existence probabilities;
- step 14's grid, *without its detection term*, as the Poisson part (§3);
- step 13's birth.

This needs A3 first.
- **Reduction test:** r = 1 with no Poisson part equals JPDA, with marginals by exact
  enumeration on both sides (D19). If PMB uses loopy belief propagation in normal runs,
  test that separately against exact enumeration on small clusters.
- **Reduction test:** one object and zero Poisson intensity equals Bernoulli with moment
  matching; the r = 0 Bernoullis created from detections are dropped.

**Step 16. PMBM, only if the data calls for it.** A2 §3.2 leaves open whether keeping
several hypotheses matters under this scope. Decide by comparing PMB with JPDA using steps
4c–7: phantom lifetime, GOSPA, NEES, and time per scan. This comparison is one of the
proposal's outcomes. If PMBM is built:
- **Reduction test:** r = 1, no birth, no Poisson part equals MHT over a fixed set of tracks.
- Decide and record its output rule (best global hypothesis or marginal, §2 rule 1).

## 7. Open questions

Carry these until they are decided, then move each answer into DECISIONS.md.

1. ~~The `ScanEvent` fields for the detection branch and the birth scan (step 2).~~
   Decided: `likelihood_ratios` and `born` (D25).
2. ~~Passing (mean, cov) instead of means to `build_scan_events` (step 3).~~ Decided:
   approved by the author on 2026-09-30 (D25).
3. ~~Which p_D form A2 claims, and which option (A to D) the filter uses (step 3b).~~
   Decided: A2 claims p_D_bar; the filter keeps option A, C is the next brick (D27).
4. ~~The fate thresholds r_conf and d_match, and the injection position and scan for the
   controlled phantom (step 4a).~~ Decided: r_conf = 0.5, d_match = 0.2 m, D8's phantom
   (D28).
5. ~~The timing log format: a separate file (recommended) or a field in the estimates log
   (step 4c).~~ Decided: a separate file (D29).
6. ~~The association figure and association-accuracy metric, and the diagnostics channel
   both need (steps 5 and 7).~~ Decided: deferred to step 11 (D30).

Questions 3 to 6 were answered by Claude at the author's request on 2026-09-30 ("keep
everything as documented and build complexity brick by brick"); 7 to 12 stay open until
their steps.
7. ~~The detection `label` field, the confusion matrix config~~ (decided, D22), and the
   derivation for the imperfect classifier (step 8b): still open, the author's.
8. ~~The prior r_0 per slot, and how the simulator places missing plants (step 8c).~~
   Decided: r_0 = 1 - p_missing = 0.9, independent per slot (D23).
9. A constant yaw-bias slot in `world/path.py` (step 8d).
10. The output rule for MHT and PMBM: best global hypothesis or marginal (§2, step 16).
11. Whether PMBM is needed (step 16).
12. Weed drift: D15 measured "hover", §5's estimate with gated clutter gives about +0.30
    per scan. Find out why they differ.

## 7b. Decisions reserved on 2026-09-30

The author decided these in the review of 2026-09-30. Write each row into DECISIONS.md in
the commit of the step named; the number is reserved so parallel sessions do not collide.

| D | Decision | Written at |
|---|---|---|
| D16 | Known pose (RTK) is a simplification for now; revisited if RTK is not on the Go2; heading error studied in step 8d | step 1 |
| D17 | Monte Carlo: the per-seed cross-check (4a) is the B3 deliverable; the event-rate model check (4b) is future work | step 4 |
| D18 | Regression and reduction comparisons use an absolute tolerance of 1e-12 on every r, mean and covariance entry, instead of byte-identical logs; `compare_r` uses absolute error on r | step 3 |
| D19 | Reduction tests that involve association marginals use exact enumeration on small clusters | step 12 |
| D20 | GNN's cost matrix uses likelihood ratios with one missed-detection column per track; its reduction test pins r = 1 | step 11 |
| D21 | The cardinality figure is restricted to the view on both sides | step 5 |
| D22 | Detections carry a class label; plant tracks do not associate weed-labelled detections; weeds may get their own map later | step 8b |
| D23 | Thesis scope is N bounded by the plan; bounded N moves forward as step 8c; N known is its limiting case | step 8c |
| D24 | Tests written before their code exists are marked `xfail(strict=True)` | step 3 |

## 8. Progress

| Step | Status | Notes |
|---|---|---|
| 1 | done | 2026-09-30: D16 reworded to "simplification for now, revisited if RTK is not on the Go2; heading error studied in step 8d" (first recorded as "thesis scope" in c049214). `world/path.py` module and `generate_path` docstrings updated. No behaviour change |
| 2 | done | 2026-09-30: the author checked the `r_sequence` docstring line by line against A2. `ScanEvent` gained `likelihood_ratios` and `born` (D25); the body of `r_sequence` was transcribed from the docstring without opening `filters/bernoulli.py`, and committed before that file was read for the derivation map. The A2 rows of `docs/derivations/README.md` were written at the author's request after that commit, and the A0/A1 rows corrected to the section titles the code cites |
| 3 | done | 2026-09-30: `build_scan_events` takes (mean, cov) (D25); `compare_r` absolute at 1e-12 (D18); `analyse --plots r_vs_analytic` checks every track, on the unpruned log when pruning (D14), and writes errors and branch coverage to metrics.json (D26). Both B3 tests pass for both p_D profiles, over births that together cover every branch; bank tracks checked too. Seed 42 phantom and bank runs match A2 to 1.2e-15. Rerun as the regression test after steps 9 and 10 |
| 3b | done | 2026-09-30: `filters/detection_prob.py`: `PdEvaluation` (miss p_D, detection p_D, missed moments), option A `AtMean` wraps the current code and is the default (D27); B, C and D are stubs that raise at construction; optional `filter.p_D_evaluation`; `build_scan_events` uses the same strategy (D33). Seed-42 logs unchanged (difference 0.0). C is the next brick when the FOV-edge effect is measured |
| 4 | done | 2026-09-30. 4a: `run_trials` general loop; per-seed cross-check (D17; 44/44 seeds pass, max error 1.0e-15); controlled phantom as birth kind `injected` (D34); fates, weed-in-gate strata and Wilson intervals via `analyse --plots phantom_fates` (D35; 200 seeds: on a plant 68 %, pruned 20 %, on a weed 7.5 %); d_match revised to 0.5 m (D28). 4b: future work, described only. 4c: timing log per filter run (D29), threads and CPU in run_meta, `scaling` command with four sweeps (D36; bank medians 0.3-2.6 ms against the 0.25 s budget). Left for later: the distance-to-weed sweep, per-seed plant proportions (step 8a), the p_D = 0 skip with its 1e-12 test |
| 5 | done | 2026-09-30, brick by brick: `gospa` with its decomposition (D31); `analysis/evaluation.py` (in-view tracks and plants, D21; cardinality, GOSPA, NEES, existence density); figures `tracks`, `existence_map`, `cardinality`, `gospa`, `nees`, `lifetimes` via `analyse --plots`, each reading only (run, filter_name) (D32). Seed-42 bank: the two plant-captured phantoms sit above the NEES band from scan 44 (overconfident). Association accuracy deferred to step 11 (D30); the "not applicable" panel comes with the first filter that needs it |
| 6 | done | 2026-09-30: `compare --run DIR --filters ...` runs the filters on one run folder (`track_all_filters`), composes every figure side by side (plots/compare_<plot>.png) and writes mean GOSPA, NEES-in-band share and median update time per filter to metrics.json (D37). bernoulli and a one-seed bank agree exactly, as D12 requires |
| 7 | done | 2026-09-30: `tests/test_figure_contract.py` renders every `ANY_FILTER_PLOTS` figure for every `FILTERS` entry (2 x 6 today), parametrised over both (D38). The "not applicable" panel comes with the first filter that needs it; the B2/B3 figures are Bernoulli-specific and outside the contract; the association figure stays deferred (D30) |
| 8a | done | 2026-09-30: `filter.plan` + `birth: null` start the bank from the plan's N slots at r = 1 (D39); configs `b4_known_n_bank{,_weeds}.yaml`; failure modes from the labels via `gate_contents` (D40). Seed 42: 70 slots, mean GOSPA 0.24 m, NEES in band 98 % of scans, 24 ms per scan; a neighbour's detection in 91 % of gates, clutter 19 %, weeds 9 % (weeds config), no plant pulled - a 3.6 cm prior moves a slot about 1 cm per wrong association. Scaling with the plan (row length 6-48 m): 13-88 ms median per scan, slope 0.92 against N (D36): cost grows with total N, so the p_D = 0 skip of 4c is worth testing. Not yet: the known-N NEES rule (track i against plant i) and multi-seed shares |
| 8b | done | 2026-09-30: `Detection.label` from a confusion matrix (`sensor.classifier`, default perfect, own RNG stream: truth and z unchanged); `filter.assumed_classifier` (perfect only) drops weed-labelled detections from plant gates; config `b4_known_n_bank_labels.yaml` (D22). Seed 42: weed returns in gates 9.2 % -> 0 %, GOSPA 0.249 -> 0.244 m. The imperfect classifier waits on the label-factor derivation (open question 7) |
| 8c | done | 2026-09-30: `world.p_missing` (independent per slot, own stream) and `plan.r_0`; configs `b1_two_rows_missing.yaml`, `b4_bounded_n_bank.yaml` (r_0 = 0.9); every slot checked against A2 (`r_initial`); `missing_plants` metric and figure, with the first "not applicable" panel (D23). Seed 42: the bank finds 0 of 4 seen empty slots, each at r = 1.0, as expected - the neighbours' detections multiply an empty slot's odds by about 7 per scan. Step 12 (JIPDA) is what should fix it. Later bricks: runs of neighbouring gaps, several seeds |
| 8d | not started | stub only; waits on NEES (step 5) for the body |
| 9 | not started | table structure fixed in step 9 |
| 10 | not started | waits on the A2 §3.1 `TODO(Wessel)` |
| 11 | not started | `assignment.py` is a stub |
| 12 | not started | `murty.py` is a stub; waits on a JPDA/JIPDA derivation |
| 13 | not started | |
| 14 | not started | |
| 15 | not started | waits on A3 |
| 16 | not started | decided by the data from steps 4c–7 |