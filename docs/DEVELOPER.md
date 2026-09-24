You are implementing an MSc thesis codebase from a completed architectural skeleton.
The architecture is settled and approved. Your job is to fill in function bodies,
not to redesign.

ENVIRONMENT (already set up — do not create anything new)

The repo is at ~/thesis/crop_mot inside the WSL2 Ubuntu-22.04 distro. WSL2 and Docker
Desktop are installed and running. Open that folder from WSL and use "Reopen in
Container" (.devcontainer/devcontainer.json). Never work under /mnt/c, never in
OneDrive, never create a second repo.

There is an older, superseded skeleton in the author's Obsidian vault at
2_Project/2_5_Implementation/crop-mot/. It has a different layout and different
filenames. Ignore it completely; it is not the repo you are working on.

READ FIRST (all inside the repo)

  1. docs/HANDOVER.md — scope, hard constraints, the four invariants, the suggested
     implementation order, and the one blocker you must not work around. Authoritative.
  2. docs/DESIGN.md — design rationale, folder tree, work-package mapping, phase-2
     migration plan.
  3. docs/derivations/README.md — how the code cites the thesis mathematics, plus
     mirrors of derivations A0-A3. Read A2 in full before you touch filters/bernoulli.py.

  Guard: docs/DESIGN.md must describe world/, sensor/, analysis/analytic.py and a filter
  interface of initial_state / predict / update / extract. If instead it describes
  simulation/, models/, or predict/update/estimates/diagnostics, it is a stale document
  from an earlier design — stop and tell me.

  Erratum: HANDOVER.md points to the design doc by a Windows path
  (C:\Users\wesse\.claude\plans\...). That path does not exist inside the container.
  Use docs/DESIGN.md.

CONTEXT IN ONE PARAGRAPH

Multi-object tracking and data association of static plants in crop rows, seen by a
camera on a quadruped. Top-down 2D. Estimation and filtering are the thesis
contribution; vision is stubbed as a black-box detector with p_D, lambda_FA and
Gaussian noise R. The repo has 45 Python files and 108 bodies that are
`raise NotImplementedError`, each with a docstring that IS its specification.

SCOPE

B1 (simulator), B2 (Bernoulli filter + r-decay plot for a phantom track), B3 (validate
r against a hand-derived closed form, plus the Monte-Carlo check). B4 —
PDA/JPDA/PMB/PMBM/GNN — is phase 2 and must NOT be implemented; its interfaces and stub
files already exist and stay empty. In association/, implement only gating.py as far as
B2 needs; leave assignment.py and murty.py alone.

TRACEABILITY — THIS IS A THESIS, NOT A PRODUCT

The derivations behind this code are mirrored in docs/derivations/ (A0 measurement and
clutter modelling, A1 Bayes to Kalman, A2 Bernoulli existence update, A3 normalisation
across global hypotheses). They are the justification for every expression you write.

  * When you implement an expression, cite it on the line or in the docstring using the
    tag form [A2 §2] — tag plus section. tests/test_derivations_in_sync.py fails if a tag
    names a derivation that does not exist.
  * Use the vault's notation exactly: x, z, g(z|x), r, p_D, p_D_bar, lambda_FA, c(z),
    lambda_u, e. Do not rename them to something more "Pythonic".
  * The mirrors are GENERATED and read-only. Never edit docs/derivations/*.md. If a
    derivation looks wrong or is missing what you need, tell me — I edit the vault note
    and re-sync. A hand-edited mirror fails the docs tests.
  * A0 and A3 contain "[figure omitted in mirror: ...]" markers where the vault note
    embeds an image. If the expression you need sits behind one, ask me for it in text.
    Do not reconstruct it from context.
  * If a docstring specifies behaviour but not the expression, and more than one
    defensible expression fits, ASK. Invented math that looks plausible is the most
    expensive thing you can hand back: it is found while the thesis text is being
    written against it.
  * Where the code deliberately uses a simplification instead of the derived form, say
    so in one docstring line: what it stands in for, and which section derives the full
    form.
  * Run `python3 -m pytest tests -q -m "docs"` before handing back a slice.

TWO HUMAN-OWNED ITEMS — NEITHER IS YOURS

  1. The TODO(human) in crop_mot/analysis/analytic.py (the existence recursion docstring)
     — see THE ONE BLOCKER below.
  2. The TODO(human) in docs/derivations/README.md (the A2 section-to-function map).
  Leave both alone. If either is still empty when you need it, ask me.

THREE MODELLING DECISIONS, ALREADY MADE — IMPLEMENT THEM AS STATED

  1. Birth existence r_birth: phase 1 uses the configured constant that
     BernoulliExistenceReference and the filter config already carry. It is an explicit
     stand-in for the measurement-driven form derived in A2 §4,
     r_b = e / (e + lambda_FA c(z)), with e = integral lambda_u(x) p_D(x) g(z|x) dx.
     Label it as such in both the filter's birth docstring and the config comment. The
     derived version arrives later as a NEW BirthModel class alongside
     SingleFromMeasurement — not as an edit to it, and not now.
  2. The filter and the analytic reference must use the SAME birth convention. If you
     find them diverging, stop and tell me; a B3 mismatch caused by that is not a bug
     you should be hunting.
  3. p_S and how the Bernoulli's density mixture collapses after an update: if the
     docstring does not pin these down, ask before choosing. Both change what r means.

THE ONE BLOCKER

crop_mot/analysis/analytic.py contains a single TODO(human) inside
BernoulliExistenceReference.r_sequence. It asks ME to write out the branches of the
existence recursion — birth, prediction, misdetection, detection, and the in_fov=False
case — in terms of the ScanEvent fields each reads. Note it is a DOCSTRING task; the
body of r_sequence is an ordinary NotImplementedError like the other 107.

The division of labour:
  * I write the docstring (the derivation). Do not write it, do not guess it, do not
    reconstruct it from the filter — that would make the B3 test a tautology.
  * You then implement r_sequence by TRANSCRIBING my docstring into code. While doing
    that, do not open filters/bernoulli.py. The reference must stay independent.
  * If my derivation needs a ScanEvent field that does not exist (current fields: k, dt,
    in_fov, p_D, lambda_FA, n_gated, n_clutter_gated), that is a schema change touching
    the recorder — raise it, do not add it silently.
  * Everything except that body can be built while it is outstanding. Ask me for it when
    you reach step 7.

ORDER OF WORK AND WHAT "GREEN" MEANS

Follow the seven-step order in HANDOVER.md, in vertical slices. Do not batch up stubs
and fill them at the end.

The suite is red by design right now: every test fails with NotImplementedError. So
"green as you go" means, after each step:
  * the tests named for that step pass;
  * every test that passed in an earlier step still passes;
  * tests for steps you have not reached still fail with NotImplementedError — never
    with ImportError or AttributeError. An ImportError means you broke the wiring.
  * run `python3 -m pytest tests -q -m "not slow"` while iterating; run the full suite
    including the Monte-Carlo test before declaring a slice done.

You may ADD tests. You may not weaken, rename, delete or loosen the tolerance of an
existing test to make a slice pass — above all
tests/test_b3_analytic_bernoulli.py::test_r_matches_analytic_recursion. If an existing
test looks wrong, say which one and why, then wait.

Commit once per slice, with the work package in the message (e.g. "B1: detector and
recording").

DEFINITION OF DONE PER WORK PACKAGE

  B1: tests/test_b1_simulator.py passes AND
      `python3 -m crop_mot simulate --config configs/b1_two_rows.yaml` produces a run
      folder containing config copy, run_meta.json, truth.jsonl, labels.jsonl and
      detections.jsonl.
  B2: tests/test_b2_bernoulli.py and test_filter_interface_contract.py pass AND the
      r-decay plot for a phantom track exists as a PNG in the run folder's plots/,
      produced through the CLI, not from a scratch script.
  B3: tests/test_b3_analytic_bernoulli.py passes, including the Monte-Carlo test, with
      r_sequence transcribed from my docstring.

CONSTRAINTS

No ROS anywhere in crop_mot/. Dependencies are numpy (<2, pinned deliberately — do not
unpin), scipy, matplotlib, PyYAML, pytest; nothing else without written justification.
No async, no web UI, no database, no ML libraries. Plain classes, dataclasses and
typing.Protocol only — the author is a Systems & Control student, not a software
engineer, and this code will be quoted in a thesis appendix. Headless: plots are saved,
never plt.show(). No hardcoded Windows paths.

WHEN YOU DISAGREE

If a docstring looks wrong once you start implementing, say which one, what breaks if it
stays, and what you propose instead — then wait. Do not silently implement something
different; the thesis text will be written from those docstrings.