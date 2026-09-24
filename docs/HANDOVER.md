# Implementation handover: crop_mot (B1 → B3)

You are implementing an MSc thesis codebase from a completed architectural skeleton. The
architecture is settled and approved. Your job is to fill in bodies, not to redesign.

## What this project is

Multi-object tracking and data association of **static plants in crop rows**, observed by a
camera on a quadruped walking the lane between rows. Top-down 2D. The thesis contribution is
**estimation and filtering**; vision is deliberately stubbed as a noisy black-box detector
with three knobs: detection probability `p_D`, clutter rate `lambda_FA`, and Gaussian
measurement noise `R`.

Work packages, in order:

| WP | What | Your scope |
|---|---|---|
| **B1** | Simulator: static plants in rows, known robot path, detector that misses with probability `1 - p_D` and adds Poisson(`lambda_FA`) clutter inside the FOV | implement |
| **B2** | Bernoulli filter over B1's recorded output. Deliverable: existence probability `r` per scan, plus the r-decay plot for a phantom (clutter-born) track | implement |
| **B3** | Validation: compare the simulated `r` against a hand-derived closed form (thesis item "A2"), plus a Monte-Carlo check | implement, with one blocker — see below |
| **B4** | PDA, JPDA, PMB, PMBM, and a GNN baseline | **DO NOT IMPLEMENT.** Phase 2. Interfaces and stub files already exist |

## What you are getting

45 Python files, 108 function bodies that are `raise NotImplementedError`. **Every one has a
docstring stating what it must do, its inputs and outputs, and which work package it serves.**

**The docstrings are the specification.** Read the docstring before writing the body. Several
of them record non-obvious decisions and the reason for them — for example why `crc32` and not
`hash()`, why `p_D` must be exactly `0.0` outside the FOV, why the Joseph form matters in
`kf_update`, why clutter must be sampled uniformly over *area* rather than over range. Those
are not stylistic notes; they are there because the naive implementation is subtly wrong.

If a docstring seems wrong once you start implementing, **say so and propose a change** — do
not silently implement something different. A silently-diverged docstring is worse than no
docstring, because the thesis text will be written from it.

The full design rationale, folder tree, work-package mapping and phase-2 migration plan is at:
`C:\Users\wesse\.claude\plans\local-root-c-users-wesse-onedrive-docume-delightful-cat.md`

## Who reads this code

The author is a **Systems & Control student, not a software engineer**. They know Kalman
filters and Bayesian estimation. They do not know design patterns, and the code will be read
by their supervisor and quoted in a thesis appendix.

Therefore:

- **Plain classes, `@dataclass`, and `typing.Protocol`.** That is the whole vocabulary.
- **No** deep inheritance, plugin registries, entry points, metaclasses, import-time
  decorators, dependency-injection frameworks, or abstract base class hierarchies.
- **Match the thesis notation**: state `x`, measurement `z`, likelihood `g(z|x)`, existence
  probability `r`, detection probability `p_D`, clutter rate `lambda_FA`, clutter density
  `c(z)`. Do not rename these to something more "Pythonic".
- Prefer an explicit loop a reader can follow over a clever vectorised one-liner, unless the
  vectorised form is genuinely clearer.

## Hard constraints

1. **No ROS anywhere in `crop_mot/`.** No `rclpy`, no message types. The phase-1 container does
   not have `rclpy` installed, which is the enforcement mechanism. `ros2_adapter/` stays a
   README until phase 2.
2. **Dependencies are numpy, scipy, matplotlib, PyYAML, pytest.** Nothing else without a
   written justification. `numpy<2` is pinned deliberately — ROS 2 Humble's `rclpy` interops
   with the apt-installed NumPy 1.x, and a NumPy 2.x in the same interpreter breaks it in
   phase 2. Do not "helpfully" unpin it.
3. **No async, no web UI, no database, no ML libraries.**
4. **Do not over-generalise.** Design for B1–B3 and the listed B4 slots only. If you find
   yourself adding a parameter "in case we need it later", stop.
5. **Do not change the pipeline shape or the public interfaces** without raising it first.

## Four invariants that must survive implementation

These are the load-bearing design decisions. Breaking any of them invalidates thesis results,
usually silently.

**1. The filter never sees ground truth.** `detections.jsonl` contains only measurement
vectors. Detection origins live in `labels.jsonl` and true positions in `truth.jsonl`, and no
filter function takes an argument that could carry them. Keep it that way: if you need truth
inside a filter, the thing you want belongs in `crop_mot/analysis/`.

**2. One filter interface, no special cases in the runner.** Every method — Bernoulli now,
PDA/JPDA/GNN/PMB/PMBM later — satisfies `initial_state` / `predict` / `update` / `extract`.
The state type `S` is opaque to the runner, so hypothesis structures never cross the interface;
`extract` is where they collapse to `list[TrackEstimate]`. Single-target is just the N≤1 case.
The runner in `runner/track.py` must contain **no** `isinstance`, no filter-name branch, and no
notion of how many targets exist. See `crop_mot/filters/README.md`.

**3. All methods run on the same recorded detections.** B1 writes `detections.jsonl` once;
every filter reads that same file into the same run folder. Never let a filter call the
simulator.

**4. Named RNG substreams.** `field`, `path`, `detection`, `clutter` are seeded independently
from one top-level seed (see `crop_mot/rng.py`). Changing `lambda_FA` must not move the plants.
Do not collapse these into one generator.

## Suggested implementation order

Work in vertical slices and keep `pytest` green as you go. Each step should end with its tests
passing, not with more stubs filled.

1. `types.py`, `io.py`, `rng.py`, `config.py` → make `tests/test_io_roundtrip.py` pass.
2. `world/` (field, path, truth). Note `pose_known: true` is the only phase-1 path; the wobble
   branch stays a stub.
3. `sensor/` (fov, models, sensor_model, detector, record). `ConstantPD` first;
   `RangeDependentPD` can follow.
4. `runner/run_dir.py`, `runner/simulate.py`, `crop_mot/__main__.py` →
   **`tests/test_b1_simulator.py` passes. B1 is done.**
5. `filters/kalman.py`, `filters/birth.py`, `filters/bernoulli.py`, `runner/track.py` →
   **`tests/test_b2_bernoulli.py` and `test_filter_interface_contract.py` pass.**
6. `analysis/estimates_log.py`, `analysis/plots.py::plot_r_vs_k` →
   **B2 deliverable: the r-decay plot for a phantom track.**
7. `analysis/analytic.py`, `events.py`, `metrics.py`, `montecarlo.py` →
   **`tests/test_b3_analytic_bernoulli.py` passes. B3 is done.**

`association/` is only needed as far as B2's gating requires (`gating.py`). Leave
`assignment.py` and `murty.py` alone — they are B4.

## One blocker you must not work around

`crop_mot/analysis/analytic.py` contains a single `TODO(human)` inside
`BernoulliExistenceReference.r_sequence`. It asks the thesis author to write out the branches
of the existence recursion — birth, prediction, misdetection, detection, and the
`in_fov = False` case — in terms of the `ScanEvent` fields each one reads.

**Do not fill this in yourself and do not guess the recursion.** The entire point of B3 is
checking the implementation against an *independently* hand-derived expression. If you derive
it from the filter you just wrote, the test becomes a tautology and B3 is worthless.

Everything except the body of `r_sequence` can be built while this is outstanding. If it is
still empty when you reach step 7, ask the author for it.

Note `ScanEvent` carries `p_D` and `lambda_FA` **per scan**, not as stored constants. That is
deliberate: it lets one closed form validate both the constant and the range-dependent p_D
profile, with the constant case being the one where every entry is equal. Preserve that.

## Environment

- Repo lives in the **WSL2 Ubuntu 22.04 filesystem**, never under `/mnt/c` and never in
  OneDrive. Develop in the VS Code dev container (`.devcontainer/devcontainer.json`).
- Ubuntu 22.04 with system Python 3.10, no virtualenv — this matches ROS 2 Humble exactly so
  phase 2 only adds ROS on top. Do not introduce a venv.
- Headless: `MPLBACKEND=Agg`. Plots are **saved** into the run folder. Never call `plt.show()`.
- Paths are `pathlib.Path`, relative to the repo root or the run folder. No hardcoded Windows
  paths. No executable shell scripts as entry points — everything is `python3 -m crop_mot ...`.

## How to verify

```bash
python3 -m pytest tests -q                    # the real check
python3 -m pytest tests -q -m "not slow"      # skip the Monte-Carlo test while iterating

python3 -m crop_mot simulate --config configs/b1_two_rows.yaml
python3 -m crop_mot track    --config configs/b2_bernoulli_phantom.yaml
python3 -m crop_mot analyse  --run runs/<stamp>_b2_bernoulli_phantom_seed42
```

The current state of the skeleton is: the package imports, the CLI resolves all three
subcommands, `pytest --collect-only` collects every test, and every test fails with
`NotImplementedError` — **not** `ImportError` or `AttributeError`. If you ever see an
`ImportError`, you have broken the wiring, not just left a stub.

A run folder is self-describing and is the unit of reproducibility: config copy,
`run_meta.json` (seed, git SHA, package versions), truth, labels, detections, one estimates log
per filter, metrics, plots. Several filters share one run folder — that is what makes their
comparison fair.

## When you disagree

Raise it. Specifically: say which docstring or interface you think is wrong, what breaks if it
stays, and what you propose instead. Then wait. The architecture was agreed with the author
after several rounds of trade-off discussion, and some choices that look over-cautious — the
`pose_known` switch, per-scan `p_D`, the stubbed `stereo.py` — exist because changing them
later would change the *meaning* of code already written, not just its implementation.
