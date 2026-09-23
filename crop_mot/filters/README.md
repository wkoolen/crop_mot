# Adding a filter (B4)

The interface was settled in phase 1 precisely so that this page can be short.

## The contract

Implement four methods from `crop_mot.filters.base.TrackingFilter`:

| Method | Does |
|---|---|
| `initial_state() -> S` | the prior, before any scan |
| `predict(state, dt) -> S` | time update: motion model + `p_S` on existence |
| `update(state, scan) -> S` | measurement update for one scan, including births |
| `extract(state) -> list[TrackEstimate]` | marginalise down to reported tracks |

`S` is *your* state type and is opaque to everything outside your file. Put whatever you
need in it.

## The two things that differ between methods, and where they go

**Single vs multi target.** Nowhere. `extract` returns a list; return one element for
Bernoulli or PDA, N for JPDA/GNN/PMB/PMBM. The runner does not branch.

**Hypotheses.** Inside `S`. PMBM's global hypothesis tree, PMB's Poisson intensity, JPDA's
marginal association probabilities — none of it crosses the interface. `extract` is the
point where it collapses to per-track `(r, mean, cov)`.

That is the whole reason `extract` is separate from `update` rather than having one
`step()`: the Bayes recursion and the estimator are different operations, and only the
estimator's output is common across methods.

## Steps

1. Write `crop_mot/filters/<name>.py` with your filter class and a
   `build_<name>(cfg: FilterConfig) -> TrackingFilter` function.
2. Add one line to `FILTERS` in `crop_mot/filters/__init__.py`.
3. Done. `tests/test_filter_interface_contract.py` picks it up automatically; the runner,
   the config loader, the detection format and the plots are untouched.

## What you get for free

- **The same detections.** Every filter reads the same `detections.jsonl` from the run
  folder, so comparison against the Bernoulli baseline and against GNN is fair by
  construction — not by remembering to use the same seed.
- **Shared Gaussian algebra.** `kalman.py` has `kf_predict`, `kf_update`,
  `predicted_measurement` and `log_predicted_likelihood`. Use them rather than writing your
  own, so that when two filters disagree the linear algebra is not a suspect.
- **Shared association.** `crop_mot/association/` has gating, the Hungarian wrapper for
  GNN, and Murty's k-best for JPDA and PMBM.
- **Optional diagnostics.** Implement `HasDiagnostics.diagnostics(state)` and the runner
  writes it alongside your estimates. Skip it and nothing breaks.

## Rules

- **Never import ground truth.** Nothing in `crop_mot.world`, and never `labels.jsonl`.
  If you find yourself wanting truth inside a filter, the thing you want probably belongs
  in `crop_mot/analysis/`.
- **Be pure.** Return new states; do not mutate the one you were given. That is what lets a
  test run a single `predict` in isolation.
- **Take everything from the config.** `p_D`, `lambda_FA` and the FOV come from
  `cfg.assumed_sensor` — the filter's *beliefs*, which may deliberately differ from the
  simulator's truth.
