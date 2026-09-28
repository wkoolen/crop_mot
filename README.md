# crop_mot

Multi-object tracking and data association of **static plants in crop rows**, seen by a
camera on a quadruped. MSc thesis codebase, phase 1.

Estimation and filtering are the contribution. Vision is deliberately stubbed as a noisy
black-box detector with three knobs: detection probability `p_D`, clutter rate
`lambda_FA`, and Gaussian measurement noise `R`.

> **This repository is a skeleton.** Every function body is `raise NotImplementedError`
> with a docstring stating what it must do, its inputs and outputs, and which work package
> it serves. No filter, simulator or mathematics is implemented yet.

## Work packages

| WP | What | Status |
|---|---|---|
| **B1** | Simulator: static plants in rows, known robot path, detector that misses with probability `1 - p_D` and adds Poisson clutter | skeleton |
| **B2** | Bernoulli filter over B1's output; existence probability `r` per scan, plus the r-decay plot for a phantom (clutter-born) track | skeleton |
| **B3** | Validation: compare the simulated `r` against the hand-derived closed form (thesis item A2), plus a Monte-Carlo check | skeleton |
| **B4** | PDA, JPDA, PMB, PMBM, and a GNN baseline — phase 2 | interfaces only |

## The pipeline

```
config + seed -> truth -> sensor model -> detections.jsonl
                                               |
                                               v
                                      filter (predict / update / extract)
                                               |
                                               v
                                      estimates_<filter>.jsonl -> analysis / plots
```

Every arrow crosses a **file on disk**, not a function call. That is what lets you re-run a
filter on last week's detections, run six filters on identical data, and trace any figure
back to the config and seed that produced it.

## Two rules the layout enforces

**The filter never sees ground truth.** `detections.jsonl` contains only measurement
vectors. Which detection came from which plant lives in `labels.jsonl`, and the true plant
positions live in `truth.jsonl` — neither of which any filter function takes as an
argument. The separation is structural, so it cannot be violated by accident.

**All methods run on the same recorded detections.** B1 writes `detections.jsonl` once;
every filter reads that same file into the same run folder. Fair comparison is a property
of the file tree rather than something you have to remember.

## Running it

```bash
python3 -m crop_mot simulate --config configs/b1_two_rows.yaml           # B1, prints scan counts
python3 -m crop_mot analyse  --run runs/<stamp>_b1_two_rows_seed42       # B1: scene + counts plots
python3 -m crop_mot track    --config configs/b2_bernoulli_phantom.yaml  # B2
python3 -m crop_mot analyse  --run runs/<stamp>_b2_bernoulli_phantom_seed42
python3 -m crop_mot candidates --run runs/<stamp>_b1_two_rows_seed42 --min-distance 1.0  # B2: phantom seeds
python3 -m crop_mot track    --config configs/b2_bernoulli_bank_phantoms.yaml  # B2: several phantoms, pruned
python3 -m crop_mot analyse  --run runs/<stamp>_b2_bernoulli_bank_phantoms_seed42 --plots hypotheses hypotheses_anim
python3 -m pytest tests -q
```

While this is a skeleton, the acceptance criterion is: the package imports, the CLI
resolves all three subcommands, `pytest --collect-only` collects every named test, and
every test fails with `NotImplementedError` — not `ImportError` or `AttributeError`. That
proves the wiring is complete and only the mathematics is missing.

## A run folder

```
runs/<timestamp>_<name>_seed<seed>/
  config.yaml                verbatim copy of the config used
  run_meta.json              seed, git sha, package versions, argv, timestamp
  truth.jsonl                plant positions + true and reported poses   (EVAL ONLY)
  labels.jsonl               per-detection origin ids, visible/detected/
                             edge-lost plant ids per scan                (EVAL ONLY)
  detections.jsonl           the only filter input
  estimates_bernoulli.jsonl  one per filter
  metrics.json
  plots/
```

Several filters share **one** run folder. Simulate once, track repeatedly into the same
directory, and the fact that they all saw identical data is visible in the file tree.

## Environment

Phase 1 is Python only: numpy, scipy, matplotlib, pytest, PyYAML.

The container is **Ubuntu 22.04 with its system Python 3.10** — the exact pair ROS 2 Humble
uses, so phase 2 only adds ROS on top. It is deliberately *not* `ros:humble`, even though
that would also work: with `rclpy` absent, the rule *the core package never imports ROS* is
enforced by the package not existing, rather than by discipline. An accidental
`import rclpy` fails immediately here.

There is no virtualenv, because Humble's `rclpy` lives in the system interpreter and a venv
would need `--system-site-packages` contortions later.

`numpy<2` is pinned for the same reason: Humble's `rclpy` does its array interop against
the apt-installed NumPy 1.x.

### Setup

The repo lives in the **WSL2 Linux filesystem**, never under `/mnt/c`. A `/mnt/c` bind
mount crosses the 9p bridge and makes pytest collection and file watching noticeably slow,
and a OneDrive-synced folder would sync every PNG in `runs/` and can corrupt `.git`.

```bash
# inside the Ubuntu-22.04 WSL distro
cd ~/thesis/crop_mot
code .                  # then: Reopen in Container
```

`remoteUser: dev` in the devcontainer is UID 1000 and must match your distro user's
`id -u`. Unlike a Windows bind mount, a WSL2 Linux bind mount carries real UIDs.

## Phase 2

`ros2_adapter/` is a placeholder with a README describing the node that will convert ROS
messages into the dataclasses in `crop_mot.types` and call the same filter methods. It
contains no code, and must not until phase 2.

`resource/crop_mot` already exists so that becoming an `ament_python` package later adds
`package.xml` and `setup.py` without moving a single core file.

## Layout

| Path | Purpose |
|---|---|
| `crop_mot/types.py` | the dataclasses every module boundary uses |
| `crop_mot/world/` | ground truth: plants, path, poses |
| `crop_mot/sensor/` | FOV, measurement model, detection process, recording |
| `crop_mot/filters/` | the one filter interface, plus Bernoulli |
| `crop_mot/association/` | gating, Hungarian assignment, Murty k-best (for B4) |
| `crop_mot/runner/` | run folders and the two entry points |
| `crop_mot/analysis/` | the B3 closed form, Monte-Carlo, metrics, plots |
| `crop_mot/filters/README.md` | **how to add a B4 filter** |
