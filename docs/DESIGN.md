# Framework skeleton for the crop-row MOT simulator (B1–B4)

## Context

`Thesis/Codebase/` is currently empty. Work package **B1** starts now (week 4 of the
preparation phase, per `Schematics/Scope-Metroline-roadmap.drawio.png`), and the roadmap
already fixes the shape of the work: B1 simulator → B2 Bernoulli filter with an r-decay
plot for a phantom track → B3 "check r vs A2" (the hand-derived Bernoulli existence
recursion) → B4 PDA/JPDA/PMB/PMBM deferred to phase 2.

Two constraints drive the whole design:

1. **B4 must be additive.** Adding JPDA or PMBM in phase 2 must mean adding a file, not
   reworking the runner. So the filter interface has to be settled *now*, before any
   filter exists.
2. **Phase 2 adds ROS 2 Humble.** Migration must be cheap, so the core package never
   imports ROS, the container already runs the Ubuntu 22.04 / Python 3.10 pair Humble
   needs, and the folder layout is already `ament_python`-shaped.

This plan produces a **skeleton only**: every function body is `raise NotImplementedError`
with a docstring stating what it must do, its inputs/outputs, and which work package it
serves. No filter, simulator or math is implemented.

Settled with the user before writing this plan:

| Decision | Choice |
|---|---|
| Measurement space | Linear world-frame Cartesian now, behind a swappable `MeasurementModel` Protocol |
| Filter interface | `initial_state` / `predict` / `update` / `extract` |
| Dimension | 2D scenario, but state/measurement are plain `np.ndarray` with `dim_x`/`dim_z` on the models |
| Association helpers | Stub `association/` now (gating, assignment, Murty) |
| Repo location | **Inside the WSL2 Linux filesystem, not OneDrive** |
| Pose knowledge | Stub gait wobble behind a `pose_known: bool` switch (true pose vs reported pose) |
| Detection probability | Both a constant and a range/occlusion-dependent p_D, switchable, with the analytic reference carrying **per-scan** p_D so both are validatable |
| B3 evidence | Closed form **and** a small Monte-Carlo sim, so the derivation is backed empirically |
| Detection log | JSONL confirmed (short phase-1 trials; long trials arrive with ROS 2 in phase 2) |
| Package name | `crop_mot` confirmed |
| 3D / stereo | Nice-to-have, stubbed at the **back** of the pipeline only — not threaded through phase 1 |

---

## 0. Phase 0 — environment setup (do this before any code)

Measured state of this machine on 2026-09-23:

| Check | Result |
|---|---|
| WSL | **Not installed** (`wsl --status` → "The Windows Subsystem for Linux is not installed") |
| Docker | **Not installed** (`docker` not on PATH) |
| OS | Windows 11 Home, build 26200, x64 — `wsl --install` is supported |
| CPU | Intel i7-11370H |
| Virtualization | **Enabled.** `HypervisorPresent = True`, VBS "Running", and `systeminfo` reports "A hypervisor has been detected". The `Win32_Processor.VirtualizationFirmwareEnabled = False` reading is a known false negative that appears precisely *because* a hypervisor is already running — not a blocker. |
| Free space on C: | 81 GB (Ubuntu distro + Docker images will want roughly 15 GB) |
| winget | Present |

So phase 0 is a genuine install, not a config tweak. Run these from an **elevated** PowerShell
on the host; everything after step 6 happens inside Linux.

1. `wsl --install -d Ubuntu-22.04` — **reboot when prompted.** Ubuntu 22.04 deliberately, so
   the distro, the phase-1 container and the phase-2 Humble container are all the same
   userland and the same Python 3.10.
2. After reboot, finish the distro's first-run prompts (UNIX username + password).
   Confirm `wsl -l -v` shows `Ubuntu-22.04` with `VERSION 2`.
3. Inside the distro, check `id -u` returns **1000**. The `Dockerfile` creates its `dev` user
   at UID/GID 1000 so that bind-mounted files are owned correctly. Unlike a Windows bind
   mount, a WSL2 Linux bind mount carries *real* UIDs, so this now matters. If `id -u` is not
   1000, change the `USER_UID`/`USER_GID` build args instead of fighting it.
4. `winget install -e --id Docker.DockerDesktop` on the host, then in Docker Desktop:
   **Settings → General → Use the WSL 2 based engine**, and **Settings → Resources → WSL
   Integration → enable `Ubuntu-22.04`**.
5. Verify from inside the distro: `docker run --rm hello-world`.
6. Install the VS Code extensions **WSL** and **Dev Containers** on the host.
7. Create the repo **inside the Linux filesystem**, never under `/mnt/c`:
   ```bash
   mkdir -p ~/thesis && cd ~/thesis
   git init crop_mot && cd crop_mot
   ```
   Open it with `code .` *from inside WSL* — VS Code attaches through the WSL remote, and
   "Reopen in Container" then bind-mounts a native Linux path at full speed.
8. Set up a **git remote (e.g. a private GitHub repo) and push.** This is not optional
   housekeeping: moving out of OneDrive removes the automatic backup you currently have, so
   the remote becomes the only copy of your code that survives a laptop failure.

**What stays in OneDrive:** the thesis text, `Papers/`, `Meetings/`, `Schematics/`. Only code
moves. The empty `Thesis/Codebase/` folder is abandoned — nothing to migrate.

**How the work splits across the reboot:** steps 1–5 need elevation and at least one reboot,
so they are yours to run (I can print the exact commands and check the results afterwards).
Once the distro exists I can create the skeleton files directly, either through the
`\\wsl$\Ubuntu-22.04\home\<you>\thesis\crop_mot` UNC path or by shelling in with
`wsl -d Ubuntu-22.04 -- ...`. Everything from §2 onward is unaffected by which route we use.

Why not just develop on the Windows-mounted path: a `/mnt/c` bind mount crosses the 9p
filesystem bridge, which makes `pytest` collection and file watching slow enough to notice;
OneDrive would sync every PNG written into `runs/`; and OneDrive has a long history of
corrupting `.git` when it syncs a repo mid-write.

---

## 1. Design rationale

**One filter interface, four methods.** All six methods of tracking (Bernoulli, PDA, JPDA,
GNN, PMB, PMBM) satisfy `initial_state → predict → update → extract`. The two things that
actually differ between them are absorbed without branching in the runner:

- *Single vs multi target*: `extract` always returns `list[TrackEstimate]`. Bernoulli and
  PDA return a list of length ≤ 1. That is the N=1 case of the same type, not a special case.
- *Hypotheses*: the filter's state `S` is opaque to the runner. PMBM's global hypothesis
  tree, PMB's Poisson intensity, JPDA's marginal association probabilities all live inside
  `S`. `extract` is the point where a filter marginalises its internal structure down to
  per-track `(r, mean, cov)`. Separating `extract` from `update` — rather than one `step()` —
  is exactly what makes this possible, and it matches how the MOT literature separates the
  Bayes recursion from the estimator.
- *Everything else the filter needs* (p_D, λ_FA, c(z), birth model, motion and measurement
  models, gating threshold) is injected at construction, so `update(state, scan)` has an
  identical signature everywhere.

**Truth is unreachable, not merely unused.** The filter reads `detections.jsonl`, which
contains only measurement vectors. Which detection came from which plant, and which plants
were in the FOV, are written to *separate* files (`labels.jsonl`, `truth.jsonl`) that the
filter code has no argument to receive. The separation is structural, so it cannot be
violated by accident. The filter's own p_D / λ_FA / FOV are configured under
`filter.assumed_sensor`, distinct from the simulator's `sensor` block — which also gives
model-mismatch experiments for free.

**Detections are recorded to disk.** B1 writes `detections.jsonl` once; every filter run
reads that same file. Fair comparison between Bernoulli, GNN, JPDA and PMBM is then a
property of the file layout rather than something you must remember to do. One run folder
holds one detection set and many `estimates_<filter>.jsonl`.

**Seed discipline by named substream.** One top-level seed spawns named generators
(`field`, `path`, `detection`, `clutter`) via `SeedSequence([seed, crc32(name)])`. Changing
λ_FA therefore does not move the plants, so parameter sweeps stay comparable. `crc32` rather
than `hash()` because Python randomises string hashing per process — a classic silent
reproducibility bug.

**Ubuntu 22.04, not `ros:humble`, in phase 1.** `ros:humble` is itself `FROM ubuntu:22.04`
with the same glibc and the same system Python 3.10, so nothing ABI-relevant differs. What
does differ: a ~2 GB image instead of ~300 MB (painful to rebuild over Docker Desktop on
Windows), and — more importantly — if `rclpy` is on the path, the rule "core never imports
ROS" is enforced only by discipline. With plain Ubuntu it is enforced by the package being
absent: an accidental `import rclpy` fails immediately. The trade-off accepted is that
ROS-side dependency conflicts surface only in phase 2; the mitigation is to pin `numpy<2`
now, because Humble's `rclpy` interops with the apt-installed NumPy 1.x and a pip NumPy 2.x
breaks it.

**No virtualenv.** Humble's `rclpy` lives in the system interpreter's site-packages, so a
venv would need `--system-site-packages` contortions in phase 2. Installing into system
Python 3.10 now is what ROS users do and is one less thing to migrate.

**Three extension slots are stubbed but unused.** `pose_known=False` (gait-induced odometry
error), `RangeDependentPD` (occlusion), and `sensor/stereo.py` (3D) each exist as a file with
`NotImplementedError` bodies that nothing in B1–B4 calls. They are stubbed rather than
omitted because each one, if added later, would otherwise change the *meaning* of something
already written: whether `Scan.pose` is truth or an estimate, whether the A2 closed form may
assume a scalar p_D, and whether a measurement is 2D. Writing the slot down now fixes those
meanings; filling it in later touches one file.

**Plain classes, dataclasses, `typing.Protocol`.** No ABC hierarchies, no plugin registry,
no metaclasses. Filters are found through a plain `dict` in `filters/__init__.py`; adding a
B4 filter is a new file plus one line in that dict. Stating it honestly: "add a file and one
dict entry", not "zero changes".

**YAML + PyYAML** is the one dependency beyond numpy/scipy/matplotlib/pytest. Justification:
Python 3.10 has no `tomllib` (that is 3.11+), so TOML would need `tomli` anyway — equal cost;
and ROS 2 parameter files are YAML and Humble already ships PyYAML, so phase 2 can feed the
same files to a node without a second format.

---

## 2. Folder tree

```
Codebase/
├── .devcontainer/devcontainer.json   VS Code dev container: build, interpreter, pytest, PYTHONPATH
├── Dockerfile                        Ubuntu 22.04 + system python3.10 + pip deps, non-root user
├── .dockerignore                     keep runs/, .git, __pycache__ out of the build context
├── .gitattributes                    force LF everywhere; mark binaries; no exec-bit reliance
├── .gitignore                        runs/, __pycache__, .pytest_cache, *.egg-info
├── pyproject.toml                    package metadata, pytest + ruff config (setup.py added in phase 2)
├── requirements.txt                  pinned numpy<2, scipy, matplotlib, pytest, PyYAML
├── README.md                         what this is, how to run B1/B2/B3, run-folder layout
├── resource/crop_mot                 empty ament marker file, added now so phase 2 moves nothing
│
├── configs/
│   ├── b1_two_rows.yaml              B1 scenario: two plant rows, lane path, detector params
│   ├── b1_two_rows_duplicate.yaml    B1 variant: detector returns extra hits per plant (multiplicity)
│   ├── b1_two_rows_extended.yaml     B1 variant: plants as extended objects, Poisson hits (multiplicity)
│   └── b2_bernoulli_phantom.yaml     B2 run: reuses the B1 scenario, Bernoulli on a clutter-born track
│
├── crop_mot/
│   ├── __init__.py                   version string only; imports nothing heavy
│   ├── types.py                      Pose2D, Detection, Scan, ScanLabels, TrackEstimate, FieldOfView
│   ├── config.py                     YAML -> typed config dataclasses; fail loudly on unknown keys
│   ├── rng.py                        named substreams from one seed (field/path/detection/clutter)
│   ├── io.py                         JSONL read/write helpers + numpy<->json encoding, one place
│   │
│   ├── world/
│   │   ├── field.py                  [B1] plant positions from row specs + planting jitter (truth)
│   │   ├── path.py                   [B1] true pose sequence + REPORTED pose (pose_known switch)
│   │   └── truth.py                  [B1] GroundTruth bundle; writes truth.jsonl (eval only)
│   │
│   ├── sensor/
│   │   ├── fov.py                    [B1] FieldOfView wedge: is a world point visible from a pose
│   │   ├── models.py                 [B1] MeasurementModel Protocol + LinearGaussianXY
│   │   ├── sensor_model.py           [B1/B2] SensorModel Protocol + ConstantPD / RangeDependentPD
│   │   ├── detector.py               [B1] sample one Scan: misses, clutter, noise (simulator side)
│   │   ├── record.py                 [B1] write/read detections.jsonl and labels.jsonl
│   │   └── stereo.py                 [nice-to-have, phase 2+] stereo depth -> 3D measurement stub
│   │
│   ├── motion/models.py              [B2] MotionModel Protocol + StaticTarget (F=I, Q=0)
│   │
│   ├── filters/
│   │   ├── __init__.py               FILTERS dict: name -> builder. B4 adds one line here.
│   │   ├── base.py                   [B2/B4] TrackingFilter Protocol, BirthModel, SurvivalModel
│   │   ├── kalman.py                 [B2] shared kf_predict / kf_update / predicted_measurement stubs
│   │   ├── bernoulli.py              [B2] BernoulliState + BernoulliFilter (the only phase-1 filter)
│   │   ├── birth.py                  [B2/B4] birth models incl. single-from-measurement (phantom)
│   │   └── README.md                 [B4] how to add a filter: 4 methods, 1 dict line, 1 contract test
│   │
│   ├── association/
│   │   ├── gating.py                 [B4] ellipsoidal / chi-square gate; also used by B2 for r
│   │   ├── assignment.py             [B4] cost matrix + scipy linear_sum_assignment wrapper (GNN)
│   │   └── murty.py                  [B4] k-best assignments (JPDA, PMBM global hypotheses)
│   │
│   ├── runner/
│   │   ├── run_dir.py                create/locate a run folder; copy config; write run_meta.json
│   │   ├── simulate.py               [B1] config -> truth + detections.jsonl in a run folder
│   │   └── track.py                  [B2] run folder + filter name -> estimates_<name>.jsonl
│   │
│   ├── analysis/
│   │   ├── estimates_log.py          read/write estimates_<filter>.jsonl
│   │   ├── counts.py                 [B1] clairvoyant per-scan counts from labels.jsonl (eval only)
│   │   ├── analytic.py               [B3] AnalyticReference Protocol + BernoulliExistenceReference (A2)
│   │   ├── events.py                 [B3] build the ScanEvent sequence (carries per-scan p_D)
│   │   ├── montecarlo.py             [B3] repeat a scenario over N seeds; mean r +/- band vs closed form
│   │   ├── metrics.py                [B3] compare_r (max/rms error, first divergence); [B4] gospa stub
│   │   └── plots.py                  scene, counts, r_vs_k, hypotheses(+gif), r_vs_analytic, r_montecarlo; saves, never shows
│   │
│   └── __main__.py                   `python -m crop_mot simulate|track|analyse` (argparse, thin)
│
├── ros2_adapter/README.md            [phase 2] placeholder: node design, no code, no ROS imports
│
├── tests/
│   ├── conftest.py                   tiny_scenario fixture, tmp run dir
│   ├── test_b1_simulator.py          determinism, no detection outside FOV, p_D and Poisson rates
│   ├── test_b2_bernoulli.py          r in [0,1], r decays under sustained misdetection, shapes
│   ├── test_b3_analytic_bernoulli.py NAMED B3 cross-check: r_sim vs A2 closed form
│   ├── test_filter_interface_contract.py  every entry in FILTERS satisfies the 4-method contract
│   └── test_io_roundtrip.py          Scan / TrackEstimate -> jsonl -> back, exactly
│
└── runs/.gitkeep                     run outputs (gitignored)
```

---

## 3. Interface definitions

Signatures are final; bodies are `raise NotImplementedError`. Docstrings below are shortened
to one line for readability — the files get the full form (what it must do, inputs, outputs,
work package, and for models the *assumption stated explicitly*).

### `crop_mot/types.py`

```python
@dataclass(frozen=True)
class Pose2D:
    """A robot pose in the world frame. [B1]

    Used for BOTH the true pose and the reported pose. Which one you hold is a property of
    where it came from: Scan.pose is always the REPORTED pose (what the filter is allowed to
    see); the true pose is written to truth.jsonl. With pose_known=True the two are equal.
    """
    x: float
    y: float
    theta: float          # heading, rad, CCW from +x

@dataclass(frozen=True)
class FieldOfView:
    """Camera footprint as a wedge in the body frame. [B1]"""
    min_range: float
    max_range: float
    half_angle: float     # rad, symmetric about the heading

    def area(self) -> float:
        """Area of the wedge in m^2; the denominator of a uniform clutter density c(z). [B1]"""
        raise NotImplementedError

@dataclass(frozen=True)
class Detection:
    """One measurement z from the black-box detector. Carries NO origin label. [B1]"""
    z: np.ndarray         # shape (dim_z,), world-frame xy in phase 1

@dataclass(frozen=True)
class Scan:
    """All detections received at one time step, with the REPORTED pose. Filter input. [B1/B2]"""
    k: int
    t: float
    pose: Pose2D          # reported, not true; equal to true when pose_known=True
    detections: tuple[Detection, ...]

@dataclass(frozen=True)
class ScanLabels:
    """Truth-side twin of Scan. Never passed to a filter; evaluation only. [B1/B3]"""
    k: int
    origin: tuple[int | None, ...]    # object id per detection, None = clutter
    visible_ids: tuple[int, ...]      # object ids inside the FOV this scan
    detected_ids: tuple[int, ...]     # object ids actually detected this scan

@dataclass(frozen=True)
class TrackEstimate:
    """One track's marginal posterior, the common output of every filter. [B2/B4]"""
    track_id: int
    r: float              # existence probability; GNN reports 0.0 or 1.0
    mean: np.ndarray      # shape (dim_x,)
    cov: np.ndarray       # shape (dim_x, dim_x)
```

### `crop_mot/world/path.py` — the pose-known switch

```python
@dataclass(frozen=True)
class PoseSample:
    """One time step's true and reported pose. [B1]"""
    t: float
    true: Pose2D
    reported: Pose2D      # what the filter receives; == true when pose_known

def generate_path(cfg: PathConfig, rng_path: Generator) -> list[PoseSample]:
    """Build the robot's walk along the lane.

    STUB BEHAVIOUR, phase 1: with cfg.pose_known=True the reported pose IS the true pose and
    the wobble parameters are ignored, which is the assumption B2 and the A2 closed form rely
    on. With pose_known=False the reported pose is the true pose perturbed by
    N(0, yaw_wobble_std^2) in heading and N(0, xy_noise_std^2) in position - gait-induced
    odometry error. Nothing downstream changes: detections are still generated from the TRUE
    pose, and the filter still un-projects using the REPORTED one, so the mismatch shows up
    as a measurement bias exactly as it would on the robot. [B1, extension slot]
    """
    raise NotImplementedError
```

The point of stubbing it now rather than later: it is the difference between `Scan.pose`
meaning "truth" and meaning "an estimate". Deciding that after B2 exists would mean revisiting
every filter's un-projection.

### `crop_mot/motion/models.py`

```python
class MotionModel(Protocol):
    """Target dynamics. ASSUMPTION stated per implementation. [B2/B4]"""
    dim_x: int

    def predict_moments(self, mean: np.ndarray, cov: np.ndarray, dt: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Chapman-Kolmogorov step for a Gaussian: return (m_pred, P_pred)."""
        raise NotImplementedError

@dataclass(frozen=True)
class StaticTarget(MotionModel):
    """ASSUMPTION: plants do not move. F = I, Q = q*dt*I with q >= 0 (q>0 only to keep P
    from collapsing numerically). [B1/B2]"""
    q: float = 0.0
```

### `crop_mot/sensor/models.py`

```python
class MeasurementModel(Protocol):
    """Maps target state to measurement space. ASSUMPTION stated per implementation. [B1/B2]"""
    dim_x: int
    dim_z: int
    R: np.ndarray

    def h(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Noise-free predicted measurement of state x seen from pose."""
        raise NotImplementedError

    def H(self, x: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Jacobian dh/dx at x, shape (dim_z, dim_x). Constant for linear models."""
        raise NotImplementedError

@dataclass(frozen=True)
class LinearGaussianXY(MeasurementModel):
    """ASSUMPTION: the detector reports plant position in WORLD xy, having already
    un-projected through the known pose. Then z = H x + v, H = I_2, v ~ N(0, R), and
    g(z|x) = N(z; x, R) is linear-Gaussian, so no Jacobian approximation is involved. [B1/B2]"""
```

### `crop_mot/sensor/sensor_model.py`

The detection process, kept separate from the measurement model. **One Protocol serves both
sides**: the simulator uses it to generate, the filter uses it to evaluate. The filter's
instance is built from `filter.assumed_sensor` and may deliberately differ from the truth.

```python
class SensorModel(Protocol):
    """Detection process: p_D, clutter rate lambda_FA, clutter density c(z). [B1/B2/B4]"""
    fov: FieldOfView
    measurement: MeasurementModel

    def p_D(self, x: np.ndarray, pose: Pose2D) -> float:
        """Detection probability of a target at x seen from pose; 0 outside the FOV."""
        raise NotImplementedError

    def lambda_FA(self, pose: Pose2D) -> float:
        """Expected number of clutter detections this scan (Poisson mean)."""
        raise NotImplementedError

    def clutter_density(self, z: np.ndarray, pose: Pose2D) -> float:
        """Clutter density c(z); uniform over the FOV means lambda_FA / fov.area()."""
        raise NotImplementedError

@dataclass(frozen=True)
class ConstantPD(SensorModel):
    """ASSUMPTION: p_D is the same everywhere inside the FOV and 0 outside. This is the
    assumption the A2 closed form is derived under, so it is the default for B2/B3. [B1/B2/B3]"""
    p_D_const: float

@dataclass(frozen=True)
class RangeDependentPD(SensorModel):
    """ASSUMPTION: p_D falls off with range (and optionally with row occlusion), modelling
    the yellow 'occluded / missed' plants in the T=2 sketch. Breaks the constant-p_D
    assumption, which is exactly why the B3 reference is written to accept a per-scan p_D
    rather than a single constant - both profiles stay validatable. [B1/B3]"""
    p_D_near: float        # at min_range
    p_D_far: float         # at max_range
    occlusion_factor: float = 1.0   # multiplier when a nearer plant shadows this one
```

`sensor/detector.py` holds the *sampling* side as free functions so the Protocol stays small
and a filter can never accidentally generate data:

```python
def sample_scan(truth: GroundTruth, pose: Pose2D, k: int, t: float,
                model: SensorModel, rng_detect: Generator, rng_clutter: Generator
                ) -> tuple[Scan, ScanLabels]:
    """Draw one scan: each visible plant detected w.p. p_D with N(0,R) noise, plus
    Poisson(lambda_FA) clutter uniform in the FOV. Detections are shuffled so order
    carries no information. Returns the filter-visible Scan and the truth-only labels. [B1]"""
    raise NotImplementedError
```

### `crop_mot/sensor/stereo.py` — 3D nice-to-have, stub only

Placed at the **back** of the pipeline, not threaded through phase 1. Nothing in B1–B4
imports it; it exists so the 3D route is written down rather than rediscovered later.

```python
@dataclass(frozen=True)
class StereoDepthModel:
    """NICE-TO-HAVE, phase 2+. Stereo pair -> 3D plant position with range-dependent noise.

    Not used by B1-B4. Written down now because the choice that makes it cheap has already
    been made: models declare their own dim_x / dim_z, so going 3D means adding
    LinearGaussianXYZ next to LinearGaussianXY and a 3D FieldOfView (a frustum instead of a
    wedge). The filters, the runner, the detection format and the analysis do not change,
    because none of them hard-code a dimension. [C1, deferred]
    """
    baseline: float           # m, stereo baseline
    focal_px: float
    disparity_std_px: float

    def depth_covariance(self, depth: float) -> np.ndarray:
        """R grows roughly with depth^2 / (baseline * focal) - the reason stereo range noise
        is strongly anisotropic and why a constant R stops being defensible in 3D."""
        raise NotImplementedError
```

### `crop_mot/filters/base.py` — the one interface

```python
S = TypeVar("S")

class TrackingFilter(Protocol[S]):
    """The single interface Bernoulli, PDA, JPDA, GNN, PMB and PMBM all satisfy.

    S is the filter's own state type and is opaque to the runner: a Bernoulli density for
    B2, a set of Bernoulli components plus a Poisson intensity for PMB, a global hypothesis
    tree for PMBM. Hypotheses therefore never cross this interface. Everything beyond the
    scan (models, p_D, lambda_FA, birth, gate) is injected at construction, so `update` has
    one signature for every filter. [B2/B4]
    """
    name: str

    def initial_state(self) -> S:
        """Prior before any scan: for B2, r = 0 (or the configured birth prior)."""
        raise NotImplementedError

    def predict(self, state: S, dt: float) -> S:
        """Time update: motion model + survival p_S on existence. Pure, returns a new state."""
        raise NotImplementedError

    def update(self, state: S, scan: Scan) -> S:
        """Measurement update for one scan, including births. Pure, returns a new state."""
        raise NotImplementedError

    def extract(self, state: S) -> list[TrackEstimate]:
        """Marginalise the posterior to reported tracks. Length <= 1 for Bernoulli/PDA,
        arbitrary for JPDA/GNN/PMB/PMBM. This is where hypotheses collapse."""
        raise NotImplementedError

class HasDiagnostics(Protocol[S]):
    """OPTIONAL. Filters that want extra per-scan output (association weights, hypothesis
    count) implement this; the runner writes it when present. Not part of the core contract
    so that B4 filters are not forced to provide it."""
    def diagnostics(self, state: S) -> dict[str, float]:
        raise NotImplementedError

class BirthModel(Protocol):
    """Where new tracks come from. [B2/B4]"""
    def birth_components(self, scan: Scan) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Return (weight_or_r_b, mean, cov) born from this scan. For B2's phantom, one
        component seeded from a clutter detection at a configured scan index."""
        raise NotImplementedError

@dataclass(frozen=True)
class SurvivalModel:
    """ASSUMPTION: plants are permanent, so p_S = 1.0 by default. [B2]"""
    p_S: float = 1.0
```

The runner, identical for all six filters, no branches:

```python
state = flt.initial_state()
for scan in scans:
    state = flt.predict(state, dt)
    state = flt.update(state, scan)
    write_estimates(scan.k, flt.extract(state))
```

### `crop_mot/analysis/analytic.py` — the B3 hook

```python
@dataclass(frozen=True)
class ScanEvent:
    """What happened at scan k, expressed in the terms the hand derivation (A2) uses.
    Built by analysis/events.py from the recorded detections + labels AFTER the filter run,
    so the filter never sees it. [B3]

    p_D and lambda_FA are carried PER SCAN, not assumed constant. With ConstantPD they are
    the same value every scan and the recursion collapses to the textbook constant-p_D form;
    with RangeDependentPD they vary as the robot approaches a plant. One reference
    implementation therefore validates both profiles, which is what makes the range-dependent
    case checkable rather than just plausible.
    """
    k: int
    dt: float
    in_fov: bool           # was the hypothesised location inside the FOV
    p_D: float             # the FILTER's assumed p_D at this scan, at the hypothesised state
    lambda_FA: float       # the FILTER's assumed clutter rate at this scan
    n_gated: int           # detections falling inside the gate
    n_clutter_gated: int   # of those, how many were clutter (truth-side)

class AnalyticReference(Protocol):
    """Closed-form reference trajectory to cross-check a filter against. [B3]"""
    name: str

    def r_sequence(self, events: Sequence[ScanEvent]) -> np.ndarray:
        """Existence probability r_k for k = 0..K-1 from the closed form alone."""
        raise NotImplementedError

@dataclass(frozen=True)
class BernoulliExistenceReference(AnalyticReference):
    """Hand-derived Bernoulli existence recursion (thesis item A2): the misdetection,
    detection and birth branches of r_k.

    p_D and lambda_FA are read from each ScanEvent, NOT stored here, so this one class covers
    both the constant and the range-dependent profile. Those values must be the FILTER's
    assumed ones, not the simulator's, or the comparison tests the wrong thing. [B3]
    """
    p_S: float
    r_birth: float
```

### `crop_mot/analysis/montecarlo.py` — the empirical backing for B3

```python
@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregate of N repeated runs of the same scenario under different seeds. [B3]"""
    r_mean: np.ndarray        # shape (K,), mean r per scan over runs
    r_std: np.ndarray         # shape (K,)
    n_runs: int
    seeds: tuple[int, ...]

def run_monte_carlo(cfg: RunConfig, n_runs: int, base_seed: int) -> MonteCarloResult:
    """Re-simulate and re-filter the same scenario over n_runs seeds and aggregate r_k.

    This is the empirical half of B3: the closed form predicts one r trajectory for one
    event sequence, but whether p_D and lambda_FA were modelled correctly shows up only in
    the average over many realisations. Each run derives its seed from base_seed so the set
    is reproducible. Kept deliberately small (tens of runs, 60 scans) - long trials belong
    to phase 2 on ROS 2. [B3]
    """
    raise NotImplementedError
```

### `crop_mot/analysis/metrics.py`

```python
@dataclass(frozen=True)
class RComparison:
    """Result of the B3 cross-check. [B3]"""
    max_abs_error: float
    rms_error: float
    first_divergence_k: int | None   # first k where |r_sim - r_ref| > tol, else None

def compare_r(r_sim: np.ndarray, r_ref: np.ndarray, tol: float = 1e-9) -> RComparison:
    """Compare a simulated r trajectory against the analytic one. [B3]"""
    raise NotImplementedError

def gospa(estimates: Sequence[TrackEstimate], truth_positions: np.ndarray,
          c: float, p: float = 2.0, alpha: float = 2.0) -> float:
    """GOSPA distance for the B4 multi-target comparison. Stub only in phase 1. [B4]"""
    raise NotImplementedError
```

### `crop_mot/filters/__init__.py` — how B4 plugs in

```python
FILTERS: dict[str, Callable[[FilterConfig], TrackingFilter]] = {
    "bernoulli": build_bernoulli,
    # B4, phase 2: "pda", "jpda", "gnn", "pmb", "pmbm" - one line each
}
```

Adding a B4 filter is: new file implementing the four methods, one line here, and it is
automatically covered by `test_filter_interface_contract.py`. The runner, the config loader,
the detection format and the plots are untouched.

---

## 4. Environment files

### `Dockerfile`

```dockerfile
# Ubuntu 22.04 with its SYSTEM python3.10 - the exact pair ROS 2 Humble uses, so phase 2
# only adds ROS on top. Deliberately NOT ros:humble: see README. No venv, because Humble's
# rclpy lives in the system interpreter.
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace
USER ${USERNAME}
WORKDIR /workspace
CMD ["bash"]
```

`MPLBACKEND=Agg` because the container is headless: plots are written into the run folder,
never displayed. `PYTHONPATH=/workspace` instead of an editable install, so nothing writes
`.egg-info` into the bind-mounted source tree and `python -m crop_mot` works with zero setup.

### `.devcontainer/devcontainer.json`

```json
{
  "name": "crop-mot (phase 1, no ROS)",
  "build": { "dockerfile": "../Dockerfile", "context": ".." },
  "workspaceFolder": "/workspace",
  "remoteUser": "dev",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-python.vscode-pylance", "charliermarsh.ruff"],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3",
        "python.testing.pytestEnabled": true,
        "python.testing.pytestArgs": ["tests"],
        "files.eol": "\n"
      }
    }
  },
  "postCreateCommand": "python3 -m pytest tests -q --collect-only"
}
```

No explicit `workspaceMount` is needed: because the repo lives in the WSL2 Linux filesystem
(phase 0, step 7), Dev Containers bind-mounts a native Linux path directly. `remoteUser: dev`
is UID 1000 and must match the distro user's `id -u`, or files created in the container will
be owned by the wrong user on the host side.

### `requirements.txt`

```
numpy>=1.24,<2.0        # <2.0: ROS 2 Humble's rclpy interops with apt NumPy 1.x
scipy>=1.10,<1.14       # linear_sum_assignment (GNN/JPDA), chi2 (gating), stats
matplotlib>=3.7,<4.0
PyYAML>=6.0             # configs; ROS 2 param files are YAML, Humble already ships it
pytest>=7.4
```

### `pyproject.toml`

Metadata (`name = "crop_mot"`, `requires-python = ">=3.10"`), setuptools backend, and
`[tool.pytest.ini_options]` + `[tool.ruff]`. **No `setup.py` yet** — it is added in phase 2,
since `ament_python` invokes it and setuptools accepts both files side by side.

### `.gitattributes`

```
* text=auto eol=lf
*.py     text eol=lf
*.yaml   text eol=lf
*.yml    text eol=lf
*.json   text eol=lf
*.jsonl  text eol=lf
*.md     text eol=lf
*.sh     text eol=lf
Dockerfile text eol=lf
*.png binary
*.gif binary
*.pdf binary
*.pptx binary
*.drawio binary
*.npz binary
```

LF is forced here rather than relying on `core.autocrlf`, so a checkout on Windows and a
build inside the Linux container see byte-identical files. With the repo now in WSL2 this is
belt-and-braces rather than load-bearing, but it still matters the moment the repo is cloned
on a Windows host — by you on another machine, or by a supervisor. Related rule enforced by design:
**no executable shell scripts as entry points** — everything is `python -m crop_mot ...`,
because Windows bind mounts cannot express the exec bit and `git update-index --chmod` drift
is a recurring annoyance. No hardcoded Windows paths anywhere; all paths are
`pathlib.Path` relative to the repo root or the run folder.

### `.dockerignore`

```
.git/
runs/
**/__pycache__/
.pytest_cache/
*.egg-info/
.venv/
.devcontainer/
*.pptx
*.drawio*
```

---

## 5. Example configs

### `configs/b1_two_rows.yaml`

```yaml
name: b1_two_rows
seed: 42

world:
  rows:                          # plant rows parallel to +y, robot walks the lane between
    - {x: -0.75, y_start: 0.0, y_end: 12.0, spacing: 0.35}
    - {x:  0.75, y_start: 0.0, y_end: 12.0, spacing: 0.35}
  position_jitter_std: 0.03      # planting irregularity, m

path:
  kind: straight_lane
  x: 0.0
  y_start: -1.0
  heading: 1.5707963             # +y, rad
  speed: 0.4                     # m/s
  n_scans: 60
  scan_period: 0.25              # s -> dt
  pose_known: true               # true: reported pose == true pose (B2/B3 assumption)
  yaw_wobble_std: 0.02           # rad,  IGNORED while pose_known: true
  xy_noise_std: 0.01             # m,    IGNORED while pose_known: true

sensor:                          # TRUTH detector. The filter gets its own assumed copy.
  fov: {min_range: 0.3, max_range: 4.0, half_angle: 0.6}   # wedge, rad
  detection:
    kind: constant               # 'constant' | 'range_dependent'
    p_D: 0.85
    # when kind: range_dependent, use instead:
    # p_D_near: 0.95
    # p_D_far: 0.55
    # occlusion_factor: 0.6
  lambda_FA: 2.0                 # expected clutter detections per scan
  measurement:
    kind: linear_xy
    R: [[0.04, 0.0], [0.0, 0.04]]    # m^2
```

### `configs/b2_bernoulli_phantom.yaml`

```yaml
name: b2_bernoulli_phantom
seed: 42
scenario: configs/b1_two_rows.yaml   # reuse B1 verbatim; no config-merge machinery

filter:
  kind: bernoulli
  motion:      {kind: static, q: 0.0}
  measurement: {kind: linear_xy, R: [[0.04, 0.0], [0.0, 0.04]]}
  assumed_sensor:                    # what the FILTER believes; may differ from truth
    fov: {min_range: 0.3, max_range: 4.0, half_angle: 0.6}
    detection: {kind: constant, p_D: 0.85}
    lambda_FA: 2.0
  birth:
    kind: single_from_measurement    # phantom: seed the track from a CLUTTER detection
    at_scan: 0
    r_b: 0.08
    init_cov: [[0.25, 0.0], [0.0, 0.25]]
  survival: {p_S: 1.0}               # static plants never disappear
  gate: {chi2_prob: 0.99}

analysis:
  b3_reference: bernoulli_existence  # the A2 closed form
  monte_carlo: {n_runs: 50}          # empirical backing; small by design in phase 1
  plots: [scene, r_vs_k, r_vs_analytic, r_montecarlo]
```

### Run folder produced

```
runs/2026-09-23T14-02-11_b2_bernoulli_phantom_seed42/
├── config.yaml                  verbatim copy of the config used
├── run_meta.json                seed, git sha, package versions, argv, timestamp
├── truth.jsonl                  plant positions + poses  (EVAL ONLY)
├── labels.jsonl                 per-detection origin ids (EVAL ONLY)
├── detections.jsonl             the ONLY filter input; shared by every filter
├── estimates_bernoulli.jsonl    one file per filter -> fair comparison by construction
├── metrics.json
└── plots/{scene.png, r_vs_k.png, r_vs_analytic.png}
```

---

## 6. Work-package mapping

| WP | Files | Interfaces / functions |
|---|---|---|
| **B1** simulator | `world/field.py`, `world/path.py`, `world/truth.py`, `sensor/fov.py`, `sensor/models.py`, `sensor/sensor_model.py`, `sensor/detector.py`, `sensor/record.py`, `runner/simulate.py`, `configs/b1_two_rows.yaml` | `Pose2D`, `FieldOfView`, `Detection`, `Scan`, `ScanLabels`, `MeasurementModel`/`LinearGaussianXY`, `SensorModel`, `sample_scan` |
| **B2** Bernoulli | `filters/base.py`, `filters/kalman.py`, `filters/bernoulli.py`, `filters/bernoulli_bank.py`, `filters/birth.py`, `motion/models.py`, `runner/track.py`, `analysis/plots.py::plot_r_vs_k`/`plot_hypotheses`, `analysis/candidates.py`, `configs/b2_bernoulli_phantom.yaml`, `configs/b2_bernoulli_bank_phantoms.yaml` | `TrackingFilter`, `BirthModel`, `SurvivalModel`, `MotionModel`/`StaticTarget`, `TrackEstimate` |
| **B3** validation | `analysis/analytic.py`, `analysis/events.py`, `analysis/montecarlo.py`, `analysis/metrics.py`, `analysis/plots.py::{plot_r_vs_analytic,plot_r_montecarlo}`, `tests/test_b3_analytic_bernoulli.py` | `AnalyticReference`, `ScanEvent` (per-scan p_D), `BernoulliExistenceReference`, `compare_r`, `RComparison`, `run_monte_carlo`, `MonteCarloResult` |
| **Extension slots** (stubbed, unused in B1–B4) | `world/path.py::generate_path` (pose_known), `sensor/sensor_model.py::RangeDependentPD`, `sensor/stereo.py` | `PoseSample`, `RangeDependentPD`, `StereoDepthModel` |
| **B4** multi-target (phase 2) | new `filters/pda.py`, `jpda.py`, `gnn.py`, `pmb.py`, `pmbm.py`; existing `association/{gating,assignment,murty}.py`; one line in `filters/__init__.py::FILTERS` | same `TrackingFilter` Protocol, unchanged runner; `gospa` in `metrics.py`; optional `HasDiagnostics` |
| **Baselines / fairness** | `sensor/record.py` (detections.jsonl), `runner/run_dir.py`, `rng.py`, `tests/test_filter_interface_contract.py` | shared detection file + one seed + one contract test across all `FILTERS` |

### Tests

- `test_b1_simulator.py` — same seed reproduces `detections.jsonl` byte-for-byte; no
  detection outside the FOV; empirical detection rate ≈ p_D; clutter counts ≈ Poisson(λ_FA);
  with `pose_known: true`, `Scan.pose` equals the true pose in `truth.jsonl` exactly.
- `test_b2_bernoulli.py` — r stays in [0,1]; r decreases monotonically under sustained
  misdetection; `predict`/`update` are pure (input state unchanged) and preserve shapes.
- **`test_b3_analytic_bernoulli.py::test_r_matches_analytic_recursion`** — the named B3
  cross-check: run the Bernoulli filter on a recorded scenario, build the `ScanEvent`
  sequence, evaluate `BernoulliExistenceReference`, assert `compare_r(...).max_abs_error`
  below tolerance. **Parametrised over `detection.kind` ∈ {`constant`, `range_dependent`}**,
  so the same closed form is exercised under both p_D profiles.
- `test_b3_analytic_bernoulli.py::test_monte_carlo_mean_r_brackets_analytic` — marked
  `@pytest.mark.slow`: the mean r over N seeds must lie within a few standard errors of the
  closed form. This is the empirical half of B3 and is expected to be the slowest test.
- `test_filter_interface_contract.py` — parametrised over every entry in `FILTERS`; asserts
  the four methods exist, `extract` returns valid `TrackEstimate`s with r ∈ [0,1] and
  consistent shapes. This is what keeps B4 additive.
- `test_io_roundtrip.py` — `Scan` / `TrackEstimate` → JSONL → back, exactly.

---

## 7. Phase-2 migration plan

**Untouched:** everything under `crop_mot/` except `__main__.py` (which merely gains no new
responsibility — the node calls the library directly), all of `tests/`, all of `configs/`,
`.gitattributes`, `requirements.txt`.

1. **Add `package.xml`** at the repo root: `<export><build_type>ament_python</build_type></export>`,
   `<exec_depend>` on `rclpy`, `python3-numpy`, `python3-scipy`, `python3-yaml`.
   `resource/crop_mot` already exists from phase 1, so nothing moves.
2. **Add `setup.py`** mirroring `pyproject.toml` metadata, with the `data_files` entries
   `ament_python` expects (`share/ament_index/resource_index/packages`, `share/crop_mot/package.xml`).
   setuptools tolerates `setup.py` and `pyproject.toml` together.
3. **Add `Dockerfile.humble`**: `FROM ros:humble-ros-base-jammy`, then the *same*
   `pip install -r requirements.txt`. Because the base is also Ubuntu 22.04 / Python 3.10,
   no dependency resolves differently; the `numpy<2` pin is what protects `rclpy` interop.
   Add a second devcontainer (`.devcontainer/humble/devcontainer.json`) pointing at it.
4. **Clone the repo into a colcon workspace** at `~/ros2_ws/src/crop_mot/` (or bind-mount it
   there). No files inside the repo move; the repo *is* the ament package.
5. **Fill in `ros2_adapter/`** as a node: subscribe to the detection topic → convert ROS
   messages to `Detection`/`Scan`/`Pose2D` dataclasses → call the *same*
   `flt.predict / flt.update / flt.extract` → convert `TrackEstimate` back to a ROS message.
   The conversion functions are the only new code, and they are the only code that imports
   `rclpy`. Optionally add a sibling `crop_mot_msgs` package for the message definitions.
6. **Point the node's ROS parameter file at the existing `configs/*.yaml`** — same format,
   so simulator and robot are configured identically.
7. **Verify** by running `pytest tests -q` inside the Humble container: the core tests must
   pass unchanged. That is the migration's acceptance criterion.

---

## 8. Verification (phase 1)

```bash
# inside the dev container
python3 -m crop_mot simulate --config configs/b1_two_rows.yaml         # B1 -> run folder
python3 -m crop_mot track    --config configs/b2_bernoulli_phantom.yaml --filter bernoulli
python3 -m crop_mot analyse  --run runs/<stamp>_b2_bernoulli_phantom_seed42
python3 -m pytest tests -q
```

Because this delivery is a skeleton, the acceptance criterion is: **the package imports, the
CLI resolves all three subcommands, `pytest --collect-only` collects every named test, and
every test fails with `NotImplementedError` rather than `ImportError` or `AttributeError`.**
That proves the wiring is complete and only the math is missing. `test_filter_interface_contract.py`
must already pass structurally against `FILTERS["bernoulli"]`.

---

## 9. Decisions recorded (all open questions now closed)

| # | Question | Decision | Where it lands |
|---|---|---|---|
| 1 | Repo location | WSL2 Linux filesystem, not OneDrive | §0; follow-up: create a git remote, since leaving OneDrive removes your current backup |
| 2 | Gait wobble | **Stub it**, behind a `pose_known: bool` | `world/path.py::generate_path`, `PoseSample`; `Scan.pose` is now explicitly the *reported* pose |
| 3 | Phantom birth | Seed the Bernoulli from a clutter detection at a chosen scan, watch r decay | `filters/birth.py::single_from_measurement`, `configs/b2_bernoulli_phantom.yaml` |
| 4 | p_D profile | **Both**, switchable, and validatable either way | `ConstantPD` + `RangeDependentPD`; `ScanEvent` carries **per-scan** p_D so one closed form covers both; B3 test parametrised over the two |
| 5 | B3 evidence | Closed form **plus** a small Monte-Carlo sim | `analysis/montecarlo.py`, `plot_r_montecarlo`, `test_monte_carlo_mean_r_brackets_analytic` |
| 6 | Detection log | JSONL, short phase-1 trials | `sensor/record.py`; long trials are a phase-2/ROS 2 concern, so no format abstraction is built now |
| 7 | Package name | `crop_mot` | valid ROS 2 package name, becomes the import name in the appendix |
| 8 | 3D / stereo | Nice-to-have, stubbed at the back of the pipeline | `sensor/stereo.py` only; nothing in B1–B4 imports it |

Two consequences worth stating explicitly, because they are the non-obvious part of
decisions 4 and 6:

- **Per-scan p_D is what keeps the range-dependent case honest.** Had `BernoulliExistenceReference`
  stored a single p_D, validating the range-dependent profile would have required a second
  derivation. Carrying p_D on each `ScanEvent` means the constant case is just the special
  case where every entry is equal — one implementation, two profiles, no duplicated algebra.
- **JSONL is a phase-1 decision, deliberately not future-proofed.** Long trials arrive with
  ROS 2, where the natural recording format is a rosbag anyway — so building an `.npz`
  backend now would be generality that gets thrown away. `sensor/record.py` is one file to
  swap if that turns out wrong.

Still open on your roadmap but **not blocking this skeleton**: C1 (2D vs 3D) — the skeleton
is 2D with n-D arrays, so 3D means new model files, not a rewrite; C4 (JPDA vs PMBM as the
core method) — both are B4 files behind the same interface, so the skeleton is neutral.

---

## 10. Planned `TODO(human)` (learning hand-off at implementation time)

One thing in this skeleton depends on *your* A2 derivation and should not be guessed:
the branch structure of `BernoulliExistenceReference.r_sequence` in
`crop_mot/analysis/analytic.py` — the misdetection, detection and birth cases written out
in terms of `ScanEvent` fields, and any field I have not anticipated.

I have specified the fields I am confident about (`p_D`, `lambda_FA`, `in_fov`, `n_gated`,
`n_clutter_gated`), because making p_D per-scan was forced by decision 4. What I will leave
as the single `TODO(human)` is the docstring of `r_sequence` stating the three branches and
the `ScanEvent` fields each one consumes. That is the point where your hand derivation meets
the code, and writing it yourself is what keeps B3 a genuine cross-check rather than a
restatement of whatever the filter already does.
