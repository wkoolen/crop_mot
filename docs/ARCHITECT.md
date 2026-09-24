# Role
You are a software architect. Design a **framework skeleton only** for an MSc thesis
simulator. Do NOT implement any algorithm. Every function body is `raise NotImplementedError`
plus a docstring saying what it must do, its inputs/outputs, and which thesis item it serves.

# Context (read carefully)
- Thesis: multi-object tracking / data association of static plants in crop rows, seen by a
  camera on a quadruped. Estimation and filtering is the core; vision is stubbed as a noisy
  black-box detector (detection probability p_D, clutter rate lambda_FA, Gaussian noise).
- The user is a Systems & Control student, not a software engineer. Code must be readable
  by someone who knows Kalman filters and Bayes, not design patterns. Prefer plain classes,
  dataclasses and typing.Protocol over deep inheritance, plugin registries or metaclasses.
- Math notation must match the thesis: state x, measurement z, likelihood g(z|x),
  existence probability r, p_D, lambda_FA, clutter density c(z).

# Scope: work packages B1 to B4
- **B1** Simulator: 2D static objects (plants in rows), a robot path (known poses),
  a detector that misses with probability 1 - p_D and adds Poisson clutter with rate lambda_FA
  inside the field of view.
- **B2** Bernoulli filter over B1 output (single target with existence r). Output: r per
  scan, plus a plot of r decaying for a phantom (clutter-born) track.
- **B3** Validation: compare the simulated r trajectory against a closed-form hand-derived
  expression (misdetection / detection / birth update). Needs a clean hook to plug in an
  analytic reference and a comparison metric/plot.
- **B4** Extension slots for multi-target filters: PDA (many measurements, one target),
  JPDA (many measurements, many targets), PMB and PMBM. Only the interfaces and where
  they plug in. B4 is implemented in phase 2, so the design must make adding them a matter
  of adding a file, not changing the pipeline.
- Also plan for baselines (GNN) behind the same filter interface, so all methods run on the
  **same recorded detections with the same seed** for a fair comparison.

# Environment
- Phase 1 (now): Windows 11 host, Docker Desktop (WSL2 backend), developed with
  VS Code Dev Containers. Python only, dependencies: numpy, scipy, matplotlib, pytest
  (justify anything else).
- Phase 2 (later): Docker image based on Ubuntu 22.04 + ROS 2 Humble, Python.
- Migration must be cheap. Therefore:
  1. Base the phase-1 image on Ubuntu 22.04 with its system Python 3.10 (what Humble uses),
     so phase 2 only adds ROS on top. Explain the choice vs. using `ros:humble` from day one.
  2. The core package must never import rclpy or any ROS message type. All data crossing
     module boundaries uses plain dataclasses (e.g. Pose2D, Detection, Scan, TrackEstimate).
  3. Lay out the package so it can later become an `ament_python` package
     (package.xml, setup.py, resource/ marker) without moving core files.
  4. Show where a thin ROS 2 adapter layer will live (a node that converts ROS messages to
     the core dataclasses and calls the same filter step). Only a placeholder folder and a
     README describing it; no ROS code now.
  5. Paths, line endings and file permissions must work on both Windows-mounted volumes
     and Linux (.gitattributes, no hardcoded Windows paths).

# Design requirements
- Pipeline: scenario config -> world/truth -> sensor model -> detections (recordable to disk)
  -> filter (predict / update per scan) -> estimates log -> analysis/plots.
- Separate: truth generation, sensor model, filter, and evaluation. The filter must never
  see ground truth.
- One filter interface that Bernoulli, PDA, JPDA, GNN, PMB and PMBM can all satisfy.
  Think about what differs (single vs. multi target, hypotheses) and state how the
  interface handles that without special cases in the runner.
- Reproducibility: every run is defined by one config file (YAML or TOML) plus a seed;
  outputs go to a run folder containing the config copy, detections, estimates and plots.
- Motion/measurement models (e.g. static target, linear Gaussian measurement) are
  swappable objects, with the assumption stated in the docstring.
- Tests: pytest skeletons, including a named test for the B3 analytic cross-check.

# Deliverables (in this order)
1. Short design rationale (max 1 page): main decisions and trade-offs, in plain language.
2. Folder tree with a one-line purpose per file.
3. Interface definitions: dataclasses and Protocols with full signatures and docstrings,
   bodies `NotImplementedError`.
4. Dockerfile, .devcontainer/devcontainer.json, requirements/pyproject, .gitattributes,
   .dockerignore.
5. Example config for a B1 scenario and a B2 run.
6. Mapping table: B1/B2/B3/B4 -> files/interfaces that serve it.
7. Phase-2 migration plan: numbered steps from this repo to a Humble container, listing
   exactly what gets added and what stays untouched.
8. Open questions you could not decide, listed for the user.

# Do not
- Implement filters, simulators or math.
- Add ROS dependencies, async code, web UIs, databases or ML libraries.
- Over-generalise: design for B1 to B4 and the listed baselines only.