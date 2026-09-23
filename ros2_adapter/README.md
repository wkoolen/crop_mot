# ROS 2 adapter — phase 2 placeholder

**There is no code in this folder, and there must not be any in phase 1.** The phase-1
container has no `rclpy` installed, so an accidental `import rclpy` anywhere in `crop_mot/`
fails immediately. That absence is the enforcement mechanism for "the core package never
imports ROS" — not discipline, and not a linter rule.

## What will live here

One node, whose entire job is conversion:

```
   ROS topic (detections)
            |
            v
   [ adapter node ]  <-- the only file in the repo that imports rclpy
            |  converts messages -> crop_mot.types dataclasses
            v
   flt.predict(state, dt)        <-- the SAME filter object as the simulator uses
   flt.update(state, scan)
   flt.extract(state)
            |  converts TrackEstimate -> ROS messages
            v
   ROS topic (tracks)
```

The node owns the filter state between callbacks and computes `dt` from message timestamps,
exactly as `crop_mot.runner.track.run_filter` does from scan timestamps.

## Why this is cheap

Nothing here re-implements anything. The conversion functions are the only new code:

| ROS side | Core side |
|---|---|
| `geometry_msgs/PoseStamped` (or a TF lookup) | `crop_mot.types.Pose2D` |
| a detection array message | `crop_mot.types.Detection`, `Scan` |
| a track array message | `crop_mot.types.TrackEstimate` |

Because every type crossing a module boundary in `crop_mot/` is already a plain dataclass
of floats and numpy arrays, there is no ROS-shaped object to unpick.

## Configuration

The node reads the **same** `configs/*.yaml` files the simulator uses. ROS 2 parameter files
are YAML, which is why PyYAML was chosen over TOML in phase 1 — the simulator and the robot
end up configured identically, from one file format.

## What does not change

Everything in `crop_mot/`, all of `tests/`, all of `configs/`. The migration's acceptance
criterion is that `pytest tests -q` passes unchanged inside the Humble container.

## Steps, when phase 2 starts

See §7 of the design plan. In short: add `package.xml` and `setup.py` at the repo root
(`resource/crop_mot` already exists), add a `Dockerfile.humble` based on
`ros:humble-ros-base-jammy` running the same `pip install -r requirements.txt`, clone the
repo into a colcon workspace, and fill in this folder.

The `numpy<2` pin in `requirements.txt` exists for this step: Humble's `rclpy` does its
array interop against the apt-installed NumPy 1.x, and a pip NumPy 2.x in the same
interpreter breaks it.
