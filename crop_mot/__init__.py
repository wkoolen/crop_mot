"""crop_mot - multi-object tracking of static plants in crop rows, seen from a quadruped.

This package is PURE PYTHON and must never import rclpy or any ROS message type. Everything
crossing a module boundary is a plain dataclass from `crop_mot.types`. The phase-2 ROS 2
adapter converts messages to those dataclasses and calls the same filter methods; see
`ros2_adapter/README.md`.

Deliberately imports nothing heavy at package level, so `import crop_mot` stays cheap and a
broken optional module cannot break the whole package.
"""

__version__ = "0.1.0"
