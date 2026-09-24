"""Field-of-view geometry: the wedge the camera sees. [B1]

ASSUMPTION: the camera footprint is a flat 2D wedge (an annular sector) in the body frame,
symmetric about the robot's heading. This matches the top-down scenario sketch, where the
quadruped walks up a lane and sees a sector of the two rows ahead of it.

A 3D version would be a frustum and would live in a new file next to this one, because
nothing here is imported by the filters.
"""

from __future__ import annotations

import numpy as np

from crop_mot.types import FieldOfView, Pose2D


def in_fov(x: np.ndarray, pose: Pose2D, fov: FieldOfView) -> bool:
    """Is a world point inside the field of view from this pose?

    Transforms x into the body frame using the given pose, then tests the range against
    [min_range, max_range] and the bearing against [-half_angle, +half_angle].

    Which pose is passed matters and is the caller's responsibility:
      * the simulator passes the TRUE pose, because visibility is a physical fact;
      * a filter passes the REPORTED pose, because that is all it knows.
    With pose_known = True these coincide.

    Serves: [B1] deciding which plants can be detected; [B2] the filter's p_D, which is 0
    outside the FOV.

    Args:
        x: shape (dim_x,), a world-frame position in metres.
        pose: the pose to view from.
        fov: the wedge parameters.

    Returns:
        True if x lies inside the wedge (boundaries included).
    """
    # World -> body frame: translate by the pose, then rotate by -theta.
    dx = x[0] - pose.x
    dy = x[1] - pose.y
    x_body = np.cos(pose.theta) * dx + np.sin(pose.theta) * dy
    y_body = -np.sin(pose.theta) * dx + np.cos(pose.theta) * dy

    rho = np.hypot(x_body, y_body)
    bearing = np.arctan2(y_body, x_body)
    return bool(fov.min_range <= rho <= fov.max_range and abs(bearing) <= fov.half_angle)


def sample_uniform_in_fov(
    pose: Pose2D, fov: FieldOfView, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n points uniformly over the FOV wedge, in world coordinates.

    Uniform over AREA, not over (range, bearing): sampling range uniformly would bunch
    points near the robot and would make the clutter density non-uniform, contradicting
    the uniform clutter density c(z) = 1 / fov.area() (clutter intensity lambda_FA * c(z))
    which the Bernoulli update assumes [A0 §Measurement model]. Sample
    r = sqrt(U * (r_max^2 - r_min^2) + r_min^2) to get it right.

    Serves: [B1] generating Poisson clutter inside the FOV.

    Args:
        pose: the pose to sample relative to (the simulator passes the TRUE pose).
        fov: the wedge parameters.
        n: how many points to draw.
        rng: the "clutter" substream generator.

    Returns:
        Shape (n, dim_x) array of world-frame positions.
    """
    u = rng.random(n)
    rho = np.sqrt(u * (fov.max_range**2 - fov.min_range**2) + fov.min_range**2)
    bearing = rng.uniform(-fov.half_angle, fov.half_angle, size=n)

    angle = pose.theta + bearing
    points = np.empty((n, 2))
    points[:, 0] = pose.x + rho * np.cos(angle)
    points[:, 1] = pose.y + rho * np.sin(angle)
    return points
