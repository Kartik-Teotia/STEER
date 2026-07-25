"""Differentiable rotation utilities for decomposition-informed losses."""

import torch
import math


def axis_angle_to_rotation_matrix(axis_angle):
    """Convert axis-angle to rotation matrix via Rodrigues formula.

    Args:
        axis_angle: (..., 3) axis-angle vectors

    Returns:
        R: (..., 3, 3) rotation matrices
    """
    shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    angle = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    axis = aa / angle  # (N, 3)

    cos_a = torch.cos(angle).squeeze(-1)  # (N,)
    sin_a = torch.sin(angle).squeeze(-1)  # (N,)
    one_minus_cos = 1.0 - cos_a

    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]

    # Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
    R = torch.stack([
        cos_a + kx * kx * one_minus_cos,
        kx * ky * one_minus_cos - kz * sin_a,
        kx * kz * one_minus_cos + ky * sin_a,

        ky * kx * one_minus_cos + kz * sin_a,
        cos_a + ky * ky * one_minus_cos,
        ky * kz * one_minus_cos - kx * sin_a,

        kz * kx * one_minus_cos - ky * sin_a,
        kz * ky * one_minus_cos + kx * sin_a,
        cos_a + kz * kz * one_minus_cos,
    ], dim=-1).reshape(-1, 3, 3)

    return R.reshape(*shape, 3, 3)


def rotation_matrix_to_euler_xyz(R):
    """Extract intrinsic XYZ Euler angles from rotation matrix.

    Matches scipy's Rotation.as_euler('xyz', degrees=True) convention:
    R = Rz(roll) @ Ry(yaw) @ Rx(pitch)

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        euler: (..., 3) angles in degrees as (pitch_x, yaw_y, roll_z)
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    # For intrinsic xyz (R = Rz @ Ry @ Rx):
    # R[2,0] = -sin(y)
    # R[2,1] = cos(y)*sin(x)
    # R[2,2] = cos(y)*cos(x)
    # R[0,0] = cos(y)*cos(z)
    # R[1,0] = cos(y)*sin(z)
    sy = -R[:, 2, 0]
    yaw = torch.asin(sy.clamp(-0.99999, 0.99999))

    cos_y = torch.cos(yaw)
    # Avoid division by zero at gimbal lock
    safe = cos_y.abs() > 1e-6

    pitch = torch.where(
        safe,
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
        torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    )
    roll = torch.where(
        safe,
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        torch.zeros_like(yaw)
    )

    euler = torch.stack([pitch, yaw, roll], dim=-1)
    euler = euler * (180.0 / math.pi)

    return euler.reshape(*shape, 3)


def axis_angle_to_euler_xyz(axis_angle):
    """Convert axis-angle to Euler XYZ angles in degrees.

    Matches the decomposition convention:
    pitch_x, yaw_y, roll_z = Rotation.from_rotvec(aa).as_euler('xyz', degrees=True)

    Args:
        axis_angle: (..., 3) axis-angle vectors

    Returns:
        euler: (..., 3) as (pitch_x, yaw_y, roll_z) in degrees
    """
    R = axis_angle_to_rotation_matrix(axis_angle)
    return rotation_matrix_to_euler_xyz(R)


def euler_xyz_to_rotation_matrix(euler_deg):
    """Convert intrinsic XYZ Euler angles (degrees) to rotation matrix.

    R = Rz(roll) @ Ry(yaw) @ Rx(pitch)

    Args:
        euler_deg: (..., 3) as (pitch_x, yaw_y, roll_z) in degrees

    Returns:
        R: (..., 3, 3) rotation matrices
    """
    shape = euler_deg.shape[:-1]
    euler = euler_deg.reshape(-1, 3) * (math.pi / 180.0)

    cx, cy, cz = torch.cos(euler[:, 0]), torch.cos(euler[:, 1]), torch.cos(euler[:, 2])
    sx, sy, sz = torch.sin(euler[:, 0]), torch.sin(euler[:, 1]), torch.sin(euler[:, 2])

    R = torch.stack([
        cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz,
        cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz,
        -sy,     sx * cy,                cx * cy,
    ], dim=-1).reshape(-1, 3, 3)

    return R.reshape(*shape, 3, 3)


def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle (gradient-safe at identity).

    Uses Taylor expansion for small angles to avoid the acos singularity
    at θ=0 where d(acos)/dx → −∞. This is critical when gaze+rhythmic
    Euler predictions start near zero during early training.

    For small θ: axis_angle ≈ skew(R) / 2  (first-order approximation)
    For large θ: standard Rodrigues via acos + skew / sin

    Args:
        R: (..., 3, 3) rotation matrices

    Returns:
        axis_angle: (..., 3) axis-angle vectors
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    # Skew-symmetric part (always numerically stable)
    skew = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)  # (N, 3)

    # angle from trace: cos(θ) = (trace(R) - 1) / 2
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)

    # Small-angle path (θ < ~0.01 rad): aa ≈ skew/2
    # Large-angle path: aa = (θ / sin(θ)) * skew/2
    # Blend smoothly to avoid branching in backward pass
    angle_sq = (1.0 - cos_angle) * 2.0  # ≈ θ² for small θ
    # θ/sin(θ) ≈ 1 + θ²/6 + ... for small θ, using Padé-like stable form:
    # factor = θ / (2*sin(θ)) → 0.5 * θ/sin(θ)
    # For small θ: factor ≈ 0.5 * (1 + θ²/6)
    # For large θ: factor = acos(cos_θ) / (2 * sin(acos(cos_θ)))
    # Use: sin²(θ) = 1 - cos²(θ), so sin(θ) = sqrt(1 - cos²(θ))
    sin_angle = torch.sqrt((1.0 - cos_angle * cos_angle).clamp(min=1e-12))

    # Stable factor: for small angles → 0.5, for large → θ/(2*sinθ)
    # acos is only used when sin_angle is large enough (angle > ~0.01)
    small = sin_angle < 1e-4
    # Small angle: factor = 0.5 * (1 + angle_sq/12)  [Taylor of θ/(2sinθ)]
    factor_small = 0.5 + angle_sq / 12.0
    # Large angle: factor = acos(cos_angle) / (2 * sin_angle)
    factor_large = torch.acos(cos_angle.clamp(-0.99999, 0.99999)) / (2.0 * sin_angle + 1e-8)

    factor = torch.where(small, factor_small, factor_large).unsqueeze(-1)

    aa = factor * skew
    return aa.reshape(*shape, 3)


def euler_xyz_to_axis_angle(euler_deg):
    """Convert Euler XYZ (degrees) to axis-angle.

    Args:
        euler_deg: (..., 3) as (pitch_x, yaw_y, roll_z) in degrees

    Returns:
        axis_angle: (..., 3) axis-angle vectors
    """
    R = euler_xyz_to_rotation_matrix(euler_deg)
    return rotation_matrix_to_axis_angle(R)
