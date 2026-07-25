"""Minimal rotation conversions (replaces the pytorch3d.transforms subset used
by the FLAME stack). Formulas follow Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks", CVPR 2019, and standard
axis-angle (Rodrigues) conversion; numerically equivalent to pytorch3d.
"""
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    eps = 1e-8
    axis = axis_angle / (angles + eps)
    cos, sin = torch.cos(angles)[..., None], torch.sin(angles)[..., None]
    x, y, z = axis[..., 0:1, None], axis[..., 1:2, None], axis[..., 2:3, None]
    zero = torch.zeros_like(x)
    K = torch.cat([
        torch.cat([zero, -z, y], dim=-1),
        torch.cat([z, zero, -x], dim=-1),
        torch.cat([-y, x, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(K.shape)
    return eye + sin * K + (1.0 - cos) * (K @ K)
