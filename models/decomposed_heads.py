"""Decomposed output heads for V4 denoiser.

Separate prediction heads for each motion channel:
  exp (50D), jaw (3D), neck_aa (3D direct), alphas (K)
  + auxiliary: gaze (3D Euler), rhythmic (3D Euler)

Neck is predicted directly as axis-angle (no Euler->AA conversion,
which has a gradient singularity at identity rotation).
Gaze and rhythmic heads are auxiliary supervision only.
"""

import torch
import torch.nn as nn

from utils.rotation import euler_xyz_to_axis_angle


class DecomposedHeads(nn.Module):
    """Decomposed output heads: separate prediction for each motion channel.

    neck_aa is predicted directly (not composed from gaze+rhythmic) to avoid
    the Euler->axis-angle singularity. Gaze and rhythmic are auxiliary heads
    supervised by decomposition GT but don't feed into the motion output.
    """

    def __init__(self, feature_dim, n_basis=4, include_eye=False, alpha_bias_init=0.0):
        super().__init__()
        half = feature_dim // 2
        self.include_eye = include_eye

        self.exp_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 50))
        self.jaw_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))
        self.neck_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))
        self.gaze_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))
        self.rhythmic_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))
        if include_eye:
            self.eye_head = nn.Sequential(
                nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))
        self.alpha_head = nn.Linear(feature_dim, n_basis)
        # Warm-start alpha bias: sigmoid(1.5) = 0.82, basis starts strongly active
        if alpha_bias_init != 0.0:
            nn.init.constant_(self.alpha_head.bias, alpha_bias_init)

    def forward(self, feat):
        """
        Args:
            feat: (N, L, D) transformer output

        Returns:
            decomposed dict with all channels + motion output + alphas
        """
        exp = self.exp_head(feat)           # (N, L, 50)
        jaw = self.jaw_head(feat)           # (N, L, 3)
        neck_aa = self.neck_head(feat)      # (N, L, 3) direct axis-angle
        gaze = self.gaze_head(feat)         # (N, L, 3) Euler degrees (auxiliary)
        rhythmic = self.rhythmic_head(feat) # (N, L, 3) Euler degrees (auxiliary)
        alphas = self.alpha_head(feat)      # (N, L, K)

        parts = [exp, jaw, neck_aa]
        if self.include_eye:
            eye = self.eye_head(feat)       # (N, L, 3)
            parts.append(eye)

        motion = torch.cat(parts, dim=-1)   # (N, L, 56 or 59)

        result = {
            'motion': motion,
            'exp': exp,
            'jaw': jaw,
            'neck_aa': neck_aa,
            'gaze': gaze,
            'rhythmic': rhythmic,
            'alphas': alphas,
        }
        if self.include_eye:
            result['eye'] = eye
        return result


class DecomposedHeadsV2(nn.Module):
    """V4.4c: Gaze + rhythmic Euler are PRIMARY neck output with Conv1D smoothing.

    Unlike DecomposedHeads where neck_aa is independently predicted and gaze/rhythmic
    are auxiliary, here gaze and rhythmic are the primary decomposition and neck_aa
    is DERIVED via euler_xyz_to_axis_angle(gaze + rhythmic).

    Conv1D temporal smoothing (kernel=5, ±2 frames) on gaze and rhythmic heads
    eliminates per-frame jitter — same mechanism as kinematic_diffusion ResBlock1D.
    """

    def __init__(self, feature_dim, n_basis=4, conv_kernel=5, include_eye=False):
        super().__init__()
        half = feature_dim // 2
        self.include_eye = include_eye

        # Exp and jaw: same MLP heads as before
        self.exp_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 50))
        self.jaw_head = nn.Sequential(
            nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))

        # Gaze: Linear -> Conv1d -> SiLU -> Conv1d -> 3D Euler
        pad = conv_kernel // 2
        self.gaze_conv = nn.Sequential(
            nn.Linear(feature_dim, half),
            _Transpose12(),  # (N, L, D) -> (N, D, L) for Conv1d
            nn.Conv1d(half, half, conv_kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(half, 3, conv_kernel, padding=pad),
            _Transpose12(),  # (N, 3, L) -> (N, L, 3)
        )

        # Rhythmic: same Conv1d structure
        self.rhythmic_conv = nn.Sequential(
            nn.Linear(feature_dim, half),
            _Transpose12(),
            nn.Conv1d(half, half, conv_kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(half, 3, conv_kernel, padding=pad),
            _Transpose12(),
        )

        # Eye head (optional, for 59D output)
        if include_eye:
            self.eye_head = nn.Sequential(
                nn.Linear(feature_dim, half), nn.GELU(), nn.Linear(half, 3))

        # Alpha head for style basis
        self.alpha_head = nn.Linear(feature_dim, n_basis)

    def forward(self, feat):
        """
        Args:
            feat: (N, L, D) transformer output

        Returns:
            decomposed dict with gaze/rhythmic as PRIMARY, neck_aa DERIVED
        """
        exp = self.exp_head(feat)                  # (N, L, 50)
        jaw = self.jaw_head(feat)                  # (N, L, 3)
        gaze = self.gaze_conv(feat)                # (N, L, 3) Euler degrees
        rhythmic = self.rhythmic_conv(feat)        # (N, L, 3) Euler degrees
        alphas = self.alpha_head(feat)             # (N, L, K)

        # Compose neck: gaze + rhythmic -> Euler -> axis-angle
        neck_euler = gaze + rhythmic               # (N, L, 3) additive Euler
        neck_aa = euler_xyz_to_axis_angle(neck_euler)  # (N, L, 3) differentiable

        parts = [exp, jaw, neck_aa]
        if self.include_eye:
            eye = self.eye_head(feat)              # (N, L, 3)
            parts.append(eye)

        motion = torch.cat(parts, dim=-1)          # (N, L, 56 or 59)

        result = {
            'motion': motion,
            'exp': exp,
            'jaw': jaw,
            'neck_aa': neck_aa,
            'gaze': gaze,
            'rhythmic': rhythmic,
            'alphas': alphas,
        }
        if self.include_eye:
            result['eye'] = eye
        return result


class _Transpose12(nn.Module):
    """Swap dims 1 and 2: (N, L, D) <-> (N, D, L) for Conv1d."""
    def forward(self, x):
        return x.transpose(1, 2)
