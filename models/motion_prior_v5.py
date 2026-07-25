"""V5 Motion Prior: Flow Matching + DiT + RoPE + DecomposedHeadsV2.

All SOTA improvements in one model:
  - Rectified flow matching (linear interpolation, Euler ODE sampling)
  - DiT blocks with AdaLN-Zero (combined time + style + label conditioning)
  - Rotary Positional Embeddings (RoPE) — replaces learned PE
  - DecomposedHeadsV2 with Conv1D temporal smoothing
  - Logit-normal timestep sampling (concentrates training on informative t)
  - Binary CFG on labels (style always active)

Architecture:
  [person_token(1), prev_motion(10), noisy_motion(100)] + RoPE
  -> N x DiTBlock(AdaLN-Zero with combined cond)
  -> DecomposedHeadsV2 -> motion(56) + gaze(3) + rhythmic(3) + alphas(K)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decomposed_heads import DecomposedHeads, DecomposedHeadsV2
from models.style_encoder_vae import StyleEncoderVAE


# ── Timestep Embedding ────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """Continuous timestep -> feature vector via sinusoidal encoding + MLP."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        """
        Args:
            t: (N,) float in [0, 1]
        Returns:
            emb: (N, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ── Rotary Positional Embedding ───────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Precompute RoPE frequencies for a given head dimension."""

    def __init__(self, dim, max_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, seq_len, device=None):
        """Returns (seq_len, dim) cos/sin frequency pairs."""
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
        return emb


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, freqs):
    """Apply RoPE to Q and K.

    Args:
        q, k: (N, H, L, D_h)
        freqs: (L, D_h)
    """
    cos = freqs.cos()[None, None, :, :]  # (1, 1, L, D_h)
    sin = freqs.sin()[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── Multi-Head Cross-Attention with RoPE ──────────────────────────────────

class RoPECrossAttention(nn.Module):
    """Cross-attention with RoPE on Q and K for temporal alignment.

    Q (from motion) and K (from audio) get positional encoding so that
    motion frame t preferentially attends to audio frame t.
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input, rope_freqs_q, rope_freqs_k):
        """
        Args:
            q_input:  (N, L_q, D)  — motion tokens (query)
            kv_input: (N, L_kv, D) — audio tokens (key/value)
            rope_freqs_q: (L_q, head_dim)  — RoPE for motion positions
            rope_freqs_k: (L_kv, head_dim) — RoPE for audio positions (offset)
        Returns:
            (N, L_q, D)
        """
        N, L_q, D = q_input.shape
        L_kv = kv_input.shape[1]

        q = self.q_proj(q_input).reshape(N, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).reshape(N, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).reshape(N, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K separately (different sequence lengths)
        cos_q = rope_freqs_q.cos()[None, None, :, :]
        sin_q = rope_freqs_q.sin()[None, None, :, :]
        q = q * cos_q + _rotate_half(q) * sin_q

        cos_k = rope_freqs_k.cos()[None, None, :, :]
        sin_k = rope_freqs_k.sin()[None, None, :, :]
        k = k * cos_k + _rotate_half(k) * sin_k

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(N, L_q, D)
        return self.out_proj(out)


# ── Multi-Head Self-Attention with RoPE ───────────────────────────────────

class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with RoPE on Q/K."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_freqs):
        """
        Args:
            x: (N, L, D)
            rope_freqs: (L, head_dim)
        """
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, H, L, D_h)
        q, k, v = qkv.unbind(0)

        q, k = apply_rotary_emb(q, k, rope_freqs)

        x = F.scaled_dot_product_attention(q, k, v)  # (N, H, L, D_h)
        x = x.transpose(1, 2).reshape(N, L, D)
        return self.proj(x)


# ── DiT Block with AdaLN-Zero ────────────────────────────────────────────

class DiTBlock(nn.Module):
    """DiT-style transformer block: AdaLN-Zero + RoPE self-attention + MLP.

    Conditioning vector c modulates both attention and MLP paths.
    Gate parameters initialized to zero → block starts as identity.
    Optional cross-attention for audio conditioning (added in Stage 2).
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, cross_attn=False,
                 dual_cross_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = RoPESelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        # AdaLN-Zero: 6 modulation params (shift, scale, gate) x 2 paths
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init so block starts as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Optional single cross-attention for audio (backward compat)
        self.has_cross_attn = cross_attn and not dual_cross_attn
        if self.has_cross_attn:
            self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn_layer = nn.MultiheadAttention(
                dim, n_heads, batch_first=True)
            self.cross_gate = nn.Parameter(torch.zeros(1, 1, dim))

        # Dual cross-attention: self-audio (lip sync) + partner-audio (reactions)
        # Uses RoPE cross-attention for temporal alignment (motion frame t → audio frame t)
        self.has_dual_cross_attn = dual_cross_attn
        if dual_cross_attn:
            self.self_audio_norm = nn.LayerNorm(dim, eps=1e-6)
            self.self_audio_cross_attn = RoPECrossAttention(dim, n_heads)
            # Gate: fixed scale (not learnable) avoids chicken-and-egg collapse
            # Set via audio_gate_scale config; default to learnable 0.02 for backward compat
            self.self_audio_gate = nn.Parameter(torch.ones(1, 1, dim) * 0.02)

            self.partner_audio_norm = nn.LayerNorm(dim, eps=1e-6)
            self.partner_audio_cross_attn = RoPECrossAttention(dim, n_heads)
            self.partner_audio_gate = nn.Parameter(torch.ones(1, 1, dim) * 0.02)

    def forward(self, x, c, rope_freqs, audio_kv=None,
                self_audio_kv=None, partner_audio_kv=None,
                audio_rope_freqs=None):
        """
        Args:
            x: (N, L, D)
            c: (N, D) combined conditioning
            rope_freqs: (L, head_dim) — RoPE for full motion sequence
            audio_kv: (N, L_audio, D) single audio KV (backward compat)
            self_audio_kv: (N, L_audio, D) self-audio for lip sync
            partner_audio_kv: (N, L_audio, D) partner-audio for reactions
            audio_rope_freqs: (L_audio, head_dim) — RoPE for audio positions
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)

        # Self-attention path
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(x_norm, rope_freqs)

        # Single cross-attention (backward compat)
        if self.has_cross_attn and audio_kv is not None:
            x_norm = self.cross_norm(x)
            ca_out, _ = self.cross_attn_layer(x_norm, audio_kv, audio_kv)
            x = x + self.cross_gate * ca_out

        # Dual cross-attention with RoPE for temporal alignment
        if self.has_dual_cross_attn:
            if self_audio_kv is not None:
                x_norm = self.self_audio_norm(x)
                sa_out = self.self_audio_cross_attn(
                    x_norm, self_audio_kv, rope_freqs, audio_rope_freqs)
                x = x + self.self_audio_gate * sa_out

            if partner_audio_kv is not None:
                x_norm = self.partner_audio_norm(x)
                pa_out = self.partner_audio_cross_attn(
                    x_norm, partner_audio_kv, rope_freqs, audio_rope_freqs)
                x = x + self.partner_audio_gate * pa_out

        # MLP path
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x


# ── Flow Matching Denoiser ────────────────────────────────────────────────

class FlowMatchingDenoiser(nn.Module):
    """DiT-style denoiser for flow matching.

    Combined conditioning: c = time_emb + style_proj(z) + label_proj(labels)
    All conditions feed into AdaLN-Zero per block.
    """

    def __init__(self, args, device='cuda'):
        super().__init__()

        self.motion_feat_dim = getattr(args, 'motion_feat_dim', 56) or 56
        self.shape_feat_dim = 100
        self.d_style = getattr(args, 'd_style', 256)
        self.d_label = getattr(args, 'd_label', 256)
        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.n_prev_motions = args.n_prev_motions
        self.n_motions = args.n_motions
        self.n_basis = getattr(args, 'n_basis', 4)

        # Continuous timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(self.feature_dim)

        # RoPE for sequence positions
        head_dim = self.feature_dim // self.n_heads
        self.rope = RotaryEmbedding(head_dim, max_len=1 + self.n_prev_motions + self.n_motions + 16)

        # Person projection: shape -> feature_dim (becomes first token)
        self.person_proj = nn.Linear(self.shape_feat_dim, self.feature_dim)

        # Motion -> feature_dim
        self.feature_proj = nn.Linear(self.motion_feat_dim, self.feature_dim)

        # Style projection into conditioning space
        self.style_proj = nn.Linear(self.d_style, self.feature_dim)

        # Discrete label embeddings -> conditioning space
        self.n_label_dims = getattr(args, 'n_label_dims', 5) or 5
        self.gaze_state_embed = nn.Embedding(4, self.feature_dim)
        self.head_motion_embed = nn.Embedding(5, self.feature_dim)
        if self.n_label_dims >= 5:
            # V5.1: 5 labels with 4-class intensity + avert_dir + role
            self.head_intensity_embed = nn.Embedding(4, self.feature_dim)
            self.avert_dir_embed = nn.Embedding(4, self.feature_dim)
            self.role_embed = nn.Embedding(3, self.feature_dim)
        else:
            # V5.0 compat: 3 labels with 3-class intensity
            self.head_intensity_embed = nn.Embedding(3, self.feature_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(self.feature_dim * self.n_label_dims, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        # Non-zero init for labels
        label_init_scale = getattr(args, 'label_init_scale', 0.02)
        if label_init_scale > 0:
            nn.init.normal_(self.label_proj[-1].weight, std=label_init_scale)
            nn.init.zeros_(self.label_proj[-1].bias)

        # Audio cross-attention (optional, for dyadic Stage 2)
        self.use_audio = getattr(args, 'use_audio', False) or False
        if self.use_audio:
            audio_dim = getattr(args, 'audio_dim', 768)  # HuBERT dim
            self.audio_proj = nn.Linear(audio_dim, self.feature_dim)

        # Dual audio: on-the-fly wav2vec2 + separate self/partner cross-attention
        self.use_dual_audio = getattr(args, 'use_dual_audio', False) or False
        if self.use_dual_audio:
            from models.wav2vec2_encoder import FrozenWav2Vec2
            self.wav2vec2 = FrozenWav2Vec2()
            audio_dim = self.wav2vec2.hidden_size  # 1024
            self.self_audio_proj = nn.Linear(audio_dim, self.feature_dim)
            self.partner_audio_proj = nn.Linear(audio_dim, self.feature_dim)
            # Learned null audio embedding — replaces zero-waveform artifacts for CFG
            self.null_audio_kv = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

        # DiT blocks
        has_single_ca = self.use_audio and not self.use_dual_audio
        self.blocks = nn.ModuleList([
            DiTBlock(self.feature_dim, self.n_heads,
                     cross_attn=has_single_ca,
                     dual_cross_attn=self.use_dual_audio)
            for _ in range(self.n_layers)
        ])

        # Final norm + output heads
        self.final_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False, eps=1e-6)
        # Final AdaLN modulation (shift + scale only, no gate)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.feature_dim, 2 * self.feature_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # Decomposed output heads
        conv_kernel = getattr(args, 'conv_kernel_size', 0) or 0
        include_eye = self.motion_feat_dim > 56
        if conv_kernel > 0:
            self.output_heads = DecomposedHeadsV2(
                self.feature_dim, self.n_basis, conv_kernel=conv_kernel,
                include_eye=include_eye)
        else:
            self.output_heads = DecomposedHeads(
                self.feature_dim, self.n_basis, include_eye=include_eye)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode_labels(self, discrete_labels):
        g_emb = self.gaze_state_embed(discrete_labels[:, 0])
        h_emb = self.head_motion_embed(discrete_labels[:, 1])
        i_emb = self.head_intensity_embed(discrete_labels[:, 2])
        if self.n_label_dims >= 5:
            n_dims = discrete_labels.shape[1]
            if n_dims >= 5:
                ad_emb = self.avert_dir_embed(discrete_labels[:, 3])
                r_emb = self.role_embed(discrete_labels[:, 4])
            else:
                ad_emb = self.avert_dir_embed(torch.zeros_like(discrete_labels[:, 0]))
                r_emb = self.role_embed(torch.zeros_like(discrete_labels[:, 0]))
            return self.label_proj(torch.cat([g_emb, h_emb, i_emb, ad_emb, r_emb], dim=-1))
        else:
            return self.label_proj(torch.cat([g_emb, h_emb, i_emb], dim=-1))

    def _extract_wav2vec2_features(self, audio_raw, frame_num):
        """Run shared frozen wav2vec2 on raw waveform.

        Args:
            audio_raw: (N, n_samples) raw 16kHz waveform
            frame_num: target output frames (e.g. 100)
        Returns:
            (N, frame_num, 1024) audio features
        """
        return self.wav2vec2(audio_raw, frame_num=frame_num)

    def forward(self, motion_feat, shape_feat, style_z,
                prev_motion_feat, t, discrete_labels=None,
                audio_feat=None,
                self_audio_raw=None, partner_audio_raw=None):
        """
        Args:
            motion_feat:      (N, L, 56) noisy motion
            shape_feat:       (N, 1, 100) shape
            style_z:          (N, d_style) from VAE
            prev_motion_feat: (N, L_p, 56) previous context
            t:                (N,) float in [0, 1] — flow matching time
            discrete_labels:  (N, 3+) LongTensor or None
            audio_feat:       (N, L_a, 768) pre-extracted audio (backward compat)
            self_audio_raw:   (N, n_samples) raw waveform of this speaker
            partner_audio_raw:(N, n_samples) raw waveform of other speaker
        """
        N = motion_feat.shape[0]

        # Combined conditioning: time + style + labels
        c = self.time_embed(t)                          # (N, D)
        c = c + self.style_proj(style_z)                # (N, D)

        if discrete_labels is not None:
            c = c + self._encode_labels(discrete_labels)  # (N, D)

        # Project audio for cross-attention (if available)
        audio_kv = None
        if self.use_audio and audio_feat is not None:
            audio_kv = self.audio_proj(audio_feat)      # (N, L_a, D)

        # Dual audio: wav2vec2 on-the-fly → project → separate KV
        self_audio_kv = None
        partner_audio_kv = None
        if self.use_dual_audio:
            frame_num = motion_feat.shape[1]  # L (100)
            if self_audio_raw is not None:
                self_feats = self._extract_wav2vec2_features(
                    self_audio_raw, frame_num)             # (N, L, 1024)
                self_audio_kv = self.self_audio_proj(self_feats)  # (N, L, D)
            if partner_audio_raw is not None:
                partner_feats = self._extract_wav2vec2_features(
                    partner_audio_raw, frame_num)          # (N, L, 1024)
                partner_audio_kv = self.partner_audio_proj(partner_feats)

            # Replace zero-waveform audio KV with learned null embedding
            if hasattr(self, 'null_audio_kv'):
                null_kv = self.null_audio_kv.expand(N, frame_num, -1)
                if self_audio_raw is not None and self_audio_kv is not None:
                    dropped = self_audio_raw.pow(2).mean(-1) < 1e-10
                    if dropped.any():
                        self_audio_kv = self_audio_kv.clone()
                        self_audio_kv[dropped] = null_kv[dropped]
                if partner_audio_raw is not None and partner_audio_kv is not None:
                    dropped_p = partner_audio_raw.pow(2).mean(-1) < 1e-10
                    if dropped_p.any():
                        partner_audio_kv = partner_audio_kv.clone()
                        partner_audio_kv[dropped_p] = null_kv[dropped_p]

        # Build sequence: [person_token, prev_motion, current_motion]
        person_feat = self.person_proj(shape_feat)      # (N, 1, D)
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        feats_in = self.feature_proj(feats_in)          # (N, L_p + L, D)
        feats_in = torch.cat([person_feat, feats_in], dim=1)  # (N, 1+L_p+L, D)

        # RoPE frequencies
        seq_len = feats_in.shape[1]
        rope_freqs = self.rope(seq_len, device=feats_in.device)

        # Audio RoPE: audio frame t aligns with motion position (1 + n_prev + t)
        # So audio uses rope positions [1+n_prev .. 1+n_prev+n_motions-1]
        audio_rope_freqs = None
        if self_audio_kv is not None or partner_audio_kv is not None:
            offset = 1 + self.n_prev_motions  # person_token(1) + prev_motion
            audio_rope_freqs = rope_freqs[offset:offset + motion_feat.shape[1]]

        # DiT blocks
        x = feats_in
        for block in self.blocks:
            x = block(x, c, rope_freqs, audio_kv=audio_kv,
                      self_audio_kv=self_audio_kv,
                      partner_audio_kv=partner_audio_kv,
                      audio_rope_freqs=audio_rope_freqs)

        # Final AdaLN + output heads (skip person token)
        shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        decomposed = self.output_heads(x[:, 1:])
        return decomposed


# ── Style Basis Decoder (same as V4.4) ────────────────────────────────────

class StyleBasisDecoder(nn.Module):
    """K additive style basis vectors, driven by style latent z."""

    def __init__(self, d_style=256, motion_dim=56, n_basis=4, hidden_dim=256):
        super().__init__()
        self.n_basis = n_basis
        self.motion_dim = motion_dim
        self.basis_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_style, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, motion_dim),
            ) for _ in range(n_basis)
        ])

    def forward(self, style_z, alphas):
        N, L, K = alphas.shape
        basis_vectors = [self.basis_mlps[k](style_z) for k in range(self.n_basis)]
        basis = torch.stack(basis_vectors, dim=1)  # (N, K, motion_dim)

        # exp+jaw (alpha-modulated), neck+eye (always-on, identity-level)
        basis_exp_jaw = basis[:, :, :53]
        basis_pose = basis[:, :, 53:]  # neck(3) [+ eye(3) if 59D]

        alpha_sigmoid = torch.sigmoid(alphas)
        offset_exp_jaw = torch.einsum('nlk,nkd->nld', alpha_sigmoid, basis_exp_jaw)
        offset_pose = basis_pose.sum(dim=1).unsqueeze(1).expand(-1, L, -1)

        return torch.cat([offset_exp_jaw, offset_pose], dim=-1)


# ── Full V5 Model ─────────────────────────────────────────────────────────

class MotionPriorV5(nn.Module):
    """V5 Motion Prior: Flow Matching + DiT + RoPE + DecomposedHeadsV2.

    Training: x_t = (1-t)*x_0 + t*noise, predict x_0.
    Sampling: Euler ODE from t=1 (noise) to t=0 (clean).
    Binary CFG on labels (style always active).
    """

    def __init__(self, args, device='cuda'):
        super().__init__()

        self.motion_feat_dim = getattr(args, 'motion_feat_dim', 56) or 56
        self.n_motions = args.n_motions
        self.n_prev_motions = args.n_prev_motions
        self.d_style = getattr(args, 'd_style', 256)
        self.n_basis = getattr(args, 'n_basis', 4)

        # Timestep sampling strategy
        self.timestep_sampling = getattr(args, 'timestep_sampling', 'logit_normal') or 'logit_normal'
        lnm = getattr(args, 'logit_normal_mean', None)
        self.logit_normal_mean = lnm if lnm is not None else 0.0
        lns = getattr(args, 'logit_normal_std', None)
        self.logit_normal_std = lns if lns is not None else 1.0

        # Style VAE
        self.style_enc_vae = StyleEncoderVAE(
            input_dim=self.motion_feat_dim, d_style=self.d_style)

        # Start tokens for prev context
        self.start_motion_feat = nn.Parameter(
            torch.randn(1, self.n_prev_motions, self.motion_feat_dim))

        # Flow matching denoiser (DiT + RoPE)
        self.denoising_net = FlowMatchingDenoiser(args, device)

        # Style basis decoder
        self.basis_decoder = StyleBasisDecoder(
            d_style=self.d_style,
            motion_dim=self.motion_feat_dim,
            n_basis=self.n_basis,
        )

        # Null labels for CFG
        self.register_buffer('null_labels', torch.zeros(
            1, self.denoising_net.n_label_dims, dtype=torch.long))

        # Dropout rates
        self.style_dropout = getattr(args, 'style_dropout', 0.15)
        self.cross_style_rate = getattr(args, 'cross_style_rate', 0.30)
        self.label_dropout = getattr(args, 'label_dropout', 0.15)
        self.audio_cfg_dropout = getattr(args, 'audio_cfg_dropout', 0.2)
        self.prev_motion_dropout = getattr(args, 'prev_motion_dropout', 0.0)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _sample_timesteps(self, batch_size):
        """Sample timesteps t in (0, 1) for flow matching."""
        if self.timestep_sampling == 'logit_normal':
            # Logit-normal: concentrates on mid-range t (most informative)
            u = torch.randn(batch_size, device=self.device)
            u = u * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)
        else:
            # Uniform
            t = torch.rand(batch_size, device=self.device)

        # Clamp away from exact 0 and 1
        t = t.clamp(1e-5, 1.0 - 1e-5)
        return t

    def forward(self, motion, shape_feat, style_input,
                prev_motion_feat=None, time_step=None,
                discrete_labels=None, audio_feat=None,
                self_audio_raw=None, partner_audio_raw=None):
        """Training forward pass with flow matching.

        Flow matching interpolation: x_t = (1-t)*x_0 + t*noise
        Network predicts x_0. Loss is on x_0 prediction.

        Args:
            audio_feat: (N, L, 768) pre-extracted audio (backward compat)
            self_audio_raw: (N, n_samples) raw waveform of this speaker
            partner_audio_raw: (N, n_samples) raw waveform of other speaker

        Returns:
            noise, dynamic_pred, gt_motion, mu, logvar, basis_offset, alphas, decomposed
        """
        batch_size = motion.shape[0]

        # 1. Style VAE encoding
        z, mu, logvar = self.style_enc_vae(style_input)

        # 2. Cross-style augmentation
        if self.training and self.cross_style_rate > 0:
            swap_mask = torch.rand(batch_size, device=self.device) < self.cross_style_rate
            if swap_mask.any():
                perm = torch.randperm(batch_size, device=self.device)
                z = torch.where(swap_mask.unsqueeze(1), z[perm], z)
                mu = torch.where(swap_mask.unsqueeze(1), mu[perm], mu)
                logvar = torch.where(swap_mask.unsqueeze(1), logvar[perm], logvar)

        # 3. Style dropout
        if self.training and self.style_dropout > 0:
            drop_mask = torch.rand(batch_size, device=self.device) < self.style_dropout
            z = torch.where(drop_mask.unsqueeze(1), torch.zeros_like(z), z)

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)

        # 4. Label dropout
        if self.training and discrete_labels is not None and self.label_dropout > 0:
            mask_lbl = torch.rand(batch_size, device=self.device) < self.label_dropout
            discrete_labels = discrete_labels.clone()
            discrete_labels[mask_lbl] = 0

        # 5. Audio dropout — JOINT dropout for CFG (drop both streams together)
        audio_cfg_dropout = getattr(self, 'audio_cfg_dropout', 0.2)
        if self.training and (self_audio_raw is not None or partner_audio_raw is not None):
            if audio_cfg_dropout > 0:
                # Joint: drop BOTH audio streams together → enables audio CFG at inference
                mask_joint = torch.rand(batch_size, device=self.device) < audio_cfg_dropout
                if mask_joint.any():
                    if self_audio_raw is not None:
                        self_audio_raw = self_audio_raw.clone()
                        self_audio_raw[mask_joint] = 0.0
                    if partner_audio_raw is not None:
                        partner_audio_raw = partner_audio_raw.clone()
                        partner_audio_raw[mask_joint] = 0.0

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)

        # Prev motion dropout — forces model to not over-rely on context
        prev_motion_dropout = getattr(self, 'prev_motion_dropout', 0.0)
        if self.training and prev_motion_dropout > 0 and prev_motion_feat is not None:
            drop_prev = torch.rand(batch_size, device=self.device) < prev_motion_dropout
            if drop_prev.any():
                prev_motion_feat = prev_motion_feat.clone()
                prev_motion_feat[drop_prev] = self.start_motion_feat.expand(drop_prev.sum(), -1, -1)

        # 6. Sample flow matching timestep t ~ (0, 1)
        if time_step is not None:
            t = time_step
        else:
            t = self._sample_timesteps(batch_size)

        # 7. Flow matching interpolation: x_t = (1-t)*x_0 + t*noise
        noise = torch.randn_like(motion)
        t_expand = t.view(-1, 1, 1)
        motion_noisy = (1.0 - t_expand) * motion + t_expand * noise

        # 8. Denoiser predicts x_0
        decomposed = self.denoising_net(
            motion_noisy, shape_feat, z,
            prev_motion_feat, t,
            discrete_labels=discrete_labels,
            audio_feat=audio_feat,
            self_audio_raw=self_audio_raw,
            partner_audio_raw=partner_audio_raw)

        dynamic_pred = decomposed['motion']
        alphas = decomposed['alphas']

        # Basis offset
        basis_offset = self.basis_decoder(z, alphas)

        # Return same interface: (noise, pred, gt, mu, logvar, basis, alphas, decomposed)
        return (noise, dynamic_pred, motion.detach(),
                mu, logvar, basis_offset, alphas, decomposed)

    @torch.no_grad()
    def sample(self, shape_feat, style_z=None, discrete_labels=None,
               prev_motion_feat=None, cfg_scale=2.0, motion_at_T=None,
               sample_steps=None, audio_feat=None,
               self_audio_raw=None, partner_audio_raw=None, **kwargs):
        """Euler ODE sampling from t=1 (noise) to t=0 (clean).

        Same interface as MotionPriorDiffusion.sample() for eval compatibility.
        audio_feat: (N, L, 768) pre-extracted audio (backward compat)
        self_audio_raw: (N, n_samples) raw waveform of this speaker
        partner_audio_raw: (N, n_samples) raw waveform of other speaker
        """
        batch_size = shape_feat.shape[0] if shape_feat.ndim >= 2 else 1

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)
        if style_z is None:
            style_z = torch.zeros(batch_size, self.d_style, device=self.device)
        if style_z.ndim == 3:
            style_z = style_z.squeeze(1)
        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if motion_at_T is None:
            motion_at_T = torch.randn(
                (batch_size, self.n_motions, self.motion_feat_dim),
                device=self.device)

        # Null labels for CFG
        lbl_null = self.null_labels.expand(batch_size, -1)
        if discrete_labels is None:
            discrete_labels = lbl_null

        # CFG scales (separate axes for labels, audio, style)
        cfg_scale_audio = kwargs.get('cfg_scale_audio', cfg_scale)
        cfg_scale_style = kwargs.get('cfg_scale_style', 1.0)  # 1.0 = no amplification (backward compat)

        # Determine CFG mode
        has_audio = (self_audio_raw is not None or partner_audio_raw is not None)
        has_style_cfg = cfg_scale_style != 1.0  # only add style axis if requested
        null_style = torch.zeros_like(style_z)

        if has_audio:
            if has_style_cfg:
                # 4-way: [no_labels+no_audio+no_style | labels+no_audio+no_style | labels+audio+no_style | labels+audio+style]
                n_entries = 4
                style_in = torch.cat([null_style, null_style, null_style, style_z], dim=0)
                labels_in = torch.cat([lbl_null, discrete_labels, discrete_labels, discrete_labels], dim=0)
                null_self = torch.zeros_like(self_audio_raw) if self_audio_raw is not None else None
                null_partner = torch.zeros_like(partner_audio_raw) if partner_audio_raw is not None else None
                self_audio_in = torch.cat([null_self, null_self, self_audio_raw, self_audio_raw], dim=0) if self_audio_raw is not None else None
                partner_audio_in = torch.cat([null_partner, null_partner, partner_audio_raw, partner_audio_raw], dim=0) if partner_audio_raw is not None else None
            else:
                # 3-way: [no_labels+no_audio | labels+no_audio | labels+audio] (style in all)
                n_entries = 3
                style_in = torch.cat([style_z] * n_entries, dim=0)
                labels_in = torch.cat([lbl_null, discrete_labels, discrete_labels], dim=0)
                null_self = torch.zeros_like(self_audio_raw) if self_audio_raw is not None else None
                null_partner = torch.zeros_like(partner_audio_raw) if partner_audio_raw is not None else None
                self_audio_in = torch.cat([null_self, null_self, self_audio_raw], dim=0) if self_audio_raw is not None else None
                partner_audio_in = torch.cat([null_partner, null_partner, partner_audio_raw], dim=0) if partner_audio_raw is not None else None
        else:
            if has_style_cfg:
                # 3-way: [no_labels+no_style | labels+no_style | labels+style]
                n_entries = 3
                style_in = torch.cat([null_style, null_style, style_z], dim=0)
                labels_in = torch.cat([lbl_null, discrete_labels, discrete_labels], dim=0)
            else:
                # 2-way: [no_labels | labels] (style in both)
                n_entries = 2
                style_in = torch.cat([style_z] * n_entries, dim=0)
                labels_in = torch.cat([lbl_null, discrete_labels], dim=0)
            self_audio_in = None
            partner_audio_in = None

        shape_in = torch.cat([shape_feat] * n_entries, dim=0)
        prev_in = torch.cat([prev_motion_feat] * n_entries, dim=0)

        audio_in = None
        if audio_feat is not None:
            audio_in = torch.cat([audio_feat] * n_entries, dim=0)

        # Euler ODE: integrate from t=1 to t=0
        if sample_steps is None:
            sample_steps = 10
        dt = 1.0 / sample_steps
        timesteps = torch.linspace(1.0, dt, sample_steps, device=self.device)

        x_t = motion_at_T

        for t_val in timesteps:
            t_batch = torch.full((batch_size,), t_val.item(), device=self.device)
            t_in = torch.cat([t_batch] * n_entries, dim=0)
            x_in = torch.cat([x_t] * n_entries, dim=0)

            decomposed = self.denoising_net(
                x_in, shape_in, style_in,
                prev_in, t_in,
                discrete_labels=labels_in,
                audio_feat=audio_in,
                self_audio_raw=self_audio_in,
                partner_audio_raw=partner_audio_in)

            dynamic_pred = decomposed['motion']
            alphas = decomposed['alphas']

            basis_offset = self.basis_decoder(style_in, alphas)
            x_0_pred = dynamic_pred + basis_offset

            if has_audio and has_style_cfg:
                # 4-way: uncond → +labels → +audio → +style
                x0_uncond = x_0_pred[:batch_size, -self.n_motions:]
                x0_labels = x_0_pred[batch_size:2*batch_size, -self.n_motions:]
                x0_audio = x_0_pred[2*batch_size:3*batch_size, -self.n_motions:]
                x0_full = x_0_pred[3*batch_size:, -self.n_motions:]
                x_0_guided = (x0_uncond
                              + cfg_scale * (x0_labels - x0_uncond)
                              + cfg_scale_audio * (x0_audio - x0_labels)
                              + cfg_scale_style * (x0_full - x0_audio))
            elif has_audio:
                # 3-way: uncond → +labels → +audio
                x0_uncond = x_0_pred[:batch_size, -self.n_motions:]
                x0_labels = x_0_pred[batch_size:2*batch_size, -self.n_motions:]
                x0_full = x_0_pred[2*batch_size:, -self.n_motions:]
                x_0_guided = (x0_uncond
                              + cfg_scale * (x0_labels - x0_uncond)
                              + cfg_scale_audio * (x0_full - x0_labels))
            elif has_style_cfg:
                # 3-way: uncond → +labels → +style
                x0_uncond = x_0_pred[:batch_size, -self.n_motions:]
                x0_labels = x_0_pred[batch_size:2*batch_size, -self.n_motions:]
                x0_full = x_0_pred[2*batch_size:, -self.n_motions:]
                x_0_guided = (x0_uncond
                              + cfg_scale * (x0_labels - x0_uncond)
                              + cfg_scale_style * (x0_full - x0_labels))
            else:
                # 2-way: uncond → +labels
                x0_uncond = x_0_pred[:batch_size, -self.n_motions:]
                x0_cond = x_0_pred[batch_size:, -self.n_motions:]
                x_0_guided = x0_uncond + cfg_scale * (x0_cond - x0_uncond)

            # Euler step
            x_t = x_t - dt * (x_t - x_0_guided) / t_val

        return x_t, motion_at_T
