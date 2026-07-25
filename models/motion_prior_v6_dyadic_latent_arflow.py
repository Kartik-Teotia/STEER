"""V6 Latent AR-Flow: Causal latent transformer with flow matching for Stage 2.

Stage 2 of a 2-stage architecture. Generates 256D latent tokens autoregressively
via flow matching, conditioned on audio and partner motion. Latent tokens come
from a frozen Stage 1 temporal motion AE (4x temporal downsampling).

Architecture:
  - Input: 25 latent tokens (256D), projected to hidden_dim (448)
  - Backbone: 8 layers x 448 dim x 8 heads, fully causal attention, RoPE
  - Sequence layout: [prev_latents(3) | target_latents(25)]
  - No shape token (avoids identity memorization)
  - Learned start_token for first-window generation

Conditioning:
  - Per-token flow timestep via AdaLN-Zero
  - Self-audio: wav2vec2 features at latent rate (25 tokens), cross-attention with RoPE
  - Partner motion latents + partner audio: fused via concat+MLP
  - Partner encoded through 2-layer causal temporal encoder
  - Per-block partner concat+MLP fusion in each DiT layer

Flow matching:
  - Rectified flow: z_t = (1-t)*z_0 + t*noise
  - Model predicts clean z_0
  - Independent per-token timesteps during training
  - Euler ODE sampling at inference (4-10 steps)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_prior_v5 import (
    RotaryEmbedding,
    SinusoidalTimestepEmbedding,
    _rotate_half,
)


# ── Causal Self-Attention with RoPE + KV Cache ───────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking.

    During training: use causal mask over the full sequence.
    During inference: only compute Q for the current token, concat cached K/V.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, L, D) — full sequence (training) or single token (inference)
            rope_freqs: (L_total, head_dim) — RoPE freqs table
            kv_cache: (K_past, V_past) each (B, H, L_past, D_h). Inference only.
            q_pos: (L,) int — explicit global positions for Q. If None, inferred.
            k_pos: (L,) int — explicit global positions for new K. If None, = q_pos.
        Returns:
            out: (B, L, D)
            new_kv_cache: (K_all, V_all) for next step
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        q, k, v = qkv.unbind(0)

        if kv_cache is not None:
            # Inference: apply RoPE to current position(s)
            if q_pos is not None:
                freqs_current = rope_freqs[q_pos]
            else:
                L_past = kv_cache[0].shape[2]
                freqs_current = rope_freqs[L_past:L_past + L]
            cos_c = freqs_current.cos()[None, None, :, :]
            sin_c = freqs_current.sin()[None, None, :, :]
            q = q * cos_c + _rotate_half(q) * sin_c

            if k_pos is not None:
                freqs_k = rope_freqs[k_pos]
            else:
                freqs_k = freqs_current
            cos_k = freqs_k.cos()[None, None, :, :]
            sin_k = freqs_k.sin()[None, None, :, :]
            k = k * cos_k + _rotate_half(k) * sin_k

            # Concat with cached K, V (already RoPE-encoded)
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

            # No mask needed — attending to all cached + current (causal by construction)
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Training: apply RoPE to full sequence, use causal mask
            freqs = rope_freqs[:L]
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            # Use built-in causal attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).reshape(B, L, D)
        new_kv = (k, v)
        return self.proj(out), new_kv


# ── Causal Cross-Attention with RoPE + KV Cache ──────────────────────────

class CausalCrossAttention(nn.Module):
    """Cross-attention with RoPE on Q and K, supporting KV caching.

    For audio cross-attention: Q comes from motion tokens (prev_latents + target
    tokens), KV from self-audio at latent rate. Uses an offset-aware causal mask
    on the uncached path so that target token t can only attend to audio keys
    0..t even when Q is prefixed by prev_latent rows.

    The uncached call auto-infers `q_position_offset = max(0, L_q - L_kv)`.
    For V6-A with n_prev_latents=3 and n_latents=25, this gives offset=3, so
    target token t (Q row n_prev+t) attends to K[0..t]. Prefix rows (first
    n_prev Q positions) are left UNMASKED to avoid all-inf attention rows
    (SDPA returns NaN otherwise); the prefix cross-attention output is still
    zeroed downstream in the DiT block as before.

    The cached path (streaming inference via sample_token) is causal by
    construction: each call passes a single current audio token and the
    cross_attn_kv_cache accumulates exactly K[0..t], so no mask is needed.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        rope_freqs_q: torch.Tensor,
        rope_freqs_k: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            k_pos: (L_kv,) int — explicit global positions for new K tokens.
                   If None, inferred from cache length (old behavior).
        """
        B, L_q, D = q_input.shape
        L_kv = kv_input.shape[1]

        q = self.q_proj(q_input).reshape(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).reshape(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).reshape(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q
        cos_q = rope_freqs_q.cos()[None, None, :, :]
        sin_q = rope_freqs_q.sin()[None, None, :, :]
        q = q * cos_q + _rotate_half(q) * sin_q

        # Apply RoPE to K
        if k_pos is not None:
            freqs_k_new = rope_freqs_k[k_pos]
        elif kv_cache is not None:
            L_past = kv_cache[0].shape[2]
            freqs_k_new = rope_freqs_k[L_past:L_past + L_kv]
        else:
            freqs_k_new = rope_freqs_k[:L_kv]
        cos_k = freqs_k_new.cos()[None, None, :, :]
        sin_k = freqs_k_new.sin()[None, None, :, :]
        k = k * cos_k + _rotate_half(k) * sin_k

        # Concat cached KV if present
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        if kv_cache is None:
            # Uncached path: training, sample(), sample_streaming_rolling(),
            # real_time_latent_streaming — all hit this branch with the full
            # audio window. Build an offset-aware causal additive mask.
            L_kv_total = k.shape[2]
            q_offset = max(0, L_q - L_kv_total)  # = n_prev for V6-A
            device = q.device
            dtype = q.dtype
            q_idx = torch.arange(L_q, device=device).unsqueeze(1)  # (L_q, 1)
            k_idx = torch.arange(L_kv_total, device=device).unsqueeze(0)  # (1, L_kv)
            # Target rows (q_idx >= q_offset): mask out k_idx > q_idx - q_offset.
            # Prefix rows (q_idx < q_offset): leave unmasked (all zeros) so
            # SDPA does not produce NaN rows. Prefix cross-attention output
            # is zeroed out in DiTBlockLatentARFlow.forward anyway.
            target_row = q_idx >= q_offset  # (L_q, 1), bool
            causal_bad = k_idx > (q_idx - q_offset)  # (L_q, L_kv), bool
            mask_bool = target_row & causal_bad  # (L_q, L_kv)
            attn_mask = torch.zeros(L_q, L_kv_total, device=device, dtype=dtype)
            attn_mask = attn_mask.masked_fill(mask_bool, float('-inf'))
            # Expand to (1, 1, L_q, L_kv) for SDPA broadcasting over B, H.
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            # Cached streaming path: sample_token() feeds one audio token per
            # call, so the accumulated cache is exactly K[0..t]. No mask.
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L_q, D)

        new_kv = (k, v)
        return self.out_proj(out), new_kv


# ── Causal Partner Temporal Block ─────────────────────────────────────────

class CausalPartnerTemporalBlock(nn.Module):
    """Temporal encoder block for the partner stream — causal variant."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), rope_freqs, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


# ── DiT Block for Latent AR-Flow ──────────────────────────────────────────

class DiTBlockLatentARFlow(nn.Module):
    """DiT block with causal self-attention, self-audio cross-attention,
    partner fusion (concat+MLP OR causal cross-attention), and AdaLN-Zero.

    Operates on latent tokens (256D projected to hidden_dim).

    Partner fusion mode, selected at build time via `use_partner_cross_attn`:
      - False (default, V6-A/B): per-token concat+MLP, listener[t] reads
        partner[t] via a fixed learned projection. AVFlow v3 pattern.
      - True (V6-C): causal cross-attention, listener target rows attend
        to all earlier partner tokens via dynamic routing. Partner fusion
        is a straight replacement — no learnable gate, no stacking.

    Both modes share the same attention/MLP pipeline; only the partner
    step differs.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 use_partner_cross_attn: bool = False):
        super().__init__()
        self.use_partner_cross_attn = use_partner_cross_attn

        # Self-attention path
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = CausalSelfAttention(dim, n_heads)

        # Self-audio cross-attention
        self.self_audio_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_audio_cross_attn = CausalCrossAttention(dim, n_heads)

        # Partner fusion — mutually exclusive modes
        self.partner_norm = nn.LayerNorm(dim, eps=1e-6)
        if use_partner_cross_attn:
            # V6-C: causal cross-attention over encoded partner_seq.
            # Q and K share target-time positions (see forward). Listener
            # target row t attends to partner tokens 0..t (strict causal).
            self.partner_cross_attn = CausalCrossAttention(dim, n_heads)
            self.partner_fuse = None
        else:
            # V6-A/B: concat+MLP fusion (AVFlow v3 pattern)
            self.partner_cross_attn = None
            self.partner_fuse = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        # MLP path
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        # AdaLN-Zero: 6 modulation params (shift, scale, gate) x 2 paths
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_freqs: torch.Tensor,
        self_audio_kv: Optional[torch.Tensor] = None,
        audio_rope_freqs_q: Optional[torch.Tensor] = None,
        audio_rope_freqs_k: Optional[torch.Tensor] = None,
        partner_ctx: Optional[torch.Tensor] = None,
        partner_kv: Optional[torch.Tensor] = None,
        partner_rope_freqs_q: Optional[torch.Tensor] = None,
        partner_rope_freqs_k: Optional[torch.Tensor] = None,
        self_attn_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        partner_cross_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        extra_adaln_mod: Optional[torch.Tensor] = None,
        self_attn_q_pos: Optional[torch.Tensor] = None,
        self_attn_k_pos: Optional[torch.Tensor] = None,
        audio_cross_k_pos: Optional[torch.Tensor] = None,
        partner_cross_k_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor],
               Optional[Tuple[torch.Tensor, torch.Tensor]],
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            c: (B, L, D) per-token conditioning (time embedding)
            partner_ctx: (B, L, D) partner context tiled to full Q length — used
                by the concat+MLP path and as the n_prev-prefixed full-sequence
                partner injection in training.
            partner_kv: (B, L_kv, D) encoded partner tokens at target-time
                positions — used by the cross-attention path as K/V. In the
                full-window path L_kv = n_target. In streaming L_kv grows with
                the committed-plus-active partner token count.
            partner_rope_freqs_q: (L_q_target, head_dim) RoPE freqs for the
                target rows of Q when using cross-attention. Shared target-time
                positions with K.
            partner_rope_freqs_k: (L_kv, head_dim) RoPE freqs for K. Same
                positions as Q (partner and listener live on the same clock).
            partner_cross_kv_cache: optional accumulated partner cross-attn KV
                cache for streaming inference.
            extra_adaln_mod: (B, L, 6*D) optional extra modulation added to
                adaLN output. Used by V7 for emotion conditioning — separate
                from time/style so there's no interference. Default None (no-op).
        Returns:
            x, self_kv_cache, audio_cross_kv_cache, partner_cross_kv_cache
        """
        # AdaLN modulation from per-token conditioning
        c_mod = self.adaLN_modulation(c)  # (B, L, 6*D)
        if extra_adaln_mod is not None:
            c_mod = c_mod + extra_adaln_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            c_mod.chunk(6, dim=-1)

        # Causal self-attention
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, self_kv = self.attn(
            x_norm, rope_freqs, kv_cache=self_attn_kv_cache,
            q_pos=self_attn_q_pos, k_pos=self_attn_k_pos)
        x = x + gate_msa * attn_out

        # Self-audio cross-attention
        cross_kv = cross_attn_kv_cache
        if self_audio_kv is not None and audio_rope_freqs_q is not None:
            x_norm = self.self_audio_norm(x)
            sa_out, cross_kv = self.self_audio_cross_attn(
                x_norm, self_audio_kv, audio_rope_freqs_q, audio_rope_freqs_k,
                kv_cache=cross_attn_kv_cache,
                k_pos=audio_cross_k_pos,
            )
            # Zero out cross-attention for prev_latent prefix tokens
            # (they shouldn't attend to future audio)
            n_total = x.shape[1]
            n_audio = self_audio_kv.shape[1]
            n_prefix = n_total - n_audio
            if n_prefix > 0 and self_attn_kv_cache is None:
                # Training mode: zero out prefix positions
                sa_out = sa_out.clone()
                sa_out[:, :n_prefix] = 0.0
            x = x + sa_out

        # Partner fusion
        new_partner_cross_kv = partner_cross_kv_cache
        if self.use_partner_cross_attn:
            # V6-C: causal cross-attention. Query only the target rows
            # (skip the n_prev prefix). Strict causal mask via the shared
            # target-time positions for both Q and K.
            if partner_kv is not None and partner_rope_freqs_q is not None:
                if self_attn_kv_cache is None:
                    # Full-window training / non-streaming inference.
                    # Split x into prefix and target, query target only.
                    n_total = x.shape[1]
                    n_target = partner_kv.shape[1]
                    n_prefix = n_total - n_target
                    if n_prefix > 0:
                        x_prefix = x[:, :n_prefix]
                        x_target = x[:, n_prefix:]
                    else:
                        x_prefix = None
                        x_target = x
                    q = self.partner_norm(x_target)
                    p_out, new_partner_cross_kv = self.partner_cross_attn(
                        q_input=q,
                        kv_input=partner_kv,
                        rope_freqs_q=partner_rope_freqs_q,
                        rope_freqs_k=partner_rope_freqs_k,
                        kv_cache=None,
                    )
                    x_target = x_target + p_out
                    if x_prefix is not None:
                        x = torch.cat([x_prefix, x_target], dim=1)
                    else:
                        x = x_target
                else:
                    # Streaming single-token path: x already contains only
                    # the current token(s), partner_kv is the current
                    # partner token, and partner_cross_kv_cache accumulates
                    # prior partner K/V.
                    q = self.partner_norm(x)
                    p_out, new_partner_cross_kv = self.partner_cross_attn(
                        q_input=q,
                        kv_input=partner_kv,
                        rope_freqs_q=partner_rope_freqs_q,
                        rope_freqs_k=partner_rope_freqs_k,
                        kv_cache=partner_cross_kv_cache,
                        k_pos=partner_cross_k_pos,
                    )
                    x = x + p_out
        else:
            # V6-A/B: concat+MLP fusion
            if partner_ctx is not None:
                x_norm = self.partner_norm(x)
                x = x + self.partner_fuse(torch.cat([x_norm, partner_ctx], dim=-1))

        # MLP
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x, self_kv, cross_kv, new_partner_cross_kv


def _make_head(in_dim, out_dim):
    hidden = max(in_dim // 2, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _make_zero_init_residual_head(in_dim, hidden_dim, out_dim):
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


# ── Inner Denoiser ────────────────────────────────────────────────────────

class FlowMatchingDenoiserLatentARFlow(nn.Module):
    """Causal latent transformer denoiser for flow matching on AE latent tokens.

    Sequence layout: [prev_latents(3) | target_latents(25)]
    All attention is token-level causal.
    Per-token timesteps during training.
    """

    def __init__(self, args, device: str = 'cuda'):
        super().__init__()

        self.latent_dim = getattr(args, 'latent_dim', 256)
        self.hidden_dim = getattr(args, 'hidden_dim', 448)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.n_layers = getattr(args, 'n_layers', 8)
        self.n_latents = getattr(args, 'n_latents', 25)
        self.n_prev_latents = getattr(args, 'n_prev_latents', 3)

        self.use_audio = getattr(args, 'use_dual_audio', False) or False
        self.use_other_audio = getattr(args, 'use_other_audio', False) or False
        self.use_other_motion = getattr(args, 'use_other_motion', False) or False
        self.use_other_latent_delta = getattr(args, 'use_other_latent_delta', False) or False
        self.use_partner_gaze = getattr(args, 'use_partner_gaze', False) or False

        # Timestep embedding — per-token conditioning
        self.time_embed = SinusoidalTimestepEmbedding(self.hidden_dim)

        # RoPE
        head_dim = self.hidden_dim // self.n_heads
        max_seq = self.n_prev_latents + self.n_latents + 64
        self.rope = RotaryEmbedding(head_dim, max_len=max_seq)
        self.partner_rope = RotaryEmbedding(head_dim, max_len=max_seq)
        self.audio_rope = RotaryEmbedding(head_dim, max_len=max_seq)

        # Input projection: latent_dim -> hidden_dim
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # Per-token timestep projection (additive to token embeddings)
        self.token_time_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Audio encoders
        if self.use_audio or self.use_other_audio:
            from models.causal_wav2vec2 import build_audio_encoder
            self.wav2vec2 = build_audio_encoder(args)
            audio_dim = self.wav2vec2.hidden_size  # 768 or 1024

            # Optional trainable audio adapter
            self.use_audio_adapter = getattr(args, 'use_audio_adapter', False) or False
            if self.use_audio_adapter:
                adapter_hidden = getattr(args, 'audio_adapter_hidden_dim', 1024) or 1024
                self.audio_adapter = nn.Sequential(
                    nn.Linear(audio_dim, adapter_hidden),
                    nn.GELU(),
                    nn.Linear(adapter_hidden, audio_dim),
                )
                # audio_dim unchanged — adapter maps audio_dim -> audio_dim

            if self.use_audio:
                self.self_audio_proj = nn.Linear(audio_dim, self.hidden_dim)
                self.null_self_audio_kv = nn.Parameter(
                    torch.randn(1, 1, self.hidden_dim) * 0.02)
            if self.use_other_audio:
                self.other_audio_proj = nn.Linear(audio_dim, self.hidden_dim)
                self.null_other_audio_kv = nn.Parameter(
                    torch.randn(1, 1, self.hidden_dim) * 0.02)

        # Partner latent motion encoder
        if self.use_other_motion:
            self.other_latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
            self.null_other_motion_kv = nn.Parameter(
                torch.randn(1, 1, self.hidden_dim) * 0.02)

        # Partner latent delta stream
        if self.use_other_latent_delta:
            self.other_latent_delta_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        if self.use_partner_gaze:
            self.partner_gaze_proj = nn.Linear(3, self.hidden_dim)
            self.null_partner_gaze_kv = nn.Parameter(
                torch.randn(1, 1, self.hidden_dim) * 0.02)

        # Partner stream fusion
        n_partner_streams = sum([self.use_other_motion, self.use_other_audio,
                                 self.use_other_latent_delta, self.use_partner_gaze])
        if n_partner_streams >= 2:
            self.partner_input_fuse = nn.Sequential(
                nn.Linear(n_partner_streams * self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.partner_input_fuse = None

        # Causal partner temporal encoder
        self.partner_encoder_layers = getattr(args, 'partner_encoder_layers', 2) or 2
        self.partner_encoder_blocks = nn.ModuleList([
            CausalPartnerTemporalBlock(self.hidden_dim, self.n_heads)
            for _ in range(self.partner_encoder_layers)
        ])

        # Partner pre-injection into latent tokens
        self.frame_partner_fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.frame_partner_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # Main DiT blocks
        self.use_partner_cross_attn = bool(
            getattr(args, 'use_partner_cross_attn', False) or False)
        # V6-D: disable the frame_partner_gate/fuse pre-injection in
        # prepare_inputs so that partner conditioning flows ONLY through the
        # per-block cross-attention. frame_partner_gate/fuse modules are
        # still instantiated (init is unchanged) to keep state_dict shapes
        # compatible across V6-B/V6-C/V6-D checkpoints — they simply aren't
        # called in the forward pass when this flag is False.
        self.use_frame_partner_preinject = bool(
            getattr(args, 'use_frame_partner_preinject', True)
            if getattr(args, 'use_frame_partner_preinject', None) is not None
            else True)
        self.blocks = nn.ModuleList([
            DiTBlockLatentARFlow(
                self.hidden_dim, self.n_heads,
                use_partner_cross_attn=self.use_partner_cross_attn,
            )
            for _ in range(self.n_layers)
        ])

        # Final norm + projection back to latent space
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

        # Output head: project back to latent_dim
        self.output_proj = _make_head(self.hidden_dim, self.latent_dim)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    # ── Audio / Partner encoding ──────────────────────────────────────────

    def _extract_wav2vec2_features(self, audio_raw, frame_num):
        feats = self.wav2vec2(audio_raw, frame_num=frame_num)
        if self.use_audio_adapter:
            feats = self.audio_adapter(feats)
        return feats

    def _make_self_audio_kv(self, audio_raw, latent_num, batch_size):
        """Encode self audio to latent-rate features (25 tokens for 100 frames)."""
        if not self.use_audio or audio_raw is None:
            return None
        # wav2vec2 outputs at video frame rate, interpolate to latent rate
        feats = self._extract_wav2vec2_features(audio_raw, latent_num)
        kv = self.self_audio_proj(feats)
        null = self.null_self_audio_kv.expand(batch_size, latent_num, -1)
        dropped = audio_raw.pow(2).mean(-1) < 1e-10
        if dropped.any():
            kv = kv.clone()
            kv[dropped] = null[dropped]
        return kv

    def _encode_other_audio(self, audio_raw, latent_num, batch_size):
        if not self.use_other_audio:
            return None
        null = self.null_other_audio_kv.expand(batch_size, latent_num, -1)
        if audio_raw is None:
            return null
        feats = self._extract_wav2vec2_features(audio_raw, latent_num)
        kv = self.other_audio_proj(feats)
        dropped = audio_raw.pow(2).mean(-1) < 1e-10
        if dropped.any():
            kv = kv.clone()
            kv[dropped] = null[dropped]
        return kv

    def _encode_other_motion(self, other_latent, latent_num, batch_size):
        if not self.use_other_motion:
            return None
        null = self.null_other_motion_kv.expand(batch_size, latent_num, -1)
        if other_latent is None:
            return null
        kv = self.other_latent_proj(other_latent)
        dropped = other_latent.abs().mean(dim=(1, 2)) < 1e-10
        if dropped.any():
            kv = kv.clone()
            kv[dropped] = null[dropped]
        return kv

    def _encode_other_latent_delta(self, other_latent, latent_num, batch_size):
        """Compute and encode partner latent deltas."""
        if not self.use_other_latent_delta:
            return None
        if other_latent is None:
            return torch.zeros(batch_size, latent_num, self.hidden_dim,
                               device=self.other_latent_delta_proj.weight.device)
        # Delta: z_{t} - z_{t-1}, first token padded with zero
        delta = other_latent[:, 1:] - other_latent[:, :-1]
        zero_pad = torch.zeros(other_latent.shape[0], 1, other_latent.shape[2],
                               device=other_latent.device, dtype=other_latent.dtype)
        delta = torch.cat([zero_pad, delta], dim=1)  # (B, latent_num, latent_dim)
        return self.other_latent_delta_proj(delta)

    def _downsample_partner_gaze(self, partner_gaze_feat, latent_num):
        """Downsample frame-rate partner gaze features to latent rate."""
        if partner_gaze_feat is None:
            return None
        if partner_gaze_feat.shape[1] == latent_num:
            return partner_gaze_feat
        gaze = partner_gaze_feat.transpose(1, 2)  # (B, 3, T)
        gaze = F.interpolate(
            gaze,
            size=latent_num,
            mode='linear',
            align_corners=False,
        )
        return gaze.transpose(1, 2)

    def _encode_partner_gaze(self, partner_gaze_feat, latent_num, batch_size):
        if not self.use_partner_gaze:
            return None
        null = self.null_partner_gaze_kv.expand(batch_size, latent_num, -1)
        if partner_gaze_feat is None:
            return null
        gaze_latent = self._downsample_partner_gaze(partner_gaze_feat, latent_num)
        kv = self.partner_gaze_proj(gaze_latent)
        dropped = gaze_latent.abs().mean(dim=(1, 2)) < 1e-10
        if dropped.any():
            kv = kv.clone()
            kv[dropped] = null[dropped]
        return kv

    def _encode_partner_stream(
        self,
        other_latent: Optional[torch.Tensor],
        other_audio_raw: Optional[torch.Tensor],
        partner_gaze_feat: Optional[torch.Tensor],
        latent_num: int,
        batch_size: int,
        partner_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Encode partner stream with causal temporal encoder."""
        motion_ctx = self._encode_other_motion(other_latent, latent_num, batch_size)
        audio_ctx = self._encode_other_audio(other_audio_raw, latent_num, batch_size)
        delta_ctx = self._encode_other_latent_delta(other_latent, latent_num, batch_size)
        gaze_ctx = self._encode_partner_gaze(partner_gaze_feat, latent_num, batch_size)

        streams = [s for s in [motion_ctx, audio_ctx, delta_ctx, gaze_ctx] if s is not None]
        if len(streams) == 0:
            return None, []
        if len(streams) >= 2:
            partner = self.partner_input_fuse(torch.cat(streams, dim=-1))
        else:
            partner = streams[0]

        # Causal temporal encoding
        total_len = partner.shape[1]
        if partner_kv_caches is not None and partner_kv_caches[0] is not None:
            past_len = partner_kv_caches[0][0].shape[2]
        else:
            past_len = 0

        partner_rope_freqs = self.partner_rope(past_len + total_len, device=partner.device)

        new_caches = []
        if partner_kv_caches is None:
            partner_kv_caches = [None] * len(self.partner_encoder_blocks)

        for i, block in enumerate(self.partner_encoder_blocks):
            partner, kv = block(partner, partner_rope_freqs, kv_cache=partner_kv_caches[i])
            new_caches.append(kv)

        return partner, new_caches

    # ── Per-token timestep conditioning ────────────────────────────────────

    def _compute_per_token_conditioning(
        self,
        t_per_token: torch.Tensor,
        n_prev: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token conditioning from per-token timesteps.

        Args:
            t_per_token: (B, n_latents) timesteps, one per target token
            n_prev: number of prev_latent tokens

        Returns:
            c_per_token: (B, n_prev + n_latents, D) — for AdaLN modulation
            c_additive: (B, n_prev + n_latents, D) — additive to token embeddings
        """
        B = t_per_token.shape[0]
        n_target = t_per_token.shape[1]

        # Per-token timestep embeddings
        t_flat = t_per_token.reshape(-1)  # (B * n_latents,)
        t_emb = self.time_embed(t_flat)  # (B * n_latents, D)
        t_emb = t_emb.reshape(B, n_target, self.hidden_dim)  # (B, n_latents, D)

        # Prefix (prev_latents) gets t=0 conditioning (clean context)
        prefix_t = self.time_embed(torch.zeros(B, device=t_per_token.device))  # (B, D)
        prefix_t = prefix_t.unsqueeze(1).expand(B, n_prev, self.hidden_dim)
        c_per_token = torch.cat([prefix_t, t_emb], dim=1)  # (B, n_prev + n_latents, D)

        # Additive token conditioning
        t_additive = self.token_time_proj(t_emb)  # (B, n_latents, D)
        prefix_additive = torch.zeros(B, n_prev, self.hidden_dim,
                                      device=t_per_token.device, dtype=t_per_token.dtype)
        c_additive = torch.cat([prefix_additive, t_additive], dim=1)

        return c_per_token, c_additive

    def finalize_hidden(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply final per-token AdaLN and project back to latent space."""
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        return self.output_proj(x)

    def prepare_inputs(
        self,
        z_noisy: torch.Tensor,
        z_prev: torch.Tensor,
        t_per_token: torch.Tensor,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        z_partner: Optional[torch.Tensor] = None,
        partner_gaze_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare token embeddings and conditioning for a full-sequence forward."""

        B = z_noisy.shape[0]
        n_target = z_noisy.shape[1]
        n_prev = z_prev.shape[1]

        self_audio_kv = self._make_self_audio_kv(self_audio_raw, n_target, B)
        partner_seq, _ = self._encode_partner_stream(
            z_partner, other_audio_raw, partner_gaze_feat, n_target, B)

        z_all = torch.cat([z_prev, z_noisy], dim=1)
        x = self.latent_proj(z_all)
        prev_tokens = x[:, :n_prev]
        target_tokens = x[:, n_prev:]

        partner_full = None
        if partner_seq is not None:
            # Position-locked pre-injection, gated by use_frame_partner_preinject.
            # V6-B / V6-C: pre-injection ON (default, backwards-compatible).
            # V6-D: pre-injection OFF — partner conditioning flows only through
            # the per-block cross-attention path.
            if self.use_frame_partner_preinject:
                partner_cat = torch.cat([target_tokens, partner_seq], dim=-1)
                target_tokens = target_tokens + self.frame_partner_gate(partner_cat) * \
                    self.frame_partner_fuse(partner_cat)
            total_len = n_prev + n_target
            partner_full = torch.zeros(
                B, total_len, self.hidden_dim,
                device=target_tokens.device, dtype=target_tokens.dtype)
            partner_full[:, n_prev:, :] = partner_seq

        x = torch.cat([prev_tokens, target_tokens], dim=1)

        c_per_token, c_additive = self._compute_per_token_conditioning(t_per_token, n_prev)
        x = x + c_additive

        # V6-F: add emotion conditioning to all target tokens (global affect).
        # _current_emotion_ctx is set by the outer model's set_emotion_control().
        emo_ctx = getattr(self, '_current_emotion_ctx', None)
        if emo_ctx is not None:
            if emo_ctx.shape[1] == 1:
                emo_ctx = emo_ctx.expand(-1, n_target, -1)
            x[:, n_prev:n_prev + emo_ctx.shape[1]] = x[:, n_prev:n_prev + emo_ctx.shape[1]] + emo_ctx

        total_len = x.shape[1]
        rope_freqs = self.rope(total_len, device=x.device)

        audio_rope_freqs_q = None
        audio_rope_freqs_k = None
        if self_audio_kv is not None:
            audio_rope_freqs_q = rope_freqs
            audio_rope_freqs_k = self.audio_rope(n_target, device=x.device)

        # Partner cross-attention RoPE freqs (target-time positions only).
        # Listener target row t and partner token t share the same latent clock;
        # we use the main rope positions [n_prev..n_prev+n_target-1] for BOTH Q
        # and K of the partner cross-attention. Strict causal mask then emerges
        # trivially from L_q == L_kv with q_offset=0 in CausalCrossAttention.
        partner_rope_freqs_q = None
        partner_rope_freqs_k = None
        if partner_seq is not None and self.use_partner_cross_attn:
            partner_rope_freqs_q = rope_freqs[n_prev:n_prev + n_target]
            partner_rope_freqs_k = rope_freqs[n_prev:n_prev + n_target]

        return {
            'x': x,
            'c_per_token': c_per_token,
            'partner_full': partner_full,
            'partner_seq': partner_seq,
            'self_audio_kv': self_audio_kv,
            'audio_rope_freqs_q': audio_rope_freqs_q,
            'audio_rope_freqs_k': audio_rope_freqs_k,
            'partner_rope_freqs_q': partner_rope_freqs_q,
            'partner_rope_freqs_k': partner_rope_freqs_k,
            'rope_freqs': rope_freqs,
            'n_prev': n_prev,
            'n_target': n_target,
        }

    # ── Training forward ──────────────────────────────────────────────────

    def forward_features(
        self,
        z_noisy: torch.Tensor,
        z_prev: torch.Tensor,
        t_per_token: torch.Tensor,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        z_partner: Optional[torch.Tensor] = None,
        partner_gaze_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass up to final hidden states."""

        prepared = self.prepare_inputs(
            z_noisy,
            z_prev,
            t_per_token,
            self_audio_raw=self_audio_raw,
            other_audio_raw=other_audio_raw,
            z_partner=z_partner,
            partner_gaze_feat=partner_gaze_feat,
        )
        x = prepared['x']
        c_per_token = prepared['c_per_token']
        partner_full = prepared['partner_full']
        self_audio_kv = prepared['self_audio_kv']
        audio_rope_freqs_q = prepared['audio_rope_freqs_q']
        audio_rope_freqs_k = prepared['audio_rope_freqs_k']
        rope_freqs = prepared['rope_freqs']
        partner_seq = prepared['partner_seq']
        partner_rope_freqs_q = prepared['partner_rope_freqs_q']
        partner_rope_freqs_k = prepared['partner_rope_freqs_k']

        # Main transformer blocks
        for block in self.blocks:
            x, _, _, _ = block(
                x, c_per_token, rope_freqs,
                self_audio_kv=self_audio_kv,
                audio_rope_freqs_q=audio_rope_freqs_q,
                audio_rope_freqs_k=audio_rope_freqs_k,
                partner_ctx=partner_full,
                partner_kv=partner_seq,
                partner_rope_freqs_q=partner_rope_freqs_q,
                partner_rope_freqs_k=partner_rope_freqs_k,
            )

        return {
            'hidden': x,
            'c_per_token': c_per_token,
            'partner_full': partner_full,
        }

    def forward(
        self,
        z_noisy: torch.Tensor,
        z_prev: torch.Tensor,
        t_per_token: torch.Tensor,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        z_partner: Optional[torch.Tensor] = None,
        partner_gaze_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(
            z_noisy,
            z_prev,
            t_per_token,
            self_audio_raw=self_audio_raw,
            other_audio_raw=other_audio_raw,
            z_partner=z_partner,
            partner_gaze_feat=partner_gaze_feat,
        )
        features['z_pred'] = self.finalize_hidden(features['hidden'], features['c_per_token'])
        return features

    # ── Streaming inference: single token generation with KV cache ────────

    def forward_single_token(
        self,
        z_token: torch.Tensor,
        t_token: torch.Tensor,
        rope_freqs: torch.Tensor,
        self_audio_kv: Optional[torch.Tensor] = None,
        audio_rope_freqs_q: Optional[torch.Tensor] = None,
        audio_rope_freqs_k: Optional[torch.Tensor] = None,
        partner_ctx: Optional[torch.Tensor] = None,
        partner_kv: Optional[torch.Tensor] = None,
        partner_rope_freqs_q: Optional[torch.Tensor] = None,
        partner_rope_freqs_k: Optional[torch.Tensor] = None,
        layer_kv_caches: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, List]:
        """Generate one token with KV caching for streaming inference.

        layer_kv_caches is now a list of 3-tuples per layer:
            (self_kv, audio_cross_kv, partner_cross_kv)
        """
        x = z_token
        new_caches = []

        for i, block in enumerate(self.blocks):
            if layer_kv_caches:
                prior = layer_kv_caches[i]
                self_kv_cache = prior[0] if len(prior) > 0 else None
                cross_kv_cache = prior[1] if len(prior) > 1 else None
                partner_cross_kv_cache = prior[2] if len(prior) > 2 else None
            else:
                self_kv_cache = None
                cross_kv_cache = None
                partner_cross_kv_cache = None

            x, self_kv, cross_kv, partner_cross_kv = block(
                x, t_token, rope_freqs,
                self_audio_kv=self_audio_kv,
                audio_rope_freqs_q=audio_rope_freqs_q,
                audio_rope_freqs_k=audio_rope_freqs_k,
                partner_ctx=partner_ctx,
                partner_kv=partner_kv,
                partner_rope_freqs_q=partner_rope_freqs_q,
                partner_rope_freqs_k=partner_rope_freqs_k,
                self_attn_kv_cache=self_kv_cache,
                cross_attn_kv_cache=cross_kv_cache,
                partner_cross_kv_cache=partner_cross_kv_cache,
            )
            new_caches.append((self_kv, cross_kv, partner_cross_kv))

        return x, new_caches


# ── Causal Mouth Refiner ──────────────────────────────────────────────────

class CausalMouthRefiner(nn.Module):
    """Frame-rate audio-driven mouth refinement sidecar.

    Takes frame-rate wav2vec2 features + decoded coarse motion (exp50+jaw3),
    predicts a residual Δ(53D) via causal dilated convolutions.
    Zero-init output so it starts as identity.
    """

    def __init__(self, audio_dim: int = 1024, motion_dim: int = 53,
                 hidden_dim: int = 128, n_layers: int = 4, kernel_size: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(audio_dim + motion_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i  # 1, 2, 4, 8
            padding = (kernel_size - 1) * dilation  # causal: full left padding
            self.convs.append(nn.Conv1d(hidden_dim, hidden_dim,
                                        kernel_size=kernel_size,
                                        dilation=dilation, padding=padding))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, motion_dim)
        # Zero-init so refiner starts as identity (Δ=0)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, audio_feats: torch.Tensor, coarse_motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_feats: (B, T, audio_dim) — wav2vec2 features at frame rate
            coarse_motion: (B, T, 53) — decoded exp50 + jaw3
        Returns:
            delta: (B, T, 53) — residual to add to coarse motion
        """
        x = self.input_proj(torch.cat([audio_feats, coarse_motion], dim=-1))  # (B, T, H)
        for conv, norm in zip(self.convs, self.norms):
            # Causal conv: pad left, trim right
            residual = x
            x_conv = conv(x.transpose(1, 2))  # (B, H, T+pad)
            x_conv = x_conv[..., :x.shape[1]]  # trim to (B, H, T) — causal
            x_conv = x_conv.transpose(1, 2)  # (B, T, H)
            x = norm(F.gelu(x_conv) + residual)
        return self.output_proj(x)


# ── Style conditioning (reference-only, no labels) ────────────────────────

class StyleRefEncoder(nn.Module):
    """Encode a target-side reference clip's AE latent tokens to a style vector.

    Input is the ref_motion already encoded through the frozen AE
    (shape (B, n_latents, latent_dim)). Produces a compact (B, style_dim)
    embedding via a small temporal transformer + mean pool.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256,
                 style_dim: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(hidden_dim, style_dim)

    def forward(self, ref_z: torch.Tensor) -> torch.Tensor:
        # ref_z: (B, n_latents, latent_dim)
        h = self.proj_in(ref_z)
        h = self.encoder(h)
        h = h.mean(dim=1)                 # (B, hidden_dim)
        return self.proj_out(h)            # (B, style_dim)


class StyleConditioner(nn.Module):
    """Project style_z -> per-block (B, 6*hidden_dim) AdaLN modulations.

    Mirrors V7's EmotionConditioner pattern. Final linear is zero-init so a
    freshly warm-started model behaves identically to the V6D backbone until
    style gradients start flowing.
    """

    def __init__(self, style_dim: int = 128, hidden_dim: int = 448,
                 n_blocks: int = 8):
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_blocks * 6 * hidden_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, style_z: torch.Tensor) -> List[torch.Tensor]:
        # style_z: (B, style_dim)
        B = style_z.shape[0]
        mod_all = self.proj(style_z)
        mod_all = mod_all.view(B, self.n_blocks, 6 * self.hidden_dim)
        return [mod_all[:, i] for i in range(self.n_blocks)]


# ── Outer Model Wrapper ───────────────────────────────────────────────────

class MotionPriorV6LatentARFlow(nn.Module):
    """Outer wrapper for V6 Latent AR-Flow model.

    Handles flow matching noise schedule, CFG dropouts, and ODE sampling.
    """

    def __init__(self, args, device='cuda'):
        super().__init__()
        self.latent_dim = getattr(args, 'latent_dim', 256)
        self.n_latents = getattr(args, 'n_latents', 25)
        self.n_prev_latents = getattr(args, 'n_prev_latents', 3)
        self.motion_dim = getattr(args, 'motion_feat_dim', 59) or 59

        self.timestep_sampling = getattr(args, 'timestep_sampling', 'logit_normal') or 'logit_normal'
        lnm = getattr(args, 'logit_normal_mean', None)
        self.logit_normal_mean = lnm if lnm is not None else 0.0
        lns = getattr(args, 'logit_normal_std', None)
        self.logit_normal_std = lns if lns is not None else 1.0

        # Learned start token for first-window generation (replaces prev latents)
        self.start_latent = nn.Parameter(
            torch.randn(1, self.n_prev_latents, self.latent_dim) * 0.02)

        # Core denoiser
        self.denoising_net = FlowMatchingDenoiserLatentARFlow(args, device=device)

        # CFG dropouts
        self.audio_cfg_dropout = getattr(args, 'audio_cfg_dropout', 0.15) or 0.15
        self.prev_latent_dropout = getattr(args, 'prev_latent_dropout', 0.15) or 0.15
        # Partner-channel dropouts. As of V6 these are hierarchical and INDEPENDENT:
        #   1. all_partner_cfg_dropout  — explicit "absent everything" rows
        #   2. on remaining rows, three independent per-channel drops:
        #      partner_motion_cfg_dropout drops ONLY z_partner
        #      partner_audio_cfg_dropout  drops ONLY other_audio_raw
        #      partner_gaze_cfg_dropout   drops ONLY partner_gaze_feat
        # Codex-reviewed recipe (see todo/codex_partner_dropout_review.md).
        # Legacy (V4e/V4f): partner_motion dropout cascaded to all three. V6+ decouples.
        self.all_partner_cfg_dropout = getattr(args, 'all_partner_cfg_dropout', 0.05) or 0.0
        self.partner_motion_cfg_dropout = getattr(args, 'partner_motion_cfg_dropout', 0.10) or 0.10
        self.partner_audio_cfg_dropout = getattr(args, 'partner_audio_cfg_dropout', 0.10) or 0.10
        self.partner_gaze_cfg_dropout = getattr(args, 'partner_gaze_cfg_dropout', 0.0) or 0.0
        # Back-compat toggle: if True, motion dropout still cascades to audio+gaze
        # (old V4e/V4f behavior). Default False for V6+.
        self.legacy_coupled_partner_dropout = bool(
            getattr(args, 'legacy_coupled_partner_dropout', False))
        self.use_partner_gaze = getattr(args, 'use_partner_gaze', False) or False
        self.use_expression_contagion = getattr(
            args, 'use_expression_contagion', False) or False
        self.use_behavior_control = getattr(args, 'use_behavior_control', False) or False
        self.behavior_control_dim = int(getattr(args, 'behavior_control_dim', 8) or 8)

        # V6-F: separate emotion control (7D one-hot → additive conditioning on target tokens)
        self.use_emotion_control = getattr(args, 'use_emotion_control', False) or False
        self.emotion_control_dim = int(getattr(args, 'emotion_control_dim', 7) or 7)
        self._current_emotion_ctx = None  # set via set_emotion_control()

        # V6D-SMIRK style extension: reference-motion -> per-block AdaLN mod.
        # Orthogonal to emotion/behavior; routed exclusively via extra_adaln_mod.
        self.use_style_conditioning = bool(getattr(args, 'use_style_conditioning', False) or False)
        self.style_cfg_dropout = float(getattr(args, 'style_cfg_dropout', 0.15) or 0.15)
        self._current_style_mods = None   # list of (B, 6*D), set via set_style_reference()
        hdim = self.denoising_net.hidden_dim
        if self.use_emotion_control:
            emo_hidden = int(getattr(args, 'emotion_control_hidden', hdim) or hdim)
            self.emotion_control_proj = nn.Sequential(
                nn.Linear(self.emotion_control_dim, emo_hidden),
                nn.SiLU(),
                nn.Linear(emo_hidden, hdim),
            )
            # Zero-init final layer so emotion starts as no-op
            nn.init.zeros_(self.emotion_control_proj[-1].weight)
            nn.init.zeros_(self.emotion_control_proj[-1].bias)
            self.null_emotion_control = nn.Parameter(
                torch.zeros(1, 1, self.emotion_control_dim))

        # Style reference encoder + conditioner (V6D-SMIRK style extension).
        # Only built when `use_style_conditioning: true`. Otherwise the V6D
        # backbone is untouched and checkpoint loading is a clean no-op.
        if self.use_style_conditioning:
            _style_dim = int(getattr(args, 'style_dim', 128) or 128)
            _style_enc_hidden = int(getattr(args, 'style_encoder_hidden', 256) or 256)
            _style_enc_layers = int(getattr(args, 'style_encoder_layers', 2) or 2)
            self.style_ref_encoder = StyleRefEncoder(
                latent_dim=self.latent_dim,
                hidden_dim=_style_enc_hidden,
                style_dim=_style_dim,
                n_layers=_style_enc_layers,
            )
            self.style_conditioner = StyleConditioner(
                style_dim=_style_dim,
                hidden_dim=hdim,
                n_blocks=self.denoising_net.n_layers,
            )
            # Learnable null style embedding used when a row is CFG-dropped or no
            # reference is available. Not zero-init — it should learn an
            # informative "generic" style manifold.
            self.null_style_embedding = nn.Parameter(torch.zeros(1, _style_dim))

        # Auxiliary heads on decoded 59D motion. Optional — set `use_decomp_heads: false`
        # to skip building them entirely (useful when using a learned observer and no
        # decomp supervision, so the heads can't drift with random gradients).
        self.use_decomp_heads = getattr(args, 'use_decomp_heads', True)
        if self.use_decomp_heads is None:
            self.use_decomp_heads = True
        if self.use_decomp_heads:
            self.decoded_gaze_head = _make_head(self.motion_dim, 3)
            self.decoded_rhythmic_head = _make_head(self.motion_dim, 3)
            self.decoded_gaze_contact_head = _make_head(self.motion_dim, 1) \
                if self.use_partner_gaze else None
        else:
            self.decoded_gaze_head = None
            self.decoded_rhythmic_head = None
            self.decoded_gaze_contact_head = None

        # Causal mouth refiner (frame-rate audio → Δ(exp50+jaw3))
        self.use_mouth_refiner = getattr(args, 'use_mouth_refiner', False) or False
        if self.use_mouth_refiner:
            mr_hidden = int(getattr(args, 'mouth_refiner_hidden', 128) or 128)
            mr_layers = int(getattr(args, 'mouth_refiner_layers', 4) or 4)
            mr_kernel = int(getattr(args, 'mouth_refiner_kernel', 5) or 5)
            mr_audio_dim = self.denoising_net.wav2vec2.hidden_size if hasattr(self.denoising_net, 'wav2vec2') else 1024
            self.mouth_refiner = CausalMouthRefiner(
                audio_dim=mr_audio_dim, motion_dim=53,
                hidden_dim=mr_hidden, n_layers=mr_layers, kernel_size=mr_kernel)

        if self.use_expression_contagion:
            expr_hidden = getattr(args, 'expression_contagion_hidden', 256) or 256
            expr_kernel = int(getattr(args, 'expression_contagion_kernel', 5) or 5)
            self.expression_contagion_kernel = max(1, expr_kernel)
            # Base target expression + raw partner expression + partner delta.
            self.expression_contagion_in = nn.Linear(151, expr_hidden)
            self.expression_contagion_conv1 = nn.Conv1d(
                expr_hidden, expr_hidden, kernel_size=self.expression_contagion_kernel)
            self.expression_contagion_conv2 = nn.Conv1d(
                expr_hidden, expr_hidden, kernel_size=self.expression_contagion_kernel)
            self.expression_contagion_out = nn.Linear(expr_hidden, 50)
            nn.init.zeros_(self.expression_contagion_out.weight)
            nn.init.zeros_(self.expression_contagion_out.bias)
        else:
            self.expression_contagion_kernel = 0

        if self.use_behavior_control:
            hidden_dim = self.denoising_net.hidden_dim
            ctrl_hidden = int(getattr(args, 'behavior_control_hidden', hidden_dim) or hidden_dim)
            tail_blocks = int(getattr(args, 'behavior_control_tail_blocks', 3) or 3)
            tail_blocks = max(1, min(tail_blocks, self.denoising_net.n_layers))
            self.behavior_control_tail_blocks = tail_blocks
            self.behavior_controlled_block_indices = list(
                range(self.denoising_net.n_layers - tail_blocks, self.denoising_net.n_layers)
            )
            self.behavior_control_stage_count = tail_blocks + 1  # tail blocks + final head stage
            self.behavior_observer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim, eps=1e-6)
                for _ in range(self.behavior_control_stage_count)
            ])
            self.behavior_observer_heads = nn.ModuleList([
                _make_head(hidden_dim, self.behavior_control_dim)
                for _ in range(self.behavior_control_stage_count)
            ])
            self.behavior_control_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim, eps=1e-6)
                for _ in range(self.behavior_control_stage_count)
            ])
            self.behavior_control_contexts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(3 * hidden_dim, ctrl_hidden),
                    nn.SiLU(),
                )
                for _ in range(self.behavior_control_stage_count)
            ])
            slices = self._behavior_group_slices()
            def _group_dim(name):
                s, e = slices[name]
                return (e - s) * 2  # control + error

            self.behavior_control_gaze = nn.ModuleList([
                _make_zero_init_residual_head(ctrl_hidden + _group_dim('gaze'), ctrl_hidden, hidden_dim)
                for _ in range(self.behavior_control_stage_count)
            ])
            # head_rhythm for 10D, head for 8D
            head_key = 'head_rhythm' if self.behavior_control_dim >= 10 else 'head'
            self.behavior_control_head = nn.ModuleList([
                _make_zero_init_residual_head(ctrl_hidden + _group_dim(head_key), ctrl_hidden, hidden_dim)
                for _ in range(self.behavior_control_stage_count)
            ])
            if self.behavior_control_dim >= 10:
                # V6-E: head_avert replaces intensity
                self.behavior_control_intensity = nn.ModuleList([
                    _make_zero_init_residual_head(ctrl_hidden + _group_dim('head_avert'), ctrl_hidden, hidden_dim)
                    for _ in range(self.behavior_control_stage_count)
                ])
            else:
                # Legacy 8D: intensity
                self.behavior_control_intensity = nn.ModuleList([
                    _make_zero_init_residual_head(ctrl_hidden + 2, ctrl_hidden, hidden_dim)
                    for _ in range(self.behavior_control_stage_count)
                ])
            self.behavior_control_speaking = nn.ModuleList([
                _make_zero_init_residual_head(ctrl_hidden + _group_dim('speaking'), ctrl_hidden, hidden_dim)
                for _ in range(self.behavior_control_stage_count)
            ])

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    # ── Style conditioning API (V6D-SMIRK style extension) ────────────────
    @torch.no_grad()
    def encode_style_reference(self, ref_motion_normed: torch.Tensor,
                               ae_model) -> Optional[torch.Tensor]:
        """Encode a target-side ref_motion clip to a (B, style_dim) style vector.

        Args:
            ref_motion_normed: (B, n_motions, motion_dim) already-normalized motion.
            ae_model: the frozen CausalTemporalMotionAE used for this run.

        Returns:
            style_z: (B, style_dim) if style conditioning is enabled, else None.
        """
        if not self.use_style_conditioning or ae_model is None:
            return None
        ref_z = ae_model.encode_mean(ref_motion_normed)   # (B, n_latents, latent_dim)
        return self.style_ref_encoder(ref_z)              # (B, style_dim)

    def set_style_reference(self, style_z: Optional[torch.Tensor] = None):
        """Persist a style vector for subsequent forward/sample calls.

        Pass None to clear. Style_z can be (B, style_dim) or (1, style_dim);
        it is repeated along the batch dim inside the backbone when needed.
        """
        if not self.use_style_conditioning or style_z is None:
            self._current_style_mods = None
            self.denoising_net._current_style_mods = None
            # Also clear the raw z so later forwards can't accidentally reuse it.
            self._current_style_z = None
            self.denoising_net._current_style_z = None
            return
        # Cache style_z, defer conditioner to backbone so it can handle CFG
        # batch doubling (matching V7's per-forward repeat pattern).
        self._current_style_z = style_z
        self.denoising_net._current_style_z = style_z
        # Invalidate any cached mods; backbone will recompute from z as needed.
        self._current_style_mods = None
        self.denoising_net._current_style_mods = None

    def _resolve_style_mods(self, batch_size: int) -> Optional[List[torch.Tensor]]:
        """Return per-block (B, 6*D) style mods for the current batch, or None.

        Handles CFG batch doubling by repeating style_z when the active batch
        is larger than what was registered via set_style_reference(). Computing
        the mods lazily per forward mirrors V7's emotion conditioning flow.
        """
        if not self.use_style_conditioning:
            return None
        style_z = getattr(self, '_current_style_z', None)
        if style_z is None:
            return None
        if style_z.shape[0] != batch_size:
            if batch_size % style_z.shape[0] != 0:
                # Mismatch that's not a clean CFG-style repeat; fall back to
                # clamping to the registered batch (caller bug). Drop style.
                return None
            n_rep = batch_size // style_z.shape[0]
            style_z = style_z.repeat(n_rep, 1)
        return self.style_conditioner(style_z)

    def set_emotion_control(self, emotion_control=None):
        """Set emotion conditioning for the next forward/sample call.

        Args:
            emotion_control: (B, 7), (B, T, 7), or None (clears).
                If (B, 7), broadcasts across all target tokens.
        """
        if not self.use_emotion_control or emotion_control is None:
            self._current_emotion_ctx = None
            self.denoising_net._current_emotion_ctx = None
            return
        if emotion_control.dim() == 2:
            emotion_control = emotion_control.unsqueeze(1)  # (B, 1, 7)
        ctx = self.emotion_control_proj(emotion_control)  # on self, not denoising_net
        self._current_emotion_ctx = ctx
        self.denoising_net._current_emotion_ctx = ctx

    def _sample_timesteps(self, batch_size, n_tokens=None):
        """Sample flow matching timesteps.

        Args:
            batch_size: number of samples
            n_tokens: if provided, sample independent per-token timesteps
                      Returns (B, n_tokens). If None, returns (B,).
        """
        if n_tokens is not None:
            total = batch_size * n_tokens
        else:
            total = batch_size

        if self.timestep_sampling == 'logit_normal':
            u = torch.randn(total, device=self.device)
            u = u * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)
        else:
            t = torch.rand(total, device=self.device)
        t = t.clamp(1e-5, 1.0 - 1e-5)

        if n_tokens is not None:
            t = t.reshape(batch_size, n_tokens)
        return t

    def predict_decoded_aux(self, decoded_motion: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Predict decomposition targets from decoded motion. Returns None if heads disabled."""
        if not self.use_decomp_heads or self.decoded_gaze_head is None:
            return None
        return {
            'gaze': self.decoded_gaze_head(decoded_motion),
            'rhythmic': self.decoded_rhythmic_head(decoded_motion),
            'gaze_contact_pred': (
                self.decoded_gaze_contact_head(decoded_motion).squeeze(-1)
                if self.decoded_gaze_contact_head is not None else None
            ),
        }

    def _behavior_group_slices(self):
        if self.behavior_control_dim >= 10:
            # V6-E 10D layout: separate gaze_strength, head_rhythm_strength,
            # head_avert_yaw, no shared intensity
            return {
                'gaze': (0, 3),          # 2D mode + 1D strength
                'head_rhythm': (3, 8),   # 4D mode + 1D strength
                'head_avert': (8, 9),    # 1D signed yaw
                'speaking': (9, 10),     # 1D
            }
        # Legacy 8D layout
        return {
            'gaze': (0, 2),
            'head': (2, 6),
            'intensity': (6, 7),
            'speaking': (7, 8),
        }

    def _replace_feature_slice(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([x[..., :start], value, x[..., end:]], dim=-1)

    def _normalize_behavior_head_simplex(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize gaze one-hot (always 0:2)
        gaze = x[..., 0:2].clamp_min(0)
        gaze = gaze / gaze.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        x = self._replace_feature_slice(x, 0, 2, gaze)
        # Normalize head mode one-hot (8D: 2:6, 10D: 3:7)
        if self.behavior_control_dim >= 10:
            head = x[..., 3:7].clamp_min(0)
            head = head / head.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return self._replace_feature_slice(x, 3, 7, head)
        else:
            head = x[..., 2:6].clamp_min(0)
            head = head / head.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return self._replace_feature_slice(x, 2, 6, head)

    def _pack_behavior(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape[-1] < self.behavior_control_dim:
            raw = F.pad(raw, (0, self.behavior_control_dim - raw.shape[-1]))
        if self.behavior_control_dim >= 10:
            # V6-E 10D layout
            gaze_mode = F.softmax(raw[..., 0:2], dim=-1)        # 2-class
            gaze_strength = torch.sigmoid(raw[..., 2:3])        # [0,1]
            head_mode = F.softmax(raw[..., 3:7], dim=-1)        # 4-class
            head_strength = torch.sigmoid(raw[..., 7:8])        # [0,1]
            head_avert_yaw = torch.tanh(raw[..., 8:9])          # [-1,1]
            speakingness = torch.sigmoid(raw[..., 9:10])        # [0,1]
            return torch.cat([gaze_mode, gaze_strength, head_mode,
                              head_strength, head_avert_yaw, speakingness], dim=-1)
        # Legacy 8D
        gaze_mode = F.softmax(raw[..., 0:2], dim=-1)
        head_type = F.softmax(raw[..., 2:6], dim=-1)
        intensity = torch.sigmoid(raw[..., 6:7])
        speakingness = torch.sigmoid(raw[..., 7:8])
        return torch.cat([gaze_mode, head_type, intensity, speakingness], dim=-1)

    def observe_behavior_from_hidden(
        self,
        hidden: torch.Tensor,
        stage_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.use_behavior_control:
            raise RuntimeError("Behavior observer is disabled")
        if stage_idx is None:
            stage_idx = self.behavior_control_stage_count - 1
        norm = self.behavior_observer_norms[stage_idx]
        head = self.behavior_observer_heads[stage_idx]
        return self._pack_behavior(head(norm(hidden)))

    def _prepare_behavior_control(
        self,
        behavior_control: Optional[torch.Tensor],
        behavior_mask: Optional[torch.Tensor],
        target_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if behavior_control is None:
            return None, None
        ctrl = behavior_control
        if ctrl.shape[-1] < self.behavior_control_dim:
            ctrl = F.pad(ctrl, (0, self.behavior_control_dim - ctrl.shape[-1]))
        elif ctrl.shape[-1] > self.behavior_control_dim:
            ctrl = ctrl[..., :self.behavior_control_dim]
        if ctrl.dim() == 2:
            ctrl = ctrl.unsqueeze(1).expand(batch_size, target_len, -1)
        if ctrl.shape[1] != target_len:
            ctrl = F.interpolate(
                ctrl.transpose(1, 2).to(dtype=torch.float32),
                size=target_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        ctrl = ctrl.to(device=device, dtype=dtype)

        if behavior_mask is None:
            mask = torch.ones_like(ctrl)
        else:
            mask = behavior_mask
            if mask.shape[-1] < self.behavior_control_dim:
                mask = F.pad(mask, (0, self.behavior_control_dim - mask.shape[-1]))
            elif mask.shape[-1] > self.behavior_control_dim:
                mask = mask[..., :self.behavior_control_dim]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(batch_size, target_len, -1)
            if mask.shape[1] != target_len:
                mask = F.interpolate(
                    mask.transpose(1, 2).to(dtype=torch.float32),
                    size=target_len,
                    mode='nearest',
                ).transpose(1, 2)
            mask = mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        ctrl = self._normalize_behavior_head_simplex(ctrl)
        return ctrl, mask

    def _control_delta_from_error(
        self,
        stage_idx: int,
        hidden: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor],
        behavior_control: torch.Tensor,
        behavior_error: torch.Tensor,
        ) -> torch.Tensor:
        partner = partner_target
        if partner is None:
            partner = torch.zeros_like(hidden)
        context = self.behavior_control_contexts[stage_idx](
            torch.cat([self.behavior_control_norms[stage_idx](hidden), c_target, partner], dim=-1)
        )
        slices = self._behavior_group_slices()
        delta = 0.0

        def _add_group(mlp_list, key):
            nonlocal delta
            s, e = slices[key]
            delta = delta + mlp_list[stage_idx](
                torch.cat([
                    context,
                    behavior_control[..., s:e],
                    behavior_error[..., s:e],
                ], dim=-1)
            )

        _add_group(self.behavior_control_gaze, 'gaze')
        head_key = 'head_rhythm' if self.behavior_control_dim >= 10 else 'head'
        _add_group(self.behavior_control_head, head_key)
        intensity_key = 'head_avert' if self.behavior_control_dim >= 10 else 'intensity'
        _add_group(self.behavior_control_intensity, intensity_key)
        _add_group(self.behavior_control_speaking, 'speaking')
        return delta

    def _apply_behavior_stage(
        self,
        stage_idx: int,
        hidden_target: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_behavior_control:
            return hidden_target, {}

        obs_base = self.observe_behavior_from_hidden(hidden_target, stage_idx=stage_idx)
        control = behavior_control
        mask = behavior_mask
        if control is None:
            zero_delta = torch.zeros_like(hidden_target)
            return hidden_target, {
                'behavior_obs_base': obs_base,
                'behavior_obs_ctrl': obs_base,
                'behavior_ctrl_energy': hidden_target.new_tensor(0.0),
                'behavior_delta_hidden': zero_delta,
                'behavior_hidden_base': hidden_target,
                'behavior_c_target': c_target,
                'behavior_partner_target': (
                    partner_target if partner_target is not None else torch.zeros_like(hidden_target)
                ),
                'behavior_control_target': None,
                'behavior_control_mask': None,
            }

        error = (control - obs_base.detach()) * mask
        # Zero masked control values so the controller MLPs don't see stale
        # values in masked dims (e.g. head_avert_yaw during contact frames).
        control_masked = control * mask
        delta = self._control_delta_from_error(
            stage_idx,
            hidden_target,
            c_target,
            partner_target,
            control_masked,
            error,
        )
        hidden_ctrl = hidden_target + delta
        obs_ctrl = self.observe_behavior_from_hidden(hidden_ctrl, stage_idx=stage_idx)
        return hidden_ctrl, {
            'behavior_obs_base': obs_base,
            'behavior_obs_ctrl': obs_ctrl,
            'behavior_ctrl_energy': (delta ** 2).mean(),
            'behavior_delta_hidden': delta,
            'behavior_hidden_base': hidden_target,
            'behavior_c_target': c_target,
            'behavior_partner_target': (
                partner_target if partner_target is not None else torch.zeros_like(hidden_target)
            ),
            'behavior_control_target': control,
            'behavior_control_mask': mask,
        }

    def _summarize_behavior_stage_infos(
        self,
        stage_infos: List[Dict[str, torch.Tensor]],
        hidden_target: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not stage_infos:
            return {}
        ctrl_energy = torch.stack([info['behavior_ctrl_energy'] for info in stage_infos]).mean()
        delta_hidden = torch.stack([info['behavior_delta_hidden'] for info in stage_infos]).sum(dim=0)
        stage_obs_base = torch.stack([info['behavior_obs_base'] for info in stage_infos], dim=1)
        stage_obs_ctrl = torch.stack([info['behavior_obs_ctrl'] for info in stage_infos], dim=1)
        stage_hidden_base = torch.stack([info['behavior_hidden_base'] for info in stage_infos], dim=1)
        stage_c_target = torch.stack([info['behavior_c_target'] for info in stage_infos], dim=1)
        stage_partner_target = torch.stack([info['behavior_partner_target'] for info in stage_infos], dim=1)
        last = stage_infos[-1]
        return {
            'behavior_obs_base': last['behavior_obs_base'],
            'behavior_obs_ctrl': last['behavior_obs_ctrl'],
            'behavior_stage_obs_base': stage_obs_base,
            'behavior_stage_obs_ctrl': stage_obs_ctrl,
            'behavior_stage_hidden_base': stage_hidden_base,
            'behavior_stage_c_target': stage_c_target,
            'behavior_stage_partner_target': stage_partner_target,
            'behavior_ctrl_energy': ctrl_energy,
            'behavior_delta_hidden': delta_hidden,
            'behavior_hidden_base': hidden_target,
            'behavior_c_target': c_target,
            'behavior_partner_target': (
                partner_target if partner_target is not None else torch.zeros_like(hidden_target)
            ),
            'behavior_control_target': last['behavior_control_target'],
            'behavior_control_mask': last['behavior_control_mask'],
        }

    def _apply_behavior_stack(
        self,
        hidden_target: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        stage_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_behavior_control:
            return hidden_target, {}
        control, mask = self._prepare_behavior_control(
            behavior_control,
            behavior_mask,
            target_len=hidden_target.shape[1],
            batch_size=hidden_target.shape[0],
            device=hidden_target.device,
            dtype=hidden_target.dtype,
        )
        if stage_indices is None:
            stage_indices = [self.behavior_control_stage_count - 1]
        stage_infos = []
        hidden = hidden_target
        hidden_base = hidden_target
        for stage_idx in stage_indices:
            hidden_base = hidden
            hidden, info = self._apply_behavior_stage(
                stage_idx,
                hidden,
                c_target,
                partner_target=partner_target,
                behavior_control=control,
                behavior_mask=mask,
            )
            stage_infos.append(info)
        return hidden, self._summarize_behavior_stage_infos(
            stage_infos,
            hidden_base,
            c_target,
            partner_target,
        )

    def apply_behavior_control(
        self,
        hidden_target: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._apply_behavior_stack(
            hidden_target,
            c_target,
            partner_target=partner_target,
            behavior_control=behavior_control,
            behavior_mask=behavior_mask,
            stage_indices=[self.behavior_control_stage_count - 1] if self.use_behavior_control else None,
        )

    def compute_behavior_crosstalk(
        self,
        hidden_target: torch.Tensor,
        c_target: torch.Tensor,
        partner_target: Optional[torch.Tensor],
        behavior_control: torch.Tensor,
        behavior_mask: Optional[torch.Tensor] = None,
        delta: float = 0.25,
        group_name: Optional[str] = None,
    ) -> torch.Tensor:
        if not self.use_behavior_control:
            return hidden_target.new_tensor(0.0)
        control, mask = self._prepare_behavior_control(
            behavior_control,
            behavior_mask,
            target_len=hidden_target.shape[1],
            batch_size=hidden_target.shape[0],
            device=hidden_target.device,
            dtype=hidden_target.dtype,
        )
        if control is None:
            return hidden_target.new_tensor(0.0)

        groups = list(self._behavior_group_slices().keys())
        if group_name is None:
            group_idx = int(torch.randint(len(groups), (1,), device=hidden_target.device))
            group_name = groups[group_idx]
        start, end = self._behavior_group_slices()[group_name]
        pert_control = control.clone()
        if group_name == 'head':
            simplex = pert_control[..., start:end]
            head_idx = int(torch.randint(0, simplex.shape[-1], (1,), device=hidden_target.device))
            boost = F.one_hot(
                torch.tensor(head_idx, device=hidden_target.device),
                num_classes=simplex.shape[-1],
            ).to(dtype=simplex.dtype).view(*([1] * (simplex.dim() - 1)), simplex.shape[-1])
            simplex = simplex + delta * boost
            simplex = simplex / simplex.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            pert_control = self._replace_feature_slice(pert_control, start, end, simplex)
        else:
            pert_control = self._replace_feature_slice(
                pert_control,
                start,
                end,
                pert_control[..., start:end] + delta,
            )

        if hidden_target.dim() == 4:
            if c_target.dim() != 4 or partner_target is None or partner_target.dim() != 4:
                raise ValueError("Stagewise behavior crosstalk expects stacked hidden/c/partner tensors")
            stage_losses = []
            for stage_idx in range(hidden_target.shape[1]):
                _, base_info = self._apply_behavior_stage(
                    stage_idx,
                    hidden_target[:, stage_idx],
                    c_target[:, stage_idx],
                    partner_target=partner_target[:, stage_idx],
                    behavior_control=control,
                    behavior_mask=mask,
                )
                _, pert_info = self._apply_behavior_stage(
                    stage_idx,
                    hidden_target[:, stage_idx],
                    c_target[:, stage_idx],
                    partner_target=partner_target[:, stage_idx],
                    behavior_control=pert_control,
                    behavior_mask=mask,
                )
                obs_delta = pert_info['behavior_obs_ctrl'] - base_info['behavior_obs_ctrl']
                off_mask = torch.ones(
                    self.behavior_control_dim,
                    device=hidden_target.device,
                    dtype=obs_delta.dtype,
                )
                off_mask[start:end] = 0.0
                stage_losses.append(((obs_delta * off_mask.view(1, 1, -1)) ** 2).mean())
            return torch.stack(stage_losses).mean()

        final_stage_idx = self.behavior_control_stage_count - 1
        _, base_info = self._apply_behavior_stage(
            final_stage_idx,
            hidden_target,
            c_target,
            partner_target=partner_target,
            behavior_control=control,
            behavior_mask=mask,
        )
        _, pert_info = self._apply_behavior_stage(
            final_stage_idx,
            hidden_target,
            c_target,
            partner_target=partner_target,
            behavior_control=pert_control,
            behavior_mask=mask,
        )
        obs_delta = pert_info['behavior_obs_ctrl'] - base_info['behavior_obs_ctrl']
        off_mask = torch.ones(self.behavior_control_dim, device=hidden_target.device, dtype=obs_delta.dtype)
        off_mask[start:end] = 0.0
        return ((obs_delta * off_mask.view(1, 1, -1)) ** 2).mean()

    def predict_behavior_from_decoded(
        self,
        decoded_motion: torch.Tensor,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_motion: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        aux = self.predict_decoded_aux(decoded_motion)
        if aux is None:
            raise RuntimeError(
                "predict_behavior_from_decoded requires decomp heads. "
                "Either enable `use_decomp_heads: true` or use a learned behavior "
                "classifier observer instead."
            )
        gaze = aux['gaze']
        rhythmic = aux['rhythmic']

        # Gaze mode: 2-class (contact / avert)
        # Use gaze head deviation + eye pose deviation as evidence
        gaze_mag = torch.linalg.norm(gaze[..., :2], dim=-1, keepdim=True)
        eye_pose = decoded_motion[..., 56:59]
        eye_mag = torch.linalg.norm(eye_pose, dim=-1, keepdim=True)
        # Contact evidence: low gaze head deviation + low eye deviation
        contact_logit = -gaze_mag / 8.0 - eye_mag / 0.15
        # Avert evidence: high gaze head deviation or high eye deviation
        avert_logit = gaze_mag / 8.0 + eye_mag / 0.15 - 1.0
        if aux.get('gaze_contact_pred') is not None:
            contact_logit = contact_logit + torch.sigmoid(
                aux['gaze_contact_pred'].unsqueeze(-1)
            )
            avert_logit = avert_logit - torch.sigmoid(
                aux['gaze_contact_pred'].unsqueeze(-1)
            )
        gaze_mode = torch.softmax(
            torch.cat([contact_logit, avert_logit], dim=-1), dim=-1,
        )

        # Head mode
        intensity = torch.clamp(
            torch.sqrt((rhythmic ** 2).mean(dim=-1, keepdim=True) + 1e-8) / 2.21,
            0.0,
            1.0,
        )
        head_energy = rhythmic.abs()
        head_axis = head_energy / head_energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        still_logit = -((intensity - 0.05) / 0.20) ** 2
        nod_logit = -((intensity - 0.75) / 0.35) ** 2 - (
            (head_axis - head_axis.new_tensor([1.0, 0.0, 0.0])) ** 2
        ).sum(dim=-1, keepdim=True) / 0.25
        shake_logit = -((intensity - 0.75) / 0.35) ** 2 - (
            (head_axis - head_axis.new_tensor([0.0, 1.0, 0.0])) ** 2
        ).sum(dim=-1, keepdim=True) / 0.25
        tilt_logit = -((intensity - 0.75) / 0.35) ** 2 - (
            (head_axis - head_axis.new_tensor([0.0, 0.0, 1.0])) ** 2
        ).sum(dim=-1, keepdim=True) / 0.25
        head_mode = torch.softmax(
            torch.cat([still_logit, nod_logit, shake_logit, tilt_logit], dim=-1),
            dim=-1,
        )

        jaw = decoded_motion[..., 50:53]
        if jaw.shape[1] > 1:
            jaw_delta = torch.linalg.norm(jaw[:, 1:] - jaw[:, :-1], dim=-1, keepdim=True)
            jaw_speed = torch.cat([jaw_delta[:, :1], jaw_delta], dim=1)
        else:
            jaw_speed = torch.zeros_like(intensity)
        jaw_scale = jaw_speed.amax(dim=1, keepdim=True).clamp_min(1e-4)
        jaw_drive = torch.clamp(jaw_speed / jaw_scale, 0.0, 1.0)

        audio_drive = torch.zeros_like(intensity)
        if self_audio_raw is not None:
            audio = self_audio_raw.to(dtype=torch.float32)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            rms = F.adaptive_avg_pool1d(
                audio.square().unsqueeze(1),
                decoded_motion.shape[1],
            ).transpose(1, 2).sqrt()
            rms_scale = rms.amax(dim=1, keepdim=True).clamp_min(1e-4)
            audio_drive = torch.clamp(rms / rms_scale, 0.0, 1.0).to(dtype=decoded_motion.dtype)

        speakingness = torch.clamp(0.7 * audio_drive + 0.3 * jaw_drive, 0.0, 1.0)
        behavior = torch.cat(
            [gaze_mode, head_mode, intensity, speakingness],
            dim=-1,
        )
        if target_len is None or decoded_motion.shape[1] == target_len:
            return behavior
        pooled = F.interpolate(
            behavior.transpose(1, 2),
            size=target_len,
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)
        # Re-normalize one-hot groups after interpolation
        gaze_p = pooled[..., 0:2].clamp_min(0)
        gaze_p = gaze_p / gaze_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        head_p = pooled[..., 2:6].clamp_min(0)
        head_p = head_p / head_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.cat([gaze_p, head_p, pooled[..., 6:]], dim=-1)
        return pooled

    def _expression_event_curve(self, exp_motion: torch.Tensor) -> torch.Tensor:
        event = exp_motion.abs().mean(dim=-1)
        if exp_motion.shape[1] > 1:
            delta_step = (exp_motion[:, 1:] - exp_motion[:, :-1]).abs().mean(dim=-1)
            delta = torch.cat([delta_step[:, :1], delta_step], dim=1)
            event = 0.5 * event + 0.5 * delta
        return event

    def _compute_affect_from_motion(
        self,
        target_motion: torch.Tensor,
        partner_motion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if partner_motion is None:
            return torch.zeros_like(target_motion[..., :1])
        partner_motion = self._resize_motion_to_frames(partner_motion, target_motion.shape[1])
        target_event = self._expression_event_curve(target_motion[..., :50])
        partner_event = self._expression_event_curve(partner_motion[..., :50])
        lag_min = max(int(round(target_motion.shape[1] * 0.04)), 2)
        lag_max = max(int(round(target_motion.shape[1] * 0.30)), lag_min)
        lagged_partner = torch.zeros_like(partner_event)
        for lag in range(lag_min, lag_max + 1):
            if lag < partner_event.shape[1]:
                lagged_partner[:, lag:] = torch.maximum(
                    lagged_partner[:, lag:],
                    partner_event[:, :-lag],
                )
        affect = lagged_partner * target_event
        scale = torch.quantile(affect.detach(), 0.9, dim=1, keepdim=True).clamp_min(1e-4)
        affect = torch.clamp(affect / scale, 0.0, 1.0)
        return affect.unsqueeze(-1)

    def _resize_motion_to_frames(
        self,
        motion: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        if motion.shape[1] == target_frames:
            return motion
        motion_t = motion.transpose(1, 2)
        motion_t = F.interpolate(
            motion_t,
            size=target_frames,
            mode='linear',
            align_corners=False,
        )
        return motion_t.transpose(1, 2)

    def _causal_conv1d(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        pad = conv.kernel_size[0] - 1
        if pad > 0:
            x = F.pad(x, (pad, 0))
        return conv(x)

    def apply_mouth_refiner(
        self,
        decoded_motion: torch.Tensor,
        self_audio_raw: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply frame-rate mouth refinement from self-audio.

        Args:
            decoded_motion: (B, T, 59) decoded coarse motion
            self_audio_raw: (B, N_samples) raw waveform at 16kHz
            hidden_states: ignored in v6 (used by v7 subclass)
        Returns:
            refined_motion: (B, T, 59) with exp50+jaw3 corrected
        """
        if not self.use_mouth_refiner or self_audio_raw is None:
            return decoded_motion
        T = decoded_motion.shape[1]
        # Extract wav2vec2 features at FRAME rate (T=100), not latent rate (25)
        audio_feats = self.denoising_net._extract_wav2vec2_features(self_audio_raw, T)  # (B, T, audio_dim)
        coarse_oral = decoded_motion[..., :53]  # exp50 + jaw3
        delta = self.mouth_refiner(audio_feats, coarse_oral)  # (B, T, 53)
        refined = torch.cat([
            decoded_motion[..., :53] + delta,
            decoded_motion[..., 53:],  # neck3 + eye3 unchanged
        ], dim=-1)
        return refined

    def apply_expression_contagion(
        self,
        decoded_motion: torch.Tensor,
        other_motion: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        return_delta: bool = False,
    ):
        """Add a causal decoded-space residual on exp[0:50] driven by partner expression."""
        if not self.use_expression_contagion or other_motion is None:
            if return_delta:
                zero = torch.zeros_like(decoded_motion[..., :50])
                return decoded_motion, zero
            return decoded_motion

        other_motion = self._resize_motion_to_frames(other_motion, decoded_motion.shape[1])
        partner_exp = other_motion[..., :50]
        if partner_exp.shape[1] > 1:
            partner_delta_step = partner_exp[:, 1:] - partner_exp[:, :-1]
            partner_delta = torch.cat([partner_delta_step[:, :1], partner_delta_step], dim=1)
        else:
            partner_delta = torch.zeros_like(partner_exp)

        base_exp = decoded_motion[..., :50]
        # Hybrid control removed social_affect from the primary control vector.
        affect_ctrl = torch.zeros_like(base_exp[..., :1])
        feat = torch.cat([base_exp, partner_exp, partner_delta, affect_ctrl], dim=-1)
        h = self.expression_contagion_in(feat).transpose(1, 2)
        h = F.gelu(self._causal_conv1d(h, self.expression_contagion_conv1))
        h = F.gelu(self._causal_conv1d(h, self.expression_contagion_conv2))
        h = h.transpose(1, 2)

        delta_exp = self.expression_contagion_out(h)

        partner_present = (
            partner_exp.abs().mean(dim=(1, 2), keepdim=True) > 1e-8
        ).to(dtype=delta_exp.dtype)
        delta_exp = delta_exp * partner_present

        motion = torch.cat([decoded_motion[..., :50] + delta_exp, decoded_motion[..., 50:]], dim=-1)

        if return_delta:
            return motion, delta_exp
        return motion

    def _prime_prev_caches(
        self,
        z_prev: torch.Tensor,
        batch_size: int,
    ) -> List[Tuple]:
        """Push clean prev latents through all layers to initialize KV caches.

        Returns a list of per-layer (self_kv, audio_cross_kv, partner_cross_kv)
        tuples. Partner cross-attn cache starts None since prev tokens don't
        attend to any partner at this prime step.
        """
        net = self.denoising_net
        rope_freqs = net.rope(self.n_prev_latents + self.n_latents, device=self.device)

        x = net.latent_proj(z_prev)
        prefix_t = net.time_embed(torch.zeros(batch_size, device=self.device))
        c_per_token = prefix_t.unsqueeze(1).expand(batch_size, self.n_prev_latents, net.hidden_dim)

        layer_kv_caches = []
        for block in net.blocks:
            x, self_kv, cross_kv, partner_cross_kv = block(
                x, c_per_token, rope_freqs,
                self_audio_kv=None,
                audio_rope_freqs_q=None,
                audio_rope_freqs_k=None,
                partner_ctx=None,
                partner_kv=None,
                partner_rope_freqs_q=None,
                partner_rope_freqs_k=None,
                self_attn_kv_cache=None,
                cross_attn_kv_cache=None,
                partner_cross_kv_cache=None,
            )
            layer_kv_caches.append((self_kv, cross_kv, partner_cross_kv))

        return layer_kv_caches

    def _run_backbone_with_behavior(
        self,
        x: torch.Tensor,
        c_per_token: torch.Tensor,
        rope_freqs: torch.Tensor,
        self_audio_kv: Optional[torch.Tensor] = None,
        audio_rope_freqs_q: Optional[torch.Tensor] = None,
        audio_rope_freqs_k: Optional[torch.Tensor] = None,
        partner_full: Optional[torch.Tensor] = None,
        partner_seq: Optional[torch.Tensor] = None,
        partner_rope_freqs_q: Optional[torch.Tensor] = None,
        partner_rope_freqs_k: Optional[torch.Tensor] = None,
        n_prev_tokens: int = 0,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x.shape[0]
        style_mods = self._resolve_style_mods(B)
        # Preallocate a (B, L, 6*D) buffer once; prev-token rows stay zero and
        # target-token rows are overwritten per block. Mirrors V7's pattern.
        _style_buf = None
        if style_mods is not None:
            _style_buf = x.new_zeros(B, x.shape[1], 6 * self.denoising_net.hidden_dim)

        def _extra_mod_for(block_idx: int) -> Optional[torch.Tensor]:
            if style_mods is None:
                return None
            mod = style_mods[block_idx]   # (B, 6*D)
            if n_prev_tokens > 0:
                _style_buf[:, n_prev_tokens:] = mod[:, None, :].expand(-1, _style_buf.shape[1] - n_prev_tokens, -1)
                _style_buf[:, :n_prev_tokens] = 0.0
            else:
                _style_buf[:] = mod[:, None, :].expand(-1, _style_buf.shape[1], -1)
            return _style_buf

        if not self.use_behavior_control:
            h = x
            for block_idx, block in enumerate(self.denoising_net.blocks):
                h, _, _, _ = block(
                    h, c_per_token, rope_freqs,
                    self_audio_kv=self_audio_kv,
                    audio_rope_freqs_q=audio_rope_freqs_q,
                    audio_rope_freqs_k=audio_rope_freqs_k,
                    partner_ctx=partner_full,
                    partner_kv=partner_seq,
                    partner_rope_freqs_q=partner_rope_freqs_q,
                    partner_rope_freqs_k=partner_rope_freqs_k,
                    extra_adaln_mod=_extra_mod_for(block_idx),
                )
            return h, {}

        h = x
        target_c = c_per_token[:, n_prev_tokens:] if n_prev_tokens > 0 else c_per_token
        target_partner = (
            partner_full[:, n_prev_tokens:]
            if partner_full is not None and n_prev_tokens > 0 else partner_full
        )
        stage_map = {
            block_idx: stage_idx
            for stage_idx, block_idx in enumerate(self.behavior_controlled_block_indices)
        }
        stage_infos = []
        for block_idx, block in enumerate(self.denoising_net.blocks):
            if block_idx in stage_map:
                target_hidden = h[:, n_prev_tokens:] if n_prev_tokens > 0 else h
                target_hidden, info = self._apply_behavior_stage(
                    stage_map[block_idx],
                    target_hidden,
                    target_c,
                    partner_target=target_partner,
                    behavior_control=behavior_control,
                    behavior_mask=behavior_mask,
                )
                stage_infos.append(info)
                h = torch.cat([h[:, :n_prev_tokens], target_hidden], dim=1) if n_prev_tokens > 0 else target_hidden
            h, _, _, _ = block(
                h, c_per_token, rope_freqs,
                self_audio_kv=self_audio_kv,
                audio_rope_freqs_q=audio_rope_freqs_q,
                audio_rope_freqs_k=audio_rope_freqs_k,
                partner_ctx=partner_full,
                partner_kv=partner_seq,
                partner_rope_freqs_q=partner_rope_freqs_q,
                partner_rope_freqs_k=partner_rope_freqs_k,
                extra_adaln_mod=_extra_mod_for(block_idx),
            )

        target_hidden = h[:, n_prev_tokens:] if n_prev_tokens > 0 else h
        target_hidden, info = self._apply_behavior_stage(
            self.behavior_control_stage_count - 1,
            target_hidden,
            target_c,
            partner_target=target_partner,
            behavior_control=behavior_control,
            behavior_mask=behavior_mask,
        )
        stage_infos.append(info)
        h = torch.cat([h[:, :n_prev_tokens], target_hidden], dim=1) if n_prev_tokens > 0 else target_hidden
        behavior_info = self._summarize_behavior_stage_infos(
            stage_infos,
            stage_infos[-1]['behavior_hidden_base'],
            stage_infos[-1]['behavior_c_target'],
            stage_infos[-1]['behavior_partner_target'],
        )
        return h, behavior_info

    def _run_single_token_with_behavior(
        self,
        x: torch.Tensor,
        c_token: torch.Tensor,
        rope_freqs: torch.Tensor,
        self_audio_kv: Optional[torch.Tensor] = None,
        audio_rope_freqs_q: Optional[torch.Tensor] = None,
        audio_rope_freqs_k: Optional[torch.Tensor] = None,
        partner_token: Optional[torch.Tensor] = None,
        partner_rope_freqs_q: Optional[torch.Tensor] = None,
        partner_rope_freqs_k: Optional[torch.Tensor] = None,
        layer_kv_caches: Optional[List] = None,
        behavior_control_token: Optional[torch.Tensor] = None,
        behavior_mask_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List]:
        """Streaming single-token forward.

        layer_kv_caches is a list of per-layer 3-tuples:
            (self_kv, audio_cross_kv, partner_cross_kv)
        """
        h = x
        new_caches = []
        stage_infos = []
        stage_map = {
            block_idx: stage_idx
            for stage_idx, block_idx in enumerate(self.behavior_controlled_block_indices)
        } if self.use_behavior_control else {}
        if layer_kv_caches is None:
            layer_kv_caches = [None] * len(self.denoising_net.blocks)

        # Resolve style mods once for this token. Single-token streaming has
        # no prev-region in `x`, so the per-block extra is (B, 1, 6*D).
        B = h.shape[0]
        style_mods = self._resolve_style_mods(B)

        for block_idx, block in enumerate(self.denoising_net.blocks):
            if self.use_behavior_control and block_idx in stage_map:
                h, info = self._apply_behavior_stage(
                    stage_map[block_idx],
                    h,
                    c_token,
                    partner_target=partner_token,
                    behavior_control=behavior_control_token,
                    behavior_mask=behavior_mask_token,
                )
                stage_infos.append(info)
            prior = layer_kv_caches[block_idx]
            if prior is not None:
                self_kv_cache = prior[0] if len(prior) > 0 else None
                cross_kv_cache = prior[1] if len(prior) > 1 else None
                partner_cross_kv_cache = prior[2] if len(prior) > 2 else None
            else:
                self_kv_cache = None
                cross_kv_cache = None
                partner_cross_kv_cache = None
            extra_mod = None
            if style_mods is not None:
                extra_mod = style_mods[block_idx].unsqueeze(1)  # (B, 1, 6*D)
            h, self_kv, cross_kv, partner_cross_kv = block(
                h, c_token, rope_freqs,
                self_audio_kv=self_audio_kv,
                audio_rope_freqs_q=audio_rope_freqs_q,
                audio_rope_freqs_k=audio_rope_freqs_k,
                partner_ctx=partner_token,
                partner_kv=partner_token,
                partner_rope_freqs_q=partner_rope_freqs_q,
                partner_rope_freqs_k=partner_rope_freqs_k,
                self_attn_kv_cache=self_kv_cache,
                cross_attn_kv_cache=cross_kv_cache,
                partner_cross_kv_cache=partner_cross_kv_cache,
                extra_adaln_mod=extra_mod,
            )
            new_caches.append((self_kv, cross_kv, partner_cross_kv))

        if self.use_behavior_control:
            h, info = self._apply_behavior_stage(
                self.behavior_control_stage_count - 1,
                h,
                c_token,
                partner_target=partner_token,
                behavior_control=behavior_control_token,
                behavior_mask=behavior_mask_token,
            )
            stage_infos.append(info)
            behavior_info = self._summarize_behavior_stage_infos(
                stage_infos,
                stage_infos[-1]['behavior_hidden_base'],
                stage_infos[-1]['behavior_c_target'],
                stage_infos[-1]['behavior_partner_target'],
            )
        else:
            behavior_info = {}
        return h, behavior_info, new_caches

    @torch.no_grad()
    def encode_partner_token(
        self,
        z_partner_token: Optional[torch.Tensor] = None,
        other_audio_token: Optional[torch.Tensor] = None,
        partner_gaze_token: Optional[torch.Tensor] = None,
        partner_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Encode one partner token causally for token-by-token streaming."""
        if z_partner_token is None and other_audio_token is None and partner_gaze_token is None:
            return None, partner_kv_caches if partner_kv_caches is not None else []

        if batch_size is None:
            if z_partner_token is not None:
                batch_size = z_partner_token.shape[0]
            elif partner_gaze_token is not None:
                batch_size = partner_gaze_token.shape[0]
            else:
                batch_size = other_audio_token.shape[0]

        return self.denoising_net._encode_partner_stream(
            z_partner_token,
            other_audio_token,
            partner_gaze_token,
            latent_num=1,
            batch_size=batch_size,
            partner_kv_caches=partner_kv_caches,
        )

    def forward(
        self,
        z_target: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
        time_step: Optional[torch.Tensor] = None,
        noise_override: Optional[torch.Tensor] = None,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        z_partner: Optional[torch.Tensor] = None,
        partner_gaze_feat: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        enable_dropouts: bool = True,
    ):
        """Training forward pass.

        Args:
            z_target: (B, n_latents, latent_dim) — clean target latent tokens
            z_prev: (B, n_prev_latents, latent_dim) — prev latent tokens
            time_step: (B, n_latents) — per-token timesteps (if None, sampled)
            noise_override: (B, n_latents, latent_dim) — noise (if None, sampled)
            self_audio_raw: (B, N_samples) raw waveform
            other_audio_raw: (B, N_samples) partner raw waveform
            z_partner: (B, n_latents, latent_dim) partner motion latent tokens
            enable_dropouts: whether to apply CFG dropouts

        Returns:
            noise: (B, n_latents, latent_dim) — noise used
            z_pred: (B, n_latents, latent_dim) — predicted clean z
            z_target: (B, n_latents, latent_dim) — ground truth z
            decomposed: dict with full predictions
        """
        batch_size = z_target.shape[0]

        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)

        # Apply CFG dropouts
        if self.training and enable_dropouts:
            if self_audio_raw is not None and self.audio_cfg_dropout > 0:
                mask_audio = torch.rand(batch_size, device=self.device) < self.audio_cfg_dropout
                if mask_audio.any():
                    self_audio_raw = self_audio_raw.clone()
                    self_audio_raw[mask_audio] = 0.0

            # ── Partner-channel dropouts ─────────────────────────────────
            # V4e/V4f legacy path: partner_motion dropout cascades to audio+gaze.
            # V6+ decoupled path: hierarchical. First sample all_partner; on remaining
            # rows, sample three independent per-channel masks.
            if self.legacy_coupled_partner_dropout:
                # Legacy V4e/V4f behavior — preserved for resume compatibility.
                if z_partner is not None and self.partner_motion_cfg_dropout > 0:
                    mask_pm = torch.rand(batch_size, device=self.device) < self.partner_motion_cfg_dropout
                    if mask_pm.any():
                        z_partner = z_partner.clone()
                        z_partner[mask_pm] = 0.0
                        if other_audio_raw is not None:
                            other_audio_raw = other_audio_raw.clone()
                            other_audio_raw[mask_pm] = 0.0
                        if partner_gaze_feat is not None:
                            partner_gaze_feat = partner_gaze_feat.clone()
                            partner_gaze_feat[mask_pm] = 0.0
                if other_audio_raw is not None and self.partner_audio_cfg_dropout > 0:
                    mask_oa = torch.rand(batch_size, device=self.device) < self.partner_audio_cfg_dropout
                    if mask_oa.any():
                        other_audio_raw = other_audio_raw.clone()
                        other_audio_raw[mask_oa] = 0.0
            else:
                # V6+ hierarchical decoupled path.
                # Step 1: "all partner absent" rows (explicit absent branch).
                mask_all = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                if self.all_partner_cfg_dropout > 0:
                    mask_all = torch.rand(batch_size, device=self.device) < self.all_partner_cfg_dropout
                # Step 2: on non-all-partner rows, sample three independent channel masks.
                not_all = ~mask_all
                def _channel_mask(p):
                    if p <= 0:
                        return torch.zeros_like(not_all)
                    m = torch.rand(batch_size, device=self.device) < p
                    return m & not_all
                mask_motion = _channel_mask(self.partner_motion_cfg_dropout)
                mask_audio_p = _channel_mask(self.partner_audio_cfg_dropout)
                mask_gaze = _channel_mask(self.partner_gaze_cfg_dropout)
                # Apply drops. "all partner" rows get all three channels zeroed.
                if z_partner is not None:
                    drop_motion = mask_all | mask_motion
                    if drop_motion.any():
                        z_partner = z_partner.clone()
                        z_partner[drop_motion] = 0.0
                if other_audio_raw is not None:
                    drop_audio = mask_all | mask_audio_p
                    if drop_audio.any():
                        other_audio_raw = other_audio_raw.clone()
                        other_audio_raw[drop_audio] = 0.0
                if partner_gaze_feat is not None:
                    drop_gaze = mask_all | mask_gaze
                    if drop_gaze.any():
                        partner_gaze_feat = partner_gaze_feat.clone()
                        partner_gaze_feat[drop_gaze] = 0.0

            if self.prev_latent_dropout > 0:
                drop_prev = torch.rand(batch_size, device=self.device) < self.prev_latent_dropout
                if drop_prev.any():
                    z_prev = z_prev.clone()
                    z_prev[drop_prev] = self.start_latent.expand(drop_prev.sum(), -1, -1)

        # Sample timesteps (per-token independent)
        if time_step is not None:
            t = time_step
        else:
            t = self._sample_timesteps(batch_size, self.n_latents)

        # Sample noise
        noise = noise_override if noise_override is not None else torch.randn_like(z_target)

        # Rectified flow: z_t = (1-t)*z_0 + t*noise
        t_exp = t.unsqueeze(-1)  # (B, n_latents, 1)
        z_noisy = (1.0 - t_exp) * z_target + t_exp * noise

        prepared = self.denoising_net.prepare_inputs(
            z_noisy,
            z_prev,
            t,
            self_audio_raw=self_audio_raw,
            other_audio_raw=other_audio_raw,
            z_partner=z_partner,
            partner_gaze_feat=partner_gaze_feat,
        )

        hidden_full, behavior_info = self._run_backbone_with_behavior(
            prepared['x'],
            prepared['c_per_token'],
            prepared['rope_freqs'],
            self_audio_kv=prepared['self_audio_kv'],
            audio_rope_freqs_q=prepared['audio_rope_freqs_q'],
            audio_rope_freqs_k=prepared['audio_rope_freqs_k'],
            partner_full=prepared['partner_full'],
            partner_seq=prepared.get('partner_seq'),
            partner_rope_freqs_q=prepared.get('partner_rope_freqs_q'),
            partner_rope_freqs_k=prepared.get('partner_rope_freqs_k'),
            n_prev_tokens=prepared['n_prev'],
            behavior_control=behavior_control,
            behavior_mask=behavior_mask,
        )
        c_per_token = prepared['c_per_token']
        decomposed = {
            'hidden': hidden_full,
            'c_per_token': c_per_token,
            'partner_full': prepared['partner_full'],
        }
        z_pred_full = self.denoising_net.finalize_hidden(hidden_full, c_per_token)
        decomposed['z_pred'] = z_pred_full
        decomposed.update(behavior_info)
        z_pred = z_pred_full[:, self.n_prev_latents:]  # (B, n_latents, latent_dim)

        return noise, z_pred, z_target.detach(), decomposed

    @torch.no_grad()
    def sample(
        self,
        z_prev: Optional[torch.Tensor] = None,
        self_audio_raw: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        z_partner: Optional[torch.Tensor] = None,
        partner_gaze_feat: Optional[torch.Tensor] = None,
        behavior_control: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        cfg_scale: float = 1.0,
        partner_cfg_scale: float = 1.0,
        behavior_cfg_scale: float = 1.0,
        sample_steps: int = 10,
        z_at_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-streaming sample: generate all n_latents tokens at once with causal mask.

        Returns:
            z_0: (B, n_latents, latent_dim) generated clean latent tokens
            z_at_T: (B, n_latents, latent_dim) initial noise

        CFG modes:
          - cfg_scale > 1: self-audio CFG (null branch drops self_audio only).
          - partner_cfg_scale > 1: partner CFG — null branch additionally drops
            z_partner, other_audio_raw, partner_gaze_feat (matches training's
            all_partner_cfg_dropout null). Amplifies partner's influence on
            the listener output.
          - both > 1: applied jointly on the same 2-branch pass (null drops
            self-audio AND partner; cond has everything). Guidance is a single
            combined scale = max(cfg_scale, partner_cfg_scale) — for finer
            control set one to 1.0.
        """
        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)
        else:
            batch_size = z_prev.shape[0]

        if z_at_T is None:
            z_at_T = torch.randn(
                batch_size, self.n_latents, self.latent_dim, device=self.device)

        has_audio = self_audio_raw is not None
        use_audio_cfg = has_audio and cfg_scale != 1.0
        use_partner_cfg = partner_cfg_scale != 1.0
        use_behavior_cfg = behavior_cfg_scale != 1.0 and behavior_control is not None
        if use_audio_cfg or use_partner_cfg or use_behavior_cfg:
            # CFG: duplicate batch for conditional and unconditional.
            # Unconditional branch gets null audio + null behavior (mask=zeros).
            # The model was trained with behavior_cfg_dropout=0.20 so it learned
            # the null-behavior baseline. Previously both halves got identical
            # labels, which cancelled behavior guidance entirely.
            # With contrastive_behavior=True (Contrastive CFG, Nov 2024), the
            # unconditional branch uses OPPOSITE labels instead of null for
            # stronger behavior separation.
            n = 2
            # Self-audio: null-branch zeros self_audio iff audio-CFG is active.
            if has_audio:
                null_self_audio = torch.zeros_like(self_audio_raw)
                null_self_use = null_self_audio if use_audio_cfg else self_audio_raw
                self_audio_in = torch.cat([null_self_use, self_audio_raw], dim=0)
            else:
                self_audio_in = None
            z_prev_in = torch.cat([z_prev, z_prev], dim=0)
            # Partner stream: null branch drops z_partner, other_audio,
            # partner_gaze together iff partner-CFG is active (matches training
            # all_partner_cfg_dropout path — see compute_loss above).
            if other_audio_raw is not None:
                null_other = torch.zeros_like(other_audio_raw)
                null_other_use = null_other if use_partner_cfg else other_audio_raw
                other_audio_in = torch.cat([null_other_use, other_audio_raw], dim=0)
            else:
                other_audio_in = None
            if z_partner is not None:
                null_zp = torch.zeros_like(z_partner)
                null_zp_use = null_zp if use_partner_cfg else z_partner
                z_partner_in = torch.cat([null_zp_use, z_partner], dim=0)
            else:
                z_partner_in = None
            if partner_gaze_feat is not None:
                null_pg = torch.zeros_like(partner_gaze_feat)
                null_pg_use = null_pg if use_partner_cfg else partner_gaze_feat
                partner_gaze_in = torch.cat([null_pg_use, partner_gaze_feat], dim=0)
            else:
                partner_gaze_in = None
            # Behavior: null branch drops behavior (null labels + mask=zeros)
            # iff behavior-CFG is active. Null-behavior baseline was learned
            # via behavior_cfg_dropout=0.20 during training.
            if behavior_control is not None:
                null_behavior = torch.zeros_like(behavior_control)
                null_beh_use = null_behavior if use_behavior_cfg else behavior_control
                behavior_control_in = torch.cat([null_beh_use, behavior_control], dim=0)
                if behavior_mask is not None:
                    null_beh_mask = torch.zeros_like(behavior_mask)
                    null_beh_mask_use = null_beh_mask if use_behavior_cfg else behavior_mask
                    behavior_mask_in = torch.cat([null_beh_mask_use, behavior_mask], dim=0)
                else:
                    behavior_mask_in = None
            else:
                behavior_control_in = None
                behavior_mask_in = None
        else:
            n = 1
            self_audio_in = self_audio_raw
            z_prev_in = z_prev
            other_audio_in = other_audio_raw
            z_partner_in = z_partner
            partner_gaze_in = partner_gaze_feat
            behavior_control_in = behavior_control
            behavior_mask_in = behavior_mask

        dt = 1.0 / sample_steps
        timesteps = torch.linspace(1.0, dt, sample_steps, device=self.device)
        z_t = z_at_T

        # CFG-Zero* disabled — was causing noisy output. The flow-matching
        # model's first-step estimate at t=1.0 is needed for trajectory quality.
        # TODO: revisit with per-step scale optimization from CFG-Zero* paper.

        for step_i, t_val in enumerate(timesteps):
            # Uniform timestep for all tokens during sampling
            t_batch = torch.full(
                (batch_size * n, self.n_latents), t_val.item(), device=self.device)
            z_in = torch.cat([z_t] * n, dim=0) if n > 1 else z_t

            prepared = self.denoising_net.prepare_inputs(
                z_in,
                z_prev_in,
                t_batch,
                self_audio_raw=self_audio_in,
                other_audio_raw=other_audio_in,
                z_partner=z_partner_in,
                partner_gaze_feat=partner_gaze_in,
            )
            hidden_full, _ = self._run_backbone_with_behavior(
                prepared['x'],
                prepared['c_per_token'],
                prepared['rope_freqs'],
                self_audio_kv=prepared['self_audio_kv'],
                audio_rope_freqs_q=prepared['audio_rope_freqs_q'],
                audio_rope_freqs_k=prepared['audio_rope_freqs_k'],
                partner_full=prepared['partner_full'],
                partner_seq=prepared.get('partner_seq'),
                partner_rope_freqs_q=prepared.get('partner_rope_freqs_q'),
                partner_rope_freqs_k=prepared.get('partner_rope_freqs_k'),
                n_prev_tokens=prepared['n_prev'],
                behavior_control=behavior_control_in,
                behavior_mask=behavior_mask_in,
            )
            z_0_pred = self.denoising_net.finalize_hidden(
                hidden_full,
                prepared['c_per_token'],
            )[:, self.n_prev_latents:]

            if n > 1:
                z0_u = z_0_pred[:batch_size]
                z0_c = z_0_pred[batch_size:]
                # Combined guidance scale: when partner-CFG is active, use
                # partner_cfg_scale; otherwise fall back to cfg_scale (audio).
                # If both are active simultaneously, use the larger scale —
                # the null branch dropped both streams so the delta combines
                # their joint contribution.
                g_scale = max(cfg_scale if use_audio_cfg else 1.0,
                              partner_cfg_scale if use_partner_cfg else 1.0,
                              behavior_cfg_scale if use_behavior_cfg else 1.0)
                z_0_g = z0_u + g_scale * (z0_c - z0_u)
            else:
                z_0_g = z_0_pred

            # Euler step: z_{t-dt} = z_t - dt * (z_t - z_0) / t
            z_t = z_t - dt * (z_t - z_0_g) / t_val

        return z_t, z_at_T

    @torch.no_grad()
    def sample_token(
        self,
        token_idx: int,
        z_prev: Optional[torch.Tensor] = None,
        self_audio_kv: Optional[torch.Tensor] = None,
        partner_ctx: Optional[torch.Tensor] = None,
        layer_kv_caches: Optional[List] = None,
        partner_kv_caches: Optional[List] = None,
        z_partner_token: Optional[torch.Tensor] = None,
        other_audio_token: Optional[torch.Tensor] = None,
        partner_gaze_token: Optional[torch.Tensor] = None,
        behavior_control_token: Optional[torch.Tensor] = None,
        behavior_mask_token: Optional[torch.Tensor] = None,
        sample_steps: int = 4,
        batch_size: int = 1,
        cfg_scale: float = 1.0,
        null_self_audio_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List, List]:
        """Streaming: generate one latent token at a time with KV cache.

        Args:
            token_idx: which token position to generate (0 to n_latents-1)
            z_prev: (B, n_prev_latents, latent_dim) — only used at token_idx=0 to
                    prepopulate KV caches with prev latent context
            self_audio_kv: (B, n_latents, hidden_dim) — precomputed audio features
            partner_ctx: (B, n_latents, hidden_dim) — precomputed partner features
            layer_kv_caches: cached KV states per layer
            partner_kv_caches: cached KV states for partner encoder
            z_partner_token: (B, 1, latent_dim) raw partner latent token for causal encoding
            other_audio_token: raw partner audio token/window for causal encoding
            sample_steps: ODE steps for this token

        Returns:
            z_token: (B, 1, latent_dim) generated latent token
            new_layer_kv_caches
            new_partner_kv_caches
        """
        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)
        else:
            batch_size = z_prev.shape[0]

        net = self.denoising_net
        total_seq_len = self.n_prev_latents + self.n_latents
        rope_freqs = net.rope(total_seq_len, device=self.device)
        past_layer_kv_caches = layer_kv_caches

        # If first token, push prev latent tokens into KV caches
        if token_idx == 0 and past_layer_kv_caches is None:
            past_layer_kv_caches = self._prime_prev_caches(z_prev, batch_size)
        elif token_idx > 0 and past_layer_kv_caches is None:
            raise ValueError(
                "sample_token requires layer_kv_caches for token_idx > 0. "
                "Call token_idx=0 first or provide primed caches."
            )

        # Partner context for this token.
        new_partner_kv_caches = partner_kv_caches
        partner_token = None
        if partner_ctx is not None:
            partner_token = partner_ctx[:, token_idx:token_idx + 1]
        elif z_partner_token is not None or other_audio_token is not None or partner_gaze_token is not None:
            partner_token, new_partner_kv_caches = self.encode_partner_token(
                z_partner_token=z_partner_token,
                other_audio_token=other_audio_token,
                partner_gaze_token=partner_gaze_token,
                partner_kv_caches=partner_kv_caches,
                batch_size=batch_size,
            )

        # Generate this token via ODE
        z_t = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        dt = 1.0 / sample_steps
        timesteps = torch.linspace(1.0, dt, sample_steps, device=self.device)

        # Pre-compute audio/partner RoPE (reused every ODE step)
        audio_kv_token = None
        audio_rope_q = None
        audio_rope_k = None
        if self_audio_kv is not None:
            audio_kv_token = self_audio_kv[:, token_idx:token_idx + 1]
            pos = self.n_prev_latents + token_idx
            audio_rope_q = rope_freqs[pos:pos + 1]
            audio_rope_k = net.audio_rope(self.n_latents, device=self.device)

        partner_rope_q_token = None
        partner_rope_k_token = None
        if partner_token is not None and net.use_partner_cross_attn:
            pos = self.n_prev_latents + token_idx
            partner_rope_q_token = rope_freqs[pos:pos + 1]
            partner_rope_k_token = rope_freqs[self.n_prev_latents:self.n_prev_latents + self.n_latents]

        # Null audio for CFG unconditional branch
        use_cfg = cfg_scale != 1.0 and audio_kv_token is not None
        null_audio_token = None
        if use_cfg:
            if null_self_audio_kv is not None:
                null_audio_token = null_self_audio_kv[:, token_idx:token_idx + 1]
            elif hasattr(net, 'null_self_audio_kv') and net.null_self_audio_kv is not None:
                null_audio_token = net.null_self_audio_kv.expand(batch_size, 1, -1)
            else:
                null_audio_token = torch.zeros_like(audio_kv_token)
            # Null behavior for unconditional branch
            null_behavior = torch.zeros_like(behavior_control_token) if behavior_control_token is not None else None
            null_behavior_mask = torch.zeros_like(behavior_mask_token) if behavior_mask_token is not None else None

        for t_val in timesteps:
            z_proj = net.latent_proj(z_t)
            t_scalar = torch.full((batch_size,), t_val.item(), device=self.device)
            t_emb = net.time_embed(t_scalar).unsqueeze(1)
            z_proj = z_proj + net.token_time_proj(t_emb)

            # Conditional forward (real audio, real behavior)
            x_c, _, _ = self._run_single_token_with_behavior(
                z_proj, t_emb, rope_freqs,
                self_audio_kv=audio_kv_token,
                audio_rope_freqs_q=audio_rope_q,
                audio_rope_freqs_k=audio_rope_k,
                partner_token=partner_token,
                partner_rope_freqs_q=partner_rope_q_token,
                partner_rope_freqs_k=partner_rope_k_token,
                layer_kv_caches=past_layer_kv_caches,
                behavior_control_token=behavior_control_token,
                behavior_mask_token=behavior_mask_token,
            )
            z_0_c = net.finalize_hidden(x_c, t_emb)

            if use_cfg:
                # Unconditional forward (null audio, null behavior)
                x_u, _, _ = self._run_single_token_with_behavior(
                    z_proj, t_emb, rope_freqs,
                    self_audio_kv=null_audio_token,
                    audio_rope_freqs_q=audio_rope_q,
                    audio_rope_freqs_k=audio_rope_k,
                    partner_token=partner_token,
                    partner_rope_freqs_q=partner_rope_q_token,
                    partner_rope_freqs_k=partner_rope_k_token,
                    layer_kv_caches=past_layer_kv_caches,
                    behavior_control_token=null_behavior,
                    behavior_mask_token=null_behavior_mask,
                )
                z_0_u = net.finalize_hidden(x_u, t_emb)
                z_0_pred = z_0_u + cfg_scale * (z_0_c - z_0_u)
            else:
                z_0_pred = z_0_c

            # Euler step
            z_t = z_t - dt * (z_t - z_0_pred) / t_val

        # After final ODE step, push this token's clean representation into KV caches
        # exactly once. The ODE steps above always condition on the fixed past cache.
        z_final_proj = net.latent_proj(z_t)
        t_zero = torch.zeros(batch_size, device=self.device)
        t_emb_zero = net.time_embed(t_zero).unsqueeze(1)
        t_additive_zero = net.token_time_proj(t_emb_zero)
        z_final_proj = z_final_proj + t_additive_zero

        audio_kv_final = None
        audio_rope_q_final = None
        audio_rope_k_final = None
        if self_audio_kv is not None:
            audio_kv_final = self_audio_kv[:, token_idx:token_idx + 1]
            pos = self.n_prev_latents + token_idx
            audio_rope_q_final = rope_freqs[pos:pos + 1]
            audio_rope_k_final = net.audio_rope(self.n_latents, device=self.device)

        partner_rope_q_final = None
        partner_rope_k_final = None
        if partner_token is not None and net.use_partner_cross_attn:
            pos = self.n_prev_latents + token_idx
            partner_rope_q_final = rope_freqs[pos:pos + 1]
            partner_rope_k_final = rope_freqs[self.n_prev_latents:self.n_prev_latents + self.n_latents]

        _, _, final_layer_kv_caches = self._run_single_token_with_behavior(
            z_final_proj, t_emb_zero, rope_freqs,
            self_audio_kv=audio_kv_final,
            audio_rope_freqs_q=audio_rope_q_final,
            audio_rope_freqs_k=audio_rope_k_final,
            partner_token=partner_token,
            partner_rope_freqs_q=partner_rope_q_final,
            partner_rope_freqs_k=partner_rope_k_final,
            layer_kv_caches=past_layer_kv_caches,
            behavior_control_token=behavior_control_token,
            behavior_mask_token=behavior_mask_token,
        )

        return z_t, final_layer_kv_caches, new_partner_kv_caches

    @torch.no_grad()
    def sample_token_sliding(
        self,
        total_tokens: int,
        z_prev: Optional[torch.Tensor] = None,
        self_audio_kv_full: Optional[torch.Tensor] = None,
        partner_ctx_full: Optional[torch.Tensor] = None,
        behavior_control_full: Optional[torch.Tensor] = None,
        behavior_mask_full: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
        sample_steps: int = 10,
        batch_size: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Token-by-token streaming with bounded sliding KV caches.

        Uses block.forward() for all cache operations (priming, ODE, commit)
        so per-layer hidden states are correct. Caches are bounded to the
        training horizon: self=27 past, audio=24 past, partner=24 past.
        Explicit global position IDs via q_pos/k_pos.
        """
        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)
        else:
            batch_size = z_prev.shape[0]

        net = self.denoising_net
        device = self.device
        n_prev = self.n_prev_latents
        n_lat = self.n_latents
        # Bounds: past cache entries (EXCLUDING current token which gets appended by attn)
        max_self_past = n_prev + n_lat - 1   # 27
        max_cross_past = n_lat - 1            # 24

        # Extend RoPE tables
        max_pos = n_prev + total_tokens + 16
        rope_freqs = net.rope(max_pos, device=device)
        audio_rope_table = net.audio_rope(max_pos, device=device)

        # CFG setup
        has_audio = self_audio_kv_full is not None
        use_cfg = has_audio and cfg_scale != 1.0
        null_audio_single = None
        if use_cfg:
            if hasattr(net, 'null_self_audio_kv') and net.null_self_audio_kv is not None:
                null_audio_single = net.null_self_audio_kv.expand(batch_size, 1, -1)
            else:
                null_audio_single = torch.zeros(batch_size, 1, net.hidden_dim, device=device)

        # Noise
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        noise_all = torch.randn(batch_size, total_tokens, self.latent_dim, generator=g).to(device)

        # Per-layer caches: list of (self_kv, audio_cross_kv, partner_cross_kv)
        n_layers = len(net.blocks)
        layer_caches = [None] * n_layers

        def _evict_caches(caches):
            """Trim caches to bounded sizes."""
            trimmed = []
            for lc in caches:
                if lc is None:
                    trimmed.append(None)
                    continue
                s_kv, a_kv, p_kv = lc
                if s_kv is not None and s_kv[0].shape[2] > max_self_past:
                    trim = s_kv[0].shape[2] - max_self_past
                    s_kv = (s_kv[0][:, :, trim:], s_kv[1][:, :, trim:])
                if a_kv is not None and a_kv[0].shape[2] > max_cross_past:
                    trim = a_kv[0].shape[2] - max_cross_past
                    a_kv = (a_kv[0][:, :, trim:], a_kv[1][:, :, trim:])
                if p_kv is not None and p_kv[0].shape[2] > max_cross_past:
                    trim = p_kv[0].shape[2] - max_cross_past
                    p_kv = (p_kv[0][:, :, trim:], p_kv[1][:, :, trim:])
                trimmed.append((s_kv, a_kv, p_kv))
            return trimmed

        def _run_through_blocks(h, c_token, pos_q, pos_k,
                                audio_kv=None, audio_k_pos=None,
                                partner_token=None, partner_k_pos=None,
                                caches=None, update_cache=True):
            """Run token through all blocks using block.forward().

            Returns (output_h, new_caches). If update_cache=False, caches
            are read but not modified (for ODE steps).
            """
            new_caches_out = []
            q_global = pos_q.item() if pos_q is not None else 0
            for layer_idx, block in enumerate(net.blocks):
                prior = caches[layer_idx] if caches else None
                s_kv = prior[0] if prior else None
                a_kv = prior[1] if prior else None
                p_kv = prior[2] if prior else None

                # Audio RoPE
                audio_rope_q = rope_freqs[q_global:q_global + 1] if audio_kv is not None else None
                audio_rope_k = audio_rope_table if audio_kv is not None else None

                # Partner RoPE
                p_rope_q = rope_freqs[q_global:q_global + 1] if (
                    partner_token is not None and block.use_partner_cross_attn) else None
                p_rope_k = rope_freqs if (
                    partner_token is not None and block.use_partner_cross_attn) else None

                h, new_s_kv, new_a_kv, new_p_kv = block(
                    h, c_token, rope_freqs,
                    self_audio_kv=audio_kv,
                    audio_rope_freqs_q=audio_rope_q,
                    audio_rope_freqs_k=audio_rope_k,
                    partner_ctx=partner_token,
                    partner_kv=partner_token if block.use_partner_cross_attn else None,
                    partner_rope_freqs_q=p_rope_q,
                    partner_rope_freqs_k=p_rope_k,
                    self_attn_kv_cache=s_kv,
                    cross_attn_kv_cache=a_kv,
                    partner_cross_kv_cache=p_kv,
                    self_attn_q_pos=pos_q,
                    self_attn_k_pos=pos_k,
                    audio_cross_k_pos=audio_k_pos,
                    partner_cross_k_pos=partner_k_pos,
                )
                if update_cache:
                    new_caches_out.append((new_s_kv, new_a_kv, new_p_kv))
                else:
                    # Keep old caches (ODE steps don't commit)
                    new_caches_out.append(prior)
            return h, new_caches_out

        # ── Prime caches with prev latent tokens (positions 0..n_prev-1) ──
        # Training gives prefix tokens ZERO additive (not token_time_proj(0)).
        z_prev_proj = net.latent_proj(z_prev)
        # c for prev tokens: time_embed(0) but NO additive
        t_zero_emb = net.time_embed(torch.zeros(batch_size, device=device)).unsqueeze(1)
        # Process prev tokens one at a time through full block stack
        for p in range(n_prev):
            token_p = z_prev_proj[:, p:p + 1]  # (B, 1, D) — NO additive
            pos = torch.tensor([p], device=device)
            # c_token = t_zero_emb (same for all prev tokens)
            _, layer_caches = _run_through_blocks(
                token_p, t_zero_emb, pos_q=pos, pos_k=pos,
                audio_kv=None, partner_token=None,
                caches=layer_caches, update_cache=True,
            )

        # ── Generate target tokens ──
        z_generated = []
        dt = 1.0 / sample_steps
        timesteps = torch.linspace(1.0, dt, sample_steps, device=device)

        for g_t in range(total_tokens):
            q_global = n_prev + g_t
            z_t = noise_all[:, g_t:g_t + 1]
            pos_q = torch.tensor([q_global], device=device)

            # Per-token conditioning inputs
            audio_kv_t = self_audio_kv_full[:, g_t:g_t + 1] if has_audio else None
            audio_k_pos = torch.tensor([g_t], device=device) if has_audio else None
            partner_t = partner_ctx_full[:, g_t:g_t + 1] if partner_ctx_full is not None else None
            partner_k_pos = torch.tensor([q_global], device=device) if (
                partner_ctx_full is not None) else None
            beh_ctrl = behavior_control_full[:, g_t:g_t + 1] if behavior_control_full is not None else None
            beh_mask = behavior_mask_full[:, g_t:g_t + 1] if behavior_mask_full is not None else None

            # ODE loop — reads caches but does NOT update them
            for t_val in timesteps:
                z_proj = net.latent_proj(z_t)
                t_scalar = torch.full((batch_size,), t_val.item(), device=device)
                t_emb = net.time_embed(t_scalar).unsqueeze(1)
                z_proj = z_proj + net.token_time_proj(t_emb)

                # Conditional forward
                z_0_c = net.finalize_hidden(
                    _run_through_blocks(
                        z_proj, t_emb, pos_q, pos_q,
                        audio_kv=audio_kv_t, audio_k_pos=audio_k_pos,
                        partner_token=partner_t, partner_k_pos=partner_k_pos,
                        caches=layer_caches, update_cache=False,
                    )[0], t_emb)

                if use_cfg:
                    # Unconditional: null audio, null behavior
                    z_0_u = net.finalize_hidden(
                        _run_through_blocks(
                            z_proj, t_emb, pos_q, pos_q,
                            audio_kv=null_audio_single, audio_k_pos=audio_k_pos,
                            partner_token=partner_t, partner_k_pos=partner_k_pos,
                            caches=layer_caches, update_cache=False,
                        )[0], t_emb)
                    z_0_pred = z_0_u + cfg_scale * (z_0_c - z_0_u)
                else:
                    z_0_pred = z_0_c

                z_t = z_t - dt * (z_t - z_0_pred) / t_val

            z_generated.append(z_t)

            # ── Commit clean token: push through blocks to update caches ──
            z_clean_proj = net.latent_proj(z_t)
            # Clean token gets t=0 conditioning with NO additive
            # (matching how committed tokens look in training's prev context)
            _, layer_caches = _run_through_blocks(
                z_clean_proj, t_zero_emb, pos_q, pos_q,
                audio_kv=audio_kv_t, audio_k_pos=audio_k_pos,
                partner_token=partner_t, partner_k_pos=partner_k_pos,
                caches=layer_caches, update_cache=True,
            )
            # Evict oldest entries to stay within training horizon
            layer_caches = _evict_caches(layer_caches)

        return torch.cat(z_generated, dim=1)

    def sample_token_sliding_OLD(
        self,
        total_tokens: int,
        z_prev: Optional[torch.Tensor] = None,
        self_audio_kv_full: Optional[torch.Tensor] = None,
        partner_ctx_full: Optional[torch.Tensor] = None,
        behavior_control_full: Optional[torch.Tensor] = None,
        behavior_mask_full: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
        sample_steps: int = 10,
        batch_size: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Token-by-token streaming with bounded sliding KV caches.

        Generates tokens one at a time. Self-attention cache is bounded to
        n_prev_latents + n_latents (training horizon). Audio and partner
        cross-attention caches are bounded to n_latents. All caches use
        explicit global position IDs for correct RoPE.

        Args:
            total_tokens: number of target tokens to generate
            z_prev: (B, n_prev, latent_dim) or None (uses start_latent)
            self_audio_kv_full: (B, total_tokens, hidden_dim) pre-encoded audio
            partner_ctx_full: (B, total_tokens, hidden_dim) pre-encoded partner
            behavior_control_full: (B, total_tokens, behavior_dim) or None
            behavior_mask_full: (B, total_tokens, behavior_dim) or None
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)
            sample_steps: ODE steps per token
            seed: random seed for noise

        Returns:
            z_generated: (B, total_tokens, latent_dim)
        """
        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)
        else:
            batch_size = z_prev.shape[0]

        net = self.denoising_net
        device = self.device
        n_prev = self.n_prev_latents
        n_latents = self.n_latents  # training window size (25)
        max_self_cache = n_prev + n_latents  # 28 — training horizon

        # Extend RoPE tables for long sequences
        max_pos = n_prev + total_tokens + 16
        rope_freqs = net.rope(max_pos, device=device)
        audio_rope_table = net.audio_rope(max_pos, device=device)
        partner_rope_table = rope_freqs  # partner shares motion clock

        # Null audio for CFG
        has_audio = self_audio_kv_full is not None
        use_cfg = has_audio and cfg_scale != 1.0
        if use_cfg:
            if hasattr(net, 'null_self_audio_kv') and net.null_self_audio_kv is not None:
                null_audio_single = net.null_self_audio_kv.expand(batch_size, 1, -1)
            else:
                null_audio_single = torch.zeros(batch_size, 1, net.hidden_dim, device=device)

        # Per-layer sliding caches: list of (self_kv, audio_cross_kv, partner_cross_kv)
        # self_kv: (K, V) each (B, H, L_cached, D_h) — bounded to max_self_cache
        # audio_cross_kv: (K, V) — bounded to n_latents
        # partner_cross_kv: (K, V) — bounded to n_latents
        # Plus: position tracking for each cache
        n_layers = len(net.blocks)
        layer_caches = [None] * n_layers  # initialized on first token

        # Track which global positions are in each self-attn cache
        self_cache_positions = []  # list of global positions in self-attn cache

        # Noise
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        noise_all = torch.randn(batch_size, total_tokens, self.latent_dim, generator=g).to(device)

        # Prime KV caches with prev latent tokens (positions 0..n_prev-1)
        z_prev_proj = net.latent_proj(z_prev)
        t_zero = torch.zeros(batch_size, device=device)
        t_emb_zero = net.time_embed(t_zero).unsqueeze(1)
        t_additive_zero = net.token_time_proj(t_emb_zero)
        # Process each prev token to build initial cache
        for p in range(n_prev):
            token_proj = z_prev_proj[:, p:p + 1] + t_additive_zero
            q_pos = torch.tensor([p], device=device)
            new_layer_caches = []
            for layer_idx, block in enumerate(net.blocks):
                prior = layer_caches[layer_idx]
                self_kv = prior[0] if prior else None
                cross_kv = prior[1] if prior else None
                pcross_kv = prior[2] if prior else None

                # Self-attention with explicit position
                x_norm = block.norm1(token_proj)  # skip AdaLN for prev (t=0, gates≈0)
                _, new_self_kv = block.attn(x_norm, rope_freqs, kv_cache=self_kv, q_pos=q_pos, k_pos=q_pos)

                # No audio/partner cross-attn for prev tokens
                new_layer_caches.append((new_self_kv, cross_kv, pcross_kv))
                # token_proj passes through unchanged (prev tokens are context only)
            layer_caches = new_layer_caches
            self_cache_positions.append(p)

        # Generate target tokens one at a time
        z_generated = []
        dt = 1.0 / sample_steps
        timesteps = torch.linspace(1.0, dt, sample_steps, device=device)

        for g_t in range(total_tokens):
            q_global_pos = n_prev + g_t  # global position of this target token
            z_t = noise_all[:, g_t:g_t + 1]  # (B, 1, latent_dim)

            # Audio token for this position
            audio_kv_token = self_audio_kv_full[:, g_t:g_t + 1] if has_audio else None
            audio_k_global_pos = torch.tensor([g_t], device=device)  # audio position
            q_pos_t = torch.tensor([q_global_pos], device=device)

            # Partner token
            partner_token = partner_ctx_full[:, g_t:g_t + 1] if partner_ctx_full is not None else None
            partner_k_pos = torch.tensor([q_global_pos], device=device)  # shared clock

            # Behavior
            beh_ctrl = behavior_control_full[:, g_t:g_t + 1] if behavior_control_full is not None else None
            beh_mask = behavior_mask_full[:, g_t:g_t + 1] if behavior_mask_full is not None else None

            # ODE loop — does NOT update caches
            for t_val in timesteps:
                z_proj = net.latent_proj(z_t)
                t_scalar = torch.full((batch_size,), t_val.item(), device=device)
                t_emb = net.time_embed(t_scalar).unsqueeze(1)
                z_proj = z_proj + net.token_time_proj(t_emb)

                def _forward_one(audio_kv, beh_c, beh_m):
                    """Single forward through all layers using cached KV."""
                    h = z_proj
                    for layer_idx, block in enumerate(net.blocks):
                        prior = layer_caches[layer_idx]
                        self_kv = prior[0] if prior else None
                        cross_kv = prior[1] if prior else None
                        pcross_kv = prior[2] if prior else None

                        # AdaLN
                        c_mod = block.adaLN_modulation(t_emb)
                        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                            c_mod.chunk(6, dim=-1)

                        # Self-attention with explicit position
                        x_norm = block.norm1(h) * (1 + scale_msa) + shift_msa
                        attn_out, _ = block.attn(
                            x_norm, rope_freqs, kv_cache=self_kv,
                            q_pos=q_pos_t, k_pos=q_pos_t)
                        h = h + gate_msa * attn_out

                        # Audio cross-attention
                        if audio_kv is not None:
                            x_norm = block.self_audio_norm(h)
                            sa_out, _ = block.self_audio_cross_attn(
                                x_norm, audio_kv,
                                rope_freqs_q=rope_freqs[q_global_pos:q_global_pos + 1],
                                rope_freqs_k=audio_rope_table,
                                kv_cache=cross_kv,
                                k_pos=audio_k_global_pos,
                            )
                            h = h + sa_out

                        # Partner
                        if partner_token is not None:
                            if block.use_partner_cross_attn:
                                q_pn = block.partner_norm(h)
                                p_out, _ = block.partner_cross_attn(
                                    q_pn, partner_token,
                                    rope_freqs_q=partner_rope_table[q_global_pos:q_global_pos + 1],
                                    rope_freqs_k=partner_rope_table,
                                    kv_cache=pcross_kv,
                                    k_pos=partner_k_pos,
                                )
                                h = h + p_out
                            else:
                                x_norm = block.partner_norm(h)
                                h = h + block.partner_fuse(torch.cat([x_norm, partner_token], dim=-1))

                        # MLP
                        x_norm = block.norm2(h) * (1 + scale_mlp) + shift_mlp
                        h = h + gate_mlp * block.mlp(x_norm)

                    return net.finalize_hidden(h, t_emb)

                z_0_c = _forward_one(audio_kv_token, beh_ctrl, beh_mask)

                if use_cfg:
                    null_beh = torch.zeros_like(beh_ctrl) if beh_ctrl is not None else None
                    null_mask = torch.zeros_like(beh_mask) if beh_mask is not None else None
                    z_0_u = _forward_one(null_audio_single, null_beh, null_mask)
                    z_0_pred = z_0_u + cfg_scale * (z_0_c - z_0_u)
                else:
                    z_0_pred = z_0_c

                z_t = z_t - dt * (z_t - z_0_pred) / t_val

            z_generated.append(z_t)

            # Push clean token into caches (once per token, not per ODE step)
            z_clean_proj = net.latent_proj(z_t)
            z_clean_proj = z_clean_proj + net.token_time_proj(
                net.time_embed(torch.zeros(batch_size, device=device)).unsqueeze(1))

            new_layer_caches = []
            for layer_idx, block in enumerate(net.blocks):
                prior = layer_caches[layer_idx]
                self_kv = prior[0] if prior else None
                cross_kv = prior[1] if prior else None
                pcross_kv = prior[2] if prior else None

                # Push into self-attn cache
                x_norm = block.norm1(z_clean_proj)  # simplified — t=0
                _, new_self_kv = block.attn(
                    x_norm, rope_freqs, kv_cache=self_kv,
                    q_pos=q_pos_t, k_pos=q_pos_t)

                # Push into audio cross-attn cache
                new_cross_kv = cross_kv
                if audio_kv_token is not None:
                    _, new_cross_kv = block.self_audio_cross_attn(
                        block.self_audio_norm(z_clean_proj),
                        audio_kv_token,
                        rope_freqs_q=rope_freqs[q_global_pos:q_global_pos + 1],
                        rope_freqs_k=audio_rope_table,
                        kv_cache=cross_kv,
                        k_pos=audio_k_global_pos,
                    )

                # Push into partner cross-attn cache
                new_pcross_kv = pcross_kv
                if partner_token is not None and block.use_partner_cross_attn:
                    _, new_pcross_kv = block.partner_cross_attn(
                        block.partner_norm(z_clean_proj),
                        partner_token,
                        rope_freqs_q=partner_rope_table[q_global_pos:q_global_pos + 1],
                        rope_freqs_k=partner_rope_table,
                        kv_cache=pcross_kv,
                        k_pos=partner_k_pos,
                    )

                # Evict oldest entries if cache exceeds training horizon
                if new_self_kv is not None and new_self_kv[0].shape[2] > max_self_cache:
                    trim = new_self_kv[0].shape[2] - max_self_cache
                    new_self_kv = (new_self_kv[0][:, :, trim:], new_self_kv[1][:, :, trim:])
                if new_cross_kv is not None and new_cross_kv[0].shape[2] > n_latents:
                    trim = new_cross_kv[0].shape[2] - n_latents
                    new_cross_kv = (new_cross_kv[0][:, :, trim:], new_cross_kv[1][:, :, trim:])
                if new_pcross_kv is not None and new_pcross_kv[0].shape[2] > n_latents:
                    trim = new_pcross_kv[0].shape[2] - n_latents
                    new_pcross_kv = (new_pcross_kv[0][:, :, trim:], new_pcross_kv[1][:, :, trim:])

                new_layer_caches.append((new_self_kv, new_cross_kv, new_pcross_kv))

            layer_caches = new_layer_caches
            self_cache_positions.append(q_global_pos)
            if len(self_cache_positions) > max_self_cache:
                self_cache_positions = self_cache_positions[-max_self_cache:]

        return torch.cat(z_generated, dim=1)

    @torch.no_grad()
    def sample_streaming_rolling(
        self,
        z_prev: Optional[torch.Tensor] = None,
        self_audio_kv_full: Optional[torch.Tensor] = None,
        z_partner_full: Optional[torch.Tensor] = None,
        other_audio_raw: Optional[torch.Tensor] = None,
        partner_gaze_full: Optional[torch.Tensor] = None,
        behavior_control_full: Optional[torch.Tensor] = None,
        behavior_mask_full: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        sample_steps: int = 10,
        window_size: int = 3,
        total_tokens: int = 25,
        noise_override: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> torch.Tensor:
        """Rolling-window streaming sampler for latent tokens.

        Diffusion Forcing style: K uncommitted tokens co-evolve,
        oldest commits when it accumulates sample_steps updates.

        Args:
            z_prev: (B, n_prev, latent_dim) or None
            self_audio_kv_full: (B, total_tokens, D) pre-encoded audio features
            z_partner_full: (B, total_tokens, latent_dim) partner latent tokens
            other_audio_raw: (B, N_samples) partner audio (for partner encoding)
            cfg_scale: guidance scale
            sample_steps: ODE steps per token
            window_size: K active tokens
            total_tokens: how many tokens to generate
            noise_override: (B, total_tokens, latent_dim)
            seed: random seed

        Returns:
            z_generated: (B, total_tokens, latent_dim) generated latent tokens
        """
        batch_size = 1
        if z_prev is not None:
            batch_size = z_prev.shape[0]
        if z_prev is None:
            z_prev = self.start_latent.expand(batch_size, -1, -1)

        K = min(window_size, total_tokens)
        device = self.device

        if noise_override is not None:
            noise_all = noise_override
        else:
            g = torch.Generator(device='cpu')
            g.manual_seed(seed)
            noise_all = torch.randn(batch_size, total_tokens, self.latent_dim,
                                     generator=g).to(device)

        # Pre-encode partner if needed
        partner_seq = None
        if z_partner_full is not None or other_audio_raw is not None or partner_gaze_full is not None:
            partner_seq, _ = self.denoising_net._encode_partner_stream(
                z_partner_full,
                other_audio_raw,
                partner_gaze_full,
                total_tokens,
                batch_size,
            )

        # Track per-token state
        steps_done = [0] * total_tokens
        z_t = noise_all.clone()  # all start as pure noise
        committed = []
        next_token = 0  # next token to add to window
        window_indices = []

        # Fill initial window
        for i in range(min(K, total_tokens)):
            window_indices.append(i)
            next_token = i + 1

        while len(committed) < total_tokens:
            # Check if oldest token is ready to commit
            if window_indices and steps_done[window_indices[0]] >= sample_steps:
                commit_idx = window_indices.pop(0)
                committed.append(commit_idx)
                # Add next token if available
                if next_token < total_tokens:
                    window_indices.append(next_token)
                    next_token += 1
                if not window_indices:
                    break
                continue

            # Build window z_t
            w_indices = window_indices
            n_w = len(w_indices)
            z_window = torch.stack([z_t[:, idx] for idx in w_indices], dim=1)  # (B, K, D)

            # Build committed prev with sliding window.
            # The model was trained with exactly n_prev_latents prev tokens
            # (typically 3). Accumulating ALL committed tokens into prev grows
            # the sequence length without bound and pushes the model into OOD
            # territory for long sequences, which manifests as per-token jitter.
            # Fix: keep only the most recent n_prev_latents committed tokens,
            # matching training's short prev context. Drops the original z_prev
            # start token once enough real tokens have committed.
            n_prev_keep = self.n_prev_latents
            if len(committed) >= n_prev_keep:
                keep = committed[-n_prev_keep:]
                committed_z = torch.stack([z_t[:, idx] for idx in keep], dim=1)
                z_prev_full = committed_z
            elif committed:
                committed_z = torch.stack([z_t[:, idx] for idx in committed], dim=1)
                z_prev_full = torch.cat(
                    [z_prev[:, -(n_prev_keep - len(committed)):], committed_z],
                    dim=1,
                )
            else:
                z_prev_full = z_prev[:, -n_prev_keep:]

            # Per-token timesteps for window
            t_window = torch.tensor(
                [1.0 - steps_done[idx] / sample_steps for idx in w_indices],
                device=device).clamp(min=1e-5).unsqueeze(0).expand(batch_size, -1)

            # Build full input: [prev | window]
            n_prev = z_prev_full.shape[1]
            z_input = torch.cat([z_prev_full, z_window], dim=1)
            x = self.denoising_net.latent_proj(z_input)

            # Partner context for window tokens
            partner_full = None
            partner_window = None
            if partner_seq is not None:
                partner_window = torch.stack(
                    [partner_seq[:, idx] for idx in w_indices], dim=1)
                # Pre-inject partner into target tokens (V6-A/B/C path;
                # V6-D has use_frame_partner_preinject=False and no gate/fuse modules)
                if getattr(self.denoising_net, 'use_frame_partner_preinject', True):
                    target_x = x[:, n_prev:]
                    partner_cat = torch.cat([target_x, partner_window], dim=-1)
                    target_x = target_x + self.denoising_net.frame_partner_gate(partner_cat) * \
                        self.denoising_net.frame_partner_fuse(partner_cat)
                    x = torch.cat([x[:, :n_prev], target_x], dim=1)
                # Full partner for per-block concat+MLP fusion (V6-B path)
                total_len = n_prev + n_w
                partner_full = torch.zeros(
                    batch_size, total_len, self.denoising_net.hidden_dim, device=device)
                partner_full[:, n_prev:] = partner_window

            # Per-token conditioning
            c_per_token, c_additive = self.denoising_net._compute_per_token_conditioning(
                t_window, n_prev)
            x = x + c_additive

            # RoPE — use GLOBAL positions, not local window positions.
            # Build a full rope table, then gather rows for prev + active tokens.
            max_rope_pos = self.n_prev_latents + total_tokens + 16
            rope_table = self.denoising_net.rope(max_rope_pos, device=device)

            # Global positions for prev tokens: most recent n_prev committed
            if len(committed) >= n_prev_keep:
                prev_global_pos = [self.n_prev_latents + c for c in committed[-n_prev_keep:]]
            elif committed:
                # Mix of original prev positions and committed
                n_orig = n_prev_keep - len(committed)
                prev_global_pos = list(range(self.n_prev_latents - n_orig, self.n_prev_latents))
                prev_global_pos += [self.n_prev_latents + c for c in committed]
            else:
                prev_global_pos = list(range(self.n_prev_latents - n_prev_keep, self.n_prev_latents))

            # Global positions for active window tokens
            active_global_pos = [self.n_prev_latents + idx for idx in w_indices]

            # Gather rope rows for the full sequence [prev | active]
            all_global_pos = prev_global_pos + active_global_pos
            rope_freqs = rope_table[torch.tensor(all_global_pos, device=device)]

            # Partner cross-attention RoPE (target-time global positions)
            partner_rope_q_win = None
            partner_rope_k_win = None
            if partner_window is not None and self.denoising_net.use_partner_cross_attn:
                active_pos_t = torch.tensor(active_global_pos, device=device)
                partner_rope_q_win = rope_table[active_pos_t]
                partner_rope_k_win = rope_table[active_pos_t]

            # Audio — window tokens only (matching training's per-token alignment),
            # but with GLOBAL RoPE positions so relative pos is correct.
            audio_rope_q = None
            audio_rope_k = None
            audio_kv_window = None
            if self_audio_kv_full is not None:
                audio_kv_window = torch.stack(
                    [self_audio_kv_full[:, idx] for idx in w_indices], dim=1)
                # Q uses global positions from the full sequence rope
                audio_rope_q = rope_freqs
                # K uses global audio positions for the active window tokens
                audio_rope_table = self.denoising_net.audio_rope(max_rope_pos, device=device)
                audio_k_positions = torch.tensor(w_indices, device=device)
                audio_rope_k = audio_rope_table[audio_k_positions]

            # CFG
            if self_audio_kv_full is not None and cfg_scale != 1.0:
                n_cfg = 2
                if hasattr(self.denoising_net, 'null_self_audio_kv') and self.denoising_net.null_self_audio_kv is not None:
                    null_audio = self.denoising_net.null_self_audio_kv.expand(batch_size, n_w, -1)
                else:
                    null_audio = torch.zeros_like(audio_kv_window)
                audio_in = torch.cat([null_audio, audio_kv_window], dim=0)
                x_in = torch.cat([x, x], dim=0)
                c_in = torch.cat([c_per_token, c_per_token], dim=0)
                rope_in = rope_freqs
                partner_in = torch.cat([partner_full, partner_full], dim=0) if partner_full is not None else None
                partner_seq_in = torch.cat([partner_window, partner_window], dim=0) if partner_window is not None else None
                audio_rope_q_in = audio_rope_q
                audio_rope_k_in = audio_rope_k
                # Behavior: unconditional branch gets null labels (mask=zeros)
                # for proper behavior guidance through CFG.
                if behavior_control_full is not None:
                    ctrl_window_cond = torch.stack([behavior_control_full[:, idx] for idx in w_indices], dim=1)
                    ctrl_window_null = torch.zeros_like(ctrl_window_cond)
                    ctrl_window = torch.cat([ctrl_window_null, ctrl_window_cond], dim=0)
                else:
                    ctrl_window = None
                if behavior_mask_full is not None:
                    mask_window_cond = torch.stack([behavior_mask_full[:, idx] for idx in w_indices], dim=1)
                    mask_window_null = torch.zeros_like(mask_window_cond)
                    mask_window = torch.cat([mask_window_null, mask_window_cond], dim=0)
                else:
                    mask_window = None
            else:
                n_cfg = 1
                audio_in = audio_kv_window
                x_in = x
                c_in = c_per_token
                rope_in = rope_freqs
                partner_in = partner_full
                partner_seq_in = partner_window
                audio_rope_q_in = audio_rope_q
                audio_rope_k_in = audio_rope_k
                ctrl_window = (
                    torch.stack([behavior_control_full[:, idx] for idx in w_indices], dim=1)
                    if behavior_control_full is not None else None
                )
                mask_window = (
                    torch.stack([behavior_mask_full[:, idx] for idx in w_indices], dim=1)
                    if behavior_mask_full is not None else None
                )

            # Forward through blocks
            h, _ = self._run_backbone_with_behavior(
                x_in,
                c_in,
                rope_in,
                self_audio_kv=audio_in,
                audio_rope_freqs_q=audio_rope_q_in,
                audio_rope_freqs_k=audio_rope_k_in,
                partner_full=partner_in,
                partner_seq=partner_seq_in,
                partner_rope_freqs_q=partner_rope_q_win,
                partner_rope_freqs_k=partner_rope_k_win,
                n_prev_tokens=n_prev,
                behavior_control=ctrl_window,
                behavior_mask=mask_window,
            )
            z_pred_all = self.denoising_net.finalize_hidden(h, c_in)

            # Extract window predictions (skip prev)
            if n_cfg == 2:
                z_pred_u = z_pred_all[:batch_size, n_prev:]
                z_pred_c = z_pred_all[batch_size:, n_prev:]
                z_0_window = z_pred_u + cfg_scale * (z_pred_c - z_pred_u)
            else:
                z_0_window = z_pred_all[:, n_prev:]

            # Euler step per token at its own timestep
            dt = 1.0 / sample_steps
            for wi, idx in enumerate(w_indices):
                t_i = 1.0 - steps_done[idx] / sample_steps
                t_i = max(t_i, 1e-5)
                z_t[:, idx] = z_t[:, idx] - dt * (z_t[:, idx] - z_0_window[:, wi]) / t_i
                steps_done[idx] += 1

        # Collect in order
        z_generated = z_t[:, :total_tokens]
        return z_generated
