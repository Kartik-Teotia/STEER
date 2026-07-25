"""Causal Temporal Motion Autoencoder for 59D FLAME motion sequences.

Causal variant of temporal_motion_ae.py for fully causal training/inference.
Key changes from bidirectional AE:
  1. Encoder Conv1d: causal padding (left-only, no future frames)
  2. Encoder Transformer: causal attention mask
  3. Decoder Conv1d: causal padding (left-only)
  4. Decoder uses fixed-lag context (configurable right_context tokens)

Same interface as TemporalMotionAE — drop-in replacement via config flag.

Positional encoding in the encoder bottleneck is selectable via
`encoder_pe_type`: 'sinusoidal' (default, added to token embeddings, with
dynamic extension at inference for length-agnosticism) or 'rope' (applied
inside self-attention to Q and K, length-agnostic by construction).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE helpers ──────────────────────────────────────────────────────

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _build_rope_freqs(seq_len: int, head_dim: int, device, dtype=torch.float32):
    """Return (seq_len, head_dim) cos/sin frequency table in the standard RoPE form."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    # Double each frequency to match the (head_dim) shape used by _rotate_half layout.
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope_freqs: torch.Tensor):
    """Apply RoPE to Q and K.

    q, k: (B, H, L, D_h)
    rope_freqs: (L, D_h)
    """
    cos = rope_freqs.cos()[None, None, :, :]
    sin = rope_freqs.sin()[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ── RoPE-based causal self-attention (for the AE encoder bottleneck) ──

class CausalSelfAttentionRoPE(nn.Module):
    """Multi-head causal self-attention with RoPE applied to Q and K.

    Purely functional — no KV cache, no streaming interface. Used inside the
    AE encoder's 2-layer bottleneck where the input is a full sequence and
    the causal mask is built on the fly via SDPA's is_causal=True.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_h)
        q, k, v = qkv.unbind(0)

        rope_freqs = _build_rope_freqs(L, self.head_dim, device=x.device, dtype=x.dtype)
        q, k = _apply_rope(q, k, rope_freqs)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    """POST-norm transformer encoder layer with RoPE causal self-attention + FFN.

    Post-norm (norm_first=False) to match PyTorch's default
    nn.TransformerEncoderLayer topology, which the sinusoidal baseline uses.
    This is important for a clean PE ablation: RoPE vs sinusoidal should be
    the ONLY difference between the two models.
    """

    def __init__(self, dim: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(dim, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: residual → add → norm, for both sublayers
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self.dropout2(self.mlp(x)))
        return x


class RoPEEncoderStack(nn.Module):
    """Stack of RoPEEncoderLayer, matches nn.TransformerEncoder interface."""

    def __init__(self, dim: int, n_heads: int, dim_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(dim, n_heads, dim_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _causal_conv1d(in_channels, out_channels, kernel_size, stride=1, right_pad=0):
    """Create a (near-)causal Conv1d: mostly left-pad, configurable right pad.

    right_pad=0 → strictly causal. right_pad>0 → allows the specified number of
    future timesteps in the conv's input (at the conv's own temporal rate).
    Total pad is kept at (kernel_size - 1) so the output length is unchanged.
    """
    total_pad = kernel_size - 1
    right_pad = max(0, min(int(right_pad), total_pad))
    left_pad = total_pad - right_pad
    return nn.Sequential(
        nn.ConstantPad1d((left_pad, right_pad), 0.0),
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0),
    )


def _build_causal_mask(seq_len, device, dtype=torch.float32):
    """Square causal mask: token i can only attend to tokens [0..i]."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class CausalTemporalMotionEncoder(nn.Module):
    """Causal encoder: (B, T, 59) → (B, T/4, latent_dim).

    Conv1d layers use causal padding (left-only).
    Transformer bottleneck uses causal attention mask.
    """

    def __init__(self, motion_dim=59, hidden_dim=128, latent_dim=256,
                 bottleneck_layers=2, bottleneck_heads=4, bottleneck_ff=512,
                 encoder_pe_type: str = 'sinusoidal'):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        assert encoder_pe_type in ('sinusoidal', 'rope'), f"Unknown encoder_pe_type={encoder_pe_type}"
        self.encoder_pe_type = encoder_pe_type

        # Two causal stride-2 convolutions: T → T/2 → T/4
        self.conv1 = _causal_conv1d(motion_dim, hidden_dim, kernel_size=5, stride=2)
        self.act1 = nn.GELU()
        self.conv2 = _causal_conv1d(hidden_dim, latent_dim, kernel_size=5, stride=2)

        if encoder_pe_type == 'sinusoidal':
            # Sinusoidal positional encoding — generated dynamically at forward
            # time so the encoder is length-agnostic. We keep a cached buffer for
            # the trained default length (128 latent tokens = 512 frames ≈ 17 s)
            # for compatibility with existing checkpoints, but the forward path
            # extends it on the fly when the input sequence is longer.
            self._pe_cached_len = 128
            pe = torch.zeros(self._pe_cached_len, latent_dim)
            position = torch.arange(0, self._pe_cached_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_encoding', pe.unsqueeze(0))

            # Causal transformer bottleneck (PyTorch default layer)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=bottleneck_heads,
                dim_feedforward=bottleneck_ff,
                batch_first=True,
                activation='gelu',
                dropout=0.1,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=bottleneck_layers)
        else:
            # RoPE variant: no additive PE at input; positions are injected
            # inside self-attention via rotary embeddings on Q and K. Length
            # agnostic by construction. Uses a custom encoder stack because
            # nn.TransformerEncoderLayer does not expose Q/K for RoPE injection.
            self.pos_encoding = None
            self.transformer = RoPEEncoderStack(
                dim=latent_dim,
                n_heads=bottleneck_heads,
                dim_ff=bottleneck_ff,
                n_layers=bottleneck_layers,
                dropout=0.1,
            )

    def forward(self, motion):
        """
        Args:
            motion: (B, T, 59) normalized motion
        Returns:
            latents: (B, T/4, latent_dim)
        """
        x = motion.permute(0, 2, 1)  # (B, 59, T)
        x = self.act1(self.conv1(x))  # (B, hidden, T/2)
        x = self.conv2(x)             # (B, latent_dim, T/4)
        x = x.permute(0, 2, 1)        # (B, T/4, latent_dim)

        if self.encoder_pe_type == 'sinusoidal':
            # Dynamic sinusoidal PE: fast path on cached buffer for short
            # sequences, fall through to on-the-fly generation for long ones.
            seq_len = x.shape[1]
            if seq_len <= self.pos_encoding.shape[1]:
                pe_slice = self.pos_encoding[:, :seq_len, :]
            else:
                latent_dim = x.shape[-1]
                position = torch.arange(
                    0, seq_len, device=x.device, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, latent_dim, 2, device=x.device, dtype=torch.float)
                    * (-math.log(10000.0) / latent_dim))
                pe = torch.zeros(seq_len, latent_dim, device=x.device, dtype=torch.float)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe_slice = pe.unsqueeze(0).to(dtype=x.dtype)
            x = x + pe_slice
            causal_mask = _build_causal_mask(x.shape[1], x.device, x.dtype)
            x = self.transformer(x, mask=causal_mask)
        else:
            # RoPE path: no additive PE, no causal mask arg (RoPEEncoderStack
            # builds its own via SDPA is_causal=True per layer).
            x = self.transformer(x)
        return x


class CausalTemporalMotionDecoder(nn.Module):
    """Causal decoder: (B, T/4, latent_dim) → (B, T, 59).

    Uses causal Conv1d. Upsample(nearest) is already causal.
    Fixed-lag option: `right_context_frames` in motion-rate frames (25 Hz).
    - 0   → strictly causal
    - 4   → 1 latent token of right context (160 ms lag)
    - 8   → 2 latent tokens (320 ms)
    - 12  → 3 latent tokens (480 ms), maximum reachable with kernel=5 × 2 conv layers

    Budget is distributed: put right-pad on the outer (motion-rate) conv2 first,
    then on the inner (T/2-rate) conv1 if more is needed. Each T/2-rate position
    contributes 2 motion frames of effective lookahead.
    """

    def __init__(self, motion_dim=59, hidden_dim=128, latent_dim=256, right_context_frames=0):
        super().__init__()
        self.motion_dim = motion_dim
        self.right_context_frames = int(right_context_frames)

        # Distribute the motion-frame right context across the two conv layers.
        # conv2 is at motion rate → each unit of right_pad = 1 motion frame of lookahead.
        # conv1 is at T/2 rate → each unit of right_pad = 2 motion frames of lookahead.
        # Fill conv2 first (up to kernel-1 = 4), then spill to conv1.
        total = self.right_context_frames
        r2 = min(4, total)                       # motion-rate right pad (≤ 4)
        remaining = max(0, total - r2)           # motion frames still to cover
        r1 = min(4, (remaining + 1) // 2)        # T/2-rate right pad (≤ 4; covers 2 motion frames each)
        self._r1 = r1
        self._r2 = r2

        # Stage 1: T/4 → T/2
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = _causal_conv1d(latent_dim, hidden_dim, kernel_size=5, right_pad=r1)
        self.act1 = nn.GELU()

        # Stage 2: T/2 → T
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = _causal_conv1d(hidden_dim, motion_dim, kernel_size=5, right_pad=r2)

    def forward(self, latents):
        """
        Args:
            latents: (B, T/4, latent_dim)
        Returns:
            recon: (B, T, 59)
        """
        x = latents.permute(0, 2, 1)  # (B, latent_dim, T/4)

        x = self.upsample1(x)         # (B, latent_dim, T/2)
        x = self.act1(self.conv1(x))   # (B, hidden, T/2)

        x = self.upsample2(x)         # (B, hidden, T)
        x = self.conv2(x)             # (B, 59, T)

        x = x.permute(0, 2, 1)        # (B, T, 59)
        return x


class CausalTemporalMotionAE(nn.Module):
    """Causal temporal autoencoder for 59D FLAME motion.

    Drop-in replacement for TemporalMotionAE with causal encoder + decoder.
    Same interface: encode(), encode_mean(), decode(), forward().

    Can transfer weights from pretrained bidirectional AE where shapes match
    (conv kernels, mu/logvar projections, transformer FFN). Transformer attention
    weights transfer but will behave differently under causal mask.
    """

    def __init__(self, motion_dim=59, hidden_dim=128, latent_dim=256,
                 bottleneck_layers=2, bottleneck_heads=4, bottleneck_ff=512,
                 use_kl=False, right_context_frames=0,
                 encoder_pe_type: str = 'sinusoidal'):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.stride = 4
        self.use_kl = use_kl
        self.encoder_pe_type = encoder_pe_type

        self.encoder = CausalTemporalMotionEncoder(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            bottleneck_layers=bottleneck_layers,
            bottleneck_heads=bottleneck_heads,
            bottleneck_ff=bottleneck_ff,
            encoder_pe_type=encoder_pe_type,
        )
        self.decoder = CausalTemporalMotionDecoder(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            right_context_frames=right_context_frames,
        )

        if use_kl:
            self.mu_proj = nn.Linear(latent_dim, latent_dim)
            self.logvar_proj = nn.Linear(latent_dim, latent_dim)
            nn.init.zeros_(self.logvar_proj.weight)
            nn.init.constant_(self.logvar_proj.bias, -4.0)

    def encode(self, motion):
        """Encode motion to latent tokens.

        Args:
            motion: (B, T, 59) — T must be divisible by 4
        Returns:
            latents: (B, T/4, latent_dim)
            kl_loss: scalar (0.0 if use_kl=False)
        """
        assert motion.shape[1] % self.stride == 0, \
            f"Temporal dim {motion.shape[1]} must be divisible by stride {self.stride}"
        h = self.encoder(motion)

        if not self.use_kl:
            return h, torch.tensor(0.0, device=h.device)

        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return z, kl

    def encode_mean(self, motion):
        """Encode to deterministic mu (for latent targets)."""
        assert motion.shape[1] % self.stride == 0
        h = self.encoder(motion)
        if self.use_kl:
            return self.mu_proj(h)
        return h

    def decode(self, latents):
        """Decode latent tokens to motion.

        Args:
            latents: (B, T/4, latent_dim)
        Returns:
            recon: (B, T, 59)
        """
        return self.decoder(latents)

    def forward(self, motion):
        """Full encode-decode forward pass.

        Args:
            motion: (B, T, 59)
        Returns:
            recon: (B, T, 59)
            latents: (B, T/4, latent_dim)
            kl_loss: scalar
        """
        latents, kl_loss = self.encode(motion)
        recon = self.decode(latents)
        return recon, latents, kl_loss

    @classmethod
    def from_bidirectional(cls, bidir_ae, right_context_frames=0,
                           encoder_pe_type='sinusoidal'):
        """Initialize from a pretrained bidirectional TemporalMotionAE.

        Transfers all weights where shapes match. Conv kernels transfer directly
        (same weights, different padding). Transformer attention weights transfer
        but will behave differently under causal mask.
        """
        # Infer hidden_dim from bidir encoder's first conv output channels
        hidden_dim = bidir_ae.encoder.conv1.out_channels if hasattr(bidir_ae.encoder.conv1, 'out_channels') else 128
        causal_ae = cls(
            motion_dim=bidir_ae.motion_dim,
            latent_dim=bidir_ae.latent_dim,
            hidden_dim=hidden_dim,
            use_kl=bidir_ae.use_kl,
            right_context_frames=right_context_frames,
            encoder_pe_type=encoder_pe_type,
        )

        # Transfer weights with shape matching
        bidir_state = bidir_ae.state_dict()
        causal_state = causal_ae.state_dict()

        matched, skipped = 0, 0
        transfer = {}
        for key, param in bidir_state.items():
            # Map bidirectional key names to causal key names
            # Conv layers: bidir has nn.Conv1d directly, causal has Sequential(Pad, Conv1d)
            # encoder.conv1.weight → encoder.conv1.1.weight (index 1 = Conv1d after Pad)
            # decoder.conv1.weight → decoder.conv1.1.weight
            causal_key = key
            parts = key.split('.')
            is_conv = (
                len(parts) >= 3
                and parts[0] in ('encoder', 'decoder')
                and parts[1] in ('conv1', 'conv2')
                and parts[2] in ('weight', 'bias')
            )
            if is_conv:
                causal_key = f"{parts[0]}.{parts[1]}.1.{'.'.join(parts[2:])}"

            if causal_key in causal_state and param.shape == causal_state[causal_key].shape:
                transfer[causal_key] = param
                matched += 1
            else:
                skipped += 1

        missing, unexpected = causal_ae.load_state_dict(transfer, strict=False)
        print(f"[CausalAE from_bidirectional] matched={matched}, skipped={skipped}, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return causal_ae


def build_motion_ae(args):
    """Factory: build causal or bidirectional AE based on config.

    Usage:
        ae = build_motion_ae(args)
    """
    use_causal = getattr(args, 'use_causal_ae', False) or False
    latent_dim = int(getattr(args, 'latent_dim', 256) or 256)
    hidden_dim = int(getattr(args, 'ae_hidden_dim', 128) or 128)
    use_kl = bool(getattr(args, 'ae_use_kl', True))
    motion_dim = int(getattr(args, 'motion_dim', 59) or 59)

    if use_causal:
        right_ctx = int(getattr(args, 'ae_decoder_right_context_frames', 0) or 0)
        enc_pe_type = getattr(args, 'encoder_pe_type', 'sinusoidal') or 'sinusoidal'
        return CausalTemporalMotionAE(
            motion_dim=motion_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_kl=use_kl,
            right_context_frames=right_ctx,
            encoder_pe_type=enc_pe_type,
        )
    else:
        from models.temporal_motion_ae import TemporalMotionAE
        return TemporalMotionAE(
            motion_dim=motion_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_kl=use_kl,
        )
