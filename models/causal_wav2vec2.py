"""Causal Wav2Vec2 encoder with configurable lookahead.

Inspired by DyStream (arxiv 2512.24408): replaces full-context wav2vec2 with
a causal variant for streaming inference. Key changes from standard wav2vec2:
  1. GroupNorm in feature extractor → LayerNorm (per-timestep, no future leakage)
  2. Convolutional positional encoding removed (kernel sees future)
  3. RoPE added to transformer attention Q/K
  4. Causal + lookahead attention mask (configurable, default 60ms)

Non-invasive: same input/output interface as FrozenWav2Vec2 in wav2vec2_encoder.py.
Swap via build_audio_encoder(args) — one-line change in the latent model constructor.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config as _Wav2Vec2Config, Wav2Vec2Model as _Wav2Vec2Model

from models.wav2vec2_encoder import linear_interpolation


# ── RoPE ──────────────────────────────────────────────────────────────────

class RotaryEmbedding1D(nn.Module):
    """1D rotary positional embedding for causal audio attention."""

    def __init__(self, dim, max_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, freqs):
    """Apply RoPE to tensor x using precomputed freqs."""
    # x: (B, H, T, D), freqs: (T, D)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ── Attention mask ────────────────────────────────────────────────────────

def build_lookahead_mask(seq_len, lookahead_steps, device, dtype=torch.float32):
    """Causal attention mask with configurable right context (lookahead).

    Token i can attend to tokens [0, ..., i + lookahead_steps].
    Returns additive bias: 0 for allowed, -inf for blocked.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    for i in range(seq_len):
        end = min(i + lookahead_steps + 1, seq_len)
        mask[i, :end] = 0.0
    return mask  # (T, T)


def ms_to_w2v_steps(lookahead_ms, native_hz=50):
    """Convert milliseconds to wav2vec2 native timesteps (~50Hz)."""
    return max(0, round(lookahead_ms / 1000.0 * native_hz))


# ── Causal attention layer ────────────────────────────────────────────────

class CausalWav2Vec2Attention(nn.Module):
    """Wav2vec2 attention with RoPE and causal+lookahead mask.

    Replaces the standard wav2vec2 self-attention. Loads pretrained Q/K/V/O weights.
    """

    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, rope_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_dim = rope_dim or self.head_dim

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self.rope = RotaryEmbedding1D(self.rope_dim)

    def forward(self, hidden_states, attention_bias=None):
        """
        Args:
            hidden_states: (B, T, D)
            attention_bias: (T, T) additive mask, 0 or -inf
        Returns:
            (B, T, D)
        """
        B, T, D = hidden_states.shape
        H = self.num_heads

        q = self.q_proj(hidden_states).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, H, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        freqs = self.rope(T, hidden_states.device)
        if self.rope_dim < self.head_dim:
            q_rope = q[..., :self.rope_dim]
            q_pass = q[..., self.rope_dim:]
            k_rope = k[..., :self.rope_dim]
            k_pass = k[..., self.rope_dim:]
            q = torch.cat([apply_rotary_emb(q_rope, freqs), q_pass], dim=-1)
            k = torch.cat([apply_rotary_emb(k_rope, freqs), k_pass], dim=-1)
        else:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_bias is not None:
            attn = attn + attention_bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ── Causal encoder layer ─────────────────────────────────────────────────

class CausalWav2Vec2EncoderLayer(nn.Module):
    """Single transformer layer matching HF wav2vec2 post-LN topology.

    HF order: attention → hidden_dropout → residual → layer_norm →
              FFN_intermediate → activation_dropout → FFN_output → hidden_dropout →
              residual → final_layer_norm
    """

    def __init__(self, hidden_size, num_heads, intermediate_size,
                 attention_dropout=0.1, activation_dropout=0.1, hidden_dropout=0.1):
        super().__init__()
        self.attention = CausalWav2Vec2Attention(
            hidden_size, num_heads, attention_dropout=attention_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_intermediate = nn.Linear(hidden_size, intermediate_size)
        self.feed_forward_activation = nn.GELU()
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.feed_forward_output = nn.Linear(intermediate_size, hidden_size)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_bias=None):
        # Attention → hidden_dropout → residual → layer_norm
        attn_out = self.attention(hidden_states, attention_bias=attention_bias)
        attn_out = self.hidden_dropout1(attn_out)
        hidden_states = hidden_states + attn_out
        hidden_states = self.layer_norm(hidden_states)

        # FFN: intermediate → activation → activation_dropout → output → hidden_dropout → residual → final_layer_norm
        ffn_out = self.feed_forward_intermediate(hidden_states)
        ffn_out = self.feed_forward_activation(ffn_out)
        ffn_out = self.activation_dropout(ffn_out)
        ffn_out = self.feed_forward_output(ffn_out)
        ffn_out = self.hidden_dropout2(ffn_out)
        hidden_states = hidden_states + ffn_out
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# ── Feature extractor norm fix ────────────────────────────────────────────

class _ChannelLayerNorm(nn.Module):
    """LayerNorm that operates on channels for (B, C, T) tensors.

    Replaces GroupNorm to avoid time-axis normalization (future leakage).
    Normalizes over C dimension only, per-timestep.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, T) → transpose → norm → transpose back
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _replace_groupnorm_with_layernorm(module):
    """Replace GroupNorm layers with channel-wise LayerNorm in the feature extractor.

    GroupNorm normalizes over the time axis → leaks future info.
    ChannelLayerNorm normalizes per-timestep over channels → causal-safe.
    Transfers affine parameters where shapes match.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            cln = _ChannelLayerNorm(child.num_channels)
            # Transfer affine params (GroupNorm weight/bias shape = (num_channels,) = LayerNorm shape)
            if child.affine and child.weight is not None:
                cln.norm.weight.data.copy_(child.weight.data)
                cln.norm.bias.data.copy_(child.bias.data)
            setattr(module, name, cln)
        else:
            _replace_groupnorm_with_layernorm(child)


# ── Main module ───────────────────────────────────────────────────────────

class CausalWav2Vec2(nn.Module):
    """Causal wav2vec2 with RoPE + lookahead for streaming inference.

    Same interface as FrozenWav2Vec2:
        input:  (B, N_samples) raw audio at 16kHz
        output: (B, frame_num, hidden_size) features at target fps
    """

    def __init__(
        self,
        model_name="facebook/wav2vec2-large-960h",
        cache_dir=None,
        lookahead_ms=60,
        encoder_layers=None,
        freeze_feature_extractor=True,
        freeze_transformer=False,
    ):
        super().__init__()
        self.lookahead_ms = lookahead_ms
        self.lookahead_steps = ms_to_w2v_steps(lookahead_ms)

        # Architecture from the bundled config; all weights come from the STEER checkpoint.
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav2vec2_large_960h_config.json")
        base = _Wav2Vec2Model(_Wav2Vec2Config.from_json_file(config_path))
        config = base.config
        self.hidden_size = config.hidden_size

        # 1. Feature extractor — copy weights, replace GroupNorm with LayerNorm
        self.feature_extractor = base.feature_extractor
        _replace_groupnorm_with_layernorm(self.feature_extractor)
        if freeze_feature_extractor:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # 2. Feature projection — copy directly
        self.feature_projection = base.feature_projection

        # 3. Encoder-level layer_norm + dropout (before layer stack, matches HF)
        self.encoder_layer_norm = nn.LayerNorm(config.hidden_size)
        self.encoder_layer_norm.weight.data.copy_(base.encoder.layer_norm.weight.data)
        self.encoder_layer_norm.bias.data.copy_(base.encoder.layer_norm.bias.data)
        self.encoder_dropout = nn.Dropout(config.hidden_dropout_prob)

        # 4. Build causal encoder layers from pretrained weights
        n_layers = encoder_layers or config.num_hidden_layers
        n_layers = min(n_layers, config.num_hidden_layers)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            hf_layer = base.encoder.layers[i]
            causal_layer = CausalWav2Vec2EncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                attention_dropout=config.attention_dropout,
                activation_dropout=getattr(config, 'activation_dropout', config.hidden_dropout_prob),
                hidden_dropout=config.hidden_dropout_prob,
            )
            # Transfer weights
            self._transfer_layer_weights(hf_layer, causal_layer)
            self.layers.append(causal_layer)

        if freeze_transformer:
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # Post-init: count transferred vs reinitialized
        del base  # free memory

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CausalWav2Vec2] {n_layers} layers, {n_params:,} params "
              f"({n_trainable:,} trainable), lookahead={lookahead_ms}ms ({self.lookahead_steps} steps)")

    def _transfer_layer_weights(self, hf_layer, causal_layer):
        """Copy weights from HF wav2vec2 layer to our causal layer."""
        attn = hf_layer.attention
        ca = causal_layer.attention

        # Q/K/V/O projections
        ca.q_proj.weight.data.copy_(attn.q_proj.weight.data)
        ca.q_proj.bias.data.copy_(attn.q_proj.bias.data)
        ca.k_proj.weight.data.copy_(attn.k_proj.weight.data)
        ca.k_proj.bias.data.copy_(attn.k_proj.bias.data)
        ca.v_proj.weight.data.copy_(attn.v_proj.weight.data)
        ca.v_proj.bias.data.copy_(attn.v_proj.bias.data)
        ca.out_proj.weight.data.copy_(attn.out_proj.weight.data)
        ca.out_proj.bias.data.copy_(attn.out_proj.bias.data)

        # Layer norms
        causal_layer.layer_norm.weight.data.copy_(hf_layer.layer_norm.weight.data)
        causal_layer.layer_norm.bias.data.copy_(hf_layer.layer_norm.bias.data)
        causal_layer.final_layer_norm.weight.data.copy_(hf_layer.final_layer_norm.weight.data)
        causal_layer.final_layer_norm.bias.data.copy_(hf_layer.final_layer_norm.bias.data)

        # FFN
        causal_layer.feed_forward_intermediate.weight.data.copy_(hf_layer.feed_forward.intermediate_dense.weight.data)
        causal_layer.feed_forward_intermediate.bias.data.copy_(hf_layer.feed_forward.intermediate_dense.bias.data)
        causal_layer.feed_forward_output.weight.data.copy_(hf_layer.feed_forward.output_dense.weight.data)
        causal_layer.feed_forward_output.bias.data.copy_(hf_layer.feed_forward.output_dense.bias.data)

    def forward(self, audio_raw, frame_num=None):
        """Extract causal audio features.

        Args:
            audio_raw: (B, N_samples) raw waveform at 16kHz
            frame_num: target number of output frames (e.g., 100 for 4sec@25fps)
        Returns:
            (B, frame_num, hidden_size) audio features
        """
        # CNN feature extraction
        hidden_states = self.feature_extractor(audio_raw)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T_cnn, D_cnn)

        # Feature projection
        hidden_states = self.feature_projection(hidden_states)[0]

        # Encoder-level norm + dropout (matches HF, skipping conv positional encoding)
        hidden_states = self.encoder_layer_norm(hidden_states)
        hidden_states = self.encoder_dropout(hidden_states)

        # Causal transformer with lookahead mask
        T = hidden_states.shape[1]
        attention_bias = build_lookahead_mask(
            T, self.lookahead_steps, hidden_states.device, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias=attention_bias)

        # Interpolate to target frame count (same as FrozenWav2Vec2)
        if frame_num is None:
            frame_num = math.ceil(audio_raw.shape[1] / 16000 * 25)
        hidden_states = linear_interpolation(
            hidden_states, 50, 25, output_len=frame_num)

        return hidden_states  # (B, frame_num, hidden_size)


# ── Factory ───────────────────────────────────────────────────────────────

def build_audio_encoder(args):
    """Build audio encoder from config. Drop-in replacement for FrozenWav2Vec2.

    Usage in model __init__:
        from models.causal_wav2vec2 import build_audio_encoder
        self.wav2vec2 = build_audio_encoder(args)
    """
    use_causal = getattr(args, 'use_causal_audio', False) or False
    model_name = getattr(args, 'wav2vec2_model_name', None) or 'facebook/wav2vec2-base-960h'
    if not use_causal:
        from models.wav2vec2_encoder import FrozenWav2Vec2
        return FrozenWav2Vec2(model_name=model_name)

    return CausalWav2Vec2(
        model_name=model_name,
        lookahead_ms=int(getattr(args, 'audio_lookahead_ms', 60) or 60),
        encoder_layers=getattr(args, 'causal_audio_encoder_layers', None),
        freeze_feature_extractor=bool(getattr(args, 'causal_audio_freeze_feature_extractor', True)),
        freeze_transformer=bool(getattr(args, 'causal_audio_freeze_transformer', False)),
    )
