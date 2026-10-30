"""Frozen Wav2Vec2 encoder for on-the-fly audio feature extraction.

Based on DualTalk's wav2vec.py. Key differences:
- Fully frozen (no training, no spec augment)
- Shared encoder for both self and partner audio
- Linear interpolation to match 25fps video frame rate
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model as _Wav2Vec2Model


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """Resample temporal features to target frame rate.

    Args:
        features: (B, T_in, D)
        input_fps: source frame rate (wav2vec2 outputs at ~50fps)
        output_fps: target frame rate (25fps for our video)
        output_len: exact output length (overrides fps calculation)
    Returns:
        (B, output_len, D)
    """
    features = features.transpose(1, 2)  # (B, D, T_in)
    if output_len is None:
        seq_len = features.shape[2] / float(input_fps)
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)  # (B, output_len, D)


class FrozenWav2Vec2(nn.Module):
    """Frozen wav2vec2 encoder for audio feature extraction.

    Loads a pretrained wav2vec2 checkpoint, freezes all parameters.
    Forward: raw waveform -> CNN features -> transformer -> linear interp to frame_num.
    Output dim depends on the chosen checkpoint, e.g. 768 for base, 1024 for large.
    """

    def __init__(self, model_name="facebook/wav2vec2-base-960h",
                 cache_dir=None):
        super().__init__()
        self.encoder = _Wav2Vec2Model.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        # Freeze everything and force eval mode permanently
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.hidden_size = self.encoder.config.hidden_size

    def train(self, mode=True):
        """Override: keep encoder in eval mode always (frozen feature extractor)."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, audio_raw, frame_num=None):
        """Extract audio features from raw waveform.

        Args:
            audio_raw: (B, N_samples) raw waveform at 16kHz
            frame_num: target number of output frames (e.g., 100 for 4sec@25fps)
                       If None, auto-computed from audio length at 25fps.
        Returns:
            (B, frame_num, hidden_size) audio features aligned to video frames
        """
        # CNN feature extraction (frozen)
        hidden_states = self.encoder.feature_extractor(audio_raw)
        hidden_states = hidden_states.transpose(1, 2)  # (B, T_cnn, D_cnn)

        # Feature projection at native resolution (~50fps)
        hidden_states = self.encoder.feature_projection(hidden_states)[0]

        # Transformer encoder at native resolution (frozen)
        encoder_outputs = self.encoder.encoder(
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, T_native, hidden_size)

        # Interpolate AFTER transformer to target frame count
        # This preserves wav2vec2's positional encoding quality
        if frame_num is None:
            frame_num = math.ceil(audio_raw.shape[1] / 16000 * 25)
        hidden_states = linear_interpolation(
            hidden_states, 50, 25, output_len=frame_num)

        return hidden_states  # (B, frame_num, hidden_size)
