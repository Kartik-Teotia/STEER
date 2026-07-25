"""Common utilities for Dyadic V1."""

import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NullableArgs:
    """Wrap a dict as an object with attribute access."""
    def __init__(self, d):
        if isinstance(d, dict):
            for k, v in d.items():
                setattr(self, k, v)
        else:
            for k in dir(d):
                if not k.startswith('_'):
                    setattr(self, k, getattr(d, k))

    def __getattr__(self, name):
        return None


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)


def pad_audio(audio, audio_unit=320, pad_threshold=80):
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')
    return audio


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """Resample features from input_fps to output_fps."""
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=False, mode='linear')
    return output_features


def nt_xent_loss(feature_a, feature_b, temperature):
    """Normalized temperature-scaled cross entropy loss."""
    batch_size = feature_a.shape[0]
    device = feature_a.device
    features = torch.cat([feature_a, feature_b], dim=0)
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(labels.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).to(device)
    loss = F.cross_entropy(logits, labels)
    return loss


def load_coef_stats(stats_path):
    """Load normalization stats (mean/std) for motion coefficients."""
    stats = np.load(stats_path)
    return {
        'mean': torch.FloatTensor(stats['mean']),
        'std': torch.FloatTensor(stats['std']),
    }


def normalize_motion(motion, mean, std):
    """Normalize motion coefficients."""
    return (motion - mean) / (std + 1e-8)


def denormalize_motion(motion, mean, std):
    """Denormalize motion coefficients."""
    return motion * (std + 1e-8) + mean


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
