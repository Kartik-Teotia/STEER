"""Style Encoder VAE for Dyadic V4.

Replaces V3's contrastive ListenerStyleEncoder with a VAE that is
jointly trained with the denoiser. Outputs a latent z that drives
K=4 style basis vectors via MSMD-style additive basis.

Three input variants:
  V4a: 56D raw FLAME motion (like MSMD)
  V4b: 7D decomposed signals (keep V3's interpretable features)
  V4c: 63D concat (56D motion + 7D decomp) — maximum information
"""

import torch
import torch.nn as nn

from utils.common import PositionalEncoding


class StyleEncoderVAE(nn.Module):
    """VAE-based style encoder for jointly-trained style learning.

    Input: (N, L, input_dim) — motion, decomp, or both
    Output: z (N, d_style), mu (N, d_style), logvar (N, d_style)
    """

    def __init__(self, input_dim=56, d_style=256, n_heads=8, n_layers=1,
                 hidden_dim=512):
        super().__init__()

        self.input_dim = input_dim
        self.d_style = d_style
        self.hidden_dim = hidden_dim

        # Input projection: Conv1D stack for local pattern extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ELU()

        # Positional encoding
        self.pe = PositionalEncoding(hidden_dim, dropout=0.1, max_len=600)

        # Transformer encoder for temporal aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim,
            activation='gelu', batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # VAE output heads
        self.fc_mu = nn.Linear(hidden_dim, d_style)
        self.fc_logvar = nn.Linear(hidden_dim, d_style)

    def encode(self, x):
        """Encode input to mu and logvar.

        Args:
            x: (N, L, input_dim) input signals
        Returns:
            mu: (N, d_style)
            logvar: (N, d_style)
        """
        # Conv1D expects (N, C, L), so transpose
        h = self.act(self.conv1(x.transpose(1, 2)))  # (N, hidden_dim, L)
        h = h.transpose(1, 2)  # (N, L, hidden_dim)
        h = self.ln1(h)
        h = self.act(self.conv2(h.transpose(1, 2)))  # (N, hidden_dim, L)
        h = h.transpose(1, 2)  # (N, L, hidden_dim)
        h = self.ln2(h)

        # Positional encoding + transformer
        h = self.pe(h)
        h = self.transformer(h)  # (N, L, hidden_dim)

        # Mean pool over time
        h = h.mean(dim=1)  # (N, hidden_dim)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Full forward: encode + reparameterize.

        Args:
            x: (N, L, input_dim) input signals
        Returns:
            z: (N, d_style) sampled latent
            mu: (N, d_style)
            logvar: (N, d_style)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
