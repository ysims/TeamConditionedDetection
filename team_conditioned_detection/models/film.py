"""FiLM (Feature-wise Linear Modulation, Perez et al. 2018) conditioning.

A FiLMGenerator maps a conditioning embedding to a per-channel (gamma, beta)
pair; applying it to a feature map rescales/shifts each channel based on
the embedding, e.g. letting the network specialize its features toward
a red-jerseyed robot vs a blue-jerseyed one.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    def __init__(self, embed_dim: int, num_channels: int, hidden_dim: int = 256):
        super().__init__()
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        # Zero-init the last layer so FiLM starts as the identity transform
        # (gamma=1, beta=0): training begins equivalent to the no-FiLM
        # baseline and only diverges as the conditioning proves useful,
        # rather than injecting random noise into the features from step 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(embedding)
        gamma_raw, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + gamma_raw
        return gamma, beta


def apply_film(feature_map: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """feature_map: (B, C, H, W); gamma, beta: (B, C)."""
    gamma = gamma[:, :, None, None]
    beta = beta[:, :, None, None]
    return feature_map * gamma + beta
