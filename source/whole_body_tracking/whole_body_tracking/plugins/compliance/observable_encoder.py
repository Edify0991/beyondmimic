"""Causal deployable observation encoder for compliance distillation."""

from __future__ import annotations

import torch
from torch import nn


class ObservableHistoryEncoder(nn.Module):
    """Small causal Conv1D + GRU encoder.

    Expected input: tensor with shape [B, T, C].
    """

    def __init__(self, in_dim: int, conv_dim: int = 64, gru_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        self.causal = nn.Conv1d(in_dim, conv_dim, kernel_size=kernel_size, padding=kernel_size - 1)
        self.norm = nn.LayerNorm(conv_dim)
        self.gru = nn.GRU(conv_dim, gru_dim, batch_first=True)
        self.out = nn.Sequential(nn.Linear(gru_dim, gru_dim), nn.SiLU())

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        x = obs_hist.transpose(1, 2)
        x = self.causal(x)
        x = x[..., : obs_hist.shape[1]]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x, _ = self.gru(x)
        return self.out(x[:, -1])
