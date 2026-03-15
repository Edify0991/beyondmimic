"""Privileged encoder used only during training."""

from __future__ import annotations

import torch
from torch import nn


class PrivilegedHistoryEncoder(nn.Module):
    """Encodes richer privileged windows with a GRU backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, priv_hist: torch.Tensor) -> torch.Tensor:
        x = self.pre(priv_hist)
        x, _ = self.gru(x)
        return self.out(x[:, -1])
