"""Short-horizon local metric predictor for counterfactual optimization."""

from __future__ import annotations

import torch
from torch import nn


class ParetoPredictor(nn.Module):
    """Predicts local metric vector y_hat from histories and compliance latent."""

    def __init__(self, obs_dim: int, priv_dim: int, z_dim: int = 5, hidden_dim: int = 256, predict_var: bool = False):
        super().__init__()
        self.predict_var = predict_var
        out_dim = 6 * (2 if predict_var else 1)
        self.obs_gru = nn.GRU(obs_dim, hidden_dim // 2, batch_first=True)
        self.priv_gru = nn.GRU(priv_dim, hidden_dim // 2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs_hist: torch.Tensor, priv_hist: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h_obs, _ = self.obs_gru(obs_hist)
        h_priv, _ = self.priv_gru(priv_hist)
        h = torch.cat([h_obs[:, -1], h_priv[:, -1], z], dim=-1)
        pred = self.head(h)
        if self.predict_var:
            mean, logvar = pred.chunk(2, dim=-1)
            return {"mean": mean, "var": torch.exp(logvar).clamp_min(1e-6)}
        return {"mean": pred}
