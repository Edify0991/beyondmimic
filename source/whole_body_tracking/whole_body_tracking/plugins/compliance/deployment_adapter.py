"""Deployment-time adapter for action/impedance/torque shaping."""

from __future__ import annotations

import torch

from .compliance_mapping import impedance_shaping, torque_shaping


class DeploymentAdapter(torch.nn.Module):
    """TorchScript-friendly plugin adapter."""

    def __init__(self, mode: str = "impedance", alpha_lpf: float = 0.2, rate_limit: float = 0.1):
        super().__init__()
        self.mode = mode
        self.alpha_lpf = alpha_lpf
        self.rate_limit = rate_limit
        self.register_buffer("prev", torch.zeros(1, 1), persistent=False)

    def _lpf(self, x: torch.Tensor) -> torch.Tensor:
        if self.prev.shape != x.shape:
            self.prev = torch.zeros_like(x)
        y = self.alpha_lpf * x + (1.0 - self.alpha_lpf) * self.prev
        self.prev = y.detach()
        return y

    def _rate_limit(self, x: torch.Tensor) -> torch.Tensor:
        dx = x - self.prev
        dx = dx.clamp(-self.rate_limit, self.rate_limit)
        return self.prev + dx

    def forward(self, base_action: torch.Tensor, obs_state: dict[str, torch.Tensor], student_out: dict[str, torch.Tensor], safe_action: torch.Tensor | None = None) -> torch.Tensor:
        z = student_out["z"]
        if self.mode == "impedance":
            shaped = impedance_shaping(z, obs_state["q"], obs_state["q_des"])["q_ref"]
        else:
            shaped = torque_shaping(z, base_action, obs_state.get("state", None))
        shaped = self._lpf(shaped)
        shaped = self._rate_limit(shaped)
        if safe_action is not None:
            valid = torch.isfinite(shaped).all(dim=-1, keepdim=True)
            shaped = torch.where(valid, shaped, safe_action)
        return shaped
