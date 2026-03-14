"""Map compliance latent vectors to control shaping quantities."""

from __future__ import annotations

import torch


def _lookup_param(q: torch.Tensor, value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(value, (int, float)):
        return torch.full_like(q, float(value))
    if value.ndim == 1:
        return value.unsqueeze(0).expand_as(q)
    return value


def impedance_shaping(
    z: torch.Tensor,
    q: torch.Tensor,
    q_des_pi: torch.Tensor,
    j_hat: float | torch.Tensor = 1.0,
    b_hat: float | torch.Tensor = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute impedance-shaping outputs from latent z."""
    omega, zeta, epsilon, alpha, beta = z.unbind(dim=-1)
    j = _lookup_param(q, j_hat)
    b = _lookup_param(q, b_hat)
    kp = j * omega.unsqueeze(-1) ** 2
    kd = 2.0 * zeta.unsqueeze(-1) * j * omega.unsqueeze(-1) - b
    q_ref = q + alpha.unsqueeze(-1) * (q_des_pi - q)
    return {"kp": kp, "kd": kd, "q_ref": q_ref, "deadband": epsilon.unsqueeze(-1), "beta": beta.unsqueeze(-1)}


def torque_shaping(z: torch.Tensor, tau_pi: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a bounded additive torque residual controlled by beta."""
    beta = z[..., 4:5]
    residual = torch.tanh(tau_pi) * beta
    if state is not None:
        residual = residual * torch.sigmoid(state[..., : tau_pi.shape[-1]])
    return tau_pi + residual
