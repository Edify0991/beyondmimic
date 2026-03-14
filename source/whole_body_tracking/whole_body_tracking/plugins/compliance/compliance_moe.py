"""Contrastive prototype-gated Mixture-of-Experts compliance teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ComplianceBounds:
    omega_min: float = 2.0
    omega_max: float = 40.0
    zeta_min: float = 0.2
    zeta_max: float = 2.0
    eps_min: float = 0.0
    eps_max: float = 0.25
    alpha_min: float = 0.1
    beta_max: float = 1.0


class _ExpertHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplianceMoE(nn.Module):
    """Prototype-based gating with regularization hooks."""

    def __init__(self, in_dim: int, num_experts: int = 4, proto_dim: int = 64, hard_top1: bool = False, temperature: float = 0.2):
        super().__init__()
        self.num_experts = num_experts
        self.hard_top1 = hard_top1
        self.temperature = temperature
        self.bounds = ComplianceBounds()

        self.project = nn.Linear(in_dim, proto_dim)
        self.prototypes = nn.Parameter(torch.randn(num_experts, proto_dim))
        self.experts = nn.ModuleList([_ExpertHead(in_dim) for _ in range(num_experts)])

    def _bound_latent(self, raw: torch.Tensor) -> torch.Tensor:
        b = self.bounds
        omega = b.omega_min + torch.sigmoid(raw[..., 0]) * (b.omega_max - b.omega_min)
        zeta = b.zeta_min + torch.sigmoid(raw[..., 1]) * (b.zeta_max - b.zeta_min)
        eps = b.eps_min + torch.sigmoid(raw[..., 2]) * (b.eps_max - b.eps_min)
        alpha = b.alpha_min + torch.sigmoid(raw[..., 3]) * (1.0 - b.alpha_min)
        beta = torch.sigmoid(raw[..., 4]) * b.beta_max
        return torch.stack([omega, zeta, eps, alpha, beta], dim=-1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        proj = F.normalize(self.project(h), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        sim = proj @ proto.t()
        gate = F.softmax(sim / self.temperature, dim=-1)
        if self.hard_top1:
            idx = torch.argmax(gate, dim=-1)
            gate = F.one_hot(idx, num_classes=self.num_experts).float()
        expert_raw = torch.stack([head(h) for head in self.experts], dim=1)
        expert_z = self._bound_latent(expert_raw)
        z_mix = torch.sum(gate.unsqueeze(-1) * expert_z, dim=1)
        return {"z_teacher": z_mix, "gate": gate, "expert_z": expert_z, "similarity": sim, "proj": proj}

    def load_balance_loss(self, gate: torch.Tensor) -> torch.Tensor:
        usage = gate.mean(dim=0)
        target = torch.full_like(usage, 1.0 / usage.numel())
        return F.mse_loss(usage, target)

    def orthogonality_loss(self) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1)
        gram = p @ p.t()
        eye = torch.eye(self.num_experts, device=gram.device)
        return ((gram - eye) ** 2).mean()

    def contrastive_assignment_loss(self, similarity: torch.Tensor, target_expert: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(similarity, target_expert)
