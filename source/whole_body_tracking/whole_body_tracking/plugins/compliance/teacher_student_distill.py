"""Teacher-student distillation module for deployable compliance inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ObservableStudentHead(nn.Module):
    """Predicts z and gating using only observable history features."""

    def __init__(self, in_dim: int, num_experts: int, latent_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim), nn.SiLU())
        self.z_head = nn.Linear(latent_dim, 5)
        self.g_head = nn.Linear(latent_dim, num_experts)

    def forward(self, h_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(h_obs)
        return {"z": self.z_head(f), "gate": torch.softmax(self.g_head(f), dim=-1), "feat": f}


def distillation_losses(
    student_out: dict[str, torch.Tensor],
    teacher_z: torch.Tensor,
    teacher_gate: torch.Tensor,
    teacher_feat: torch.Tensor | None = None,
    rollout_loss: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute core distillation losses."""
    lz = F.mse_loss(student_out["z"], teacher_z)
    lkl = F.kl_div(torch.log(student_out["gate"] + 1e-8), teacher_gate, reduction="batchmean")
    lfeat = torch.tensor(0.0, device=teacher_z.device)
    if teacher_feat is not None:
        lfeat = F.mse_loss(student_out["feat"], teacher_feat)
    lroll = torch.tensor(0.0, device=teacher_z.device) if rollout_loss is None else rollout_loss
    return {"L_student_z": lz, "L_student_KL": lkl, "L_feat": lfeat, "L_roll": lroll}
