"""Compliance metric and analysis utilities."""

from __future__ import annotations

import torch


def tracking_rmse(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((q - q_ref) ** 2).mean(dim=-1))


def impact_cost(impact: torch.Tensor) -> torch.Tensor:
    return impact.mean(dim=-1) if impact.ndim > 1 else impact


def oscillation_cost(osc: torch.Tensor) -> torch.Tensor:
    return osc.mean(dim=-1) if osc.ndim > 1 else osc


def payload_vibration_cost(vib: torch.Tensor) -> torch.Tensor:
    return vib.mean(dim=-1) if vib.ndim > 1 else vib


def effort_cost(effort: torch.Tensor) -> torch.Tensor:
    return effort.mean(dim=-1) if effort.ndim > 1 else effort


def limit_risk(risk: torch.Tensor) -> torch.Tensor:
    return risk.mean(dim=-1) if risk.ndim > 1 else risk


def expert_specialization_stats(gate: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    usage = gate.mean(dim=0)
    entropy = -(gate * (gate + 1e-8).log()).sum(dim=-1).mean()
    out = {"usage": usage, "gate_entropy": entropy}
    if labels is not None:
        num_labels = int(labels.max().item()) + 1
        by_label = []
        for i in range(num_labels):
            mask = labels == i
            by_label.append(gate[mask].mean(dim=0) if mask.any() else torch.zeros_like(usage))
        out["usage_by_label"] = torch.stack(by_label)
    return out
