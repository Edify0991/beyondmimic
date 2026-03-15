"""Compliance metric and analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
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


def compute_tracking_preservation(plugin_err: np.ndarray, baseline_err: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalized tracking preservation in [0, 1], higher is better."""
    ratio = plugin_err / (baseline_err + eps)
    score = 1.0 - ratio
    return np.clip(score, 0.0, 1.0)


def compute_recovery_success(
    torso_att_err: np.ndarray,
    com_vel: np.ndarray,
    contact_count: np.ndarray,
    att_thresh: float = 0.35,
    vel_thresh: float = 0.7,
    min_contacts: int = 1,
) -> np.ndarray:
    """Binary success flag indicating stable recovery within a window."""
    stable = (torso_att_err < att_thresh) & (com_vel < vel_thresh) & (contact_count >= min_contacts)
    return stable.astype(np.float32)


def compute_payload_vibration_proxy(imu_acc: np.ndarray) -> np.ndarray:
    hf = np.diff(imu_acc, axis=-2) if imu_acc.ndim >= 2 else np.diff(imu_acc)
    return np.sqrt(np.mean(hf**2, axis=tuple(range(1, hf.ndim))))


def compute_impact_proxy(contact_force: np.ndarray | None = None, joint_acc: np.ndarray | None = None) -> np.ndarray:
    if contact_force is not None:
        return np.linalg.norm(contact_force, axis=-1).mean(axis=-1) if contact_force.ndim > 2 else np.abs(contact_force)
    if joint_acc is not None:
        return np.max(np.abs(joint_acc), axis=-1)
    raise ValueError("Provide either contact_force or joint_acc for impact proxy.")


def compute_event_window_metrics(
    tracking_err: np.ndarray,
    impact: np.ndarray,
    payload_vib: np.ndarray,
    oscillation: np.ndarray,
    effort: np.ndarray,
    event_labels: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Aggregate key metrics by event label."""
    out: dict[str, dict[str, float]] = {}
    for ev in np.unique(event_labels):
        mask = event_labels == ev
        out[str(int(ev))] = {
            "tracking_error": float(np.mean(tracking_err[mask])),
            "impact_cost": float(np.mean(impact[mask])),
            "payload_vibration_cost": float(np.mean(payload_vib[mask])),
            "oscillation_cost": float(np.mean(oscillation[mask])),
            "effort_cost": float(np.mean(effort[mask])),
        }
    return out


def aggregate_fixed_grid_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return best setting and summary stats for fixed-grid sweeps."""
    if not rows:
        return {"best": None, "count": 0}

    def score(r: dict[str, Any]) -> float:
        return (
            1.5 * float(r.get("tracking_preservation", 0.0))
            - 0.8 * float(r.get("impact_cost", 0.0))
            - 0.6 * float(r.get("payload_vibration_cost", 0.0))
            - 0.4 * float(r.get("oscillation_cost", 0.0))
            - 0.2 * float(r.get("effort_cost", 0.0))
            + 0.5 * float(r.get("recovery_success", 0.0))
            - 0.8 * float(r.get("fall_rate", 0.0))
        )

    best = max(rows, key=score)
    return {
        "count": len(rows),
        "best": best,
        "mean_tracking_preservation": float(np.mean([r["tracking_preservation"] for r in rows])),
        "mean_impact_cost": float(np.mean([r["impact_cost"] for r in rows])),
    }


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
