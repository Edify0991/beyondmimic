"""Proxy signal computation utilities for compliance learning/logging."""

from __future__ import annotations

import torch


def impact_proxy(contact_force: torch.Tensor | None, joint_acc: torch.Tensor | None, imu_acc: torch.Tensor | None) -> torch.Tensor:
    if contact_force is not None:
        return contact_force.norm(dim=-1)
    if joint_acc is not None:
        return joint_acc.abs().amax(dim=-1)
    if imu_acc is None:
        raise ValueError("Need one source for impact proxy.")
    hf = imu_acc[:, 1:] - imu_acc[:, :-1]
    return hf.pow(2).mean(dim=(-1, -2)).sqrt()


def oscillation_proxy(signal: torch.Tensor, split_hz: int = 5) -> torch.Tensor:
    spec = torch.fft.rfft(signal, dim=-1).abs().pow(2)
    n = spec.shape[-1]
    low = spec[..., : max(1, int(n * split_hz / 20.0))].mean(dim=-1)
    high = spec[..., max(1, int(n * split_hz / 20.0)) :].mean(dim=-1)
    return high / (low + 1e-6)


def payload_vibration_proxy(imu_acc: torch.Tensor, jerk: torch.Tensor | None = None) -> torch.Tensor:
    hf = imu_acc[:, 1:] - imu_acc[:, :-1]
    rms = hf.pow(2).mean(dim=(-1, -2)).sqrt()
    if jerk is not None:
        rms = 0.5 * rms + 0.5 * jerk.pow(2).mean(dim=-1).sqrt()
    return rms


def effort_proxy(torque: torch.Tensor, reversal_window: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    effort = torque.abs().mean(dim=-1)
    reversals = torch.zeros_like(effort)
    if reversal_window is not None and reversal_window.shape[1] > 1:
        reversals = (torch.sign(reversal_window[:, 1:]) != torch.sign(reversal_window[:, :-1])).float().sum(dim=(1, 2))
    return effort, reversals


def limit_risk_proxy(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    low = torch.relu((q_min + margin) - q)
    high = torch.relu(q - (q_max - margin))
    return (low + high).mean(dim=-1)


def support_confidence(contact_prob: torch.Tensor, momentum: float = 0.9, prev: torch.Tensor | None = None) -> torch.Tensor:
    est = contact_prob if prev is None else momentum * prev + (1 - momentum) * contact_prob
    return est.clamp(0.0, 1.0)
