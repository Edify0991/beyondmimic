"""Typed containers used by the compliance plugin pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObservableStateWindow:
    """Deployable observation history window."""

    q: torch.Tensor
    dq: torch.Tensor
    q_des: torch.Tensor
    action_pi: torch.Tensor
    q_margin: torch.Tensor
    torso_imu: torch.Tensor
    effort_proxy: torch.Tensor


@dataclass
class PrivilegedStateWindow(ObservableStateWindow):
    """Training-time privileged history with non-deployable channels."""

    contact_force: torch.Tensor
    impact_proxy: torch.Tensor
    oscillation_proxy: torch.Tensor
    payload_vibration_proxy: torch.Tensor
    contact_mode: torch.Tensor
    true_torque: torch.Tensor | None = None


@dataclass
class ComplianceLatent:
    """Bounded compliance latent z = [omega, zeta, epsilon, alpha, beta]."""

    value: torch.Tensor


@dataclass
class ParetoMetricVector:
    """Predicted/realized local metric vector."""

    value: torch.Tensor


@dataclass
class ExpertAssignmentInfo:
    """MoE assignment information for analysis and distillation."""

    gate: torch.Tensor
    topk_idx: torch.Tensor
    prototype_sim: torch.Tensor
