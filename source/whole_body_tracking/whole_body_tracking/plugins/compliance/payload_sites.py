"""Payload site/body extraction helpers for compliance rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PayloadExtractionCfg:
    payload_body_names: list[str]
    payload_site_names: list[str]
    torso_reference_body_name: str = "torso_link"


class PayloadExtractor:
    """Extract payload and torso-relative kinematics from robot body states."""

    def __init__(self, robot, cfg: PayloadExtractionCfg):
        self.robot = robot
        self.cfg = cfg
        self.payload_ids = self._resolve_body_ids(cfg.payload_body_names or cfg.payload_site_names)
        self.torso_ids = self._resolve_body_ids([cfg.torso_reference_body_name])
        self.prev_vel: torch.Tensor | None = None

    def _resolve_body_ids(self, names: list[str]) -> torch.Tensor:
        if not names:
            return torch.tensor([0], dtype=torch.long, device=self.robot.device)
        try:
            ids = self.robot.find_bodies(names, preserve_order=True)[0]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long, device=self.robot.device)
            return ids if ids.numel() > 0 else torch.tensor([0], dtype=torch.long, device=self.robot.device)
        except Exception:
            return torch.tensor([0], dtype=torch.long, device=self.robot.device)

    def extract(self, dt: float) -> dict[str, torch.Tensor]:
        pos = self.robot.data.body_pos_w[:, self.payload_ids].mean(dim=1)
        vel = self.robot.data.body_lin_vel_w[:, self.payload_ids].mean(dim=1)
        torso_pos = self.robot.data.body_pos_w[:, self.torso_ids].mean(dim=1)
        torso_vel = self.robot.data.body_lin_vel_w[:, self.torso_ids].mean(dim=1)

        if self.prev_vel is None:
            acc = torch.zeros_like(vel)
        else:
            acc = (vel - self.prev_vel) / max(dt, 1e-6)
        self.prev_vel = vel.detach().clone()

        return {
            "payload_pos": pos,
            "payload_vel": vel,
            "payload_acc": acc,
            "payload_rel_pos": pos - torso_pos,
            "payload_rel_vel": vel - torso_vel,
        }
