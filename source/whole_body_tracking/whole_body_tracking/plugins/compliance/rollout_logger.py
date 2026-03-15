"""Step-level rollout logger for compliance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .hdf5_schema import append_rows, ensure_schema
from .payload_sites import PayloadExtractionCfg, PayloadExtractor


@dataclass
class RolloutLoggerCfg:
    save_path: str
    task_name: str
    motion_name: str
    run_id: str
    seed: int
    selected_joint_names: list[str]
    payload_body_names: list[str]
    payload_site_names: list[str]
    torso_reference_body_name: str


class ComplianceRolloutLogger:
    """Collects step tensors and flushes per-episode data to HDF5."""

    def __init__(self, env, cfg: RolloutLoggerCfg):
        import h5py

        self.env = env
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.step_count = 0
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_contact = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_root_lin_vel: torch.Tensor | None = None

        self.robot = env.scene["robot"]
        self.joint_ids = self._resolve_joint_ids(cfg.selected_joint_names)
        self.payload_extractor = PayloadExtractor(
            self.robot,
            PayloadExtractionCfg(
                payload_body_names=cfg.payload_body_names,
                payload_site_names=cfg.payload_site_names,
                torso_reference_body_name=cfg.torso_reference_body_name,
            ),
        )

        self.buffers = [self._new_episode_buffer(i) for i in range(self.num_envs)]

        self.path = Path(cfg.save_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5f = h5py.File(self.path, "a")
        ensure_schema(self.h5f)

    def close(self):
        for env_id in range(self.num_envs):
            if self.buffers[env_id]["time/sim_step"]:
                self._flush_env(env_id)
        self.h5f.flush()
        self.h5f.close()

    def _new_episode_buffer(self, env_id: int) -> dict[str, list[np.ndarray]]:
        keys = [
            "time/sim_step",
            "time/sim_time",
            "meta/env_id",
            "meta/episode_id",
            "meta/task_name",
            "meta/motion_name",
            "meta/seed",
            "meta/run_id",
            "joint/q",
            "joint/dq",
            "joint/q_des",
            "joint/action_pi",
            "joint/tau_cmd",
            "joint/effort_proxy",
            "root/pos",
            "root/quat",
            "root/lin_vel",
            "root/ang_vel",
            "imu/torso",
            "contact/bool",
            "contact/force",
            "payload/pos",
            "payload/vel",
            "payload/acc",
            "payload/rel_pos",
            "payload/rel_vel",
            "event/aerial_hint",
            "event/touchdown_hint",
        ]
        return {k: [] for k in keys}

    def _resolve_joint_ids(self, names: list[str]) -> torch.Tensor:
        if not names:
            return torch.arange(self.robot.num_joints, device=self.device)
        try:
            ids = self.robot.find_joints(names, preserve_order=True)[0]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            return ids if ids.numel() > 0 else torch.arange(self.robot.num_joints, device=self.device)
        except Exception:
            return torch.arange(self.robot.num_joints, device=self.device)

    def _safe_cmd_joint_pos(self) -> torch.Tensor | None:
        try:
            cmd = self.env.command_manager.get_term("motion")
            if hasattr(cmd, "joint_pos"):
                return cmd.joint_pos[:, self.joint_ids]
        except Exception:
            pass
        return None

    def append_step(self, actions: torch.Tensor, dones: torch.Tensor) -> None:
        dt = float(getattr(self.env, "step_dt", 0.02))
        robot = self.robot

        q = robot.data.joint_pos[:, self.joint_ids]
        dq = robot.data.joint_vel[:, self.joint_ids]
        q_des = self._safe_cmd_joint_pos()
        if q_des is None:
            q_des = torch.full_like(q, float("nan"))
        tau = getattr(robot.data, "applied_torque", None)
        tau_sel = tau[:, self.joint_ids] if tau is not None else torch.full_like(q, float("nan"))
        effort = torch.nanmean(torch.abs(torch.nan_to_num(tau_sel, nan=0.0)), dim=-1, keepdim=True)

        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        root_lin = robot.data.root_lin_vel_w
        root_ang = robot.data.root_ang_vel_w

        if self.prev_root_lin_vel is None:
            lin_acc = torch.zeros_like(root_lin)
        else:
            lin_acc = (root_lin - self.prev_root_lin_vel) / max(dt, 1e-6)
        self.prev_root_lin_vel = root_lin.detach().clone()
        imu_like = torch.cat([lin_acc, root_ang], dim=-1)

        contact_sensor = self.env.scene.sensors.get("contact_forces", None)
        if contact_sensor is not None:
            force_w = contact_sensor.data.net_forces_w.reshape(self.num_envs, -1, 3)
            contact_force = torch.norm(force_w, dim=-1).mean(dim=-1, keepdim=True)
            contact_bool = (contact_force > 1.0).float()
        else:
            contact_force = torch.zeros(self.num_envs, 1, device=self.device)
            contact_bool = torch.zeros(self.num_envs, 1, device=self.device)

        payload = self.payload_extractor.extract(dt)

        aerial_hint = (contact_bool < 0.5).float()
        touchdown_hint = ((self.prev_contact.unsqueeze(-1) < 0.5) & (contact_bool >= 0.5)).float()
        self.prev_contact = contact_bool.squeeze(-1)

        sim_step = np.full((self.num_envs,), self.step_count, dtype=np.int64)
        sim_time = np.full((self.num_envs,), self.step_count * dt, dtype=np.float64)
        self.step_count += 1

        for env_id in range(self.num_envs):
            b = self.buffers[env_id]
            b["time/sim_step"].append(sim_step[env_id])
            b["time/sim_time"].append(sim_time[env_id])
            b["meta/env_id"].append(np.int32(env_id))
            b["meta/episode_id"].append(np.int64(self.episode_ids[env_id].item()))
            b["meta/task_name"].append(self.cfg.task_name)
            b["meta/motion_name"].append(self.cfg.motion_name)
            b["meta/seed"].append(np.int64(self.cfg.seed))
            b["meta/run_id"].append(self.cfg.run_id)
            b["joint/q"].append(q[env_id].detach().cpu().numpy())
            b["joint/dq"].append(dq[env_id].detach().cpu().numpy())
            b["joint/q_des"].append(q_des[env_id].detach().cpu().numpy())
            b["joint/action_pi"].append(actions[env_id].detach().cpu().numpy())
            b["joint/tau_cmd"].append(tau_sel[env_id].detach().cpu().numpy())
            b["joint/effort_proxy"].append(effort[env_id].detach().cpu().numpy())
            b["root/pos"].append(root_pos[env_id].detach().cpu().numpy())
            b["root/quat"].append(root_quat[env_id].detach().cpu().numpy())
            b["root/lin_vel"].append(root_lin[env_id].detach().cpu().numpy())
            b["root/ang_vel"].append(root_ang[env_id].detach().cpu().numpy())
            b["imu/torso"].append(imu_like[env_id].detach().cpu().numpy())
            b["contact/bool"].append(contact_bool[env_id].detach().cpu().numpy())
            b["contact/force"].append(contact_force[env_id].detach().cpu().numpy())
            b["payload/pos"].append(payload["payload_pos"][env_id].detach().cpu().numpy())
            b["payload/vel"].append(payload["payload_vel"][env_id].detach().cpu().numpy())
            b["payload/acc"].append(payload["payload_acc"][env_id].detach().cpu().numpy())
            b["payload/rel_pos"].append(payload["payload_rel_pos"][env_id].detach().cpu().numpy())
            b["payload/rel_vel"].append(payload["payload_rel_vel"][env_id].detach().cpu().numpy())
            b["event/aerial_hint"].append(aerial_hint[env_id].detach().cpu().numpy())
            b["event/touchdown_hint"].append(touchdown_hint[env_id].detach().cpu().numpy())

            if bool(dones[env_id].item()):
                self._flush_env(env_id)
                self.episode_ids[env_id] += 1
                self.buffers[env_id] = self._new_episode_buffer(env_id)

    def _flush_env(self, env_id: int):
        b = self.buffers[env_id]
        for k, v in b.items():
            if not v:
                continue
            path = "/" + k
            arr = np.asarray(v)
            append_rows(self.h5f, path, arr)
        self.h5f.flush()
