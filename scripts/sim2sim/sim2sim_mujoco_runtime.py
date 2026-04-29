#!/usr/bin/env python3
"""MuJoCo runtime wrapper for sim2sim rollout and realtime control."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ActuationArrays:
    actuator_ids: np.ndarray  # shape [N], int64 (-1 for unavailable)
    qpos_adrs: np.ndarray     # shape [N], int64
    qvel_adrs: np.ndarray     # shape [N], int64
    kp: np.ndarray            # shape [N], float64
    kd: np.ndarray            # shape [N], float64
    effort_limit: np.ndarray  # shape [N], float64 (<=0 means no clip)



def launch_viewer_if_needed(render: bool, model: mujoco.MjModel, data: mujoco.MjData):
    if not render:
        return None
    try:
        import mujoco.viewer

        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        print(f"[WARN] Failed to open MuJoCo passive viewer: {exc}")
        return None


class MujocoRuntime:
    """Encapsulates control stepping and optional realtime viewer sync."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuation: ActuationArrays,
        actuation_mode: str,
        sim_substeps: int,
        effective_control_dt: float,
        viewer=None,
        realtime: bool = True,
        realtime_factor: float = 1.0,
    ):
        self.model = model
        self.data = data
        self.actuation = actuation
        self.actuation_mode = str(actuation_mode).strip().lower()
        self.sim_substeps = max(1, int(sim_substeps))
        self.effective_control_dt = float(effective_control_dt)
        self.viewer = viewer
        self.realtime = bool(realtime)
        self.realtime_factor = max(1.0e-3, float(realtime_factor))

    def _apply_position_control(self, desired_joint_pos: np.ndarray):
        ids = self.actuation.actuator_ids
        for i in range(ids.size):
            aid = int(ids[i])
            if aid < 0:
                continue
            self.data.ctrl[aid] = float(desired_joint_pos[i])

    def _apply_torque_pd(self, desired_joint_pos: np.ndarray):
        ids = self.actuation.actuator_ids
        for i in range(ids.size):
            aid = int(ids[i])
            if aid < 0:
                continue
            q = float(self.data.qpos[int(self.actuation.qpos_adrs[i])])
            dq = float(self.data.qvel[int(self.actuation.qvel_adrs[i])])
            tau = float(self.actuation.kp[i]) * (float(desired_joint_pos[i]) - q) - float(self.actuation.kd[i]) * dq
            effort = float(self.actuation.effort_limit[i])
            if effort > 0.0:
                tau = float(np.clip(tau, -effort, effort))
            self.data.ctrl[aid] = tau

    def apply_and_step(self, desired_joint_pos: np.ndarray):
        desired = np.asarray(desired_joint_pos, dtype=np.float64).reshape(-1)
        if desired.size != int(self.actuation.actuator_ids.size):
            raise ValueError(
                f"desired_joint_pos size mismatch: got {desired.size}, "
                f"expected {self.actuation.actuator_ids.size}"
            )

        for _ in range(self.sim_substeps):
            if self.actuation_mode == "position":
                self._apply_position_control(desired)
            else:
                self._apply_torque_pd(desired)
            mujoco.mj_step(self.model, self.data)

    def sync_viewer(self, *, step: int, wall_start: float):
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass

        if not self.realtime:
            return

        sim_elapsed = (step + 1) * self.effective_control_dt
        target_wall = sim_elapsed / self.realtime_factor
        now_wall = time.perf_counter() - wall_start
        remaining = target_wall - now_wall
        if remaining > 0.0:
            time.sleep(remaining)
