# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import numbers
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]
        self.policy_obs_shape = self._resolve_policy_obs_shape(env)

    @staticmethod
    def _resolve_policy_obs_shape(env: ManagerBasedRLEnv) -> tuple[int, ...]:
        """Infer policy observation shape from observation manager/group metadata."""
        group_dim = env.observation_manager.group_obs_dim.get("policy")
        if group_dim is None:
            raise ValueError("Observation group 'policy' not found in env.observation_manager.group_obs_dim.")

        if isinstance(group_dim, tuple):
            return tuple(int(x) for x in group_dim)

        if isinstance(group_dim, list):
            flat: list[int] = []
            for term_shape in group_dim:
                if not isinstance(term_shape, tuple):
                    raise ValueError(f"Unsupported policy term shape type: {type(term_shape)}")
                flat.extend(int(x) for x in term_shape)
            return (int(sum(flat)),)

        raise ValueError(f"Unsupported policy observation shape type: {type(group_dim)}")

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros((1, *self.policy_obs_shape), dtype=torch.float32)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 6, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(float(x)) if isinstance(x, numbers.Real) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    robot = env.scene["robot"]
    motion_term: MotionCommand = env.command_manager.get_term("motion")
    action_term = env.action_manager.get_term("joint_pos")

    # action joint ordering (matches policy action output dimension/order)
    action_joint_names = list(getattr(action_term, "_joint_names", []))
    if not action_joint_names:
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        if action_joint_ids is not None:
            action_joint_names = [robot.data.joint_names[i] for i in action_joint_ids]

    # policy joint observation ordering (used by joint_pos / joint_vel terms)
    policy_joint_names = list(robot.data.joint_names)
    policy_cfg = getattr(env.observation_manager.cfg, "policy", None)
    joint_pos_term_cfg = getattr(policy_cfg, "joint_pos", None) if policy_cfg is not None else None
    if joint_pos_term_cfg is not None:
        params = getattr(joint_pos_term_cfg, "params", None) or {}
        asset_cfg = params.get("asset_cfg")
        if asset_cfg is not None:
            obs_joint_names = getattr(asset_cfg, "joint_names", None)
            if obs_joint_names is not None:
                _, resolved = robot.find_joints(obs_joint_names, preserve_order=True)
                if len(resolved) > 0:
                    policy_joint_names = list(resolved)

    metadata = {
        "run_path": run_path,
        "joint_names": robot.data.joint_names,
        "joint_stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot.data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos_nominal.cpu().tolist(),
        "default_joint_vel": robot.data.default_joint_vel[0].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": action_term._scale[0].cpu().tolist(),
        "action_joint_names": action_joint_names,
        "command_joint_names": list(motion_term.joint_names),
        "policy_joint_names": policy_joint_names,
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": motion_term.cfg.body_names,
        "root_body_name": robot.data.body_names[0],
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.decimation,
        "control_dt": env.cfg.sim.dt * env.cfg.decimation,
        "motion_fps": float(motion_term.motion.fps),
        "motion_num_frames": int(motion_term.motion.time_step_total),
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
