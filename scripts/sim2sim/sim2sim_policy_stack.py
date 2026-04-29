#!/usr/bin/env python3
"""Policy and auxiliary ONNX stage stack for SONIC-style sim2sim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort



def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]



def parse_csv_floats(raw: str | None) -> list[float]:
    out: list[float] = []
    for item in parse_csv_list(raw):
        out.append(float(item))
    return out



def parse_csv_ints(raw: str | None) -> list[int]:
    out: list[int] = []
    for item in parse_csv_list(raw):
        out.append(int(float(item)))
    return out


@dataclass(frozen=True)
class MainPolicyMetadata:
    observation_names: list[str]
    observation_history_lengths: list[int]
    body_names: list[str]
    joint_names: list[str]
    action_joint_names: list[str]
    command_joint_names: list[str]
    policy_joint_names: list[str]
    joint_stiffness: list[float]
    joint_damping: list[float]
    default_joint_pos: list[float]
    default_joint_vel: list[float]
    action_scale: list[float]
    anchor_body_name: str
    root_body_name: str
    control_dt: float
    motion_num_frames: int


class OnnxModel:
    """Thin ONNX Runtime wrapper with typed helpers."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        providers = providers or ["CPUExecutionProvider"]
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)

    def infer(self, feeds: dict[str, np.ndarray], output_names: list[str] | None = None) -> dict[str, np.ndarray]:
        requested = output_names if output_names is not None else self.output_names
        outputs = self.session.run(requested, feeds)
        return {name: np.asarray(value) for name, value in zip(requested, outputs)}



def parse_main_policy_metadata(model: OnnxModel) -> MainPolicyMetadata:
    meta = model.metadata
    observation_names = parse_csv_list(meta.get("observation_names"))
    observation_history_lengths = parse_csv_ints(meta.get("observation_history_lengths"))
    body_names = parse_csv_list(meta.get("body_names"))
    joint_names = parse_csv_list(meta.get("joint_names"))

    action_joint_names = parse_csv_list(meta.get("action_joint_names"))
    command_joint_names = parse_csv_list(meta.get("command_joint_names"))
    policy_joint_names = parse_csv_list(meta.get("policy_joint_names"))

    return MainPolicyMetadata(
        observation_names=observation_names,
        observation_history_lengths=observation_history_lengths,
        body_names=body_names,
        joint_names=joint_names,
        action_joint_names=action_joint_names,
        command_joint_names=command_joint_names,
        policy_joint_names=policy_joint_names,
        joint_stiffness=parse_csv_floats(meta.get("joint_stiffness")),
        joint_damping=parse_csv_floats(meta.get("joint_damping")),
        default_joint_pos=parse_csv_floats(meta.get("default_joint_pos")),
        default_joint_vel=parse_csv_floats(meta.get("default_joint_vel")),
        action_scale=parse_csv_floats(meta.get("action_scale")),
        anchor_body_name=str(meta.get("anchor_body_name", "")).strip(),
        root_body_name=str(meta.get("root_body_name", "")).strip(),
        control_dt=float(meta.get("control_dt", "0") or 0.0),
        motion_num_frames=int(float(meta.get("motion_num_frames", "0") or 0.0)),
    )



def _ensure_batch_2d(obs_vec: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs_vec, dtype=np.float32).reshape(1, -1)
    return obs



def _ensure_time_input(time_step: int) -> np.ndarray:
    return np.asarray([[int(time_step)]], dtype=np.float32)


@dataclass(frozen=True)
class AuxStageConfig:
    name: str
    onnx_path: str
    pipeline: str
    obs_input_name: str = "obs"
    time_input_name: str = "time_step"
    output_bindings: dict[str, str] | None = None


class AuxiliaryStage:
    def __init__(self, cfg: AuxStageConfig, providers: list[str] | None = None):
        self.cfg = cfg
        self.model = OnnxModel(cfg.onnx_path, providers=providers)

    def run(self, *, obs_vec: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {self.cfg.obs_input_name: _ensure_batch_2d(obs_vec)}
        if self.cfg.time_input_name in self.model.input_names:
            feeds[self.cfg.time_input_name] = _ensure_time_input(time_step)
        raw_outputs = self.model.infer(feeds)

        if not self.cfg.output_bindings:
            return {k: np.asarray(v).reshape(-1) for k, v in raw_outputs.items()}

        mapped: dict[str, np.ndarray] = {}
        for output_name, target_name in self.cfg.output_bindings.items():
            if output_name not in raw_outputs:
                raise KeyError(
                    f"Aux stage '{self.cfg.name}' output '{output_name}' not found. "
                    f"Available: {sorted(raw_outputs.keys())}"
                )
            mapped[str(target_name)] = np.asarray(raw_outputs[output_name]).reshape(-1)
        return mapped


class MainPolicy:
    """Main BeyondMimic policy ONNX wrapper with action/reference extraction."""

    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        self.model = OnnxModel(onnx_path, providers=providers)
        self.metadata = parse_main_policy_metadata(self.model)

        self.action_output_name = self._detect_action_output_name()
        self.reference_outputs = [
            name
            for name in [
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ]
            if name in self.model.output_names
        ]

    def _detect_action_output_name(self) -> str:
        for name in ["actions", "action"]:
            if name in self.model.output_names:
                return name
        # Fallback: use first output if model naming differs.
        if not self.model.output_names:
            raise ValueError("Main policy ONNX has no outputs.")
        return self.model.output_names[0]

    def infer(self, *, obs_vec: np.ndarray, time_step: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        feeds = {
            "obs": _ensure_batch_2d(obs_vec),
            "time_step": _ensure_time_input(time_step),
        }

        # Graceful fallback if ONNX input names differ.
        if "obs" not in self.model.input_names:
            feeds = {self.model.input_names[0]: _ensure_batch_2d(obs_vec)}
            if len(self.model.input_names) > 1:
                feeds[self.model.input_names[1]] = _ensure_time_input(time_step)

        outputs = self.model.infer(feeds)
        if self.action_output_name not in outputs:
            raise KeyError(
                f"Action output '{self.action_output_name}' missing. "
                f"Available outputs: {sorted(outputs.keys())}"
            )
        action = np.asarray(outputs[self.action_output_name], dtype=np.float32).reshape(-1)

        refs: dict[str, np.ndarray] = {}
        for name in self.reference_outputs:
            refs[name] = np.asarray(outputs[name])
        return action, refs


class PolicyStack:
    """Main policy + optional auxiliary stage chain."""

    def __init__(
        self,
        *,
        main_onnx_path: str,
        aux_stage_cfgs: list[AuxStageConfig] | None = None,
        providers: list[str] | None = None,
    ):
        self.main_policy = MainPolicy(main_onnx_path, providers=providers)
        self.aux_stages = [AuxiliaryStage(cfg, providers=providers) for cfg in (aux_stage_cfgs or [])]

    def run_aux_stages(
        self,
        *,
        stage_obs: dict[str, np.ndarray],
        time_step: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        merged: dict[str, np.ndarray] = {}
        traces: dict[str, dict[str, np.ndarray]] = {}
        for stage in self.aux_stages:
            if stage.cfg.name not in stage_obs:
                raise KeyError(
                    f"Aux stage '{stage.cfg.name}' has no prepared observation vector in stage_obs map."
                )
            outputs = stage.run(obs_vec=stage_obs[stage.cfg.name], time_step=time_step)
            traces[stage.cfg.name] = outputs
            merged.update(outputs)
        return merged, traces

    def run_main_policy(self, *, obs_vec: np.ndarray, time_step: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.main_policy.infer(obs_vec=obs_vec, time_step=time_step)



def build_aux_stage_configs(policy_stack_cfg: dict, config_base_dir: str) -> list[AuxStageConfig]:
    """Parse auxiliary stage definitions from config dict."""

    stages_raw = policy_stack_cfg.get("aux_models", policy_stack_cfg.get("stages", []))
    if not isinstance(stages_raw, list):
        return []

    out: list[AuxStageConfig] = []
    for idx, item in enumerate(stages_raw):
        if not isinstance(item, dict):
            raise ValueError("policy_stack auxiliary stage items must be dict objects.")
        enabled = bool(item.get("enabled", True))
        if not enabled:
            continue

        name = str(item.get("name", f"aux_{idx}")).strip() or f"aux_{idx}"
        onnx_path = str(item.get("onnx_path", "")).strip()
        if not onnx_path:
            raise ValueError(f"Aux stage '{name}' missing onnx_path.")

        pipeline = str(item.get("pipeline", name)).strip() or name
        obs_input_name = str(item.get("obs_input_name", "obs")).strip() or "obs"
        time_input_name = str(item.get("time_input_name", "time_step")).strip() or "time_step"

        bindings_raw = item.get("output_bindings", item.get("outputs", {}))
        bindings: dict[str, str] | None = None
        if isinstance(bindings_raw, dict) and bindings_raw:
            bindings = {str(k): str(v) for k, v in bindings_raw.items()}

        out.append(
            AuxStageConfig(
                name=name,
                onnx_path=onnx_path,
                pipeline=pipeline,
                obs_input_name=obs_input_name,
                time_input_name=time_input_name,
                output_bindings=bindings,
            )
        )
    return out
