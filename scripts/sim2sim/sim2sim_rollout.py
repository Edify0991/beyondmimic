#!/usr/bin/env python3
"""SONIC-style MuJoCo sim2sim rollout for BeyondMimic policies."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from sim2sim_mujoco_runtime import ActuationArrays, MujocoRuntime, launch_viewer_if_needed
from sim2sim_observation import ObservationAssembler, render_term_table
from sim2sim_policy_stack import (
    MainPolicy,
    PolicyStack,
    build_aux_stage_configs,
)
from sonic_fsm import ProgramFSM, ProgramState
from sonic_io import TelemetryPacket, build_input_channel, build_output_channels
from sonic_observation_config import materialize_observation_block, resolve_observation_selection
from sonic_observation_registry import (
    ObservationShapeContext,
    build_default_observation_registry,
    load_observation_registry_plugin,
)



def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]



def _parse_csv_floats(raw: str | None) -> list[float]:
    out: list[float] = []
    for item in _parse_csv_list(raw):
        out.append(float(item))
    return out



def _parse_csv_ints(raw: str | None) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(raw):
        out.append(int(float(item)))
    return out



def _load_dict_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in [".yaml", ".yml"]:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be dict: {path}")
    return data



def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default



def _coerce_name_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list/str for names, got {type(value)}")



def _coerce_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(float(item.strip())) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"Expected list/str for ints, got {type(value)}")



def _coerce_name_map(value) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
    if isinstance(value, str):
        mapping: dict[str, str] = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid mapping item: {item}")
            a, b = item.split(":", 1)
            a = a.strip()
            b = b.strip()
            if a and b:
                mapping[a] = b
        return mapping
    raise ValueError(f"Expected dict/str for name map, got {type(value)}")



def _coerce_float_map(value) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("joint_gains/action_scale map must be dict")
    return {str(k): float(v) for k, v in value.items()}



def _cfg_pick(cfg: dict, candidates: list[tuple[str, ...] | str], default=None):
    for candidate in candidates:
        path = (candidate,) if isinstance(candidate, str) else tuple(candidate)
        cur: Any = cfg
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            if not (isinstance(cur, str) and cur.strip() == ""):
                return cur
    return default



def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    x = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(x)
    if n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return x / n



def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    qn = _normalize_quat_wxyz(q)
    return np.asarray([qn[0], -qn[1], -qn[2], -qn[3]], dtype=np.float64)



def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = _normalize_quat_wxyz(q2)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )



def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )



def _quat_rotate_inv_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    rot = _quat_to_rotmat_wxyz(q)
    return rot.T @ np.asarray(v, dtype=np.float64)



def _rotmat_first_two_columns_flat(q: np.ndarray) -> np.ndarray:
    mat = _quat_to_rotmat_wxyz(q)
    return mat[:, :2].reshape(-1)



def _quat_angle_error_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    a = _normalize_quat_wxyz(q1)
    b = _normalize_quat_wxyz(q2)
    dot = float(np.clip(np.abs(np.dot(a, b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)



def _split_velocity6(vel6: np.ndarray, velocity_order: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(vel6, dtype=np.float64).reshape(6)
    if velocity_order == "ang_lin":
        ang = x[:3]
        lin = x[3:6]
    else:
        lin = x[:3]
        ang = x[3:6]
    return lin, ang


@dataclass(frozen=True)
class JointSpec:
    train_name: str
    mj_name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int
    actuator_id: int
    default_pos: float
    default_vel: float
    kp: float
    kd: float
    effort_limit: float



def _find_actuator_id(model: mujoco.MjModel, joint_name: str, actuator_name_map: dict[str, str]) -> int:
    candidate_names = [
        actuator_name_map.get(joint_name, ""),
        joint_name,
        f"motor_{joint_name}",
        f"{joint_name}_motor",
    ]
    for name in candidate_names:
        if not name:
            continue
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except Exception:
            aid = -1
        if aid >= 0:
            return int(aid)
    return -1



def _resolve_reference_arrays_from_onnx(refs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]
    for key in required:
        if key not in refs:
            raise KeyError(f"Main policy ONNX output missing reference key '{key}'.")

    joint_pos = np.asarray(refs["joint_pos"], dtype=np.float64).reshape(-1)
    joint_vel = np.asarray(refs["joint_vel"], dtype=np.float64).reshape(-1)

    body_pos = np.asarray(refs["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(refs["body_quat_w"], dtype=np.float64)

    if body_pos.ndim == 3:
        body_pos = body_pos[0]
    if body_quat.ndim == 3:
        body_quat = body_quat[0]
    if body_pos.ndim != 2 or body_quat.ndim != 2:
        raise ValueError("Invalid ONNX reference body tensors shape.")
    return joint_pos, joint_vel, body_pos, body_quat



def _prepare_reference_motion(
    motion_file: str,
    *,
    command_joint_dim: int,
    body_count: int,
    reference_joint_indices: list[int],
    reference_body_indices: list[int],
):
    data = np.load(motion_file)
    for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]:
        if key not in data:
            raise ValueError(f"Reference motion missing key '{key}': {motion_file}")

    ref_joint_pos_all = np.asarray(data["joint_pos"], dtype=np.float64)
    ref_joint_vel_all = np.asarray(data["joint_vel"], dtype=np.float64)
    ref_body_pos_all = np.asarray(data["body_pos_w"], dtype=np.float64)
    ref_body_quat_all = np.asarray(data["body_quat_w"], dtype=np.float64)

    if ref_joint_pos_all.ndim != 2 or ref_joint_vel_all.ndim != 2:
        raise ValueError("Reference joint arrays must be [T, J].")
    if ref_body_pos_all.ndim != 3 or ref_body_quat_all.ndim != 3:
        raise ValueError("Reference body arrays must be [T, B, D].")
    if ref_joint_pos_all.shape[0] <= 0:
        raise ValueError("Reference motion has zero frames.")

    if reference_joint_indices:
        if len(reference_joint_indices) != command_joint_dim:
            raise ValueError(
                "reference.joint_indices length mismatch with command joint dim: "
                f"{len(reference_joint_indices)} vs {command_joint_dim}"
            )
        ref_joint_pos = ref_joint_pos_all[:, reference_joint_indices]
        ref_joint_vel = ref_joint_vel_all[:, reference_joint_indices]
    elif ref_joint_pos_all.shape[1] == command_joint_dim:
        ref_joint_pos = ref_joint_pos_all
        ref_joint_vel = ref_joint_vel_all
    else:
        raise ValueError(
            "Reference joint dimension mismatch. Provide reference.joint_indices when motion order differs."
        )

    if reference_body_indices:
        if len(reference_body_indices) != body_count:
            raise ValueError(
                "reference.body_indices length mismatch with body_count: "
                f"{len(reference_body_indices)} vs {body_count}"
            )
        ref_body_pos = ref_body_pos_all[:, reference_body_indices]
        ref_body_quat = ref_body_quat_all[:, reference_body_indices]
    elif ref_body_pos_all.shape[1] == body_count:
        ref_body_pos = ref_body_pos_all
        ref_body_quat = ref_body_quat_all
    else:
        raise ValueError("Reference body count mismatch. Provide reference.body_indices when needed.")

    num_frames = int(ref_joint_pos.shape[0])

    def at_step(step: int):
        idx = int(np.clip(step, 0, num_frames - 1))
        return (
            ref_joint_pos[idx],
            ref_joint_vel[idx],
            ref_body_pos[idx],
            ref_body_quat[idx],
        )

    return num_frames, at_step



def _resolve_action_scale(
    scale_cfg: Any,
    *,
    action_specs: list[JointSpec],
    metadata_action_scale: list[float],
    metadata_action_joint_names: list[str],
) -> np.ndarray:
    action_dim = len(action_specs)
    train_names = [spec.train_name for spec in action_specs]

    metadata_scale = None
    if metadata_action_scale and len(metadata_action_scale) == action_dim:
        metadata_scale = np.asarray(metadata_action_scale, dtype=np.float64)
        if metadata_action_joint_names and len(metadata_action_joint_names) == action_dim:
            if metadata_action_joint_names != train_names:
                by_name = {n: metadata_scale[i] for i, n in enumerate(metadata_action_joint_names)}
                metadata_scale = np.asarray([by_name[n] for n in train_names], dtype=np.float64)

    if scale_cfg is None:
        if metadata_scale is None:
            raise ValueError("Missing action_scale: neither config nor ONNX metadata provides valid scale.")
        return metadata_scale

    if isinstance(scale_cfg, (int, float)):
        return np.full(action_dim, float(scale_cfg), dtype=np.float64)

    if isinstance(scale_cfg, list):
        if len(scale_cfg) != action_dim:
            raise ValueError(
                f"control.action_scale list length mismatch: {len(scale_cfg)} vs action_dim={action_dim}"
            )
        return np.asarray([float(x) for x in scale_cfg], dtype=np.float64)

    if isinstance(scale_cfg, dict):
        out: list[float] = []
        for i, spec in enumerate(action_specs):
            raw = scale_cfg.get(spec.train_name, scale_cfg.get(spec.mj_name, None))
            if raw is None:
                if metadata_scale is None:
                    raise ValueError(
                        f"control.action_scale is missing '{spec.train_name}' and no metadata fallback exists."
                    )
                out.append(float(metadata_scale[i]))
            else:
                out.append(float(raw))
        return np.asarray(out, dtype=np.float64)

    raise ValueError("control.action_scale must be float/list/dict.")



def _normalize_friction_triplet(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return np.asarray([v, v, v], dtype=np.float64)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            v = float(value[0])
            return np.asarray([v, v, v], dtype=np.float64)
        if len(value) >= 3:
            return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    return None


def _apply_mujoco_runtime_overrides(
    model: mujoco.MjModel,
    *,
    cfg: dict,
    action_specs: list[JointSpec],
) -> list[str]:
    notes: list[str] = []

    floor_friction_cfg = _cfg_pick(cfg, candidates=[("simulation", "floor_friction")], default=None)
    floor_triplet = _normalize_friction_triplet(floor_friction_cfg)
    if floor_triplet is not None:
        changed = 0
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_gid >= 0:
            model.geom_friction[floor_gid, :3] = floor_triplet
            changed = 1
        else:
            for gid in range(model.ngeom):
                is_world = int(model.geom_bodyid[gid]) == 0
                is_plane = int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                if is_world and is_plane:
                    model.geom_friction[gid, :3] = floor_triplet
                    changed += 1
        if changed > 0:
            notes.append(
                "floor_friction="
                f"[{floor_triplet[0]:.3f}, {floor_triplet[1]:.3f}, {floor_triplet[2]:.3f}]"
            )

    dof_damping_scale = float(_cfg_pick(cfg, candidates=[("control", "dof_damping_scale")], default=1.0) or 1.0)
    dof_frictionloss_scale = float(
        _cfg_pick(cfg, candidates=[("control", "dof_frictionloss_scale")], default=1.0) or 1.0
    )
    if abs(dof_damping_scale - 1.0) > 1.0e-12 or abs(dof_frictionloss_scale - 1.0) > 1.0e-12:
        for spec in action_specs:
            d = int(spec.dof_adr)
            if 0 <= d < int(model.nv):
                model.dof_damping[d] = float(model.dof_damping[d]) * dof_damping_scale
                model.dof_frictionloss[d] = float(model.dof_frictionloss[d]) * dof_frictionloss_scale
        notes.append(
            f"dof_damping_scale={dof_damping_scale:.3f} dof_frictionloss_scale={dof_frictionloss_scale:.3f}"
        )

    return notes


def run_with_config_file(config_file: str):
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    cfg = _load_dict_file(config_file)
    cfg = materialize_observation_block(cfg)
    config_base_dir = os.path.dirname(os.path.abspath(config_file))

    resources_cfg = cfg.get("resources", {}) if isinstance(cfg.get("resources"), dict) else {}
    robot_cfg = cfg.get("robot", {}) if isinstance(cfg.get("robot"), dict) else {}
    control_cfg = cfg.get("control", {}) if isinstance(cfg.get("control"), dict) else {}
    sim_cfg = cfg.get("simulation", {}) if isinstance(cfg.get("simulation"), dict) else {}
    mapping_cfg = cfg.get("mapping", {}) if isinstance(cfg.get("mapping"), dict) else {}
    logging_cfg = cfg.get("logging", {}) if isinstance(cfg.get("logging"), dict) else {}
    reference_cfg = cfg.get("reference", {}) if isinstance(cfg.get("reference"), dict) else {}

    policy_stack_cfg = cfg.get("policy_stack")
    if not isinstance(policy_stack_cfg, dict):
        sonic_cfg = cfg.get("sonic", {}) if isinstance(cfg.get("sonic"), dict) else {}
        policy_stack_cfg = sonic_cfg.get("policy_stack", {}) if isinstance(sonic_cfg.get("policy_stack"), dict) else {}

    # ---------------------------------------------------------------------
    # Policy stack and metadata
    # ---------------------------------------------------------------------
    onnx_path = str(resources_cfg.get("onnx_path", "")).strip()
    if not onnx_path:
        raise ValueError("resources.onnx_path is required")

    aux_stage_cfgs = build_aux_stage_configs(policy_stack_cfg, config_base_dir=config_base_dir)
    policy_stack = PolicyStack(main_onnx_path=onnx_path, aux_stage_cfgs=aux_stage_cfgs)
    main_policy: MainPolicy = policy_stack.main_policy
    main_meta = main_policy.metadata

    # ---------------------------------------------------------------------
    # Resolve observation selection / registry
    # ---------------------------------------------------------------------
    mode_id = policy_stack_cfg.get("mode_id", None)
    mode_name = policy_stack_cfg.get("mode_name", None)
    token_dim = int(_cfg_pick(cfg, candidates=[("observation", "token_dim")], default=0) or 0)

    action_output_name = main_policy.action_output_name
    action_output_dim = None
    for output_info in main_policy.model.session.get_outputs():
        if output_info.name == action_output_name:
            shape = list(output_info.shape or [])
            if shape:
                tail = shape[-1]
                if isinstance(tail, (int, np.integer)):
                    action_output_dim = int(tail)
            break
    if action_output_dim is None:
        raise ValueError(
            f"Unable to resolve action dimension from ONNX output '{action_output_name}'. "
            "Please export policy with static action shape."
        )
    action_dim = int(action_output_dim)

    prefer_onnx_joint_order = _parse_bool(mapping_cfg.get("prefer_onnx_joint_order", False), default=False)

    # fallback joints first (may be overwritten below)
    action_joint_names = _coerce_name_list(robot_cfg.get("action_joint_names"))
    action_joint_source = "config.robot.action_joint_names"
    if prefer_onnx_joint_order:
        if len(main_meta.action_joint_names) == action_dim:
            action_joint_names = list(main_meta.action_joint_names)
            action_joint_source = "onnx.action_joint_names"
        elif len(main_meta.joint_names) == action_dim:
            action_joint_names = list(main_meta.joint_names)
            action_joint_source = "onnx.joint_names"
    if not action_joint_names:
        action_joint_names = list(main_meta.action_joint_names or main_meta.joint_names)
        action_joint_source = "onnx.fallback"
    if len(action_joint_names) != action_dim:
        raise ValueError(
            f"Action joint count mismatch: got {len(action_joint_names)}, expected action_dim={action_dim}. "
            f"source={action_joint_source}"
        )

    command_joint_names = _coerce_name_list(robot_cfg.get("command_joint_names"))
    command_joint_source = "config.robot.command_joint_names"
    if not command_joint_names:
        command_joint_names = list(main_meta.command_joint_names or action_joint_names)
        command_joint_source = "onnx.fallback"

    policy_joint_names = _coerce_name_list(robot_cfg.get("policy_joint_names"))
    policy_joint_source = "config.robot.policy_joint_names"
    if prefer_onnx_joint_order:
        if len(main_meta.policy_joint_names) > 0:
            policy_joint_names = list(main_meta.policy_joint_names)
            policy_joint_source = "onnx.policy_joint_names"
        elif len(main_meta.joint_names) > 0:
            policy_joint_names = list(main_meta.joint_names)
            policy_joint_source = "onnx.joint_names"
    if not policy_joint_names:
        policy_joint_names = list(main_meta.policy_joint_names or action_joint_names)
        policy_joint_source = "onnx.fallback"

    body_names = _coerce_name_list(robot_cfg.get("body_names"))
    if not body_names:
        body_names = list(main_meta.body_names)

    obs_selection = resolve_observation_selection(
        cfg,
        onnx_observation_names=list(main_meta.observation_names),
        onnx_observation_history_lengths=list(main_meta.observation_history_lengths),
        default_pipeline=str(policy_stack_cfg.get("main_pipeline", "policy_main")).strip() or None,
        mode_id=int(mode_id) if mode_id is not None else None,
        mode_name=str(mode_name).strip() if mode_name is not None else None,
    )

    obs_ctx = ObservationShapeContext(
        action_dim=action_dim,
        command_joint_dim=len(command_joint_names),
        policy_joint_dim=len(policy_joint_names),
        body_count=len(body_names),
        token_dim=token_dim,
    )
    obs_registry = build_default_observation_registry(obs_ctx)

    plugin_file = str(
        _cfg_pick(
            cfg,
            candidates=[
                ("observation", "registry_plugin"),
                ("sonic", "observation_registry", "plugin_file"),
            ],
            default="",
        )
        or ""
    ).strip()
    if plugin_file:
        load_observation_registry_plugin(obs_registry, plugin_file)
        print(f"[INFO] Loaded observation registry plugin: {plugin_file}")

    # ---------------------------------------------------------------------
    # MuJoCo model
    # ---------------------------------------------------------------------
    mjcf_path = str(resources_cfg.get("mjcf_path", "")).strip()
    if not mjcf_path:
        raise ValueError("resources.mjcf_path is required")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Name mappings
    joint_name_map = _coerce_name_map(mapping_cfg.get("joint_name_map", {}))
    body_name_map = _coerce_name_map(mapping_cfg.get("body_name_map", {}))
    actuator_name_map = _coerce_name_map(mapping_cfg.get("actuator_name_map", {}))

    def map_joint(name: str) -> str:
        return joint_name_map.get(name, name)

    def map_body(name: str) -> str:
        return body_name_map.get(name, name)

    # Metadata maps for defaults/gains
    md_joint_names = list(main_meta.joint_names)
    md_default_pos = list(main_meta.default_joint_pos)
    md_default_vel = list(main_meta.default_joint_vel)
    md_kp = list(main_meta.joint_stiffness)
    md_kd = list(main_meta.joint_damping)

    default_pos_map: dict[str, float] = {}
    default_vel_map: dict[str, float] = {}
    kp_map: dict[str, float] = {}
    kd_map: dict[str, float] = {}
    for i, jn in enumerate(md_joint_names):
        if i < len(md_default_pos):
            default_pos_map[jn] = float(md_default_pos[i])
        if i < len(md_default_vel):
            default_vel_map[jn] = float(md_default_vel[i])
        if i < len(md_kp):
            kp_map[jn] = float(md_kp[i])
        if i < len(md_kd):
            kd_map[jn] = float(md_kd[i])

    cfg_default_angles = _coerce_float_map(robot_cfg.get("default_joint_angles", {}))
    cfg_joint_gains = control_cfg.get("joint_gains", {}) if isinstance(control_cfg.get("joint_gains"), dict) else {}

    def resolve_default_pos(train_name: str, mj_name: str) -> float:
        if train_name in cfg_default_angles:
            return float(cfg_default_angles[train_name])
        if mj_name in cfg_default_angles:
            return float(cfg_default_angles[mj_name])
        if train_name in default_pos_map:
            return float(default_pos_map[train_name])
        if mj_name in default_pos_map:
            return float(default_pos_map[mj_name])
        return 0.0

    def resolve_default_vel(train_name: str, mj_name: str) -> float:
        if train_name in default_vel_map:
            return float(default_vel_map[train_name])
        if mj_name in default_vel_map:
            return float(default_vel_map[mj_name])
        return 0.0

    def resolve_joint_gain(train_name: str, mj_name: str, key: str, md_map: dict[str, float], default: float) -> float:
        source = cfg_joint_gains.get(train_name, None)
        if source is None:
            source = cfg_joint_gains.get(mj_name, None)
        if isinstance(source, dict) and key in source:
            return float(source[key])
        if train_name in md_map:
            return float(md_map[train_name])
        if mj_name in md_map:
            return float(md_map[mj_name])
        return float(default)

    def resolve_effort_limit(train_name: str, mj_name: str) -> float:
        source = cfg_joint_gains.get(train_name, None)
        if source is None:
            source = cfg_joint_gains.get(mj_name, None)
        if isinstance(source, dict):
            for key in ["effort_limit", "effort_limit_sim", "limit"]:
                if key in source:
                    return float(source[key])
        return -1.0

    def build_joint_specs(train_names: list[str]) -> list[JointSpec]:
        specs: list[JointSpec] = []
        for train_name in train_names:
            mj_name = map_joint(train_name)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jid < 0:
                raise ValueError(f"Joint '{train_name}' mapped to '{mj_name}' but not found in MJCF.")
            qpos_adr = int(model.jnt_qposadr[jid])
            dof_adr = int(model.jnt_dofadr[jid])
            actuator_id = _find_actuator_id(model, mj_name, actuator_name_map)
            spec = JointSpec(
                train_name=train_name,
                mj_name=mj_name,
                joint_id=int(jid),
                qpos_adr=qpos_adr,
                dof_adr=dof_adr,
                actuator_id=int(actuator_id),
                default_pos=resolve_default_pos(train_name, mj_name),
                default_vel=resolve_default_vel(train_name, mj_name),
                kp=resolve_joint_gain(train_name, mj_name, "kp", kp_map, default=0.0),
                kd=resolve_joint_gain(train_name, mj_name, "kd", kd_map, default=0.0),
                effort_limit=resolve_effort_limit(train_name, mj_name),
            )
            specs.append(spec)
        return specs

    action_specs = build_joint_specs(action_joint_names)
    command_specs = build_joint_specs(command_joint_names)
    policy_specs = build_joint_specs(policy_joint_names)
    runtime_override_notes = _apply_mujoco_runtime_overrides(
        model,
        cfg=cfg,
        action_specs=action_specs,
    )

    # Body ids in motion order
    if not body_names:
        raise ValueError("body_names is empty. Provide robot.body_names or ONNX metadata body_names.")

    body_names_mj = [map_body(name) for name in body_names]
    body_ids = []
    for b in body_names_mj:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid < 0:
            raise ValueError(f"Body '{b}' not found in MJCF.")
        body_ids.append(int(bid))

    anchor_body_name_train = str(
        _cfg_pick(
            cfg,
            candidates=[("robot", "anchor_body_name")],
            default=main_meta.anchor_body_name,
        )
        or ""
    ).strip()
    if not anchor_body_name_train:
        anchor_body_name_train = body_names[0]
    anchor_body_name_mj = map_body(anchor_body_name_train)
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name_mj)
    if anchor_body_id < 0:
        raise ValueError(f"Anchor body '{anchor_body_name_mj}' not found in MJCF.")

    if anchor_body_name_train in body_names:
        motion_anchor_body_index = body_names.index(anchor_body_name_train)
    elif anchor_body_name_mj in body_names_mj:
        motion_anchor_body_index = body_names_mj.index(anchor_body_name_mj)
    else:
        raise ValueError(
            f"Anchor body '{anchor_body_name_train}' is not included in body_names: {body_names}"
        )

    root_body_name_train = str(
        _cfg_pick(
            cfg,
            candidates=[("robot", "root_body_name")],
            default=main_meta.root_body_name,
        )
        or ""
    ).strip()
    if not root_body_name_train:
        root_body_name_train = anchor_body_name_train
    root_body_name_mj = map_body(root_body_name_train)
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name_mj)
    if root_body_id < 0:
        raise ValueError(f"Root body '{root_body_name_mj}' not found in MJCF.")

    # Action scale
    action_scale = _resolve_action_scale(
        control_cfg.get("action_scale", None),
        action_specs=action_specs,
        metadata_action_scale=list(main_meta.action_scale),
        metadata_action_joint_names=list(main_meta.action_joint_names),
    )

    # ---------------------------------------------------------------------
    # Observation dimensions + assemblers
    # ---------------------------------------------------------------------
    term_dims = obs_registry.build_term_dimensions(obs_selection.observation_names)
    obs_dim_expected = sum(
        int(term_dims[name]) * int(hist)
        for name, hist in zip(obs_selection.observation_names, obs_selection.observation_history_lengths)
    )

    model_obs_dim = int(main_policy.model.session.get_inputs()[0].shape[-1])
    if model_obs_dim != obs_dim_expected:
        rows = [
            (name, int(term_dims[name]), int(hist))
            for name, hist in zip(obs_selection.observation_names, obs_selection.observation_history_lengths)
        ]
        raise ValueError(
            "Main policy observation dim mismatch.\n"
            f"model obs dim: {model_obs_dim}\n"
            f"resolved obs dim: {obs_dim_expected}\n"
            f"{render_term_table(rows)}"
        )

    main_obs_assembler = ObservationAssembler(
        observation_names=list(obs_selection.observation_names),
        observation_history_lengths=list(obs_selection.observation_history_lengths),
        term_dims=term_dims,
        obs_dim=model_obs_dim,
    )

    # Auxiliary stage assemblers
    aux_assemblers: dict[str, ObservationAssembler] = {}
    for stage in policy_stack.aux_stages:
        stage_selection = resolve_observation_selection(
            cfg,
            onnx_observation_names=[],
            onnx_observation_history_lengths=[],
            default_pipeline=stage.cfg.pipeline,
            mode_id=int(mode_id) if mode_id is not None else None,
            mode_name=str(mode_name).strip() if mode_name is not None else None,
        )
        stage_dims = obs_registry.build_term_dimensions(stage_selection.observation_names)
        stage_obs_dim = sum(
            int(stage_dims[name]) * int(hist)
            for name, hist in zip(stage_selection.observation_names, stage_selection.observation_history_lengths)
        )

        stage_input = stage.model.session.get_inputs()[0]
        stage_model_dim = int(stage_input.shape[-1])
        if stage_obs_dim != stage_model_dim:
            rows = [
                (name, int(stage_dims[name]), int(hist))
                for name, hist in zip(stage_selection.observation_names, stage_selection.observation_history_lengths)
            ]
            raise ValueError(
                f"Aux stage '{stage.cfg.name}' observation dim mismatch.\n"
                f"model obs dim: {stage_model_dim}\n"
                f"resolved obs dim: {stage_obs_dim}\n"
                f"{render_term_table(rows)}"
            )

        aux_assemblers[stage.cfg.name] = ObservationAssembler(
            observation_names=list(stage_selection.observation_names),
            observation_history_lengths=list(stage_selection.observation_history_lengths),
            term_dims=stage_dims,
            obs_dim=stage_model_dim,
        )

    # ---------------------------------------------------------------------
    # Timing / runtime setup
    # ---------------------------------------------------------------------
    sim_dt = float(model.opt.timestep)
    sim_dt_cfg = float(_cfg_pick(cfg, candidates=[("simulation", "sim_dt")], default=0.0) or 0.0)
    if sim_dt_cfg > 0.0 and abs(sim_dt_cfg - sim_dt) > 1.0e-12:
        model.opt.timestep = float(sim_dt_cfg)
        sim_dt = float(model.opt.timestep)

    control_dt_cfg = float(_cfg_pick(cfg, candidates=[("simulation", "control_dt")], default=0.0) or 0.0)
    if control_dt_cfg <= 0.0:
        control_dt_cfg = float(main_meta.control_dt) if float(main_meta.control_dt) > 0.0 else sim_dt

    sim_substeps = max(1, int(round(control_dt_cfg / sim_dt)))
    effective_control_dt = sim_substeps * sim_dt

    policy_dt_cfg = float(_cfg_pick(cfg, candidates=[("simulation", "policy_dt")], default=0.0) or 0.0)
    policy_dt = effective_control_dt if policy_dt_cfg <= 0.0 else policy_dt_cfg
    policy_every_n = max(1, int(round(policy_dt / effective_control_dt)))
    effective_policy_dt = policy_every_n * effective_control_dt

    num_steps = int(_cfg_pick(cfg, candidates=[("simulation", "num_steps")], default=0) or 0)
    duration_s = float(_cfg_pick(cfg, candidates=[("simulation", "duration_s")], default=0.0) or 0.0)
    if num_steps <= 0 and duration_s > 0.0:
        num_steps = int(math.ceil(duration_s / effective_control_dt))

    run_forever = _parse_bool(_cfg_pick(cfg, candidates=[("simulation", "run_forever")], default=False), False)
    if run_forever:
        num_steps = 0

    print_every = int(_cfg_pick(cfg, candidates=[("simulation", "print_every")], default=50) or 50)
    realtime = _parse_bool(_cfg_pick(cfg, candidates=[("simulation", "realtime")], default=True), True)
    realtime_factor = float(_cfg_pick(cfg, candidates=[("simulation", "realtime_factor")], default=1.0) or 1.0)

    render = _parse_bool(_cfg_pick(cfg, candidates=[("simulation", "render")], default=False), False)
    camera_follow_anchor = _parse_bool(
        _cfg_pick(cfg, candidates=[("simulation", "camera_follow_anchor")], default=True),
        True,
    )
    camera_lookat_height_offset = float(
        _cfg_pick(cfg, candidates=[("simulation", "camera_lookat_height_offset")], default=0.4) or 0.4
    )
    camera_distance_cfg = _cfg_pick(cfg, candidates=[("simulation", "camera_distance")], default=None)
    camera_azimuth_cfg = _cfg_pick(cfg, candidates=[("simulation", "camera_azimuth")], default=None)
    camera_elevation_cfg = _cfg_pick(cfg, candidates=[("simulation", "camera_elevation")], default=None)

    actuation_mode = str(_cfg_pick(cfg, candidates=[("control", "actuation_mode")], default="torque_pd")).strip().lower()
    if actuation_mode not in {"torque_pd", "position"}:
        raise ValueError("control.actuation_mode must be one of: torque_pd, position")

    velocity_order = str(_cfg_pick(cfg, candidates=[("control", "velocity_order")], default="ang_lin")).strip().lower()
    if velocity_order not in {"ang_lin", "lin_ang"}:
        raise ValueError("control.velocity_order must be 'ang_lin' or 'lin_ang'.")

    kp_scale = float(_cfg_pick(cfg, candidates=[("control", "kp_scale")], default=1.0) or 1.0)
    kd_scale = float(_cfg_pick(cfg, candidates=[("control", "kd_scale")], default=1.0) or 1.0)

    actuation_arrays = ActuationArrays(
        actuator_ids=np.asarray([spec.actuator_id for spec in action_specs], dtype=np.int64),
        qpos_adrs=np.asarray([spec.qpos_adr for spec in action_specs], dtype=np.int64),
        qvel_adrs=np.asarray([spec.dof_adr for spec in action_specs], dtype=np.int64),
        kp=kp_scale * np.asarray([spec.kp for spec in action_specs], dtype=np.float64),
        kd=kd_scale * np.asarray([spec.kd for spec in action_specs], dtype=np.float64),
        effort_limit=np.asarray([spec.effort_limit for spec in action_specs], dtype=np.float64),
    )

    viewer = launch_viewer_if_needed(render=render, model=model, data=data)
    runtime = MujocoRuntime(
        model=model,
        data=data,
        actuation=actuation_arrays,
        actuation_mode=actuation_mode,
        sim_substeps=sim_substeps,
        effective_control_dt=effective_control_dt,
        viewer=viewer,
        realtime=realtime,
        realtime_factor=realtime_factor,
    )

    def update_viewer_camera(anchor_pos_w: np.ndarray):
        if viewer is None or (not camera_follow_anchor):
            return
        try:
            # Force free-camera mode so lookat updates are always effective.
            viewer.cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
            viewer.cam.trackbodyid = -1
            lookat = np.asarray(anchor_pos_w, dtype=np.float64).copy()
            lookat[2] += float(camera_lookat_height_offset)
            viewer.cam.lookat[:] = lookat
            if camera_distance_cfg is not None:
                viewer.cam.distance = float(camera_distance_cfg)
            if camera_azimuth_cfg is not None:
                viewer.cam.azimuth = float(camera_azimuth_cfg)
            if camera_elevation_cfg is not None:
                viewer.cam.elevation = float(camera_elevation_cfg)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Reference source
    # ---------------------------------------------------------------------
    reference_source = str(_cfg_pick(cfg, candidates=[("reference", "source")], default="onnx")).strip().lower()
    reference_motion_file = str(
        _cfg_pick(cfg, candidates=[("reference", "motion_file"), ("resources", "motion_file")], default="")
        or ""
    ).strip()
    reference_joint_indices = _coerce_int_list(_cfg_pick(cfg, candidates=[("reference", "joint_indices")], default=[]))
    reference_body_indices = _coerce_int_list(_cfg_pick(cfg, candidates=[("reference", "body_indices")], default=[]))

    motion_num_frames = int(main_meta.motion_num_frames) if int(main_meta.motion_num_frames) > 0 else 0

    zero_obs_for_ref = np.zeros(model_obs_dim, dtype=np.float32)

    if reference_source == "motion_file":
        if not reference_motion_file:
            raise ValueError("reference.source=motion_file but reference.motion_file is empty.")
        motion_num_frames, ref_at_step = _prepare_reference_motion(
            reference_motion_file,
            command_joint_dim=len(command_specs),
            body_count=len(body_names),
            reference_joint_indices=reference_joint_indices,
            reference_body_indices=reference_body_indices,
        )
    else:

        def ref_at_step(step: int):
            _, refs = policy_stack.run_main_policy(obs_vec=zero_obs_for_ref, time_step=step)
            return _resolve_reference_arrays_from_onnx(refs)

        # Infer frame count from motion file when available.
        if motion_num_frames <= 0 and reference_motion_file and os.path.isfile(reference_motion_file):
            try:
                temp = np.load(reference_motion_file)
                if "joint_pos" in temp:
                    motion_num_frames = int(np.asarray(temp["joint_pos"]).shape[0])
            except Exception:
                pass

    no_motion_loop = _parse_bool(_cfg_pick(cfg, candidates=[("simulation", "no_motion_loop")], default=False), False)
    reset_on_motion_cycle = _parse_bool(
        _cfg_pick(cfg, candidates=[("simulation", "reset_on_motion_cycle")], default=False),
        False,
    )
    startup_pose_mode = str(_cfg_pick(cfg, candidates=[("simulation", "startup_pose_mode")], default="training_default")).strip().lower()
    cycle_reset_pose_mode = str(_cfg_pick(cfg, candidates=[("simulation", "cycle_reset_pose_mode")], default="training_default")).strip().lower()
    align_root_to_reference = _parse_bool(
        _cfg_pick(cfg, candidates=[("simulation", "align_root_to_reference")], default=True),
        True,
    )

    valid_pose_modes = {"reference_t0", "robot_init", "training_default"}
    if startup_pose_mode not in valid_pose_modes:
        raise ValueError(f"Invalid startup_pose_mode: {startup_pose_mode}")
    if cycle_reset_pose_mode not in valid_pose_modes:
        raise ValueError(f"Invalid cycle_reset_pose_mode: {cycle_reset_pose_mode}")

    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    initial_qvel = np.asarray(data.qvel, dtype=np.float64).copy()

    root_free_joint = None
    if model.njnt > 0:
        for jid in range(model.njnt):
            b = int(model.jnt_bodyid[jid])
            if b == int(root_body_id) and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                root_free_joint = jid
                break

    root_qpos_adr = int(model.jnt_qposadr[root_free_joint]) if root_free_joint is not None else -1

    def apply_pose_mode(mode: str):
        qpos = initial_qpos.copy()
        qvel = initial_qvel.copy()

        if mode in {"training_default", "robot_init"}:
            for spec in action_specs:
                qpos[spec.qpos_adr] = float(spec.default_pos)
                qvel[spec.dof_adr] = 0.0

        elif mode == "reference_t0":
            rjp, _, rbp, rbq = ref_at_step(0)
            # Apply selected command joints to robot action joints by index overlap.
            n = min(len(rjp), len(action_specs))
            for i in range(n):
                qpos[action_specs[i].qpos_adr] = float(rjp[i])
                qvel[action_specs[i].dof_adr] = 0.0

            if align_root_to_reference and root_qpos_adr >= 0 and rbp.shape[0] > 0 and rbq.shape[0] > 0:
                # Use first body as reference root (commonly anchor/root body).
                qpos[root_qpos_adr : root_qpos_adr + 3] = np.asarray(rbp[0], dtype=np.float64)
                qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = _normalize_quat_wxyz(np.asarray(rbq[0], dtype=np.float64))

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

    apply_pose_mode(startup_pose_mode)
    update_viewer_camera(np.asarray(data.xpos[anchor_body_id], dtype=np.float64))

    # ---------------------------------------------------------------------
    # SONIC FSM + I/O channels
    # ---------------------------------------------------------------------
    sonic_cfg = cfg.get("sonic", {}) if isinstance(cfg.get("sonic"), dict) else {}
    sonic_fsm_cfg = sonic_cfg.get("fsm", {}) if isinstance(sonic_cfg.get("fsm"), dict) else {}
    sonic_input_cfg = sonic_cfg.get("input", {}) if isinstance(sonic_cfg.get("input"), dict) else {}
    sonic_output_cfg = sonic_cfg.get("output", {}) if isinstance(sonic_cfg.get("output"), dict) else {}

    fsm = ProgramFSM.from_dict(sonic_fsm_cfg)
    input_channel = build_input_channel(sonic_input_cfg)
    output_channels = build_output_channels(sonic_output_cfg)

    # CSV logging
    save_csv = str(logging_cfg.get("save_csv", "")).strip()
    csv_file = None
    csv_writer = None
    if save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        csv_file = open(save_csv, "w", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "time_s", "joint_rmse", "anchor_pos_err", "anchor_ori_err", "action_l2"])

    # ---------------------------------------------------------------------
    # Runtime loop
    # ---------------------------------------------------------------------
    print(
        f"[INFO] sim2sim start | obs_dim={model_obs_dim} action_dim={action_dim} "
        f"control_dt={effective_control_dt:.6f}s sim_dt={sim_dt:.6f}s substeps={sim_substeps} "
        f"actuation_mode={actuation_mode} velocity_order={velocity_order} reference={reference_source}"
    )
    print(
        f"[INFO] timing | policy_dt={effective_policy_dt:.6f}s "
        f"(policy_every_n={policy_every_n}) realtime={realtime} realtime_factor={realtime_factor:.3f}"
    )
    print(f"[INFO] model | onnx={onnx_path} mjcf={mjcf_path}")
    if runtime_override_notes:
        print("[INFO] mujoco runtime overrides:")
        for note in runtime_override_notes:
            print(f"  - {note}")
    print(
        "[INFO] joint order source | "
        f"action={action_joint_source} command={command_joint_source} policy={policy_joint_source}"
    )
    print("[INFO] joint mapping (train -> mujoco):")
    for spec in action_specs:
        print(
            f"  {spec.train_name} -> {spec.mj_name} "
            f"(qpos_adr={spec.qpos_adr}, dof_adr={spec.dof_adr}, actuator_id={spec.actuator_id})"
        )

    rows = [
        (name, int(term_dims[name]), int(hist))
        for name, hist in zip(obs_selection.observation_names, obs_selection.observation_history_lengths)
    ]
    print("[INFO] observation terms:")
    print(render_term_table(rows))
    print(
        f"[INFO] rollout | run_forever={run_forever} reset_on_motion_cycle={reset_on_motion_cycle} "
        f"startup_pose_mode={startup_pose_mode} cycle_reset_pose_mode={cycle_reset_pose_mode} "
        f"align_root_to_reference={align_root_to_reference}"
    )
    if motion_num_frames > 0:
        print(f"[INFO] motion frames per cycle={motion_num_frames}")

    last_actions = np.zeros(action_dim, dtype=np.float64)

    metric_joint_rmse: list[float] = []
    metric_anchor_pos_err: list[float] = []
    metric_anchor_ori_err: list[float] = []
    metric_action_l2: list[float] = []

    wall_start = time.perf_counter()
    interrupted_by_user = False

    try:
        step = 0
        while num_steps <= 0 or step < num_steps:
            cmd = input_channel.poll(step=step, time_s=step * effective_control_dt)
            state = fsm.update(cmd=cmd, dt=effective_control_dt)
            if state == ProgramState.STOPPED:
                print("[INFO] FSM requested stop, terminating loop.")
                break
            if state in {ProgramState.INIT, ProgramState.WAIT_FOR_CONTROL}:
                hold_pos = np.asarray([data.qpos[spec.qpos_adr] for spec in action_specs], dtype=np.float64)
                runtime.apply_and_step(desired_joint_pos=hold_pos)
                update_viewer_camera(np.asarray(data.xpos[anchor_body_id], dtype=np.float64))
                if print_every > 0 and (step == 0 or ((step + 1) % print_every == 0)):
                    print(f"[fsm step {step + 1}] state={state.name.lower()}")
                runtime.sync_viewer(step=step, wall_start=wall_start)
                step += 1
                continue

            # Time-step over reference trajectory.
            if motion_num_frames > 0:
                if no_motion_loop:
                    time_step = min(step, motion_num_frames - 1)
                else:
                    time_step = step % motion_num_frames
            else:
                time_step = step

            if (
                reset_on_motion_cycle
                and not no_motion_loop
                and motion_num_frames > 0
                and step > 0
                and time_step == 0
            ):
                cycle_id = step // motion_num_frames
                print(f"[INFO] motion cycle {cycle_id} completed, resetting pose: {cycle_reset_pose_mode}")
                apply_pose_mode(cycle_reset_pose_mode)
                update_viewer_camera(np.asarray(data.xpos[anchor_body_id], dtype=np.float64))
                last_actions[:] = 0.0
                main_obs_assembler.clear()
                for asm in aux_assemblers.values():
                    asm.clear()

            # Reference targets.
            ref_joint_pos, ref_joint_vel, ref_body_pos_w, ref_body_quat_w = ref_at_step(time_step)
            motion_anchor_pos_w = np.asarray(ref_body_pos_w[motion_anchor_body_index], dtype=np.float64)
            motion_anchor_quat_w = _normalize_quat_wxyz(np.asarray(ref_body_quat_w[motion_anchor_body_index], dtype=np.float64))

            # Current robot state.
            robot_anchor_pos_w = np.asarray(data.xpos[anchor_body_id], dtype=np.float64)
            robot_anchor_quat_w = _normalize_quat_wxyz(np.asarray(data.xquat[anchor_body_id], dtype=np.float64))

            vel6_root_local = np.zeros(6, dtype=np.float64)
            vel6_anchor_world = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_body_id, vel6_root_local, 1)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, anchor_body_id, vel6_anchor_world, 0)
            base_lin_vel, base_ang_vel = _split_velocity6(vel6_root_local, velocity_order)
            robot_anchor_lin_vel_w, robot_anchor_ang_vel_w = _split_velocity6(vel6_anchor_world, velocity_order)

            policy_joint_pos = np.asarray([data.qpos[spec.qpos_adr] for spec in policy_specs], dtype=np.float64)
            policy_joint_vel = np.asarray([data.qvel[spec.dof_adr] for spec in policy_specs], dtype=np.float64)
            policy_default_pos = np.asarray([spec.default_pos for spec in policy_specs], dtype=np.float64)
            policy_default_vel = np.asarray([spec.default_vel for spec in policy_specs], dtype=np.float64)
            joint_pos_rel = policy_joint_pos - policy_default_pos
            joint_vel_rel = policy_joint_vel - policy_default_vel

            command_vec = np.concatenate([ref_joint_pos, ref_joint_vel], axis=0)
            motion_anchor_pos_b = _quat_rotate_inv_wxyz(robot_anchor_quat_w, motion_anchor_pos_w - robot_anchor_pos_w)
            motion_anchor_ori_b = _rotmat_first_two_columns_flat(
                _quat_mul_wxyz(_quat_inv_wxyz(robot_anchor_quat_w), motion_anchor_quat_w)
            )

            robot_body_pos_w = np.stack([np.asarray(data.xpos[i], dtype=np.float64) for i in body_ids], axis=0)
            robot_body_quat_w = np.stack(
                [_normalize_quat_wxyz(np.asarray(data.xquat[i], dtype=np.float64)) for i in body_ids],
                axis=0,
            )
            body_pos_b_list: list[np.ndarray] = []
            body_ori_b_list: list[np.ndarray] = []
            for pos_w, quat_w in zip(robot_body_pos_w, robot_body_quat_w):
                body_pos_b_list.append(_quat_rotate_inv_wxyz(robot_anchor_quat_w, pos_w - robot_anchor_pos_w))
                body_quat_b = _quat_mul_wxyz(_quat_inv_wxyz(robot_anchor_quat_w), quat_w)
                body_ori_b_list.append(_rotmat_first_two_columns_flat(body_quat_b))
            robot_body_pos_b = np.concatenate(body_pos_b_list, axis=0)
            robot_body_ori_b = np.concatenate(body_ori_b_list, axis=0)

            source_terms: dict[str, np.ndarray] = {
                "command": np.asarray(command_vec, dtype=np.float32),
                "motion_anchor_pos_b": np.asarray(motion_anchor_pos_b, dtype=np.float32),
                "motion_anchor_ori_b": np.asarray(motion_anchor_ori_b, dtype=np.float32),
                "base_lin_vel": np.asarray(base_lin_vel, dtype=np.float32),
                "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float32),
                "joint_pos": np.asarray(joint_pos_rel, dtype=np.float32),
                "joint_vel": np.asarray(joint_vel_rel, dtype=np.float32),
                "actions": np.asarray(last_actions, dtype=np.float32),
                "robot_anchor_ori_w": np.asarray(_rotmat_first_two_columns_flat(robot_anchor_quat_w), dtype=np.float32),
                "robot_anchor_lin_vel_w": np.asarray(robot_anchor_lin_vel_w, dtype=np.float32),
                "robot_anchor_ang_vel_w": np.asarray(robot_anchor_ang_vel_w, dtype=np.float32),
                "robot_body_pos_b": np.asarray(robot_body_pos_b, dtype=np.float32),
                "robot_body_ori_b": np.asarray(robot_body_ori_b, dtype=np.float32),
            }

            # Auxiliary models (if any).
            aux_traces: dict[str, dict[str, np.ndarray]] = {}
            if policy_stack.aux_stages:
                stage_obs: dict[str, np.ndarray] = {}
                for stage in policy_stack.aux_stages:
                    asm = aux_assemblers[stage.cfg.name]
                    stage_term_values = obs_registry.extract_terms(asm.observation_names, source_terms)
                    stage_obs[stage.cfg.name] = asm.build(stage_term_values)

                aux_outputs, aux_traces = policy_stack.run_aux_stages(stage_obs=stage_obs, time_step=time_step)
                for name, value in aux_outputs.items():
                    source_terms[name] = np.asarray(value, dtype=np.float32).reshape(-1)

            # Main policy
            term_values = obs_registry.extract_terms(main_obs_assembler.observation_names, source_terms)
            obs_vec = main_obs_assembler.build(term_values=term_values)

            if step % policy_every_n == 0:
                policy_action, _ = policy_stack.run_main_policy(obs_vec=obs_vec, time_step=time_step)
            else:
                policy_action = last_actions.copy()

            action_default_pos = np.asarray([spec.default_pos for spec in action_specs], dtype=np.float64)
            desired_joint_pos = action_default_pos + action_scale * np.asarray(policy_action, dtype=np.float64)
            runtime.apply_and_step(desired_joint_pos=desired_joint_pos)
            last_actions = np.asarray(policy_action, dtype=np.float64)

            # Metrics
            robot_anchor_pos_w = np.asarray(data.xpos[anchor_body_id], dtype=np.float64)
            robot_anchor_quat_w = _normalize_quat_wxyz(np.asarray(data.xquat[anchor_body_id], dtype=np.float64))
            anchor_pos_err = float(np.linalg.norm(motion_anchor_pos_w - robot_anchor_pos_w))
            anchor_ori_err = float(_quat_angle_error_rad(motion_anchor_quat_w, robot_anchor_quat_w))

            command_joint_pos = np.asarray([data.qpos[spec.qpos_adr] for spec in command_specs], dtype=np.float64)
            joint_rmse = float(np.sqrt(np.mean((command_joint_pos - ref_joint_pos) ** 2)))
            action_l2 = float(np.linalg.norm(policy_action))

            metric_joint_rmse.append(joint_rmse)
            metric_anchor_pos_err.append(anchor_pos_err)
            metric_anchor_ori_err.append(anchor_ori_err)
            metric_action_l2.append(action_l2)

            telemetry = TelemetryPacket(
                step=step,
                time_s=step * effective_control_dt,
                joint_rmse=joint_rmse,
                anchor_pos_err=anchor_pos_err,
                anchor_ori_err=anchor_ori_err,
                action_l2=action_l2,
                extras={
                    "fsm_state": fsm.state.name,
                    "time_step": int(time_step),
                    "aux_stage_count": int(len(aux_traces)),
                },
            )
            for channel in output_channels:
                channel.publish(telemetry)

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        step,
                        step * effective_control_dt,
                        joint_rmse,
                        anchor_pos_err,
                        anchor_ori_err,
                        action_l2,
                    ]
                )

            is_last = num_steps > 0 and (step + 1) == num_steps
            if print_every > 0 and ((step + 1) % print_every == 0 or step == 0 or is_last):
                total = "inf" if num_steps <= 0 else str(num_steps)
                print(
                    f"[step {step + 1:6d}/{total}] "
                    f"joint_rmse={joint_rmse:.5f} "
                    f"anchor_pos_err={anchor_pos_err:.5f}m "
                    f"anchor_ori_err={anchor_ori_err:.5f}rad "
                    f"action_l2={action_l2:.5f}"
                )

            update_viewer_camera(robot_anchor_pos_w)
            runtime.sync_viewer(step=step, wall_start=wall_start)
            step += 1

    except KeyboardInterrupt:
        interrupted_by_user = True
        print("[INFO] KeyboardInterrupt received, stopping sim2sim loop.")

    finally:
        try:
            input_channel.close()
        except Exception:
            pass
        for channel in output_channels:
            try:
                channel.close()
            except Exception:
                pass
        if csv_file is not None:
            csv_file.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    if metric_joint_rmse:
        mean_joint = float(np.mean(metric_joint_rmse))
        std_joint = float(np.std(metric_joint_rmse))
        mean_pos = float(np.mean(metric_anchor_pos_err))
        std_pos = float(np.std(metric_anchor_pos_err))
        mean_ori = float(np.mean(metric_anchor_ori_err))
        std_ori = float(np.std(metric_anchor_ori_err))
        mean_l2 = float(np.mean(metric_action_l2))
        std_l2 = float(np.std(metric_action_l2))
    else:
        mean_joint = std_joint = mean_pos = std_pos = mean_ori = std_ori = mean_l2 = std_l2 = float("nan")

    terminated_by = "keyboard_interrupt" if interrupted_by_user else ("max_steps" if num_steps > 0 else "fsm_or_external")
    steps_executed = len(metric_joint_rmse)

    print("\n[RESULT] sim2sim summary")
    print(f"  steps_executed      : {steps_executed}")
    print(f"  steps_configured    : {'infinite' if num_steps <= 0 else num_steps}")
    print(f"  terminated_by       : {terminated_by}")
    print(f"  effective_control_dt: {effective_control_dt:.6f} s")
    print(f"  effective_policy_dt : {effective_policy_dt:.6f} s")
    print(f"  joint_rmse          : {mean_joint:.6f} ± {std_joint:.6f} rad")
    print(f"  anchor_pos_err      : {mean_pos:.6f} ± {std_pos:.6f} m")
    print(f"  anchor_ori_err      : {mean_ori:.6f} ± {std_ori:.6f} rad")
    print(f"  action_l2           : {mean_l2:.6f} ± {std_l2:.6f}")
    if save_csv:
        print(f"  metrics_csv         : {save_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MuJoCo sim2sim rollout from config file.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to resolved config file.")
    args = parser.parse_args()
    run_with_config_file(args.config_file)
