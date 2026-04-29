#!/usr/bin/env python3
"""Config loading and training-alignment utilities for sim2sim."""

from __future__ import annotations

import json
import os
import re
import runpy
import time
from typing import Any


DEFAULT_CONFIG_ENV = "SIM2SIM_CONFIG_FILE"
DEFAULT_CONFIG_PATH = "scripts/sim2sim/jingchu01_deploy.yaml"



def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False



def _nested_get(cfg: dict, path: tuple[str, ...]):
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur



def _cfg_pick(cfg: dict, candidates: list[tuple[str, ...] | str], default=None):
    for item in candidates:
        path = (item,) if isinstance(item, str) else tuple(item)
        value = _nested_get(cfg, path)
        if not _is_missing(value):
            return value
    return default



def _resolve_path(path: str, base_dirs: list[str | None]) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    for base in base_dirs:
        if not base:
            continue
        candidate = os.path.abspath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)



def _load_dict_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    data: Any = None

    if suffix == ".py":
        namespace = runpy.run_path(path)
        for key in ["CONFIG", "config", "MAPPING", "mapping"]:
            value = namespace.get(key)
            if isinstance(value, dict):
                data = value
                break
        if data is None:
            raise ValueError(
                "Python config file must define a dict named one of: CONFIG/config/MAPPING/mapping."
            )
    elif suffix == ".json":
        data = json.loads(text)
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for .yaml config files.") from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except Exception:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ValueError("Unknown config extension. Use .yaml/.yml/.json/.py.") from exc
            data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object/dict.")
    return data



def _resolve_config_path(config_file: str | None) -> str:
    if config_file and config_file.strip():
        path = config_file.strip()
    else:
        env_path = os.getenv(DEFAULT_CONFIG_ENV, "").strip()
        path = env_path if env_path else DEFAULT_CONFIG_PATH
    return os.path.abspath(os.path.expanduser(path))



def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["1", "true", "yes", "y", "on"]:
            return True
        if v in ["0", "false", "no", "n", "off", ""]:
            return False
    return default



def _ensure_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []



def _regex_match(pattern: str, name: str) -> bool:
    if pattern == name:
        return True
    try:
        return re.fullmatch(pattern, name) is not None
    except re.error:
        return False



def _resolve_pattern_value(value_obj: Any, name: str, default: float | None = None) -> float | None:
    if value_obj is None:
        return default
    if isinstance(value_obj, (int, float)):
        return float(value_obj)
    if isinstance(value_obj, dict):
        if name in value_obj:
            return float(value_obj[name])
        for key, value in value_obj.items():
            key_str = str(key)
            if _regex_match(key_str, name):
                return float(value)
        return default
    return default



def _set_if_missing_or_forced(container: dict, key: str, value: Any, force: bool):
    if force or _is_missing(container.get(key)):
        container[key] = value



def _load_training_env_snapshot(path: str) -> dict:
    text = open(path, "r", encoding="utf-8").read().strip()
    safe_globals = {"__builtins__": {}}
    safe_locals = {"slice": slice}
    data = eval(text, safe_globals, safe_locals)  # noqa: S307
    if isinstance(data, str):
        data = eval(data, safe_globals, safe_locals)  # noqa: S307
    if not isinstance(data, dict):
        raise ValueError("Training env snapshot must evaluate to a dict.")
    return data



def _apply_training_alignment_from_snapshot(cfg: dict, base_dir: str) -> dict:
    align_cfg = cfg.get("training_alignment", {})
    if not isinstance(align_cfg, dict):
        return cfg
    if not _to_bool(align_cfg.get("enabled", False), default=False):
        return cfg

    env_snapshot_path = str(align_cfg.get("env_snapshot_path", "")).strip()
    strict = _to_bool(align_cfg.get("strict", True), default=True)
    force_override = _to_bool(align_cfg.get("force_override", True), default=True)
    align_sim_dt = _to_bool(align_cfg.get("align_sim_dt", False), default=False)

    if not env_snapshot_path:
        if strict:
            raise ValueError("training_alignment.enabled=true but env_snapshot_path is empty.")
        return cfg

    env_snapshot_path = _resolve_path(env_snapshot_path, [base_dir])
    if not os.path.isfile(env_snapshot_path):
        if strict:
            raise FileNotFoundError(f"training env snapshot not found: {env_snapshot_path}")
        return cfg

    snapshot = _load_training_env_snapshot(env_snapshot_path)

    robot_cfg = cfg.setdefault("robot", {})
    sim_cfg = cfg.setdefault("simulation", {})
    control_cfg = cfg.setdefault("control", {})

    sim_dt = _cfg_pick(snapshot, candidates=[("sim", "dt")], default=None)
    decimation = snapshot.get("decimation", None)
    if sim_dt is not None and decimation is not None:
        sim_dt = float(sim_dt)
        decimation = int(decimation)
        control_dt = sim_dt * decimation
        if align_sim_dt:
            _set_if_missing_or_forced(sim_cfg, "sim_dt", sim_dt, force_override)
        _set_if_missing_or_forced(sim_cfg, "control_dt", control_dt, force_override)
        _set_if_missing_or_forced(sim_cfg, "policy_dt", control_dt, force_override)

    train_joint_names = _ensure_list_str(_cfg_pick(snapshot, candidates=[("actions", "joint_pos", "joint_names")], default=[]))
    action_joint_preserve_order = _to_bool(
        _cfg_pick(snapshot, candidates=[("actions", "joint_pos", "preserve_order")], default=False),
        default=False,
    )
    policy_joint_preserve_order = _to_bool(
        _cfg_pick(
            snapshot,
            candidates=[("observations", "policy", "joint_pos", "params", "asset_cfg", "preserve_order")],
            default=False,
        ),
        default=False,
    )

    _set_if_missing_or_forced(robot_cfg, "action_joint_preserve_order", action_joint_preserve_order, force_override)
    _set_if_missing_or_forced(robot_cfg, "policy_joint_preserve_order", policy_joint_preserve_order, force_override)

    mapping_cfg = cfg.setdefault("mapping", {})
    if (not action_joint_preserve_order) or (not policy_joint_preserve_order):
        _set_if_missing_or_forced(mapping_cfg, "prefer_onnx_joint_order", True, force_override)

    if train_joint_names:
        _set_if_missing_or_forced(robot_cfg, "command_joint_names", list(train_joint_names), force_override)
        if force_override or action_joint_preserve_order:
            _set_if_missing_or_forced(robot_cfg, "action_joint_names", list(train_joint_names), force_override)
        if force_override or policy_joint_preserve_order:
            _set_if_missing_or_forced(robot_cfg, "policy_joint_names", list(train_joint_names), force_override)

    init_joint_pos = _cfg_pick(snapshot, candidates=[("scene", "robot", "init_state", "joint_pos")], default={})
    action_joint_names = _ensure_list_str(robot_cfg.get("action_joint_names", []))
    if not action_joint_names and train_joint_names:
        action_joint_names = list(train_joint_names)

    if isinstance(init_joint_pos, dict) and action_joint_names:
        default_joint_angles: dict[str, float] = {}
        for name in action_joint_names:
            value = _resolve_pattern_value(init_joint_pos, name, default=0.0)
            default_joint_angles[name] = float(0.0 if value is None else value)
        _set_if_missing_or_forced(robot_cfg, "default_joint_angles", default_joint_angles, force_override)

    joint_gains: dict[str, dict[str, float]] = {}
    actuators = _cfg_pick(snapshot, candidates=[("scene", "robot", "actuators")], default={})
    if isinstance(actuators, dict) and action_joint_names:
        for joint_name in action_joint_names:
            kp = None
            kd = None
            effort_limit = None
            for actuator in actuators.values():
                if not isinstance(actuator, dict):
                    continue
                exprs = _ensure_list_str(actuator.get("joint_names_expr", []))
                if exprs and not any(_regex_match(expr, joint_name) for expr in exprs):
                    continue
                kp = _resolve_pattern_value(actuator.get("stiffness"), joint_name, default=kp)
                kd = _resolve_pattern_value(actuator.get("damping"), joint_name, default=kd)
                effort_limit = _resolve_pattern_value(
                    actuator.get("effort_limit_sim", actuator.get("effort_limit")),
                    joint_name,
                    default=effort_limit,
                )

            if kp is None and kd is None and effort_limit is None:
                continue
            joint_gains[joint_name] = {}
            if kp is not None:
                joint_gains[joint_name]["kp"] = float(kp)
            if kd is not None:
                joint_gains[joint_name]["kd"] = float(kd)
            if effort_limit is not None:
                joint_gains[joint_name]["effort_limit"] = float(effort_limit)

    if joint_gains:
        _set_if_missing_or_forced(control_cfg, "joint_gains", joint_gains, force_override)

    align_action_scale = _to_bool(align_cfg.get("align_action_scale", True), default=True)
    if align_action_scale:
        action_scale_cfg = _cfg_pick(snapshot, candidates=[("actions", "joint_pos", "scale")], default=None)
        if action_joint_names and action_scale_cfg is not None:
            action_scale_by_joint: dict[str, float] = {}
            for name in action_joint_names:
                value = _resolve_pattern_value(action_scale_cfg, name, default=None)
                if value is not None:
                    action_scale_by_joint[name] = float(value)
            if len(action_scale_by_joint) == len(action_joint_names):
                _set_if_missing_or_forced(control_cfg, "action_scale", action_scale_by_joint, force_override)

    if _to_bool(align_cfg.get("align_control_mode", True), default=True):
        _set_if_missing_or_forced(control_cfg, "actuation_mode", "torque_pd", force_override)

    align_cfg["resolved_env_snapshot_path"] = env_snapshot_path
    cfg["training_alignment"] = align_cfg
    return cfg



def _inject_joint_name_map_from_order(cfg: dict):
    robot_cfg = cfg.setdefault("robot", {})
    mapping_cfg = cfg.setdefault("mapping", {})

    action_joint_names = _ensure_list_str(robot_cfg.get("action_joint_names", []))
    explicit_map = mapping_cfg.get("joint_name_map", {})
    if not isinstance(explicit_map, dict):
        explicit_map = {}

    mj_names_in_train_order = _ensure_list_str(mapping_cfg.get("mujoco_joint_names_in_train_order", []))

    if mj_names_in_train_order:
        if len(action_joint_names) != len(mj_names_in_train_order):
            raise ValueError(
                "mapping.mujoco_joint_names_in_train_order length must match robot.action_joint_names: "
                f"{len(mj_names_in_train_order)} vs {len(action_joint_names)}"
            )
        auto_map = {
            train_name: mj_name
            for train_name, mj_name in zip(action_joint_names, mj_names_in_train_order)
        }
        auto_map.update({str(k): str(v) for k, v in explicit_map.items()})
        mapping_cfg["joint_name_map"] = auto_map
    elif action_joint_names:
        auto_map = {name: name for name in action_joint_names}
        auto_map.update({str(k): str(v) for k, v in explicit_map.items()})
        mapping_cfg["joint_name_map"] = auto_map

    if "body_name_map" not in mapping_cfg or not isinstance(mapping_cfg.get("body_name_map"), dict):
        mapping_cfg["body_name_map"] = {}

    cfg["mapping"] = mapping_cfg



def _resolve_cfg_paths(cfg: dict, base_dir: str):
    path_fields: list[tuple[str, ...]] = [
        ("resources", "onnx_path"),
        ("resources", "mjcf_path"),
        ("resources", "motion_file"),
        ("reference", "motion_file"),
        ("logging", "save_csv"),
        ("training_alignment", "env_snapshot_path"),
        ("observation", "registry_plugin"),
        ("observation", "config_file"),
        ("sonic", "observation_registry", "plugin_file"),
    ]

    for path in path_fields:
        parent: Any = cfg
        for key in path[:-1]:
            node = parent.get(key)
            if not isinstance(node, dict):
                parent = None
                break
            parent = node
        if parent is None:
            continue

        leaf = path[-1]
        raw_value = parent.get(leaf)
        if isinstance(raw_value, str) and raw_value.strip():
            parent[leaf] = _resolve_path(raw_value, [base_dir])

    policy_stack_cfg = cfg.get("policy_stack")
    if not isinstance(policy_stack_cfg, dict):
        policy_stack_cfg = cfg.get("sonic", {}).get("policy_stack") if isinstance(cfg.get("sonic"), dict) else None
    if isinstance(policy_stack_cfg, dict):
        stages_raw = policy_stack_cfg.get("aux_models", policy_stack_cfg.get("stages", []))
        if isinstance(stages_raw, list):
            for item in stages_raw:
                if not isinstance(item, dict):
                    continue
                onnx_path = item.get("onnx_path")
                if isinstance(onnx_path, str) and onnx_path.strip():
                    item["onnx_path"] = _resolve_path(onnx_path, [base_dir])



def _validate_required_paths(cfg: dict):
    onnx_path = _cfg_pick(cfg, candidates=[("resources", "onnx_path"), "onnx_path"], default="")
    mjcf_path = _cfg_pick(cfg, candidates=[("resources", "mjcf_path"), "mjcf_path"], default="")

    if not isinstance(onnx_path, str) or not onnx_path.strip():
        raise ValueError("Missing required field: resources.onnx_path")
    if not isinstance(mjcf_path, str) or not mjcf_path.strip():
        raise ValueError("Missing required field: resources.mjcf_path")



def load_effective_config(config_file: str | None = None) -> tuple[dict, str]:
    config_path = _resolve_config_path(config_file)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Set env {DEFAULT_CONFIG_ENV} or create {DEFAULT_CONFIG_PATH}."
        )

    cfg = _load_dict_file(config_path)
    base_dir = os.path.dirname(config_path)

    cfg = _apply_training_alignment_from_snapshot(cfg, base_dir)
    _inject_joint_name_map_from_order(cfg)
    _resolve_cfg_paths(cfg, base_dir)
    _validate_required_paths(cfg)

    return cfg, config_path



def write_effective_config_file(cfg: dict, config_path: str) -> str:
    stamp = int(time.time())
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    out_path = os.path.abspath(f"/tmp/{cfg_name}_effective_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return out_path



def prepare_effective_config_file(config_file: str | None = None) -> tuple[str, dict, str]:
    cfg, config_path = load_effective_config(config_file=config_file)
    effective_path = write_effective_config_file(cfg, config_path)
    return effective_path, cfg, config_path
