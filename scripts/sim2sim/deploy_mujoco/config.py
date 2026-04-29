#!/usr/bin/env python3
"""Deploy entry config for MuJoCo sim2sim."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import runpy
from typing import Any


DEFAULT_DEPLOY_ENV = "SIM2SIM_DEPLOY_MUJOCO_CFG"
DEFAULT_DEPLOY_CFG = "scripts/sim2sim/deploy_mujoco/config/mujoco.yaml"


def _load_dict_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for deploy YAML files.") from exc
        data = yaml.safe_load(text)
    elif suffix == ".py":
        namespace = runpy.run_path(path)
        data = namespace.get("CONFIG", namespace.get("config"))
        if data is None:
            raise ValueError("Python deploy config must define CONFIG (dict).")
    else:
        data = json.loads(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Deploy config root must be dict/object.")
    return data


def _resolve_path(path: str, base_dir: str) -> str:
    raw = os.path.expanduser(path)
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(base_dir, raw))


@dataclass(frozen=True)
class DeployEntryConfig:
    deploy_config_file: str
    sim2sim_config_file: str
    raw: dict[str, Any]


def load_deploy_entry_config(config_file: str | None = None) -> DeployEntryConfig:
    if isinstance(config_file, str) and config_file.strip():
        raw_path = config_file.strip()
    else:
        raw_path = os.getenv(DEFAULT_DEPLOY_ENV, "").strip() or DEFAULT_DEPLOY_CFG

    deploy_cfg_path = os.path.abspath(os.path.expanduser(raw_path))
    if not os.path.isfile(deploy_cfg_path):
        raise FileNotFoundError(
            f"Deploy config file not found: {deploy_cfg_path}. "
            f"Set {DEFAULT_DEPLOY_ENV} or create {DEFAULT_DEPLOY_CFG}."
        )

    deploy_cfg = _load_dict_file(deploy_cfg_path)
    base_dir = os.path.dirname(deploy_cfg_path)

    sim2sim_cfg = ""
    block = deploy_cfg.get("sim2sim")
    if isinstance(block, dict):
        sim2sim_cfg = str(block.get("config_file", "")).strip()
    if not sim2sim_cfg:
        sim2sim_cfg = str(deploy_cfg.get("sim2sim_config_file", deploy_cfg.get("config_file", ""))).strip()
    if not sim2sim_cfg:
        raise ValueError("Deploy config must provide sim2sim.config_file (or sim2sim_config_file).")

    resolved_sim2sim_cfg = _resolve_path(sim2sim_cfg, base_dir)

    return DeployEntryConfig(
        deploy_config_file=deploy_cfg_path,
        sim2sim_config_file=resolved_sim2sim_cfg,
        raw=deploy_cfg,
    )
