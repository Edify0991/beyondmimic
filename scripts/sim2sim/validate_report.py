#!/usr/bin/env python3
"""Startup report utilities for sim2sim validation."""

from __future__ import annotations

from typing import Any

from validate_context import ValidationContext


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


def print_startup_report(context: ValidationContext) -> None:
    cfg = context.effective_config

    print(f"[INFO] Source config: {context.source_config_file}")
    print(f"[INFO] Effective config written to: {context.effective_config_file}")

    if context.training_snapshot_path:
        print(f"[INFO] Training alignment snapshot: {context.training_snapshot_path}")

    print(f"[INFO] Loaded config file: {context.effective_config_file}")

    onnx_path = _cfg_pick(cfg, candidates=[("resources", "onnx_path")], default="")
    mjcf_path = _cfg_pick(cfg, candidates=[("resources", "mjcf_path")], default="")
    motion_file = _cfg_pick(
        cfg,
        candidates=[("reference", "motion_file"), ("resources", "motion_file")],
        default="",
    )
    if onnx_path:
        print(f"[INFO] resources.onnx_path={onnx_path}")
    if mjcf_path:
        print(f"[INFO] resources.mjcf_path={mjcf_path}")
    if motion_file:
        print(f"[INFO] reference.motion_file={motion_file}")

    obs_cfg = cfg.get("observation", {}) if isinstance(cfg.get("observation"), dict) else {}
    obs_source = "inline"
    if isinstance(obs_cfg.get("config_file"), str) and obs_cfg.get("config_file", "").strip():
        obs_source = f"external:{obs_cfg['config_file']}"
    pipeline = str(obs_cfg.get("pipeline", "")).strip() or "(auto)"
    print(f"[INFO] observation config source={obs_source} pipeline={pipeline}")
