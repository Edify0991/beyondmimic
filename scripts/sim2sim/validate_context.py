#!/usr/bin/env python3
"""Validation context builder for SONIC-style sim2sim pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any

from sim2sim_config import (
    DEFAULT_CONFIG_ENV,
    DEFAULT_CONFIG_PATH,
    prepare_effective_config_file,
)


@dataclass(frozen=True)
class ValidationContext:
    source_config_file: str
    effective_config_file: str
    source_config: dict[str, Any]
    effective_config: dict[str, Any]

    @property
    def training_snapshot_path(self) -> str:
        align_cfg = self.effective_config.get("training_alignment")
        if not isinstance(align_cfg, dict):
            return ""
        path = str(align_cfg.get("resolved_env_snapshot_path", "")).strip()
        if path:
            return path
        return str(align_cfg.get("env_snapshot_path", "")).strip()


def resolve_requested_config_path(config_file: str | None = None) -> str:
    if isinstance(config_file, str) and config_file.strip():
        raw = config_file.strip()
    else:
        raw = os.getenv(DEFAULT_CONFIG_ENV, "").strip() or DEFAULT_CONFIG_PATH
    return os.path.abspath(os.path.expanduser(raw))


def guard_no_cli_args(argv: list[str] | None = None, *, script_name: str = "validate_policy_mujoco.py") -> None:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) <= 1:
        return
    extras = " ".join(argv[1:])
    raise ValueError(
        f"{script_name} no longer parses runtime CLI arguments; received: {extras}\n"
        f"Please configure YAML and optionally set {DEFAULT_CONFIG_ENV}=<path>."
    )


def build_validation_context(config_file: str | None = None) -> ValidationContext:
    effective_file, cfg, source_cfg_file = prepare_effective_config_file(config_file)
    return ValidationContext(
        source_config_file=source_cfg_file,
        effective_config_file=effective_file,
        source_config=dict(cfg),
        effective_config=dict(cfg),
    )
