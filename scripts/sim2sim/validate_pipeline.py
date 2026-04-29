#!/usr/bin/env python3
"""Orchestration pipeline for MuJoCo sim2sim validation."""

from __future__ import annotations

from sim2sim_rollout import run_with_config_file
from validate_context import build_validation_context, guard_no_cli_args
from validate_report import print_startup_report


def run_validation_from_config(config_file: str | None = None, *, enforce_no_cli_args: bool = False):
    if enforce_no_cli_args:
        guard_no_cli_args()

    context = build_validation_context(config_file=config_file)
    print_startup_report(context)
    run_with_config_file(context.effective_config_file)
    return context
