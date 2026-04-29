#!/usr/bin/env python3
"""RoboMimic-style deploy entry for SONIC-structured MuJoCo sim2sim."""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM2SIM_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SIM2SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM2SIM_DIR)

from validate_pipeline import run_validation_from_config

from config import DEFAULT_DEPLOY_ENV, load_deploy_entry_config


def _guard_no_cli_args() -> None:
    if len(sys.argv) <= 1:
        return
    extras = " ".join(sys.argv[1:])
    raise ValueError(
        "deploy_mujoco.py does not parse runtime CLI args anymore; "
        f"received: {extras}. Configure YAML and use env {DEFAULT_DEPLOY_ENV} when needed."
    )


def main() -> None:
    _guard_no_cli_args()

    entry_cfg = load_deploy_entry_config(config_file=None)
    print(f"[INFO] deploy config: {entry_cfg.deploy_config_file}")
    print(f"[INFO] sim2sim config: {entry_cfg.sim2sim_config_file}")

    # Keep compatibility with validate entry's env fallback behavior.
    os.environ["SIM2SIM_CONFIG_FILE"] = entry_cfg.sim2sim_config_file
    run_validation_from_config(config_file=entry_cfg.sim2sim_config_file, enforce_no_cli_args=False)


if __name__ == "__main__":
    main()
