#!/usr/bin/env python3
"""Main entry for MuJoCo sim2sim validation (config-driven)."""

from __future__ import annotations

from validate_pipeline import run_validation_from_config


def main() -> None:
    # Keep runtime entry config-driven. No CLI argument parsing here.
    run_validation_from_config(config_file=None, enforce_no_cli_args=True)


if __name__ == "__main__":
    main()
