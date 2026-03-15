#!/usr/bin/env python3
"""Wrapper around rsl_rl train entrypoint enabling compliance logging flags."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train base tracking policy with optional compliance logging.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--registry_name", required=True)
    parser.add_argument("--log_compliance_windows", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args forwarded to scripts/rsl_rl/train.py")
    args = parser.parse_args()

    cmd = [sys.executable, "scripts/rsl_rl/train.py", "--task", args.task, "--registry_name", args.registry_name]
    if args.log_compliance_windows:
        cmd += ["--", "env.compliance.log_buffers=True"]
    if args.extra:
        cmd.extend(args.extra)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
