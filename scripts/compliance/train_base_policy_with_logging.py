#!/usr/bin/env python3
"""Train/replay base policy with compliance rollout logging enabled."""

from __future__ import annotations

import argparse
import subprocess
import sys


def _csv(text: str) -> str:
    return ",".join([x.strip() for x in text.split(",") if x.strip()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or rollout base policy with compliance logging.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--registry_name", default=None)
    parser.add_argument("--wandb_path", default=None)
    parser.add_argument("--motion_file", default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--episode_length_s", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save_dir", default="outputs/rollouts/default")
    parser.add_argument("--payload_body_names", default="")
    parser.add_argument("--payload_site_names", default="")
    parser.add_argument("--train_mode", action="store_true")
    parser.add_argument("--rollout_mode", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.train_mode == args.rollout_mode:
        raise ValueError("Select exactly one mode: --train_mode or --rollout_mode")

    if args.train_mode:
        cmd = [sys.executable, "scripts/rsl_rl/train.py", "--task", args.task, "--num_envs", str(args.num_envs)]
        if args.registry_name:
            cmd += ["--registry_name", args.registry_name]
        if args.motion_file:
            cmd += ["--motion_file", args.motion_file]
        if args.headless:
            cmd += ["--headless"]
        # Hydra overrides
        cmd += ["--", "env.compliance.log_buffers=True", f"env.episode_length_s={args.episode_length_s}"]
    else:
        cmd = [
            sys.executable,
            "scripts/rsl_rl/play.py",
            "--task",
            args.task,
            "--num_envs",
            str(args.num_envs),
            "--enable_compliance_plugin",
            "--compliance_log_rollouts",
            "--compliance_save_dir",
            args.save_dir,
            "--payload_body_names",
            _csv(args.payload_body_names),
            "--payload_site_names",
            _csv(args.payload_site_names),
        ]
        if args.wandb_path:
            cmd += ["--wandb_path", args.wandb_path]
        if args.motion_file:
            cmd += ["--motion_file", args.motion_file]
        if args.max_steps > 0:
            cmd += ["--max_steps", str(args.max_steps)]
        if args.headless:
            cmd += ["--headless"]

    if args.extra:
        cmd.extend(args.extra)

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
