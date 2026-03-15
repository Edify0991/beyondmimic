#!/usr/bin/env python3
"""Frame-by-frame CSV segment selector for short-motion extraction.

Supports two interaction modes:
1) terminal-only frame stepping (default)
2) simulator-assisted visualization (`--visualize`) to inspect pose changes before
   marking segment start/end.

Output is a short CSV compatible with `scripts/csv_to_npz.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _clamp_frame(idx: int, total: int) -> int:
    return max(0, min(total - 1, idx))


def _print_frame_preview(data: np.ndarray, frame_idx: int) -> None:
    row = data[frame_idx]
    root_pos = row[:3]
    root_quat = row[3:7]
    print(
        f"[frame={frame_idx:06d}] root_pos={np.array2string(root_pos, precision=4)} "
        f"root_quat(xyzw)={np.array2string(root_quat, precision=4)}"
    )


def _save_segment(data: np.ndarray, start: int, end: int, out_csv: Path) -> None:
    if start > end:
        start, end = end, start
    seg = data[start : end + 1]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, seg, delimiter=",", fmt="%.10f")
    print(f"[INFO] saved segment: {out_csv} (frames {start}..{end}, count={len(seg)})")


def _interactive_commands_help() -> None:
    print("\n=== CSV Frame Segmenter ===")
    print("Commands:")
    print("  n                 -> next frame")
    print("  p                 -> previous frame")
    print("  f <k>             -> move +k frames")
    print("  b <k>             -> move -k frames")
    print("  j <idx>           -> jump to frame idx")
    print("  play <k>          -> autoplay next k frames")
    print("  s                 -> mark start = current")
    print("  e                 -> mark end = current")
    print("  i                 -> print info")
    print("  w                 -> write current [start,end] to output csv")
    print("  q                 -> quit")
    print("---------------------------")


def _parse_step_command(raw: str) -> tuple[str, int | None]:
    parts = raw.split()
    cmd = parts[0].lower()
    val = None
    if len(parts) == 2 and parts[1].lstrip("-").isdigit():
        val = int(parts[1])
    return cmd, val


def _interactive_loop_terminal(data: np.ndarray, out_csv: Path, init_start: int, init_end: int, step_size: int) -> None:
    total = data.shape[0]
    cur = _clamp_frame(init_start, total)
    start = _clamp_frame(init_start, total)
    end = _clamp_frame(init_end, total)

    _interactive_commands_help()
    _print_frame_preview(data, cur)
    print(f"[range] start={start}, end={end}, total={total}")

    while True:
        raw = input("segmenter> ").strip() or "n"
        cmd, val = _parse_step_command(raw)

        if cmd == "n":
            cur = _clamp_frame(cur + step_size, total)
            _print_frame_preview(data, cur)
        elif cmd == "p":
            cur = _clamp_frame(cur - step_size, total)
            _print_frame_preview(data, cur)
        elif cmd == "f" and val is not None:
            cur = _clamp_frame(cur + val, total)
            _print_frame_preview(data, cur)
        elif cmd == "b" and val is not None:
            cur = _clamp_frame(cur - val, total)
            _print_frame_preview(data, cur)
        elif cmd == "j" and val is not None:
            cur = _clamp_frame(val, total)
            _print_frame_preview(data, cur)
        elif cmd == "play" and val is not None:
            for _ in range(max(1, val)):
                cur = _clamp_frame(cur + step_size, total)
                _print_frame_preview(data, cur)
        elif cmd == "s":
            start = cur
            print(f"[INFO] start <- {start}")
        elif cmd == "e":
            end = cur
            print(f"[INFO] end <- {end}")
        elif cmd == "i":
            print(f"[info] current={cur}, start={start}, end={end}, segment_len={abs(end-start)+1}")
            _print_frame_preview(data, cur)
        elif cmd == "w":
            _save_segment(data, start, end, out_csv)
        elif cmd == "q":
            print("[INFO] quit without additional save.")
            break
        else:
            print("[WARN] unknown command. use n/p/f/b/j/play/s/e/i/w/q")


def _interactive_loop_with_sim(
    data: np.ndarray,
    out_csv: Path,
    init_start: int,
    init_end: int,
    step_size: int,
    app_extra_args: list[str],
) -> None:
    """Simulator-assisted frame stepping loop.

    The CSV format is assumed as:
    [root_pos_xyz, root_quat_xyzw, joint_pos...]
    """
    import torch
    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(app_extra_args)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(device=app_args.device)
    sim_cfg.dt = 1.0 / 50.0
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

    total = data.shape[0]
    cur = _clamp_frame(init_start, total)
    start = _clamp_frame(init_start, total)
    end = _clamp_frame(init_end, total)

    def _render_frame(frame_idx: int) -> None:
        row = torch.tensor(data[frame_idx], dtype=torch.float32, device=sim.device)[None, :]
        root_pos = row[:, :3]
        # CSV is xyzw, simulator expects wxyz
        root_quat = row[:, 3:7][:, [3, 0, 1, 2]]
        joint_pos = row[:, 7 : 7 + len(robot_joint_indexes)]

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_quat
        root_states[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_states)

        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[:, robot_joint_indexes] = joint_pos
        jv[:, robot_joint_indexes] = 0.0
        robot.write_joint_state_to_sim(jp, jv)

        sim.render()
        scene.update(sim.get_physics_dt())
        pos_lookat = root_states[0, :3].detach().cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

    _interactive_commands_help()
    _render_frame(cur)
    _print_frame_preview(data, cur)
    print(f"[range] start={start}, end={end}, total={total}")

    while simulation_app.is_running():
        raw = input("segmenter(sim)> ").strip() or "n"
        cmd, val = _parse_step_command(raw)

        if cmd == "n":
            cur = _clamp_frame(cur + step_size, total)
            _render_frame(cur)
            _print_frame_preview(data, cur)
        elif cmd == "p":
            cur = _clamp_frame(cur - step_size, total)
            _render_frame(cur)
            _print_frame_preview(data, cur)
        elif cmd == "f" and val is not None:
            cur = _clamp_frame(cur + val, total)
            _render_frame(cur)
            _print_frame_preview(data, cur)
        elif cmd == "b" and val is not None:
            cur = _clamp_frame(cur - val, total)
            _render_frame(cur)
            _print_frame_preview(data, cur)
        elif cmd == "j" and val is not None:
            cur = _clamp_frame(val, total)
            _render_frame(cur)
            _print_frame_preview(data, cur)
        elif cmd == "play" and val is not None:
            for _ in range(max(1, val)):
                if not simulation_app.is_running():
                    break
                cur = _clamp_frame(cur + step_size, total)
                _render_frame(cur)
                _print_frame_preview(data, cur)
        elif cmd == "s":
            start = cur
            print(f"[INFO] start <- {start}")
        elif cmd == "e":
            end = cur
            print(f"[INFO] end <- {end}")
        elif cmd == "i":
            print(f"[info] current={cur}, start={start}, end={end}, segment_len={abs(end-start)+1}")
            _print_frame_preview(data, cur)
        elif cmd == "w":
            _save_segment(data, start, end, out_csv)
        elif cmd == "q":
            print("[INFO] quit.")
            break
        else:
            print("[WARN] unknown command. use n/p/f/b/j/play/s/e/i/w/q")

    simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-by-frame CSV segment selector for short-motion extraction.")
    parser.add_argument("--input_csv", required=True, help="Input long motion csv.")
    parser.add_argument("--output_csv", default="outputs/motions/selected_segment.csv", help="Output short segment csv.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive frame stepping and marking.")
    parser.add_argument("--visualize", action="store_true", help="Use Isaac simulator visualization during stepping.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (0-based).")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (0-based, inclusive).")
    parser.add_argument("--step_size", type=int, default=1, help="Default step for n/p in interactive mode.")
    args, unknown = parser.parse_known_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    data = np.loadtxt(input_csv, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("CSV format looks invalid: expected 2D array with at least 8 columns.")

    total = data.shape[0]
    start = _clamp_frame(args.start_frame, total)
    end = _clamp_frame(total - 1 if args.end_frame < 0 else args.end_frame, total)
    out_csv = Path(args.output_csv)

    print(f"[INFO] loaded {input_csv}, frames={total}, dims={data.shape[1]}")

    if args.interactive and args.visualize:
        _interactive_loop_with_sim(data, out_csv, start, end, max(1, args.step_size), unknown)
    elif args.interactive:
        _interactive_loop_terminal(data, out_csv, start, end, max(1, args.step_size))
    else:
        _save_segment(data, start, end, out_csv)


if __name__ == "__main__":
    main()
