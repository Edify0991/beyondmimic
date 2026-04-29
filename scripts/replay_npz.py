"""Replay converted motions and optionally trim a frame range interactively.

Default behavior remains unchanged: loop-play a motion npz.

Examples:

    python scripts/replay_npz.py --motion_file data/motions/lafan_walk_short.npz --robot g1

Interactive frame-by-frame trimming:

    python scripts/replay_npz.py \
        --robot jingchu01 \
        --motion_file data/motions/jingchu01_walk1_subject1_leg.npz \
        --interactive_trim
"""

# Launch Isaac Sim Simulator first.

from __future__ import annotations

import argparse
import pathlib
import select
import sys
import termios
import tty

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion .npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="WandB registry name (optional).")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    choices=["g1", "jingchu01"],
    help="Robot profile used to spawn articulation during replay.",
)

# Optional interactive trimming controls (additive; default replay behavior is unchanged)
parser.add_argument(
    "--interactive_trim",
    action="store_true",
    default=False,
    help="Enable keyboard frame-by-frame replay and save selected [start, end] frames to a new npz.",
)
parser.add_argument(
    "--interactive_play",
    action="store_true",
    default=False,
    help="When --interactive_trim is on, start in auto-play instead of paused mode.",
)
parser.add_argument(
    "--trim_output_dir",
    type=str,
    default="data/motions",
    help="Output directory for trimmed npz in interactive mode.",
)
parser.add_argument(
    "--trim_output_name",
    type=str,
    default=None,
    help="Output file name (with/without .npz). Default: <input_stem>_seg<start>-<end>.npz",
)
parser.add_argument(
    "--trim_overwrite",
    action="store_true",
    default=False,
    help="Allow overwriting existing trimmed npz output file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.robot_info import get_robot_profile

ROBOT_PROFILE = get_robot_profile(args_cli.robot)


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_PROFILE.articulation_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


class TerminalKeyReader:
    """Non-blocking single-key reader for Linux terminals."""

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("--interactive_trim requires an interactive terminal (stdin is not a TTY).")
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._closed = False

    def close(self):
        if not self._closed:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._closed = True

    def poll(self) -> str | None:
        if self._closed:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None

        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch

        # Parse arrow keys: ESC [ C (right), ESC [ D (left)
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        ch2 = sys.stdin.read(1)
        if ch2 != "[":
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        ch3 = sys.stdin.read(1)
        if ch3 == "C":
            return "RIGHT"
        if ch3 == "D":
            return "LEFT"
        return None


def _print_interactive_help(total_frames: int):
    print("[INFO] Interactive trim mode enabled.")
    print(f"[INFO] Total frames: {total_frames}")
    print("[INFO] Controls:")
    print("       space/p : play/pause")
    print("       d/right : next frame (also pauses)")
    print("       a/left  : previous frame (also pauses)")
    print("       s       : mark start frame")
    print("       e       : mark end frame")
    print("       w       : write selected segment to npz")
    print("       h       : print help")
    print("       q       : quit")


def _resolve_motion_file() -> str:
    registry_name = args_cli.registry_name
    if registry_name is not None and ":" not in registry_name:
        registry_name += ":latest"

    if args_cli.motion_file is not None:
        motion_file = args_cli.motion_file
        print(f"[INFO]: Loading motion from local file: {motion_file}")
        return motion_file

    if registry_name is not None:
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO]: Loading motion from wandb artifact: {motion_file}")
        return motion_file

    raise ValueError("Must provide either --motion_file or --registry_name")


def _load_motion_npz(motion_file: str) -> dict[str, np.ndarray]:
    with np.load(motion_file) as data:
        raw = {k: np.array(data[k]) for k in data.files}

    required = ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Motion file missing required keys: {missing}")

    t = int(raw["joint_pos"].shape[0])
    for k in ("joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if raw[k].shape[0] != t:
            raise ValueError(f"Inconsistent frame count: joint_pos has {t}, but {k} has {raw[k].shape[0]}")

    return raw


def _build_trim_output_path(source_motion_file: str, start_frame: int, end_frame: int) -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    output_dir = pathlib.Path(args_cli.trim_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.trim_output_name:
        output_name = args_cli.trim_output_name
    else:
        stem = pathlib.Path(source_motion_file).stem
        lo, hi = sorted((start_frame, end_frame))
        output_name = f"{stem}_seg{lo}-{hi}"

    if output_name.endswith(".npz"):
        file_name = output_name
    else:
        file_name = f"{output_name}.npz"

    out_path = output_dir / file_name
    if out_path.exists() and not args_cli.trim_overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_path}. Use --trim_overwrite to allow overwriting."
        )
    return out_path


def _save_trimmed_segment(raw_motion: dict[str, np.ndarray], source_motion_file: str, start_frame: int, end_frame: int):
    total = int(raw_motion["joint_pos"].shape[0])
    lo, hi = sorted((start_frame, end_frame))

    if lo < 0 or hi >= total:
        raise ValueError(f"Invalid trim range [{lo}, {hi}] for total frames {total}.")

    trimmed: dict[str, np.ndarray] = {}
    for key, value in raw_motion.items():
        if key == "fps":
            trimmed[key] = value.copy()
            continue
        if value.ndim >= 1 and value.shape[0] == total:
            trimmed[key] = value[lo : hi + 1].copy()
        else:
            trimmed[key] = value.copy()

    out_path = _build_trim_output_path(source_motion_file, lo, hi)
    np.savez(out_path, **trimmed)
    print(f"[INFO] Saved trimmed motion: {out_path} | frame_range=[{lo}, {hi}] | frames={hi - lo + 1}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion_file = _resolve_motion_file()
    raw_motion = _load_motion_npz(motion_file)

    joint_pos = torch.tensor(raw_motion["joint_pos"], dtype=torch.float32, device=sim.device)
    joint_vel = torch.tensor(raw_motion["joint_vel"], dtype=torch.float32, device=sim.device)
    body_pos_w = torch.tensor(raw_motion["body_pos_w"], dtype=torch.float32, device=sim.device)
    body_quat_w = torch.tensor(raw_motion["body_quat_w"], dtype=torch.float32, device=sim.device)
    body_lin_vel_w = torch.tensor(raw_motion["body_lin_vel_w"], dtype=torch.float32, device=sim.device)
    body_ang_vel_w = torch.tensor(raw_motion["body_ang_vel_w"], dtype=torch.float32, device=sim.device)

    if joint_pos.shape[1] != robot.data.joint_pos.shape[1]:
        raise ValueError(
            f"Motion DOF mismatch: npz has {joint_pos.shape[1]} joints but robot "
            f"{ROBOT_PROFILE.name} expects {robot.data.joint_pos.shape[1]} joints."
        )
    if body_pos_w.ndim != 3 or body_pos_w.shape[1] < 1:
        raise ValueError(f"body_pos_w must be [T, B, 3] with B>=1, got {tuple(body_pos_w.shape)}")

    total_frames = int(joint_pos.shape[0])
    current_frame = 0

    interactive = bool(args_cli.interactive_trim)
    paused = interactive and (not args_cli.interactive_play)
    start_frame: int | None = None
    end_frame: int | None = None

    key_reader: TerminalKeyReader | None = None
    if interactive:
        key_reader = TerminalKeyReader()
        _print_interactive_help(total_frames)
        print(f"[INFO] Start mode: {'PAUSED' if paused else 'PLAYING'}")

    print(f"[INFO]: Using robot profile: {ROBOT_PROFILE.name}")

    try:
        # Simulation loop
        while simulation_app.is_running():
            if interactive and key_reader is not None:
                key = key_reader.poll()
                if key is not None:
                    if key in ("q", "Q", "\x03"):
                        print("[INFO] Quit interactive replay.")
                        break
                    elif key in ("h", "H"):
                        _print_interactive_help(total_frames)
                    elif key in (" ", "p", "P"):
                        paused = not paused
                        print(f"[INFO] {'PAUSED' if paused else 'PLAYING'} at frame {current_frame}")
                    elif key in ("d", "D", "RIGHT"):
                        current_frame = (current_frame + 1) % total_frames
                        paused = True
                        print(f"[INFO] Frame -> {current_frame}")
                    elif key in ("a", "A", "LEFT"):
                        current_frame = (current_frame - 1 + total_frames) % total_frames
                        paused = True
                        print(f"[INFO] Frame -> {current_frame}")
                    elif key in ("s", "S"):
                        start_frame = current_frame
                        print(f"[INFO] Marked start_frame={start_frame}")
                    elif key in ("e", "E"):
                        end_frame = current_frame
                        print(f"[INFO] Marked end_frame={end_frame}")
                    elif key in ("w", "W"):
                        if start_frame is None or end_frame is None:
                            print("[WARN] Need both start_frame and end_frame before saving.")
                        else:
                            _save_trimmed_segment(raw_motion, motion_file, start_frame, end_frame)
                    else:
                        # Ignore unknown key
                        pass

            if (not interactive) or (interactive and not paused):
                current_frame = (current_frame + 1) % total_frames

            time_steps = torch.tensor([current_frame], dtype=torch.long, device=sim.device)

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos[time_steps], joint_vel[time_steps])
            scene.write_data_to_sim()
            sim.render()  # We don't want physics stepping (sim.step())
            scene.update(sim_dt)

            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
    finally:
        if key_reader is not None:
            key_reader.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
