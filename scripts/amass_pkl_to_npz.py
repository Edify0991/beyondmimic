"""Convert AMASS-retargeted PKL motions to BeyondMimic NPZ format.

This script converts flattened AMASS-retargeted lower-body motion PKLs into the
motion NPZ schema consumed by BeyondMimic:
- fps
- joint_pos
- joint_vel
- body_pos_w
- body_quat_w
- body_lin_vel_w
- body_ang_vel_w

Supported PKL layouts:
1) amass_flat_frames (default)
   dict with keys:
   - frames: list/array with shape (T, 6 + D)
     columns: [root_pos_xyz, root_euler_xyz, joint_pos_D]
   - fps: optional

2) standard_dict
   dict with keys:
   - root_pos: (T, 3)
   - root_rot: (T, 4) quaternion (wxyz)
   - dof_pos: (T, D)
   - fps: optional
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert AMASS-retargeted PKL to BeyondMimic NPZ.")
parser.add_argument("--input_file", type=str, required=True, help="Path to input PKL motion file.")
parser.add_argument("--output_name", type=str, required=True, help="Output motion name (saved to data/motions/{name}.npz).")
parser.add_argument(
    "--layout",
    type=str,
    default="auto",
    choices=["auto", "amass_flat_frames", "standard_dict"],
    help=(
        "Input PKL layout type. Use auto to infer from keys: "
        "frames -> amass_flat_frames; root_pos/root_rot/dof_pos -> standard_dict."
    ),
)
parser.add_argument(
    "--standard_root_rot_order",
    type=str,
    default="auto",
    choices=["auto", "wxyz", "xyzw"],
    help=(
        "Quaternion component order for root_rot in standard_dict layout. "
        "auto selects the more upright interpretation."
    ),
)
parser.add_argument("--input_fps", type=float, default=None, help="Override input FPS. If not set, use value in PKL.")
parser.add_argument("--output_fps", type=int, default=50, help="Output NPZ FPS.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Optional frame range [START, END] inclusive, 0-based.",
)
parser.add_argument("--robot", type=str, default="jingchu01", choices=["g1", "jingchu01"], help="Robot profile for FK replay.")
parser.add_argument("--joint_names", type=str, default=None, help="Optional comma-separated joint names. Default uses robot profile.")
parser.add_argument(
    "--joint_index_mode",
    type=str,
    default="declared",
    choices=["declared", "articulation"],
    help=(
        "How input DOF columns map to articulation joint indexes. "
        "'declared' enforces identity mapping (0..D-1) in the provided --joint_names order; "
        "'articulation' maps via robot.find_joints(...)."
    ),
)
parser.add_argument("--upload_wandb", action="store_true", default=False, help="Upload output NPZ to wandb registry.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_euler_xyz, quat_mul, quat_slerp

from whole_body_tracking.robots.robot_info import get_robot_profile, parse_joint_names_arg

ROBOT_PROFILE = get_robot_profile(args_cli.robot)


_NUMPY_PICKLE_MODULE_REMAP = {
    "numpy._core": "numpy.core",
    "numpy._core.multiarray": "numpy.core.multiarray",
    "numpy._core.numeric": "numpy.core.numeric",
    "numpy._core.umath": "numpy.core.umath",
    "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
}


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        mapped_module = _NUMPY_PICKLE_MODULE_REMAP.get(module, module)
        return super().find_class(mapped_module, name)


def _load_pickle_compat(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as err:
            if "numpy._core" not in str(err):
                raise

    print("[WARN] Detected numpy pickle module mismatch (numpy._core). Retrying with compatibility remap.")
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for replay scene used to compute FK-based body states."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

    robot: ArticulationCfg = ROBOT_PROFILE.articulation_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        layout: str,
        standard_root_rot_order: str,
        input_fps: float | None,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.layout = layout
        self.standard_root_rot_order = standard_root_rot_order
        self.input_fps_override = input_fps
        self.output_fps = float(output_fps)
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range

        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _slice_frame_range(self, *arrays: np.ndarray) -> list[np.ndarray]:
        if self.frame_range is None:
            return list(arrays)

        start, end = self.frame_range
        if start < 0 or end < start:
            raise ValueError(f"Invalid --frame_range {self.frame_range}. Expected START>=0 and END>=START.")

        sliced = [arr[start : end + 1] for arr in arrays]
        if sliced and sliced[0].shape[0] == 0:
            raise ValueError(f"Frame range {self.frame_range} produced empty motion.")
        return sliced

    def _load_from_amass_flat_frames(self, payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if "frames" not in payload:
            raise KeyError("Missing key 'frames' for layout 'amass_flat_frames'.")

        frames = np.asarray(payload["frames"], dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] < 7:
            raise ValueError(
                f"Expected frames with shape (T, 6 + D). Got shape {frames.shape}."
            )

        root_pos = frames[:, 0:3]
        root_euler = frames[:, 3:6]
        dof_pos = frames[:, 6:]

        root_quat = quat_from_euler_xyz(
            torch.from_numpy(root_euler[:, 0]),
            torch.from_numpy(root_euler[:, 1]),
            torch.from_numpy(root_euler[:, 2]),
        ).cpu().numpy()

        fps = float(payload.get("fps", 30.0))
        return root_pos, root_quat, dof_pos, fps

    def _normalize_quat_array(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return quat / norms

    def _median_upright_tilt_deg_wxyz(self, quat_wxyz: np.ndarray) -> float:
        # For R(q), R[2,2] = 1 - 2*(x^2 + y^2), i.e. body-z dot world-z.
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        r_zz = 1.0 - 2.0 * (x * x + y * y)
        tilt_deg = np.degrees(np.arccos(np.clip(r_zz, -1.0, 1.0)))
        return float(np.median(tilt_deg))

    def _resolve_standard_root_rot(self, root_rot_raw: np.ndarray) -> np.ndarray:
        if root_rot_raw.ndim != 2 or root_rot_raw.shape[1] != 4:
            raise ValueError(f"root_rot must be (T,4), got {root_rot_raw.shape}.")

        if self.standard_root_rot_order == "wxyz":
            self.resolved_root_rot_order = "wxyz"
            return self._normalize_quat_array(root_rot_raw)

        if self.standard_root_rot_order == "xyzw":
            self.resolved_root_rot_order = "xyzw"
            return self._normalize_quat_array(root_rot_raw[:, [3, 0, 1, 2]])

        quat_as_wxyz = self._normalize_quat_array(root_rot_raw)
        quat_as_xyzw = self._normalize_quat_array(root_rot_raw[:, [3, 0, 1, 2]])

        tilt_wxyz = self._median_upright_tilt_deg_wxyz(quat_as_wxyz)
        tilt_xyzw = self._median_upright_tilt_deg_wxyz(quat_as_xyzw)

        if tilt_xyzw + 1e-3 < tilt_wxyz:
            self.resolved_root_rot_order = "xyzw"
            chosen = quat_as_xyzw
        else:
            self.resolved_root_rot_order = "wxyz"
            chosen = quat_as_wxyz

        print(
            "[INFO] standard_dict root_rot auto-detect: "
            f"tilt_med(wxyz)={tilt_wxyz:.2f}deg, tilt_med(xyzw)={tilt_xyzw:.2f}deg, "
            f"chosen={self.resolved_root_rot_order}"
        )
        return chosen

    def _load_from_standard_dict(self, payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        for key in ("root_pos", "root_rot", "dof_pos"):
            if key not in payload:
                raise KeyError(f"Missing key '{key}' for layout 'standard_dict'.")

        root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
        root_rot_raw = np.asarray(payload["root_rot"], dtype=np.float32)
        root_quat = self._resolve_standard_root_rot(root_rot_raw)
        dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be (T,3), got {root_pos.shape}.")
        if dof_pos.ndim != 2:
            raise ValueError(f"dof_pos must be (T,D), got {dof_pos.shape}.")

        fps = float(payload.get("fps", 30.0))
        return root_pos, root_quat, dof_pos, fps

    def _resolve_layout(self, payload: dict) -> str:
        if self.layout != "auto":
            return self.layout

        if "frames" in payload:
            return "amass_flat_frames"

        if all(k in payload for k in ("root_pos", "root_rot", "dof_pos")):
            return "standard_dict"

        raise ValueError(
            "Unable to infer --layout from PKL keys. "
            "Expected either key 'frames' or keys {'root_pos','root_rot','dof_pos'}. "
            f"Found keys: {sorted(payload.keys())}"
        )

    def _load_motion(self):
        payload = _load_pickle_compat(self.motion_file)

        if not isinstance(payload, dict):
            raise TypeError(f"PKL payload must be dict, got {type(payload)}")

        resolved_layout = self._resolve_layout(payload)
        if resolved_layout == "amass_flat_frames":
            root_pos, root_quat, dof_pos, fps = self._load_from_amass_flat_frames(payload)
        else:
            root_pos, root_quat, dof_pos, fps = self._load_from_standard_dict(payload)

        if self.input_fps_override is not None:
            fps = float(self.input_fps_override)
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")

        root_pos, root_quat, dof_pos = self._slice_frame_range(root_pos, root_quat, dof_pos)

        if not (root_pos.shape[0] == root_quat.shape[0] == dof_pos.shape[0]):
            raise ValueError(
                "Frame count mismatch among root_pos/root_rot/dof_pos: "
                f"{root_pos.shape[0]}, {root_quat.shape[0]}, {dof_pos.shape[0]}"
            )

        self.input_fps = fps
        self.input_dt = 1.0 / self.input_fps

        self.motion_base_poss_input = torch.from_numpy(root_pos).to(torch.float32).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(root_quat).to(torch.float32).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(torch.float32).to(self.device)

        self.input_frames = int(self.motion_base_poss_input.shape[0])
        if self.input_frames < 2:
            raise ValueError("Need at least 2 frames to compute velocities and interpolation.")

        self.duration = (self.input_frames - 1) * self.input_dt
        rot_order_msg = ""
        if resolved_layout == "standard_dict":
            rot_order_msg = f" | root_rot_order={getattr(self, 'resolved_root_rot_order', self.standard_root_rot_order)}"
        print(
            f"[INFO] Motion loaded: {self.motion_file} | layout={resolved_layout}{rot_order_msg} | "
            f"frames={self.input_frames} | input_fps={self.input_fps} | dof={self.motion_dof_poss_input.shape[1]}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        if times.numel() < 2:
            times = torch.tensor([0.0, self.duration], device=self.device, dtype=torch.float32)

        self.output_frames = int(times.shape[0])
        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

        print(
            f"[INFO] Interpolated motion: input_fps={self.input_fps} -> output_fps={self.output_fps} | "
            f"output_frames={self.output_frames}"
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def get_next_state(self) -> tuple[tuple[torch.Tensor, ...], bool]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        layout=args_cli.layout,
        standard_root_rot_order=args_cli.standard_root_rot_order,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=torch.device(args_cli.device),
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]

    if motion.motion_dof_poss.shape[1] != len(joint_names):
        raise ValueError(
            f"Joint dimension mismatch: motion has {motion.motion_dof_poss.shape[1]} DOFs, "
            f"but {len(joint_names)} joint names were provided for robot {ROBOT_PROFILE.name}."
        )

    resolved_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(resolved_joint_indexes) != len(joint_names):
        raise ValueError(
            "Failed to resolve all requested joints in robot articulation. "
            f"Resolved {len(resolved_joint_indexes)} / {len(joint_names)} names: {joint_names}"
        )
    resolved_joint_indexes = [int(i) for i in resolved_joint_indexes]
    joint_name_to_index = {name: idx for name, idx in zip(joint_names, resolved_joint_indexes, strict=False)}
    articulation_order_names = ["<unmapped>"] * len(joint_names)
    for name, idx in joint_name_to_index.items():
        if 0 <= idx < len(articulation_order_names):
            articulation_order_names[idx] = name

    if args_cli.joint_index_mode == "declared":
        robot_dof = int(robot.data.default_joint_pos.shape[1])
        if robot_dof != len(joint_names):
            raise ValueError(
                "joint_index_mode='declared' requires robot dof count == len(joint_names). "
                f"Got robot_dof={robot_dof}, len(joint_names)={len(joint_names)}."
            )
        declared_indexes = list(range(len(joint_names)))
        if resolved_joint_indexes != declared_indexes:
            raise ValueError(
                "joint_index_mode='declared' was requested, but articulation order does not match provided "
                "--joint_names order.\n"
                f"resolved_indexes={resolved_joint_indexes}\n"
                f"declared_indexes={declared_indexes}\n"
                f"joint_name_to_index={joint_name_to_index}\n"
                f"articulation_order_names={articulation_order_names}\n"
                "If you want to map by articulation names, set --joint_index_mode articulation."
            )
        robot_joint_indexes = declared_indexes
    else:
        robot_joint_indexes = resolved_joint_indexes

    print(
        f"[INFO] Joint mapping mode={args_cli.joint_index_mode} | "
        f"joint_names={joint_names} | indexes={robot_joint_indexes}"
    )

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)

            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "motions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{args_cli.output_name}.npz")
            np.savez(output_path, **log)
            print(f"[INFO] Saved BeyondMimic motion npz to: {output_path}")

            if args_cli.upload_wandb:
                import wandb

                collection = args_cli.output_name
                run = wandb.init(project="pkl_to_npz", name=collection)
                registry = "motions"
                artifact = run.log_artifact(artifact_or_path=output_path, name=collection, type=registry)
                run.link_artifact(artifact=artifact, target_path=f"wandb-registry-{registry}/{collection}")
                print(f"[INFO] Uploaded to wandb registry: {registry}/{collection}")
                run.finish()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete.")
    print(f"[INFO] Robot profile: {ROBOT_PROFILE.name}")

    joint_names = parse_joint_names_arg(args_cli.joint_names, ROBOT_PROFILE.motion_joint_names)
    run_simulator(sim, scene, joint_names=joint_names)


if __name__ == "__main__":
    main()
    simulation_app.close()
