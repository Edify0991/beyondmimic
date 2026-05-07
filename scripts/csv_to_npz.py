"""This script replays a motion from a csv/pkl file and outputs it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50

    python csv_to_npz.py --input_file motion.pkl --input_format pkl --robot jingchu01 \
    --output_name dance1_subject2 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv/pkl file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv/pkl file.")
parser.add_argument(
    "--input_format",
    type=str,
    default="auto",
    choices=["auto", "csv", "pkl"],
    help="Input motion format. PKL supports dict keys root_pos/root_rot/dof_pos/fps.",
)
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--robot", type=str, default="g1", choices=["g1", "jingchu01"], help="Robot profile for simulation and joint/body conventions.")
parser.add_argument("--joint_names", type=str, default=None, help="Optional comma-separated joint names overriding the default order for the selected robot.")
parser.add_argument(
    "--pkl_root_rot_order",
    type=str,
    default="auto",
    choices=["auto", "wxyz", "xyzw"],
    help="Quaternion order for PKL root_rot. Use auto for upright-tilt heuristic.",
)
# parser.add_argument(
#     "--device",
#     type=str,
#     default="cpu",
#     help="Device to run on, e.g., 'cpu', 'cuda:0', 'cuda:1', etc."
# )
parser.add_argument("--upload_wandb", action="store_true", default=False, help="Also upload motion to wandb registry.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
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
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_PROFILE.articulation_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from a csv or pkl file."""
        input_format = args_cli.input_format
        if input_format == "auto":
            input_format = "pkl" if self.motion_file.lower().endswith(".pkl") else "csv"

        if input_format == "pkl":
            self._load_motion_from_pkl()
        elif input_format == "csv":
            self._load_motion_from_csv()
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def _slice_arrays(self, *arrays: np.ndarray) -> list[np.ndarray]:
        if self.frame_range is None:
            return list(arrays)
        start, end = self.frame_range
        if start < 1 or end < start:
            raise ValueError(f"Invalid --frame_range {self.frame_range}. Expected START>=1 and END>=START.")
        start_idx = start - 1
        end_idx = end
        sliced = [arr[start_idx:end_idx] for arr in arrays]
        if sliced and sliced[0].shape[0] == 0:
            raise ValueError(f"Frame range {self.frame_range} produced empty motion.")
        return sliced

    def _load_motion_from_csv(self):
        if self.frame_range is None:
            motion_np = np.loadtxt(self.motion_file, delimiter=",")
        else:
            # CSV frame range follows the original script convention: 1-indexed, inclusive.
            motion_np = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        motion = torch.from_numpy(motion_np)
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), format=csv, duration: {self.duration} sec, frames: {self.input_frames}")

    @staticmethod
    def _normalize_quat_array(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return quat / norms

    @staticmethod
    def _median_upright_tilt_deg_wxyz(quat_wxyz: np.ndarray) -> float:
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        r_zz = 1.0 - 2.0 * (x * x + y * y)
        return float(np.median(np.degrees(np.arccos(np.clip(r_zz, -1.0, 1.0)))))

    def _resolve_pkl_root_rot(self, root_rot_raw: np.ndarray) -> np.ndarray:
        if root_rot_raw.ndim != 2 or root_rot_raw.shape[1] != 4:
            raise ValueError(f"PKL root_rot must have shape (T, 4), got {root_rot_raw.shape}.")

        if args_cli.pkl_root_rot_order == "wxyz":
            self.resolved_pkl_root_rot_order = "wxyz"
            return self._normalize_quat_array(root_rot_raw)
        if args_cli.pkl_root_rot_order == "xyzw":
            self.resolved_pkl_root_rot_order = "xyzw"
            return self._normalize_quat_array(root_rot_raw[:, [3, 0, 1, 2]])

        quat_as_wxyz = self._normalize_quat_array(root_rot_raw)
        quat_as_xyzw = self._normalize_quat_array(root_rot_raw[:, [3, 0, 1, 2]])
        tilt_wxyz = self._median_upright_tilt_deg_wxyz(quat_as_wxyz)
        tilt_xyzw = self._median_upright_tilt_deg_wxyz(quat_as_xyzw)
        if tilt_xyzw + 1e-3 < tilt_wxyz:
            self.resolved_pkl_root_rot_order = "xyzw"
            chosen = quat_as_xyzw
        else:
            self.resolved_pkl_root_rot_order = "wxyz"
            chosen = quat_as_wxyz
        print(
            "[INFO] PKL root_rot auto-detect: "
            f"tilt_med(wxyz)={tilt_wxyz:.2f}deg, tilt_med(xyzw)={tilt_xyzw:.2f}deg, "
            f"chosen={self.resolved_pkl_root_rot_order}"
        )
        return chosen

    def _load_motion_from_pkl(self):
        payload = _load_pickle_compat(self.motion_file)
        if not isinstance(payload, dict):
            raise TypeError(f"PKL payload must be dict, got {type(payload)}")
        for key in ("root_pos", "root_rot", "dof_pos"):
            if key not in payload:
                raise KeyError(f"Missing key '{key}' in PKL motion.")

        root_pos = np.asarray(payload["root_pos"], dtype=np.float32)
        root_rot = self._resolve_pkl_root_rot(np.asarray(payload["root_rot"], dtype=np.float32))
        dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
        root_pos, root_rot, dof_pos = self._slice_arrays(root_pos, root_rot, dof_pos)

        fps = float(payload.get("fps", self.input_fps))
        if fps <= 0.0:
            raise ValueError(f"Invalid PKL fps: {fps}")
        self.input_fps = fps
        self.input_dt = 1.0 / self.input_fps

        self.motion_base_poss_input = torch.from_numpy(root_pos).to(torch.float32).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(root_rot).to(torch.float32).to(self.device)
        self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(torch.float32).to(self.device)

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), format=pkl, duration: {self.duration} sec, "
            f"frames: {self.input_frames}, input_fps: {self.input_fps}, dof: {self.motion_dof_poss_input.shape[1]}"
        )

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
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
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
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
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        # device=sim.device,
        device=torch.device(args_cli.device),
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    if motion.motion_dof_poss.shape[1] != len(joint_names):
        raise ValueError(
            f"Joint dimension mismatch: motion file has {motion.motion_dof_poss.shape[1]} DOFs, "
            f"but {len(joint_names)} joint names were provided for robot {ROBOT_PROFILE.name}."
        )

    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    if len(robot_joint_indexes) != len(joint_names):
        raise ValueError(
            "Failed to resolve all requested joints in the robot articulation. "
            f"Resolved {len(robot_joint_indexes)} / {len(joint_names)} names: {joint_names}"
        )

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "source_joint_names": np.array(joint_names),
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
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

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
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
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            log["joint_names"] = np.array(robot.data.joint_names)
            log["body_names"] = np.array(robot.data.body_names)
            log["source_motion_file"] = np.array(args_cli.input_file)
            log["source_joint_layout"] = np.array("input_joint_names")
            log["saved_joint_layout"] = np.array("isaaclab_articulation_order")
            log["saved_body_layout"] = np.array("isaaclab_body_order")

            # np.savez("tmp/motion.npz", **log)

            # import wandb

            # COLLECTION = args_cli.output_name
            # run = wandb.init(project="csv_to_npz", name=COLLECTION)
            # # print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
            # REGISTRY = "motions"
            # logged_artifact = run.log_artifact(artifact_or_path="tmp/motion.npz", name=COLLECTION, type=REGISTRY)
            # run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
            # print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

            import os
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "motions")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{args_cli.output_name}.npz")
            np.savez(output_path, **log)
            print(f"[INFO]: Motion saved locally to: {output_path}")

            # Optionally upload to wandb registry (controlled by --upload_wandb flag)
            if args_cli.upload_wandb:
                import wandb
                COLLECTION = args_cli.output_name
                run = wandb.init(project="csv_to_npz", name=COLLECTION)
                REGISTRY = "motions"
                logged_artifact = run.log_artifact(artifact_or_path=output_path, name=COLLECTION, type=REGISTRY)
                run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
                print(f"[INFO]: Motion also saved to wandb registry: {REGISTRY}/{COLLECTION}")
                run.finish()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    joint_names = parse_joint_names_arg(args_cli.joint_names, ROBOT_PROFILE.motion_joint_names)
    print(f"[INFO]: Using robot profile: {ROBOT_PROFILE.name}")
    run_simulator(sim, scene, joint_names=joint_names)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
