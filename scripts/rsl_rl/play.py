"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import inspect
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--enable_compliance_plugin", action="store_true", default=False, help="Enable compliance plugin integration.")
parser.add_argument("--compliance_mode", type=str, default="off", choices=["off", "teacher", "student", "adapter"], help="Compliance runtime mode.")
parser.add_argument("--compliance_log_rollouts", action="store_true", default=False, help="Enable raw compliance rollout logging.")
parser.add_argument("--compliance_save_dir", type=str, default="outputs/rollouts/default", help="Directory to save raw rollout h5.")
parser.add_argument("--payload_body_names", type=str, default="", help="Comma-separated payload body names.")
parser.add_argument("--payload_site_names", type=str, default="", help="Comma-separated payload site names.")
parser.add_argument("--torso_reference_body_name", type=str, default="torso_link", help="Torso reference body name for relative payload motion.")
parser.add_argument("--compliance_joint_names", type=str, default="", help="Comma-separated joint names to log; empty logs all joints.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional maximum rollout steps (0 means unlimited).")
parser.add_argument("--policy_io_log", action="store_true", default=False, help="Log per-step policy inputs and outputs.")
parser.add_argument(
    "--policy_io_save_path",
    type=str,
    default="outputs/policy_io/play_policy_io.h5",
    help="HDF5 path for --policy_io_log.",
)
parser.add_argument(
    "--policy_io_env_ids",
    type=str,
    default="0",
    help="Comma-separated env ids to log, or 'all'. Default logs env 0 only.",
)
parser.add_argument(
    "--policy_io_flush_interval",
    type=int,
    default=50,
    help="Flush policy IO HDF5 every N steps.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")
except metadata.PackageNotFoundError:
    INSTALLED_RSL_RL_VERSION = "0.0.0"

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from packaging import version

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.plugins.compliance.rollout_logger import ComplianceRolloutLogger, RolloutLoggerCfg


def _as_json_attr(value) -> str:
    """Serialize small metadata objects for HDF5 attributes."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, slice):
        value = {"slice": [value.start, value.stop, value.step]}
    return json.dumps(value, ensure_ascii=True)


def _parse_policy_io_env_ids(env_ids_arg: str, num_envs: int) -> list[int]:
    env_ids_arg = env_ids_arg.strip().lower()
    if env_ids_arg == "all":
        return list(range(num_envs))
    env_ids = [int(x.strip()) for x in env_ids_arg.split(",") if x.strip()]
    if not env_ids:
        raise ValueError("--policy_io_env_ids did not contain any valid env ids.")
    invalid = [idx for idx in env_ids if idx < 0 or idx >= num_envs]
    if invalid:
        raise ValueError(f"--policy_io_env_ids contains invalid ids {invalid}; num_envs={num_envs}.")
    return env_ids


class PolicyIOLogger:
    """Streaming HDF5 logger for exact policy-call inputs and outputs during play."""

    def __init__(self, env, path: str, env_ids: list[int], flush_interval: int = 50):
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError("Policy IO logging requires h5py to be installed.") from exc

        self.env = env
        self.path = os.path.abspath(path)
        self.env_ids = list(env_ids)
        self.flush_interval = max(int(flush_interval), 1)
        self._step_count = 0
        self._datasets_created = False
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.h5f = h5py.File(self.path, "w")
        self._write_metadata()

    def _write_metadata(self) -> None:
        robot = self.env.scene["robot"]
        attrs = self.h5f.attrs
        attrs["env_ids"] = _as_json_attr(self.env_ids)
        attrs["joint_names"] = _as_json_attr(list(robot.data.joint_names))
        attrs["body_names"] = _as_json_attr(list(robot.data.body_names))
        attrs["default_joint_pos"] = _as_json_attr(robot.data.default_joint_pos[0])
        attrs["default_joint_vel"] = _as_json_attr(robot.data.default_joint_vel[0])

        action_term = self.env.action_manager.get_term("joint_pos")
        attrs["action_joint_names"] = _as_json_attr(list(getattr(action_term, "_joint_names", [])))
        attrs["action_joint_ids"] = _as_json_attr(
            self._joint_ids_to_list(getattr(action_term, "_joint_ids", []), robot.num_joints)
        )
        attrs["action_scale"] = _as_json_attr(self._first_env_value(getattr(action_term, "_scale", torch.empty(0))))
        attrs["action_offset"] = _as_json_attr(self._first_env_value(getattr(action_term, "_offset", torch.empty(0))))

        motion_term = self.env.command_manager.get_term("motion")
        attrs["command_joint_names"] = _as_json_attr(list(getattr(motion_term, "joint_names", [])))
        attrs["command_body_names"] = _as_json_attr(list(getattr(motion_term.cfg, "body_names", [])))
        attrs["anchor_body_name"] = str(getattr(motion_term.cfg, "anchor_body_name", ""))

        obs_mgr = self.env.observation_manager
        attrs["policy_observation_names"] = _as_json_attr(list(obs_mgr.active_terms.get("policy", [])))
        attrs["policy_group_obs_dim"] = _as_json_attr(obs_mgr.group_obs_dim.get("policy"))
        attrs["policy_observation_slices"] = _as_json_attr(self._observation_slices())
        attrs["logged_policy_obs_key"] = "policy"
        attrs["policy_joint_pos_names"] = _as_json_attr(self._resolved_obs_joint_names("joint_pos"))
        attrs["policy_joint_vel_names"] = _as_json_attr(self._resolved_obs_joint_names("joint_vel"))
        attrs["sim_dt"] = float(self.env.cfg.sim.dt)
        attrs["decimation"] = int(self.env.cfg.decimation)
        attrs["control_dt"] = float(self.env.cfg.sim.dt * self.env.cfg.decimation)

    @staticmethod
    def _joint_ids_to_list(joint_ids, num_joints: int) -> list[int]:
        if isinstance(joint_ids, slice):
            return list(range(num_joints))[joint_ids]
        if isinstance(joint_ids, torch.Tensor):
            return [int(x) for x in joint_ids.detach().cpu().tolist()]
        return [int(x) for x in joint_ids]

    @staticmethod
    def _first_env_value(value):
        if isinstance(value, torch.Tensor) and value.ndim > 1:
            return value[0]
        return value

    def _resolved_obs_joint_names(self, term_name: str) -> list[str]:
        robot = self.env.scene["robot"]
        policy_cfg = getattr(self.env.observation_manager.cfg, "policy", None)
        term_cfg = getattr(policy_cfg, term_name, None) if policy_cfg is not None else None
        params = getattr(term_cfg, "params", None) or {}
        asset_cfg = params.get("asset_cfg")
        if asset_cfg is None:
            return list(robot.data.joint_names)
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if isinstance(joint_ids, slice):
            return list(robot.data.joint_names)[joint_ids]
        return [robot.data.joint_names[int(i)] for i in joint_ids]

    def _observation_slices(self) -> dict[str, list[int]]:
        """Return flat observation slices for each active policy term."""
        obs_mgr = self.env.observation_manager
        names = list(obs_mgr.active_terms.get("policy", []))
        dims = obs_mgr.group_obs_term_dim.get("policy")
        if dims is None:
            return {}

        slices: dict[str, list[int]] = {}
        cursor = 0
        for name, shape in zip(names, dims, strict=False):
            if not isinstance(shape, tuple):
                continue
            width = 1
            for value in shape:
                width *= int(value)
            slices[name] = [cursor, cursor + width]
            cursor += width
        return slices

    def _create_datasets(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor) -> None:
        num_logged_envs = len(self.env_ids)
        obs_dim = int(obs.shape[-1])
        action_dim = int(actions.shape[-1])
        next_obs_dim = int(next_obs.shape[-1])
        done_dim = 1 if dones.ndim == 1 else int(dones.shape[-1])

        self.h5f.create_dataset("/time/step", shape=(0,), maxshape=(None,), dtype="i8")
        self.h5f.create_dataset("/policy/time_step", shape=(0, num_logged_envs), maxshape=(None, num_logged_envs), dtype="i8")
        self.h5f.create_dataset(
            "/policy/obs", shape=(0, num_logged_envs, obs_dim), maxshape=(None, num_logged_envs, obs_dim), dtype="f4"
        )
        self.h5f.create_dataset(
            "/policy/actions",
            shape=(0, num_logged_envs, action_dim),
            maxshape=(None, num_logged_envs, action_dim),
            dtype="f4",
        )
        self.h5f.create_dataset(
            "/policy/next_obs",
            shape=(0, num_logged_envs, next_obs_dim),
            maxshape=(None, num_logged_envs, next_obs_dim),
            dtype="f4",
        )
        self.h5f.create_dataset("/env/dones", shape=(0, num_logged_envs, done_dim), maxshape=(None, num_logged_envs, done_dim), dtype="?")
        self._datasets_created = True

    def _motion_time_steps(self) -> torch.Tensor:
        motion_term = self.env.command_manager.get_term("motion")
        time_steps = getattr(motion_term, "time_steps", None)
        if time_steps is None:
            return torch.full((self.env.num_envs,), -1, dtype=torch.long)
        return time_steps.detach().long().cpu()

    @staticmethod
    def _extract_policy_obs_tensor(obs) -> torch.Tensor:
        """Extract the actor/policy tensor from TensorDict or plain observation tensors."""
        if isinstance(obs, torch.Tensor):
            return obs
        for key in ("policy", "actor", "obs", "observations"):
            try:
                if key in obs:
                    value = obs[key]
                    if isinstance(value, torch.Tensor):
                        return value
            except Exception:
                pass
        try:
            values = list(obs.values())
        except Exception as exc:
            raise TypeError(f"Unsupported policy observation type for logging: {type(obs)}") from exc
        tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]
        raise TypeError(
            "Could not infer which observation tensor to log. "
            f"Available keys: {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}"
        )

    def append_step(
        self,
        step: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        policy_time_steps: torch.Tensor,
    ) -> None:
        obs_cpu = self._extract_policy_obs_tensor(obs).detach().cpu()
        actions_cpu = actions.detach().cpu()
        next_obs_cpu = self._extract_policy_obs_tensor(next_obs).detach().cpu()
        dones_cpu = dones.detach().cpu()
        if dones_cpu.ndim == 1:
            dones_cpu = dones_cpu[:, None]
        if not self._datasets_created:
            self._create_datasets(obs_cpu, actions_cpu, next_obs_cpu, dones_cpu)

        row = self.h5f["/time/step"].shape[0]
        for path in ("/time/step", "/policy/time_step", "/policy/obs", "/policy/actions", "/policy/next_obs", "/env/dones"):
            ds = self.h5f[path]
            ds.resize((row + 1, *ds.shape[1:]))

        env_ids = torch.tensor(self.env_ids, dtype=torch.long)
        self.h5f["/time/step"][row] = int(step)
        self.h5f["/policy/time_step"][row] = policy_time_steps[env_ids].numpy()
        self.h5f["/policy/obs"][row] = obs_cpu[env_ids].numpy().astype("float32", copy=False)
        self.h5f["/policy/actions"][row] = actions_cpu[env_ids].numpy().astype("float32", copy=False)
        self.h5f["/policy/next_obs"][row] = next_obs_cpu[env_ids].numpy().astype("float32", copy=False)
        self.h5f["/env/dones"][row] = dones_cpu[env_ids].numpy().astype(bool, copy=False)

        self._step_count += 1
        if self._step_count % self.flush_interval == 0:
            self.h5f.flush()

    def close(self) -> None:
        self.h5f.flush()
        self.h5f.close()


def _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
    """Drop algorithm keys unsupported by the installed rsl-rl runtime."""
    runner_cfg = agent_cfg.to_dict()
    algorithm_cfg = runner_cfg.get("algorithm")
    if not isinstance(algorithm_cfg, dict):
        return runner_cfg

    algorithm_name = str(algorithm_cfg.get("class_name", ""))
    if algorithm_name and algorithm_name != "PPO":
        return runner_cfg

    supported_keys = set(inspect.signature(PPO.__init__).parameters.keys())
    supported_keys.discard("self")
    supported_keys.discard("policy")

    removed_keys: list[str] = []
    for key in list(algorithm_cfg.keys()):
        if key == "class_name":
            continue
        if key not in supported_keys:
            algorithm_cfg.pop(key)
            removed_keys.append(key)

    if removed_keys:
        print(
            "[WARN] Dropping unsupported PPO cfg keys for installed rsl-rl "
            f"({INSTALLED_RSL_RL_VERSION}): {', '.join(removed_keys)}"
        )

    return runner_cfg


def _should_use_deprecated_cfg_handler(installed_version: str) -> bool:
    """Use IsaacLab's deprecated-config adapter only for rsl-rl >= 4.0.0."""
    try:
        return version.parse(installed_version) >= version.parse("4.0.0")
    except Exception:
        return False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if _should_use_deprecated_cfg_handler(INSTALLED_RSL_RL_VERSION):
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if hasattr(env_cfg, "compliance") and args_cli.enable_compliance_plugin:
        env_cfg.compliance.enable = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # --- Load model from local or wandb ---
    if args_cli.checkpoint:
        # Load from local checkpoint
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from local: {resume_path}")

        # Load motion from local file if specified
        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
            print(f"[INFO]: Using local motion file: {env_cfg.commands.motion.motion_file}")

    elif args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env,
        _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg),
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        # normalizer=ppo_runner.obs_normalizer,
        normalizer=ppo_runner.obs_normalizer if hasattr(ppo_runner, "obs_normalizer") else None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    _obs_pack = env.get_observations()
    if isinstance(_obs_pack, tuple):
        obs = _obs_pack[0]
    else:
        obs = _obs_pack
    timestep = 0

    logger = None
    if args_cli.compliance_log_rollouts:
        motion_name = os.path.basename(env_cfg.commands.motion.motion_file) if hasattr(env_cfg.commands.motion, "motion_file") else "unknown"
        run_tag = os.path.basename(os.path.dirname(resume_path)) or "play"
        save_path = os.path.join(args_cli.compliance_save_dir, "raw_rollouts.h5")
        logger = ComplianceRolloutLogger(
            env.unwrapped,
            RolloutLoggerCfg(
                save_path=save_path,
                task_name=args_cli.task or "unknown",
                motion_name=motion_name,
                seed=int(getattr(agent_cfg, "seed", 0) or 0),
                run_id=run_tag,
                selected_joint_names=[x.strip() for x in args_cli.compliance_joint_names.split(",") if x.strip()],
                payload_body_names=[x.strip() for x in args_cli.payload_body_names.split(",") if x.strip()],
                payload_site_names=[x.strip() for x in args_cli.payload_site_names.split(",") if x.strip()],
                torso_reference_body_name=args_cli.torso_reference_body_name,
            ),
        )
        print(f"[INFO] Compliance raw rollout logging enabled: {save_path}")

    policy_io_logger = None
    if args_cli.policy_io_log:
        policy_io_env_ids = _parse_policy_io_env_ids(args_cli.policy_io_env_ids, env.unwrapped.num_envs)
        policy_io_logger = PolicyIOLogger(
            env.unwrapped,
            args_cli.policy_io_save_path,
            env_ids=policy_io_env_ids,
            flush_interval=args_cli.policy_io_flush_interval,
        )
        print(
            "[INFO] Policy IO logging enabled: "
            f"{policy_io_logger.path} | env_ids={policy_io_env_ids}"
        )
    # simulate environment
    try:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                policy_obs = obs
                policy_time_steps = (
                    policy_io_logger._motion_time_steps() if policy_io_logger is not None else None
                )
                policy_obs_for_log = policy_obs.detach().clone() if policy_io_logger is not None else None
                actions = policy(policy_obs)
                if actions.ndim == 1:
                    actions = actions[None, :]  # 增加 batch 维度，变成 shape [1, 29]
                # env stepping
                obs, _, dones, infos = env.step(actions)
                if policy_io_logger is not None:
                    policy_io_logger.append_step(timestep, policy_obs_for_log, actions, obs, dones, policy_time_steps)
                if logger is not None:
                    logger.append_step(actions, dones)
            timestep += 1
            if args_cli.video and timestep == args_cli.video_length:
                break
            if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                break
    finally:
        if policy_io_logger is not None:
            policy_io_logger.close()
        if logger is not None:
            logger.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
