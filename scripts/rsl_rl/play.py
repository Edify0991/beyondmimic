"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import inspect
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
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if actions.ndim == 1:
                actions = actions[None, :]  # 增加 batch 维度，变成 shape [1, 29]
            # env stepping
            obs, _, dones, infos = env.step(actions)
            if logger is not None:
                logger.append_step(actions, dones)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

    if logger is not None:
        logger.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
