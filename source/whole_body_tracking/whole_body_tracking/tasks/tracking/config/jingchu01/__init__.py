import gymnasium as gym

from . import agents, flat_env_cfg, flat_env_leg12_cfg, flat_env_leg12_strict_cfg


# Register Gym environments.

gym.register(
    id="Tracking-Flat-Jingchu01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Jingchu01FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Jingchu01FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Jingchu01FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_cfg.Jingchu01Leg12FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_cfg.Jingchu01Leg12FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_cfg.Jingchu01Leg12FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-Strict-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_strict_cfg.Jingchu01Leg12StrictFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-Strict-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_strict_cfg.Jingchu01Leg12StrictFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Jingchu01-Leg12-Strict-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_leg12_strict_cfg.Jingchu01Leg12StrictFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Jingchu01FlatLowFreqPPORunnerCfg",
    },
)
