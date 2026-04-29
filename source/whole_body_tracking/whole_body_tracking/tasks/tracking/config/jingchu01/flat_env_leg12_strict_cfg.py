from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.jingchu01.flat_env_leg12_cfg import (
    LEG12_JOINT_NAMES,
    Jingchu01Leg12FlatEnvCfg,
    Jingchu01Leg12FlatLowFreqEnvCfg,
    Jingchu01Leg12FlatWoStateEstimationEnvCfg,
)


def _apply_leg12_joint_filters(cfg) -> None:
    """Restrict joint-related terms to the 12 lower-body joints only."""

    cfg.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)}
    cfg.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)}
    cfg.observations.critic.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)}
    cfg.observations.critic.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)}
    cfg.rewards.joint_limit.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)}
    cfg.events.add_joint_default_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG12_JOINT_NAMES)


@configclass
class Jingchu01Leg12StrictFlatEnvCfg(Jingchu01Leg12FlatEnvCfg):
    """Lower-body task with joint observations/rewards explicitly filtered to leg12."""

    def __post_init__(self):
        super().__post_init__()
        _apply_leg12_joint_filters(self)


@configclass
class Jingchu01Leg12StrictFlatWoStateEstimationEnvCfg(Jingchu01Leg12FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_leg12_joint_filters(self)


@configclass
class Jingchu01Leg12StrictFlatLowFreqEnvCfg(Jingchu01Leg12FlatLowFreqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_leg12_joint_filters(self)
