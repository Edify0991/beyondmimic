from isaaclab.utils import configclass

from whole_body_tracking.robots.jingchu01 import JINGCHU01_MOTION_JOINT_NAMES_LOWER_BODY
from whole_body_tracking.tasks.tracking.config.jingchu01.flat_env_cfg import (
    Jingchu01FlatEnvCfg,
    Jingchu01FlatLowFreqEnvCfg,
    Jingchu01FlatWoStateEstimationEnvCfg,
)


LEG12_JOINT_NAMES = list(JINGCHU01_MOTION_JOINT_NAMES_LOWER_BODY)


@configclass
class Jingchu01Leg12FlatEnvCfg(Jingchu01FlatEnvCfg):
    """Dedicated lower-body (12-DOF) Jingchu01 tracking config."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = LEG12_JOINT_NAMES
        self.commands.motion.joint_names = LEG12_JOINT_NAMES


@configclass
class Jingchu01Leg12FlatWoStateEstimationEnvCfg(Jingchu01FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = LEG12_JOINT_NAMES
        self.commands.motion.joint_names = LEG12_JOINT_NAMES


@configclass
class Jingchu01Leg12FlatLowFreqEnvCfg(Jingchu01FlatLowFreqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.joint_names = LEG12_JOINT_NAMES
        self.commands.motion.joint_names = LEG12_JOINT_NAMES
