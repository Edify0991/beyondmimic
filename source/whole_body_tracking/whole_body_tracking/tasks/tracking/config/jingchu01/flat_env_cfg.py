import re

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.robot_info import get_robot_profile
from whole_body_tracking.tasks.tracking.config.jingchu01.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

JINGCHU01_PROFILE = get_robot_profile("jingchu01")


def _build_undesired_contacts_regex(contact_exempt_body_names: tuple[str, ...]) -> str:
    if not contact_exempt_body_names:
        return r".+"
    lookahead = "".join(f"(?!{re.escape(name)}$)" for name in contact_exempt_body_names)
    return rf"^{lookahead}.+$"


@configclass
class Jingchu01FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = JINGCHU01_PROFILE.articulation_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = JINGCHU01_PROFILE.action_scale
        self.actions.joint_pos.joint_names = list(JINGCHU01_PROFILE.motion_joint_names)
        self.commands.motion.joint_names = list(JINGCHU01_PROFILE.motion_joint_names)
        self.commands.motion.anchor_body_name = JINGCHU01_PROFILE.motion_anchor_body_name
        self.commands.motion.body_names = list(JINGCHU01_PROFILE.motion_body_names)

        self.events.base_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=JINGCHU01_PROFILE.motion_anchor_body_name
        )

        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[_build_undesired_contacts_regex(JINGCHU01_PROFILE.contact_exempt_body_names)],
        )

        self.terminations.ee_body_pos.params["body_names"] = list(JINGCHU01_PROFILE.termination_body_names)


@configclass
class Jingchu01FlatWoStateEstimationEnvCfg(Jingchu01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Jingchu01FlatLowFreqEnvCfg(Jingchu01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
