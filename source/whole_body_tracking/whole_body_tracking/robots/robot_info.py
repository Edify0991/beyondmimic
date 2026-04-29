from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.robots.jingchu01 import (
    JINGCHU01_ACTION_SCALE,
    JINGCHU01_ANCHOR_BODY_NAME,
    JINGCHU01_CFG,
    JINGCHU01_CONTACT_EXEMPT_BODY_NAMES,
    JINGCHU01_MOTION_BODY_NAMES,
    JINGCHU01_MOTION_JOINT_NAMES,
    JINGCHU01_TERMINATION_BODY_NAMES,
)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    articulation_cfg: ArticulationCfg
    action_scale: dict[str, float]
    motion_joint_names: tuple[str, ...]
    motion_anchor_body_name: str
    motion_body_names: tuple[str, ...]
    contact_exempt_body_names: tuple[str, ...]
    termination_body_names: tuple[str, ...]
    torso_reference_body_name: str


G1_MOTION_JOINT_NAMES = (
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
)

G1_MOTION_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

G1_CONTACT_EXEMPT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


ROBOT_PROFILES = {
    "g1": RobotProfile(
        name="g1",
        articulation_cfg=G1_CYLINDER_CFG,
        action_scale=G1_ACTION_SCALE,
        motion_joint_names=G1_MOTION_JOINT_NAMES,
        motion_anchor_body_name="torso_link",
        motion_body_names=G1_MOTION_BODY_NAMES,
        contact_exempt_body_names=G1_CONTACT_EXEMPT_BODY_NAMES,
        termination_body_names=G1_CONTACT_EXEMPT_BODY_NAMES,
        torso_reference_body_name="torso_link",
    ),
    "jingchu01": RobotProfile(
        name="jingchu01",
        articulation_cfg=JINGCHU01_CFG,
        action_scale=JINGCHU01_ACTION_SCALE,
        motion_joint_names=tuple(JINGCHU01_MOTION_JOINT_NAMES),
        motion_anchor_body_name=JINGCHU01_ANCHOR_BODY_NAME,
        motion_body_names=tuple(JINGCHU01_MOTION_BODY_NAMES),
        contact_exempt_body_names=tuple(JINGCHU01_CONTACT_EXEMPT_BODY_NAMES),
        termination_body_names=tuple(JINGCHU01_TERMINATION_BODY_NAMES),
        torso_reference_body_name=JINGCHU01_ANCHOR_BODY_NAME,
    ),
}


_ROBOT_ALIASES = {
    "g1": "g1",
    "jingchu01": "jingchu01",
    "jingchu": "jingchu01",
    "jc01": "jingchu01",
}


def list_robot_names() -> list[str]:
    return sorted(ROBOT_PROFILES.keys())


def get_robot_profile(name: str) -> RobotProfile:
    if not name:
        raise ValueError("Robot name must be a non-empty string.")
    key = _ROBOT_ALIASES.get(name.strip().lower())
    if key is None:
        available = ", ".join(list_robot_names())
        raise ValueError(f"Unsupported robot '{name}'. Available options: {available}")
    return ROBOT_PROFILES[key]


def parse_joint_names_arg(joint_names_arg: str | None, default_joint_names: Sequence[str]) -> list[str]:
    if joint_names_arg is None:
        return list(default_joint_names)

    parsed = [name.strip() for name in joint_names_arg.split(",") if name.strip()]
    if not parsed:
        raise ValueError("--joint_names was provided but no valid names were found.")
    return parsed
