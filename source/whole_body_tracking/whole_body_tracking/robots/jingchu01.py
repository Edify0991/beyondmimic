import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR


def _env_list(var_name: str, default: list[str]) -> tuple[str, ...]:
    raw = os.getenv(var_name)
    if raw is None:
        return tuple(default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values if values else tuple(default)


# -----------------------------------------------------------------------------
# Runtime mode selection
# -----------------------------------------------------------------------------
# WBT_JINGCHU01_CONTROL_MODE options:
# - lower_body: 12-DOF leg-only control/tracking (default)
# - full_body: keep full-body interfaces for future use
SUPPORTED_CONTROL_MODES = ("lower_body", "full_body")
JINGCHU01_CONTROL_MODE = os.getenv("WBT_JINGCHU01_CONTROL_MODE", "lower_body").strip().lower()
if JINGCHU01_CONTROL_MODE not in SUPPORTED_CONTROL_MODES:
    raise ValueError(
        f"Invalid WBT_JINGCHU01_CONTROL_MODE='{JINGCHU01_CONTROL_MODE}'. "
        f"Expected one of: {SUPPORTED_CONTROL_MODES}."
    )


# Set this in your shell when URDF is outside this repository.
JINGCHU01_URDF_PATH = os.path.abspath(
    os.path.expanduser(
        os.getenv(
            "WBT_JINGCHU01_URDF",
            f"/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF_legs.urdf",
        )
    )
)


# -----------------------------------------------------------------------------
# Motion/body metadata defaults
# -----------------------------------------------------------------------------
JINGCHU01_MOTION_JOINT_NAMES_LOWER_BODY = (
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
)

JINGCHU01_MOTION_JOINT_NAMES_FULL_BODY = (
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "waist_roll",
    "waist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
)

JINGCHU01_MOTION_BODY_NAMES_LOWER_BODY = (
    "Robotbase",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
)

JINGCHU01_MOTION_BODY_NAMES_FULL_BODY = (
    "Robotbase",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "waist_roll",
    "waist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
)

JINGCHU01_CONTACT_EXEMPT_BODY_NAMES_LOWER_BODY = (
    "right_ankle_roll",
    "left_ankle_roll",
)

JINGCHU01_CONTACT_EXEMPT_BODY_NAMES_FULL_BODY = (
    "right_ankle_roll",
    "left_ankle_roll",
    "right_wrist_roll",
    "left_wrist_roll",
)


def _default_for_mode(lower_body_values: tuple[str, ...], full_body_values: tuple[str, ...]) -> list[str]:
    if JINGCHU01_CONTROL_MODE == "full_body":
        return list(full_body_values)
    return list(lower_body_values)


# Override via env vars when your naming differs:
# - WBT_JINGCHU01_JOINT_NAMES
# - WBT_JINGCHU01_ANCHOR_BODY
# - WBT_JINGCHU01_BODY_NAMES
# - WBT_JINGCHU01_CONTACT_EXEMPT_BODIES
# - WBT_JINGCHU01_TERMINATION_BODIES
JINGCHU01_MOTION_JOINT_NAMES = _env_list(
    "WBT_JINGCHU01_JOINT_NAMES",
    _default_for_mode(JINGCHU01_MOTION_JOINT_NAMES_LOWER_BODY, JINGCHU01_MOTION_JOINT_NAMES_FULL_BODY),
)

JINGCHU01_ANCHOR_BODY_NAME = os.getenv("WBT_JINGCHU01_ANCHOR_BODY", "Robotbase")

JINGCHU01_MOTION_BODY_NAMES = _env_list(
    "WBT_JINGCHU01_BODY_NAMES",
    _default_for_mode(JINGCHU01_MOTION_BODY_NAMES_LOWER_BODY, JINGCHU01_MOTION_BODY_NAMES_FULL_BODY),
)

JINGCHU01_CONTACT_EXEMPT_BODY_NAMES = _env_list(
    "WBT_JINGCHU01_CONTACT_EXEMPT_BODIES",
    _default_for_mode(
        JINGCHU01_CONTACT_EXEMPT_BODY_NAMES_LOWER_BODY,
        JINGCHU01_CONTACT_EXEMPT_BODY_NAMES_FULL_BODY,
    ),
)

JINGCHU01_TERMINATION_BODY_NAMES = _env_list(
    "WBT_JINGCHU01_TERMINATION_BODIES",
    list(JINGCHU01_CONTACT_EXEMPT_BODY_NAMES),
)


# -----------------------------------------------------------------------------
# Motor parameters
# -----------------------------------------------------------------------------
# Kp = armature * natural_freq^2
# Kd = 2 * damping_ratio * armature * natural_freq
NATURAL_FREQ = 6.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 1.1

ARMATURE_A10020_P224 = 0.2773762228
ARMATURE_A10020_P112 = 0.07001770124
ARMATURE_A8112_P118 = 0.0485578476
ARMATURE_A6408_P225 = 0.03960461065
ARMATURE_A4310_P236 = 0.02422284137

STIFFNESS_A10020_P224 = ARMATURE_A10020_P224 * NATURAL_FREQ**2
STIFFNESS_A10020_P112 = ARMATURE_A10020_P112 * NATURAL_FREQ**2
STIFFNESS_A8112_P118 = ARMATURE_A8112_P118 * NATURAL_FREQ**2
STIFFNESS_A6408_P225 = ARMATURE_A6408_P225 * NATURAL_FREQ**2
STIFFNESS_A4310_P236 = ARMATURE_A4310_P236 * NATURAL_FREQ**2

DAMPING_A10020_P224 = 2.0 * DAMPING_RATIO * ARMATURE_A10020_P224 * NATURAL_FREQ
DAMPING_A10020_P112 = 2.0 * DAMPING_RATIO * ARMATURE_A10020_P112 * NATURAL_FREQ
DAMPING_A8112_P118 = 2.0 * DAMPING_RATIO * ARMATURE_A8112_P118 * NATURAL_FREQ
DAMPING_A6408_P225 = 2.0 * DAMPING_RATIO * ARMATURE_A6408_P225 * NATURAL_FREQ
DAMPING_A4310_P236 = 2.0 * DAMPING_RATIO * ARMATURE_A4310_P236 * NATURAL_FREQ

ARMATURE_KNEE = 300.0 / (NATURAL_FREQ**2)
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ

# Joint matching patterns
HIP_ROLL = r".*_hip_roll(?:_joint)?$"
HIP_YAW = r".*_hip_yaw(?:_joint)?$"
HIP_PITCH = r".*_hip_pitch(?:_joint)?$"
KNEE_PITCH = r".*_knee_pitch(?:_joint)?$"
ANKLE_PITCH = r".*_ankle_pitch(?:_joint)?$"
ANKLE_ROLL = r".*_ankle_roll(?:_joint)?$"
WAIST_ROLL = r"waist_roll(?:_joint)?$"
WAIST_YAW = r"waist_yaw(?:_joint)?$"
SHOULDER_PITCH = r".*_shoulder_pitch(?:_joint)?$"
SHOULDER_ROLL = r".*_shoulder_roll(?:_joint)?$"
SHOULDER_YAW = r".*_shoulder_yaw(?:_joint)?$"
ELBOW_PITCH = r".*_elbow_pitch(?:_joint)?$"
ELBOW_YAW = r".*_elbow_yaw(?:_joint)?$"
WRIST_PITCH = r".*_wrist_pitch(?:_joint)?$"
WRIST_ROLL = r".*_wrist_roll(?:_joint)?$"


FULL_BODY_INIT_JOINT_POS = {
    HIP_PITCH: -0.24,
    KNEE_PITCH: 0.48,
    ANKLE_PITCH: -0.24,
    SHOULDER_PITCH: 0.2,
    SHOULDER_ROLL: 0.0,
    ELBOW_PITCH: 0.5,
}

LOWER_BODY_INIT_JOINT_POS = {
    HIP_PITCH: -0.24,
    KNEE_PITCH: 0.48,
    ANKLE_PITCH: -0.24,
}


def _build_full_body_actuators() -> dict[str, ImplicitActuatorCfg]:
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[HIP_ROLL, HIP_YAW, HIP_PITCH, KNEE_PITCH],
            effort_limit_sim={
                HIP_ROLL: 130.0,
                HIP_YAW: 130.0,
                HIP_PITCH: 130.0,
                KNEE_PITCH: 130.0,
            },
            velocity_limit_sim={
                HIP_ROLL: 2.094,
                HIP_YAW: 2.618,
                HIP_PITCH: 3.926,
                KNEE_PITCH: 2.770,
            },
            stiffness={
                HIP_ROLL: STIFFNESS_A10020_P224,
                HIP_YAW: STIFFNESS_A10020_P112,
                HIP_PITCH: STIFFNESS_A10020_P224,
                KNEE_PITCH: STIFFNESS_KNEE,
            },
            damping={
                HIP_ROLL: DAMPING_A10020_P224,
                HIP_YAW: DAMPING_A10020_P112,
                HIP_PITCH: DAMPING_A10020_P224,
                KNEE_PITCH: DAMPING_KNEE,
            },
            armature={
                HIP_ROLL: ARMATURE_A10020_P224,
                HIP_YAW: ARMATURE_A10020_P112,
                HIP_PITCH: ARMATURE_A10020_P224,
                KNEE_PITCH: ARMATURE_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[ANKLE_PITCH, ANKLE_ROLL],
            effort_limit_sim={
                ANKLE_PITCH: 130.0,
                ANKLE_ROLL: 130.0,
            },
            velocity_limit_sim={
                ANKLE_PITCH: 3.124,
                ANKLE_ROLL: 4.160,
            },
            stiffness=2.0*STIFFNESS_A8112_P118,
            damping=2.0*DAMPING_A8112_P118,
            armature=2.0*ARMATURE_A8112_P118,
        ),
        "waist_joints": ImplicitActuatorCfg(
            joint_names_expr=[WAIST_ROLL, WAIST_YAW],
            effort_limit_sim={
                WAIST_ROLL: 50.0,
                WAIST_YAW: 50.0,
            },
            velocity_limit_sim={
                WAIST_ROLL: 2.0,
                WAIST_YAW: 2.0,
            },
            stiffness={
                WAIST_ROLL: STIFFNESS_A10020_P224,
                WAIST_YAW: STIFFNESS_A10020_P112,
            },
            damping={
                WAIST_ROLL: DAMPING_A10020_P224,
                WAIST_YAW: DAMPING_A10020_P112,
            },
            armature={
                WAIST_ROLL: ARMATURE_A10020_P224,
                WAIST_YAW: ARMATURE_A10020_P112,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[SHOULDER_PITCH, SHOULDER_ROLL, SHOULDER_YAW, ELBOW_PITCH, ELBOW_YAW, WRIST_PITCH, WRIST_ROLL],
            effort_limit_sim={
                SHOULDER_PITCH: 25.0,
                SHOULDER_ROLL: 25.0,
                SHOULDER_YAW: 25.0,
                ELBOW_PITCH: 25.0,
                ELBOW_YAW: 25.0,
                WRIST_PITCH: 10.0,
                WRIST_ROLL: 10.0,
            },
            velocity_limit_sim={
                SHOULDER_PITCH: 2.0,
                SHOULDER_ROLL: 2.0,
                SHOULDER_YAW: 2.0,
                ELBOW_PITCH: 2.0,
                ELBOW_YAW: 2.0,
                WRIST_PITCH: 2.0,
                WRIST_ROLL: 2.0,
            },
            stiffness={
                SHOULDER_PITCH: STIFFNESS_A8112_P118,
                SHOULDER_ROLL: STIFFNESS_A8112_P118,
                SHOULDER_YAW: STIFFNESS_A6408_P225,
                ELBOW_PITCH: STIFFNESS_A6408_P225,
                ELBOW_YAW: STIFFNESS_A4310_P236,
                WRIST_PITCH: STIFFNESS_A4310_P236,
                WRIST_ROLL: STIFFNESS_A4310_P236,
            },
            damping={
                SHOULDER_PITCH: DAMPING_A8112_P118,
                SHOULDER_ROLL: DAMPING_A8112_P118,
                SHOULDER_YAW: DAMPING_A6408_P225,
                ELBOW_PITCH: DAMPING_A6408_P225,
                ELBOW_YAW: DAMPING_A4310_P236,
                WRIST_PITCH: DAMPING_A4310_P236,
                WRIST_ROLL: DAMPING_A4310_P236,
            },
            armature={
                SHOULDER_PITCH: ARMATURE_A8112_P118,
                SHOULDER_ROLL: ARMATURE_A8112_P118,
                SHOULDER_YAW: ARMATURE_A6408_P225,
                ELBOW_PITCH: ARMATURE_A6408_P225,
                ELBOW_YAW: ARMATURE_A4310_P236,
                WRIST_PITCH: ARMATURE_A4310_P236,
                WRIST_ROLL: ARMATURE_A4310_P236,
            },
        ),
    }


def _select_actuators_for_mode() -> dict[str, ImplicitActuatorCfg]:
    full_body = _build_full_body_actuators()
    if JINGCHU01_CONTROL_MODE == "full_body":
        return full_body
    return {
        "legs": full_body["legs"],
        "feet": full_body["feet"],
    }


def _select_init_joint_pos_for_mode() -> dict[str, float]:
    if JINGCHU01_CONTROL_MODE == "full_body":
        return FULL_BODY_INIT_JOINT_POS
    return LOWER_BODY_INIT_JOINT_POS


JINGCHU01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=JINGCHU01_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos=_select_init_joint_pos_for_mode(),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=_select_actuators_for_mode(),
)


JINGCHU01_ACTION_SCALE = {}
for actuator_cfg in JINGCHU01_CFG.actuators.values():
    effort_limit = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    names = actuator_cfg.joint_names_expr
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort_limit and name in stiffness and stiffness[name]:
            JINGCHU01_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]
