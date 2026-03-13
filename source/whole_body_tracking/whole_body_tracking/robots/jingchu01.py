import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# ============ 电机参数计算 ============
# 保持与G1相同的自然频率和阻尼比
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 62.83 rad/s (10Hz)
DAMPING_RATIO = 2.0

# 根据目标Kp反推ARMATURE值
# 公式: Kp = ARMATURE * NATURAL_FREQ^2
# 公式: Kd = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

# 髋关节目标: Kp≈150, Kd≈10
ARMATURE_HIP = 150.0 / (NATURAL_FREQ**2)
STIFFNESS_HIP = ARMATURE_HIP * NATURAL_FREQ**2  # ≈150
DAMPING_HIP = 2.0 * DAMPING_RATIO * ARMATURE_HIP * NATURAL_FREQ  # ≈9.55

# 膝关节目标: Kp≈200, Kd≈15
ARMATURE_KNEE = 200.0 / (NATURAL_FREQ**2)
STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2  # ≈200
DAMPING_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ  # ≈12.73

# 踝关节目标: Kp≈50, Kd≈3
ARMATURE_ANKLE = 50.0 / (NATURAL_FREQ**2)
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2  # ≈50
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ  # ≈3.18

# 手臂关节（假设使用较小的电机，类似G1的5020电机）
ARMATURE_ARM = 0.003609725  # 与G1的5020电机相同
STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ**2
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ

# 腰部关节（假设使用中等电机）
ARMATURE_WAIST = 0.003609725  # 与G1的5020电机相同
STIFFNESS_WAIST = ARMATURE_WAIST * NATURAL_FREQ**2
DAMPING_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_WAIST * NATURAL_FREQ


JINGCHU01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/../../../jingchu01/jingchu01_legs.urdf",  # 请根据实际路径修改
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
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),  # 初始高度，根据机器人调整
        joint_pos={
            # 腿部初始姿态（站立姿态）
            ".*_hip_pitch": -0.24,
            ".*_knee_pitch": 0.48,
            ".*_ankle_pitch": -0.24,
            # 手臂初始姿态
            ".*_shoulder_pitch": 0.2,
            ".*_shoulder_roll": 0.0,
            ".*_elbow_pitch": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # ============ 髋关节 (Kp≈150, Kd≈9.55) ============
        "hip_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll",
                ".*_hip_yaw",
                ".*_hip_pitch",
            ],
            # 从URDF读取的effort和velocity限制
            effort_limit_sim={
                ".*_hip_roll": 163.0,
                ".*_hip_yaw": 130.0,
                ".*_hip_pitch": 484.23,
            },
            velocity_limit_sim={
                ".*_hip_roll": 2.094,
                ".*_hip_yaw": 2.618,
                ".*_hip_pitch": 3.926,
            },
            stiffness=STIFFNESS_HIP,
            damping=DAMPING_HIP,
            armature=ARMATURE_HIP,
        ),
        
        # ============ 膝关节 (Kp≈200, Kd≈12.73) ============
        "knee_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch"],
            effort_limit_sim=306.0,
            velocity_limit_sim=2.770,
            stiffness=STIFFNESS_KNEE,
            damping=DAMPING_KNEE,
            armature=ARMATURE_KNEE,
        ),
        
        # ============ 踝关节 (Kp≈50, Kd≈3.18) ============
        "ankle_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch",
                ".*_ankle_roll",
            ],
            effort_limit_sim={
                ".*_ankle_pitch": 509.163,
                ".*_ankle_roll": 314.930,
            },
            velocity_limit_sim={
                ".*_ankle_pitch": 3.124,
                ".*_ankle_roll": 4.160,
            },
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            armature=ARMATURE_ANKLE,
        ),
        
        # ============ 腰部关节 ============
        "waist_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_roll",
                "waist_yaw",
            ],
            effort_limit_sim=50.0,  # 需要根据实际电机调整
            velocity_limit_sim=2.0,
            stiffness=STIFFNESS_WAIST,
            damping=DAMPING_WAIST,
            armature=ARMATURE_WAIST,
        ),
        
        # ============ 手臂关节 ============
        "shoulder_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch",
                ".*_shoulder_roll",
                ".*_shoulder_yaw",
            ],
            effort_limit_sim=25.0,  # 需要根据实际电机调整
            velocity_limit_sim=2.0,
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_ARM,
        ),
        
        "elbow_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch",
                ".*_elbow_yaw",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=2.0,
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_ARM,
        ),
        
        "wrist_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=STIFFNESS_ARM,
            damping=DAMPING_ARM,
            armature=ARMATURE_ARM,
        ),
    },
)

# ============ 动作缩放因子计算 ============
JINGCHU01_ACTION_SCALE = {}
for a in JINGCHU01_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            JINGCHU01_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# ============ 参数验证打印 ============
print("=" * 60)
print("电机参数验证 (自然频率 = {:.2f} rad/s)".format(NATURAL_FREQ))
print("=" * 60)
print(f"髋关节: Kp={STIFFNESS_HIP:.2f}, Kd={DAMPING_HIP:.2f}, Armature={ARMATURE_HIP:.6f}")
print(f"膝关节: Kp={STIFFNESS_KNEE:.2f}, Kd={DAMPING_KNEE:.2f}, Armature={ARMATURE_KNEE:.6f}")
print(f"踝关节: Kp={STIFFNESS_ANKLE:.2f}, Kd={DAMPING_ANKLE:.2f}, Armature={ARMATURE_ANKLE:.6f}")
print("=" * 60)