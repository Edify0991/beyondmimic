# Jingchu01 训练与重放完整手册（下半身 12DOF / 保留全身接口）

## 1) 模式切换（核心）
`source/whole_body_tracking/whole_body_tracking/robots/jingchu01.py` 里已经支持两种模式：

```bash
# 默认：下半身 12DOF（推荐当前使用）
export WBT_JINGCHU01_CONTROL_MODE=lower_body

# 未来切回全身接口
export WBT_JINGCHU01_CONTROL_MODE=full_body
```

如果不设置，默认 `lower_body`。

## 2) 基础环境变量
```bash
cd /home/user/wmd/beyondmimic

# 你的 URDF 绝对路径
export WBT_JINGCHU01_URDF="/abs/path/to/JC01-URDF_legs.urdf"

# 当前建议固定下半身模式
export WBT_JINGCHU01_CONTROL_MODE=lower_body

# 你的下半身关节顺序（右腿在前，与你 URDF/配置一致）
export JINGCHU12="right_hip_roll,right_hip_yaw,right_hip_pitch,right_knee_pitch,right_ankle_pitch,right_ankle_roll,left_hip_roll,left_hip_yaw,left_hip_pitch,left_knee_pitch,left_ankle_pitch,left_ankle_roll"
```

## 3) AMASS 重定向 PKL -> BeyondMimic NPZ

### 3.1 推荐命令（你的 standard_dict pkl）
当 pkl 里是 `root_pos/root_rot/dof_pos`（你目前这个就是）：

```bash
python scripts/amass_pkl_to_npz.py \
  --robot jingchu01 \
  --layout standard_dict \
  --standard_root_rot_order xyzw \
  --input_file "data/motions/jingchu01_walk1_subject1_leg.pkl" \
  --output_name jingchu01_walk1_subject1_leg \
  --joint_names "$JINGCHU12" \
  --joint_index_mode articulation \
  --output_fps 50 \
  --headless
```

### 3.2 如果你不确定 root_rot 顺序
可以先用自动判定：

```bash
--standard_root_rot_order auto
```

脚本会打印 upright tilt 对比并自动选 `wxyz/xyzw`。

### 3.3 如果 pkl 是 `frames` 扁平格式
使用：

```bash
--layout amass_flat_frames
```

## 4) NPZ 重放检查

```bash
python scripts/replay_npz.py \
  --robot jingchu01 \
  --motion_file data/motions/jingchu01_walk1_subject1_leg.npz
```

## 5) 交互式截帧（逐帧检查 + 保存子片段）

```bash
python scripts/replay_npz.py \
  --robot jingchu01 \
  --motion_file data/motions/jingchu01_walk1_subject1_leg.npz \
  --interactive_trim
```

可选：启动即自动播放

```bash
--interactive_play
```

按键：
- `space`/`p`：播放/暂停
- `d`/右方向键：下一帧（并暂停）
- `a`/左方向键：上一帧（并暂停）
- `s`：标记起始帧
- `e`：标记结束帧
- `w`：保存截取段为新的 npz
- `q`：退出

可选输出参数：
- `--trim_output_dir data/motions`
- `--trim_output_name my_clip`
- `--trim_overwrite`

## 6) 训练 task 选择（新增 Leg12 / Leg12-Strict）

已注册 task：
- `Tracking-Flat-Jingchu01-Leg12-v0`
- `Tracking-Flat-Jingchu01-Leg12-Wo-State-Estimation-v0`
- `Tracking-Flat-Jingchu01-Leg12-Low-Freq-v0`
- `Tracking-Flat-Jingchu01-Leg12-Strict-v0`
- `Tracking-Flat-Jingchu01-Leg12-Strict-Wo-State-Estimation-v0`
- `Tracking-Flat-Jingchu01-Leg12-Strict-Low-Freq-v0`

说明：
- `Leg12`：动作与 motion command 对齐到 12 关节。
- `Leg12-Strict`：额外把 joint 相关观测/奖励/startup joint 随机化也限制到这 12 关节（更稳，推荐优先用它）。

## 7) 训练命令（推荐 Strict）

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Jingchu01-Leg12-Strict-v0 \
  --motion_file data/motions/jingchu01_walk1_subject1_leg.npz \
  --headless \
  --logger wandb \
  --log_project_name beyondmimic_jingchu01 \
  --run_name leg12_strict_walk1
```

训练日志默认在：
`logs/rsl_rl/jingchu01_flat/<时间戳>_leg12_strict_walk1/`

## 8) Play / Eval

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-Jingchu01-Leg12-Strict-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/jingchu01_flat/<run_dir>/model_XXXX.pt \
  --motion_file data/motions/jingchu01_walk1_subject1_leg.npz
```

## 9) 保留全身接口（未来切换）
当你未来要回全身：

```bash
export WBT_JINGCHU01_CONTROL_MODE=full_body
```

然后改用全身数据（全身关节维度与顺序一致），训练命令本身不用变；task 可继续用 `Tracking-Flat-Jingchu01-v0` 系列或按需新增 full-body strict 配置。

## 10) 快速自检

```bash
python -c "from whole_body_tracking.robots.jingchu01 import JINGCHU01_CONTROL_MODE,JINGCHU01_MOTION_JOINT_NAMES; print('mode=',JINGCHU01_CONTROL_MODE); print('joint_count=',len(JINGCHU01_MOTION_JOINT_NAMES)); print('joints=',JINGCHU01_MOTION_JOINT_NAMES)"

python -c "import numpy as np; d=np.load('data/motions/jingchu01_walk1_subject1_leg.npz'); print('joint_pos', d['joint_pos'].shape); print('joint_vel', d['joint_vel'].shape); print('body_pos_w', d['body_pos_w'].shape)"
```


## 11) 常见报错修复：`base_com:asset_cfg` / `torso_link: []`

如果出现：

```text
ValueError: Error while parsing 'base_com:asset_cfg' ...
torso_link: []
Available strings: ['Robotbase', 'left_hip_roll', ...]
```

说明：当前机器人 URDF 里没有 `torso_link` 这个 body 名称，实际根 body 是 `Robotbase`。

本仓库现在默认已改为 `Robotbase`，但如果你之前 shell 里残留了旧环境变量，仍可能覆盖成错误值。执行：

```bash
# 清理旧值（如有）
unset WBT_JINGCHU01_ANCHOR_BODY
unset WBT_JINGCHU01_BODY_NAMES
unset WBT_JINGCHU01_CONTACT_EXEMPT_BODIES
unset WBT_JINGCHU01_TERMINATION_BODIES

# 显式设置为当前 URDF 命名（最稳）
export WBT_JINGCHU01_ANCHOR_BODY="Robotbase"
export WBT_JINGCHU01_BODY_NAMES="Robotbase,left_hip_roll,left_hip_yaw,left_hip_pitch,left_knee_pitch,left_ankle_pitch,left_ankle_roll,right_hip_roll,right_hip_yaw,right_hip_pitch,right_knee_pitch,right_ankle_pitch,right_ankle_roll"
export WBT_JINGCHU01_CONTACT_EXEMPT_BODIES="left_ankle_roll,right_ankle_roll"
export WBT_JINGCHU01_TERMINATION_BODIES="left_ankle_roll,right_ankle_roll"
```

然后重新运行 train/play/replay 命令。
