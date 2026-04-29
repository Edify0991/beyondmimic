# 1. 必须在最前面启动 SimulationApp
from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True})
app = launcher.app

# 2. 导入 PyTorch 检查底层环境
import torch
print("\n" + "="*50)
print(f"[PyTorch 检查] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[PyTorch 检查] Device Name: {torch.cuda.get_device_name(0)}")

# 3. 显式且暴力地配置 SimulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils

print("\n[配置检查] 正在创建强行绑定 cuda:0 的配置...")
sim_cfg = SimulationCfg(
    dt=0.01,
    device="cuda:0",  # <--- 暴力锁死在第一张 4090 上
)
sim_cfg.physx.use_gpu = True 

# 4. 创建 Context 并验尸
sim = SimulationContext(sim_cfg)
physx_ctx = sim.get_physics_context()

print("\n" + "="*50)
print(f"Target Config Device : {sim_cfg.device}")
print(f"Context Actual : {physx_ctx.use_fabric}")    # 对应你截图里的 _device
# print(f"Context Target GPU      : {physx_ctx.use_gpu}")  # 对
print(f"Context Use GPU      : {physx_ctx._use_gpu}")  # 对应你截图里的 _use_gpu
print("="*50 + "\n")

app.close()
