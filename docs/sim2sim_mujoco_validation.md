# MuJoCo Sim2Sim Validation

## Structure (SONIC-style split)

Core runtime modules under `scripts/sim2sim`:

- `validate_policy_mujoco.py`: main entry (no runtime CLI parsing)
- `validate_pipeline.py`: orchestration (`context -> report -> rollout`)
- `validate_context.py`: config/effective-config context build
- `validate_report.py`: startup info report
- `sim2sim_config.py`: config loading + training alignment
- `sim2sim_rollout.py`: MuJoCo rollout main loop
- `sim2sim_mujoco_runtime.py`: MuJoCo stepping/render/realtime pacing
- `sim2sim_policy_stack.py`: ONNX main policy + optional aux-stage stack
- `sim2sim_observation.py`: observation history assembly
- `sonic_observation_config.py`: observation pipeline selection + external observation file loading
- `sonic_observation_registry.py`: observation term registry
- `sonic_fsm.py`: lifecycle FSM (`INIT/WAIT_FOR_CONTROL/CONTROL/STOPPED`)
- `sonic_io.py`: input/output channel abstraction (static / zmq_json / zmq_packed)

Deploy entry (`RoboMimic-style`) under `scripts/sim2sim/deploy_mujoco`:

- `deploy_mujoco.py`
- `config.py`
- `config/mujoco.yaml`

## Where Observation Is Configured

If `observation_registry` is already implemented, observation *selection* is configured in:

- Inline: `scripts/sim2sim/jingchu01_deploy.yaml` -> `observation`
  - `observation.pipeline`
  - `observation.pipelines.<name>.terms`
  - optional `observation.modes`
- External file: set `observation.config_file`
  - example: `scripts/sim2sim/deploy_mujoco/config/observation_policy_main.yaml`

`observation_registry` only defines available term names and dimensions; actual active terms/order/history come from the observation config.

## Default Config

Main sim2sim config:

- `scripts/sim2sim/jingchu01_deploy.yaml`

Deploy wrapper config:

- `scripts/sim2sim/deploy_mujoco/config/mujoco.yaml`

## Run

### Direct validator entry

```bash
conda run -n lyc-sim2sim python scripts/sim2sim/validate_policy_mujoco.py
```

Optional config switch via env var:

```bash
SIM2SIM_CONFIG_FILE=/abs/path/to/your_config.yaml \
conda run -n lyc-sim2sim python scripts/sim2sim/validate_policy_mujoco.py
```

### Deploy wrapper entry

```bash
conda run -n lyc-sim2sim python scripts/sim2sim/deploy_mujoco/deploy_mujoco.py
```

Optional deploy-config switch via env var:

```bash
SIM2SIM_DEPLOY_MUJOCO_CFG=/abs/path/to/mujoco.yaml \
conda run -n lyc-sim2sim python scripts/sim2sim/deploy_mujoco/deploy_mujoco.py
```

## Notes

- `training_alignment.enabled=true` aligns dt/joint-order/gains/action-scale from training snapshot (`env.yaml`).
- `simulation.run_forever=true` means no step-limit stop; complete motion cycles can loop.
- `simulation.reset_on_motion_cycle=true` with `cycle_reset_pose_mode` controls posture reset between reference cycles.
- For headless runs set `simulation.render=false`.
