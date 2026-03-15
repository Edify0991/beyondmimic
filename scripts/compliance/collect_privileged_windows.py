#!/usr/bin/env python3
"""Collect privileged jump windows from raw rollout HDF5 or synthetic fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EVENT_NAMES = ["pre_takeoff", "aerial", "pre_landing", "landing", "post_landing_recovery"]
EVENT_TO_ID = {name: i for i, name in enumerate(EVENT_NAMES)}


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_raw_h5(path: str) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        data = {
            "sim_step": np.asarray(f["/time/sim_step"]),
            "env_id": np.asarray(f["/meta/env_id"]),
            "episode_id": np.asarray(f["/meta/episode_id"]),
            "q": np.asarray(f["/joint/q"]),
            "dq": np.asarray(f["/joint/dq"]),
            "q_des": np.asarray(f["/joint/q_des"]),
            "action": np.asarray(f["/joint/action_pi"]),
            "tau_cmd": np.asarray(f["/joint/tau_cmd"]),
            "effort_proxy": np.asarray(f["/joint/effort_proxy"]),
            "root_pos": np.asarray(f["/root/pos"]),
            "root_lin_vel": np.asarray(f["/root/lin_vel"]),
            "imu_torso": np.asarray(f["/imu/torso"]),
            "contact_bool": np.asarray(f["/contact/bool"]),
            "contact_force": np.asarray(f["/contact/force"]),
            "payload_acc": np.asarray(f["/payload/acc"]),
            "payload_vel": np.asarray(f["/payload/vel"]),
        }
    return data


def _reshape_sequences(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    env_ids = data["env_id"].astype(int)
    unique_env = np.unique(env_ids)
    seq = {}
    for key, val in data.items():
        if key in {"env_id", "episode_id", "sim_step"}:
            continue
        chunks = [val[env_ids == e] for e in unique_env]
        min_t = min(c.shape[0] for c in chunks)
        chunks = [c[:min_t] for c in chunks]
        seq[key] = np.stack(chunks, axis=0)
    return seq


def _synthetic(args: argparse.Namespace) -> dict[str, np.ndarray]:
    n_env, t = args.num_envs, 240
    tt = np.linspace(0.0, 1.0, t)
    contact = np.ones((n_env, t, 1), dtype=np.float32)
    contact[:, 70:130] = 0
    contact[:, 130:140] = 1
    root_pos = np.zeros((n_env, t, 3), dtype=np.float32)
    root_pos[..., 2] = 0.95 + 0.15 * np.exp(-((tt - 0.45) / 0.12) ** 2)
    root_lin_vel = np.gradient(root_pos, axis=1)
    q = np.zeros((n_env, t, 29), dtype=np.float32)
    dq = np.zeros((n_env, t, 29), dtype=np.float32)
    q_des = np.zeros((n_env, t, 29), dtype=np.float32)
    action = np.zeros((n_env, t, 29), dtype=np.float32)
    tau = np.zeros((n_env, t, 29), dtype=np.float32)
    effort = np.mean(np.abs(tau), axis=-1, keepdims=True)
    force = contact * (60 + 200 * np.exp(-((tt - 0.58) / 0.02) ** 2))[None, :, None]
    payload_acc = np.repeat(0.2 * np.exp(-((tt - 0.6) / 0.05) ** 2)[None, :, None], n_env, axis=0)
    payload_vel = np.cumsum(payload_acc, axis=1)
    imu_torso = np.concatenate([root_lin_vel, np.zeros_like(root_lin_vel)], axis=-1)
    return {
        "q": q,
        "dq": dq,
        "q_des": q_des,
        "action": action,
        "tau_cmd": tau,
        "effort_proxy": effort,
        "root_pos": root_pos,
        "root_lin_vel": root_lin_vel,
        "imu_torso": imu_torso,
        "contact_bool": contact,
        "contact_force": force,
        "payload_acc": payload_acc,
        "payload_vel": payload_vel,
    }


def _tag_events(contact: np.ndarray, root_vz: np.ndarray, root_z: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    n_env, t = contact.shape
    labels = np.full((n_env, t), EVENT_TO_ID["pre_takeoff"], dtype=np.int64)
    for e in range(n_env):
        c = contact[e] > 0.5
        aerial = (~c) & (np.abs(root_vz[e]) > cfg["vz_threshold"])
        labels[e, aerial] = EVENT_TO_ID["aerial"]
        aerial_idx = np.where(aerial)[0]
        touchdown = None
        if len(aerial_idx) > 0:
            after = np.where(c & (np.arange(t) > aerial_idx[-1]))[0]
            if len(after) > 0:
                touchdown = int(after[0])
        if touchdown is not None:
            s = max(0, touchdown - int(cfg["pre_landing_steps"]))
            labels[e, s:touchdown] = EVENT_TO_ID["pre_landing"]
            labels[e, touchdown : min(t, touchdown + 2)] = EVENT_TO_ID["landing"]
            labels[e, touchdown + 2 : min(t, touchdown + 2 + int(cfg["post_landing_steps"]))] = EVENT_TO_ID[
                "post_landing_recovery"
            ]
        labels[e, (root_z[e] < cfg["recovery_height_max"]) & c] = EVENT_TO_ID["post_landing_recovery"]
    return labels


def _psd_ratio(signal: np.ndarray, fs: float = 50.0) -> np.ndarray:
    x = signal - signal.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freq = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    low = spec[..., (freq >= 0) & (freq < 5)].mean(axis=-1)
    high = spec[..., (freq >= 5) & (freq <= 20)].mean(axis=-1)
    return high / (low + 1e-6)


def _window(seq: dict[str, np.ndarray], labels: np.ndarray, window: int, stride: int) -> dict[str, np.ndarray]:
    out = {k: [] for k in ["q", "dq", "q_des", "action", "tau_cmd", "effort_proxy", "impact_proxy", "oscillation_proxy", "payload_vibration_proxy", "contact_force", "contact_bool", "event_label", "event_name"]}
    n_env, t = labels.shape
    for e in range(n_env):
        for end in range(window, t + 1, stride):
            start = end - window
            mid = (start + end) // 2
            ev = int(labels[e, mid])
            q = seq["q"][e, start:end]
            dq = seq["dq"][e, start:end]
            q_des = seq["q_des"][e, start:end]
            action = seq["action"][e, start:end]
            tau = seq["tau_cmd"][e, start:end]
            effort = seq["effort_proxy"][e, start:end]
            cf = seq["contact_force"][e, start:end]
            cb = seq["contact_bool"][e, start:end]
            # proxies
            if np.isfinite(cf).any() and np.abs(cf).sum() > 0:
                impact = np.max(np.abs(cf), axis=(0, 1), keepdims=True)
            else:
                ddq = np.diff(dq, axis=0)
                impact = np.max(np.abs(ddq), axis=(0, 1), keepdims=True)
                if not np.isfinite(impact).all():
                    imu = seq["imu_torso"][e, start:end, :3]
                    impact = np.sqrt(np.mean(np.diff(imu, axis=0) ** 2, keepdims=True))
            osc = _psd_ratio(dq.T).mean(keepdims=True)
            if "payload_acc" in seq:
                pa = seq["payload_acc"][e, start:end]
            else:
                pa = np.diff(seq["payload_vel"][e, start:end], axis=0, prepend=seq["payload_vel"][e, start : start + 1])
            payload_vib = np.sqrt(np.mean((pa - pa.mean(axis=0, keepdims=True)) ** 2, axis=(0, 1), keepdims=True))

            out["q"].append(q)
            out["dq"].append(dq)
            out["q_des"].append(q_des)
            out["action"].append(action)
            out["tau_cmd"].append(tau)
            out["effort_proxy"].append(effort)
            out["impact_proxy"].append(np.asarray(impact).reshape(1, 1))
            out["oscillation_proxy"].append(np.asarray(osc).reshape(1, 1))
            out["payload_vibration_proxy"].append(np.asarray(payload_vib).reshape(1, 1))
            out["contact_force"].append(cf)
            out["contact_bool"].append(cb)
            out["event_label"].append(ev)
            out["event_name"].append(EVENT_NAMES[ev])

    for k in out:
        if k == "event_name":
            out[k] = np.asarray(out[k], dtype="S32")
        else:
            out[k] = np.asarray(out[k])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect privileged windows from raw rollout data.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--wandb_path", default=None)
    parser.add_argument("--raw_h5", default=None, help="Path to raw_rollouts.h5")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--event_types", default="pre_landing,landing,aerial,post_landing_recovery")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out_file", default="outputs/datasets/jump_privileged_windows.h5")
    parser.add_argument("--vz_threshold", type=float, default=0.015)
    parser.add_argument("--pre_landing_steps", type=int, default=6)
    parser.add_argument("--post_landing_steps", type=int, default=18)
    parser.add_argument("--recovery_height_max", type=float, default=1.05)
    args = parser.parse_args()

    if args.raw_h5:
        seq = _reshape_sequences(_load_raw_h5(args.raw_h5))
    else:
        seq = _synthetic(args)

    labels = _tag_events(
        seq["contact_bool"].squeeze(-1),
        seq["root_lin_vel"][..., 2],
        seq["root_pos"][..., 2],
        {
            "vz_threshold": args.vz_threshold,
            "pre_landing_steps": args.pre_landing_steps,
            "post_landing_steps": args.post_landing_steps,
            "recovery_height_max": args.recovery_height_max,
        },
    )

    windows = _window(seq, labels, args.window_size, args.stride)
    selected = set(_parse_csv_list(args.event_types))
    keep = np.array([name.decode("utf-8") in selected for name in windows["event_name"]], dtype=bool)
    for k in list(windows.keys()):
        windows[k] = windows[k][keep]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    with h5py.File(out_path, "w") as f:
        for k, v in windows.items():
            f.create_dataset(k, data=v)
        f.create_dataset("event_names", data=np.asarray(EVENT_NAMES, dtype="S32"))
        f.attrs["task"] = args.task or "unknown"
        f.attrs["wandb_path"] = args.wandb_path or "none"
        f.attrs["source_raw_h5"] = args.raw_h5 or "synthetic"

    print(f"[INFO] Saved {len(windows['event_label'])} windows to {out_path}")


if __name__ == "__main__":
    main()
