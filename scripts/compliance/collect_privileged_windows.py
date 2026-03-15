#!/usr/bin/env python3
"""Collect jump-centric observable/privileged windows and tag jump events.

This week-1 utility supports either:
1) reading an existing rollout npz via --input_npz, or
2) generating synthetic rollout-like traces for quick pipeline validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EVENT_NAMES = ["pre_takeoff", "aerial", "pre_landing", "landing", "post_landing_recovery"]
EVENT_TO_ID = {name: i for i, name in enumerate(EVENT_NAMES)}


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_or_generate(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.input_npz:
        data = dict(np.load(args.input_npz))
        if "contact" not in data:
            data["contact"] = (data.get("contact_force", np.zeros((data["q"].shape[0], data["q"].shape[1]))) > 5.0).astype(np.float32)
        return data

    # Synthetic fallback for week-1 dry-runs when simulator integration is unavailable.
    n_env = args.num_envs
    t = 240
    tt = np.linspace(0.0, 1.0, t)
    contact = np.ones((n_env, t), dtype=np.float32)
    contact[:, 70:130] = 0.0
    contact[:, 130:140] = 1.0
    base_height = (0.95 + 0.15 * np.exp(-((tt - 0.45) / 0.12) ** 2))[None, :].repeat(n_env, axis=0)
    base_vz = np.gradient(base_height, axis=1)
    contact_force = contact * (60 + 200 * np.exp(-((tt - 0.58) / 0.02) ** 2))[None, :]
    payload_vib = 0.15 + 0.3 * np.exp(-((tt - 0.6) / 0.07) ** 2)[None, :].repeat(n_env, axis=0)
    track_err = 0.05 + 0.08 * np.exp(-((tt - 0.6) / 0.08) ** 2)[None, :].repeat(n_env, axis=0)
    effort = 0.3 + 0.25 * np.exp(-((tt - 0.55) / 0.09) ** 2)[None, :].repeat(n_env, axis=0)
    q = np.zeros((n_env, t, 29), dtype=np.float32)
    dq = np.zeros((n_env, t, 29), dtype=np.float32)
    q_des = np.zeros((n_env, t, 29), dtype=np.float32)
    action = np.zeros((n_env, t, 29), dtype=np.float32)
    return {
        "q": q,
        "dq": dq,
        "q_des": q_des,
        "action": action,
        "contact": contact,
        "contact_force": contact_force,
        "base_vz": base_vz,
        "base_height": base_height,
        "payload_vibration_proxy": payload_vib,
        "tracking_error": track_err,
        "effort_proxy": effort,
    }


def _tag_events(contact: np.ndarray, base_vz: np.ndarray, base_height: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    """Tag each timestep with a jump-phase event ID.

    Default assumptions:
    - aerial when contact is false and vertical speed exceeds threshold magnitude,
    - landing anchored on first touchdown after aerial,
    - pre_landing is a configurable window before landing,
    - post_landing_recovery is a configurable window after landing.
    """
    n_env, t = contact.shape
    labels = np.full((n_env, t), EVENT_TO_ID["pre_takeoff"], dtype=np.int64)
    vz_thr = cfg["vz_threshold"]
    pre_land = int(cfg["pre_landing_steps"])
    post_land = int(cfg["post_landing_steps"])
    for e in range(n_env):
        c = contact[e] > 0.5
        vz = base_vz[e]
        h = base_height[e]

        airborne = (~c) & (np.abs(vz) >= vz_thr)
        labels[e, airborne] = EVENT_TO_ID["aerial"]

        touchdown_idx = None
        aerial_idx = np.where(airborne)[0]
        if aerial_idx.size > 0:
            after = np.where(c & (np.arange(t) > aerial_idx[-1]))[0]
            if after.size > 0:
                touchdown_idx = int(after[0])

        if touchdown_idx is not None:
            s = max(0, touchdown_idx - pre_land)
            labels[e, s:touchdown_idx] = EVENT_TO_ID["pre_landing"]
            labels[e, touchdown_idx : min(t, touchdown_idx + 2)] = EVENT_TO_ID["landing"]
            labels[e, touchdown_idx + 2 : min(t, touchdown_idx + 2 + post_land)] = EVENT_TO_ID["post_landing_recovery"]

        # extra heuristic: if height is below a soft bound and in contact late, treat as recovery
        low_height = h < cfg["recovery_height_max"]
        labels[e, low_height & c & (np.arange(t) > (touchdown_idx or 0))] = EVENT_TO_ID["post_landing_recovery"]

    return labels


def _window_data(data: dict[str, np.ndarray], labels: np.ndarray, window: int, stride: int) -> dict[str, np.ndarray]:
    keys = ["q", "dq", "q_des", "action", "contact_force", "payload_vibration_proxy", "tracking_error", "effort_proxy"]
    keys = [k for k in keys if k in data]
    n_env, t = labels.shape
    out = {k: [] for k in keys}
    out["event_label"] = []
    out["event_name"] = []

    for e in range(n_env):
        for end in range(window, t + 1, stride):
            start = end - window
            mid = (start + end) // 2
            ev_id = int(labels[e, mid])
            for k in keys:
                v = data[k]
                if v.ndim == 3:
                    out[k].append(v[e, start:end])
                else:
                    out[k].append(v[e, start:end, None])
            out["event_label"].append(ev_id)
            out["event_name"].append(EVENT_NAMES[ev_id])

    for k, v in out.items():
        if k == "event_name":
            out[k] = np.asarray(v, dtype="S32")
        else:
            out[k] = np.asarray(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect privileged jump windows and event labels.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--wandb_path", default=None)
    parser.add_argument("--input_npz", default=None, help="Optional local rollout npz for offline processing.")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--event_types", default="pre_landing,landing,aerial,post_landing_recovery")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out_file", required=True)

    # configurable detection parameters
    parser.add_argument("--vz_threshold", type=float, default=0.015)
    parser.add_argument("--pre_landing_steps", type=int, default=6)
    parser.add_argument("--post_landing_steps", type=int, default=18)
    parser.add_argument("--recovery_height_max", type=float, default=1.05)
    args = parser.parse_args()

    data = _load_or_generate(args)
    labels = _tag_events(
        data["contact"],
        data["base_vz"],
        data["base_height"],
        {
            "vz_threshold": args.vz_threshold,
            "pre_landing_steps": args.pre_landing_steps,
            "post_landing_steps": args.post_landing_steps,
            "recovery_height_max": args.recovery_height_max,
        },
    )
    windows = _window_data(data, labels, args.window_size, args.stride)

    selected = set(_parse_csv_list(args.event_types))
    keep = np.array([name.decode("utf-8") in selected for name in windows["event_name"]], dtype=bool)
    for k in list(windows.keys()):
        windows[k] = windows[k][keep]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to write HDF5 outputs. Please install h5py.") from exc

    with h5py.File(out_path, "w") as f:
        for k, v in windows.items():
            f.create_dataset(k, data=v)
        f.create_dataset("event_names", data=np.asarray(EVENT_NAMES, dtype="S32"))
        f.attrs["task"] = args.task or "unknown"
        f.attrs["wandb_path"] = args.wandb_path or "none"

    print(f"[INFO] Saved {len(windows['event_label'])} windows to {out_path}")


if __name__ == "__main__":
    main()
