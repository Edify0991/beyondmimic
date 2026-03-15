#!/usr/bin/env python3
"""Debug utility for inspecting raw compliance rollout logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _list_keys(h5obj, prefix=""):
    keys = []
    for k, v in h5obj.items():
        p = f"{prefix}/{k}"
        if hasattr(v, "keys"):
            keys.extend(_list_keys(v, p))
        else:
            keys.append(p)
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect raw rollout h5 and save quick sanity plots.")
    p.add_argument("--raw_h5", required=True)
    p.add_argument("--out_dir", default="outputs/figures/debug_logging")
    p.add_argument("--env_id", type=int, default=0)
    args = p.parse_args()

    import h5py
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.raw_h5, "r") as f:
        keys = _list_keys(f)
        print("[INFO] available datasets:")
        for k in keys:
            print("  ", k)

        env = np.asarray(f["/meta/env_id"])
        m = env == args.env_id

        z = np.asarray(f["/root/pos"])[m, 2]
        cb = np.asarray(f["/contact/bool"])[m, 0]
        cf = np.asarray(f["/contact/force"])[m, 0]
        pr = np.asarray(f["/payload/rel_pos"])[m, 2]
        pa = np.asarray(f["/payload/acc"])[m, 2]
        q = np.asarray(f["/joint/q"])[m]
        dq = np.asarray(f["/joint/dq"])[m]

    knee_idx = min(3, q.shape[1] - 1)
    t = np.arange(len(z))

    plt.figure(figsize=(10, 9))
    plt.subplot(5, 1, 1)
    plt.plot(t, z)
    plt.ylabel("root z")
    plt.subplot(5, 1, 2)
    plt.plot(t, cb, label="contact bool")
    plt.plot(t, cf, label="contact force")
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(t, pr, label="payload rel z")
    plt.plot(t, pa, label="payload acc z")
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(t, q[:, knee_idx])
    plt.ylabel("knee q")
    plt.subplot(5, 1, 5)
    plt.plot(t, dq[:, knee_idx])
    plt.ylabel("knee dq")
    plt.xlabel("step")
    plt.tight_layout()
    out = out_dir / "debug_rollout_overview.png"
    plt.savefig(out)
    print(f"[INFO] saved {out}")


if __name__ == "__main__":
    main()
