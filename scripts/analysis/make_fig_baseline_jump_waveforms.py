#!/usr/bin/env python3
"""Plot representative baseline jump waveforms around landing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pick_indices(impact_scores: np.ndarray) -> dict[str, int]:
    sorted_idx = np.argsort(impact_scores)
    return {
        "median-impact": int(sorted_idx[len(sorted_idx) // 2]),
        "worst-case": int(sorted_idx[-1]),
        "random": int(np.random.default_rng(0).integers(0, len(impact_scores))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create baseline jump waveform figure around landing windows.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", default="outputs/figures/week1")
    parser.add_argument("--selection", default="median-impact", choices=["median-impact", "worst-case", "random"])
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for reading the dataset.") from exc

    with h5py.File(args.dataset, "r") as f:
        cf = np.asarray(f["contact_force"])
        pv = np.asarray(f["payload_vibration_proxy"])
        te = np.asarray(f["tracking_error"])
        ep = np.asarray(f["effort_proxy"])

    impact = cf.mean(axis=(1, 2))
    idx = _pick_indices(impact)[args.selection]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    for i, (name, arr) in enumerate(
        [
            ("contact force/proxy", cf[idx, :, 0]),
            ("payload vibration proxy", pv[idx, :, 0]),
            ("tracking error", te[idx, :, 0]),
            ("effort/current proxy", ep[idx, :, 0]),
        ],
        start=1,
    ):
        plt.subplot(4, 1, i)
        plt.plot(arr)
        plt.ylabel(name)
    plt.xlabel("window timestep")
    plt.tight_layout()
    out_path = out_dir / "baseline_jump_landing_waveforms.png"
    plt.savefig(out_path)
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()
