#!/usr/bin/env python3
"""Plot one short event window with baseline and optional counterfactual targets."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Plot jump event windows and optional targets.")
    p.add_argument("--mode", default="with_targets", choices=["baseline_only", "with_targets"])
    p.add_argument("--dataset", required=True, help="HDF5 dataset path")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--teacher_npz", default=None)
    p.add_argument("--output_png", required=True)
    a = p.parse_args()

    import matplotlib.pyplot as plt

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for reading the dataset.") from exc

    with h5py.File(a.dataset, "r") as f:
        cf = np.asarray(f["contact_force"])[a.index, :, 0]
        pv = np.asarray(f["payload_vibration_proxy"])[a.index, :, 0]
        te = np.asarray(f["tracking_error"])[a.index, :, 0]
        ef = np.asarray(f["effort_proxy"])[a.index, :, 0]
        z_t = np.asarray(f.get("z_target", np.zeros((1, 5))))[min(a.index, len(f.get("z_target", np.zeros((1, 5)))) - 1)]

    if a.mode == "baseline_only":
        plt.figure(figsize=(10, 6))
        for i, (n, arr) in enumerate([
            ("contact", cf),
            ("payload vib", pv),
            ("tracking err", te),
            ("effort", ef),
        ], start=1):
            plt.subplot(4, 1, i)
            plt.plot(arr)
            plt.ylabel(n)
        plt.tight_layout()
        plt.savefig(a.output_png)
        return

    plt.figure(figsize=(11, 7))
    plt.subplot(3, 1, 1)
    plt.plot(te, label="baseline tracking error")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(5), z_t)
    plt.xticks(np.arange(5), ["omega", "zeta", "eps", "alpha", "beta"])
    plt.ylabel("optimal z*")
    plt.subplot(3, 1, 3)
    plt.plot(cf, label="impact")
    plt.plot(pv, label="payload vib")
    if a.teacher_npz:
        t = np.load(a.teacher_npz)
        if "teacher_metric" in t:
            plt.plot(t["teacher_metric"], label="teacher metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(a.output_png)


if __name__ == "__main__":
    main()
