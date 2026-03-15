#!/usr/bin/env python3
"""Visualize distributions of optimal counterfactual compliance latents z*."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create z* distribution plots grouped by event type/expert.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", default="outputs/figures/week1")
    parser.add_argument("--group_by_expert", action="store_true")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for reading the dataset.") from exc

    with h5py.File(args.dataset, "r") as f:
        z = np.asarray(f["z_target"])
        event = np.asarray(f.get("event_label", np.zeros(z.shape[0], dtype=np.int64)))
        best = np.asarray(f.get("best_expert", np.zeros(z.shape[0], dtype=np.int64)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["omega*", "zeta*", "epsilon*", "alpha*", "beta*"]
    event_groups = {"landing": 3, "aerial": 1, "recovery": 4}

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for i, ax in enumerate(axes):
        data = [z[event == eid, i] for _, eid in event_groups.items()]
        labels = list(event_groups.keys())
        ax.boxplot([d if len(d) else np.array([np.nan]) for d in data], labels=labels)
        ax.set_title(names[i])
    fig.tight_layout()
    fig.savefig(out_dir / "optimal_z_distribution.png")

    if args.group_by_expert:
        plt.figure(figsize=(8, 4))
        for ex in np.unique(best):
            vals = z[best == ex, 0]
            if len(vals):
                plt.hist(vals, bins=20, alpha=0.4, label=f"expert {int(ex)}")
        plt.legend()
        plt.title("omega* grouped by best expert")
        plt.tight_layout()
        plt.savefig(out_dir / "optimal_z_distribution_by_expert.png")


if __name__ == "__main__":
    main()
