#!/usr/bin/env python3
"""Visualize expert usage and specialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize expert usage and specialization.")
    p.add_argument("--gate_pt", default=None)
    p.add_argument("--output_png", default=None)
    p.add_argument("--teacher_ckpt", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    import matplotlib.pyplot as plt

    if a.teacher_ckpt:
        import torch
        ckpt = torch.load(a.teacher_ckpt, map_location="cpu")
        gate = ckpt.get("gate", torch.zeros(1, 4)).float().numpy()
        out = Path(a.out_dir or "outputs/figures/week1") / "expert_specialization.png"
    else:
        import torch
        gate = torch.load(a.gate_pt, map_location="cpu")["gate"].float().numpy()
        out = Path(a.output_png or "expert_specialization.png")

    out.parent.mkdir(parents=True, exist_ok=True)
    usage = gate.mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(usage)), usage)
    plt.xlabel("expert")
    plt.ylabel("usage")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
