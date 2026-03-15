#!/usr/bin/env python3
"""Compare payload protection metrics across eval dirs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _mean_from_csv(path: Path, key: str) -> float:
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals.append(float(r[key]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare payload vibration/impact metrics.")
    p.add_argument("--baseline_npz", default=None)
    p.add_argument("--plugin_npz", default=None)
    p.add_argument("--output_png", default=None)
    p.add_argument("--eval_root", default=None)
    p.add_argument("--compare_dirs", default=None)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    import matplotlib.pyplot as plt

    if a.eval_root and a.compare_dirs:
        names = [x.strip() for x in a.compare_dirs.split(",") if x.strip()]
        vib = []
        imp = []
        for name in names:
            csv_path = Path(a.eval_root) / name / "fixed_grid_results.csv"
            vib.append(_mean_from_csv(csv_path, "payload_vibration_cost"))
            imp.append(_mean_from_csv(csv_path, "impact_cost"))
        out = Path(a.out_dir or "outputs/figures/week1") / "payload_protection.png"
        labels = names
    else:
        b = np.load(a.baseline_npz)
        s = np.load(a.plugin_npz)
        vib = [float(b["payload_vibration_cost"].mean()), float(s["payload_vibration_cost"].mean())]
        imp = [float(b["impact_cost"].mean()), float(s["impact_cost"].mean())]
        labels = ["baseline", "plugin"]
        out = Path(a.output_png or "payload_protection.png")

    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, vib, width=w, label="payload vibration")
    plt.bar(x + w / 2, imp, width=w, label="impact")
    plt.xticks(x, labels, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
