#!/usr/bin/env python3
"""Plot Pareto tradeoff curves.

Supports:
- legacy metrics json input
- week1 fixed-grid csv discovery via --eval_root
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path



def _from_metrics_json(path: str) -> list[tuple[str, float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return [(name, v.get("tracking_rmse", 0.0), v.get("impact_cost", 0.0)) for name, v in m.items()]


def _from_eval_root(root: str) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for csv_file in Path(root).rglob("fixed_grid_results.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    (
                        f"{csv_file.parent.name}|w={row['omega']},z={row['zeta']},a={row['alpha']}",
                        float(row["tracking_preservation"]),
                        float(row["impact_cost"]),
                    )
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Pareto tradeoff between tracking and impact.")
    p.add_argument("--metrics_json", default=None)
    p.add_argument("--output_png", default=None)
    p.add_argument("--eval_root", default=None)
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    import matplotlib.pyplot as plt

    if a.eval_root:
        pts = _from_eval_root(a.eval_root)
        out = Path(a.out_dir or "outputs/figures/week1") / "pareto_tradeoff.png"
    elif a.metrics_json:
        pts = _from_metrics_json(a.metrics_json)
        out = Path(a.output_png or "pareto_tradeoff.png")
    else:
        raise ValueError("Provide either --eval_root or --metrics_json")

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for _, x, y in pts:
        plt.scatter(x, y, c="tab:blue", alpha=0.6)
    plt.xlabel("tracking preservation")
    plt.ylabel("impact cost")
    plt.tight_layout()
    plt.savefig(out)
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
    main()
