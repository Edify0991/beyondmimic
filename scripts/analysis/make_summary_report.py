#!/usr/bin/env python3
"""Generate markdown summary report with figures and tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _best_fixed_grid(eval_root: str) -> dict | None:
    best = None
    for csv_path in Path(eval_root).rglob("fixed_grid_results.csv"):
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                score = (
                    float(row.get("tracking_preservation", 0.0))
                    - 0.6 * float(row.get("impact_cost", 0.0))
                    - 0.5 * float(row.get("payload_vibration_cost", 0.0))
                )
                cand = {"source": str(csv_path.parent), **row, "score": score}
                if best is None or score > best["score"]:
                    best = cand
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Generate markdown summary report with figures and tables.")
    p.add_argument("--metrics_json", required=True)
    p.add_argument("--fig_dir", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument("--week1_jump_mode", action="store_true")
    p.add_argument("--eval_root", default=None)
    a = p.parse_args()

    with open(a.metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    lines = ["# Compliance Plugin Summary", "", "## Metrics", "```json", json.dumps(m, indent=2), "```", ""]

    if a.week1_jump_mode:
        lines += ["## Week-1 Jump Figures", ""]
        figs = [
            ("baseline jump landing waveforms", "baseline_jump_landing_waveforms.png"),
            ("fixed-grid Pareto tradeoff", "pareto_tradeoff.png"),
            ("optimal z* distribution", "optimal_z_distribution.png"),
            ("expert specialization", "expert_specialization.png"),
            ("payload protection comparison", "payload_protection.png"),
        ]
        for title, file in figs:
            fig_path = Path(a.fig_dir) / file
            if fig_path.exists():
                lines.append(f"### {title}")
                lines.append(f"![{title}]({fig_path})")
                lines.append("")

        lines += ["## Week-1 Summary Table", "", "| item | value |", "|---|---|"]
        best = _best_fixed_grid(a.eval_root) if a.eval_root else None
        lines.append(f"| best fixed-grid setting | `{best}` |")
        base_imp = m.get("baseline", {}).get("impact_cost", None)
        teacher_imp = m.get("teacher", {}).get("impact_cost", None)
        improv = None if base_imp is None or teacher_imp is None else (base_imp - teacher_imp)
        lines.append(f"| teacher MoE improvement over baseline (impact drop) | `{improv}` |")
        top3 = m.get("top_payload_windows", [])[:3]
        lines.append(f"| top-3 event windows with highest payload vibration | `{top3}` |")
    else:
        lines += ["## Figures"]
        for name in [
            "pareto_tradeoff",
            "expert_specialization",
            "gate_quality",
            "payload_protection",
            "window_counterfactual",
            "student_vs_teacher",
        ]:
            lines.append(f"![{name}]({a.fig_dir}/{name}.png)")

    with open(a.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
