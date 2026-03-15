#!/usr/bin/env python3
"""Evaluate compliance rollouts.

Week-1 adds `--mode fixed_grid` for frozen base jump policy sweeps.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np



def aggregate_fixed_grid_results(rows):
    if not rows:
        return {"best": None, "count": 0}
    def score(r):
        return (1.5*r["tracking_preservation"] - 0.8*r["impact_cost"] - 0.6*r["payload_vibration_cost"] - 0.4*r["oscillation_cost"] - 0.2*r["effort_cost"] + 0.5*r["recovery_success"] - 0.8*r["fall_rate"])
    best=max(rows,key=score)
    return {"count": len(rows), "best": best}


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _simulate_metrics(omega: float, zeta: float, alpha: float, epsilon: float, beta: float) -> dict[str, float]:
    """Week-1 surrogate metric model for fixed-grid scoring.

    This keeps the script runnable without coupling to Isaac execution internals.
    """
    # Nominal landing-friendly setting around which we create smooth tradeoffs.
    d = ((omega - 24.0) / 10.0) ** 2 + ((zeta - 1.0) / 0.5) ** 2 + ((alpha - 0.8) / 0.25) ** 2
    impact = 0.9 + 1.2 * d + 0.2 * beta
    payload = 0.7 + 1.0 * d + 0.15 * beta
    oscillation = 0.5 + 0.8 * ((omega - 22.0) / 12.0) ** 2 + 0.3 * (zeta - 1.1) ** 2
    effort = 0.4 + 0.6 * ((omega - 24.0) / 10.0) ** 2 + 0.2 * (1.0 - alpha) + 0.2 * beta
    track_pres = float(np.clip(1.0 - 0.3 * d - 0.05 * epsilon, 0.0, 1.0))
    recovery_success = float(track_pres > 0.55 and impact < 2.3)
    fall_rate = float(np.clip(1.0 - recovery_success + 0.1 * max(0.0, impact - 2.3), 0.0, 1.0))
    return {
        "tracking_preservation": track_pres,
        "impact_cost": float(impact),
        "payload_vibration_cost": float(payload),
        "oscillation_cost": float(oscillation),
        "effort_cost": float(effort),
        "recovery_success": recovery_success,
        "fall_rate": fall_rate,
    }


def _fixed_grid(args: argparse.Namespace) -> None:
    omegas = _parse_grid(args.grid_omega)
    zetas = _parse_grid(args.grid_zeta)
    alphas = _parse_grid(args.grid_alpha)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for omega, zeta, alpha in itertools.product(omegas, zetas, alphas):
        metrics = _simulate_metrics(omega, zeta, alpha, args.epsilon_fixed, args.beta_fixed)
        row = {
            "omega": omega,
            "zeta": zeta,
            "alpha": alpha,
            "epsilon": args.epsilon_fixed,
            "beta": args.beta_fixed,
            **metrics,
        }
        rows.append(row)

    csv_path = out_dir / "fixed_grid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate_fixed_grid_results(rows)
    with (out_dir / "fixed_grid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved fixed-grid results to {csv_path}")


def _aggregate(npz_path: str) -> dict[str, float]:
    d = np.load(npz_path)
    out = {}
    for k in [
        "tracking_rmse",
        "impact_cost",
        "oscillation_cost",
        "payload_vibration_cost",
        "effort_cost",
        "limit_risk",
    ]:
        if k in d:
            out[k] = float(np.mean(d[k]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline/teacher/student/plugin trajectories.")
    parser.add_argument("--mode", default="compare", choices=["compare", "fixed_grid"])
    parser.add_argument("--wandb_path", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--headless", action="store_true")

    # fixed-grid options
    parser.add_argument("--grid_omega", default="18,24,30")
    parser.add_argument("--grid_zeta", default="0.6,1.0,1.4")
    parser.add_argument("--grid_alpha", default="0.6,0.8,1.0")
    parser.add_argument("--epsilon_fixed", type=float, default=0.05)
    parser.add_argument("--beta_fixed", type=float, default=0.05)
    parser.add_argument("--out_dir", default="outputs/eval/default")

    # comparison mode options
    parser.add_argument("--baseline", required=False)
    parser.add_argument("--teacher", required=False)
    parser.add_argument("--student", required=False)
    parser.add_argument("--adapter", required=False)
    parser.add_argument("--output_json", required=False)
    args = parser.parse_args()

    if args.mode == "fixed_grid":
        _fixed_grid(args)
        return

    if not args.baseline or not args.output_json:
        raise ValueError("--baseline and --output_json are required for mode=compare")
    report = {"baseline": _aggregate(args.baseline)}
    if args.teacher:
        report["teacher"] = _aggregate(args.teacher)
    if args.student:
        report["student"] = _aggregate(args.student)
    if args.adapter:
        report["adapter"] = _aggregate(args.adapter)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
