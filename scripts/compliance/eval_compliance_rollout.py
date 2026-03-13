#!/usr/bin/env python3
"""Evaluate baseline/teacher/student/plugin rollouts from logged trajectory files."""

from __future__ import annotations

import argparse
import json

import numpy as np


def _aggregate(npz_path: str) -> dict[str, float]:
    d = np.load(npz_path)
    out = {}
    for k in ["tracking_rmse", "impact_cost", "oscillation_cost", "payload_vibration_cost", "effort_cost", "limit_risk"]:
        if k in d:
            out[k] = float(np.mean(d[k]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs teacher vs student vs adapter trajectories.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--teacher", required=False)
    parser.add_argument("--student", required=False)
    parser.add_argument("--adapter", required=False)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

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
