#!/usr/bin/env python3
"""Solve local counterfactual compliance targets using learned predictor."""

from __future__ import annotations

import argparse

import h5py
import torch

from whole_body_tracking.plugins.compliance.counterfactual_solver import CounterfactualSolver
from whole_body_tracking.plugins.compliance.pareto_predictor import ParetoPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve counterfactual z* targets.")
    parser.add_argument("--dataset_h5", required=True)
    parser.add_argument("--predictor_ckpt", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--method", default="gradient", choices=["gradient", "random", "cem"])
    args = parser.parse_args()

    with h5py.File(args.dataset_h5, "r") as f:
        obs = torch.tensor(f["obs"][:], dtype=torch.float32)
        priv = torch.tensor(f["priv"][:], dtype=torch.float32)
        z_init = torch.tensor(f["z_init"][:], dtype=torch.float32)
        z_prev = torch.tensor(f["z_prev"][:], dtype=torch.float32)
        z_nom = torch.tensor(f["z_nom"][:], dtype=torch.float32)
        rho = torch.tensor(f["rho"][:], dtype=torch.float32)
    pred = ParetoPredictor(obs.shape[-1], priv.shape[-1])
    pred.load_state_dict(torch.load(args.predictor_ckpt, map_location="cpu"))
    solver = CounterfactualSolver(method=args.method)
    out = solver.solve(pred, obs, priv, z_init, z_prev, z_nom, rho)
    torch.save({"z_star": out.z_star_by_expert, "best": out.best_expert, "obj": out.objective_values}, args.output_pt)


if __name__ == "__main__":
    main()
