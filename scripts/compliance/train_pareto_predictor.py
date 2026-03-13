#!/usr/bin/env python3
"""Train a lightweight Pareto predictor model."""

from __future__ import annotations

import argparse

import h5py
import torch
import torch.nn.functional as F

from whole_body_tracking.plugins.compliance.pareto_predictor import ParetoPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train short-horizon Pareto predictor.")
    parser.add_argument("--dataset_h5", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    with h5py.File(args.dataset_h5, "r") as f:
        obs = torch.tensor(f["obs"][:], dtype=torch.float32)
        priv = torch.tensor(f["priv"][:], dtype=torch.float32)
        z = torch.tensor(f["z"][:], dtype=torch.float32)
        y = torch.tensor(f["y"][:], dtype=torch.float32)

    model = ParetoPredictor(obs.shape[-1], priv.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(args.epochs):
        pred = model(obs, priv, z)["mean"]
        loss = F.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), args.output_ckpt)


if __name__ == "__main__":
    main()
