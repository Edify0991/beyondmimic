#!/usr/bin/env python3
"""Train a lightweight Pareto predictor model."""

from __future__ import annotations

import argparse
from pathlib import Path


import importlib.util


def _load_pareto_predictor_cls():
    mod_path = Path("source/whole_body_tracking/whole_body_tracking/plugins/compliance/pareto_predictor.py")
    spec = importlib.util.spec_from_file_location("pareto_predictor_module", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.ParetoPredictor


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

import h5py
import torch
import torch.nn.functional as F

from whole_body_tracking.plugins.compliance.pareto_predictor import ParetoPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train short-horizon Pareto predictor.")
    parser.add_argument("--dataset", "--dataset_h5", dest="dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--output_ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    cfg = _load_yaml(args.config)
    hidden_dim = int(cfg.get("hidden_dim", 256))
    lr = float(cfg.get("optimizer", {}).get("lr", 3e-4))

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for this script.") from exc

    with h5py.File(args.dataset, "r") as f:
        obs = torch.tensor(f["q"][:], dtype=torch.float32)
        priv = torch.tensor(f.get("dq", f["q"])[:], dtype=torch.float32)
        z = torch.zeros(obs.shape[0], 5, dtype=torch.float32)
        y = torch.cat(
            [
                torch.tensor(f.get("tracking_error", f["effort_proxy"])[:].mean(axis=(1, 2)), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(f.get("contact_force", f["effort_proxy"])[:].mean(axis=(1, 2)), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(f.get("payload_vibration_proxy", f["effort_proxy"])[:].mean(axis=(1, 2)), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(f.get("payload_vibration_proxy", f["effort_proxy"])[:].std(axis=(1, 2)), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(f["effort_proxy"][:].mean(axis=(1, 2)), dtype=torch.float32).unsqueeze(-1),
                torch.zeros(obs.shape[0], 1),
            ],
            dim=-1,
        )

    ParetoPredictor = _load_pareto_predictor_cls()
    model = ParetoPredictor(obs.shape[-1], priv.shape[-1], hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(args.epochs):
        pred = model(obs, priv, z)["mean"]
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "best.pt"
    else:
        out = Path(args.output_ckpt or "pareto_predictor.pt")
        out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"[INFO] saved predictor checkpoint: {out}")


if __name__ == "__main__":
    main()
