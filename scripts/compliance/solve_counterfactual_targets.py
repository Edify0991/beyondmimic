#!/usr/bin/env python3
"""Solve local counterfactual compliance targets using learned predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import importlib.util


def _load_cls(path, cls_name):
    mod_path = Path(path)
    spec = importlib.util.spec_from_file_location(f"tmp_{cls_name}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


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

from whole_body_tracking.plugins.compliance.counterfactual_solver import CounterfactualSolver
from whole_body_tracking.plugins.compliance.pareto_predictor import ParetoPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve counterfactual z* targets.")
    parser.add_argument("--dataset", "--dataset_h5", dest="dataset", required=True)
    parser.add_argument("--predictor_ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out_file", "--output_pt", dest="out_file", required=True)
    parser.add_argument("--method", default=None, choices=["gradient", "random", "cem", None])
    args = parser.parse_args()

    import torch

    cfg = _load_yaml(args.config)
    method = args.method or cfg.get("method", "gradient")

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for this script.") from exc

    with h5py.File(args.dataset, "r") as f:
        obs = torch.tensor(f["q"][:], dtype=torch.float32)
        priv = torch.tensor(f.get("dq", f["q"])[:], dtype=torch.float32)
        event_label = np.asarray(f.get("event_label", np.zeros(obs.shape[0], dtype=np.int64)))

    # filter by event when requested (e.g., landing windows only)
    event_filter = cfg.get("event_filter", {})
    only_landing = bool(event_filter.get("landing_only", False))
    if only_landing:
        keep = event_label == 3
        obs = obs[keep]
        priv = priv[keep]
        event_label = event_label[keep]

    n = obs.shape[0]
    n_exp = 4
    z_init = torch.rand(n, n_exp, 5)
    z_prev = torch.zeros(n, 5)
    z_nom = torch.tensor([[24.0, 1.0, 0.05, 0.8, 0.05]], dtype=torch.float32).repeat(n, 1)
    rho = torch.tensor(cfg.get("expert_preferences", [[1, 0.8, 0.8, 0.8, 0.3, 0.2]] * n_exp), dtype=torch.float32)

    ParetoPredictor = _load_cls("source/whole_body_tracking/whole_body_tracking/plugins/compliance/pareto_predictor.py", "ParetoPredictor")
    CounterfactualSolver = _load_cls("source/whole_body_tracking/whole_body_tracking/plugins/compliance/counterfactual_solver.py", "CounterfactualSolver")
    pred = ParetoPredictor(obs.shape[-1], priv.shape[-1])
    pred.load_state_dict(torch.load(args.predictor_ckpt, map_location="cpu"))
    solver = CounterfactualSolver(method=method, steps=int(cfg.get("steps", 16)), lr=float(cfg.get("lr", 0.05)))
    out = solver.solve(
        pred,
        obs,
        priv,
        z_init,
        z_prev,
        z_nom,
        rho,
        lambda_delta=float(cfg.get("lambda_delta", 1e-2)),
        lambda_nom=float(cfg.get("lambda_nom", 1e-2)),
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_traces = bool(cfg.get("save_optimization_traces", False))
    with h5py.File(out_path, "w") as f:
        f.create_dataset("obs_hist", data=obs.numpy())
        f.create_dataset("priv_hist", data=priv.numpy())
        f.create_dataset("z_star", data=out.z_star_by_expert.numpy())
        f.create_dataset("best_expert", data=out.best_expert.numpy())
        f.create_dataset("objective_values", data=out.objective_values.numpy())
        f.create_dataset("event_label", data=event_label)
        # teacher targets = best expert z*
        idx = out.best_expert[:, None, None].expand(-1, 1, out.z_star_by_expert.shape[-1])
        z_target = out.z_star_by_expert.gather(1, idx).squeeze(1)
        f.create_dataset("z_target", data=z_target.numpy())
        if save_traces:
            f.create_dataset("z_init", data=z_init.numpy())
    print(f"[INFO] saved counterfactual targets: {out_path}")


if __name__ == "__main__":
    main()
