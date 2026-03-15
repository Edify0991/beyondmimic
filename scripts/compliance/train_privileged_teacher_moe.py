#!/usr/bin/env python3
"""Train privileged contrastive MoE teacher against counterfactual targets."""

from __future__ import annotations

import argparse
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train privileged MoE teacher.")
    parser.add_argument("--dataset", "--dataset_pt", dest="dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--output_ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    cfg = _load_yaml(args.config)

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for this script.") from exc

    with h5py.File(args.dataset, "r") as f:
        priv_hist = torch.tensor(f["priv_hist"][:], dtype=torch.float32)
        z_target = torch.tensor(f["z_target"][:], dtype=torch.float32)
        best_expert = torch.tensor(f["best_expert"][:], dtype=torch.long)

    num_experts = int(cfg.get("num_experts", 4))
    ComplianceMoE = _load_cls("source/whole_body_tracking/whole_body_tracking/plugins/compliance/compliance_moe.py", "ComplianceMoE")
    PrivilegedHistoryEncoder = _load_cls("source/whole_body_tracking/whole_body_tracking/plugins/compliance/privileged_encoder.py", "PrivilegedHistoryEncoder")
    enc = PrivilegedHistoryEncoder(priv_hist.shape[-1])
    moe = ComplianceMoE(192, num_experts=num_experts)
    opt = torch.optim.Adam(list(enc.parameters()) + list(moe.parameters()), lr=3e-4)
    w = cfg.get("loss_weights", {})
    last_gate = None
    for _ in range(args.epochs):
        h = enc(priv_hist)
        out = moe(h)
        last_gate = out["gate"].detach().cpu()
        loss = w.get("L_opt", 1.0) * F.mse_loss(out["z_teacher"], z_target)
        loss = loss + w.get("L_ctr", 0.2) * moe.contrastive_assignment_loss(out["similarity"], best_expert)
        loss = loss + w.get("L_ortho", 0.05) * moe.orthogonality_loss()
        loss = loss + w.get("L_bal", 0.05) * moe.load_balance_loss(out["gate"])
        opt.zero_grad()
        loss.backward()
        opt.step()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "best.pt"
    else:
        out = Path(args.output_ckpt or "teacher_moe.pt")
        out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "encoder": enc.state_dict(),
            "moe": moe.state_dict(),
            "prototypes": moe.prototypes.detach().cpu(),
            "gate": last_gate if last_gate is not None else torch.zeros(1, num_experts),
        },
        out,
    )
    print(f"[INFO] saved teacher checkpoint: {out}")


if __name__ == "__main__":
    main()
