#!/usr/bin/env python3
"""Train privileged contrastive MoE teacher against counterfactual targets."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from whole_body_tracking.plugins.compliance.compliance_moe import ComplianceMoE
from whole_body_tracking.plugins.compliance.privileged_encoder import PrivilegedHistoryEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train privileged MoE teacher.")
    parser.add_argument("--dataset_pt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data = torch.load(args.dataset_pt, map_location="cpu")
    priv_hist = data["priv_hist"].float()
    z_target = data["z_target"].float()
    best_expert = data["best_expert"].long()

    enc = PrivilegedHistoryEncoder(priv_hist.shape[-1])
    moe = ComplianceMoE(192, num_experts=data.get("num_experts", 4))
    opt = torch.optim.Adam(list(enc.parameters()) + list(moe.parameters()), lr=3e-4)
    for _ in range(args.epochs):
        h = enc(priv_hist)
        out = moe(h)
        loss = F.mse_loss(out["z_teacher"], z_target)
        loss = loss + 0.2 * moe.contrastive_assignment_loss(out["similarity"], best_expert)
        loss = loss + 0.05 * moe.orthogonality_loss() + 0.05 * moe.load_balance_loss(out["gate"])
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"encoder": enc.state_dict(), "moe": moe.state_dict()}, args.output_ckpt)


if __name__ == "__main__":
    main()
