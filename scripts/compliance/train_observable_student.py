#!/usr/bin/env python3
"""Train observable student head and export TorchScript."""

from __future__ import annotations

import argparse

import torch

from whole_body_tracking.plugins.compliance.observable_encoder import ObservableHistoryEncoder
from whole_body_tracking.plugins.compliance.teacher_student_distill import ObservableStudentHead, distillation_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill observable student from teacher labels.")
    parser.add_argument("--dataset_pt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--output_ts", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    data = torch.load(args.dataset_pt, map_location="cpu")
    obs_hist = data["obs_hist"].float()
    teacher_z = data["teacher_z"].float()
    teacher_gate = data["teacher_gate"].float()

    enc = ObservableHistoryEncoder(obs_hist.shape[-1])
    stu = ObservableStudentHead(128, teacher_gate.shape[-1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(stu.parameters()), lr=3e-4)
    for _ in range(args.epochs):
        h = enc(obs_hist)
        out = stu(h)
        losses = distillation_losses(out, teacher_z, teacher_gate)
        loss = sum(losses.values())
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"encoder": enc.state_dict(), "student": stu.state_dict()}, args.output_ckpt)
    scripted = torch.jit.script(stu)
    scripted.save(args.output_ts)


if __name__ == "__main__":
    main()
