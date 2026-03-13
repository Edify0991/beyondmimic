#!/usr/bin/env python3
from __future__ import annotations
import argparse, torch, matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Compare student vs teacher z and gate KL.')
p.add_argument('--compare_pt',required=True); p.add_argument('--output_png',required=True); a=p.parse_args()
d=torch.load(a.compare_pt,map_location='cpu')
err=(d['student_z']-d['teacher_z']).pow(2).mean(-1).sqrt(); plt.figure(); plt.plot(err.numpy()); plt.ylabel('z error'); plt.tight_layout(); plt.savefig(a.output_png)
