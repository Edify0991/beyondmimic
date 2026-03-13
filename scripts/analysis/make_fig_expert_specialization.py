#!/usr/bin/env python3
from __future__ import annotations
import argparse, torch, matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Visualize expert usage and specialization.')
p.add_argument('--gate_pt',required=True); p.add_argument('--output_png',required=True); a=p.parse_args()
g=torch.load(a.gate_pt,map_location='cpu')['gate'].float(); plt.figure(); plt.bar(range(g.shape[1]), g.mean(0).numpy()); plt.xlabel('expert'); plt.ylabel('usage'); plt.tight_layout(); plt.savefig(a.output_png)
