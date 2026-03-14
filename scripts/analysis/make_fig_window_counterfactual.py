#!/usr/bin/env python3
from __future__ import annotations
import argparse, torch, matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Plot short event window with base action and z* trajectories.')
p.add_argument('--window_pt',required=True); p.add_argument('--output_png',required=True); a=p.parse_args()
d=torch.load(a.window_pt,map_location='cpu'); plt.figure(); plt.plot(d['base_action'][0].numpy(), label='base'); plt.plot(d['z_star'][0,:,0].numpy(), label='z*_omega'); plt.legend(); plt.tight_layout(); plt.savefig(a.output_png)
