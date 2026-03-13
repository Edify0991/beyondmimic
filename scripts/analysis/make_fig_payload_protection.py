#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Compare payload vibration/impulse metrics.')
p.add_argument('--baseline_npz',required=True); p.add_argument('--plugin_npz',required=True); p.add_argument('--output_png',required=True); a=p.parse_args()
b=np.load(a.baseline_npz); s=np.load(a.plugin_npz)
vals=[b['payload_vibration_cost'].mean(), s['payload_vibration_cost'].mean(), b['impact_cost'].mean(), s['impact_cost'].mean()]
plt.figure(); plt.bar(['base vib','plugin vib','base impact','plugin impact'], vals); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(a.output_png)
