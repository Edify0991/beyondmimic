#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description='Plot Pareto tradeoff between tracking and impact/vibration/oscillation.')
p.add_argument('--metrics_json',required=True)
p.add_argument('--output_png',required=True)
a=p.parse_args()
with open(a.metrics_json,'r',encoding='utf-8') as f: m=json.load(f)
plt.figure();
for name,v in m.items(): plt.scatter(v.get('tracking_rmse',0), v.get('impact_cost',0), label=name)
plt.xlabel('tracking preservation (RMSE)'); plt.ylabel('impact cost'); plt.legend(); plt.tight_layout(); plt.savefig(a.output_png)
