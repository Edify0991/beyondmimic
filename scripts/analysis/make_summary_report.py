#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
p=argparse.ArgumentParser(description='Generate markdown summary report with figures and tables.')
p.add_argument('--metrics_json',required=True); p.add_argument('--fig_dir',required=True); p.add_argument('--output_md',required=True); a=p.parse_args()
with open(a.metrics_json,'r',encoding='utf-8') as f: m=json.load(f)
lines=['# Compliance Plugin Summary','', '## Metrics','```json', json.dumps(m,indent=2),'```','','## Figures']
for name in ['pareto_tradeoff','expert_specialization','gate_quality','payload_protection','window_counterfactual','student_vs_teacher']:
    lines.append(f'![{name}]({a.fig_dir}/{name}.png)')
with open(a.output_md,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
