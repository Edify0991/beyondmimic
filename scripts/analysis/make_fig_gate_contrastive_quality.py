#!/usr/bin/env python3
from __future__ import annotations
import argparse, torch, matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Plot prototype cosine similarity / orthogonality.')
p.add_argument('--proto_pt',required=True); p.add_argument('--output_png',required=True); a=p.parse_args()
P=torch.nn.functional.normalize(torch.load(a.proto_pt,map_location='cpu')['prototypes'].float(),dim=-1); S=P@P.t(); plt.figure(); plt.imshow(S.numpy(),vmin=-1,vmax=1,cmap='coolwarm'); plt.colorbar(); plt.tight_layout(); plt.savefig(a.output_png)
