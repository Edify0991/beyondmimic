#!/usr/bin/env python3
"""Collect observable/privileged windows and save as HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect privileged windows from rollout logs.")
    parser.add_argument("--input_npz", required=True, help="Input rollout npz with arrays.")
    parser.add_argument("--output_h5", required=True)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    q = data["q"]
    with h5py.File(Path(args.output_h5), "w") as f:
        f.create_dataset("observable_window", data=q[:, -args.window :])
        for k in ["privileged", "action", "torque", "task_id", "motion_id", "event_tags"]:
            if k in data:
                f.create_dataset(k, data=data[k])


if __name__ == "__main__":
    main()
