#!/usr/bin/env python3
"""Frame-by-frame CSV segment selector for short-motion extraction.

This utility is designed for long motion CSVs that are difficult to auto-segment.
You can step through frames manually, mark start/end, and export a new short CSV
for subsequent `csv_to_npz.py` conversion and base-policy training.

Typical workflow:
1) python scripts/csv_frame_segmenter.py --input_csv long.csv --interactive
2) mark start/end in terminal (s/e), then write (w)
3) convert the exported short csv using scripts/csv_to_npz.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _clamp_frame(idx: int, total: int) -> int:
    return max(0, min(total - 1, idx))


def _print_frame_preview(data: np.ndarray, frame_idx: int) -> None:
    row = data[frame_idx]
    root_pos = row[:3]
    root_quat = row[3:7]
    print(
        f"[frame={frame_idx:06d}] root_pos={np.array2string(root_pos, precision=4)} "
        f"root_quat={np.array2string(root_quat, precision=4)}"
    )


def _save_segment(data: np.ndarray, start: int, end: int, out_csv: Path) -> None:
    if start > end:
        start, end = end, start
    seg = data[start : end + 1]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, seg, delimiter=",", fmt="%.10f")
    print(f"[INFO] saved segment: {out_csv} (frames {start}..{end}, count={len(seg)})")


def _interactive_loop(data: np.ndarray, out_csv: Path, init_start: int, init_end: int, step_size: int) -> None:
    total = data.shape[0]
    cur = init_start
    start = _clamp_frame(init_start, total)
    end = _clamp_frame(init_end, total)

    print("\n=== CSV Frame Segmenter (interactive) ===")
    print("Commands:")
    print("  n                 -> next frame")
    print("  p                 -> previous frame")
    print("  f <k>             -> move +k frames")
    print("  b <k>             -> move -k frames")
    print("  j <idx>           -> jump to frame idx")
    print("  s                 -> mark start = current")
    print("  e                 -> mark end = current")
    print("  i                 -> print info")
    print("  w                 -> write current [start,end] to output csv")
    print("  q                 -> quit")
    print("---------------------------------------")

    _print_frame_preview(data, cur)
    print(f"[range] start={start}, end={end}, total={total}")

    while True:
        raw = input("segmenter> ").strip()
        if not raw:
            raw = "n"
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "n":
            cur = _clamp_frame(cur + step_size, total)
            _print_frame_preview(data, cur)
        elif cmd == "p":
            cur = _clamp_frame(cur - step_size, total)
            _print_frame_preview(data, cur)
        elif cmd == "f" and len(parts) == 2 and parts[1].lstrip("-").isdigit():
            cur = _clamp_frame(cur + int(parts[1]), total)
            _print_frame_preview(data, cur)
        elif cmd == "b" and len(parts) == 2 and parts[1].lstrip("-").isdigit():
            cur = _clamp_frame(cur - int(parts[1]), total)
            _print_frame_preview(data, cur)
        elif cmd == "j" and len(parts) == 2 and parts[1].lstrip("-").isdigit():
            cur = _clamp_frame(int(parts[1]), total)
            _print_frame_preview(data, cur)
        elif cmd == "s":
            start = cur
            print(f"[INFO] start <- {start}")
        elif cmd == "e":
            end = cur
            print(f"[INFO] end <- {end}")
        elif cmd == "i":
            print(f"[info] current={cur}, start={start}, end={end}, segment_len={abs(end-start)+1}")
            _print_frame_preview(data, cur)
        elif cmd == "w":
            _save_segment(data, start, end, out_csv)
        elif cmd == "q":
            print("[INFO] quit without additional save.")
            break
        else:
            print("[WARN] unknown command. use n/p/f/b/j/s/e/i/w/q")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-by-frame CSV segment selector for short-motion extraction.")
    parser.add_argument("--input_csv", required=True, help="Input long motion csv.")
    parser.add_argument(
        "--output_csv",
        default="outputs/motions/selected_segment.csv",
        help="Output short segment csv path.",
    )
    parser.add_argument("--interactive", action="store_true", help="Enable interactive frame stepping and marking.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (0-based).")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (0-based, inclusive).")
    parser.add_argument("--step_size", type=int, default=1, help="Default step for n/p in interactive mode.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    data = np.loadtxt(input_csv, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("CSV format looks invalid: expected 2D array with at least 8 columns.")

    total = data.shape[0]
    start = _clamp_frame(args.start_frame, total)
    end = _clamp_frame(total - 1 if args.end_frame < 0 else args.end_frame, total)
    out_csv = Path(args.output_csv)

    print(f"[INFO] loaded {input_csv}, frames={total}, dims={data.shape[1]}")

    if args.interactive:
        _interactive_loop(data, out_csv, start, end, max(1, args.step_size))
    else:
        _save_segment(data, start, end, out_csv)


if __name__ == "__main__":
    main()
