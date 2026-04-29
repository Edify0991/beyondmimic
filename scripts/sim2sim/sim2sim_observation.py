#!/usr/bin/env python3
"""Observation history assembly helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np



def render_term_table(rows: list[tuple[str, int, int]]) -> str:
    if not rows:
        return ""
    name_w = max(len("term"), max(len(r[0]) for r in rows))
    dim_w = max(len("dim"), max(len(str(r[1])) for r in rows))
    hist_w = max(len("hist"), max(len(str(r[2])) for r in rows))
    lines = [f"{'term':<{name_w}}  {'dim':>{dim_w}}  {'hist':>{hist_w}}"]
    lines.append(f"{'-' * name_w}  {'-' * dim_w}  {'-' * hist_w}")
    for name, dim, hist in rows:
        lines.append(f"{name:<{name_w}}  {dim:>{dim_w}}  {hist:>{hist_w}}")
    return "\n".join(lines)


@dataclass
class ObservationAssembler:
    """Build flattened policy observation vectors with per-term history."""

    observation_names: list[str]
    observation_history_lengths: list[int]
    term_dims: dict[str, int]
    obs_dim: int
    histories: dict[str, deque[np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        self.histories = {
            name: deque(maxlen=max(1, hist))
            for name, hist in zip(self.observation_names, self.observation_history_lengths)
        }

    def clear(self) -> None:
        for history in self.histories.values():
            history.clear()

    def build(self, term_values: dict[str, np.ndarray]) -> np.ndarray:
        obs_terms: list[np.ndarray] = []
        for name, hist in zip(self.observation_names, self.observation_history_lengths):
            value = np.asarray(term_values[name], dtype=np.float32).reshape(-1)
            expected_dim = int(self.term_dims[name])
            if value.size != expected_dim:
                raise ValueError(
                    f"Term '{name}' dim mismatch: value dim={value.size}, expected={expected_dim}."
                )

            buffer = self.histories[name]
            if not buffer:
                for _ in range(max(1, hist)):
                    buffer.append(value.copy())
            else:
                buffer.append(value.copy())
                while len(buffer) < max(1, hist):
                    buffer.appendleft(buffer[0].copy())

            if hist > 1:
                obs_terms.append(np.concatenate(list(buffer), axis=0))
            else:
                obs_terms.append(buffer[-1])

        obs_vec = np.concatenate(obs_terms, axis=0).astype(np.float32)
        if obs_vec.size != self.obs_dim:
            raise ValueError(f"Built obs dim {obs_vec.size} does not match ONNX input dim {self.obs_dim}.")
        return obs_vec
