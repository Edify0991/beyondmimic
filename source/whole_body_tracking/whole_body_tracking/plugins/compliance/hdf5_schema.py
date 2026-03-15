"""HDF5 schema helpers for compliance rollout logging."""

from __future__ import annotations

from typing import Any

import numpy as np


def _vlen_utf8_dtype():
    try:
        import h5py

        return h5py.string_dtype(encoding="utf-8")
    except Exception:
        return object


SCHEMA = {
    "/time/sim_step": {"shape": (0,), "maxshape": (None,), "dtype": np.int64},
    "/time/sim_time": {"shape": (0,), "maxshape": (None,), "dtype": np.float64},
    "/meta/env_id": {"shape": (0,), "maxshape": (None,), "dtype": np.int32},
    "/meta/episode_id": {"shape": (0,), "maxshape": (None,), "dtype": np.int64},
    "/meta/task_name": {"shape": (0,), "maxshape": (None,), "dtype": _vlen_utf8_dtype()},
    "/meta/motion_name": {"shape": (0,), "maxshape": (None,), "dtype": _vlen_utf8_dtype()},
    "/meta/seed": {"shape": (0,), "maxshape": (None,), "dtype": np.int64},
    "/meta/run_id": {"shape": (0,), "maxshape": (None,), "dtype": _vlen_utf8_dtype()},
    "/joint/q": {"shape": (0, 0), "maxshape": (None, None), "dtype": np.float32},
    "/joint/dq": {"shape": (0, 0), "maxshape": (None, None), "dtype": np.float32},
    "/joint/q_des": {"shape": (0, 0), "maxshape": (None, None), "dtype": np.float32},
    "/joint/action_pi": {"shape": (0, 0), "maxshape": (None, None), "dtype": np.float32},
    "/joint/tau_cmd": {"shape": (0, 0), "maxshape": (None, None), "dtype": np.float32},
    "/joint/effort_proxy": {"shape": (0, 1), "maxshape": (None, 1), "dtype": np.float32},
    "/root/pos": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/root/quat": {"shape": (0, 4), "maxshape": (None, 4), "dtype": np.float32},
    "/root/lin_vel": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/root/ang_vel": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/imu/torso": {"shape": (0, 6), "maxshape": (None, 6), "dtype": np.float32},
    "/contact/bool": {"shape": (0, 1), "maxshape": (None, 1), "dtype": np.float32},
    "/contact/force": {"shape": (0, 1), "maxshape": (None, 1), "dtype": np.float32},
    "/payload/pos": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/payload/vel": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/payload/acc": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/payload/rel_pos": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/payload/rel_vel": {"shape": (0, 3), "maxshape": (None, 3), "dtype": np.float32},
    "/event/aerial_hint": {"shape": (0, 1), "maxshape": (None, 1), "dtype": np.float32},
    "/event/touchdown_hint": {"shape": (0, 1), "maxshape": (None, 1), "dtype": np.float32},
}


def ensure_schema(h5f) -> None:
    for path, cfg in SCHEMA.items():
        if path in h5f:
            continue
        grp = "/".join(path.split("/")[:-1])
        if grp and grp not in h5f:
            h5f.require_group(grp)
        h5f.create_dataset(path, shape=cfg["shape"], maxshape=cfg["maxshape"], dtype=cfg["dtype"])


def append_rows(h5f, path: str, values: np.ndarray) -> None:
    ds = h5f[path]
    values = np.asarray(values)
    if ds.ndim == 1:
        values = values.reshape(-1)
    elif values.ndim == 1:
        values = values.reshape(-1, ds.shape[1])

    old_n = ds.shape[0]
    new_n = old_n + values.shape[0]
    ds.resize((new_n, *ds.shape[1:]))
    ds[old_n:new_n] = values


def optional_or_nan(value: Any, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    return arr
