#!/usr/bin/env python3
"""Example plugin that extends observation_registry terms."""

from __future__ import annotations

import numpy as np


def register_observation_terms(registry, context=None):
    # gravity_dir is often used by legged locomotion policies.
    registry.register(
        "gravity_dir",
        3,
        extractor=lambda s: np.asarray(s.get("gravity_dir", [0.0, 0.0, -1.0]), dtype=np.float32),
        description="Optional gravity direction in base frame.",
    )

    # token_state can be supplied by auxiliary encoders or external estimators.
    if context is not None and int(getattr(context, "token_dim", 0)) > 0:
        registry.register(
            "token_state",
            int(context.token_dim),
            extractor=lambda s, n=int(context.token_dim): np.asarray(
                s.get("token_state", np.zeros(n, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1),
            description="Optional latent token from auxiliary stage.",
        )
