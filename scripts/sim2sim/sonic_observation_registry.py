#!/usr/bin/env python3
"""Observation registry for SONIC-style, multi-policy sim2sim pipelines."""

from __future__ import annotations

import inspect
import os
import runpy
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


ExtractorFn = Callable[[Dict[str, np.ndarray]], np.ndarray]
DimensionFn = Callable[[], int]


@dataclass(frozen=True)
class ObservationShapeContext:
    """Static dimension context used to build observation terms."""

    action_dim: int
    command_joint_dim: int
    policy_joint_dim: int
    body_count: int
    token_dim: int = 0


@dataclass(frozen=True)
class ObservationTermSpec:
    """Registry entry for one observation term."""

    name: str
    dimension_fn: DimensionFn
    extractor_fn: ExtractorFn
    description: str = ""

    def dimension(self) -> int:
        dim = int(self.dimension_fn())
        if dim <= 0:
            raise ValueError(f"Observation term '{self.name}' has invalid dimension {dim}.")
        return dim

    def extract(self, source: dict[str, np.ndarray]) -> np.ndarray:
        value = np.asarray(self.extractor_fn(source), dtype=np.float32).reshape(-1)
        expected = self.dimension()
        if value.size != expected:
            raise ValueError(
                f"Observation term '{self.name}' dim mismatch: got {value.size}, expected {expected}."
            )
        return value


class ObservationRegistry:
    """Name -> term-spec registry with dimensional validation."""

    def __init__(self, context: ObservationShapeContext):
        self.context = context
        self._specs: dict[str, ObservationTermSpec] = {}

    def register(
        self,
        name: str,
        dimension: int | DimensionFn,
        extractor: ExtractorFn | None = None,
        description: str = "",
    ) -> None:
        term_name = str(name).strip()
        if not term_name:
            raise ValueError("Observation term name cannot be empty.")

        if extractor is None:
            extractor = lambda source, key=term_name: source[key]

        if callable(dimension):
            dim_fn = dimension
        else:
            dim_value = int(dimension)
            dim_fn = lambda value=dim_value: value

        self._specs[term_name] = ObservationTermSpec(
            name=term_name,
            dimension_fn=dim_fn,
            extractor_fn=extractor,
            description=description,
        )

    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> ObservationTermSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs.keys()))
            raise KeyError(f"Unknown observation term '{name}'. Known terms: {known}") from exc

    def build_term_dimensions(self, observation_names: list[str]) -> dict[str, int]:
        return {name: self.get(name).dimension() for name in observation_names}

    def extract_terms(self, observation_names: list[str], source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in observation_names:
            out[name] = self.get(name).extract(source)
        return out

    def list_terms(self) -> list[str]:
        return sorted(self._specs.keys())


def build_default_observation_registry(context: ObservationShapeContext) -> ObservationRegistry:
    """Build built-in terms covering BeyondMimic + SONIC naming conventions."""

    registry = ObservationRegistry(context)
    cdim = int(context.command_joint_dim)
    pdim = int(context.policy_joint_dim)
    adim = int(context.action_dim)
    bdim = int(context.body_count)

    # BeyondMimic canonical names.
    registry.register("command", 2 * cdim)
    registry.register("motion_anchor_pos_b", 3)
    registry.register("motion_anchor_ori_b", 6)
    registry.register("base_lin_vel", 3)
    registry.register("base_ang_vel", 3)
    registry.register("joint_pos", pdim)
    registry.register("joint_vel", pdim)
    registry.register("actions", adim)
    registry.register("robot_anchor_ori_w", 6)
    registry.register("robot_anchor_lin_vel_w", 3)
    registry.register("robot_anchor_ang_vel_w", 3)
    registry.register("robot_body_pos_b", 3 * bdim)
    registry.register("robot_body_ori_b", 6 * bdim)

    # SONIC-ish names / aliases.
    registry.register(
        "motion_joint_positions",
        cdim,
        extractor=lambda s, n=cdim: np.asarray(s["command"], dtype=np.float32)[:n],
        description="Reference motion joint positions.",
    )
    registry.register(
        "motion_joint_velocities",
        cdim,
        extractor=lambda s, n=cdim: np.asarray(s["command"], dtype=np.float32)[n : 2 * n],
        description="Reference motion joint velocities.",
    )
    registry.register("motion_anchor_orientation", 6, extractor=lambda s: s["motion_anchor_ori_b"])
    registry.register("base_linear_velocity", 3, extractor=lambda s: s["base_lin_vel"])
    registry.register("base_angular_velocity", 3, extractor=lambda s: s["base_ang_vel"])
    registry.register("body_joint_positions", pdim, extractor=lambda s: s["joint_pos"])
    registry.register("body_joint_velocities", pdim, extractor=lambda s: s["joint_vel"])
    registry.register("last_actions", adim, extractor=lambda s: s["actions"])

    if int(context.token_dim) > 0:
        registry.register("token_state", int(context.token_dim))

    return registry


def load_observation_registry_plugin(registry: ObservationRegistry, plugin_file: str) -> None:
    """Load a python plugin and extend/override observation terms."""

    plugin_path = os.path.abspath(os.path.expanduser(plugin_file))
    if not os.path.isfile(plugin_path):
        raise FileNotFoundError(f"Observation registry plugin not found: {plugin_path}")

    namespace = runpy.run_path(plugin_path)
    register_fn = namespace.get("register_observation_terms")
    if not callable(register_fn):
        raise ValueError(
            "Observation plugin must define callable 'register_observation_terms(registry, context=None)'."
        )

    sig = inspect.signature(register_fn)
    if len(sig.parameters) <= 1:
        register_fn(registry)
    else:
        register_fn(registry, registry.context)
