#!/usr/bin/env python3
"""SONIC-style runtime state machine for sim2sim deployment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ProgramState(Enum):
    """Lifecycle states aligned with SONIC control loop behavior."""

    INIT = auto()
    WAIT_FOR_CONTROL = auto()
    CONTROL = auto()
    STOPPED = auto()


@dataclass(frozen=True)
class OperatorCommand:
    """External control command packet consumed by ProgramFSM."""

    start: bool = False
    stop: bool = False
    play: bool = True


@dataclass(frozen=True)
class ProgramFsmConfig:
    """Configurable FSM behavior."""

    init_steps: int = 1
    auto_start: bool = True
    wait_timeout_s: float = 0.0
    start_in_control: bool = False


class ProgramFSM:
    """Deterministic state machine for rollout lifecycle control."""

    def __init__(self, config: ProgramFsmConfig):
        self.config = config
        self.state = ProgramState.CONTROL if config.start_in_control else ProgramState.INIT
        self._init_counter = 0
        self._wait_elapsed_s = 0.0

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "ProgramFSM":
        cfg = cfg or {}
        return cls(
            ProgramFsmConfig(
                init_steps=max(0, int(cfg.get("init_steps", 1))),
                auto_start=bool(cfg.get("auto_start", True)),
                wait_timeout_s=max(0.0, float(cfg.get("wait_timeout_s", 0.0))),
                start_in_control=bool(cfg.get("start_in_control", False)),
            )
        )

    def update(self, cmd: OperatorCommand, dt: float) -> ProgramState:
        if self.state == ProgramState.STOPPED:
            return self.state

        if cmd.stop:
            self.state = ProgramState.STOPPED
            return self.state

        if self.state == ProgramState.INIT:
            self._init_counter += 1
            if self._init_counter >= self.config.init_steps:
                self.state = ProgramState.WAIT_FOR_CONTROL
                if self.config.auto_start:
                    self.state = ProgramState.CONTROL
            return self.state

        if self.state == ProgramState.WAIT_FOR_CONTROL:
            self._wait_elapsed_s += max(0.0, float(dt))
            timeout = self.config.wait_timeout_s > 0.0 and self._wait_elapsed_s >= self.config.wait_timeout_s
            if cmd.start or self.config.auto_start or timeout:
                self.state = ProgramState.CONTROL
            return self.state

        if self.state == ProgramState.CONTROL:
            if not cmd.play:
                self.state = ProgramState.WAIT_FOR_CONTROL
            return self.state

        return self.state
