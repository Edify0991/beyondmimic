#!/usr/bin/env python3
"""SONIC-style input/output channel abstractions for sim2sim."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from sonic_fsm import OperatorCommand


@dataclass(frozen=True)
class TelemetryPacket:
    """Structured runtime telemetry published by output channels."""

    step: int
    time_s: float
    joint_rmse: float
    anchor_pos_err: float
    anchor_ori_err: float
    action_l2: float
    extras: dict[str, Any]


_PACKED_HEADER_SIZE_DEFAULT = 1280
_PACKED_DTYPE_TO_NP: dict[str, np.dtype] = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i8": np.dtype("i1"),
    "u8": np.dtype("u1"),
    "i16": np.dtype("<i2"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bool": np.dtype("u1"),
}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _shape_product(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _normalize_packed_dtype(dtype: str) -> str:
    token = str(dtype).strip().lower()
    if token not in _PACKED_DTYPE_TO_NP:
        raise ValueError(f"Unsupported packed dtype: {dtype}")
    return token


def _coerce_field_shape(shape_raw: Any) -> tuple[int, ...]:
    if shape_raw is None:
        return (1,)
    if isinstance(shape_raw, int):
        dim = int(shape_raw)
        if dim <= 0:
            raise ValueError(f"Invalid packed field shape dim: {dim}")
        return (dim,)
    if isinstance(shape_raw, list):
        if len(shape_raw) == 0:
            return (1,)
        shape: list[int] = []
        for item in shape_raw:
            dim = int(item)
            if dim <= 0:
                raise ValueError(f"Invalid packed field shape dim: {dim}")
            shape.append(dim)
        return tuple(shape)
    raise ValueError(f"Packed field shape must be int/list, got: {type(shape_raw)}")


def _strip_topic_prefix(raw: bytes, topic: str) -> bytes | None:
    if not topic:
        return raw
    prefix = topic.encode("utf-8")
    if raw.startswith(prefix):
        return raw[len(prefix) :]
    return None


def _parse_packed_message(raw_payload: bytes, header_size: int) -> dict[str, np.ndarray]:
    if len(raw_payload) < header_size:
        raise ValueError(f"Packed payload too small: {len(raw_payload)} < header_size={header_size}")

    header_buf = raw_payload[:header_size]
    header_end = header_buf.find(b"\x00")
    if header_end < 0:
        header_end = header_size
    header_text = header_buf[:header_end].decode("utf-8").strip()
    if not header_text:
        raise ValueError("Packed payload contains empty JSON header.")

    header = json.loads(header_text)
    fields = header.get("fields")
    if not isinstance(fields, list):
        raise ValueError("Packed header must contain a list field 'fields'.")

    data = memoryview(raw_payload)[header_size:]
    offset = 0
    out: dict[str, np.ndarray] = {}
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("Each packed header field spec must be a dict.")
        name = str(field.get("name", "")).strip()
        if not name:
            raise ValueError("Packed header field has empty 'name'.")
        dtype_token = _normalize_packed_dtype(str(field.get("dtype", "f32")))
        shape = _coerce_field_shape(field.get("shape", [1]))
        elem_count = _shape_product(shape)
        np_dtype = _PACKED_DTYPE_TO_NP[dtype_token]
        nbytes = elem_count * np_dtype.itemsize
        if offset + nbytes > len(data):
            raise ValueError(
                f"Packed field '{name}' overflows payload: "
                f"offset={offset}, nbytes={nbytes}, payload={len(data)}"
            )
        buf = data[offset : offset + nbytes]
        arr = np.frombuffer(buf, dtype=np_dtype, count=elem_count)
        arr = arr.reshape(shape)
        out[name] = arr
        offset += nbytes
    return out


def _first_scalar_bool(fields: dict[str, np.ndarray], field_name: str, default: bool) -> bool:
    arr = fields.get(field_name)
    if arr is None or arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return _parse_bool(value, default=default)


def _as_le_array(value: Any, dtype_token: str) -> np.ndarray:
    dtype_name = _normalize_packed_dtype(dtype_token)
    np_dtype = _PACKED_DTYPE_TO_NP[dtype_name]
    arr = np.asarray(value, dtype=np_dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


@dataclass(frozen=True)
class _PackedField:
    name: str
    dtype: str
    value: np.ndarray


def _build_packed_message(topic: str, header_size: int, fields: list[_PackedField]) -> bytes:
    header_fields: list[dict[str, Any]] = []
    binary_parts: list[bytes] = []
    for field in fields:
        dtype_name = _normalize_packed_dtype(field.dtype)
        arr = _as_le_array(field.value, dtype_name)
        header_fields.append(
            {
                "name": field.name,
                "dtype": dtype_name,
                "shape": list(arr.shape),
            }
        )
        binary_parts.append(arr.tobytes(order="C"))

    header = {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": header_fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) >= header_size:
        raise ValueError(
            f"Packed header exceeds header_size ({len(header_json)} >= {header_size}). "
            "Reduce fields/extras or increase sonic.*.header_size."
        )
    header_bytes = header_json.ljust(header_size, b"\x00")
    topic_bytes = topic.encode("utf-8") if topic else b""
    return topic_bytes + header_bytes + b"".join(binary_parts)


class BaseInputChannel:
    """Abstract operator command source."""

    def poll(self, *, step: int, time_s: float) -> OperatorCommand:
        raise NotImplementedError

    def close(self) -> None:
        return


class StaticInputChannel(BaseInputChannel):
    """Simple always-on command source (auto start, optional auto-stop)."""

    def __init__(self, *, auto_start: bool = True, stop_after_steps: int = 0):
        self.auto_start = bool(auto_start)
        self.stop_after_steps = int(stop_after_steps)
        self._start_sent = False

    def poll(self, *, step: int, time_s: float) -> OperatorCommand:
        if self.stop_after_steps > 0 and step >= self.stop_after_steps:
            return OperatorCommand(stop=True)
        if self.auto_start and not self._start_sent:
            self._start_sent = True
            return OperatorCommand(start=True)
        return OperatorCommand()


class ZmqJsonInputChannel(BaseInputChannel):
    """Optional ZMQ JSON command source for start/stop/play control."""

    def __init__(self, *, host: str, port: int, topic: str = "command"):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is required for sonic.input.type='zmq_json'. Install with: pip install pyzmq"
            ) from exc

        self._zmq = zmq
        self._topic = topic or ""
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, 1)
        self._sock.connect(f"tcp://{host}:{int(port)}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, self._topic)

    def poll(self, *, step: int, time_s: float) -> OperatorCommand:
        try:
            raw = self._sock.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return OperatorCommand()
        except Exception:
            return OperatorCommand()

        payload = raw
        if self._topic and raw.startswith(self._topic.encode("utf-8")):
            payload = raw[len(self._topic) :]
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            return OperatorCommand()

        return OperatorCommand(
            start=bool(data.get("start", False)),
            stop=bool(data.get("stop", False)),
            play=bool(data.get("play", True)),
        )

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class ZmqPackedInputChannel(BaseInputChannel):
    """SONIC packed-message command source: [topic][fixed JSON header][binary payload]."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        topic: str = "command",
        header_size: int = _PACKED_HEADER_SIZE_DEFAULT,
        start_field: str = "start",
        stop_field: str = "stop",
        play_field: str = "play",
        planner_field: str = "planner",
        timeout_ms: int = 1,
        conflate: bool = True,
    ):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is required for sonic.input.type='zmq_packed'. Install with: pip install pyzmq"
            ) from exc

        self._zmq = zmq
        self._topic = topic or ""
        self._header_size = max(64, int(header_size))
        self._start_field = str(start_field).strip() or "start"
        self._stop_field = str(stop_field).strip() or "stop"
        self._play_field = str(play_field).strip() or "play"
        self._planner_field = str(planner_field).strip() or "planner"

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_ms)))
        if bool(conflate):
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(f"tcp://{host}:{int(port)}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, self._topic)

    def poll(self, *, step: int, time_s: float) -> OperatorCommand:
        try:
            raw = self._sock.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return OperatorCommand()
        except Exception:
            return OperatorCommand()

        payload = _strip_topic_prefix(raw, self._topic)
        if payload is None:
            return OperatorCommand()
        try:
            fields = _parse_packed_message(payload, self._header_size)
        except Exception:
            return OperatorCommand()

        start = _first_scalar_bool(fields, self._start_field, default=False)
        stop = _first_scalar_bool(fields, self._stop_field, default=False)

        if self._play_field in fields:
            play = _first_scalar_bool(fields, self._play_field, default=True)
        elif self._planner_field in fields:
            # SONIC command topic often uses "planner" instead of "play".
            play = _first_scalar_bool(fields, self._planner_field, default=True)
        else:
            play = True

        return OperatorCommand(start=start, stop=stop, play=play)

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class BaseOutputChannel:
    """Abstract runtime telemetry sink."""

    def publish(self, packet: TelemetryPacket) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class ConsoleOutputChannel(BaseOutputChannel):
    """Human-readable periodic telemetry logging."""

    def __init__(self, *, print_every: int = 50):
        self.print_every = max(1, int(print_every))

    def publish(self, packet: TelemetryPacket) -> None:
        if packet.step == 0 or (packet.step + 1) % self.print_every == 0:
            print(
                f"[sonic-io step {packet.step + 1}] "
                f"rmse={packet.joint_rmse:.5f} "
                f"pos={packet.anchor_pos_err:.5f}m "
                f"ori={packet.anchor_ori_err:.5f}rad "
                f"action={packet.action_l2:.5f}"
            )


class ZmqJsonOutputChannel(BaseOutputChannel):
    """Optional ZMQ JSON telemetry publisher."""

    def __init__(self, *, host: str, port: int, topic: str = "sim2sim_debug"):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is required for sonic.output channel type='zmq_json'. Install with: pip install pyzmq"
            ) from exc

        self._zmq = zmq
        self._topic = topic or ""
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{int(port)}")
        self._boot_time = time.time()

    def publish(self, packet: TelemetryPacket) -> None:
        message = {
            "step": packet.step,
            "time_s": packet.time_s,
            "joint_rmse": packet.joint_rmse,
            "anchor_pos_err": packet.anchor_pos_err,
            "anchor_ori_err": packet.anchor_ori_err,
            "action_l2": packet.action_l2,
            "wall_time_s": time.time() - self._boot_time,
            "extras": packet.extras,
        }
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        data = (self._topic.encode("utf-8") + payload) if self._topic else payload
        try:
            self._sock.send(data, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class ZmqPackedOutputChannel(BaseOutputChannel):
    """SONIC packed-message telemetry publisher."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        topic: str = "sim2sim_debug",
        header_size: int = _PACKED_HEADER_SIZE_DEFAULT,
        float_dtype: str = "f32",
        include_extras: bool = False,
    ):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError(
                "pyzmq is required for sonic.output channel type='zmq_packed'. Install with: pip install pyzmq"
            ) from exc

        self._zmq = zmq
        self._topic = topic or ""
        self._header_size = max(64, int(header_size))
        float_dtype = str(float_dtype).strip().lower() or "f32"
        if float_dtype not in {"f32", "f64"}:
            raise ValueError("sonic.output.zmq_packed.float_dtype must be 'f32' or 'f64'.")
        self._float_dtype = float_dtype
        self._include_extras = bool(include_extras)

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{int(port)}")
        self._boot_time = time.time()

    def _extras_to_fields(self, extras: dict[str, Any]) -> list[_PackedField]:
        if not self._include_extras or not isinstance(extras, dict):
            return []
        packed: list[_PackedField] = []
        for key, value in extras.items():
            name = str(key).strip()
            if not name:
                continue
            if isinstance(value, bool):
                packed.append(_PackedField(name=name, dtype="u8", value=np.asarray([1 if value else 0], dtype=np.uint8)))
                continue
            if isinstance(value, (int, np.integer)):
                packed.append(_PackedField(name=name, dtype="i64", value=np.asarray([int(value)], dtype=np.int64)))
                continue
            if isinstance(value, (float, np.floating)):
                packed.append(
                    _PackedField(
                        name=name,
                        dtype=self._float_dtype,
                        value=np.asarray([float(value)], dtype=_PACKED_DTYPE_TO_NP[self._float_dtype]),
                    )
                )
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.size == 0:
                    continue
                if arr.dtype.kind in {"b"}:
                    packed.append(_PackedField(name=name, dtype="u8", value=arr.astype(np.uint8)))
                elif arr.dtype.kind in {"i", "u"}:
                    packed.append(_PackedField(name=name, dtype="i64", value=arr.astype(np.int64)))
                elif arr.dtype.kind in {"f"}:
                    packed.append(
                        _PackedField(
                            name=name,
                            dtype=self._float_dtype,
                            value=arr.astype(_PACKED_DTYPE_TO_NP[self._float_dtype]),
                        )
                    )
                # skip unsupported extra types (strings/objects)
                continue
        return packed

    def publish(self, packet: TelemetryPacket) -> None:
        base_fields: list[_PackedField] = [
            _PackedField("step", "i64", np.asarray([packet.step], dtype=np.int64)),
            _PackedField("time_s", "f64", np.asarray([packet.time_s], dtype=np.float64)),
            _PackedField("joint_rmse", self._float_dtype, np.asarray([packet.joint_rmse])),
            _PackedField("anchor_pos_err", self._float_dtype, np.asarray([packet.anchor_pos_err])),
            _PackedField("anchor_ori_err", self._float_dtype, np.asarray([packet.anchor_ori_err])),
            _PackedField("action_l2", self._float_dtype, np.asarray([packet.action_l2])),
            _PackedField("wall_time_s", "f64", np.asarray([time.time() - self._boot_time], dtype=np.float64)),
        ]
        all_fields = base_fields + self._extras_to_fields(packet.extras)
        try:
            payload = _build_packed_message(self._topic, self._header_size, all_fields)
            self._sock.send(payload, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass
        except Exception:
            # Keep runtime robust even if telemetry transport fails.
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def build_input_channel(cfg: dict | None) -> BaseInputChannel:
    cfg = cfg or {}
    channel_type = str(cfg.get("type", "static")).strip().lower()
    if channel_type == "static":
        return StaticInputChannel(
            auto_start=bool(cfg.get("auto_start", True)),
            stop_after_steps=int(cfg.get("stop_after_steps", 0)),
        )
    if channel_type == "zmq_json":
        return ZmqJsonInputChannel(
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 5558)),
            topic=str(cfg.get("topic", "command")),
        )
    if channel_type == "zmq_packed":
        return ZmqPackedInputChannel(
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 5558)),
            topic=str(cfg.get("topic", "command")),
            header_size=int(cfg.get("header_size", _PACKED_HEADER_SIZE_DEFAULT)),
            start_field=str(cfg.get("start_field", "start")),
            stop_field=str(cfg.get("stop_field", "stop")),
            play_field=str(cfg.get("play_field", "play")),
            planner_field=str(cfg.get("planner_field", "planner")),
            timeout_ms=int(cfg.get("timeout_ms", 1)),
            conflate=bool(cfg.get("conflate", True)),
        )
    raise ValueError(f"Unsupported sonic input channel type: {channel_type}")


def build_output_channels(cfg: dict | None) -> list[BaseOutputChannel]:
    cfg = cfg or {}
    channels_cfg = cfg.get("channels")

    if channels_cfg is None:
        default_type = str(cfg.get("type", "")).strip().lower()
        if not default_type:
            return []
        channels_cfg = [dict(cfg)]

    if not isinstance(channels_cfg, list):
        raise ValueError("sonic.output.channels must be a list of channel configs.")

    channels: list[BaseOutputChannel] = []
    for item in channels_cfg:
        if not isinstance(item, dict):
            raise ValueError("Each sonic.output.channels item must be a dict/object.")
        channel_type = str(item.get("type", "console")).strip().lower()
        if channel_type == "console":
            channels.append(ConsoleOutputChannel(print_every=int(item.get("print_every", 50))))
        elif channel_type == "zmq_json":
            channels.append(
                ZmqJsonOutputChannel(
                    host=str(item.get("host", "*")),
                    port=int(item.get("port", 5560)),
                    topic=str(item.get("topic", "sim2sim_debug")),
                )
            )
        elif channel_type == "zmq_packed":
            channels.append(
                ZmqPackedOutputChannel(
                    host=str(item.get("host", "*")),
                    port=int(item.get("port", 5560)),
                    topic=str(item.get("topic", "sim2sim_debug")),
                    header_size=int(item.get("header_size", _PACKED_HEADER_SIZE_DEFAULT)),
                    float_dtype=str(item.get("float_dtype", "f32")),
                    include_extras=bool(item.get("include_extras", False)),
                )
            )
        elif channel_type in ["none", "null", "disabled"]:
            continue
        else:
            raise ValueError(f"Unsupported sonic output channel type: {channel_type}")
    return channels
