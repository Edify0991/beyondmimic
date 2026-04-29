#!/usr/bin/env python3
"""YAML-driven observation pipeline selection for multi-policy sim2sim."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import runpy
from typing import Any


@dataclass(frozen=True)
class ObservationTermConfig:
    name: str
    history: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class ObservationPipeline:
    name: str
    terms: list[ObservationTermConfig]


@dataclass(frozen=True)
class ObservationSelection:
    pipeline_name: str
    observation_names: list[str]
    observation_history_lengths: list[int]
    source: str


def _load_dict_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML observation config files.") from exc
        data = yaml.safe_load(text)
    elif suffix == ".py":
        namespace = runpy.run_path(path)
        data = namespace.get("CONFIG", namespace.get("config"))
        if data is None:
            raise ValueError("Python observation config must define CONFIG (dict).")
    else:
        data = json.loads(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Observation config root must be dict/object: {path}")
    return data


def _to_term_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Observation term list must be a list.")
    terms: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            terms.append({"name": name, "enabled": True, "history": 1})
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if not name:
                raise ValueError(f"Observation term item missing name: {item}")
            terms.append(
                {
                    "name": name,
                    "enabled": bool(item.get("enabled", True)),
                    "history": int(item.get("history", item.get("history_length", 1))),
                }
            )
            continue
        raise ValueError(f"Unsupported observation item type: {type(item)}")
    return terms


def _normalize_external_observation_data(data: dict) -> dict:
    # Already in new style.
    if isinstance(data.get("observation"), dict):
        return dict(data["observation"])
    if "pipelines" in data and isinstance(data.get("pipelines"), dict):
        return dict(data)

    normalized: dict[str, Any] = {}
    pipelines: dict[str, Any] = {}

    # SONIC release style: top-level observations list.
    if isinstance(data.get("observations"), list):
        pipelines["policy_main"] = {"terms": _to_term_list(data.get("observations"))}
        normalized["pipeline"] = "policy_main"

    encoder_cfg = data.get("encoder")
    if isinstance(encoder_cfg, dict):
        if isinstance(encoder_cfg.get("encoder_observations"), list):
            pipelines["encoder"] = {"terms": _to_term_list(encoder_cfg.get("encoder_observations"))}

        encoder_modes = encoder_cfg.get("encoder_modes")
        if isinstance(encoder_modes, list):
            modes: list[dict[str, Any]] = []
            for mode in encoder_modes:
                if not isinstance(mode, dict):
                    continue
                mode_name = str(mode.get("name", "")).strip()
                modes.append(
                    {
                        "name": mode_name if mode_name else None,
                        "mode_id": int(mode.get("mode_id", -1)),
                        "pipeline": "encoder",
                        "required_observations": list(mode.get("required_observations", [])),
                    }
                )
            if modes:
                normalized["modes"] = modes

    if pipelines:
        normalized["pipelines"] = pipelines
    return normalized


def materialize_observation_block(cfg: dict) -> dict:
    """Load optional observation.config_file and merge into cfg["observation"].

    Priority:
    1) explicit fields already in cfg["observation"] (inline override)
    2) loaded external observation config file content
    """

    if not isinstance(cfg, dict):
        raise ValueError("Config must be dict.")
    observation = cfg.get("observation")
    if not isinstance(observation, dict):
        observation = {}
        cfg["observation"] = observation

    config_file = str(observation.get("config_file", "")).strip()
    if not config_file:
        return cfg
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"observation.config_file not found: {config_file}")

    loaded = _load_dict_file(config_file)
    normalized_loaded = _normalize_external_observation_data(loaded)
    if not isinstance(normalized_loaded, dict):
        raise ValueError("Normalized observation config must be dict.")

    merged: dict[str, Any] = {}
    merged.update(normalized_loaded)
    merged.update(observation)
    cfg["observation"] = merged
    return cfg


def _parse_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _parse_term_item(item: Any) -> ObservationTermConfig:
    if isinstance(item, str):
        name = item.strip()
        if not name:
            raise ValueError("Observation term string cannot be empty.")
        return ObservationTermConfig(name=name, history=1, enabled=True)
    if isinstance(item, dict):
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Observation term item missing name: {item}")
        history = int(item.get("history", item.get("history_length", 1)))
        if history <= 0:
            history = 1
        enabled = _parse_bool(item.get("enabled", True), default=True)
        return ObservationTermConfig(name=name, history=history, enabled=enabled)
    raise ValueError(f"Invalid observation term item type: {type(item)}")


def _normalize_pipeline_terms(raw: Any) -> list[ObservationTermConfig]:
    if raw is None:
        return []
    if isinstance(raw, dict) and "terms" in raw:
        raw = raw["terms"]
    if not isinstance(raw, list):
        raise ValueError("Observation pipeline terms must be a list.")

    parsed = [_parse_term_item(item) for item in raw]
    return [item for item in parsed if item.enabled]


def _get_mode_block(cfg: dict, mode_id: int | None, mode_name: str | None) -> dict | None:
    obs_cfg = cfg.get("observation")
    if not isinstance(obs_cfg, dict):
        return None
    modes = obs_cfg.get("modes")
    if not isinstance(modes, list):
        return None

    wanted_name = (mode_name or "").strip()
    for block in modes:
        if not isinstance(block, dict):
            continue
        if wanted_name and str(block.get("name", "")).strip() == wanted_name:
            return block
        if mode_id is not None and int(block.get("mode_id", -999999)) == int(mode_id):
            return block
    return None


def _resolve_pipeline_name(cfg: dict, default_pipeline: str | None, mode_block: dict | None) -> str | None:
    if isinstance(mode_block, dict):
        mode_pipeline = str(mode_block.get("pipeline", "")).strip()
        if mode_pipeline:
            return mode_pipeline

    obs_cfg = cfg.get("observation")
    if isinstance(obs_cfg, dict):
        from_cfg = str(obs_cfg.get("pipeline", "")).strip()
        if from_cfg:
            return from_cfg

    if default_pipeline and default_pipeline.strip():
        return default_pipeline.strip()
    return None


def _parse_pipeline(cfg: dict, pipeline_name: str) -> ObservationPipeline | None:
    obs_cfg = cfg.get("observation")
    if not isinstance(obs_cfg, dict):
        return None

    pipelines = obs_cfg.get("pipelines")
    if isinstance(pipelines, dict):
        block = pipelines.get(pipeline_name)
        if block is not None:
            return ObservationPipeline(name=pipeline_name, terms=_normalize_pipeline_terms(block))

    if pipeline_name == "default":
        raw_terms = obs_cfg.get("terms") or obs_cfg.get("observation_terms")
        if raw_terms is not None:
            return ObservationPipeline(name="default", terms=_normalize_pipeline_terms(raw_terms))

    return None


def _build_selection_from_pipeline(
    pipeline: ObservationPipeline,
    mode_block: dict | None,
) -> ObservationSelection:
    terms = list(pipeline.terms)

    if isinstance(mode_block, dict):
        required = mode_block.get("required_observations")
        if isinstance(required, list) and len(required) > 0:
            required_set = {str(item).strip() for item in required if str(item).strip()}
            terms = [term for term in terms if term.name in required_set]

    names = [term.name for term in terms]
    hists = [int(term.history) for term in terms]
    return ObservationSelection(
        pipeline_name=pipeline.name,
        observation_names=names,
        observation_history_lengths=hists,
        source="pipeline_config",
    )


def _build_selection_from_onnx(
    onnx_names: list[str],
    onnx_hists: list[int],
) -> ObservationSelection:
    names = list(onnx_names)
    if not names:
        raise ValueError("ONNX metadata has empty observation_names and no observation pipeline is configured.")
    if onnx_hists and len(onnx_hists) == len(names):
        hists = [max(1, int(v)) for v in onnx_hists]
    else:
        hists = [1] * len(names)
    return ObservationSelection(
        pipeline_name="onnx_metadata",
        observation_names=names,
        observation_history_lengths=hists,
        source="onnx_metadata",
    )


def resolve_observation_selection(
    cfg: dict,
    *,
    onnx_observation_names: list[str],
    onnx_observation_history_lengths: list[int],
    default_pipeline: str | None = None,
    mode_id: int | None = None,
    mode_name: str | None = None,
) -> ObservationSelection:
    """Resolve observation terms/history from YAML pipeline config with ONNX fallback."""

    mode_block = _get_mode_block(cfg, mode_id=mode_id, mode_name=mode_name)
    pipeline_name = _resolve_pipeline_name(cfg, default_pipeline=default_pipeline, mode_block=mode_block)

    pipeline = None
    if pipeline_name:
        pipeline = _parse_pipeline(cfg, pipeline_name)
        if pipeline is None:
            raise ValueError(f"Observation pipeline '{pipeline_name}' is requested but not found in config.")

    if pipeline is not None:
        selection = _build_selection_from_pipeline(pipeline, mode_block=mode_block)
        if not selection.observation_names:
            raise ValueError(
                f"Observation pipeline '{selection.pipeline_name}' resolved to empty terms "
                f"(mode filter may be too strict)."
            )
        return selection

    return _build_selection_from_onnx(
        onnx_names=onnx_observation_names,
        onnx_hists=onnx_observation_history_lengths,
    )
