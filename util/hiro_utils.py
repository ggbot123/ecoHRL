import json
import os
from copy import deepcopy
from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Tuple

from rl.algos.sac.sac import SAC
from rl.algos.HRL.goal_samplers import GoalSamplerConfig
from rl.algos.HRL.hiro import HIROConfig, LowSafetyFilterConfig
from util.config_utils import deep_update


def unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    idx = 1
    while True:
        cand = f"{base}_{idx}{ext}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def find_hiro_run_config(model_dir: str) -> Optional[str]:
    """Find run_config.json beside the model or in a same-name log directory."""
    model_dir_abs = os.path.abspath(model_dir)
    run_name = os.path.basename(os.path.normpath(model_dir_abs))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates = [
        os.path.join(model_dir_abs, "run_config.json"),
        os.path.join(repo_root, "logs", "current", run_name, "run_config.json"),
        os.path.join(repo_root, "logs", run_name, "run_config.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_hiro_run_config(model_dir: str) -> Tuple[Dict[str, Any], str]:
    path = find_hiro_run_config(model_dir)
    if path is None:
        raise FileNotFoundError(
            "Missing run_config.json beside the model and in same-name log "
            f"directories for: {os.path.abspath(model_dir)}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"HIRO run config must be a JSON object: {path}")
    return payload, path


def env_config_from_run_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("run_config.json is missing the 'environment' object")
    env_cfg = environment.get("env0_config")
    if not isinstance(env_cfg, Mapping):
        raise ValueError(
            "run_config.json is missing 'environment.env0_config'"
        )
    return deepcopy(dict(env_cfg))


def _hiro_config_from_mapping(saved: Mapping[str, Any]) -> HIROConfig:
    allowed = {field.name for field in fields(HIROConfig)}
    missing = allowed - set(saved)
    unknown = set(saved) - allowed
    if missing or unknown:
        raise ValueError(
            "Invalid HIRO config fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    kwargs = deepcopy(dict(saved))
    goal_sampler = kwargs.get("goal_sampler")
    low_safety_filter = kwargs.get("low_safety_filter")
    goal_sampler_fields = {field.name for field in fields(GoalSamplerConfig)}
    low_safety_filter_fields = {field.name for field in fields(LowSafetyFilterConfig)}
    if not isinstance(goal_sampler, Mapping):
        raise ValueError("HIRO config 'goal_sampler' must be an object")
    goal_missing = goal_sampler_fields - set(goal_sampler)
    goal_unknown = set(goal_sampler) - goal_sampler_fields
    if goal_missing or goal_unknown:
        raise ValueError(
            "Invalid goal_sampler fields: "
            f"missing={sorted(goal_missing)}, unknown={sorted(goal_unknown)}"
        )
    kwargs["goal_sampler"] = GoalSamplerConfig(**dict(goal_sampler))

    if low_safety_filter is None:
        kwargs["low_safety_filter"] = None
    elif isinstance(low_safety_filter, Mapping):
        safety_missing = low_safety_filter_fields - set(low_safety_filter)
        safety_unknown = set(low_safety_filter) - low_safety_filter_fields
        if safety_missing or safety_unknown:
            raise ValueError(
                "Invalid low_safety_filter fields: "
                f"missing={sorted(safety_missing)}, unknown={sorted(safety_unknown)}"
            )
        kwargs["low_safety_filter"] = LowSafetyFilterConfig(**dict(low_safety_filter))
    else:
        raise ValueError("HIRO config 'low_safety_filter' must be an object or null")
    return HIROConfig(**kwargs)


def hiro_config_from_run_config(payload: Mapping[str, Any]) -> HIROConfig:
    hiro_section = payload.get("hiro")
    if not isinstance(hiro_section, Mapping):
        raise ValueError("run_config.json is missing the 'hiro' object")
    saved = hiro_section.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("run_config.json is missing 'hiro.config'")
    return _hiro_config_from_mapping(saved)


def apply_hiro_config_overrides(
    config: HIROConfig,
    overrides: Mapping[str, Any],
) -> HIROConfig:
    merged = asdict(config)
    deep_update(merged, overrides)
    return _hiro_config_from_mapping(merged)


def load_hiro_models(
    model_dir: str,
    *,
    high_model_dir: Optional[str] = None,
    low_model_dir: Optional[str] = None,
    model_suffix: Optional[str] = None,
) -> Tuple[SAC, SAC]:
    """Load HIRO high/low models.

    Fixed filenames:
    - hiro_high_final.zip
    - hiro_low_final.zip

    Defaults to loading both models from `model_dir`.
    You can override high/low to come from different directories.
    """
    high_dir = high_model_dir or model_dir
    low_dir = low_model_dir or model_dir
    suffix = model_suffix or "final"
    high_name = f"hiro_high_{suffix}"
    low_name = f"hiro_low_{suffix}"
    high_path = os.path.join(high_dir, f"{high_name}.zip")
    low_path = os.path.join(low_dir, f"{low_name}.zip")
    return SAC.load(high_path), SAC.load(low_path)


def load_hiro_high_model(model_dir: str, model_suffix: str = "final") -> SAC:
    return SAC.load(os.path.join(model_dir, f"hiro_high_{model_suffix}.zip"))


def load_hiro_low_model(model_dir: str, model_suffix: str = "final") -> SAC:
    return SAC.load(os.path.join(model_dir, f"hiro_low_{model_suffix}.zip"))
