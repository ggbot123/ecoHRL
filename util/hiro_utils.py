import json
import os
import warnings
from copy import deepcopy
from dataclasses import MISSING, asdict, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Type

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


def _fill_missing_dataclass_defaults(
    saved: Mapping[str, Any],
    config_type: Type[Any],
    config_name: str,
) -> Dict[str, Any]:
    config_fields = {field.name: field for field in fields(config_type)}
    unknown = set(saved) - set(config_fields)
    missing = set(config_fields) - set(saved)
    missing_required = {
        name
        for name in missing
        if config_fields[name].default is MISSING
        and config_fields[name].default_factory is MISSING
    }
    if missing_required or unknown:
        raise ValueError(
            f"Invalid {config_name} fields: "
            f"missing={sorted(missing_required)}, unknown={sorted(unknown)}"
        )

    kwargs = deepcopy(dict(saved))
    filled = {}
    for name in sorted(missing):
        field = config_fields[name]
        value = (
            field.default_factory()
            if field.default_factory is not MISSING
            else deepcopy(field.default)
        )
        kwargs[name] = value
        filled[name] = value

    if filled:
        warnings.warn(
            f"{config_name} is missing fields from an older run config; "
            f"using current defaults: {filled}",
            UserWarning,
            stacklevel=3,
        )
    return kwargs


def _hiro_config_from_mapping(saved: Mapping[str, Any]) -> HIROConfig:
    kwargs = _fill_missing_dataclass_defaults(saved, HIROConfig, "HIRO config")
    goal_sampler = kwargs.get("goal_sampler")
    low_safety_filter = kwargs.get("low_safety_filter")
    if not isinstance(goal_sampler, Mapping):
        raise ValueError("HIRO config 'goal_sampler' must be an object")
    kwargs["goal_sampler"] = GoalSamplerConfig(
        **_fill_missing_dataclass_defaults(
            goal_sampler,
            GoalSamplerConfig,
            "goal_sampler config",
        )
    )

    if low_safety_filter is None:
        kwargs["low_safety_filter"] = None
    elif isinstance(low_safety_filter, Mapping):
        kwargs["low_safety_filter"] = LowSafetyFilterConfig(
            **_fill_missing_dataclass_defaults(
                low_safety_filter,
                LowSafetyFilterConfig,
                "low_safety_filter config",
            )
        )
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
