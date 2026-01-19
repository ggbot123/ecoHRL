import os
from typing import Optional, Tuple

from rl.algos.sac.sac import SAC


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


def load_hiro_models(
    model_dir: str,
    *,
    high_model_dir: Optional[str] = None,
    low_model_dir: Optional[str] = None,
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
    high_path = os.path.join(high_dir, "hiro_high_final.zip")
    low_path = os.path.join(low_dir, "hiro_low_final.zip")
    return SAC.load(high_path), SAC.load(low_path)


def load_hiro_high_model(model_dir: str) -> SAC:
    return SAC.load(os.path.join(model_dir, "hiro_high_final.zip"))
