from __future__ import annotations

import warnings
from typing import Any

# Suppress a noisy third-party warning from pygame (setuptools/pkg_resources deprecation).
# Keep this at the very top so it applies before any indirect pygame imports.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

from configs.conf import TRAIN_CONFIG
from util.train_launcher import make_env, run_training, set_global_seed


def main(**overrides: Any) -> None:
    """Run training from configs.conf.TRAIN_CONFIG, with optional keyword overrides."""
    cfg = dict(TRAIN_CONFIG)
    cfg.update(overrides)
    run_training(**cfg)


if __name__ == "__main__":
    main()
