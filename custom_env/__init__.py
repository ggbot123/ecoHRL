"""ecoHRL custom environment package.

This module applies global warning filters early so they also affect
multiprocessing worker processes (e.g., SubprocVecEnv) which may import
`custom_env` without running the top-level training entry script.
"""

from __future__ import annotations

import warnings


# Suppress a noisy third-party warning from pygame (setuptools/pkg_resources deprecation).
# Targets only the message emitted during `pygame.pkgdata` import.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)
