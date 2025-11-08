"""
Backward-compatible import shim for historical `RL_enviroment`.

The actual implementation lives in `minesweeper_env_rl`. Keep this module so
existing notebooks or projects that still import RL_enviroment don't break.
"""
from __future__ import annotations

import warnings

from minesweeper_env_rl import *  # noqa: F401,F403 - re-export public API
from minesweeper_env_rl import __all__ as _new_all
from minesweeper_env_rl import __version__

warnings.warn(
    "RL_enviroment is deprecated; import minesweeper_env_rl instead. "
    "This shim will be removed in a future major release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _new_all
