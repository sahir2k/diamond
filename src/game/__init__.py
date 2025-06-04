"""Game environment utilities.

This module exposes :class:`PlayEnv` without importing the optional
``pygame`` dependency required for :class:`Game`.  When ``pygame`` is
installed the :class:`Game` class is also made available; otherwise the
attribute is set to ``None`` so ``from game import Game`` does not fail.
"""

from .play_env import PlayEnv

try:  # ``game.Game`` relies on pygame which might not be installed
    from .game import Game  # type: ignore
except ModuleNotFoundError:
    Game = None  # type: ignore

__all__ = ["PlayEnv", "Game"]
