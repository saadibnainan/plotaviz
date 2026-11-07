"""Shared plumbing for providers that run a locally installed, already signed-in CLI.

Claude Code and the Codex CLI both let PlotaViz use a subscription the user already has, with no
API key to paste. Running someone else's executable from a desktop app has two traps, handled
once here: finding it when the app was launched with a short ``PATH``, and not leaking the
bundle's own libraries into the child process.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path


def find_executable(name: str, fallbacks: Iterable[str] = ()) -> str | None:
    """Locate an executable on ``PATH``, then in the usual install locations.

    A desktop launch (an AppImage, a .desktop file, Finder) often runs with a shorter ``PATH`` than
    the user's shell, so ``fallbacks`` lists where installers put the tool.
    """
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def child_env(drop: Iterable[str] = ()) -> dict[str, str]:
    """The environment for a CLI child process.

    A PyInstaller bundle (the AppImage and the macOS app) points ``LD_LIBRARY_PATH`` at its own
    bundled libraries, which can break an unrelated native executable. PyInstaller keeps the
    original value in ``*_ORIG``; restore it, or drop the variable if there was none.

    Args:
        drop: Variables to remove, such as markers that the parent runs inside another session.
    """
    env = dict(os.environ)
    for name in drop:
        env.pop(name, None)
    if getattr(sys, "frozen", False):
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            original = env.pop(f"{var}_ORIG", None)
            if original is not None:
                env[var] = original
            else:
                env.pop(var, None)
    return env
