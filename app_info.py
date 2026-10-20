"""
Application name, version and resource-path helper.

Bump __version__ (and RELEASE_DATE) here for every release — window title,
splash screen, Handbook and the .exe build all read from this file.
Version scheme: MAJOR.MINOR.PATCH
    PATCH  – bug fixes only
    MINOR  – new features / handbook sections, results unchanged
    MAJOR  – changes that can alter numerical results of existing workflows
"""

import sys
from pathlib import Path

APP_NAME     = "THE_Solution"
APP_SUBTITLE = "Quantum Yield & Photochemistry Analysis"
__version__  = "1.0.0"
RELEASE_DATE = "2026-10-20"


def resource_path(relative: str) -> Path:
    """
    Absolute path to a bundled resource (e.g. "gui/style.qss").

    Works both from source and inside a PyInstaller build, where data files
    are unpacked under sys._MEIPASS instead of next to the .py files.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative
