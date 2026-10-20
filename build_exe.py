"""
Build the Windows executable with PyInstaller.

    .venv/Scripts/python build_exe.py

Result: dist/THE_Solution_<version>/THE_Solution.exe  (one-folder build).
Copy the whole THE_Solution_<version> folder to the server — the .exe only
works together with the _internal folder next to it.

Before building a release: bump __version__ / RELEASE_DATE in app_info.py and
add release notes to the Handbook ("Version & Last Updated").
"""

import shutil
import subprocess
import sys
from pathlib import Path

from app_info import APP_NAME, __version__

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
TARGET = DIST / f"{APP_NAME}_{__version__}"

# (source, destination inside the bundle) — keep in sync with resource_path() users
DATA = [
    ("gui/style.qss", "gui"),
    ("gui/assets",    "gui/assets"),
    ("config",        "config"),
]


def main():
    if TARGET.exists():
        sys.exit(f"{TARGET} already exists — bump __version__ in app_info.py "
                 "or delete the folder.")

    cmd = [
        sys.executable, "-m", "PyInstaller", str(ROOT / "run_gui.py"),
        "--name", APP_NAME,
        "--onedir", "--windowed", "--noconfirm", "--clean",
        "--distpath", str(DIST / "_build"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--exclude-module", "tkinter",
        "--collect-all", "originpro", "--collect-all", "OriginExt",
        "--icon", str(ROOT / "gui" / "assets" / "app_icon.ico"),
    ]
    for src, dest in DATA:
        cmd += ["--add-data", f"{ROOT / src}{';'}{dest}"]

    subprocess.run(cmd, check=True)

    shutil.move(str(DIST / "_build" / APP_NAME), str(TARGET))
    shutil.rmtree(DIST / "_build", ignore_errors=True)
    print(f"\nBuilt {APP_NAME} {__version__}  ->  {TARGET / (APP_NAME + '.exe')}")


if __name__ == "__main__":
    main()
