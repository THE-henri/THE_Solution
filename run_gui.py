"""
THE_Solution — GUI entry point.

Run from the THE_Solution directory:
    python run_gui.py
or with the venv active:
    .venv/Scripts/python run_gui.py
"""

import sys
import traceback
from pathlib import Path

# Ensure THE_Solution root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication, QAbstractSpinBox, QMessageBox
from PyQt6.QtCore import QLocale, QObject, QEvent
from PyQt6.QtGui import QIcon
from gui.splash import SplashScreen
from gui.main_window import MainWindow
from app_info import APP_NAME, __version__, resource_path


class _NoScrollSpinFilter(QObject):
    """Application-level filter: ignore wheel events on all spin boxes."""
    def eventFilter(self, obj, event):
        if isinstance(obj, QAbstractSpinBox) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False


def _excepthook(exc_type, exc, tb):
    """Show unhandled errors instead of silently closing (the .exe has no console)."""
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    print(f"[ERROR] Unhandled exception:\n{text}")
    if QApplication.instance() is not None:
        box = QMessageBox(QMessageBox.Icon.Critical, f"{APP_NAME} — unexpected error",
                          f"{exc_type.__name__}: {exc}\n\n"
                          "The program will try to continue. Please report this "
                          f"(with version {__version__}) if it keeps happening.")
        box.setDetailedText(text)
        box.exec()


def _load_stylesheet(app: QApplication):
    gui_dir = resource_path("gui")
    qss = gui_dir / "style.qss"
    if qss.exists():
        assets = (gui_dir / "assets").as_posix()
        with open(qss, encoding="utf-8") as f:
            app.setStyleSheet(f.read().replace("$ASSETS", assets))


def main():
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(resource_path("gui/assets/app_icon.png"))))
    if sys.platform == "win32":
        # own taskbar entry/icon instead of the generic python.exe one
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_NAME)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    _load_stylesheet(app)
    _scroll_filter = _NoScrollSpinFilter(app)
    app.installEventFilter(_scroll_filter)

    splash = SplashScreen()
    result = splash.exec()

    if result != SplashScreen.DialogCode.Accepted or not splash.agreed:
        sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
