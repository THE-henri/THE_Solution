"""
QY Tool — GUI entry point.

Run from the QY_Tool directory:
    python run_gui.py
or with the venv active:
    .venv/Scripts/python run_gui.py
"""

import sys
from pathlib import Path

# Ensure QY_Tool root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication, QAbstractSpinBox
from PyQt6.QtCore import QLocale, QObject, QEvent
from gui.splash import SplashScreen
from gui.main_window import MainWindow


class _NoScrollSpinFilter(QObject):
    """Application-level filter: ignore wheel events on all spin boxes."""
    def eventFilter(self, obj, event):
        if isinstance(obj, QAbstractSpinBox) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False


def _load_stylesheet(app: QApplication):
    gui_dir = Path(__file__).resolve().parent / "gui"
    qss = gui_dir / "style.qss"
    if qss.exists():
        assets = (gui_dir / "assets").as_posix()
        with open(qss, encoding="utf-8") as f:
            app.setStyleSheet(f.read().replace("$ASSETS", assets))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("QY Tool")
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
