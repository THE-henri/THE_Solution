import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QSizePolicy, QLabel,
)
from PyQt6.QtCore import Qt

from .info_button import InfoButton


class PlotWidget(QWidget):
    """
    Embeds a matplotlib Figure with a custom toolbar:
      Pan | Zoom | Reset     [ℹ]     💾 Save image…

    Usage
    -----
    pw = PlotWidget(info_title="…", info_text="…")
    pw.set_figure(fig)   # replaces any previous figure
    pw.clear()           # removes figure, shows placeholder
    pw.set_save_dir(path)  # default directory for the Save dialog
    """

    def __init__(self, info_title: str = "Plot", info_text: str = "",
                 min_height: int = 320, parent=None):
        super().__init__(parent)
        self._info_title  = info_title
        self._info_text   = info_text
        self._min_height  = min_height
        self._fig         = None
        self._canvas      = None
        self._mpl_toolbar = None
        self._save_dir:   Path | None = None
        self._default_filename: str = ""
        self._build_ui()

    # ── Build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)

        # Canvas area
        self._canvas_wrap = QWidget()
        self._canvas_wrap.setMinimumHeight(self._min_height)
        self._canvas_wrap.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas_layout = QVBoxLayout(self._canvas_wrap)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("No data loaded")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color:#444466; font-size:11pt;")
        self._canvas_layout.addWidget(self._placeholder)

        root.addWidget(self._canvas_wrap)

        # Toolbar row
        tb = QHBoxLayout()
        tb.setSpacing(4)
        tb.setContentsMargins(0, 0, 0, 0)

        self._btn_pan  = QPushButton("⇔ Pan")
        self._btn_zoom = QPushButton("⌕ Zoom")
        self._btn_home = QPushButton("↩ Reset")
        for btn in (self._btn_pan, self._btn_zoom, self._btn_home):
            btn.setObjectName("plot_tool_btn")
            btn.setFixedHeight(24)
            btn.setCheckable(True)
            tb.addWidget(btn)

        tb.addStretch()

        info_btn = InfoButton(self._info_title, self._info_text)
        tb.addWidget(info_btn)

        self._btn_save = QPushButton("💾 Save image…")
        self._btn_save.setObjectName("plot_tool_btn")
        self._btn_save.setFixedHeight(24)
        self._btn_save.clicked.connect(self._save_image)
        tb.addWidget(self._btn_save)

        root.addLayout(tb)

        self._btn_pan.clicked.connect(self._on_pan)
        self._btn_zoom.clicked.connect(self._on_zoom)
        self._btn_home.clicked.connect(self._on_home)

    # ── Public API ────────────────────────────────────────────────────────

    def set_figure(self, fig):
        """Replace the current figure. Closes the old one."""
        if self._fig is not None:
            plt.close(self._fig)

        # Remove old canvas / placeholder
        while self._canvas_layout.count():
            item = self._canvas_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._mpl_toolbar is not None:
            self._mpl_toolbar.deleteLater()
            self._mpl_toolbar = None

        self._fig    = fig
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas_layout.addWidget(self._canvas)

        # Hidden toolbar for nav actions
        self._mpl_toolbar = NavigationToolbar2QT(self._canvas, self)
        self._mpl_toolbar.hide()

        self._canvas.draw()

    def clear(self):
        """Remove figure and show the empty placeholder."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

        while self._canvas_layout.count():
            item = self._canvas_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._mpl_toolbar is not None:
            self._mpl_toolbar.deleteLater()
            self._mpl_toolbar = None

        self._canvas = None
        self._canvas_layout.addWidget(self._placeholder)
        self._placeholder.setVisible(True)

        for btn in (self._btn_pan, self._btn_zoom):
            btn.setChecked(False)

    # ── Navigation ────────────────────────────────────────────────────────

    def _on_pan(self):
        if self._mpl_toolbar is None:
            self._btn_pan.setChecked(False)
            return
        if self._btn_zoom.isChecked():
            self._btn_zoom.setChecked(False)
        self._mpl_toolbar.pan()

    def _on_zoom(self):
        if self._mpl_toolbar is None:
            self._btn_zoom.setChecked(False)
            return
        if self._btn_pan.isChecked():
            self._btn_pan.setChecked(False)
        self._mpl_toolbar.zoom()

    def _on_home(self):
        self._btn_pan.setChecked(False)
        self._btn_zoom.setChecked(False)
        if self._mpl_toolbar is not None:
            self._mpl_toolbar.home()

    # ── Save dir ──────────────────────────────────────────────────────────

    def set_save_dir(self, path: Path | None):
        """Set the default directory opened by the Save dialog."""
        self._save_dir = path

    def set_default_filename(self, name: str):
        """Set the suggested filename that pre-fills the Save dialog."""
        self._default_filename = name

    # ── Save ──────────────────────────────────────────────────────────────

    def _save_image(self):
        if self._fig is None:
            return
        if self._save_dir and self._default_filename:
            start = str(self._save_dir / self._default_filename)
        elif self._save_dir:
            start = str(self._save_dir)
        else:
            start = self._default_filename
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", start,
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Plot saved → {path}")
