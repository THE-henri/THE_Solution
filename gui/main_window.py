from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QTabWidget, QFileDialog,
)
from PyQt6.QtCore import Qt

from gui.folder_header import FolderHeader
from gui.log_panel import LogPanel
from gui.tabs.half_life_tab import HalfLifeTab
from gui.tabs.actinometer_tab import ActinometerTab
from gui.tabs.thermal_tab import ThermalTab
from gui.tabs.extinction_coeff_tab import ExtinctionCoeffTab
from gui.tabs.spectra_tab import SpectraTab
from gui.tabs.qy_tab import QuantumYieldTab
from gui.tabs.placeholder_tab import PlaceholderTab
from gui.tabs.handbook_tab import HandbookTab
from gui.project_prefs import ProjectPrefs


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QY Tool")
        self.setMinimumSize(1100, 780)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Folder header ──────────────────────────────────────────────────
        self._folder_header = FolderHeader()
        root.addWidget(self._folder_header)

        # ── Splitter: tabs (top) | log (bottom) ───────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        # ── Tab widget ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setObjectName("main_tabs")

        self._half_life_tab = HalfLifeTab(self._folder_header)
        self._actinometer_tab = ActinometerTab()
        self._thermal_tab = ThermalTab()
        self._extinction_coeff_tab = ExtinctionCoeffTab()
        self._spectra_tab = SpectraTab()
        self._qy_tab = QuantumYieldTab()
        self._tabs.addTab(self._extinction_coeff_tab,             "Ext. Coeff.")
        self._tabs.addTab(self._spectra_tab,                      "Spectra")
        self._tabs.addTab(self._half_life_tab,                    "Half-Life")
        self._tabs.addTab(self._thermal_tab,                      "Thermal")
        self._tabs.addTab(self._actinometer_tab,                  "Actinometer")
        self._tabs.addTab(self._qy_tab,                           "Quantum Yield")
        self._tabs.addTab(PlaceholderTab("Parameters"),           "Parameters")
        self._tabs.addTab(HandbookTab(),                          "Handbook")

        splitter.addWidget(self._tabs)

        # ── Log panel ──────────────────────────────────────────────────────
        self._log = LogPanel()
        self._log.setMinimumHeight(110)
        splitter.addWidget(self._log)

        splitter.setSizes([600, 160])

        # ── Signal wiring ──────────────────────────────────────────────────
        self._folder_header.raw_folder_changed.connect(self._half_life_tab.set_raw_path)
        self._folder_header.raw_folder_changed.connect(self._actinometer_tab.set_raw_path)
        self._folder_header.raw_folder_changed.connect(self._qy_tab.set_raw_path)
        self._folder_header.raw_folder_changed.connect(self._extinction_coeff_tab.set_raw_path)
        self._folder_header.compound_name_changed.connect(self._extinction_coeff_tab.set_compound_name)
        self._folder_header.output_folder_changed.connect(self._on_output_changed)
        self._folder_header.output_folder_changed.connect(self._log.set_log_dir)
        self._folder_header.prefs_save_requested.connect(self._save_prefs)
        self._folder_header.prefs_load_requested.connect(self._load_prefs)

        # Forward worker log signals from half-life sub-panels to log panel
        self._half_life_tab._kinetics_panel.log_signal.connect(self._log.append)
        self._half_life_tab._scanning_panel.log_signal.connect(self._log.append)

        # Actinometer
        self._actinometer_tab.log_signal.connect(self._log.append)

        # Thermal
        self._thermal_tab.log_signal.connect(self._log.append)

        # Extinction coefficients
        self._extinction_coeff_tab.log_signal.connect(self._log.append)

        # Spectra
        self._spectra_tab.log_signal.connect(self._log.append)

        # Quantum Yield
        self._qy_tab.log_signal.connect(self._log.append)

    def _on_output_changed(self, path: Path):
        self._half_life_tab.set_output_path(path)
        self._actinometer_tab.set_output_path(path)
        self._thermal_tab.set_output_path(path)
        self._extinction_coeff_tab.set_output_path(path)
        self._spectra_tab.set_output_path(path)
        self._qy_tab.set_output_path(path)
        print(f"Output folder set → {path}")

    def _apply_prefs(self, prefs: ProjectPrefs):
        self._half_life_tab.apply_prefs(prefs)
        self._actinometer_tab.apply_prefs(prefs)
        self._thermal_tab.apply_prefs(prefs)
        self._extinction_coeff_tab.apply_prefs(prefs)
        self._spectra_tab.apply_prefs(prefs)
        self._qy_tab.apply_prefs(prefs)

    def _save_prefs(self):
        start_dir = str(self._folder_header.output_path or Path.home())
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preferences",
            str(Path(start_dir) / "prefs.json"),
            "JSON files (*.json);;All files (*)")
        if not file_path:
            return
        dest = Path(file_path)
        prefs = ProjectPrefs()
        self._half_life_tab.collect_prefs(prefs)
        self._actinometer_tab.collect_prefs(prefs)
        self._thermal_tab.collect_prefs(prefs)
        self._extinction_coeff_tab.collect_prefs(prefs)
        self._spectra_tab.collect_prefs(prefs)
        self._qy_tab.collect_prefs(prefs)
        prefs.save(dest)
        self._folder_header.set_prefs_status(f"Saved → {dest}")
        print(f"[Preferences] Saved to {dest}")

    def _load_prefs(self):
        start_dir = str(self._folder_header.output_path or Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Preferences", start_dir,
            "JSON files (*.json);;All files (*)")
        if not file_path:
            return
        prefs = ProjectPrefs.load_from_file(Path(file_path))
        if prefs is None:
            self._folder_header.set_prefs_status(
                f"Failed to load {Path(file_path).name}", ok=False)
            return
        self._apply_prefs(prefs)
        self._folder_header.set_prefs_status(f"Loaded → {Path(file_path).name}")
        print(f"[Preferences] Loaded from {file_path}")

    def closeEvent(self, event):
        self._log.restore_stdout()
        super().closeEvent(event)
