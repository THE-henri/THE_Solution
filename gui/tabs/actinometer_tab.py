"""
Actinometer tab — three sub-panels:

  A. Chemical Actinometry   (workflows/actinometer_analysis.py)
  B. LED Characterisation   (LED block from workflows/quantum_yield.py)
  C. LED Actinometry        independent N_chem from actinometer solutions
                            measured under LED irradiation; compared to N_LED
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem,
    QFileDialog, QAbstractItemView, QAbstractScrollArea, QFrame, QTabWidget,
)

from gui.tabs.actinometer_core import (
    ACTINOMETERS,
    ActinometerResult, LEDResult, LEDActinometerResult,
    run_actinometry_file, plot_actinometry_result,
    run_led_characterization, plot_led_result,
    plot_led_diagnostic,
    run_led_actinometry_file, plot_led_actinometry_result,
)
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.widgets.info_button import InfoButton
from gui.worker import Worker


# ══════════════════════════════════════════════════════════════════════════════
# Sub-panel A — Chemical Actinometry
# ══════════════════════════════════════════════════════════════════════════════

class _ChemActinometerPanel(QWidget):
    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path: Optional[Path] = None
        self._raw_path:    Optional[Path] = None
        self._results: list[ActinometerResult] = []
        self._current_idx: int = 0
        self._worker: Optional[Worker] = None
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        # ── Stage 1 — Files ────────────────────────────────────────────────
        self._stage1 = StageCard("Stage 1 — Files")
        self._stage1.add_info_button(
            "Input Files",
            "Select one or more actinometry CSV files exported from the Cary 60 "
            "spectrophotometer.\n\n"
            "Each file should contain a time-resolved absorbance scan series "
            "recorded while the actinometer solution is irradiated.\n\n"
            "Multiple files are processed independently and each yields one "
            "photon flux value."
        )

        hint = QLabel("Select one or more actinometry CSV files (Cary 60 format).")
        hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint)

        ctrl = QHBoxLayout()
        self._select_btn = QPushButton("Select files…")
        self._select_btn.setFixedWidth(120)
        self._select_btn.clicked.connect(self._select_files)
        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.setFixedWidth(120)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.setFixedWidth(70)
        self._clear_btn.clicked.connect(self._clear_files)
        ctrl.addWidget(self._select_btn)
        ctrl.addWidget(self._remove_btn)
        ctrl.addWidget(self._clear_btn)
        ctrl.addStretch()
        self._stage1.add_layout(ctrl)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(120)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._stage1.add_widget(self._file_list)
        self._stage1.set_status(WAITING)
        layout.addWidget(self._stage1)

        # ── Stage 2 — Parameters ───────────────────────────────────────────
        self._stage2 = StageCard("Stage 2 — Parameters")
        self._stage2.add_info_button(
            "Actinometry Parameters",
            "Actinometer: choose the chemical actinometer that matches your "
            "irradiation wavelength and experimental conditions.\n\n"
            "λ_irr (nm): irradiation wavelength used during the experiment.\n\n"
            "Irradiation time (s): total duration of one irradiation period.\n\n"
            "Volume (mL) / Path length (cm): cuvette geometry for "
            "concentration calculations.\n\n"
            "Scans / group: number of consecutive scans averaged into one "
            "time point.\n\n"
            "λ tolerance (nm): window around the probe wavelength used to "
            "locate the absorption peak in each scan."
        )

        row1 = QHBoxLayout()
        # Actinometer choice
        act_col = QVBoxLayout()
        act_col.setSpacing(4)
        _act_lbl = QLabel("Actinometer")
        _act_lbl.setObjectName("pref_label")
        _act_hdr = QHBoxLayout()
        _act_hdr.setSpacing(4)
        _act_hdr.addWidget(_act_lbl)
        _act_hdr.addWidget(InfoButton(
            "Actinometer",
            "Select the chemical actinometer that matches your irradiation\n"
            "wavelength and experimental conditions.\n\n"
            "Each entry shows the applicable wavelength range.\n"
            "The most common choice for visible light is Reinecke's salt\n"
            "or potassium ferrioxalate depending on wavelength."))
        _act_hdr.addStretch()
        act_col.addLayout(_act_hdr)
        self._act_combo = QComboBox()
        for key, val in ACTINOMETERS.items():
            rng = val["wavelength_range_nm"]
            self._act_combo.addItem(
                f"{key} — {val['name']} ({rng[0]}–{rng[1]} nm)", key)
        self._act_combo.setMinimumWidth(260)
        act_col.addWidget(self._act_combo)
        row1.addLayout(act_col)
        self._lambda_spin  = self._dspin(row1, "λ_irr (nm)", 300.0, 900.0, 579.0, 1.0, 1, pref=True,
            info_title="λ_irr (nm)",
            info_text="Irradiation wavelength in nanometres.\nMust match the wavelength used during the actinometry experiment.\nUsed to look up the actinometer quantum yield and molar absorptivity.")
        self._irr_time_spin = self._dspin(row1, "Irradiation time (s)", 0.1, 10000.0, 60.0, 1.0, pref=True,
            info_title="Irradiation time (s)",
            info_text="Duration of a single irradiation period in seconds.\nThis is the time the actinometer solution was exposed to light\nbetween consecutive UV-Vis measurements.")
        row1.addStretch()
        self._stage2.add_layout(row1)

        row2 = QHBoxLayout()
        self._volume_spin   = self._dspin(row2, "Volume (mL)", 0.01, 100.0, 2.0, 0.1, pref=True,
            info_title="Volume (mL)",
            info_text="Volume of actinometer solution in the cuvette (mL).\nUsed together with path length to convert absorbance changes\ninto molar concentration changes and ultimately photon flux.")
        self._path_spin     = self._dspin(row2, "Path length (cm)", 0.001, 100.0, 1.0, 0.1, pref=True,
            info_title="Path length (cm)",
            info_text="Optical path length of the cuvette in centimetres (typically 1 cm).\nEnter the actual cuvette path length used — errors here propagate\ndirectly into the reported photon flux.")
        self._spg_spin      = self._ispin(row2, "Scans / group", 1, 50, 3, pref=True,
            info_title="Scans / group",
            info_text="Number of consecutive UV-Vis scans averaged into one time point.\nHigher values reduce noise but lower temporal resolution.\nMatch to how the raw data was collected on the spectrometer.")
        self._tol_spin      = self._dspin(row2, "λ tolerance (nm)", 0.1, 10.0, 1.0, 0.5, pref=True,
            info_title="λ tolerance (nm)",
            info_text="Window around the actinometer probe wavelength (nm) used to locate\nand average the absorption peak in each scan.\nIncrease if the peak is slightly shifted or the spectrum is noisy.")
        row2.addStretch()
        self._stage2.add_layout(row2)

        for sig in (
            self._act_combo.currentIndexChanged,
            self._lambda_spin.valueChanged,
            self._irr_time_spin.valueChanged,
            self._volume_spin.valueChanged,
            self._path_spin.valueChanged,
            self._spg_spin.valueChanged,
            self._tol_spin.valueChanged,
        ):
            sig.connect(self._mark_stale)

        self._stage2.set_status(READY)
        layout.addWidget(self._stage2)

        # ── Stage 3 — Run & Results ────────────────────────────────────────
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(
            "Chemical Actinometry Results",
            "The plot shows the actinometer rate function vs irradiation time. "
            "A linear fit through the origin gives the slope N (mol s⁻¹), "
            "the photon flux at the irradiation wavelength.\n\n"
            "Results table columns:\n"
            "  N (mol s⁻¹)     — photon flux\n"
            "  N_std (mol s⁻¹) — standard deviation from the fit\n"
            "  R²              — goodness of linear fit\n\n"
            "Use 'Save to master CSV' to write all results to "
            "photon_flux_master.csv for later use in the Quantum Yield tab."
        )

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setFixedWidth(100)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:9pt;")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._status_lbl)
        run_row.addStretch()
        self._stage3.add_layout(run_row)

        # Plot navigation (shown when > 1 file processed)
        self._nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setFixedWidth(80)
        self._prev_btn.clicked.connect(self._show_prev)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setFixedWidth(80)
        self._next_btn.clicked.connect(self._show_next)
        self._nav_lbl = QLabel("")
        self._nav_row.addWidget(self._prev_btn)
        self._nav_row.addWidget(self._nav_lbl)
        self._nav_row.addWidget(self._next_btn)
        self._nav_row.addStretch()
        nav_widget = QWidget()
        nav_widget.setLayout(self._nav_row)
        nav_widget.setVisible(False)
        self._nav_widget = nav_widget
        self._stage3.add_widget(nav_widget)

        self._plot = PlotWidget(
            info_title="Chemical Actinometry",
            info_text=(
                "Linear fit of the actinometer rate function vs irradiation time.\n"
                "Slope = photon flux N (mol s⁻¹)."
            ),
            min_height=320,
        )
        self._stage3.add_widget(self._plot)

        # Results summary table
        self._res_table = QTableWidget(0, 7)
        self._res_table.setHorizontalHeaderLabels(
            ["File", "Actinometer", "λ_irr (nm)", "ε (M⁻¹cm⁻¹)",
             "N (mol s⁻¹)", "N_std (mol s⁻¹)", "R²"])
        rh = self._res_table.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 7):
            rh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._res_table.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self._res_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._res_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._stage3.add_widget(self._res_table)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save to master CSV")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_master_csv)
        save_row.addWidget(self._save_btn)
        save_row.addStretch()
        self._stage3.add_layout(save_row)

        self._stage3.set_status(WAITING)
        layout.addWidget(self._stage3)
        layout.addStretch()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _dspin(self, parent, label, lo, hi, val, step, decimals=3, pref=False,
               info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setMinimumWidth(110)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    def _ispin(self, parent, label, lo, hi, val, pref=False,
               info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setMinimumWidth(80)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    # ── Public slots ───────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        plots_dir = path / "actinometer" / "results" / "plots"
        self._plot.set_save_dir(plots_dir)

    def set_raw_path(self, path: Path):
        self._raw_path = path

    # ── File list ──────────────────────────────────────────────────────────

    def _select_files(self):
        start = str(self._raw_path or self._output_path or Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select actinometry CSV files", start, "CSV files (*.csv)")
        existing = {self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self._file_list.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.ItemDataRole.UserRole, p)
                self._file_list.addItem(item)
        self._update_stage1()

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))
        self._update_stage1()
        self._mark_stale()

    def _clear_files(self):
        self._file_list.clear()
        self._update_stage1()
        self._mark_stale()

    def _update_stage1(self):
        if self._file_list.count() == 0:
            self._stage1.set_status(WAITING)
            self._run_btn.setEnabled(False)
        else:
            self._stage1.set_status(READY)
            self._run_btn.setEnabled(True)

    def _mark_stale(self):
        if self._results:
            self._stage3.set_status(STALE)

    # ── Run ────────────────────────────────────────────────────────────────

    def _run(self):
        if self._file_list.count() == 0:
            return

        files = [Path(self._file_list.item(i).data(Qt.ItemDataRole.UserRole))
                 for i in range(self._file_list.count())]
        act_key  = self._act_combo.currentData()
        lam_irr  = self._lambda_spin.value()
        irr_time = self._irr_time_spin.value()
        volume   = self._volume_spin.value()
        path_len = self._path_spin.value()
        spg      = self._spg_spin.value()
        tol      = self._tol_spin.value()

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._stage3.set_status(WAITING)
        self._results = []

        def _run_all():
            results = []
            for fp in files:
                print(f"\n{'='*50}")
                print(f"Processing {fp.name}")
                r = run_actinometry_file(
                    filepath=fp,
                    actinometer_choice=act_key,
                    irradiation_wavelength_nm=lam_irr,
                    irradiation_time_s=irr_time,
                    volume_mL=volume,
                    path_length_cm=path_len,
                    scans_per_group=spg,
                    wavelength_tolerance_nm=tol,
                )
                results.append(r)
            return results

        self._worker = Worker(_run_all)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, results: list):
        self._results = results
        self._current_idx = 0
        self._populate_table()
        self._show_result(0)
        self._nav_widget.setVisible(len(results) > 1)
        self._stage3.set_status(DONE)
        self._status_lbl.setText(
            f"Done — {len(results)} file(s) processed.")
        self._save_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._stage3.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_finished(self):
        self._run_btn.setEnabled(True)

    # ── Results display ────────────────────────────────────────────────────

    def _populate_table(self):
        self._res_table.setRowCount(0)
        for r in self._results:
            row = self._res_table.rowCount()
            self._res_table.insertRow(row)
            vals = [
                r.file,
                r.actinometer_name,
                f"{r.irradiation_nm:.0f}",
                f"{r.epsilon_M_cm:.4e}",
                f"{r.photon_flux_mol_s:.4e}",
                f"{r.photon_flux_std_mol_s:.4e}",
                f"{r.r2:.4f}",
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._res_table.setItem(row, col, item)

    def _show_result(self, idx: int):
        if not self._results:
            return
        r = self._results[idx]
        fig = plot_actinometry_result(r)
        stem = Path(r.file).stem
        self._plot.set_default_filename(
            f"{stem}_Act{self._act_combo.currentData()}_"
            f"{r.irradiation_nm:.0f}nm.png")
        self._plot.set_figure(fig)
        self._nav_lbl.setText(
            f"  {idx + 1} / {len(self._results)}  ")

    def _show_prev(self):
        if self._results:
            self._current_idx = max(0, self._current_idx - 1)
            self._show_result(self._current_idx)

    def _show_next(self):
        if self._results:
            self._current_idx = min(len(self._results) - 1,
                                    self._current_idx + 1)
            self._show_result(self._current_idx)

    # ── Save master CSV ────────────────────────────────────────────────────

    def _save_master_csv(self):
        if not self._results:
            return
        if self._output_path:
            out_dir = self._output_path / "actinometer" / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path.home()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save master CSV",
            str(out_dir / "photon_flux_master.csv"), "CSV (*.csv)")
        if not path:
            return
        rows = [{
            "File":                  r.file,
            "Actinometer":           r.actinometer_name,
            "Irradiation_nm":        r.irradiation_nm,
            "QY":                    r.QY,
            "Epsilon_M_cm":          r.epsilon_M_cm,
            "Volume_mL":             r.volume_mL,
            "Path_length_cm":        r.path_length_cm,
            "Photon_flux_mol_s":     r.photon_flux_mol_s,
            "Photon_flux_std_mol_s": r.photon_flux_std_mol_s,
            "R2":                    r.r2,
        } for r in self._results]
        df_new = pd.DataFrame(rows)
        p = Path(path)
        if p.exists():
            df_existing = pd.read_csv(p)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        df_new.to_csv(p, index=False)
        print(f"[Actinometer] Master CSV saved → {path}")
        self.log_signal.emit(f"[Actinometer] Master CSV saved → {path}", "INFO")

    # ── Preferences ────────────────────────────────────────────────────────

    def apply_prefs(self, prefs):
        p = prefs.actinometer
        idx = self._act_combo.findData(p.actinometer_choice)
        if idx >= 0:
            self._act_combo.setCurrentIndex(idx)
        self._lambda_spin.setValue(p.irradiation_wavelength_nm)
        self._irr_time_spin.setValue(p.irradiation_time_s)
        self._volume_spin.setValue(p.volume_mL)
        self._path_spin.setValue(p.path_length_cm)
        self._spg_spin.setValue(p.scans_per_group)
        self._tol_spin.setValue(p.wavelength_tolerance_nm)

    def collect_prefs(self, prefs):
        prefs.actinometer.actinometer_choice        = self._act_combo.currentData()
        prefs.actinometer.irradiation_wavelength_nm = self._lambda_spin.value()
        prefs.actinometer.irradiation_time_s        = self._irr_time_spin.value()
        prefs.actinometer.volume_mL                 = self._volume_spin.value()
        prefs.actinometer.path_length_cm            = self._path_spin.value()
        prefs.actinometer.scans_per_group           = self._spg_spin.value()
        prefs.actinometer.wavelength_tolerance_nm   = self._tol_spin.value()


# ══════════════════════════════════════════════════════════════════════════════
# Sub-panel B — LED Characterisation
# ══════════════════════════════════════════════════════════════════════════════

class _LEDCharacterisationPanel(QWidget):
    log_signal        = pyqtSignal(str, str)
    led_result_ready  = pyqtSignal(object)   # emits LEDResult after a successful run

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path: Optional[Path] = None
        self._raw_path:    Optional[Path] = None
        self._result: Optional[LEDResult] = None
        self._worker: Optional[Worker] = None
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        # ── Stage 1 — Input Files ──────────────────────────────────────────
        self._stage1 = StageCard("Stage 1 — Input Files")
        self._stage1.add_info_button(
            "LED Input Files",
            "Provide the emission spectrum and power time series for your LED.\n\n"
            "Emission spectrum CSV: columns wavelength_nm and intensity_au. "
            "One file before irradiation is required; one after is optional "
            "(will be averaged with the 'before' spectrum).\n\n"
            "Power time series CSV: columns time_s and power_mW. "
            "Records the optical power measured at the sample position. "
            "A 'before' file is required; an 'after' file is optional."
        )

        hint = QLabel(
            "Emission CSV columns: wavelength_nm, intensity_au\n"
            "Power CSV columns: time_s, power_mW"
        )
        hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint)

        self._em_before = self._file_row(
            self._stage1, "Emission spectrum (before)", required=True)
        self._em_after  = self._file_row(
            self._stage1, "Emission spectrum (after, optional)", required=False)
        self._pw_before = self._file_row(
            self._stage1, "Power time series (before)", required=True)
        self._pw_after  = self._file_row(
            self._stage1, "Power time series (after, optional)", required=False)

        # Diagnostic toggle + plot
        diag_row = QHBoxLayout()
        self._diag_chk = QCheckBox("Show data diagnostic")
        self._diag_chk.setObjectName("pref_cb")
        self._diag_chk.toggled.connect(self._on_diag_toggled)
        diag_row.addWidget(self._diag_chk)
        diag_row.addStretch()
        self._stage1.add_layout(diag_row)

        self._diag_plot = PlotWidget(
            info_title="Data diagnostic",
            info_text=(
                "Left: raw emission spectra (not normalised) — overlay lets you\n"
                "check that the before/after spectra have the same shape.\n\n"
                "Right: power time series before and after irradiation.\n"
                "The drift percentage is shown in the panel title."
            ),
            min_height=300,
        )
        self._diag_plot.setVisible(False)
        self._stage1.add_widget(self._diag_plot)

        self._stage1.set_status(WAITING)
        layout.addWidget(self._stage1)

        # ── Stage 2 — Processing ───────────────────────────────────────────
        self._stage2 = StageCard("Stage 2 — Processing")
        self._stage2.add_info_button(
            "LED Processing Parameters",
            "Power to use: which power measurement(s) to use for converting "
            "the spectral shape to absolute photon flux.\n\n"
            "Integration mode:\n"
            "  scalar — fast: computes a flux-weighted effective wavelength λ_eff "
            "and a single N value (mol s⁻¹).\n"
            "  full — spectral ODE integration: returns N(λ) as a spectrum CSV "
            "for use with the full spectral QY calculation.\n\n"
            "Emission threshold: fraction of peak intensity below which the "
            "spectrum is set to zero (removes noise in the wings).\n\n"
            "Savitzky-Golay smoothing: polynomial smoothing applied to the "
            "emission spectrum before integration."
        )

        row1 = QHBoxLayout()
        pw_col = QVBoxLayout()
        pw_col.setSpacing(4)
        _pw_lbl = QLabel("Power to use")
        _pw_lbl.setObjectName("pref_label")
        _pw_hdr = QHBoxLayout()
        _pw_hdr.setSpacing(4)
        _pw_hdr.addWidget(_pw_lbl)
        _pw_hdr.addWidget(InfoButton(
            "Power to use",
            "Which power measurement(s) to use when scaling the emission spectrum\n"
            "to absolute photon flux.\n\n"
            "'before' — power recorded before irradiation\n"
            "'after' — power recorded after irradiation\n"
            "'average' — mean of before and after\n\n"
            "Use 'average' when the LED power drifts noticeably over the experiment."))
        _pw_hdr.addStretch()
        pw_col.addLayout(_pw_hdr)
        self._pw_combo = QComboBox()
        self._pw_combo.addItems(["before", "after", "average"])
        self._pw_combo.setMinimumWidth(100)
        pw_col.addWidget(self._pw_combo)
        row1.addLayout(pw_col)

        mode_col = QVBoxLayout()
        mode_col.setSpacing(4)
        _mode_lbl = QLabel("Integration mode")
        _mode_lbl.setObjectName("pref_label")
        _mode_hdr = QHBoxLayout()
        _mode_hdr.setSpacing(4)
        _mode_hdr.addWidget(_mode_lbl)
        _mode_hdr.addWidget(InfoButton(
            "Integration mode",
            "'scalar' — fast mode: computes a single flux-weighted effective\n"
            "wavelength λ_eff and one N value (mol s⁻¹). Use for monochromatic\n"
            "or narrow-band LEDs.\n\n"
            "'full' — spectral ODE integration: returns N(λ) as a spectrum CSV\n"
            "for use with the full spectral quantum yield calculation. Required\n"
            "for broad-band sources."))
        _mode_hdr.addStretch()
        mode_col.addLayout(_mode_hdr)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("scalar  (fast, flux-weighted λ_eff)", "scalar")
        self._mode_combo.addItem("full  (spectral ODE integration)", "full")
        self._mode_combo.setCurrentIndex(1)
        self._mode_combo.setMinimumWidth(220)
        mode_col.addWidget(self._mode_combo)
        row1.addLayout(mode_col)
        row1.addStretch()
        self._stage2.add_layout(row1)

        row2 = QHBoxLayout()
        self._threshold_spin = self._dspin_s2(
            row2, "Emission threshold\n(fraction of peak)", 0.0, 1.0, 0.005, 0.001, pref=True,
            info_title="Emission threshold",
            info_text="Fraction of peak emission intensity below which spectral values are\nset to zero. Removes noise in the wings of the spectrum.\n\nTypical value: 0.005 (0.5 % of peak). Increase if wing noise is\nvisible in the photon flux spectrum.")
        self._std_spin = self._dspin_s2(
            row2, "Manual N_std (mol s⁻¹)\n0 = auto from drift", 0.0, 1.0, 0.0, 1e-10, pref=True,
            info_title="Manual N_std",
            info_text="Manual standard deviation for the photon flux (mol s⁻¹).\nLeave at 0 to compute it automatically from LED power drift\n(difference between before/after power readings).\n\nSet manually only when you have an independent uncertainty estimate.")
        self._std_spin.setDecimals(10)
        row2.addStretch()
        self._stage2.add_layout(row2)

        row3 = QHBoxLayout()
        smth_col = QVBoxLayout()
        smth_col.setSpacing(4)
        self._smooth_chk = QCheckBox("Savitzky-Golay smoothing")
        self._smooth_chk.setObjectName("pref_cb")
        self._smooth_chk.setChecked(True)
        self._smooth_chk.stateChanged.connect(self._on_smooth_toggled)
        _smth_hdr = QHBoxLayout()
        _smth_hdr.setSpacing(4)
        _smth_hdr.addWidget(self._smooth_chk)
        _smth_hdr.addWidget(InfoButton(
            "Savitzky–Golay smoothing",
            "Applies polynomial smoothing to the emission spectrum before\n"
            "integration. Reduces noise without distorting peak positions.\n\n"
            "Recommended for most measurements. Disable only if the raw\n"
            "spectrum is already very clean or shows artefacts from smoothing."))
        _smth_hdr.addStretch()
        smth_col.addWidget(QLabel(""))   # spacer to align with spinboxes
        smth_col.addLayout(_smth_hdr)
        row3.addLayout(smth_col)
        self._sg_window = self._ispin_s2(row3, "SG window\n(odd integer)", 3, 201, 11, pref=True,
            info_title="SG window",
            info_text="Window size for Savitzky–Golay smoothing (must be odd).\nLarger windows give smoother spectra but can broaden peaks.\nTypical value: 11. Set to 3 for minimal smoothing.")
        self._sg_order  = self._ispin_s2(row3, "SG order", 1, 10, 3, pref=True,
            info_title="SG order",
            info_text="Polynomial order for Savitzky–Golay smoothing.\nHigher order preserves sharper spectral features.\nMust be less than the window size.")
        row3.addStretch()
        self._stage2.add_layout(row3)

        for sig in (
            self._pw_combo.currentIndexChanged,
            self._mode_combo.currentIndexChanged,
            self._threshold_spin.valueChanged,
            self._std_spin.valueChanged,
            self._smooth_chk.stateChanged,
            self._sg_window.valueChanged,
            self._sg_order.valueChanged,
        ):
            sig.connect(self._mark_stale)

        self._stage2.set_status(READY)
        layout.addWidget(self._stage2)

        # ── Stage 3 — Run & Results ────────────────────────────────────────
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(
            "LED Characterisation Results",
            "Top panel: normalised emission spectrum with threshold boundaries "
            "marking the integration region.\n\n"
            "Bottom panel: spectral photon flux density N(λ) [pmol s⁻¹ nm⁻¹] — "
            "how many photons per second are delivered at each wavelength.\n\n"
            "The saved CSV contains N(λ) and can be loaded in the Quantum Yield "
            "tab as the photon flux source for a full spectral calculation."
        )

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setFixedWidth(100)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:9pt;")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._status_lbl)
        run_row.addStretch()
        self._stage3.add_layout(run_row)

        self._plot = PlotWidget(
            info_title="LED Characterisation",
            info_text=(
                "Top: emission spectrum (normalised) with threshold boundaries.\n"
                "Bottom: spectral photon flux density N(λ) [pmol s⁻¹ nm⁻¹]."
            ),
            min_height=420,
        )
        self._stage3.add_widget(self._plot)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save spectrum CSV")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_csv)
        save_row.addWidget(self._save_btn)
        save_row.addStretch()
        self._stage3.add_layout(save_row)

        self._stage3.set_status(WAITING)
        layout.addWidget(self._stage3)
        layout.addStretch()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _file_row(self, card: StageCard, label: str, required: bool) -> QLineEdit:
        row = QHBoxLayout()
        lbl_text = label if not required else label + "  *"
        row.addWidget(QLabel(lbl_text))
        edit = QLineEdit()
        edit.setPlaceholderText("(not selected)")
        edit.setReadOnly(True)
        edit.setMinimumWidth(260)
        edit.textChanged.connect(self._update_stage1)
        row.addWidget(edit, stretch=1)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(lambda _, e=edit: self._browse_file(e))
        row.addWidget(btn)
        clr = QPushButton("✕")
        clr.setFixedWidth(28)
        clr.setToolTip("Clear")
        clr.clicked.connect(lambda _, e=edit: e.clear())
        row.addWidget(clr)
        card.add_layout(row)
        return edit

    def _browse_file(self, edit: QLineEdit):
        start = str(self._raw_path or self._output_path or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", start, "CSV files (*.csv)")
        if path:
            edit.setText(path)

    def _dspin_s2(self, parent, label, lo, hi, val, step, decimals=4, pref=False,
                  info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setMinimumWidth(120)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    def _ispin_s2(self, parent, label, lo, hi, val, pref=False,
                  info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setMinimumWidth(80)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    def _on_smooth_toggled(self):
        enabled = self._smooth_chk.isChecked()
        self._sg_window.setEnabled(enabled)
        self._sg_order.setEnabled(enabled)
        self._mark_stale()

    # ── Public slots ───────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        plots_dir = path / "led" / "results" / "plots"
        self._plot.set_save_dir(plots_dir)

    def set_raw_path(self, path: Path):
        self._raw_path = path

    # ── Stage 1 status ─────────────────────────────────────────────────────

    def _update_stage1(self):
        em_ok = bool(self._em_before.text().strip())
        pw_ok = bool(self._pw_before.text().strip())
        if em_ok and pw_ok:
            self._stage1.set_status(READY)
            self._run_btn.setEnabled(True)
        else:
            self._stage1.set_status(WAITING)
            self._run_btn.setEnabled(False)
        self._mark_stale()
        self._refresh_diagnostic()

    def _on_diag_toggled(self, checked: bool):
        self._diag_plot.setVisible(checked)
        if checked:
            self._refresh_diagnostic()

    def _refresh_diagnostic(self):
        if not self._diag_chk.isChecked():
            return
        em_b_txt = self._em_before.text().strip()
        pw_b_txt = self._pw_before.text().strip()
        if not em_b_txt or not pw_b_txt:
            return
        em_b = Path(em_b_txt)
        pw_b = Path(pw_b_txt)
        if not em_b.exists() or not pw_b.exists():
            return
        em_a_txt = self._em_after.text().strip()
        pw_a_txt = self._pw_after.text().strip()
        em_a = Path(em_a_txt) if em_a_txt else None
        pw_a = Path(pw_a_txt) if pw_a_txt else None
        try:
            fig = plot_led_diagnostic(em_b, em_a, pw_b, pw_a)
            self._diag_plot.set_figure(fig)
        except Exception as exc:
            print(f"[LED diagnostic] {exc}")

    def _mark_stale(self):
        if self._result is not None:
            self._stage3.set_status(STALE)

    # ── Run ────────────────────────────────────────────────────────────────

    def _run(self):
        em_b = Path(self._em_before.text().strip())
        em_a_txt = self._em_after.text().strip()
        em_a = Path(em_a_txt) if em_a_txt else None
        pw_b = Path(self._pw_before.text().strip())
        pw_a_txt = self._pw_after.text().strip()
        pw_a = Path(pw_a_txt) if pw_a_txt else None

        pw_use       = self._pw_combo.currentText()
        mode         = self._mode_combo.currentData()
        threshold    = self._threshold_spin.value()
        std_manual   = self._std_spin.value()
        smoothing    = self._smooth_chk.isChecked()
        sg_win       = self._sg_window.value()
        sg_ord       = self._sg_order.value()

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._stage3.set_status(WAITING)

        self._worker = Worker(
            run_led_characterization,
            emission_before_path=em_b,
            emission_after_path=em_a,
            power_before_path=pw_b,
            power_after_path=pw_a,
            power_use=pw_use,
            emission_threshold=threshold,
            smoothing_enabled=smoothing,
            smoothing_window=sg_win,
            smoothing_order=sg_ord,
            integration_mode=mode,
            photon_flux_std_manual=std_manual,
        )
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, result: LEDResult):
        self._result = result
        fig = plot_led_result(result)
        stem = Path(self._em_before.text()).stem
        self._plot.set_default_filename(f"{stem}_LED_spectrum.png")
        self._plot.set_figure(fig)

        parts = [
            f"N_total = {result.N_mol_s:.4e} mol s⁻¹",
            f"N_std   = {result.N_std_mol_s:.4e} mol s⁻¹",
            f"P_used  = {result.P_used_mW:.4f} mW",
            f"Mode    : {result.integration_mode}",
        ]
        if result.lam_eff is not None:
            parts.append(f"λ_eff   = {result.lam_eff:.1f} nm")
        self._status_lbl.setText("  |  ".join(parts[:3]))
        self._stage3.set_status(DONE)
        self._save_btn.setEnabled(True)
        self.led_result_ready.emit(result)

    def _on_error(self, msg: str):
        self._stage3.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_finished(self):
        self._run_btn.setEnabled(True)

    # ── Save CSV ───────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._result is None:
            return
        r = self._result
        stem = Path(self._em_before.text()).stem
        default_name = f"{stem}_LED_photon_flux.csv"

        if self._output_path:
            out_dir = self._output_path / "led" / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path.home()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save LED spectrum CSV",
            str(out_dir / default_name), "CSV (*.csv)")
        if not path:
            return

        df = pd.DataFrame({
            "wavelength_nm":    r.wl_arr,
            "N_mol_s_per_nm":   r.N_arr,
        })
        # Store scalar metadata as extra columns on row 0
        df["N_total_mol_s"]      = np.nan
        df["N_std_mol_s"]        = np.nan
        df["P_used_mW"]          = np.nan
        df["integration_mode"]   = ""
        df["lam_eff_nm"]         = np.nan
        df.loc[0, "N_total_mol_s"]    = r.N_mol_s
        df.loc[0, "N_std_mol_s"]      = r.N_std_mol_s
        df.loc[0, "P_used_mW"]        = r.P_used_mW
        df.loc[0, "integration_mode"] = r.integration_mode
        if r.lam_eff is not None:
            df.loc[0, "lam_eff_nm"]   = r.lam_eff

        df.to_csv(path, index=False)
        print(f"[LED] Spectrum CSV saved → {path}")
        self.log_signal.emit(f"[LED] Spectrum CSV saved → {path}", "INFO")

    # ── Preferences ────────────────────────────────────────────────────────

    def apply_prefs(self, prefs):
        p = prefs.led_characterisation
        idx = self._pw_combo.findText(p.power_use)
        if idx >= 0:
            self._pw_combo.setCurrentIndex(idx)
        idx2 = self._mode_combo.findData(p.integration_mode)
        if idx2 >= 0:
            self._mode_combo.setCurrentIndex(idx2)
        self._threshold_spin.setValue(p.emission_threshold)
        self._smooth_chk.setChecked(p.smoothing_enabled)
        self._sg_window.setValue(p.smoothing_window)
        self._sg_order.setValue(p.smoothing_order)
        self._std_spin.setValue(p.photon_flux_std_manual)

    def collect_prefs(self, prefs):
        prefs.led_characterisation.power_use              = self._pw_combo.currentText()
        prefs.led_characterisation.integration_mode       = self._mode_combo.currentData()
        prefs.led_characterisation.emission_threshold     = self._threshold_spin.value()
        prefs.led_characterisation.smoothing_enabled      = self._smooth_chk.isChecked()
        prefs.led_characterisation.smoothing_window       = self._sg_window.value()
        prefs.led_characterisation.smoothing_order        = self._sg_order.value()
        prefs.led_characterisation.photon_flux_std_manual = self._std_spin.value()


# ══════════════════════════════════════════════════════════════════════════════
# Sub-panel C — LED Chemical Actinometry (cross-check)
# ══════════════════════════════════════════════════════════════════════════════

class _LEDActinometryPanel(QWidget):
    """
    Use actinometer solutions measured under LED irradiation to compute an
    independent photon flux N_chem and compare it against N_LED from the
    LED Characterisation panel.

    Two integration modes (matching the QY workflow):
      scalar   – λ_eff (flux-weighted centroid) used as single irr. wavelength
      spectral – flux-weighted ε_eff and QY_eff integrated over the full band
    """

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path:  Optional[Path]               = None
        self._raw_path:     Optional[Path]               = None
        self._led_result:   Optional[LEDResult]          = None
        self._results:      list[LEDActinometerResult]   = []
        self._current_idx:  int                          = 0
        self._worker:       Optional[Worker]             = None
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)

        # ── Stage 1 — LED source ───────────────────────────────────────────
        self._stage1 = StageCard("Stage 1 — LED Emission Source")
        self._stage1.add_info_button(
            "LED Emission Source",
            "The LED emission spectrum (N(λ), mol s⁻¹ nm⁻¹) is needed to compute\n"
            "the effective ε and QY for the actinometer under broadband irradiation.\n\n"
            "Sources:\n"
            "  • Automatic — populated when you run the LED Characterisation tab\n"
            "  • CSV — load a previously saved LED spectrum CSV\n\n"
            "The displayed λ_eff is the flux-weighted centroid and is used as the\n"
            "monitoring wavelength in the actinometry rate function regardless of\n"
            "the integration mode chosen in Stage 2."
        )

        hint_led = QLabel(
            "Run the LED Characterisation tab first, or load a saved LED spectrum CSV."
        )
        hint_led.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint_led)

        # Status label
        self._led_status_lbl = QLabel("No LED data loaded.")
        self._led_status_lbl.setStyleSheet("color:#c0392b; font-size:9pt;")
        self._stage1.add_widget(self._led_status_lbl)

        led_btn_row = QHBoxLayout()
        self._load_led_btn = QPushButton("Load LED spectrum CSV…")
        self._load_led_btn.setFixedWidth(180)
        self._load_led_btn.clicked.connect(self._load_led_csv)
        led_btn_row.addWidget(self._load_led_btn)
        led_btn_row.addStretch()
        self._stage1.add_layout(led_btn_row)

        self._stage1.set_status(WAITING)
        layout.addWidget(self._stage1)

        # ── Stage 2 — Actinometry files & parameters ───────────────────────
        self._stage2 = StageCard("Stage 2 — Actinometry Files & Parameters")
        self._stage2.add_info_button(
            "LED Actinometry Parameters",
            "Load CSV files of the actinometer solution recorded while\n"
            "irradiated by the LED.\n\n"
            "Data type:\n"
            "  scanning kinetics — standard Cary 60 multi-scan format\n"
            "  kinetic           — fixed-wavelength time-series format;\n"
            "                     requires an initial scan for Beer-Lambert scaling\n\n"
            "Integration mode:\n"
            "  scalar   — uses λ_eff as a single effective wavelength; ε and QY\n"
            "             are evaluated at λ_eff only (fast, approximate)\n"
            "  spectral — flux-weighted averages over the full LED emission band:\n"
            "             ε_eff = ∫ f(λ)·ε(λ) dλ,  QY_eff = ∫ f(λ)·QY(λ) dλ\n"
            "             where f(λ) = N(λ)/N_total is the normalised LED shape\n"
            "             and ε(λ) is derived via Beer-Lambert scaling.\n\n"
            "λ_irr is NOT entered here — it is determined entirely by the LED\n"
            "emission spectrum loaded in Stage 1."
        )

        file_hint = QLabel("Select CSV file(s) of the actinometer under LED irradiation.")
        file_hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage2.add_widget(file_hint)

        # File list controls
        file_ctrl = QHBoxLayout()
        self._sel_btn = QPushButton("Select files…")
        self._sel_btn.setFixedWidth(120)
        self._sel_btn.clicked.connect(self._select_files)
        self._rem_btn = QPushButton("Remove selected")
        self._rem_btn.setFixedWidth(120)
        self._rem_btn.clicked.connect(self._remove_selected)
        self._clr_btn = QPushButton("Clear all")
        self._clr_btn.setFixedWidth(70)
        self._clr_btn.clicked.connect(self._clear_files)
        file_ctrl.addWidget(self._sel_btn)
        file_ctrl.addWidget(self._rem_btn)
        file_ctrl.addWidget(self._clr_btn)
        file_ctrl.addStretch()
        self._stage2.add_layout(file_ctrl)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(120)
        self._file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self._stage2.add_widget(self._file_list)

        # Data type
        dtype_row = QHBoxLayout()
        dtype_col = QVBoxLayout()
        dtype_col.setSpacing(4)
        _dtype_lbl = QLabel("Data type")
        _dtype_lbl.setObjectName("pref_label")
        _dtype_hdr = QHBoxLayout()
        _dtype_hdr.setSpacing(4)
        _dtype_hdr.addWidget(_dtype_lbl)
        _dtype_hdr.addWidget(InfoButton(
            "Data type",
            "'scanning kinetics' — standard Cary 60 multi-scan format.\n"
            "Each file contains groups of full-spectrum scans; scans within\n"
            "a group are averaged to one absorbance per time point.\n\n"
            "'kinetic' — fixed-wavelength time-series format (Cary 60 kinetic\n"
            "mode). Row 0 has channel labels (e.g. '25C_672nm'); rows 2+ are\n"
            "time/absorbance pairs. An initial full-spectrum scan must be\n"
            "provided for Beer-Lambert scaling and offset correction."))
        _dtype_hdr.addStretch()
        dtype_col.addLayout(_dtype_hdr)
        self._dtype_combo = QComboBox()
        self._dtype_combo.addItem("Scanning kinetics  (Cary 60 scan series)", "scanning")
        self._dtype_combo.addItem("Kinetic  (fixed λ, time series)", "kinetic")
        self._dtype_combo.setMinimumWidth(260)
        self._dtype_combo.currentIndexChanged.connect(self._on_data_type_changed)
        dtype_col.addWidget(self._dtype_combo)
        dtype_row.addLayout(dtype_col)
        dtype_row.addStretch()
        self._stage2.add_layout(dtype_row)

        # Integration mode
        mode_row = QHBoxLayout()
        mode_col = QVBoxLayout()
        mode_col.setSpacing(4)
        _mode_lbl = QLabel("Integration mode")
        _mode_lbl.setObjectName("pref_label")
        _mode_hdr = QHBoxLayout()
        _mode_hdr.setSpacing(4)
        _mode_hdr.addWidget(_mode_lbl)
        _mode_hdr.addWidget(InfoButton(
            "Integration mode",
            "'scalar' — uses λ_eff as a single effective wavelength.\n"
            "Fast and equivalent to standard actinometry at λ_eff.\n\n"
            "'spectral' — integrates ε(λ) and QY(λ) weighted by the LED\n"
            "spectral shape f(λ) across the full emission band. More accurate\n"
            "for broad-band LEDs or when ε and QY vary significantly with λ."))
        _mode_hdr.addStretch()
        mode_col.addLayout(_mode_hdr)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("scalar  (λ_eff, fast)", "scalar")
        self._mode_combo.addItem("spectral  (flux-weighted ε and QY)", "spectral")
        self._mode_combo.setCurrentIndex(1)
        self._mode_combo.setMinimumWidth(240)
        mode_col.addWidget(self._mode_combo)
        mode_row.addLayout(mode_col)
        mode_row.addStretch()
        self._stage2.add_layout(mode_row)

        # Actinometer
        param_row1 = QHBoxLayout()
        act_col = QVBoxLayout()
        act_col.setSpacing(4)
        _act_lbl = QLabel("Actinometer")
        _act_lbl.setObjectName("pref_label")
        _act_hdr = QHBoxLayout()
        _act_hdr.setSpacing(4)
        _act_hdr.addWidget(_act_lbl)
        _act_hdr.addWidget(InfoButton(
            "Actinometer",
            "Select the chemical actinometer solution used in this measurement.\n\n"
            "The actinometer's ε_ref (at its reference wavelength) and QY(λ) formula are used to\n"
            "compute ε(λ) via Beer-Lambert scaling and the effective quantum yield."))
        _act_hdr.addStretch()
        act_col.addLayout(_act_hdr)
        self._act_combo = QComboBox()
        for key, val in ACTINOMETERS.items():
            rng = val["wavelength_range_nm"]
            self._act_combo.addItem(
                f"{key} — {val['name']} ({rng[0]}–{rng[1]} nm)", key)
        self._act_combo.setMinimumWidth(260)
        act_col.addWidget(self._act_combo)
        param_row1.addLayout(act_col)
        param_row1.addStretch()
        self._stage2.add_layout(param_row1)

        # Scanning-only parameters (hidden when kinetic mode selected)
        self._scan_params_widget = QWidget()
        scan_layout = QHBoxLayout(self._scan_params_widget)
        scan_layout.setContentsMargins(0, 4, 0, 0)
        self._irr_time_spin = self._dspin(
            scan_layout, "Irradiation time (s)", 0.1, 10000.0, 60.0, 1.0, pref=True,
            info_title="Irradiation time (s)",
            info_text="Duration of one irradiation interval between consecutive scans (s).")
        self._spg_spin = self._ispin(
            scan_layout, "Scans / group", 1, 50, 3, pref=True,
            info_title="Scans / group",
            info_text="Number of consecutive scans averaged into one time point.")
        self._skip_spin = self._ispin(
            scan_layout, "Skip initial groups", 0, 100, 0, pref=True,
            info_title="Skip initial groups",
            info_text="Number of scan groups to discard from the start of the file.\n"
                      "Use this when irradiation does not begin at the first time point\n"
                      "(e.g. a dark reference group was recorded first).\n"
                      "The time axis is reset to zero at the first kept group.\n"
                      "Note: the Beer-Lambert reference is always taken from the\n"
                      "second scan of the first group, regardless of this setting.")
        scan_layout.addStretch()
        self._stage2.add_widget(self._scan_params_widget)

        # Kinetic-only parameters (hidden until kinetic mode selected)
        self._kin_params_widget = QWidget()
        kin_layout = QVBoxLayout(self._kin_params_widget)
        kin_layout.setContentsMargins(0, 4, 0, 0)
        kin_layout.setSpacing(6)

        init_hdr_row = QHBoxLayout()
        _init_lbl = QLabel("Initial spectrum (Cary 60 scan CSV)")
        _init_lbl.setObjectName("pref_label")
        init_hdr_row.addWidget(_init_lbl)
        init_hdr_row.addWidget(InfoButton(
            "Initial spectrum",
            "A Cary 60 full-spectrum scan of the actinometer before irradiation\n"
            "(or at t=0). Used for:\n"
            "  • Beer-Lambert scaling: ε(λ) = ε_ref × A(λ)/A(λ_ref)\n"
            "  • Offset correction: A_offset = A_initial(λ_eff) − mean(A_kin[0:5])\n\n"
            "Required when 'kinetic' data type is selected."))
        init_hdr_row.addStretch()
        kin_layout.addLayout(init_hdr_row)

        init_file_row = QHBoxLayout()
        self._init_spec_edit = QLineEdit()
        self._init_spec_edit.setPlaceholderText("Select initial spectrum CSV…")
        self._init_spec_edit.setReadOnly(True)
        self._init_spec_edit.textChanged.connect(self._update_run_btn)
        self._init_spec_browse_btn = QPushButton("Browse…")
        self._init_spec_browse_btn.setFixedWidth(80)
        self._init_spec_browse_btn.clicked.connect(self._browse_init_spectrum)
        init_file_row.addWidget(self._init_spec_edit)
        init_file_row.addWidget(self._init_spec_browse_btn)
        kin_layout.addLayout(init_file_row)

        fit_spin_row = QHBoxLayout()
        self._fit_start_spin = self._dspin(
            fit_spin_row, "Fit start (s)", 0.0, 100000.0, 0.0, 10.0, decimals=1, pref=True,
            info_title="Fit start (s)",
            info_text="Kinetic data before this time is excluded from the linear fit.\n"
                      "Set to 0 to start from the beginning of the time series.")
        self._fit_end_spin = self._dspin(
            fit_spin_row, "Fit end (s)", 0.0, 100000.0, 0.0, 10.0, decimals=1, pref=True,
            info_title="Fit end (s)",
            info_text="Kinetic data after this time is excluded from the linear fit.\n"
                      "Set to 0 to use all data up to the end of the time series.")
        fit_spin_row.addStretch()
        kin_layout.addLayout(fit_spin_row)

        self._kin_params_widget.setVisible(False)
        self._stage2.add_widget(self._kin_params_widget)

        # Common parameters
        param_row2 = QHBoxLayout()
        self._volume_spin   = self._dspin(
            param_row2, "Volume (mL)", 0.01, 100.0, 2.0, 0.1, pref=True,
            info_title="Volume (mL)",
            info_text="Volume of actinometer solution in the cuvette (mL).")
        self._path_spin     = self._dspin(
            param_row2, "Path length (cm)", 0.001, 100.0, 1.0, 0.1, pref=True,
            info_title="Path length (cm)",
            info_text="Optical path length of the cuvette (cm).")
        self._tol_spin      = self._dspin(
            param_row2, "λ tolerance (nm)", 0.1, 10.0, 1.0, 0.5, pref=True,
            info_title="λ tolerance (nm)",
            info_text="Match window around λ_eff for absorbance extraction (nm).")
        param_row2.addStretch()
        self._stage2.add_layout(param_row2)

        for sig in (
            self._dtype_combo.currentIndexChanged,
            self._mode_combo.currentIndexChanged,
            self._act_combo.currentIndexChanged,
            self._irr_time_spin.valueChanged,
            self._volume_spin.valueChanged,
            self._path_spin.valueChanged,
            self._spg_spin.valueChanged,
            self._skip_spin.valueChanged,
            self._tol_spin.valueChanged,
            self._fit_start_spin.valueChanged,
            self._fit_end_spin.valueChanged,
        ):
            sig.connect(self._mark_stale)

        self._stage2.set_status(WAITING)
        layout.addWidget(self._stage2)

        # ── Stage 3 — Run & Results ────────────────────────────────────────
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(
            "LED Actinometry Results",
            "Left panel: rate function vs irradiation time with linear fit.\n"
            "The slope gives N_chem (mol s⁻¹).\n\n"
            "Right panel (when LED N is available): bar chart comparing N_chem\n"
            "and N_LED with error bars. The deviation (%) is shown in the title.\n\n"
            "The annotation box lists the effective ε and QY used — compare\n"
            "these between scalar and spectral modes to assess the approximation."
        )

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setFixedWidth(100)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:9pt;")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._status_lbl)
        run_row.addStretch()
        self._stage3.add_layout(run_row)

        # Navigation
        self._nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setFixedWidth(80)
        self._prev_btn.clicked.connect(self._show_prev)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setFixedWidth(80)
        self._next_btn.clicked.connect(self._show_next)
        self._nav_lbl = QLabel("")
        self._nav_row.addWidget(self._prev_btn)
        self._nav_row.addWidget(self._nav_lbl)
        self._nav_row.addWidget(self._next_btn)
        self._nav_row.addStretch()
        nav_widget = QWidget()
        nav_widget.setLayout(self._nav_row)
        nav_widget.setVisible(False)
        self._nav_widget = nav_widget
        self._stage3.add_widget(nav_widget)

        self._plot = PlotWidget(
            info_title="LED Actinometry",
            info_text=(
                "Left: rate function vs time — slope = N_chem (mol s⁻¹).\n"
                "Right: comparison bar chart N_chem vs N_LED."
            ),
            min_height=350,
        )
        self._stage3.add_widget(self._plot)

        # Results table
        self._res_table = QTableWidget(0, 13)
        self._res_table.setHorizontalHeaderLabels([
            "File", "Actinometer", "Mode", "λ_eff (nm)",
            "ε_eff (M⁻¹cm⁻¹)", "QY_eff",
            "N_chem (mol s⁻¹)", "N_std (mol s⁻¹)", "R²", "Intercept (mol)",
            "N_LED (mol s⁻¹)", "N_LED std (mol s⁻¹)", "Deviation (%)",
        ])
        rh = self._res_table.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 13):
            rh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._res_table.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self._res_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._res_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._stage3.add_widget(self._res_table)

        self._stage3.set_status(WAITING)
        layout.addWidget(self._stage3)
        layout.addStretch()

    # ── Spin box helpers (same pattern as the other panels) ─────────────────

    def _dspin(self, parent, label, lo, hi, val, step, decimals=3, pref=False,
               info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setMinimumWidth(110)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    def _ispin(self, parent, label, lo, hi, val, pref=False,
               info_title="", info_text=""):
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        if info_text:
            hdr = QHBoxLayout()
            hdr.setSpacing(4)
            hdr.addWidget(lbl)
            hdr.addWidget(InfoButton(info_title, info_text))
            hdr.addStretch()
            col.addLayout(hdr)
        else:
            col.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setMinimumWidth(80)
        col.addWidget(spin)
        parent.addLayout(col)
        return spin

    # ── Public slot: receive live LED result from LED Characterisation panel ─

    def set_led_source(self, result: LEDResult):
        """Called by ActinometerTab when the LED Characterisation panel finishes."""
        self._led_result = result
        # Compute λ_eff if not stored (full-mode LED result may have lam_eff=None)
        if result.lam_eff is not None:
            lam_eff = result.lam_eff
        else:
            lam_eff = float(
                np.trapezoid(result.wl_arr * result.N_arr, result.wl_arr)
                / result.N_mol_s
            )
        self._led_status_lbl.setText(
            f"Live result — λ_eff = {lam_eff:.1f} nm  |  "
            f"N_LED = {result.N_mol_s:.4e} mol s⁻¹  |  "
            f"range {result.wl_arr[0]:.0f}–{result.wl_arr[-1]:.0f} nm"
        )
        self._led_status_lbl.setStyleSheet("color:#27ae60; font-size:9pt;")
        self._stage1.set_status(READY)
        self._update_run_btn()
        self._mark_stale()

    # ── Public paths ────────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        plots_dir = path / "actinometer" / "led_actinometry" / "plots"
        self._plot.set_save_dir(plots_dir)

    def set_raw_path(self, path: Path):
        self._raw_path = path

    # ── LED CSV loader ──────────────────────────────────────────────────────

    def _load_led_csv(self):
        start = str(self._raw_path or self._output_path or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load LED spectrum CSV", start, "CSV files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            wl_arr = df["wavelength_nm"].values.astype(float)
            N_arr  = df["N_mol_s_per_nm"].values.astype(float)
            N_total = float(df.loc[0, "N_total_mol_s"])
            N_std   = float(df.loc[0, "N_std_mol_s"]) \
                      if "N_std_mol_s" in df.columns else 0.0
            lam_eff_val = df.loc[0, "lam_eff_nm"] \
                          if "lam_eff_nm" in df.columns else float("nan")
            if np.isnan(lam_eff_val):
                lam_eff_val = float(
                    np.trapezoid(wl_arr * N_arr, wl_arr) / N_total)

            # Construct a LEDResult with dummy power fields so the rest of
            # the code (which only reads wl_arr, N_arr, N_mol_s, etc.) works.
            dummy = np.array([0.0])
            self._led_result = LEDResult(
                wl_arr=wl_arr,
                N_arr=N_arr,
                N_mol_s=N_total,
                N_std_mol_s=N_std,
                lam_eff=lam_eff_val,
                integration_mode="full",
                P_before_mW=0.0,
                P_after_mW=None,
                P_used_mW=0.0,
                wl_full=wl_arr,
                N_arr_full=N_arr,
                em_int_pre_smooth=np.ones_like(wl_arr),
                em_int_raw_b=np.ones_like(wl_arr),
                em_int_raw_a=None,
                pwr_time_b=dummy,
                pwr_vals_b=dummy,
                pwr_time_a=None,
                pwr_vals_a=None,
            )
            self._led_status_lbl.setText(
                f"CSV: {Path(path).name}  |  "
                f"λ_eff = {lam_eff_val:.1f} nm  |  "
                f"N_LED = {N_total:.4e} mol s⁻¹"
            )
            self._led_status_lbl.setStyleSheet("color:#27ae60; font-size:9pt;")
            self._stage1.set_status(READY)
            self._update_run_btn()
            self._mark_stale()
        except Exception as exc:
            self._led_status_lbl.setText(f"Error loading CSV: {exc}")
            self._led_status_lbl.setStyleSheet("color:#c0392b; font-size:9pt;")

    # ── Data type toggle ────────────────────────────────────────────────────

    def _on_data_type_changed(self):
        is_kinetic = self._dtype_combo.currentData() == "kinetic"
        self._scan_params_widget.setVisible(not is_kinetic)
        self._kin_params_widget.setVisible(is_kinetic)
        self._update_run_btn()
        self._mark_stale()

    def _browse_init_spectrum(self):
        start = str(self._raw_path or self._output_path or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select initial spectrum CSV", start, "CSV files (*.csv)")
        if path:
            self._init_spec_edit.setText(path)

    # ── File list ───────────────────────────────────────────────────────────

    def _select_files(self):
        start = str(self._raw_path or self._output_path or Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select actinometry CSV files", start, "CSV files (*.csv)")
        existing = {self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self._file_list.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.ItemDataRole.UserRole, p)
                self._file_list.addItem(item)
        self._update_run_btn()

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))
        self._update_run_btn()
        self._mark_stale()

    def _clear_files(self):
        self._file_list.clear()
        self._update_run_btn()
        self._mark_stale()

    def _update_run_btn(self):
        if self._dtype_combo.currentData() == "kinetic":
            has_init_spec = bool(self._init_spec_edit.text().strip())
        else:
            has_init_spec = True
        ready = (self._led_result is not None
                 and self._file_list.count() > 0
                 and has_init_spec)
        self._run_btn.setEnabled(ready)
        if self._file_list.count() > 0:
            self._stage2.set_status(READY)
        else:
            self._stage2.set_status(WAITING)

    def _mark_stale(self):
        if self._results:
            self._stage3.set_status(STALE)

    # ── Run ─────────────────────────────────────────────────────────────────

    def _run(self):
        if self._led_result is None or self._file_list.count() == 0:
            return

        r = self._led_result
        # Resolve λ_eff
        if r.lam_eff is not None:
            lam_eff = r.lam_eff
        else:
            lam_eff = float(
                np.trapezoid(r.wl_arr * r.N_arr, r.wl_arr) / r.N_mol_s)

        files        = [Path(self._file_list.item(i).data(Qt.ItemDataRole.UserRole))
                        for i in range(self._file_list.count())]
        act_key      = self._act_combo.currentData()
        mode         = self._mode_combo.currentData()
        data_type    = self._dtype_combo.currentData()
        irr_time     = self._irr_time_spin.value()
        volume       = self._volume_spin.value()
        path_len     = self._path_spin.value()
        spg          = self._spg_spin.value()
        skip         = self._skip_spin.value()
        tol          = self._tol_spin.value()
        init_spec    = (Path(self._init_spec_edit.text().strip())
                        if data_type == "kinetic" and self._init_spec_edit.text().strip()
                        else None)
        fit_start    = self._fit_start_spin.value() or None   # 0.0 → None
        fit_end      = self._fit_end_spin.value() or None     # 0.0 → None
        N_LED        = r.N_mol_s
        N_LED_std    = r.N_std_mol_s

        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._stage3.set_status(WAITING)
        self._results = []

        led_wl  = r.wl_arr.copy()
        led_N   = r.N_arr.copy()

        def _run_all():
            results = []
            for fp in files:
                print(f"\n{'='*50}\nLED actinometry: {fp.name}")
                res = run_led_actinometry_file(
                    filepath=fp,
                    actinometer_choice=act_key,
                    led_wl_arr=led_wl,
                    led_N_arr=led_N,
                    lam_eff=lam_eff,
                    integration_mode=mode,
                    data_type=data_type,
                    irradiation_time_s=irr_time,
                    volume_mL=volume,
                    path_length_cm=path_len,
                    scans_per_group=spg,
                    wavelength_tolerance_nm=tol,
                    skip_groups=skip,
                    initial_spectrum_path=init_spec,
                    fit_time_start_s=fit_start,
                    fit_time_end_s=fit_end,
                    N_LED_mol_s=N_LED,
                    N_LED_std_mol_s=N_LED_std,
                )
                results.append(res)
            return results

        self._worker = Worker(_run_all)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, results: list):
        self._results = results
        self._current_idx = 0
        self._populate_table()
        self._show_result(0)
        self._nav_widget.setVisible(len(results) > 1)
        self._stage3.set_status(DONE)
        self._status_lbl.setText(f"Done — {len(results)} file(s) processed.")

    def _on_error(self, msg: str):
        self._stage3.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_finished(self):
        self._run_btn.setEnabled(
            self._led_result is not None and self._file_list.count() > 0)

    # ── Results display ──────────────────────────────────────────────────────

    def _populate_table(self):
        self._res_table.setRowCount(0)
        for r in self._results:
            row = self._res_table.rowCount()
            self._res_table.insertRow(row)
            if r.N_LED_mol_s is not None:
                n_led_str     = f"{r.N_LED_mol_s:.4e}"
                n_led_std_str = (f"{r.N_LED_std_mol_s:.4e}"
                                 if r.N_LED_std_mol_s is not None else "—")
                dev = (r.photon_flux_mol_s - r.N_LED_mol_s) / r.N_LED_mol_s * 100.0
                dev_str = f"{dev:+.2f}"
            else:
                n_led_str = n_led_std_str = dev_str = "—"
            vals = [
                r.file,
                r.actinometer_name,
                r.integration_mode,
                f"{r.lam_eff:.1f}",
                f"{r.epsilon_eff_M_cm:.4e}",
                f"{r.QY_eff:.4f}",
                f"{r.photon_flux_mol_s:.4e}",
                f"{r.photon_flux_std_mol_s:.4e}",
                f"{r.r2:.4f}",
                f"{r.intercept:.4e}",
                n_led_str,
                n_led_std_str,
                dev_str,
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._res_table.setItem(row, col, item)
        self._res_table.resizeRowsToContents()

    def _show_result(self, idx: int):
        if not self._results:
            return
        r = self._results[idx]
        fig = plot_led_actinometry_result(r)
        self._plot.set_default_filename(
            f"{Path(r.file).stem}_{r.actinometer_name.replace(' ','_')}"
            f"_{r.integration_mode}.png")
        self._plot.set_figure(fig)
        self._nav_lbl.setText(f"  {idx + 1} / {len(self._results)}  ")

    def _show_prev(self):
        if self._results:
            self._current_idx = max(0, self._current_idx - 1)
            self._show_result(self._current_idx)

    def _show_next(self):
        if self._results:
            self._current_idx = min(
                len(self._results) - 1, self._current_idx + 1)
            self._show_result(self._current_idx)


# ══════════════════════════════════════════════════════════════════════════════
# Top-level ActinometerTab
# ══════════════════════════════════════════════════════════════════════════════

class ActinometerTab(QWidget):
    """Actinometer tab: Chemical Actinometry + LED Characterisation + LED Actinometry."""

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        inner = QTabWidget()
        inner.setObjectName("inner_tabs")

        self._chem_panel      = _ChemActinometerPanel()
        self._led_panel       = _LEDCharacterisationPanel()
        self._led_act_panel   = _LEDActinometryPanel()
        inner.addTab(self._chem_panel,    "Chemical Actinometry")
        inner.addTab(self._led_panel,     "LED Characterisation")
        inner.addTab(self._led_act_panel, "LED Actinometry")
        layout.addWidget(inner)

        self._chem_panel.log_signal.connect(self.log_signal)
        self._led_panel.log_signal.connect(self.log_signal)
        self._led_act_panel.log_signal.connect(self.log_signal)

        # Forward LED result to the LED Actinometry panel automatically
        self._led_panel.led_result_ready.connect(
            self._led_act_panel.set_led_source)

    # ── Public slots ───────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._chem_panel.set_output_path(path)
        self._led_panel.set_output_path(path)
        self._led_act_panel.set_output_path(path)

    def set_raw_path(self, path: Path):
        self._chem_panel.set_raw_path(path)
        self._led_panel.set_raw_path(path)
        self._led_act_panel.set_raw_path(path)

    def apply_prefs(self, prefs):
        self._chem_panel.apply_prefs(prefs)
        self._led_panel.apply_prefs(prefs)

    def collect_prefs(self, prefs):
        self._chem_panel.collect_prefs(prefs)
        self._led_panel.collect_prefs(prefs)
