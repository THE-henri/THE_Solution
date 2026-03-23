"""
Quantum Yield tab — wraps qy_core.py in a 4-stage PyQt6 GUI.

Stages:
  1. Input files, data type, and photon flux source
  2. Experimental parameters (sample, baseline, fit window, k_th)
  3. Extinction coefficients
  4. Fitting parameters, run, and results
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QListWidget, QListWidgetItem,
    QFileDialog, QAbstractItemView, QFrame,
    QGroupBox,
)

from gui.tabs.qy_core import (
    QYParams, QYFileResult,
    load_photon_flux, load_epsilon_at_wavelength, load_epsilon_at_wavelengths,
    load_k_th, run_qy_file, plot_qy_result,
)
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.worker import Worker


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hbox(*widgets, spacing=6):
    row = QHBoxLayout()
    row.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QWidget):
            row.addWidget(w)
        elif w == "stretch":
            row.addStretch()
        else:
            row.addSpacing(w)
    return row


def _label(text, *, color=None, italic=False):
    lbl = QLabel(text)
    css = ""
    if color:
        css += f"color:{color};"
    if italic:
        css += "font-style:italic;"
    if css:
        lbl.setStyleSheet(css)
    return lbl


def _field_row(label_text, widget, width=None, pref=False):
    """Return an HBoxLayout with a fixed-width label + widget."""
    lbl = QLabel(label_text)
    lbl.setFixedWidth(200)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if pref:
        lbl.setObjectName("pref_label")
    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(lbl)
    if width is not None and isinstance(widget, QWidget):
        widget.setFixedWidth(width)
    row.addWidget(widget)
    row.addStretch()
    return row


def _browse_row(label_text, line_edit, button):
    row = _field_row(label_text, line_edit, width=320)
    row.insertWidget(row.count() - 1, button)
    return row


# ══════════════════════════════════════════════════════════════════════════════
# QuantumYieldTab
# ══════════════════════════════════════════════════════════════════════════════

class QuantumYieldTab(QWidget):
    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path: Optional[Path] = None
        self._results:     list[QYFileResult] = []
        self._current_idx: int = 0
        self._worker:      Optional[Worker] = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

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

        self._build_stage1(layout)
        self._build_stage2(layout)
        self._build_stage3(layout)
        self._build_stage4(layout)
        layout.addStretch()

    # ── Stage 1 — Input Files & Photon Flux ──────────────────────────────────

    def _build_stage1(self, parent_layout):
        self._stage1 = StageCard("Stage 1 — Input Files & Photon Flux")
        self._stage1.add_info_button(
            "Input Files & Photon Flux",
            "Data files: one or more CSV files from the kinetic or Cary 60 "
            "scanning experiment. For scanning mode, set Δt and scans/group "
            "to convert scan index to elapsed time.\n\n"
            "Photon flux source:\n"
            "  manual mol/s — enter N and its uncertainty directly.\n"
            "  manual µW — enter optical power; the tool converts to mol/s "
            "using the irradiation wavelength.\n"
            "  actinometry CSV — load photon_flux_master.csv produced by "
            "the Actinometer tab.\n"
            "  LED spectrum CSV — load the spectral N(λ) file from the "
            "LED Characterisation panel for a full spectral calculation."
        )

        # ── Data files ────────────────────────────────────────────────────────
        hint = _label(
            "Select one or more QY data CSV files (kinetic or Cary 60 scanning).",
            color="#888")
        self._stage1.add_widget(hint)

        ctrl = QHBoxLayout()
        self._sel_btn = QPushButton("Select files…")
        self._sel_btn.setFixedWidth(120)
        self._sel_btn.clicked.connect(self._select_files)
        self._clr_btn = QPushButton("Clear")
        self._clr_btn.setFixedWidth(60)
        self._clr_btn.clicked.connect(self._clear_files)
        self._file_count_lbl = _label("No files selected.", color="#888")
        ctrl.addWidget(self._sel_btn)
        ctrl.addWidget(self._clr_btn)
        ctrl.addWidget(self._file_count_lbl)
        ctrl.addStretch()
        self._stage1.add_layout(ctrl)

        self._file_list = QListWidget()
        self._file_list.setFixedHeight(90)
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._stage1.add_widget(self._file_list)

        # ── Data type ────────────────────────────────────────────────────────
        self._data_type_combo = QComboBox()
        self._data_type_combo.addItems(["kinetic", "scanning"])
        self._stage1.add_layout(_field_row("Data type:", self._data_type_combo, 120, pref=True))
        self._data_type_combo.currentTextChanged.connect(self._on_data_type_changed)

        # ── Scanning-specific ─────────────────────────────────────────────────
        self._scan_grp = QGroupBox("Scanning-specific parameters")
        sg = QVBoxLayout(self._scan_grp)
        sg.setSpacing(4)
        self._delta_t_spin = QDoubleSpinBox()
        self._delta_t_spin.setRange(0.01, 1e6)
        self._delta_t_spin.setDecimals(3)
        self._delta_t_spin.setValue(12.0)
        self._delta_t_spin.setSuffix(" s")
        sg.addLayout(_field_row("Δt per group (s):", self._delta_t_spin, 100, pref=True))

        self._scans_per_grp_spin = QSpinBox()
        self._scans_per_grp_spin.setRange(1, 100)
        self._scans_per_grp_spin.setValue(1)
        sg.addLayout(_field_row("Scans per group:", self._scans_per_grp_spin, 80, pref=True))

        self._first_cycle_off_chk = QCheckBox("Skip first scan group")
        sg.addWidget(self._first_cycle_off_chk)

        self._stage1.add_widget(self._scan_grp)
        self._scan_grp.setVisible(False)

        # ── Photon flux source ────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;")
        self._stage1.add_widget(sep)

        self._irr_wl_spin = QDoubleSpinBox()
        self._irr_wl_spin.setRange(200, 1100)
        self._irr_wl_spin.setDecimals(1)
        self._irr_wl_spin.setValue(530.0)
        self._irr_wl_spin.setSuffix(" nm")
        self._stage1.add_layout(_field_row(
            "Irradiation λ:", self._irr_wl_spin, 120, pref=True))

        self._flux_src_combo = QComboBox()
        self._flux_src_combo.addItems([
            "manual_mol_s", "manual_uW", "actinometry", "led_spectrum"])
        self._stage1.add_layout(_field_row(
            "Photon flux source:", self._flux_src_combo, 140, pref=True))
        self._flux_src_combo.currentTextChanged.connect(self._on_flux_src_changed)

        # manual mol/s
        self._flux_manual_grp = QGroupBox("Manual photon flux (mol s⁻¹)")
        mg = QVBoxLayout(self._flux_manual_grp)
        self._flux_mol_s_spin = QDoubleSpinBox()
        self._flux_mol_s_spin.setRange(0, 1)
        self._flux_mol_s_spin.setDecimals(6)
        self._flux_mol_s_spin.setSingleStep(1e-10)
        self._flux_mol_s_spin.setValue(1e-9)
        mg.addLayout(_field_row("N (mol s⁻¹):", self._flux_mol_s_spin, 120, pref=True))
        self._flux_std_spin = QDoubleSpinBox()
        self._flux_std_spin.setRange(0, 1)
        self._flux_std_spin.setDecimals(6)
        self._flux_std_spin.setSingleStep(1e-11)
        self._flux_std_spin.setValue(0.0)
        mg.addLayout(_field_row("N std (mol s⁻¹):", self._flux_std_spin, 120, pref=True))
        self._stage1.add_widget(self._flux_manual_grp)

        # manual µW
        self._flux_uw_grp = QGroupBox("Manual photon flux (µW)")
        ug = QVBoxLayout(self._flux_uw_grp)
        self._flux_uw_spin = QDoubleSpinBox()
        self._flux_uw_spin.setRange(0, 1e6)
        self._flux_uw_spin.setDecimals(4)
        self._flux_uw_spin.setValue(0.0)
        self._flux_uw_spin.setSuffix(" µW")
        ug.addLayout(_field_row("Power:", self._flux_uw_spin, 120))
        self._flux_uw_std_spin = QDoubleSpinBox()
        self._flux_uw_std_spin.setRange(0, 1)
        self._flux_uw_std_spin.setDecimals(6)
        self._flux_uw_std_spin.setValue(0.0)
        ug.addLayout(_field_row("N std (mol s⁻¹):", self._flux_uw_std_spin, 120))
        self._stage1.add_widget(self._flux_uw_grp)

        # actinometry CSV
        self._flux_actin_grp = QGroupBox("Actinometry CSV")
        ag = QVBoxLayout(self._flux_actin_grp)
        self._actin_csv_edit = QLineEdit()
        self._actin_csv_edit.setPlaceholderText("photon_flux_master.csv …")
        self._actin_csv_browse = QPushButton("Browse…")
        self._actin_csv_browse.setFixedWidth(80)
        self._actin_csv_browse.clicked.connect(self._browse_actin_csv)
        ag.addLayout(_browse_row("CSV file:", self._actin_csv_edit,
                                  self._actin_csv_browse))
        self._actin_filter_spin = QDoubleSpinBox()
        self._actin_filter_spin.setRange(0, 1100)
        self._actin_filter_spin.setDecimals(1)
        self._actin_filter_spin.setValue(0.0)
        self._actin_filter_spin.setSpecialValueText("(auto from irr λ)")
        ag.addLayout(_field_row("Filter λ (nm):", self._actin_filter_spin, 100))
        self._stage1.add_widget(self._flux_actin_grp)

        # LED spectrum CSV
        self._flux_led_grp = QGroupBox("LED spectrum CSV")
        lg = QVBoxLayout(self._flux_led_grp)
        self._led_csv_edit = QLineEdit()
        self._led_csv_edit.setPlaceholderText("led_spectrum_…csv")
        self._led_csv_browse = QPushButton("Browse…")
        self._led_csv_browse.setFixedWidth(80)
        self._led_csv_browse.clicked.connect(self._browse_led_csv)
        lg.addLayout(_browse_row("CSV file:", self._led_csv_edit,
                                  self._led_csv_browse))
        self._led_integ_combo = QComboBox()
        self._led_integ_combo.addItems(["scalar", "full"])
        lg.addLayout(_field_row("Integration mode:", self._led_integ_combo, 80))
        self._stage1.add_widget(self._flux_led_grp)

        parent_layout.addWidget(self._stage1)
        self._on_flux_src_changed("manual_mol_s")

    # ── Stage 2 — Experimental Parameters ────────────────────────────────────

    def _build_stage2(self, parent_layout):
        self._stage2 = StageCard("Stage 2 — Experimental Parameters")
        self._stage2.add_info_button(
            "Experimental Parameters",
            "Case:\n"
            "  A_only — only species A is converted (irreversible reaction or "
            "first irradiation direction). Fit yields Φ_AB.\n"
            "  AB_both — both forward (A→B) and backward (B→A) reactions occur; "
            "fit yields Φ_AB and Φ_BA simultaneously.\n"
            "  A_thermal_PSS — includes thermal relaxation of B; k_th must be "
            "supplied.\n\n"
            "Monitoring wavelengths: absorption wavelengths where the kinetics "
            "are tracked. Multiple wavelengths are fitted jointly.\n\n"
            "Baseline correction: subtract the absorbance at a reference "
            "wavelength from all spectra to remove drift."
        )

        # Case & identifiers
        self._case_combo = QComboBox()
        self._case_combo.addItems(["A_only", "AB_both", "A_thermal_PSS"])
        self._stage2.add_layout(_field_row("Case:", self._case_combo, 140, pref=True))

        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(-100, 200)
        self._temp_spin.setDecimals(1)
        self._temp_spin.setValue(25.0)
        self._temp_spin.setSuffix(" °C")
        self._stage2.add_layout(_field_row("Temperature:", self._temp_spin, 100, pref=True))

        self._solvent_edit = QLineEdit()
        self._solvent_edit.setPlaceholderText("e.g. acetonitrile")
        self._stage2.add_layout(_field_row("Solvent:", self._solvent_edit, 160, pref=True))

        # Optical
        self._path_spin = QDoubleSpinBox()
        self._path_spin.setRange(0.001, 100)
        self._path_spin.setDecimals(4)
        self._path_spin.setValue(1.0)
        self._path_spin.setSuffix(" cm")
        self._stage2.add_layout(_field_row("Path length:", self._path_spin, 100, pref=True))

        self._vol_spin = QDoubleSpinBox()
        self._vol_spin.setRange(0.001, 1000)
        self._vol_spin.setDecimals(3)
        self._vol_spin.setValue(2.0)
        self._vol_spin.setSuffix(" mL")
        self._stage2.add_layout(_field_row("Volume:", self._vol_spin, 100, pref=True))

        # Monitoring wavelengths
        self._mon_wl_edit = QLineEdit()
        self._mon_wl_edit.setPlaceholderText("e.g. 580, 620 — blank = auto from kinetic headers")
        self._stage2.add_layout(_field_row("Monitoring λ (nm):", self._mon_wl_edit, 280, pref=True))

        self._wl_tol_spin = QDoubleSpinBox()
        self._wl_tol_spin.setRange(0.1, 20)
        self._wl_tol_spin.setDecimals(1)
        self._wl_tol_spin.setValue(2.0)
        self._wl_tol_spin.setSuffix(" nm")
        self._stage2.add_layout(_field_row("Wavelength tolerance:", self._wl_tol_spin, 100, pref=True))

        # Baseline correction
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;"); self._stage2.add_widget(sep)

        self._baseline_combo = QComboBox()
        self._baseline_combo.addItems(["first_point", "none", "plateau", "file"])
        self._stage2.add_layout(
            _field_row("Offset correction:", self._baseline_combo, 120))
        self._baseline_combo.currentTextChanged.connect(self._on_baseline_changed)

        self._baseline_plat_grp = QGroupBox("Plateau offset")
        bpg = QVBoxLayout(self._baseline_plat_grp)
        self._plat_dur_spin = QDoubleSpinBox()
        self._plat_dur_spin.setRange(0, 1e5)
        self._plat_dur_spin.setDecimals(1)
        self._plat_dur_spin.setValue(20.0)
        self._plat_dur_spin.setSuffix(" s")
        bpg.addLayout(_field_row("Plateau duration:", self._plat_dur_spin, 100))
        self._plat_start_spin = QDoubleSpinBox()
        self._plat_start_spin.setRange(0, 1e6)
        self._plat_start_spin.setDecimals(1)
        self._plat_start_spin.setValue(0.0)
        self._plat_start_spin.setSpecialValueText("(auto)")
        bpg.addLayout(_field_row("Start (s):", self._plat_start_spin, 100))
        self._plat_end_spin = QDoubleSpinBox()
        self._plat_end_spin.setRange(0, 1e6)
        self._plat_end_spin.setDecimals(1)
        self._plat_end_spin.setValue(0.0)
        self._plat_end_spin.setSpecialValueText("(auto)")
        bpg.addLayout(_field_row("End (s):", self._plat_end_spin, 100))
        self._stage2.add_widget(self._baseline_plat_grp)
        self._baseline_plat_grp.setVisible(False)

        self._baseline_file_grp = QGroupBox("Offset file")
        bfg = QVBoxLayout(self._baseline_file_grp)
        self._baseline_file_edit = QLineEdit()
        self._baseline_file_edit.setPlaceholderText("baseline CSV …")
        self._baseline_file_browse = QPushButton("Browse…")
        self._baseline_file_browse.setFixedWidth(80)
        self._baseline_file_browse.clicked.connect(self._browse_baseline_file)
        bfg.addLayout(_browse_row("CSV:", self._baseline_file_edit,
                                   self._baseline_file_browse))
        self._stage2.add_widget(self._baseline_file_grp)
        self._baseline_file_grp.setVisible(False)

        # Fit window
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#555;"); self._stage2.add_widget(sep2)

        self._auto_detect_chk = QCheckBox("Auto-detect irradiation start")
        self._auto_detect_chk.setChecked(True)
        self._stage2.add_widget(self._auto_detect_chk)
        self._auto_detect_chk.toggled.connect(self._on_autodetect_toggled)

        self._auto_detect_grp = QGroupBox("Auto-detect parameters")
        adg = QVBoxLayout(self._auto_detect_grp)
        self._n_plat_spin = QSpinBox()
        self._n_plat_spin.setRange(3, 200)
        self._n_plat_spin.setValue(20)
        adg.addLayout(_field_row("Plateau points:", self._n_plat_spin, 80))
        self._detect_thresh_spin = QDoubleSpinBox()
        self._detect_thresh_spin.setRange(0.5, 100)
        self._detect_thresh_spin.setDecimals(1)
        self._detect_thresh_spin.setValue(5.0)
        adg.addLayout(_field_row("Threshold (σ):", self._detect_thresh_spin, 80))
        self._min_consec_spin = QSpinBox()
        self._min_consec_spin.setRange(1, 20)
        self._min_consec_spin.setValue(3)
        adg.addLayout(_field_row("Min consecutive:", self._min_consec_spin, 80))
        self._stage2.add_widget(self._auto_detect_grp)

        self._manual_window_grp = QGroupBox("Manual fit window")
        mwg = QVBoxLayout(self._manual_window_grp)
        self._fit_start_spin = QDoubleSpinBox()
        self._fit_start_spin.setRange(0, 1e6)
        self._fit_start_spin.setDecimals(1)
        self._fit_start_spin.setValue(0.0)
        self._fit_start_spin.setSpecialValueText("(auto)")
        mwg.addLayout(_field_row("Fit start (s):", self._fit_start_spin, 100))
        self._fit_end_spin = QDoubleSpinBox()
        self._fit_end_spin.setRange(0, 1e6)
        self._fit_end_spin.setDecimals(1)
        self._fit_end_spin.setValue(0.0)
        self._fit_end_spin.setSpecialValueText("(auto)")
        mwg.addLayout(_field_row("Fit end (s):", self._fit_end_spin, 100))
        self._stage2.add_widget(self._manual_window_grp)
        self._manual_window_grp.setVisible(False)

        # k_th
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("color:#555;"); self._stage2.add_widget(sep3)

        self._kth_src_combo = QComboBox()
        self._kth_src_combo.addItems(
            ["none", "manual", "half_life_master", "eyring", "arrhenius"])
        self._stage2.add_layout(
            _field_row("k_th source:", self._kth_src_combo, 140, pref=True))
        self._kth_src_combo.currentTextChanged.connect(self._on_kth_src_changed)

        self._kth_manual_grp = QGroupBox("Manual k_th")
        kmg = QVBoxLayout(self._kth_manual_grp)
        self._kth_manual_spin = QDoubleSpinBox()
        self._kth_manual_spin.setRange(0, 1e6)
        self._kth_manual_spin.setDecimals(8)
        self._kth_manual_spin.setSingleStep(1e-6)
        self._kth_manual_spin.setValue(0.0)
        self._kth_manual_spin.setSuffix(" s⁻¹")
        kmg.addLayout(_field_row("k_th:", self._kth_manual_spin, 140))
        self._kth_manual_std_spin = QDoubleSpinBox()
        self._kth_manual_std_spin.setRange(0, 1e6)
        self._kth_manual_std_spin.setDecimals(8)
        self._kth_manual_std_spin.setValue(0.0)
        self._kth_manual_std_spin.setSuffix(" s⁻¹")
        kmg.addLayout(_field_row("k_th std:", self._kth_manual_std_spin, 140))
        self._stage2.add_widget(self._kth_manual_grp)
        self._kth_manual_grp.setVisible(False)

        self._kth_csv_grp = QGroupBox("k_th from file")
        kcg = QVBoxLayout(self._kth_csv_grp)
        self._kth_csv_edit = QLineEdit()
        self._kth_csv_edit.setPlaceholderText("half_life_master.csv / eyring_results.csv …")
        self._kth_csv_browse = QPushButton("Browse…")
        self._kth_csv_browse.setFixedWidth(80)
        self._kth_csv_browse.clicked.connect(self._browse_kth_csv)
        kcg.addLayout(_browse_row("CSV:", self._kth_csv_edit,
                                   self._kth_csv_browse))
        self._kth_temp_spin = QDoubleSpinBox()
        self._kth_temp_spin.setRange(-100, 200)
        self._kth_temp_spin.setDecimals(1)
        self._kth_temp_spin.setValue(25.0)
        self._kth_temp_spin.setSuffix(" °C")
        kcg.addLayout(_field_row("Temperature:", self._kth_temp_spin, 100, pref=True))
        self._stage2.add_widget(self._kth_csv_grp)
        self._kth_csv_grp.setVisible(False)

        # PSS group (only for A_thermal_PSS case)
        sep4 = QFrame(); sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setStyleSheet("color:#555;"); self._stage2.add_widget(sep4)
        self._pss_grp = QGroupBox("PSS parameters (A_thermal_PSS case)")
        psg = QVBoxLayout(self._pss_grp)
        self._pss_src_combo = QComboBox()
        self._pss_src_combo.addItems(["manual_fraction", "manual_absorbance"])
        psg.addLayout(_field_row("PSS source:", self._pss_src_combo, 160))
        self._pss_frac_spin = QDoubleSpinBox()
        self._pss_frac_spin.setRange(0, 1)
        self._pss_frac_spin.setDecimals(4)
        self._pss_frac_spin.setValue(0.85)
        psg.addLayout(_field_row("PSS fraction B:", self._pss_frac_spin, 100))
        self._pss_abs_spin = QDoubleSpinBox()
        self._pss_abs_spin.setRange(0, 10)
        self._pss_abs_spin.setDecimals(4)
        self._pss_abs_spin.setValue(0.0)
        psg.addLayout(_field_row("A_abs at PSS:", self._pss_abs_spin, 100))
        self._stage2.add_widget(self._pss_grp)
        self._pss_grp.setVisible(False)
        self._case_combo.currentTextChanged.connect(self._on_case_changed)

        # Initial conditions
        sep5 = QFrame(); sep5.setFrameShape(QFrame.Shape.HLine)
        sep5.setStyleSheet("color:#555;"); self._stage2.add_widget(sep5)
        self._init_src_combo = QComboBox()
        self._init_src_combo.addItems(["absorbance", "manual"])
        self._stage2.add_layout(
            _field_row("Initial conc. source:", self._init_src_combo, 120))
        self._init_src_combo.currentTextChanged.connect(self._on_init_src_changed)

        self._init_manual_grp = QGroupBox("Manual initial concentrations")
        img = QVBoxLayout(self._init_manual_grp)
        self._conc_A0_spin = QDoubleSpinBox()
        self._conc_A0_spin.setRange(0, 1)
        self._conc_A0_spin.setDecimals(8)
        self._conc_A0_spin.setValue(0.0)
        self._conc_A0_spin.setSuffix(" mol/L")
        img.addLayout(_field_row("[A]₀ (mol/L):", self._conc_A0_spin, 140))
        self._conc_B0_spin = QDoubleSpinBox()
        self._conc_B0_spin.setRange(0, 1)
        self._conc_B0_spin.setDecimals(8)
        self._conc_B0_spin.setValue(0.0)
        self._conc_B0_spin.setSuffix(" mol/L")
        img.addLayout(_field_row("[B]₀ (mol/L):", self._conc_B0_spin, 140))
        self._stage2.add_widget(self._init_manual_grp)
        self._init_manual_grp.setVisible(False)

        parent_layout.addWidget(self._stage2)

    # ── Stage 3 — Extinction Coefficients ────────────────────────────────────

    def _build_stage3(self, parent_layout):
        self._stage3 = StageCard("Stage 3 — Extinction Coefficients")
        self._stage3.add_info_button(
            "Extinction Coefficients",
            "ε_A and ε_B at the irradiation wavelength are required to "
            "convert absorbance into the fraction of absorbed light.\n\n"
            "Sources:\n"
            "  manual — enter the molar absorption coefficient directly "
            "(M⁻¹ cm⁻¹).\n"
            "  csv — load a spectrum CSV (wavelength_nm, epsilon columns) "
            "produced by the Extinction Coefficient tab; the value at the "
            "irradiation wavelength is interpolated automatically.\n\n"
            "For an A-only case, ε_B at the irradiation wavelength can be "
            "left at 0."
        )

        hint = _label(
            "ε at the irradiation wavelength; also needed at monitoring wavelengths for B "
            "(leave B manual at 0 if A-only).",
            color="#888")
        hint.setWordWrap(True)
        self._stage3.add_widget(hint)

        # ε_A
        eps_a_grp = QGroupBox("ε_A (isomer A / reactant)")
        eag = QVBoxLayout(eps_a_grp)
        self._eps_a_src_combo = QComboBox()
        self._eps_a_src_combo.addItems(["manual", "csv"])
        eag.addLayout(_field_row("Source:", self._eps_a_src_combo, 80, pref=True))
        self._eps_a_src_combo.currentTextChanged.connect(
            lambda t: self._on_eps_src_changed("A", t))

        self._eps_a_manual_grp = QGroupBox("")
        eamg = QVBoxLayout(self._eps_a_manual_grp)
        self._eps_a_irr_spin = QDoubleSpinBox()
        self._eps_a_irr_spin.setRange(0, 1e7)
        self._eps_a_irr_spin.setDecimals(1)
        self._eps_a_irr_spin.setValue(10000.0)
        self._eps_a_irr_spin.setSuffix(" M⁻¹cm⁻¹")
        eamg.addLayout(_field_row("ε_A at irr λ:", self._eps_a_irr_spin, 140, pref=True))
        eag.addWidget(self._eps_a_manual_grp)

        self._eps_a_csv_grp = QGroupBox("")
        eacg = QVBoxLayout(self._eps_a_csv_grp)
        self._eps_a_csv_edit = QLineEdit()
        self._eps_a_csv_edit.setPlaceholderText("extinction_coeff_master.csv …")
        self._eps_a_csv_browse = QPushButton("Browse…")
        self._eps_a_csv_browse.setFixedWidth(80)
        self._eps_a_csv_browse.clicked.connect(self._browse_eps_a_csv)
        eacg.addLayout(_browse_row("CSV:", self._eps_a_csv_edit, self._eps_a_csv_browse))
        self._eps_a_col_edit = QLineEdit("Mean")
        eacg.addLayout(_field_row("Column:", self._eps_a_col_edit, 100))
        eag.addWidget(self._eps_a_csv_grp)
        self._eps_a_csv_grp.setVisible(False)
        self._stage3.add_widget(eps_a_grp)

        # ε_B
        eps_b_grp = QGroupBox("ε_B (isomer B / product)")
        ebg = QVBoxLayout(eps_b_grp)
        self._eps_b_src_combo = QComboBox()
        self._eps_b_src_combo.addItems(["manual", "csv"])
        ebg.addLayout(_field_row("Source:", self._eps_b_src_combo, 80, pref=True))
        self._eps_b_src_combo.currentTextChanged.connect(
            lambda t: self._on_eps_src_changed("B", t))

        self._eps_b_manual_grp = QGroupBox("")
        ebmg = QVBoxLayout(self._eps_b_manual_grp)
        self._eps_b_irr_spin = QDoubleSpinBox()
        self._eps_b_irr_spin.setRange(0, 1e7)
        self._eps_b_irr_spin.setDecimals(1)
        self._eps_b_irr_spin.setValue(0.0)
        self._eps_b_irr_spin.setSuffix(" M⁻¹cm⁻¹")
        ebmg.addLayout(_field_row("ε_B at irr λ:", self._eps_b_irr_spin, 140, pref=True))
        ebg.addWidget(self._eps_b_manual_grp)

        self._eps_b_csv_grp = QGroupBox("")
        ebcg = QVBoxLayout(self._eps_b_csv_grp)
        self._eps_b_csv_edit = QLineEdit()
        self._eps_b_csv_edit.setPlaceholderText("isomer_B_extinction.csv …")
        self._eps_b_csv_browse = QPushButton("Browse…")
        self._eps_b_csv_browse.setFixedWidth(80)
        self._eps_b_csv_browse.clicked.connect(self._browse_eps_b_csv)
        ebcg.addLayout(_browse_row("CSV:", self._eps_b_csv_edit, self._eps_b_csv_browse))
        self._eps_b_col_edit = QLineEdit("Mean")
        ebcg.addLayout(_field_row("Column:", self._eps_b_col_edit, 100))
        ebg.addWidget(self._eps_b_csv_grp)
        self._eps_b_csv_grp.setVisible(False)
        self._stage3.add_widget(eps_b_grp)

        parent_layout.addWidget(self._stage3)

    # ── Stage 4 — Fitting & Results ───────────────────────────────────────────

    def _build_stage4(self, parent_layout):
        self._stage4 = StageCard("Stage 4 — Fitting & Results")
        self._stage4.add_info_button(
            "Fitting & Results",
            "Initial guesses (Φ_AB, Φ_BA): starting values for the "
            "non-linear least-squares optimiser. Results are not sensitive "
            "to these unless the problem is ill-conditioned.\n\n"
            "Bounds: lower and upper limits for the quantum yield values "
            "(default 0–1). Set both to 0 to leave unconstrained.\n\n"
            "The fit minimises the residuals between the measured absorbance "
            "kinetics and the ODE-simulated model at all monitoring wavelengths.\n\n"
            "Results are shown per file and per wavelength. Save CSV writes "
            "the fitted Φ values, uncertainties, and goodness-of-fit metrics."
        )

        # Fitting parameters
        self._qy_ab_init_spin = QDoubleSpinBox()
        self._qy_ab_init_spin.setRange(1e-6, 1.0)
        self._qy_ab_init_spin.setDecimals(5)
        self._qy_ab_init_spin.setValue(0.1)
        self._stage4.add_layout(_field_row("Φ_AB initial:", self._qy_ab_init_spin, 100, pref=True))

        self._qy_ba_init_spin = QDoubleSpinBox()
        self._qy_ba_init_spin.setRange(1e-6, 1.0)
        self._qy_ba_init_spin.setDecimals(5)
        self._qy_ba_init_spin.setValue(0.05)
        self._stage4.add_layout(_field_row("Φ_BA initial:", self._qy_ba_init_spin, 100, pref=True))

        self._qy_lo_spin = QDoubleSpinBox()
        self._qy_lo_spin.setRange(0, 1)
        self._qy_lo_spin.setDecimals(6)
        self._qy_lo_spin.setValue(1e-6)
        self._stage4.add_layout(_field_row("Bounds lower:", self._qy_lo_spin, 100))

        self._qy_hi_spin = QDoubleSpinBox()
        self._qy_hi_spin.setRange(0, 10)
        self._qy_hi_spin.setDecimals(4)
        self._qy_hi_spin.setValue(1.0)
        self._stage4.add_layout(_field_row("Bounds upper:", self._qy_hi_spin, 100))

        self._unconstrained_chk = QCheckBox("Unconstrained fit (no bounds)")
        self._stage4.add_widget(self._unconstrained_chk)

        self._n_slopes_spin = QSpinBox()
        self._n_slopes_spin.setRange(3, 100)
        self._n_slopes_spin.setValue(8)
        self._stage4.add_layout(
            _field_row("Initial slopes pts:", self._n_slopes_spin, 80))

        self._qy_ref_edit = QLineEdit()
        self._qy_ref_edit.setPlaceholderText("(optional) reference Φ_AB for fixed-BA fit")
        self._stage4.add_layout(_field_row("Reference Φ_AB:", self._qy_ref_edit, 160))

        # Run button
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#555;"); self._stage4.add_widget(sep)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run All Files")
        self._run_btn.setFixedWidth(140)
        self._run_btn.clicked.connect(self._run)
        self._run_btn.setEnabled(False)
        self._status_lbl = _label("Waiting for input…", color="#888")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._status_lbl)
        run_row.addStretch()
        self._stage4.add_layout(run_row)

        # Plot navigator
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.clicked.connect(self._prev_result)
        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(32)
        self._next_btn.clicked.connect(self._next_result)
        self._nav_lbl = _label("—")
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._nav_lbl)
        nav.addWidget(self._next_btn)
        nav.addStretch()
        self._stage4.add_layout(nav)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

        # Plot widget
        self._plot = PlotWidget()
        self._plot.setMinimumHeight(320)
        self._stage4.add_widget(self._plot)

        # Summary table
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "File", "λ mon (nm)", "Φ_AB", "σ_fit", "σ_total", "R²"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(120)
        self._table.verticalHeader().setVisible(False)
        self._stage4.add_widget(self._table)

        # Save CSV
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save master CSV…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_csv)
        save_row.addWidget(self._save_btn)
        save_row.addStretch()
        self._stage4.add_layout(save_row)

        parent_layout.addWidget(self._stage4)

    # ── Visibility toggles ────────────────────────────────────────────────────

    def _on_data_type_changed(self, text):
        self._scan_grp.setVisible(text == "scanning")

    def _on_flux_src_changed(self, text):
        self._flux_manual_grp.setVisible(text == "manual_mol_s")
        self._flux_uw_grp.setVisible(text == "manual_uW")
        self._flux_actin_grp.setVisible(text == "actinometry")
        self._flux_led_grp.setVisible(text == "led_spectrum")

    def _on_baseline_changed(self, text):
        self._baseline_plat_grp.setVisible(text == "plateau")
        self._baseline_file_grp.setVisible(text == "file")

    def _on_autodetect_toggled(self, checked):
        self._auto_detect_grp.setVisible(checked)
        self._manual_window_grp.setVisible(not checked)

    def _on_kth_src_changed(self, text):
        self._kth_manual_grp.setVisible(text == "manual")
        self._kth_csv_grp.setVisible(text in ("half_life_master", "eyring", "arrhenius"))

    def _on_case_changed(self, text):
        self._pss_grp.setVisible(text == "A_thermal_PSS")

    def _on_init_src_changed(self, text):
        self._init_manual_grp.setVisible(text == "manual")

    def _on_eps_src_changed(self, which, text):
        if which == "A":
            self._eps_a_manual_grp.setVisible(text == "manual")
            self._eps_a_csv_grp.setVisible(text == "csv")
        else:
            self._eps_b_manual_grp.setVisible(text == "manual")
            self._eps_b_csv_grp.setVisible(text == "csv")

    # ── File selection ────────────────────────────────────────────────────────

    def _select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select QY data CSV files",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if not paths:
            return
        self._file_list.clear()
        for p in paths:
            self._file_list.addItem(QListWidgetItem(p))
        n = len(paths)
        self._file_count_lbl.setText(f"{n} file{'s' if n != 1 else ''} selected.")
        self._run_btn.setEnabled(True)
        self._stage1.set_status(READY)

    def _clear_files(self):
        self._file_list.clear()
        self._file_count_lbl.setText("No files selected.")
        self._run_btn.setEnabled(False)
        self._stage1.set_status(WAITING)

    def _get_file_paths(self) -> list[Path]:
        return [Path(self._file_list.item(i).text())
                for i in range(self._file_list.count())]

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_actin_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Actinometry CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._actin_csv_edit.setText(p)

    def _browse_led_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "LED spectrum CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._led_csv_edit.setText(p)

    def _browse_baseline_file(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Baseline CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._baseline_file_edit.setText(p)

    def _browse_kth_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "k_th CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._kth_csv_edit.setText(p)

    def _browse_eps_a_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "ε_A CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._eps_a_csv_edit.setText(p)

    def _browse_eps_b_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "ε_B CSV",
            str(self._output_path or Path.home()),
            "CSV files (*.csv);;All files (*)")
        if p:
            self._eps_b_csv_edit.setText(p)

    # ── Collect QYParams from GUI ─────────────────────────────────────────────

    def _collect_params(self) -> QYParams:
        p = QYParams()

        # Stage 1 — flux
        p.data_type = self._data_type_combo.currentText()
        if p.data_type == "scanning":
            p.delta_t_s       = self._delta_t_spin.value()
            p.scans_per_group = self._scans_per_grp_spin.value()
            p.first_cycle_off = self._first_cycle_off_chk.isChecked()

        p.irradiation_wavelength_nm = self._irr_wl_spin.value()

        src = self._flux_src_combo.currentText()
        p.photon_flux_source = src
        if src == "manual_mol_s":
            p.photon_flux_mol_s     = self._flux_mol_s_spin.value()
            p.photon_flux_std_mol_s = self._flux_std_spin.value()
        elif src == "manual_uW":
            p.photon_flux_uW        = self._flux_uw_spin.value()
            p.photon_flux_std_mol_s = self._flux_uw_std_spin.value()
        elif src == "actinometry":
            txt = self._actin_csv_edit.text().strip()
            p.actinometry_csv = Path(txt) if txt else None
            v = self._actin_filter_spin.value()
            p.actinometry_filter_nm = v if v > 0 else p.irradiation_wavelength_nm
        elif src == "led_spectrum":
            txt = self._led_csv_edit.text().strip()
            p.led_spectrum_csv    = Path(txt) if txt else None
            p.led_integration_mode = self._led_integ_combo.currentText()

        # Stage 2 — experiment
        p.case          = self._case_combo.currentText()
        p.temperature_C = self._temp_spin.value()
        p.solvent       = self._solvent_edit.text().strip()
        p.path_length_cm = self._path_spin.value()
        p.volume_mL      = self._vol_spin.value()

        wl_text = self._mon_wl_edit.text().strip()
        if wl_text:
            try:
                p.monitoring_wavelengths = [
                    float(x) for x in wl_text.replace(";", ",").split(",")
                    if x.strip()]
            except ValueError:
                p.monitoring_wavelengths = None
        else:
            p.monitoring_wavelengths = None
        p.wavelength_tolerance_nm = self._wl_tol_spin.value()

        p.baseline_correction = self._baseline_combo.currentText()
        if p.baseline_correction == "plateau":
            v = self._plat_dur_spin.value()
            p.offset_plateau_duration_s  = v if v > 0 else None
            vs = self._plat_start_spin.value()
            p.baseline_plateau_start_s   = vs if vs > 0 else None
            ve = self._plat_end_spin.value()
            p.baseline_plateau_end_s     = ve if ve > 0 else None
        elif p.baseline_correction == "file":
            txt = self._baseline_file_edit.text().strip()
            p.baseline_file = Path(txt) if txt else None

        p.auto_detect_irr_start = self._auto_detect_chk.isChecked()
        if p.auto_detect_irr_start:
            p.auto_detect_n_plateau  = self._n_plat_spin.value()
            p.auto_detect_threshold  = self._detect_thresh_spin.value()
            p.auto_detect_min_consec = self._min_consec_spin.value()
        else:
            vs = self._fit_start_spin.value()
            ve = self._fit_end_spin.value()
            p.fit_time_start_s = vs if vs > 0 else None
            p.fit_time_end_s   = ve if ve > 0 else None

        p.k_th_source = self._kth_src_combo.currentText()
        if p.k_th_source == "manual":
            p.k_th_manual     = self._kth_manual_spin.value()
            p.k_th_manual_std = self._kth_manual_std_spin.value()
        elif p.k_th_source in ("half_life_master", "eyring", "arrhenius"):
            txt = self._kth_csv_edit.text().strip()
            p.k_th_csv           = Path(txt) if txt else None
            p.k_th_temperature_C = self._kth_temp_spin.value()

        if p.case == "A_thermal_PSS":
            p.pss_source = self._pss_src_combo.currentText()
            if p.pss_source == "manual_fraction":
                p.pss_fraction_B_manual = self._pss_frac_spin.value()
            else:
                p.pss_A_abs_pss_manual = self._pss_abs_spin.value()

        p.initial_conc_source = self._init_src_combo.currentText()
        if p.initial_conc_source == "manual":
            p.initial_conc_A_manual = self._conc_A0_spin.value()
            p.initial_conc_B_manual = self._conc_B0_spin.value()

        # Stage 3 — extinction coefficients
        p.epsilon_source_A = self._eps_a_src_combo.currentText()
        if p.epsilon_source_A == "manual":
            p.epsilon_A_irr = self._eps_a_irr_spin.value()
        else:
            txt = self._eps_a_csv_edit.text().strip()
            p.epsilon_A_csv = Path(txt) if txt else None
            p.epsilon_A_col = self._eps_a_col_edit.text().strip() or "Mean"

        p.epsilon_source_B = self._eps_b_src_combo.currentText()
        if p.epsilon_source_B == "manual":
            p.epsilon_B_irr = self._eps_b_irr_spin.value()
        else:
            txt = self._eps_b_csv_edit.text().strip()
            p.epsilon_B_csv = Path(txt) if txt else None
            p.epsilon_B_col = self._eps_b_col_edit.text().strip() or "Mean"

        # Stage 4 — fitting
        p.QY_AB_init           = self._qy_ab_init_spin.value()
        p.QY_BA_init           = self._qy_ba_init_spin.value()
        p.QY_bounds_lo         = self._qy_lo_spin.value()
        p.QY_bounds_hi         = self._qy_hi_spin.value()
        p.QY_unconstrained     = self._unconstrained_chk.isChecked()
        p.n_initial_slopes_pts = self._n_slopes_spin.value()
        ref_text = self._qy_ref_edit.text().strip()
        if ref_text:
            try:
                p.QY_AB_reference = float(ref_text)
            except ValueError:
                pass

        return p

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        files = self._get_file_paths()
        if not files:
            self._status_lbl.setText("No files selected.")
            return

        params = self._collect_params()

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._results.clear()
        self._table.setRowCount(0)
        self._stage4.set_status(WAITING)
        self._status_lbl.setText(f"Running {len(files)} file(s)…")

        def _compute():
            # Resolve shared inputs once
            N_mol_s, N_std, led_wl, led_N = load_photon_flux(params)
            k_th, k_th_std = load_k_th(params)
            irr_wl = params.irradiation_wavelength_nm

            # ε_A and ε_B at irradiation wavelength
            eps_a_irr = load_epsilon_at_wavelength(
                params.epsilon_source_A, params.epsilon_A_csv,
                params.epsilon_A_col, irr_wl, params.epsilon_A_irr, "A")
            eps_b_irr = load_epsilon_at_wavelength(
                params.epsilon_source_B, params.epsilon_B_csv,
                params.epsilon_B_col, irr_wl, params.epsilon_B_irr, "B")

            # LED ε arrays (if LED full-integration)
            led_eps_a = None
            led_eps_b = None
            if led_wl is not None and params.led_integration_mode == "full":
                led_eps_a = load_epsilon_at_wavelengths(
                    params.epsilon_source_A, params.epsilon_A_csv,
                    params.epsilon_A_col, list(led_wl), params.epsilon_A_irr)
                led_eps_b = load_epsilon_at_wavelengths(
                    params.epsilon_source_B, params.epsilon_B_csv,
                    params.epsilon_B_col, list(led_wl), params.epsilon_B_irr)

            results = []
            for f in files:
                r = run_qy_file(params, f,
                                N_mol_s, N_std, k_th, k_th_std,
                                eps_a_irr, eps_b_irr,
                                led_wl, led_N, led_eps_a, led_eps_b)
                results.append(r)
            return results

        self._worker = Worker(_compute)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_results)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_results(self, results: list):
        self._results = results
        self._current_idx = 0
        self._populate_table()
        self._update_nav()
        self._show_result(0)
        self._stage4.set_status(DONE)
        self._status_lbl.setText(
            f"Done — {len(results)} file(s) processed.")
        self._save_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._stage4.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")
        self.log_signal.emit(msg, "error")

    def _on_finished(self):
        self._run_btn.setEnabled(True)
        self._worker = None

    # ── Navigation & display ──────────────────────────────────────────────────

    def _update_nav(self):
        n = len(self._results)
        self._prev_btn.setEnabled(n > 1)
        self._next_btn.setEnabled(n > 1)
        if n:
            self._nav_lbl.setText(
                f"File {self._current_idx + 1} / {n}: "
                f"{self._results[self._current_idx].file_name}")
        else:
            self._nav_lbl.setText("—")

    def _prev_result(self):
        if self._results:
            self._current_idx = (self._current_idx - 1) % len(self._results)
            self._update_nav()
            self._show_result(self._current_idx)

    def _next_result(self):
        if self._results:
            self._current_idx = (self._current_idx + 1) % len(self._results)
            self._update_nav()
            self._show_result(self._current_idx)

    def _show_result(self, idx: int):
        if not self._results or idx >= len(self._results):
            return
        r = self._results[idx]
        fig = plot_qy_result(r)
        self._plot.set_figure(fig)
        if self._output_path:
            self._plot.set_save_dir(
                self._output_path / "quantum_yield" / "results" / "plots")
        stem = Path(r.file_name).stem if r.file_name else f"qy_{idx}"
        self._plot.set_default_filename(f"{stem}_QY.png")

    def _populate_table(self):
        self._table.setRowCount(0)
        for r in self._results:
            for j, wl in enumerate(r.mon_wls):
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(r.file_name))
                self._table.setItem(row, 1, QTableWidgetItem(f"{wl:.1f}"))
                qy_ab = r.QY_AB_per_wl[j] if j < len(r.QY_AB_per_wl) else r.QY_AB
                sf    = r.stderr_AB_per_wl[j] if j < len(r.stderr_AB_per_wl) else r.QY_AB_sigma_fit
                st    = r.sigma_total_per_wl[j] if j < len(r.sigma_total_per_wl) else r.QY_AB_sigma_total
                r2    = r.r2_per_wl[j] if j < len(r.r2_per_wl) else r.r2
                self._table.setItem(row, 2, QTableWidgetItem(f"{qy_ab:.5f}"))
                self._table.setItem(row, 3, QTableWidgetItem(f"{sf:.5f}"))
                self._table.setItem(row, 4, QTableWidgetItem(f"{st:.5f}"))
                self._table.setItem(row, 5, QTableWidgetItem(f"{r2:.4f}"))

    # ── Save CSV ──────────────────────────────────────────────────────────────

    def _save_csv(self):
        if not self._results:
            return
        import pandas as pd
        rows = []
        for r in self._results:
            for j, wl in enumerate(r.mon_wls):
                qy_ab = r.QY_AB_per_wl[j] if j < len(r.QY_AB_per_wl) else r.QY_AB
                qy_ba = r.QY_BA_per_wl[j] if j < len(r.QY_BA_per_wl) else r.QY_BA
                sf    = r.stderr_AB_per_wl[j] if j < len(r.stderr_AB_per_wl) else r.QY_AB_sigma_fit
                st    = r.sigma_total_per_wl[j] if j < len(r.sigma_total_per_wl) else r.QY_AB_sigma_total
                r2    = r.r2_per_wl[j] if j < len(r.r2_per_wl) else r.r2
                rows.append({
                    "File":              r.file_name,
                    "Compound":          r.compound,
                    "Case":              r.case,
                    "Mon_wl_nm":         wl,
                    "N_mol_s":           r.N_mol_s,
                    "N_std_mol_s":       r.N_std_mol_s,
                    "k_th_s":            r.k_th,
                    "eps_A_irr":         r.eps_A_irr,
                    "eps_B_irr":         r.eps_B_irr,
                    "Temperature_C":     r.temperature_C,
                    "Solvent":           r.solvent,
                    "QY_AB":             qy_ab,
                    "QY_BA":             qy_ba,
                    "sigma_fit_AB":      sf,
                    "sigma_total_AB":    st,
                    "R2":                r2,
                    "Method":            r.method,
                })
        df = pd.DataFrame(rows)
        default_name = "qy_master.csv"
        default_dir  = str(self._output_path / "quantum_yield" / "results"
                           if self._output_path else Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save QY master CSV",
            str(Path(default_dir) / default_name),
            "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            self.log_signal.emit(f"QY master CSV saved → {path}", "info")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        self._plot.set_save_dir(path / "quantum_yield" / "results" / "plots")

    def apply_prefs(self, prefs):
        if not hasattr(prefs, "quantum_yield"):
            return
        qp = prefs.quantum_yield
        self._case_combo.setCurrentText(qp.case)
        self._temp_spin.setValue(qp.temperature_C)
        self._solvent_edit.setText(qp.solvent)
        self._path_spin.setValue(qp.path_length_cm)
        self._vol_spin.setValue(qp.volume_mL)
        self._flux_src_combo.setCurrentText(qp.photon_flux_source)
        self._flux_mol_s_spin.setValue(qp.photon_flux_mol_s)
        self._flux_std_spin.setValue(qp.photon_flux_std_mol_s)
        self._irr_wl_spin.setValue(qp.irradiation_wavelength_nm)
        self._data_type_combo.setCurrentText(qp.data_type)
        self._delta_t_spin.setValue(qp.delta_t_s)
        self._scans_per_grp_spin.setValue(qp.scans_per_group)
        self._kth_src_combo.setCurrentText(qp.k_th_source)
        self._kth_temp_spin.setValue(qp.k_th_temperature_C)
        self._eps_a_src_combo.setCurrentText(qp.epsilon_source_A)
        self._eps_a_irr_spin.setValue(qp.epsilon_A_irr)
        self._eps_b_src_combo.setCurrentText(qp.epsilon_source_B)
        self._eps_b_irr_spin.setValue(qp.epsilon_B_irr)
        self._qy_ab_init_spin.setValue(qp.QY_AB_init)
        self._qy_ba_init_spin.setValue(qp.QY_BA_init)
        self._wl_tol_spin.setValue(qp.wavelength_tolerance_nm)
        if qp.monitoring_wavelengths:
            self._mon_wl_edit.setText(
                ", ".join(str(x) for x in qp.monitoring_wavelengths))

    def collect_prefs(self, prefs):
        if not hasattr(prefs, "quantum_yield"):
            return
        qp = prefs.quantum_yield
        qp.case                    = self._case_combo.currentText()
        qp.temperature_C           = self._temp_spin.value()
        qp.solvent                 = self._solvent_edit.text().strip()
        qp.path_length_cm          = self._path_spin.value()
        qp.volume_mL               = self._vol_spin.value()
        qp.photon_flux_source      = self._flux_src_combo.currentText()
        qp.photon_flux_mol_s       = self._flux_mol_s_spin.value()
        qp.photon_flux_std_mol_s   = self._flux_std_spin.value()
        qp.irradiation_wavelength_nm = self._irr_wl_spin.value()
        qp.data_type               = self._data_type_combo.currentText()
        qp.delta_t_s               = self._delta_t_spin.value()
        qp.scans_per_group         = self._scans_per_grp_spin.value()
        qp.k_th_source             = self._kth_src_combo.currentText()
        qp.k_th_temperature_C      = self._kth_temp_spin.value()
        qp.epsilon_source_A        = self._eps_a_src_combo.currentText()
        qp.epsilon_A_irr           = self._eps_a_irr_spin.value()
        qp.epsilon_source_B        = self._eps_b_src_combo.currentText()
        qp.epsilon_B_irr           = self._eps_b_irr_spin.value()
        qp.QY_AB_init              = self._qy_ab_init_spin.value()
        qp.QY_BA_init              = self._qy_ba_init_spin.value()
        qp.wavelength_tolerance_nm = self._wl_tol_spin.value()
        wl_text = self._mon_wl_edit.text().strip()
        if wl_text:
            try:
                qp.monitoring_wavelengths = [
                    float(x) for x in wl_text.replace(";", ",").split(",")
                    if x.strip()]
            except ValueError:
                qp.monitoring_wavelengths = []
        else:
            qp.monitoring_wavelengths = []
