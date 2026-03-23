"""
Spectra tab — extract pure species B spectrum from irradiation series.

Four extraction modes:
  negative      – reference-wavelength subtraction
  negative_pca  – PCA non-negativity extrapolation (bleaching system)
  positive_pca  – PCA non-negativity extrapolation (build-up system)
  positive_pss  – known PSS conversion fraction

Stage layout
────────────
  Stage 1 — Input files    (initial, irradiation, optional PSS)
  Stage 2 — Parameters     (mode, compound, extraction params)
  Stage 3 — Run & Results  (extract + plot + save CSV)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QFrame,
    QGroupBox,
)

from gui.tabs.spectra_core import (
    SpectraParams, SpectraResult,
    load_and_average_files, load_irradiation_series_files, load_pss_files,
    run_spectra_extraction,
    plot_overview, plot_extraction_result,
    plot_sb_diagnostic, plot_pca_diagnostic, plot_convergence,
)
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.worker import Worker
from gui.project_prefs import ProjectPrefs, SpectraPrefs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _files_label(paths: list[Path]) -> str:
    if not paths:
        return "No files selected"
    if len(paths) == 1:
        return paths[0].name
    return f"{len(paths)} files  ({paths[0].name}, …)"


# ══════════════════════════════════════════════════════════════════════════════
# SpectraTab
# ══════════════════════════════════════════════════════════════════════════════

class SpectraTab(QWidget):
    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path:    Optional[Path] = None
        self._initial_files:  list[Path]     = []
        self._irrad_files:    list[Path]     = []
        self._pss_files:      list[Path]     = []
        self._grid:           Optional[np.ndarray] = None
        self._S_A:            Optional[np.ndarray] = None
        self._series:         Optional[np.ndarray] = None
        self._S_PSS:          Optional[np.ndarray] = None
        self._result:         Optional[SpectraResult] = None
        self._worker:         Optional[Worker] = None
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

        self._build_stage1(layout)
        self._build_stage2(layout)
        self._build_stage3(layout)
        layout.addStretch()

    # ── Stage 1 ─────────────────────────────────────────────────────────────

    def _build_stage1(self, parent_layout):
        self._stage1 = StageCard("Stage 1 — Input Files")
        self._stage1.add_info_button(
            "Input Files",
            "Three file groups are needed:\n\n"
            "Initial spectrum: one or more CSV scans of pure species A before "
            "any irradiation. These are averaged to form S_A.\n\n"
            "Irradiation series: scans taken during photolysis, sorted "
            "chronologically by filename. The algorithm tracks how the mixture "
            "spectrum evolves to extract S_B.\n\n"
            "PSS files (optional): scans at the photostationary state — only "
            "required for the positive_pss extraction mode."
        )

        hint = QLabel(
            "Select the CSV files for each group.\n"
            "Initial spectrum: one or more scans of pure species A.\n"
            "Irradiation series: scans taken during photolysis (sorted by filename).\n"
            "PSS files: only required for positive_pss mode."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint)

        # ── Initial files
        init_row = QHBoxLayout()
        self._initial_btn = QPushButton("Select initial files…")
        self._initial_btn.setFixedWidth(160)
        self._initial_btn.clicked.connect(self._select_initial)
        self._initial_lbl = QLabel("No files selected")
        self._initial_lbl.setStyleSheet("color:#555;")
        init_row.addWidget(self._initial_btn)
        init_row.addWidget(self._initial_lbl, 1)
        self._stage1.add_layout(init_row)

        # ── Irradiation files
        irr_row = QHBoxLayout()
        self._irrad_btn = QPushButton("Select irradiation files…")
        self._irrad_btn.setFixedWidth(160)
        self._irrad_btn.clicked.connect(self._select_irrad)
        self._irrad_lbl = QLabel("No files selected")
        self._irrad_lbl.setStyleSheet("color:#555;")
        irr_row.addWidget(self._irrad_btn)
        irr_row.addWidget(self._irrad_lbl, 1)
        self._stage1.add_layout(irr_row)

        # ── PSS files (optional)
        pss_row = QHBoxLayout()
        self._pss_btn = QPushButton("Select PSS files… (optional)")
        self._pss_btn.setFixedWidth(200)
        self._pss_btn.clicked.connect(self._select_pss)
        self._pss_lbl = QLabel("None selected")
        self._pss_lbl.setStyleSheet("color:#555;")
        pss_row.addWidget(self._pss_btn)
        pss_row.addWidget(self._pss_lbl, 1)
        self._stage1.add_layout(pss_row)

        # ── Load button
        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load & Preview Data")
        self._load_btn.setFixedWidth(160)
        self._load_btn.clicked.connect(self._load_data)
        self._load_status_lbl = QLabel("")
        self._load_status_lbl.setStyleSheet("color:#555;")
        load_row.addWidget(self._load_btn)
        load_row.addWidget(self._load_status_lbl, 1)
        load_row.addStretch()
        self._stage1.add_layout(load_row)

        # Overview plot
        self._overview_plot = PlotWidget(info_title="Data Overview",
                                         info_text="Irradiation series coloured "
                                                   "blue (early) → red (late). "
                                                   "Inset shows the long-wavelength "
                                                   "baseline region.")
        self._overview_plot.setMinimumHeight(300)
        self._overview_plot.hide()
        self._stage1.add_widget(self._overview_plot)

        parent_layout.addWidget(self._stage1)

    # ── Stage 2 ─────────────────────────────────────────────────────────────

    def _build_stage2(self, parent_layout):
        self._stage2 = StageCard("Stage 2 — Parameters")
        self._stage2.add_info_button(
            "Extraction Parameters",
            "Extraction mode:\n"
            "  negative — subtract the spectrum at a reference wavelength "
            "where only species A absorbs.\n"
            "  negative_pca — SVD non-negativity extrapolation for a "
            "system where species A bleaches.\n"
            "  positive_pca — SVD non-negativity extrapolation for a "
            "system where species B builds up.\n"
            "  positive_pss — known PSS conversion fraction; provide "
            "PSS files in Stage 1.\n\n"
            "Reference λ: anchor wavelength for the negative subtraction mode. "
            "Should be an isosbestic or A-only absorption point.\n\n"
            "Bootstrap iterations: number of resampled fits used to estimate "
            "uncertainty on the extracted spectrum."
        )
        self._stage2.set_status(WAITING)

        # ── Mode selector
        mode_row = QHBoxLayout()
        _mode_lbl = QLabel("Extraction mode:")
        _mode_lbl.setObjectName("pref_label")
        mode_row.addWidget(_mode_lbl)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "negative",
            "negative_pca",
            "positive_pca",
            "positive_pss",
        ])
        self._mode_combo.setFixedWidth(140)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        self._stage2.add_layout(mode_row)

        mode_hint = QLabel(
            "negative: reference λ subtraction  |  "
            "negative/positive_pca: SVD + non-negativity extrapolation  |  "
            "positive_pss: known PSS fraction"
        )
        mode_hint.setWordWrap(True)
        mode_hint.setStyleSheet("color:#888; font-size:8pt;")
        self._stage2.add_widget(mode_hint)

        # ── Compound + concentration group (shared)
        shared_grp = QGroupBox("Sample")
        shared_lay = QVBoxLayout(shared_grp)

        c1 = QHBoxLayout()
        _cn_lbl = QLabel("Compound name:")
        _cn_lbl.setObjectName("pref_label")
        c1.addWidget(_cn_lbl)
        self._compound_edit = QLineEdit()
        self._compound_edit.setPlaceholderText("e.g. DAE_001")
        self._compound_edit.setFixedWidth(180)
        c1.addWidget(self._compound_edit)
        c1.addStretch()
        shared_lay.addLayout(c1)

        c2 = QHBoxLayout()
        _pl_lbl = QLabel("Path length (cm):")
        _pl_lbl.setObjectName("pref_label")
        c2.addWidget(_pl_lbl)
        self._path_length_spin = QDoubleSpinBox()
        self._path_length_spin.setRange(0.001, 100.0)
        self._path_length_spin.setDecimals(3)
        self._path_length_spin.setValue(1.0)
        self._path_length_spin.setFixedWidth(90)
        c2.addWidget(self._path_length_spin)
        c2.addSpacing(20)
        c2.addWidget(QLabel("Concentration (mol/L, blank = absorbance output):"))
        self._conc_edit = QLineEdit()
        self._conc_edit.setPlaceholderText("leave blank for absorbance")
        self._conc_edit.setFixedWidth(200)
        c2.addWidget(self._conc_edit)
        c2.addStretch()
        shared_lay.addLayout(c2)

        self._stage2.add_widget(shared_grp)

        # ── Baseline group (shared)
        bl_grp = QGroupBox("Baseline")
        bl_lay = QVBoxLayout(bl_grp)

        bl_row = QHBoxLayout()
        bl_row.addWidget(QLabel("Offset to add to all spectra:"))
        self._baseline_offset_spin = QDoubleSpinBox()
        self._baseline_offset_spin.setRange(-1.0, 1.0)
        self._baseline_offset_spin.setDecimals(6)
        self._baseline_offset_spin.setSingleStep(0.001)
        self._baseline_offset_spin.setValue(0.0)
        self._baseline_offset_spin.setFixedWidth(100)
        bl_row.addWidget(self._baseline_offset_spin)
        bl_row.addSpacing(20)
        _sbt_lbl = QLabel("S_B tolerance (multiples of baseline σ):")
        _sbt_lbl.setObjectName("pref_label")
        bl_row.addWidget(_sbt_lbl)
        self._sb_tol_spin = QDoubleSpinBox()
        self._sb_tol_spin.setRange(0.0, 100.0)
        self._sb_tol_spin.setDecimals(1)
        self._sb_tol_spin.setValue(3.0)
        self._sb_tol_spin.setFixedWidth(70)
        bl_row.addWidget(self._sb_tol_spin)
        bl_row.addStretch()
        bl_lay.addLayout(bl_row)

        self._stage2.add_widget(bl_grp)

        # ── Spectrum index selection (shared)
        idx_grp = QGroupBox("Spectrum index filter (optional)")
        idx_lay = QVBoxLayout(idx_grp)
        idx_hint = QLabel(
            "Leave blank to use all irradiation spectra.\n"
            "Range: enter \"start:stop\" (0-based, stop exclusive).  "
            "List: comma-separated indices, e.g. \"0,2,5,7\"."
        )
        idx_hint.setWordWrap(True)
        idx_hint.setStyleSheet("color:#888; font-size:8pt;")
        idx_lay.addWidget(idx_hint)
        idx_row = QHBoxLayout()
        idx_row.addWidget(QLabel("Indices:"))
        self._indices_edit = QLineEdit()
        self._indices_edit.setPlaceholderText("blank = use all")
        self._indices_edit.setFixedWidth(200)
        idx_row.addWidget(self._indices_edit)
        idx_row.addStretch()
        idx_lay.addLayout(idx_row)
        self._stage2.add_widget(idx_grp)

        # ── Mode-specific groups ───────────────────────────────────────────

        # negative mode
        self._neg_grp = QGroupBox("Negative switch parameters")
        neg_lay = QVBoxLayout(self._neg_grp)

        ref_row = QHBoxLayout()
        _ref_lbl = QLabel("Reference λ (nm):")
        _ref_lbl.setObjectName("pref_label")
        ref_row.addWidget(_ref_lbl)
        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("e.g. 500  or  490:520  or  490,510,520")
        self._ref_edit.setFixedWidth(220)
        ref_row.addWidget(self._ref_edit)
        ref_row.addStretch()
        neg_lay.addLayout(ref_row)

        ref_hint = QLabel(
            "Single value: 500  |  Band: 490:520  |  "
            "List: 490,510,520"
        )
        ref_hint.setStyleSheet("color:#888; font-size:8pt;")
        neg_lay.addWidget(ref_hint)

        self._ref_weighted_cb = QCheckBox("Weighted reference (least-squares α estimation)")
        self._ref_weighted_cb.setObjectName("pref_cb")
        self._ref_weighted_cb.setChecked(True)
        neg_lay.addWidget(self._ref_weighted_cb)

        alpha_row = QHBoxLayout()
        _amin_lbl = QLabel("α window  min:")
        _amin_lbl.setObjectName("pref_label")
        alpha_row.addWidget(_amin_lbl)
        self._min_alpha_spin = QDoubleSpinBox()
        self._min_alpha_spin.setRange(0.0, 1.0)
        self._min_alpha_spin.setDecimals(2)
        self._min_alpha_spin.setSingleStep(0.05)
        self._min_alpha_spin.setValue(0.2)
        self._min_alpha_spin.setFixedWidth(70)
        alpha_row.addWidget(self._min_alpha_spin)
        alpha_row.addSpacing(10)
        _amax_lbl = QLabel("max:")
        _amax_lbl.setObjectName("pref_label")
        alpha_row.addWidget(_amax_lbl)
        self._max_alpha_spin = QDoubleSpinBox()
        self._max_alpha_spin.setRange(0.0, 1.0)
        self._max_alpha_spin.setDecimals(2)
        self._max_alpha_spin.setSingleStep(0.05)
        self._max_alpha_spin.setValue(0.6)
        self._max_alpha_spin.setFixedWidth(70)
        alpha_row.addWidget(self._max_alpha_spin)
        alpha_row.addStretch()
        neg_lay.addLayout(alpha_row)

        self._excl_neg_cb = QCheckBox("Exclude spectra with negative S_B values")
        self._excl_neg_cb.setObjectName("pref_cb")
        self._excl_neg_cb.setChecked(True)
        neg_lay.addWidget(self._excl_neg_cb)

        self._stage2.add_widget(self._neg_grp)

        # PCA mode
        self._pca_grp = QGroupBox("PCA parameters")
        pca_lay = QVBoxLayout(self._pca_grp)
        pca_row = QHBoxLayout()
        _bt_lbl = QLabel("Bootstrap iterations:")
        _bt_lbl.setObjectName("pref_label")
        pca_row.addWidget(_bt_lbl)
        self._n_bootstrap_spin = QSpinBox()
        self._n_bootstrap_spin.setRange(100, 10000)
        self._n_bootstrap_spin.setSingleStep(500)
        self._n_bootstrap_spin.setValue(2000)
        self._n_bootstrap_spin.setFixedWidth(90)
        pca_row.addWidget(self._n_bootstrap_spin)
        pca_row.addStretch()
        pca_lay.addLayout(pca_row)
        self._stage2.add_widget(self._pca_grp)

        # PSS mode
        self._pss_grp = QGroupBox("PSS fraction parameters")
        pss_lay = QVBoxLayout(self._pss_grp)
        pss_row = QHBoxLayout()
        _fb_lbl = QLabel("f_B at PSS:")
        _fb_lbl.setObjectName("pref_label")
        pss_row.addWidget(_fb_lbl)
        self._pss_fb_spin = QDoubleSpinBox()
        self._pss_fb_spin.setRange(0.01, 0.99)
        self._pss_fb_spin.setDecimals(3)
        self._pss_fb_spin.setValue(0.85)
        self._pss_fb_spin.setFixedWidth(80)
        pss_row.addWidget(self._pss_fb_spin)
        pss_row.addSpacing(20)
        _fbe_lbl = QLabel("± error:")
        _fbe_lbl.setObjectName("pref_label")
        pss_row.addWidget(_fbe_lbl)
        self._pss_fb_err_spin = QDoubleSpinBox()
        self._pss_fb_err_spin.setRange(0.001, 0.5)
        self._pss_fb_err_spin.setDecimals(3)
        self._pss_fb_err_spin.setValue(0.02)
        self._pss_fb_err_spin.setFixedWidth(80)
        pss_row.addWidget(self._pss_fb_err_spin)
        pss_row.addStretch()
        pss_lay.addLayout(pss_row)
        self._stage2.add_widget(self._pss_grp)

        # Initial mode visibility
        self._on_mode_changed("negative")

        parent_layout.addWidget(self._stage2)

    # ── Stage 3 ─────────────────────────────────────────────────────────────

    def _build_stage3(self, parent_layout):
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(
            "Extraction Results",
            "Overview plot: the full irradiation series coloured blue (early) "
            "to red (late), overlaid with the extracted pure-B spectrum.\n\n"
            "Extraction plot: S_B (extracted species B spectrum) with its "
            "bootstrap uncertainty band. Also shown: S_A (initial) and "
            "the mixture spectra for context.\n\n"
            "The output CSV contains wavelength_nm, S_A, S_B, S_B_std columns "
            "and can be loaded in the Extinction Coefficient tab to compute ε."
        )
        self._stage3.set_status(WAITING)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Extraction")
        self._run_btn.setFixedWidth(150)
        self._run_btn.clicked.connect(self._run_extraction)
        self._run_status_lbl = QLabel("")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._run_status_lbl, 1)
        run_row.addStretch()
        self._stage3.add_layout(run_row)

        self._diag_chk = QCheckBox("Show diagnostics  "
                                   "(individual estimates / PCA scores / convergence)")
        self._stage3.add_widget(self._diag_chk)

        self._result_plot = PlotWidget(
            info_title="Extraction Result",
            info_text="Extracted S_B spectrum with ±1σ / 95% CI shading. "
                      "Grey traces are a subset of the irradiation series "
                      "shown for context.",
        )
        self._result_plot.setMinimumHeight(320)
        self._result_plot.hide()
        self._stage3.add_widget(self._result_plot)

        # Diagnostic plots (hidden until diagnostics are run)
        self._diag_plot1 = PlotWidget(
            info_title="Diagnostic: estimates / PCA",
            info_text="For negative mode: individual S_B estimates coloured by α.\n"
                      "For PCA modes: PC1 score progression and bootstrap samples.",
        )
        self._diag_plot1.setMinimumHeight(300)
        self._diag_plot1.hide()
        self._stage3.add_widget(self._diag_plot1)

        self._diag_plot2 = PlotWidget(
            info_title="Diagnostic: convergence",
            info_text="S_B extracted from growing fractions of the irradiation "
                      "series (25 % → 100 %). Stable overlapping curves indicate "
                      "a reliable extraction.",
        )
        self._diag_plot2.setMinimumHeight(300)
        self._diag_plot2.hide()
        self._stage3.add_widget(self._diag_plot2)

        # Save row
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save extracted spectra CSV…")
        self._save_btn.setFixedWidth(230)
        self._save_btn.clicked.connect(self._save_result)
        self._save_btn.setEnabled(False)
        self._save_status_lbl = QLabel("")
        self._save_status_lbl.setStyleSheet("color:#555;")
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._save_status_lbl, 1)
        save_row.addStretch()
        self._stage3.add_layout(save_row)

        parent_layout.addWidget(self._stage3)

    # ── File selection ────────────────────────────────────────────────────────

    def _select_initial(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select initial spectrum files", "", "CSV files (*.csv)")
        if paths:
            self._initial_files = [Path(p) for p in paths]
            self._initial_lbl.setText(_files_label(self._initial_files))
            self._reset_loaded()

    def _select_irrad(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select irradiation series files", "", "CSV files (*.csv)")
        if paths:
            self._irrad_files = [Path(p) for p in paths]
            self._irrad_lbl.setText(_files_label(self._irrad_files))
            self._reset_loaded()

    def _select_pss(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PSS spectrum files", "", "CSV files (*.csv)")
        if paths:
            self._pss_files = [Path(p) for p in paths]
            self._pss_lbl.setText(_files_label(self._pss_files))
            self._reset_loaded()

    def _reset_loaded(self):
        self._grid   = None
        self._S_A    = None
        self._series = None
        self._S_PSS  = None
        self._stage1.set_status(WAITING)
        self._stage2.set_status(WAITING)
        self._stage3.set_status(WAITING)
        self._overview_plot.hide()
        self._result_plot.hide()
        self._diag_plot1.hide()
        self._diag_plot2.hide()
        self._save_btn.setEnabled(False)

    # ── Mode change ───────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str):
        self._neg_grp.setVisible(mode == "negative")
        self._pca_grp.setVisible(mode in ("negative_pca", "positive_pca"))
        self._pss_grp.setVisible(mode == "positive_pss")

    # ── Load data ─────────────────────────────────────────────────────────────

    def _load_data(self):
        if not self._initial_files:
            self._load_status_lbl.setText("Select initial spectrum files first.")
            return
        if not self._irrad_files:
            self._load_status_lbl.setText("Select irradiation series files first.")
            return

        if self._worker and self._worker.isRunning():
            return

        self._load_btn.setEnabled(False)
        self._stage1.set_status(WAITING)
        self._load_status_lbl.setText("Loading…")

        initial_files = list(self._initial_files)
        irrad_files   = list(self._irrad_files)
        pss_files     = list(self._pss_files)
        mode          = self._mode_combo.currentText()

        def _do_load():
            grid, S_A, n_init = load_and_average_files(initial_files)
            print(f"Initial spectrum: {n_init} scan(s) averaged, "
                  f"grid {grid[0]}–{grid[-1]} nm")
            series = load_irradiation_series_files(irrad_files, grid)
            print(f"Irradiation series: {len(series)} spectra")
            S_PSS = None
            if pss_files:
                S_PSS = load_pss_files(pss_files, grid)
                print(f"PSS spectrum loaded ({len(pss_files)} file(s))")
            fig = plot_overview(
                grid, S_A, series, S_PSS,
                mode=mode,
                reference_wavelength_nm=None,   # will be set per run
                compound_name="",
            )
            return grid, S_A, series, S_PSS, fig

        self._worker = Worker(_do_load)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_load_done)
        self._worker.error_signal.connect(self._on_load_error)
        self._worker.finished_signal.connect(
            lambda: self._load_btn.setEnabled(True))
        self._worker.start()

    def _on_load_done(self, res):
        grid, S_A, series, S_PSS, fig = res
        self._grid   = grid
        self._S_A    = S_A
        self._series = series
        self._S_PSS  = S_PSS

        self._overview_plot.set_figure(fig)
        if self._output_path:
            self._overview_plot.set_save_dir(self._output_path / "spectra" / "results" / "plots")
        self._overview_plot.set_default_filename("data_overview.png")
        self._overview_plot.show()

        n = len(series)
        self._load_status_lbl.setText(
            f"Loaded — grid {grid[0]}–{grid[-1]} nm, {n} irradiation spectra")
        self._stage1.set_status(DONE)
        self._stage2.set_status(READY)
        self._stage3.set_status(WAITING)
        self._result_plot.hide()
        self._save_btn.setEnabled(False)

    def _on_load_error(self, msg):
        self._load_status_lbl.setText(f"Error: {msg}")
        self._stage1.set_status(ERROR)

    # ── Parse reference wavelength ────────────────────────────────────────────

    def _parse_reference(self) -> object:
        """Parse the reference λ text field into float, (lo,hi) tuple, or list."""
        text = self._ref_edit.text().strip()
        if not text:
            raise ValueError("Reference wavelength is required for negative mode.")
        # Range: "490:520"
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2:
                raise ValueError(f"Cannot parse reference range: '{text}'")
            return (float(parts[0]), float(parts[1]))
        # List: "490,510,520"
        if "," in text:
            return [float(x) for x in text.split(",")]
        # Single
        return float(text)

    @staticmethod
    def _parse_indices(text: str, n_total: int) -> object:
        text = text.strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            return (int(parts[0]), int(parts[1]))
        return [int(x) for x in text.split(",")]

    # ── Run extraction ────────────────────────────────────────────────────────

    def _run_extraction(self):
        if self._grid is None or self._S_A is None or self._series is None:
            self._run_status_lbl.setText("Load data first (Stage 1).")
            return
        if self._worker and self._worker.isRunning():
            return

        mode = self._mode_combo.currentText()

        # Build params
        try:
            ref = (self._parse_reference()
                   if mode == "negative" else 500.0)
        except ValueError as e:
            self._run_status_lbl.setText(str(e))
            return

        conc_text = self._conc_edit.text().strip()
        concentration = float(conc_text) if conc_text else None

        try:
            indices = self._parse_indices(
                self._indices_edit.text(), len(self._series))
        except ValueError:
            self._run_status_lbl.setText("Invalid spectrum indices.")
            return

        params = SpectraParams(
            mode                    = mode,
            compound_name           = self._compound_edit.text().strip(),
            path_length_cm          = self._path_length_spin.value(),
            concentration_mol_L     = concentration,
            baseline_offset         = self._baseline_offset_spin.value(),
            reference_wavelength_nm = ref,
            reference_weighted      = self._ref_weighted_cb.isChecked(),
            min_alpha               = self._min_alpha_spin.value(),
            max_alpha               = self._max_alpha_spin.value(),
            exclude_negative_SB     = self._excl_neg_cb.isChecked(),
            sb_tolerance_sigma      = self._sb_tol_spin.value(),
            n_bootstrap             = self._n_bootstrap_spin.value(),
            pss_fraction_B          = self._pss_fb_spin.value(),
            pss_fraction_B_error    = self._pss_fb_err_spin.value(),
            spectrum_indices        = indices,
            show_diagnostics        = self._diag_chk.isChecked(),
        )

        if mode == "positive_pss" and not self._pss_files:
            self._run_status_lbl.setText(
                "positive_pss mode requires PSS files (Stage 1).")
            return

        grid   = self._grid
        S_A    = self._S_A
        series = self._series
        S_PSS  = self._S_PSS

        self._run_btn.setEnabled(False)
        self._stage3.set_status(WAITING)
        self._run_status_lbl.setText("Running…")

        def _do_extract():
            result = run_spectra_extraction(params, grid, S_A, series, S_PSS)
            fig    = plot_extraction_result(result)
            return result, fig

        self._worker = Worker(_do_extract)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_extract_done)
        self._worker.error_signal.connect(self._on_extract_error)
        self._worker.finished_signal.connect(
            lambda: self._run_btn.setEnabled(True))
        self._worker.start()

    def _on_extract_done(self, res):
        result, fig = res
        self._result = result

        self._result_plot.set_figure(fig)
        if self._output_path:
            self._result_plot.set_save_dir(self._output_path / "spectra" / "results" / "plots")
        cname = result.compound or "spectra"
        self._result_plot.set_default_filename(
            f"{cname}_{result.mode}_extraction.png")
        self._result_plot.show()

        # Diagnostic plots
        self._diag_plot1.hide()
        self._diag_plot2.hide()
        if self._diag_chk.isChecked():
            if result.mode == "negative" and result.B_estimates is not None:
                self._diag_plot1.set_figure(plot_sb_diagnostic(result))
                self._diag_plot1.show()
            elif result.mode in ("negative_pca", "positive_pca") \
                    and result.pca_scores is not None:
                self._diag_plot1.set_figure(plot_pca_diagnostic(result))
                self._diag_plot1.show()
            if result.convergence_results is not None:
                self._diag_plot2.set_figure(plot_convergence(result))
                self._diag_plot2.show()

        self._run_status_lbl.setText(
            f"Done — {result.n_spectra_used} of {result.n_spectra_total} spectra used")
        self._stage3.set_status(DONE)
        self._save_btn.setEnabled(True)

    def _on_extract_error(self, msg):
        self._run_status_lbl.setText(f"Error: {msg}")
        self._stage3.set_status(ERROR)

    # ── Save CSV ──────────────────────────────────────────────────────────────

    def _save_result(self):
        if self._result is None:
            return
        r = self._result
        default_dir = str(self._output_path / "spectra" / "results") \
            if self._output_path else ""
        cname = r.compound or "spectra"
        default_name = f"{cname}_{r.mode}_spectra.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save extracted spectra",
            str(Path(default_dir) / default_name) if default_dir else default_name,
            "CSV files (*.csv)",
        )
        if not path:
            return

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "Wavelength_nm": r.grid,
            "Species_A":     r.S_A,
            "Species_B":     r.S_B,
            "Species_B_lo":  r.S_B_lo,
            "Species_B_hi":  r.S_B_hi,
            "Species_B_std": r.S_B_std,
        })
        with open(out, "w", encoding="utf-8", newline="") as fh:
            for line in r.meta_lines:
                fh.write(f"# {line}\n")
            df.to_csv(fh, index=False)

        self._save_status_lbl.setText(f"Saved → {out.name}")
        self.log_signal.emit(
            f"[Spectra] Extracted spectra saved to {out}", "info")

    # ── Prefs ─────────────────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        if self._output_path:
            self._result_plot.set_save_dir(path / "spectra" / "plots")
            self._overview_plot.set_save_dir(path / "spectra" / "plots")

    def apply_prefs(self, prefs: ProjectPrefs):
        p = prefs.spectra
        self._compound_edit.setText(p.compound_name)
        self._path_length_spin.setValue(p.path_length_cm)
        mode_idx = self._mode_combo.findText(p.mode)
        if mode_idx >= 0:
            self._mode_combo.setCurrentIndex(mode_idx)
        self._ref_edit.setText(p.reference_wavelength_nm)
        self._ref_weighted_cb.setChecked(p.reference_weighted)
        self._min_alpha_spin.setValue(p.min_alpha)
        self._max_alpha_spin.setValue(p.max_alpha)
        self._excl_neg_cb.setChecked(p.exclude_negative_SB)
        self._sb_tol_spin.setValue(p.sb_tolerance_sigma)
        self._n_bootstrap_spin.setValue(p.n_bootstrap)
        self._pss_fb_spin.setValue(p.pss_fraction_B)
        self._pss_fb_err_spin.setValue(p.pss_fraction_B_error)

    def collect_prefs(self, prefs: ProjectPrefs):
        prefs.spectra.compound_name           = self._compound_edit.text().strip()
        prefs.spectra.path_length_cm          = self._path_length_spin.value()
        prefs.spectra.mode                    = self._mode_combo.currentText()
        prefs.spectra.reference_wavelength_nm = self._ref_edit.text().strip()
        prefs.spectra.reference_weighted      = self._ref_weighted_cb.isChecked()
        prefs.spectra.min_alpha               = self._min_alpha_spin.value()
        prefs.spectra.max_alpha               = self._max_alpha_spin.value()
        prefs.spectra.exclude_negative_SB     = self._excl_neg_cb.isChecked()
        prefs.spectra.sb_tolerance_sigma      = self._sb_tol_spin.value()
        prefs.spectra.n_bootstrap             = self._n_bootstrap_spin.value()
        prefs.spectra.pss_fraction_B          = self._pss_fb_spin.value()
        prefs.spectra.pss_fraction_B_error    = self._pss_fb_err_spin.value()
