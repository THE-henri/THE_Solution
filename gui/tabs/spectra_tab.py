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
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QFrame,
    QGroupBox, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QMenu,
)

from gui.tabs.spectra_core import (
    SpectraParams, SpectraResult,
    load_and_average_files, load_irradiation_series_files, load_pss_files,
    run_spectra_extraction,
    plot_overview, plot_extraction_result,
    plot_sb_diagnostic, plot_pca_diagnostic, plot_convergence,
)
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.info_button import InfoButton
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
        self._raw_path:       Optional[Path] = None
        self._initial_files:     list[Path]  = []
        self._irrad_files:       list[Path]  = []
        self._irrad_labels:      list[str]   = []
        self._irrad_sources:     list[Path]  = []
        self._irrad_disabled_idx: set[int]   = set()
        self._pss_files:         list[Path]  = []
        self._grid:           Optional[np.ndarray] = None
        self._S_A:            Optional[np.ndarray] = None
        self._series:         Optional[np.ndarray] = None
        self._S_PSS:          Optional[np.ndarray] = None
        self._result:         Optional[SpectraResult] = None
        self._worker:         Optional[Worker] = None
        # EC spectrum for concentration calculation
        self._ec_wl_arr:      Optional[np.ndarray] = None
        self._ec_eps_arr:     Optional[np.ndarray] = None
        self._ec_computed_conc: Optional[float]    = None
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
            "Initial spectrum: one or more scans of pure species A (required).\n"
            "Irradiation series: scans during photolysis (sorted by filename). "
            "Optional when only a PSS spectrum is available.\n"
            "PSS files: required for positive_pss mode; also used by negative mode "
            "when no irradiation series is loaded (single-spectrum path)."
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
        self._pss_btn = QPushButton("Select PSS files…")
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

        # Spectrum toggle list (shown after loading)
        self._spectra_list_lbl = QLabel(
            "Uncheck individual spectra to exclude them from the extraction.  "
            "Right-click to open the source file.")
        self._spectra_list_lbl.setStyleSheet("color:#888; font-size:8pt;")
        self._spectra_list_lbl.setWordWrap(True)
        self._spectra_list_lbl.hide()
        self._stage1.add_widget(self._spectra_list_lbl)

        self._irrad_list = QListWidget()
        self._irrad_list.setMaximumHeight(160)
        self._irrad_list.setStyleSheet("font-size:8pt;")
        self._irrad_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._irrad_list.customContextMenuRequested.connect(
            self._spectra_context_menu)
        self._irrad_list.itemChanged.connect(self._on_spectra_item_toggled)
        self._irrad_list.hide()
        self._stage1.add_widget(self._irrad_list)

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
        mode_row.addWidget(InfoButton(
            "Extraction mode",
            "Algorithm used to separate the pure-component spectra:\n\n"
            "'negative' — iterative S_B estimation from alpha-scaled spectral differences. Use when A→B conversion is the only process.\n\n"
            "'negative_pca' — same as negative but uses PCA scores to determine the mixing fraction α. More robust for noisy data.\n\n"
            "'positive_pca' — PCA-based extraction for A→B photoswitches measured from the B side.\n\n"
            "'positive_pss' — uses a known PSS composition to anchor the extraction.",
        ))
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
        c1.addWidget(InfoButton(
            "Compound name",
            "Identifier used in plot titles and output filenames.\n"
            "Does not affect the calculation.",
        ))
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
        c2.addWidget(InfoButton(
            "Path length (cm)",
            "Optical path length of the cuvette in centimetres (typically 1 cm).\n"
            "Used when converting absorbance to molar units if a concentration\n"
            "is provided. Does not affect spectral shape extraction.",
        ))
        self._path_length_spin = QDoubleSpinBox()
        self._path_length_spin.setRange(0.001, 100.0)
        self._path_length_spin.setDecimals(3)
        self._path_length_spin.setValue(1.0)
        self._path_length_spin.setFixedWidth(90)
        self._path_length_spin.valueChanged.connect(self._update_computed_conc)
        c2.addWidget(self._path_length_spin)
        c2.addStretch()
        shared_lay.addLayout(c2)

        # ── Concentration source ───────────────────────────────────────────
        conc_mode_row = QHBoxLayout()
        conc_mode_row.setSpacing(8)
        self._conc_manual_rb = QRadioButton("Concentration (mol/L):")
        self._conc_manual_rb.setChecked(True)
        self._conc_ec_rb     = QRadioButton("Compute from \u03b5 spectrum")
        _conc_bg = QButtonGroup(self)
        _conc_bg.addButton(self._conc_manual_rb)
        _conc_bg.addButton(self._conc_ec_rb)
        conc_mode_row.addWidget(self._conc_manual_rb)
        conc_mode_row.addWidget(InfoButton(
            "Concentration (mol/L)",
            "Solution concentration in mol/L.\n"
            "When provided, extracted spectra are output in units of\n"
            "M\u207b\u00b9cm\u207b\u00b9 (molar absorptivity).\n"
            "Leave blank to output in raw absorbance units.",
        ))
        self._conc_edit = QLineEdit()
        self._conc_edit.setPlaceholderText("leave blank for absorbance units")
        self._conc_edit.setFixedWidth(180)
        conc_mode_row.addWidget(self._conc_edit)
        conc_mode_row.addSpacing(20)
        conc_mode_row.addWidget(self._conc_ec_rb)
        conc_mode_row.addWidget(InfoButton(
            "Compute concentration from \u03b5 spectrum",
            "Load an extinction coefficient spectrum CSV (from the Extinction\n"
            "Coefficient tab or any CSV with Wavelength and \u03b5 columns).\n\n"
            "Choose a reference wavelength \u03bb. The concentration is then\n"
            "computed as:\n"
            "  c = A(\u03bb) / (\u03b5(\u03bb) \u00d7 l)\n\n"
            "where A(\u03bb) is read from the averaged initial spectrum and \u03b5(\u03bb)\n"
            "is interpolated from the loaded \u03b5 spectrum. The computed\n"
            "concentration is shown before running.",
        ))
        conc_mode_row.addStretch()
        shared_lay.addLayout(conc_mode_row)

        # EC spectrum row (shown only in EC mode)
        self._ec_row_widget = QWidget()
        ec_row = QHBoxLayout(self._ec_row_widget)
        ec_row.setContentsMargins(24, 0, 0, 0)
        self._ec_btn = QPushButton("Select \u03b5 spectrum CSV\u2026")
        self._ec_btn.setFixedWidth(180)
        self._ec_btn.clicked.connect(self._select_ec_file)
        self._ec_file_lbl = QLabel("No file selected")
        self._ec_file_lbl.setStyleSheet("color:#555; font-size:9pt;")
        ec_row.addWidget(self._ec_btn)
        ec_row.addWidget(self._ec_file_lbl, 1)
        ec_row.addSpacing(16)
        _ec_wl_lbl = QLabel("\u03bb for c (nm):")
        _ec_wl_lbl.setObjectName("pref_label")
        ec_row.addWidget(_ec_wl_lbl)
        self._ec_wl_spin = QDoubleSpinBox()
        self._ec_wl_spin.setRange(200.0, 1100.0)
        self._ec_wl_spin.setDecimals(1)
        self._ec_wl_spin.setValue(400.0)
        self._ec_wl_spin.setFixedWidth(80)
        self._ec_wl_spin.valueChanged.connect(self._update_computed_conc)
        ec_row.addWidget(self._ec_wl_spin)
        ec_row.addStretch()
        shared_lay.addWidget(self._ec_row_widget)

        # Computed concentration label
        self._ec_conc_lbl = QLabel("")
        self._ec_conc_lbl.setStyleSheet("color:#5b8dee; font-size:9pt; padding-left:24px;")
        shared_lay.addWidget(self._ec_conc_lbl)

        # Wire mode toggle
        self._conc_manual_rb.toggled.connect(self._on_conc_mode_changed)
        self._on_conc_mode_changed()  # set initial visibility

        self._stage2.add_widget(shared_grp)

        # ── Baseline group (shared)
        bl_grp = QGroupBox("Baseline")
        bl_lay = QVBoxLayout(bl_grp)

        bl_row = QHBoxLayout()
        bl_row.addWidget(QLabel("Offset to add to all spectra:"))
        bl_row.addWidget(InfoButton(
            "Spectral offset",
            "Constant added to every spectrum before extraction.\n"
            "Use to correct a residual baseline offset (e.g. dust or cuvette\n"
            "scattering). Estimate from the flat spectral region outside any\n"
            "absorption bands. Typical range: \u22120.05 to +0.05.",
        ))
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
        bl_row.addWidget(InfoButton(
            "S_B tolerance",
            "Outlier rejection threshold for individual S_B estimates,\n"
            "expressed as multiples of the baseline noise \u03c3.\n\n"
            "Estimates whose values deviate by more than this factor from the\n"
            "median are excluded. Set to 0 to disable outlier rejection.\n"
            "Typical value: 3\u20135.",
        ))
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
        idx_row.addWidget(InfoButton(
            "Spectrum indices",
            "Filter which spectra from the irradiation series are used.\n"
            "Examples:\n"
            "  '0:20'  \u2014 first 20 spectra\n"
            "  '5,10,15' \u2014 specific indices\n"
            "  blank \u2014 use all spectra\n\n"
            "Useful to exclude early transients or late plateau spectra.",
        ))
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
        _ref_lbl = QLabel("Reference \u03bb (nm):")
        _ref_lbl.setObjectName("pref_label")
        ref_row.addWidget(_ref_lbl)
        ref_row.addWidget(InfoButton(
            "Reference wavelength (nm)",
            "Anchor wavelength for the negative subtraction mode.\n"
            "Should be an isosbestic point or a wavelength where only\n"
            "species A absorbs (species B is transparent).\n\n"
            "A good reference minimises the residual at that wavelength\n"
            "across all spectra in the series.",
        ))
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

        ref_weighted_row = QHBoxLayout()
        self._ref_weighted_cb = QCheckBox("Weighted reference (least-squares α estimation)")
        self._ref_weighted_cb.setObjectName("pref_cb")
        self._ref_weighted_cb.setChecked(True)
        ref_weighted_row.addWidget(self._ref_weighted_cb)
        ref_weighted_row.addWidget(InfoButton(
            "Weighted reference (\u03b1 estimation)",
            "When checked, the mixing fraction \u03b1 is estimated by\n"
            "least-squares minimisation across all wavelengths (weighted).\n\n"
            "When unchecked, \u03b1 is estimated from the ratio at the single\n"
            "reference wavelength only. The weighted mode is more robust\n"
            "when the reference wavelength is in a noisy region.",
        ))
        ref_weighted_row.addStretch()
        neg_lay.addLayout(ref_weighted_row)

        alpha_row = QHBoxLayout()
        _amin_lbl = QLabel("\u03b1 window  min:")
        _amin_lbl.setObjectName("pref_label")
        alpha_row.addWidget(_amin_lbl)
        alpha_row.addWidget(InfoButton(
            "\u03b1 minimum",
            "Lower bound on the mixing fraction \u03b1.\n"
            "\u03b1 represents how much of species B spectrum is present in a\n"
            "given scan. Physical range: 0 (pure A) to 1 (pure B).\n\n"
            "Set a lower bound > 0 only if you are certain B is always present.",
        ))
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
        alpha_row.addWidget(InfoButton(
            "\u03b1 maximum",
            "Upper bound on the mixing fraction \u03b1.\n"
            "Set to 1.0 for unrestricted conversion. Lower values constrain\n"
            "the extraction to partial conversion data and can improve\n"
            "stability when PSS has not been reached.",
        ))
        self._max_alpha_spin = QDoubleSpinBox()
        self._max_alpha_spin.setRange(0.0, 1.0)
        self._max_alpha_spin.setDecimals(2)
        self._max_alpha_spin.setSingleStep(0.05)
        self._max_alpha_spin.setValue(0.6)
        self._max_alpha_spin.setFixedWidth(70)
        alpha_row.addWidget(self._max_alpha_spin)
        alpha_row.addStretch()
        neg_lay.addLayout(alpha_row)

        excl_neg_row = QHBoxLayout()
        self._excl_neg_cb = QCheckBox("Exclude spectra with negative S_B values")
        self._excl_neg_cb.setObjectName("pref_cb")
        self._excl_neg_cb.setChecked(True)
        excl_neg_row.addWidget(self._excl_neg_cb)
        excl_neg_row.addWidget(InfoButton(
            "Exclude negative S_B spectra",
            "When checked, spectra that produce negative values in the\n"
            "extracted S_B spectrum are discarded.\n\n"
            "Negative values are unphysical for absorbance spectra. Excluding\n"
            "these frames usually improves the quality of the extraction but\n"
            "may reduce the number of points in the bootstrap.",
        ))
        excl_neg_row.addStretch()
        neg_lay.addLayout(excl_neg_row)

        self._stage2.add_widget(self._neg_grp)

        # PCA mode
        self._pca_grp = QGroupBox("PCA parameters")
        pca_lay = QVBoxLayout(self._pca_grp)
        pca_row = QHBoxLayout()
        _bt_lbl = QLabel("Bootstrap iterations:")
        _bt_lbl.setObjectName("pref_label")
        pca_row.addWidget(_bt_lbl)
        pca_row.addWidget(InfoButton(
            "Bootstrap iterations",
            "Number of resampled fits used to estimate uncertainty on S_B.\n"
            "Higher values give more precise confidence bands but increase\n"
            "computation time.\n\n"
            "Recommended: \u2265 500 for publication-quality uncertainty.\n"
            "100\u2013200 is sufficient for exploratory analysis.",
        ))
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
        pss_row.addWidget(InfoButton(
            "f_B at PSS",
            "Fraction of species B at the photostationary state (PSS).\n"
            "Determines how the final spectrum of the mixture is decomposed\n"
            "into pure A and pure B contributions.\n\n"
            "Measure experimentally from NMR, HPLC, or another analytical\n"
            "method. Typical values: 0.5\u20130.95 for efficient photoswitches.",
        ))
        self._pss_fb_spin = QDoubleSpinBox()
        self._pss_fb_spin.setRange(0.01, 0.99)
        self._pss_fb_spin.setDecimals(3)
        self._pss_fb_spin.setValue(0.85)
        self._pss_fb_spin.setFixedWidth(80)
        pss_row.addWidget(self._pss_fb_spin)
        pss_row.addSpacing(20)
        _fbe_lbl = QLabel("\u00b1 error:")
        _fbe_lbl.setObjectName("pref_label")
        pss_row.addWidget(_fbe_lbl)
        pss_row.addWidget(InfoButton(
            "f_B error",
            "Uncertainty (\u00b1) on the PSS composition f_B.\n"
            "Used in the bootstrap to propagate f_B uncertainty into\n"
            "the extracted spectrum confidence band.\n\n"
            "Typically 0.02\u20130.05 (absolute, e.g. from NMR integration error).",
        ))
        self._pss_fb_err_spin = QDoubleSpinBox()
        self._pss_fb_err_spin.setRange(0.001, 0.5)
        self._pss_fb_err_spin.setDecimals(3)
        self._pss_fb_err_spin.setValue(0.02)
        self._pss_fb_err_spin.setFixedWidth(80)
        pss_row.addWidget(self._pss_fb_err_spin)
        pss_row.addStretch()
        pss_lay.addLayout(pss_row)

        # ── Auto-compute f_B from observation wavelength (negative photoswitch)
        pss_auto_row = QHBoxLayout()
        self._pss_auto_fb_cb = QCheckBox(
            "Auto-compute f_B from bleaching at observation λ (negative photoswitch)")
        self._pss_auto_fb_cb.setObjectName("pref_cb")
        self._pss_auto_fb_cb.setChecked(False)
        pss_auto_row.addWidget(self._pss_auto_fb_cb)
        pss_auto_row.addWidget(InfoButton(
            "Auto-compute f_B from observation λ",
            "For negative photoswitches where species A bleaches at λ_obs\n"
            "and species B has negligible absorbance there:\n\n"
            "  f_B = 1 − A_PSS(λ_obs) / A₀(λ_obs)\n\n"
            "Set λ_obs to the peak absorption wavelength of species A.\n"
            "Requires initial and PSS spectra to be loaded (Stage 1).\n"
            "The ± error field above still sets the f_B uncertainty.",
        ))
        pss_auto_row.addStretch()
        pss_lay.addLayout(pss_auto_row)

        self._pss_obs_row = QWidget()
        obs_row = QHBoxLayout(self._pss_obs_row)
        obs_row.setContentsMargins(24, 0, 0, 0)
        _obs_lbl = QLabel("Observation λ (nm):")
        _obs_lbl.setObjectName("pref_label")
        obs_row.addWidget(_obs_lbl)
        obs_row.addWidget(InfoButton(
            "Observation wavelength",
            "Wavelength used to calculate f_B from the bleaching ratio.\n"
            "Choose the maximum absorbance of species A where species B\n"
            "is transparent.",
        ))
        self._pss_obs_wl_spin = QDoubleSpinBox()
        self._pss_obs_wl_spin.setRange(200.0, 1100.0)
        self._pss_obs_wl_spin.setDecimals(1)
        self._pss_obs_wl_spin.setValue(400.0)
        self._pss_obs_wl_spin.setFixedWidth(80)
        self._pss_obs_wl_spin.valueChanged.connect(self._update_auto_fb)
        obs_row.addWidget(self._pss_obs_wl_spin)
        obs_row.addStretch()
        pss_lay.addWidget(self._pss_obs_row)

        self._pss_auto_fb_lbl = QLabel("")
        self._pss_auto_fb_lbl.setStyleSheet(
            "color:#5b8dee; font-size:9pt; padding-left:24px;")
        pss_lay.addWidget(self._pss_auto_fb_lbl)

        self._pss_auto_fb_cb.toggled.connect(self._on_pss_auto_fb_changed)
        self._on_pss_auto_fb_changed()

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

        diag_row = QHBoxLayout()
        self._diag_chk = QCheckBox("Show diagnostics  "
                                   "(individual estimates / PCA scores / convergence)")
        diag_row.addWidget(self._diag_chk)
        diag_row.addWidget(InfoButton(
            "Show diagnostics",
            "When checked, additional diagnostic plots are shown after extraction:\n\n"
            "\u2022 negative mode: individual S_B estimates across the series\n"
            "\u2022 PCA mode: PC1 scores vs. irradiation time\n"
            "\u2022 Both modes: convergence test \u2014 S_B extracted from growing\n"
            "  fractions of the data (25 %\u2192100 %). Stable overlapping curves\n"
            "  indicate a reliable extraction.",
        ))
        diag_row.addStretch()
        self._stage3.add_layout(diag_row)

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

    def _file_start_dir(self) -> str:
        """Return the best starting directory for file dialogs."""
        # Prefer the directory of already-selected files, then raw path, then home
        for files in (self._initial_files, self._irrad_files, self._pss_files):
            if files:
                return str(files[0].parent)
        if self._raw_path and self._raw_path.exists():
            return str(self._raw_path)
        return str(Path.home())

    def _select_initial(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select initial spectrum files",
            self._file_start_dir(), "CSV files (*.csv)")
        if paths:
            self._initial_files = [Path(p) for p in paths]
            self._initial_lbl.setText(_files_label(self._initial_files))
            self._reset_loaded()

    def _select_irrad(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select irradiation series files",
            self._file_start_dir(), "CSV files (*.csv)")
        if paths:
            self._irrad_files = [Path(p) for p in paths]
            self._irrad_lbl.setText(_files_label(self._irrad_files))
            self._reset_loaded()

    def _populate_spectra_list(self, labels: list[str], sources: list[Path]):
        self._irrad_labels  = labels
        self._irrad_sources = sources
        self._irrad_disabled_idx.clear()
        self._irrad_list.blockSignals(True)
        self._irrad_list.clear()
        for idx, (label, src) in enumerate(zip(labels, sources)):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (idx, src))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(str(src))
            self._irrad_list.addItem(item)
        self._irrad_list.blockSignals(False)
        if labels:
            self._irrad_list.show()
            self._spectra_list_lbl.show()
        else:
            self._irrad_list.hide()
            self._spectra_list_lbl.hide()

    def _on_spectra_item_toggled(self, item: QListWidgetItem):
        idx, _src = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Unchecked:
            self._irrad_disabled_idx.add(idx)
        else:
            self._irrad_disabled_idx.discard(idx)
        # Update label to show how many are active
        total  = len(self._irrad_labels)
        active = total - len(self._irrad_disabled_idx)
        base   = _files_label(self._irrad_files)
        self._irrad_lbl.setText(
            base if active == total else f"{base}  ({active}/{total} spectra active)")

    def _spectra_context_menu(self, pos):
        item = self._irrad_list.itemAt(pos)
        if item is None:
            return
        _idx, src = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        open_act = menu.addAction("Open source file")
        action = menu.exec(self._irrad_list.mapToGlobal(pos))
        if action == open_act:
            os.startfile(str(src))

    def _select_pss(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PSS spectrum files",
            self._file_start_dir(), "CSV files (*.csv)")
        if paths:
            self._pss_files = [Path(p) for p in paths]
            self._pss_lbl.setText(_files_label(self._pss_files))
            self._reset_loaded()

    def _reset_loaded(self):
        self._grid   = None
        self._S_A    = None
        self._series = None
        self._S_PSS  = None
        self._irrad_labels      = []
        self._irrad_sources     = []
        self._irrad_disabled_idx.clear()
        self._stage1.set_status(WAITING)
        self._stage2.set_status(WAITING)
        self._stage3.set_status(WAITING)
        self._overview_plot.hide()
        self._irrad_list.hide()
        self._spectra_list_lbl.hide()
        self._result_plot.hide()
        self._diag_plot1.hide()
        self._diag_plot2.hide()
        self._save_btn.setEnabled(False)

    # ── Mode change ───────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str):
        self._neg_grp.setVisible(mode == "negative")
        self._pca_grp.setVisible(mode in ("negative_pca", "positive_pca"))
        self._pss_grp.setVisible(mode == "positive_pss")

    def _on_pss_auto_fb_changed(self):
        auto = self._pss_auto_fb_cb.isChecked()
        self._pss_fb_spin.setEnabled(not auto)
        self._pss_obs_row.setVisible(auto)
        self._pss_auto_fb_lbl.setVisible(auto)
        self._update_auto_fb()

    def _update_auto_fb(self):
        """Compute and display f_B = 1 − A_PSS(λ) / A₀(λ) for the PSS auto mode."""
        self._pss_auto_fb_lbl.setText("")
        if (not self._pss_auto_fb_cb.isChecked()
                or self._S_A is None or self._S_PSS is None
                or self._grid is None):
            return
        lam      = self._pss_obs_wl_spin.value()
        sa_obs   = float(np.interp(lam, self._grid, self._S_A))
        spss_obs = float(np.interp(lam, self._grid, self._S_PSS))
        if sa_obs <= 0:
            self._pss_auto_fb_lbl.setText(
                f"A₀({lam:.0f} nm) ≤ 0 — choose a wavelength where species A absorbs")
            return
        f_B = 1.0 - spss_obs / sa_obs
        if not (0.01 < f_B < 0.99):
            self._pss_auto_fb_lbl.setText(
                f"  f_B = {f_B:.3f} — out of valid range (0.01–0.99), adjust λ_obs")
            return
        self._pss_auto_fb_lbl.setText(
            f"  → f_B = {f_B:.4f}   "
            f"[A₀({lam:.0f} nm) = {sa_obs:.4f},  "
            f"A_PSS({lam:.0f} nm) = {spss_obs:.4f}]")

    # ── Load data ─────────────────────────────────────────────────────────────

    def _load_data(self):
        if not self._initial_files:
            self._load_status_lbl.setText("Select initial spectrum files first.")
            return
        mode = self._mode_combo.currentText()
        if not self._irrad_files and mode != "positive_pss":
            if not self._pss_files:
                self._load_status_lbl.setText(
                    "Select irradiation series files first — or, for PSS-only "
                    "extraction (negative / positive_pss modes), load only "
                    "initial + PSS files.")
                return

        if self._worker and self._worker.isRunning():
            return

        self._load_btn.setEnabled(False)
        self._stage1.set_status(WAITING)
        self._load_status_lbl.setText("Loading…")

        initial_files = list(self._initial_files)
        irrad_files   = list(self._irrad_files)
        pss_files     = list(self._pss_files)

        def _do_load():
            grid, S_A, n_init = load_and_average_files(initial_files)
            print(f"Initial spectrum: {n_init} scan(s) averaged, "
                  f"grid {grid[0]}–{grid[-1]} nm")
            if irrad_files:
                series, labels, sources = load_irradiation_series_files(irrad_files, grid)
                print(f"Irradiation series: {len(series)} spectra")
            else:
                series  = np.empty((0, len(grid)))
                labels  = []
                sources = []
                print("No irradiation files — PSS-only mode")
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
            return grid, S_A, series, labels, sources, S_PSS, fig

        self._worker = Worker(_do_load)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_load_done)
        self._worker.error_signal.connect(self._on_load_error)
        self._worker.finished_signal.connect(
            lambda: self._load_btn.setEnabled(True))
        self._worker.start()

    def _on_load_done(self, res):
        grid, S_A, series, labels, sources, S_PSS, fig = res
        self._grid   = grid
        self._S_A    = S_A
        self._series = series
        self._S_PSS  = S_PSS
        self._update_computed_conc()   # refresh computed c now that S_A is known

        self._overview_plot.set_figure(fig)
        if self._output_path:
            self._overview_plot.set_save_dir(self._output_path / "spectra" / "results" / "plots")
        self._overview_plot.set_default_filename("data_overview.png")
        self._overview_plot.show()

        self._populate_spectra_list(labels, sources)
        self._update_auto_fb()

        n = len(series)
        irrad_info = f"{n} irradiation spectra" if n > 0 else "no irradiation series (PSS-only)"
        self._load_status_lbl.setText(
            f"Loaded — grid {grid[0]}–{grid[-1]} nm, {irrad_info}")
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

    # ── Concentration mode helpers ─────────────────────────────────────────────

    def _on_conc_mode_changed(self):
        manual = self._conc_manual_rb.isChecked()
        self._conc_edit.setEnabled(manual)
        self._ec_row_widget.setVisible(not manual)
        self._ec_conc_lbl.setVisible(not manual)
        if not manual:
            self._update_computed_conc()

    def _select_ec_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select extinction coefficient CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, comment="#")
            # Try to find wavelength and epsilon columns (flexible naming)
            wl_col  = next((c for c in df.columns
                            if "wavelength" in c.lower() or "wl" in c.lower()
                            or c.strip().lower() in ("nm", "lambda")), None)
            eps_col = next((c for c in df.columns
                            if "mean" in c.lower() or "epsilon" in c.lower()
                            or "eps" in c.lower() or "molar" in c.lower()
                            or "m-1" in c.lower() or "m^-1" in c.lower()), None)
            if wl_col is None or eps_col is None:
                self._ec_file_lbl.setText(
                    f"Could not find wavelength/epsilon columns in {Path(path).name}")
                return
            self._ec_wl_arr  = pd.to_numeric(df[wl_col],  errors="coerce").values
            self._ec_eps_arr = pd.to_numeric(df[eps_col], errors="coerce").values
            valid = np.isfinite(self._ec_wl_arr) & np.isfinite(self._ec_eps_arr)
            self._ec_wl_arr  = self._ec_wl_arr[valid]
            self._ec_eps_arr = self._ec_eps_arr[valid]
            self._ec_file_lbl.setText(
                f"{Path(path).name}  ({wl_col} / {eps_col})")
            # Set spinbox range to loaded spectrum range
            self._ec_wl_spin.setRange(
                float(self._ec_wl_arr.min()), float(self._ec_wl_arr.max()))
            self._update_computed_conc()
        except Exception as exc:
            self._ec_file_lbl.setText(f"Error: {exc}")

    def _update_computed_conc(self):
        """Compute c = A(λ) / (ε(λ) × l) and show it."""
        self._ec_computed_conc = None
        if (self._ec_wl_arr is None or self._ec_eps_arr is None
                or self._S_A is None or self._grid is None):
            self._ec_conc_lbl.setText(
                "Load data (Stage 1) and select an \u03b5 spectrum to compute c.")
            return
        lam = self._ec_wl_spin.value()
        l   = self._path_length_spin.value()
        # Interpolate ε at λ
        eps_at_lam = float(np.interp(lam, self._ec_wl_arr, self._ec_eps_arr))
        # Interpolate A_initial at λ
        A_at_lam   = float(np.interp(lam, self._grid, self._S_A))
        if eps_at_lam <= 0:
            self._ec_conc_lbl.setText(
                f"\u03b5({lam:.1f} nm) = {eps_at_lam:.2f} — cannot divide by zero")
            return
        c = A_at_lam / (eps_at_lam * l)
        self._ec_computed_conc = c
        self._ec_conc_lbl.setText(
            f"  \u2192  c = {c:.4e} mol/L   "
            f"[\u03b5({lam:.1f} nm) = {eps_at_lam:.2f} M\u207b\u00b9cm\u207b\u00b9, "
            f"A\u2080({lam:.1f} nm) = {A_at_lam:.4f}]")

    def _get_concentration(self) -> Optional[float]:
        """Return the concentration to use for unit conversion."""
        if self._conc_manual_rb.isChecked():
            text = self._conc_edit.text().strip()
            return float(text) if text else None
        return self._ec_computed_conc

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

        concentration = self._get_concentration()

        try:
            indices = self._parse_indices(
                self._indices_edit.text(), len(self._series))
        except ValueError:
            self._run_status_lbl.setText("Invalid spectrum indices.")
            return

        pss_obs_wl = (
            self._pss_obs_wl_spin.value()
            if mode == "positive_pss" and self._pss_auto_fb_cb.isChecked()
            else None
        )

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
            pss_obs_wavelength_nm   = pss_obs_wl,
            spectrum_indices        = indices,
            show_diagnostics        = self._diag_chk.isChecked(),
        )

        if mode == "positive_pss" and not self._pss_files:
            self._run_status_lbl.setText(
                "positive_pss mode requires PSS files (Stage 1).")
            return

        grid   = self._grid
        S_A    = self._S_A
        S_PSS  = self._S_PSS
        if self._irrad_disabled_idx:
            active = [i for i in range(len(self._series))
                      if i not in self._irrad_disabled_idx]
            if not active:
                self._run_status_lbl.setText(
                    "All spectra are disabled — enable at least one.")
                return
            series = self._series[active]
        else:
            series = self._series

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
        default_dir = self._output_path / "spectra" / "results" \
            if self._output_path else None
        if default_dir:
            default_dir.mkdir(parents=True, exist_ok=True)
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

    def set_raw_path(self, path: Path):
        self._raw_path = path

    def set_output_path(self, path: Path):
        self._output_path = path
        if self._output_path:
            self._result_plot.set_save_dir(path / "spectra" / "results" / "plots")
            self._overview_plot.set_save_dir(path / "spectra" / "results" / "plots")

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
        self._pss_auto_fb_cb.setChecked(p.pss_auto_fb)
        self._pss_obs_wl_spin.setValue(p.pss_obs_wavelength_nm)

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
        prefs.spectra.pss_auto_fb             = self._pss_auto_fb_cb.isChecked()
        prefs.spectra.pss_obs_wavelength_nm   = self._pss_obs_wl_spin.value()
