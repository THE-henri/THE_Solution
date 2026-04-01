"""
Half-Life tab  –  Kinetics and Scanning Kinetics sub-panels.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox,
    QScrollArea, QCheckBox, QFileDialog, QLineEdit,
    QSizePolicy, QFrame, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from gui.worker import Worker
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.widgets.info_button import InfoButton
from gui.widgets.master_csv_table import MasterCsvTable
from gui.tabs.half_life_core import (
    detect_time_unit, load_kinetics_csv,
    load_scanning_kinetics_csv, load_reference_spectrum,
    run_half_life_fit, run_scanning_fit, FitResult,
    detect_time_window, find_thermal_segments,
)
from core.plotting import plot_half_life_with_linear

# ── Colour palette for multi-channel plots ─────────────────────────────────
_CHANNEL_COLORS = [
    "#5b8dee", "#e8a020", "#3cb371", "#e84d4d",
    "#a06de0", "#40c8c8", "#e06080", "#a0c040",
]


def _make_label_row(label_text: str, info_title: str = "",
                    info_text: str = "") -> tuple[QHBoxLayout, QLabel]:
    """Helper: label + optional ℹ button in a row, returns (layout, label)."""
    row = QHBoxLayout()
    row.setSpacing(4)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(200)
    row.addWidget(lbl)
    if info_title:
        row.addWidget(InfoButton(info_title, info_text))
    return row, lbl


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setObjectName("card_separator")
    return f


# ═══════════════════════════════════════════════════════════════════════════
# Kinetics panel
# ═══════════════════════════════════════════════════════════════════════════

class KineticsPanel(QWidget):
    """Five-stage pipeline for multi-wavelength kinetics."""

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channels:   dict  = {}      # {label: (time_s, abs)}
        self._time_unit:  str   = "sec"
        self._fit_done:   bool  = False
        self._worker:     Worker | None = None
        self._output_path: Path | None  = None
        self._raw_path:    Path | None  = None
        self._last_results: list = []     # FitResult objects from last fit run
        self._segments:   list  = []      # [(t_start, t_end), ...]
        self._seg_idx:    int   = 0
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
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(12, 12, 12, 12)
        self._main_layout.setSpacing(0)
        scroll.setWidget(container)

        self._card1 = self._build_card1()
        self._card2 = self._build_card2()
        self._card3 = self._build_card3()
        self._card4 = self._build_card4()
        self._card5 = self._build_card5()

        for card in (self._card1, self._card2, self._card3,
                     self._card4, self._card5):
            self._main_layout.addWidget(card)

        self._main_layout.addStretch()

        # Initial state
        for card in (self._card2, self._card3, self._card4, self._card5):
            card.set_card_enabled(False)

    # ── Stage 1: Load data ─────────────────────────────────────────────────

    def _build_card1(self) -> StageCard:
        card = StageCard("Stage 1 — Load data")
        card.set_status(READY)

        # File row
        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        file_row.addWidget(QLabel("File:"))
        self._k_file_edit = QLineEdit()
        self._k_file_edit.setPlaceholderText("Select kinetics CSV…")
        self._k_file_edit.setReadOnly(True)
        file_row.addWidget(self._k_file_edit)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(72)
        btn_browse.clicked.connect(self._k_browse_file)
        file_row.addWidget(btn_browse)
        info = InfoButton(
            "Kinetics CSV format",
            "Multi-wavelength CSV (Cary 60 style):\n"
            "  Row 0 : channel labels (e.g. 45C_672)\n"
            "  Row 1 : 'Time (sec)' / 'Time (min)', 'Abs'\n"
            "  Row 2+: time / absorbance pairs\n\n"
            "Time unit is auto-detected from row 1. "
            "All internal calculations use seconds.",
        )
        file_row.addWidget(info)
        card.add_layout(file_row)

        # Unit detection label
        self._k_unit_lbl = QLabel("")
        self._k_unit_lbl.setObjectName("detected_label")
        card.add_widget(self._k_unit_lbl)

        # Raw trace plot
        self._k_raw_plot = PlotWidget(
            info_title="Raw kinetic traces",
            info_text=(
                "All wavelength channels from the loaded file, each in a "
                "different colour. Toggle channels off in Stage 2 to exclude "
                "them from fitting. The orange shaded region shows the current "
                "time window (set in Stage 3). Look for clean exponential "
                "decay or rise before proceeding."
            ),
            min_height=300,
        )
        card.add_widget(self._k_raw_plot)

        # ── Publication mode ───────────────────────────────────────────────
        pub_row = QHBoxLayout()
        self._k_pub_chk = QCheckBox("Save for publication")
        self._k_pub_chk.setToolTip(
            "When enabled, manual save buttons appear to export data\n"
            "into a publication/ subfolder for the Publication Composer.")
        self._k_pub_chk.toggled.connect(self._k_pub_mode_changed)
        pub_row.addWidget(self._k_pub_chk)
        pub_row.addStretch()
        card.add_layout(pub_row)

        return card

    # ── Stage 2: Channel selection ─────────────────────────────────────────

    def _build_card2(self) -> StageCard:
        card = StageCard("Stage 2 — Channel selection")

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Active channels:"))
        info_row.addWidget(InfoButton(
            "Channel selection",
            "Deselect channels that have poor signal, artefacts, or should "
            "not be included in this fitting run. Deselected channels are "
            "hidden in the plot and skipped during fitting.",
        ))
        info_row.addStretch()
        card.add_layout(info_row)

        self._k_channel_widget = QWidget()
        self._k_channel_layout = QVBoxLayout(self._k_channel_widget)
        self._k_channel_layout.setContentsMargins(0, 0, 0, 0)
        self._k_channel_layout.setSpacing(4)
        card.add_widget(self._k_channel_widget)

        self._k_checkboxes: dict[str, QCheckBox] = {}

        return card

    # ── Stage 3: Time window ───────────────────────────────────────────────

    def _build_card3(self) -> StageCard:
        card = StageCard("Stage 3 — Time window")

        # ── Photoswitch direction (moved here from Stage 4) ────────────────
        sw_row = QHBoxLayout()
        sw_lbl = QLabel("Photoswitch direction:")
        sw_lbl.setObjectName("pref_label")
        sw_row.addWidget(sw_lbl)
        sw_row.addWidget(InfoButton(
            "Photoswitch direction",
            "Negative photochromic photoswitch (build-up):\n"
            "  Irradiation DECREASES absorbance at the monitoring wavelength.\n"
            "  Thermal relaxation causes absorbance to BUILD UP back to baseline.\n"
            "  Fits A(t) = A∞ + (A₀−A∞)·e^(−kt)  [A₀ < A∞].\n\n"
            "Positive photochromic photoswitch (decay):\n"
            "  Irradiation INCREASES absorbance at the monitoring wavelength.\n"
            "  Thermal relaxation causes absorbance to DECAY back to baseline.\n"
            "  Fits A(t) = A₀·e^(−kt).\n\n"
            "The selection also controls the direction used by Auto-detect.",
        ))
        self._k_sw_neg = QRadioButton("Negative photochromic photoswitch (build-up)")
        self._k_sw_neg.setObjectName("pref_rb")
        self._k_sw_pos = QRadioButton("Positive photochromic photoswitch (decay)")
        self._k_sw_pos.setObjectName("pref_rb")
        self._k_sw_neg.setChecked(True)
        bg_sw = QButtonGroup(self)
        bg_sw.addButton(self._k_sw_neg)
        bg_sw.addButton(self._k_sw_pos)
        sw_row.addWidget(self._k_sw_neg)
        sw_row.addWidget(self._k_sw_pos)
        sw_row.addStretch()
        card.add_layout(sw_row)
        self._k_sw_neg.toggled.connect(self._k_mark_stale)

        # ── Auto-detect ────────────────────────────────────────────────────
        auto_row = QHBoxLayout()
        btn_auto = QPushButton("Auto-detect")
        btn_auto.setFixedWidth(110)
        btn_auto.clicked.connect(self._k_auto_detect_window)
        btn_auto.setToolTip(
            "Analyse the selected channel and detect where thermal relaxation begins.\n"
            "Multiple cycles are split into separate segments — use the\n"
            "← / → buttons to switch between them.")
        auto_row.addWidget(btn_auto)
        auto_row.addWidget(InfoButton(
            "Auto-detect window",
            "Detects irradiation events (peak for negative photochromic / valley "
            "for positive photochromic) and extracts each subsequent thermal-recovery "
            "phase as a separate fitting segment.\n\n"
            "Choose the channel to analyse from the drop-down. "
            "'Auto' picks the channel with the largest absorbance range.\n\n"
            "Use ← / → to cycle between segments if multiple were found.",
        ))

        # Channel selector for auto-detect
        self._k_detect_ch_combo = QComboBox()
        self._k_detect_ch_combo.setFixedWidth(160)
        self._k_detect_ch_combo.setToolTip(
            "Channel used for segment detection. 'Auto' picks the channel "
            "with the largest absorbance range.")
        self._k_detect_ch_combo.addItem("Auto (max Δabs)")
        auto_row.addWidget(self._k_detect_ch_combo)

        self._k_detect_lbl = QLabel("")
        self._k_detect_lbl.setObjectName("detected_label")
        auto_row.addWidget(self._k_detect_lbl)
        auto_row.addStretch()
        card.add_layout(auto_row)

        # Segment navigation row (hidden until >1 segment found)
        self._k_seg_nav = QWidget()
        seg_nav_l = QHBoxLayout(self._k_seg_nav)
        seg_nav_l.setContentsMargins(0, 0, 0, 0)
        seg_nav_l.setSpacing(6)
        self._k_seg_prev_btn = QPushButton("← Prev")
        self._k_seg_prev_btn.setFixedWidth(70)
        self._k_seg_prev_btn.clicked.connect(self._k_seg_prev)
        self._k_seg_next_btn = QPushButton("Next →")
        self._k_seg_next_btn.setFixedWidth(70)
        self._k_seg_next_btn.clicked.connect(self._k_seg_next)
        self._k_seg_lbl = QLabel("")
        seg_nav_l.addWidget(self._k_seg_prev_btn)
        seg_nav_l.addWidget(self._k_seg_lbl)
        seg_nav_l.addWidget(self._k_seg_next_btn)
        seg_nav_l.addStretch()
        self._k_seg_nav.setVisible(False)
        card.add_widget(self._k_seg_nav)

        # ── Start / End spinboxes ──────────────────────────────────────────
        row, _ = _make_label_row(
            "Start time (s):",
            "Start time",
            "Beginning of the fitting window in seconds. Points before this "
            "time are shown in grey (excluded from fit).",
        )
        self._k_t_start = QDoubleSpinBox()
        self._k_t_start.setRange(0, 1e9)
        self._k_t_start.setDecimals(1)
        self._k_t_start.setSuffix(" s")
        self._k_t_start.valueChanged.connect(self._k_update_current_segment)
        self._k_t_start.valueChanged.connect(self._k_update_raw_plot)
        self._k_t_start.valueChanged.connect(self._k_mark_stale)
        row.addWidget(self._k_t_start)
        row.addStretch()
        card.add_layout(row)

        row2, _ = _make_label_row(
            "End time (s):",
            "End time",
            "End of the fitting window in seconds. Points after this time "
            "are shown in grey and excluded from fit.",
        )
        self._k_t_end = QDoubleSpinBox()
        self._k_t_end.setRange(0, 1e9)
        self._k_t_end.setDecimals(1)
        self._k_t_end.setSuffix(" s")
        self._k_t_end.valueChanged.connect(self._k_update_current_segment)
        self._k_t_end.valueChanged.connect(self._k_update_raw_plot)
        self._k_t_end.valueChanged.connect(self._k_mark_stale)
        row2.addWidget(self._k_t_end)
        row2.addStretch()
        card.add_layout(row2)

        # ── Manual segments ────────────────────────────────────────────────
        card.add_widget(_sep())
        man_row = QHBoxLayout()
        man_lbl = QLabel("Manual segments:")
        man_lbl.setObjectName("pref_label")
        man_row.addWidget(man_lbl)
        man_row.addWidget(InfoButton(
            "Manual segments",
            "Divide the time range into N equal segments to fit separately.\n\n"
            "After setting the count the current Start / End values are split "
            "evenly. Use ← / → to navigate between segments and adjust each "
            "one's Start / End times — edits are stored per segment.\n\n"
            "All segments are shown simultaneously in the raw-data plot above: "
            "the active segment is highlighted, others are shown faintly.",
        ))
        self._k_n_manual_segs = QSpinBox()
        self._k_n_manual_segs.setRange(1, 20)
        self._k_n_manual_segs.setValue(1)
        self._k_n_manual_segs.setFixedWidth(60)
        self._k_n_manual_segs.setToolTip("Number of fitting segments (1 = single window)")
        self._k_n_manual_segs.valueChanged.connect(self._k_manual_segments_init)
        man_row.addWidget(self._k_n_manual_segs)
        man_row.addStretch()
        card.add_layout(man_row)

        # Publication save button (hidden until pub mode on)
        self._k_pub_save_raw_btn = QPushButton("Save raw + segments for publication")
        self._k_pub_save_raw_btn.setVisible(False)
        self._k_pub_save_raw_btn.clicked.connect(self._k_save_pub_raw)
        card.add_widget(self._k_pub_save_raw_btn)

        return card

    def _k_auto_detect_window(self):
        """Run segment detection on the selected (or best) channel."""
        if not self._channels:
            return
        active = [l for l, cb in self._k_checkboxes.items() if cb.isChecked()]
        if not active:
            return

        # Determine which channel to use
        chosen = self._k_detect_ch_combo.currentText()
        if chosen == "Auto (max Δabs)" or chosen not in self._channels:
            best = max(active, key=lambda l: float(np.ptp(self._channels[l][1])))
        else:
            best = chosen if chosen in active else max(
                active, key=lambda l: float(np.ptp(self._channels[l][1])))

        switch = "negative" if self._k_sw_neg.isChecked() else "positive"
        time, absorbance = self._channels[best]

        segs = find_thermal_segments(time, absorbance, switch=switch)
        if not segs:
            self._k_detect_lbl.setText("No segments detected")
            self._k_detect_lbl.setObjectName("warning_label")
            return

        self._segments = segs
        self._seg_idx  = 0
        self._k_apply_segment(0)
        n = len(segs)
        self._k_detect_lbl.setText(
            f"Found {n} segment{'s' if n > 1 else ''}  ·  channel: {best}")
        self._k_detect_lbl.setObjectName("detected_label")
        self._k_seg_nav.setVisible(n > 1)

    def _k_apply_segment(self, idx: int):
        t_start, t_end = self._segments[idx]
        self._k_t_start.blockSignals(True)
        self._k_t_end.blockSignals(True)
        self._k_t_start.setValue(t_start)
        self._k_t_end.setValue(t_end)
        self._k_t_start.blockSignals(False)
        self._k_t_end.blockSignals(False)
        n = len(self._segments)
        self._k_seg_lbl.setText(f"Segment {idx + 1} / {n}  "
                                 f"({t_start:.1f} – {t_end:.1f} s)")
        self._k_seg_prev_btn.setEnabled(idx > 0)
        self._k_seg_next_btn.setEnabled(idx < n - 1)
        self._k_mark_stale()
        self._k_update_raw_plot()

    def _k_seg_prev(self):
        if self._seg_idx > 0:
            self._seg_idx -= 1
            self._k_apply_segment(self._seg_idx)

    def _k_seg_next(self):
        if self._seg_idx < len(self._segments) - 1:
            self._seg_idx += 1
            self._k_apply_segment(self._seg_idx)

    def _k_manual_segments_init(self):
        """Split current Start–End range into N equal segments."""
        n = self._k_n_manual_segs.value()
        t0 = self._k_t_start.value()
        t1 = self._k_t_end.value()
        if t1 <= t0 or not self._channels:
            return
        step = (t1 - t0) / n
        self._segments = [(t0 + i * step, t0 + (i + 1) * step) for i in range(n)]
        self._seg_idx = 0
        # Clear any auto-detect status text to avoid confusion
        self._k_detect_lbl.setText("")
        self._k_seg_nav.setVisible(n > 1)
        self._k_apply_segment(0)

    def _k_update_current_segment(self):
        """Sync spinbox values back into the active segment entry."""
        if not self._segments or not (0 <= self._seg_idx < len(self._segments)):
            return
        t_start = self._k_t_start.value()
        t_end   = self._k_t_end.value()
        self._segments[self._seg_idx] = (t_start, t_end)
        self._k_seg_lbl.setText(
            f"Segment {self._seg_idx + 1} / {len(self._segments)}  "
            f"({t_start:.1f} – {t_end:.1f} s)"
        )

    def _k_seg_label(self) -> str:
        """Return e.g. 'Seg1 (0–120 s)' for the current segment."""
        if self._segments and 0 <= self._seg_idx < len(self._segments):
            t0, t1 = self._segments[self._seg_idx]
            return f"Seg{self._seg_idx + 1} ({t0:.0f}–{t1:.0f} s)"
        return "–"

    # ── Publication helpers ────────────────────────────────────────────────

    def _k_pub_folder(self) -> "Path | None":
        if self._output_path is None:
            return None
        return self._output_path / "half_life" / "results" / "publication"

    def _k_pub_mode_changed(self, checked: bool):
        self._k_pub_save_raw_btn.setVisible(checked)
        self._k_pub_save_seg_btn.setVisible(checked)
        if checked:
            pub = self._k_pub_folder()
            if pub is not None:
                pub.mkdir(parents=True, exist_ok=True)

    def _k_save_pub_raw(self):
        """Save raw_data.csv and segments.csv into publication/raw/."""
        import pandas as pd
        pub = self._k_pub_folder()
        if pub is None:
            return
        active = [lbl for lbl, cb in self._k_checkboxes.items() if cb.isChecked()]
        if not active or not self._channels:
            return
        raw_dir = pub / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # raw_data.csv
        data: dict = {}
        for label in active:
            time, absorbance = self._channels[label]
            if "time_s" not in data:
                data["time_s"] = time
            data[label] = absorbance
        pd.DataFrame(data).to_csv(raw_dir / "raw_data.csv", index=False)
        # segments.csv
        if self._segments:
            rows = [
                {"segment": i + 1, "t_start_s": s0, "t_end_s": s1,
                 "label": f"Seg{i+1} ({s0:.0f}–{s1:.0f} s)"}
                for i, (s0, s1) in enumerate(self._segments)
            ]
            pd.DataFrame(rows).to_csv(raw_dir / "segments.csv", index=False)
        print(f"Publication raw data → {raw_dir}")

    def _k_save_pub_segment(self):
        """Save data_points.csv, fit_line.csv, fit_params.csv for current segment."""
        import pandas as pd
        pub = self._k_pub_folder()
        if pub is None or not self._last_results:
            return
        n = self._seg_idx + 1
        seg_dir = pub / f"segment_{n}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        data_cols: dict = {}
        fit_cols: dict  = {}
        params_rows: list = []
        for r in self._last_results:
            if not r.success:
                continue
            popt   = r.popt
            switch = r.switch
            if switch == "negative":
                a_inf = float(popt[1]); a0 = float(popt[0]); k = float(popt[2])
                ln_data = np.log(np.maximum(r.abs_clean - a_inf, 1e-20))
                t_dense = np.linspace(r.time_clean[0], r.time_clean[-1], 300)
                ln_fit  = np.log(abs(a0 - a_inf)) - k * (t_dense - r.time_clean[0])
            else:
                a0 = float(popt[0]); k = float(popt[1]); a_inf = 0.0
                ln_data = np.log(np.maximum(r.abs_clean, 1e-20))
                t_dense = np.linspace(r.time_clean[0], r.time_clean[-1], 300)
                ln_fit  = np.log(abs(a0)) - k * (t_dense - r.time_clean[0])
            if "time_s" not in data_cols:
                data_cols["time_s"] = r.time_clean
            if "time_s" not in fit_cols:
                fit_cols["time_s"] = t_dense
            data_cols[f"{r.label}_abs"]  = r.abs_clean
            data_cols[f"{r.label}_ln_A"] = ln_data
            fit_cols[f"{r.label}_fit"]   = ln_fit
            params_rows.append({
                "channel": r.label, "k_s-1": k, "t_half_s": r.t_half,
                "A0": a0, "A_inf": a_inf, "R2": r.r2,
            })
        if data_cols:
            pd.DataFrame(data_cols).to_csv(seg_dir / "data_points.csv", index=False)
        if fit_cols:
            pd.DataFrame(fit_cols).to_csv(seg_dir / "fit_line.csv", index=False)
        if params_rows:
            pd.DataFrame(params_rows).to_csv(seg_dir / "fit_params.csv", index=False)
        print(f"Publication segment {n} → {seg_dir}")

    # ── Stage 4: Fit parameters ────────────────────────────────────────────

    def _build_card4(self) -> StageCard:
        card = StageCard("Stage 4 — Fit parameters")

        # A∞
        ainf_row = QHBoxLayout()
        ainf_lbl = QLabel("A∞ (asymptote):")
        ainf_lbl.setObjectName("pref_label")
        ainf_row.addWidget(ainf_lbl)
        ainf_row.addWidget(InfoButton(
            "A∞ — asymptote",
            "The absorbance at infinite time (thermal equilibrium).\n\n"
            "'Fit freely': A∞ is a free parameter — use when the plateau is "
            "visible in the data.\n\n"
            "'Fix to value': A∞ is held fixed — use when the decay doesn't "
            "reach the plateau within the measurement window, or when you know "
            "the equilibrium value (e.g. 0 for full bleaching).",
        ))
        self._k_ainf_free  = QRadioButton("Fit freely")
        self._k_ainf_fixed = QRadioButton("Fix to:")
        self._k_ainf_free.setChecked(True)
        bg2 = QButtonGroup(self)
        bg2.addButton(self._k_ainf_free)
        bg2.addButton(self._k_ainf_fixed)
        self._k_ainf_val = QDoubleSpinBox()
        self._k_ainf_val.setRange(-10, 10)
        self._k_ainf_val.setDecimals(4)
        self._k_ainf_val.setValue(0.0)
        self._k_ainf_val.setFixedWidth(90)
        self._k_ainf_free.toggled.connect(
            lambda on: self._k_ainf_val.setEnabled(not on))
        self._k_ainf_val.setEnabled(False)
        ainf_row.addWidget(self._k_ainf_free)
        ainf_row.addWidget(self._k_ainf_fixed)
        ainf_row.addWidget(self._k_ainf_val)
        ainf_row.addStretch()
        card.add_layout(ainf_row)
        self._k_ainf_free.toggled.connect(self._k_mark_stale)
        self._k_ainf_val.valueChanged.connect(self._k_mark_stale)

        card.add_widget(_sep())

        # IQR factor
        iqr_row, iqr_lbl = _make_label_row(
            "Outlier IQR factor:",
            "Outlier IQR factor",
            "Points whose residual from the initial fit exceeds "
            "factor × IQR are excluded before the final fit.\n\n"
            "Typical values: 3 (strict) to 50 (very lenient).\n"
            "Watch out: very low values may remove real data near the "
            "end of the decay.",
        )
        self._k_iqr = QDoubleSpinBox()
        self._k_iqr.setRange(0.1, 1000)
        self._k_iqr.setDecimals(1)
        self._k_iqr.setValue(3.0)
        self._k_iqr.setFixedWidth(80)
        iqr_lbl.setObjectName("pref_label")
        iqr_row.addWidget(self._k_iqr)
        iqr_row.addStretch()
        card.add_layout(iqr_row)
        self._k_iqr.valueChanged.connect(self._k_mark_stale)

        card.add_widget(_sep())

        # Temperature
        temp_row, _ = _make_label_row(
            "Temperature (°C):",
            "Temperature",
            "Measurement temperature. Stored in the master CSV and used "
            "for Arrhenius / Eyring analysis.",
        )
        self._k_temp = QDoubleSpinBox()
        self._k_temp.setRange(-100, 500)
        self._k_temp.setDecimals(1)
        self._k_temp.setValue(25.0)
        self._k_temp.setFixedWidth(80)
        temp_row.addWidget(self._k_temp)
        temp_row.addStretch()
        card.add_layout(temp_row)
        self._k_temp.valueChanged.connect(self._k_mark_stale)

        return card

    # ── Stage 5: Fit & Results ─────────────────────────────────────────────

    def _build_card5(self) -> StageCard:
        card = StageCard("Stage 5 — Fit & Results")

        # Run button
        run_row = QHBoxLayout()
        self._k_run_btn = QPushButton("▶  Run Fit")
        self._k_run_btn.setObjectName("run_btn")
        self._k_run_btn.setFixedHeight(34)
        self._k_run_btn.clicked.connect(self._k_run_fit)
        run_row.addWidget(self._k_run_btn)
        self._k_pub_save_seg_btn = QPushButton("Save segment fit for publication")
        self._k_pub_save_seg_btn.setVisible(False)
        self._k_pub_save_seg_btn.setEnabled(False)
        self._k_pub_save_seg_btn.clicked.connect(self._k_save_pub_segment)
        run_row.addWidget(self._k_pub_save_seg_btn)
        run_row.addStretch()
        card.add_layout(run_row)

        # Stale banner (hidden until stale)
        self._k_stale_banner = QLabel(
            "⚠  Parameters changed — re-run the fit to update results.")
        self._k_stale_banner.setObjectName("stale_banner")
        self._k_stale_banner.setVisible(False)
        card.add_widget(self._k_stale_banner)

        # Plot area — will be populated per channel after fit
        self._k_plot_area = QWidget()
        self._k_plot_layout = QVBoxLayout(self._k_plot_area)
        self._k_plot_layout.setContentsMargins(0, 0, 0, 0)
        self._k_plot_layout.setSpacing(8)
        card.add_widget(self._k_plot_area)

        # Master CSV table
        card.add_widget(_sep())
        self._k_master = MasterCsvTable()
        card.add_widget(self._k_master)

        return card

    # ── Stage 1 actions ────────────────────────────────────────────────────

    def _k_browse_file(self):
        start = str(self._raw_path) if self._raw_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select kinetics CSV", start, "CSV files (*.csv)")
        if not path:
            return
        self._k_file_edit.setText(path)
        self._k_load_file(Path(path))

    def _k_load_file(self, path: Path):
        try:
            self._time_unit = detect_time_unit(path)
            self._channels  = load_kinetics_csv(path, convert_to_seconds=True)
        except Exception as exc:
            self._card1.set_status(ERROR)
            self._k_unit_lbl.setText(f"Error: {exc}")
            self._k_unit_lbl.setObjectName("error_label")
            return

        unit_str = "minutes (converted to seconds)" if self._time_unit == "min" else "seconds"
        self._k_unit_lbl.setText(f"Detected time unit: {unit_str}")
        self._k_unit_lbl.setObjectName("detected_label")
        self._card1.set_status(DONE)

        # Populate channel checkboxes
        self._k_populate_channels()

        # Set time range defaults
        all_times = np.concatenate([t for t, _ in self._channels.values()])
        t_min, t_max = float(all_times.min()), float(all_times.max())
        self._k_t_start.blockSignals(True)
        self._k_t_end.blockSignals(True)
        self._k_t_start.setRange(t_min, t_max)
        self._k_t_end.setRange(t_min, t_max)
        self._k_t_start.setValue(t_min)
        self._k_t_end.setValue(t_max)
        self._k_t_start.blockSignals(False)
        self._k_t_end.blockSignals(False)

        # Enable downstream cards
        for card in (self._card2, self._card3, self._card4, self._card5):
            card.set_card_enabled(True)
            card.set_status(READY)

        self._fit_done = False
        self._k_update_raw_plot()
        print(f"Loaded {len(self._channels)} channels from {path.name}")

    def _k_populate_channels(self):
        # Clear old checkboxes
        while self._k_channel_layout.count():
            item = self._k_channel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._k_checkboxes.clear()

        # Repopulate the auto-detect channel selector
        self._k_detect_ch_combo.blockSignals(True)
        self._k_detect_ch_combo.clear()
        self._k_detect_ch_combo.addItem("Auto (max Δabs)")
        for label in self._channels:
            self._k_detect_ch_combo.addItem(label)
        self._k_detect_ch_combo.blockSignals(False)

        for label in self._channels:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._k_update_raw_plot)
            cb.toggled.connect(self._k_mark_stale)
            self._k_checkboxes[label] = cb
            self._k_channel_layout.addWidget(cb)

    def _k_update_raw_plot(self):
        if not self._channels:
            return
        t_start = self._k_t_start.value()
        t_end   = self._k_t_end.value()

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        active = [lbl for lbl, cb in self._k_checkboxes.items() if cb.isChecked()]

        for i, (label, (time, absorbance)) in enumerate(self._channels.items()):
            color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            if label not in active:
                # Show inactive channels faintly in grey without a legend entry
                ax.plot(time, absorbance, "o", color="#555566", alpha=0.18,
                        markersize=2, markerfacecolor="none", markeredgewidth=0.5)
            else:
                ax.plot(time, absorbance, "o", color=color, alpha=0.85,
                        markersize=3, markerfacecolor="none",
                        markeredgewidth=0.8, label=label)

        # Draw segment highlights
        if len(self._segments) > 1:
            for i, (s0, s1) in enumerate(self._segments):
                if s1 <= s0:
                    continue
                is_active = (i == self._seg_idx)
                ax.axvspan(s0, s1, color="#e8a020",
                           alpha=0.22 if is_active else 0.07,
                           label="Active segment" if is_active else None)
                ax.text((s0 + s1) / 2, 1.0, str(i + 1),
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8,
                        color="#b06010" if is_active else "#c09040",
                        fontweight="bold" if is_active else "normal")
        elif self._segments:
            s0, s1 = self._segments[0]
            if s1 > s0:
                ax.axvspan(s0, s1, color="#e8a020", alpha=0.12, label="Selection window")
        elif t_end > t_start:
            ax.axvspan(t_start, t_end, color="#e8a020", alpha=0.12, label="Selection window")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absorbance")
        ax.set_title("Raw kinetic traces")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._k_raw_plot.set_figure(fig)
        plt.close(fig)

    # ── Stage 5 actions ────────────────────────────────────────────────────

    def _k_mark_stale(self):
        if self._fit_done:
            self._card5.set_status(STALE)
            self._k_stale_banner.setVisible(True)

    def _k_run_fit(self):
        active = [lbl for lbl, cb in self._k_checkboxes.items() if cb.isChecked()]
        if not active:
            return

        switch      = "negative" if self._k_sw_neg.isChecked() else "positive"
        a_inf_mode  = "free" if self._k_ainf_free.isChecked() else "fixed"
        a_inf_value = self._k_ainf_val.value() if a_inf_mode == "fixed" else None
        iqr         = self._k_iqr.value()
        temp        = self._k_temp.value()
        t_start     = self._k_t_start.value()
        t_end       = self._k_t_end.value()

        channels_subset = {l: self._channels[l] for l in active}

        self._k_run_btn.setEnabled(False)
        self._card5.set_status(WAITING)

        def _fit_all():
            results = []
            for label, (time, absorbance) in channels_subset.items():
                print(f"\nFitting channel: {label}")
                r = run_half_life_fit(
                    label=label, time=time, absorbance=absorbance,
                    t_start_s=t_start, t_end_s=t_end,
                    switch=switch, a_inf_mode=a_inf_mode,
                    a_inf_value=a_inf_value,
                    iqr_factor=iqr, temperature_c=temp,
                )
                results.append(r)
            return results

        self._worker = Worker(_fit_all)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._k_on_fit_done)
        self._worker.error_signal.connect(self._k_on_fit_error)
        self._worker.finished_signal.connect(
            lambda: self._k_run_btn.setEnabled(True))
        self._worker.start()

    @pyqtSlot(object)
    def _k_on_fit_done(self, results: list[FitResult]):
        self._fit_done = True
        self._card5.set_status(DONE)
        self._k_stale_banner.setVisible(False)

        # Clear old plots
        while self._k_plot_layout.count():
            item = self._k_plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        file_stem = Path(self._k_file_edit.text()).stem
        result_entries = []

        for r in results:
            if not r.success:
                lbl = QLabel(f"  {r.label}: {r.error_msg}")
                lbl.setObjectName("error_label")
                self._k_plot_layout.addWidget(lbl)
                continue

            # Build figure using the existing core plotting function
            fig, _ = plot_half_life_with_linear(
                r.time_full, r.abs_full,
                start_idx=r.start_idx, end_idx=r.end_idx,
                time_sel=r.time_clean, absorbance_sel=r.abs_clean,
                time_outliers=r.time_outliers, absorbance_outliers=r.abs_outliers,
                fitted_curve=r.fitted_curve, r_squared=r.r2,
                popt=r.popt, t_half=r.t_half, switch=r.switch,
                title=f"{file_stem} | {r.label} | T={r.temperature_c:.0f}°C",
                show=False,
            )
            pw = PlotWidget(
                info_title="Fit plot",
                info_text=(
                    "Left panel: exponential fit overlay.\n"
                    "  Grey circles = full trace\n"
                    "  Orange circles = selected inliers\n"
                    "  Red × = outliers\n"
                    "  Red dashed = fit\n\n"
                    "Right panel: linearised transform (ln|A−A∞| vs time). "
                    "Should be linear for a first-order process. "
                    "Deviations indicate non-first-order kinetics or a "
                    "poor selection window."
                ),
                min_height=300,
            )
            pw.set_save_dir(self._output_path / "half_life" / "results" / "plots"
                            if self._output_path else None)
            pw.set_default_filename(
                f"{file_stem}_{r.label}_{r.temperature_c:.0f}C.png"
                .replace(" ", "_").replace("/", "-"))
            pw.set_figure(fig)
            plt.close(fig)
            self._k_plot_layout.addWidget(pw)

            # Add result to master table
            switch_val = r.switch
            popt = r.popt
            result_entries.append({
                "File":          Path(self._k_file_edit.text()).name,
                "Segment":       self._k_seg_label(),
                "Wavelength":    r.label,
                "Type":          "Kinetics",
                "Temperature_C": r.temperature_c,
                "Switch":        switch_val,
                "A0":            popt[0] if popt is not None else None,
                "A_inf":         popt[1] if (popt is not None and switch_val == "negative") else None,
                "k":             popt[-1] if popt is not None else None,
                "Half_life_s":   r.t_half,
                "R2":            r.r2,
            })

        self._last_results = results
        if result_entries:
            self._k_master.add_pending(result_entries)
        if self._k_pub_chk.isChecked() and result_entries:
            self._k_pub_save_seg_btn.setEnabled(True)

    @pyqtSlot(str)
    def _k_on_fit_error(self, msg: str):
        self._card5.set_status(ERROR)
        self.log_signal.emit(f"Fit error: {msg}", "ERROR")

    # ── Path relays ────────────────────────────────────────────────────────

    # ── Preferences ────────────────────────────────────────────────────────

    def apply_prefs(self, prefs):
        """Populate preference widgets from a HalfLifeKineticsPrefs object."""
        if prefs.switch == "positive":
            self._k_sw_pos.setChecked(True)
        else:
            self._k_sw_neg.setChecked(True)
        if prefs.a_inf_mode == "fixed":
            self._k_ainf_fixed.setChecked(True)
        else:
            self._k_ainf_free.setChecked(True)
        self._k_ainf_val.setValue(prefs.a_inf_value)
        self._k_iqr.setValue(prefs.iqr_factor)
        self._k_temp.setValue(prefs.temperature_c)

    def collect_prefs(self, prefs):
        """Write current widget values into a HalfLifeKineticsPrefs object."""
        prefs.switch       = "positive" if self._k_sw_pos.isChecked() else "negative"
        prefs.a_inf_mode   = "fixed" if self._k_ainf_fixed.isChecked() else "free"
        prefs.a_inf_value  = self._k_ainf_val.value()
        prefs.iqr_factor   = self._k_iqr.value()
        prefs.temperature_c = self._k_temp.value()

    def set_raw_path(self, path: Path):
        self._raw_path = path

    def set_output_path(self, path: Path):
        self._output_path = path
        self._k_master.set_output_path(path)
        plots_dir = path / "half_life" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self._k_raw_plot.set_save_dir(plots_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Scanning Kinetics panel
# ═══════════════════════════════════════════════════════════════════════════

class ScanningKineticsPanel(QWidget):
    """Six-stage pipeline for scanning kinetics."""

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scans:          list  = []
        self._ref_scans:      list  = []
        self._n_scans:        int   = 0
        self._fit_done:       bool  = False
        self._worker:         Worker | None = None
        self._output_path:    Path | None   = None
        self._raw_path:       Path | None   = None
        self._wavelength_inputs: list[QDoubleSpinBox] = []
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
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(12, 12, 12, 12)
        self._main_layout.setSpacing(0)
        scroll.setWidget(container)

        self._card1 = self._build_sk_card1()
        self._card2 = self._build_sk_card2()
        self._card3 = self._build_sk_card3()
        self._card4 = self._build_sk_card4()
        self._card5 = self._build_sk_card5()
        self._card6 = self._build_sk_card6()

        for card in (self._card1, self._card2, self._card3,
                     self._card4, self._card5, self._card6):
            self._main_layout.addWidget(card)
        self._main_layout.addStretch()

        for card in (self._card2, self._card3, self._card4,
                     self._card5, self._card6):
            card.set_card_enabled(False)

    # ── Stage 1 ────────────────────────────────────────────────────────────

    def _build_sk_card1(self) -> StageCard:
        card = StageCard("Stage 1 — Load data")
        card.set_status(READY)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._sk_file_edit = QLineEdit()
        self._sk_file_edit.setPlaceholderText("Select scanning kinetics CSV…")
        self._sk_file_edit.setReadOnly(True)
        file_row.addWidget(self._sk_file_edit)
        btn = QPushButton("Browse")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._sk_browse_file)
        file_row.addWidget(btn)
        file_row.addWidget(InfoButton(
            "Scanning kinetics CSV format",
            "Column-pair CSV (Cary 60 style):\n"
            "  Row 0 : scan labels\n"
            "  Row 1 : 'Wavelength (nm)', 'Abs' repeated per scan\n"
            "  Row 2+: wavelength / absorbance data\n\n"
            "Each column pair is one full spectrum at one time point.",
        ))
        card.add_layout(file_row)

        # Time interval
        int_row, _ = _make_label_row(
            "Time interval:",
            "Time interval between scans",
            "Time elapsed between consecutive spectra. "
            "The time axis is built as: t = scan_index × interval.\n\n"
            "Select the unit matching your instrument settings.",
        )
        self._sk_interval = QDoubleSpinBox()
        self._sk_interval.setRange(0.1, 1e6)
        self._sk_interval.setDecimals(1)
        self._sk_interval.setValue(300.0)
        self._sk_interval.setFixedWidth(90)
        int_row.addWidget(self._sk_interval)

        self._sk_unit_min = QRadioButton("minutes")
        self._sk_unit_sec = QRadioButton("seconds")
        self._sk_unit_sec.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._sk_unit_sec)
        bg.addButton(self._sk_unit_min)
        int_row.addWidget(self._sk_unit_sec)
        int_row.addWidget(self._sk_unit_min)
        int_row.addStretch()
        card.add_layout(int_row)
        self._sk_interval.valueChanged.connect(self._sk_mark_stale)
        self._sk_unit_min.toggled.connect(self._sk_mark_stale)

        self._sk_scan_count_lbl = QLabel("")
        self._sk_scan_count_lbl.setObjectName("detected_label")
        card.add_widget(self._sk_scan_count_lbl)

        # Waterfall plot
        self._sk_raw_plot = PlotWidget(
            info_title="Spectral waterfall",
            info_text=(
                "All scans overlaid, coloured from early (blue) to late (red). "
                "Look for consistent spectral shifts or amplitude changes at "
                "your target wavelengths over time."
            ),
            min_height=280,
        )
        card.add_widget(self._sk_raw_plot)

        return card

    # ── Stage 2: Target wavelengths ────────────────────────────────────────

    def _build_sk_card2(self) -> StageCard:
        card = StageCard("Stage 2 — Target wavelengths")

        card.add_widget(InfoButton(
            "Target wavelengths",
            "Absorbance at each target wavelength is extracted as the mean "
            "of all data points within ±tolerance nm. The resulting kinetic "
            "trace is fitted with an exponential model.",
        ))

        self._sk_wl_container = QWidget()
        self._sk_wl_layout = QVBoxLayout(self._sk_wl_container)
        self._sk_wl_layout.setContentsMargins(0, 0, 0, 0)
        self._sk_wl_layout.setSpacing(4)
        card.add_widget(self._sk_wl_container)

        for nm in (359.0, 539.0, 672.0):
            self._sk_add_wl_row(nm)

        add_row = QHBoxLayout()
        btn_add = QPushButton("+ Add wavelength")
        btn_add.setFixedWidth(150)
        btn_add.clicked.connect(lambda: self._sk_add_wl_row(500.0))
        add_row.addWidget(btn_add)
        add_row.addStretch()
        card.add_layout(add_row)

        tol_row, _ = _make_label_row(
            "Tolerance (nm):",
            "Wavelength tolerance",
            "Mean absorbance is computed over [target ± tolerance] nm. "
            "Increase if spectra are sparsely sampled.",
        )
        self._sk_tol = QDoubleSpinBox()
        self._sk_tol.setRange(0.1, 20)
        self._sk_tol.setDecimals(1)
        self._sk_tol.setValue(1.0)
        self._sk_tol.setFixedWidth(70)
        tol_row.addWidget(self._sk_tol)
        tol_row.addStretch()
        card.add_layout(tol_row)
        self._sk_tol.valueChanged.connect(self._sk_mark_stale)

        return card

    def _sk_add_wl_row(self, default_nm: float):
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        sb = QDoubleSpinBox()
        sb.setRange(200, 1100)
        sb.setDecimals(1)
        sb.setValue(default_nm)
        sb.setSuffix(" nm")
        sb.setFixedWidth(100)
        sb.valueChanged.connect(self._sk_mark_stale)
        self._wavelength_inputs.append(sb)
        row_l.addWidget(sb)
        btn_del = QPushButton("✕")
        btn_del.setObjectName("delete_row_btn")
        btn_del.setFixedSize(22, 22)
        btn_del.clicked.connect(lambda: self._sk_remove_wl_row(row_w, sb))
        row_l.addWidget(btn_del)
        row_l.addStretch()
        self._sk_wl_layout.addWidget(row_w)

    def _sk_remove_wl_row(self, row_w: QWidget, sb: QDoubleSpinBox):
        if sb in self._wavelength_inputs:
            self._wavelength_inputs.remove(sb)
        row_w.deleteLater()

    # ── Stage 3: Reference / A∞ ────────────────────────────────────────────

    def _build_sk_card3(self) -> StageCard:
        card = StageCard("Stage 3 — Reference spectrum  (A∞)")

        card.add_widget(InfoButton(
            "A∞ — asymptote",
            "A∞ is the absorbance at full thermal relaxation (equilibrium).\n\n"
            "'Use reference file': A∞ at each target wavelength is extracted "
            "from a spectrum of the fully relaxed sample.\n\n"
            "'Fix to value': A∞ is held at the given number for all wavelengths "
            "(use 0 if the compound fully bleaches).\n\n"
            "'Fit freely': A∞ is a free fit parameter.",
        ))

        self._sk_ref_file  = QRadioButton("Use reference file")
        self._sk_ref_fixed = QRadioButton("Fix A∞ to:")
        self._sk_ref_free  = QRadioButton("Fit freely")
        self._sk_ref_file.setChecked(True)
        bg = QButtonGroup(self)
        for r in (self._sk_ref_file, self._sk_ref_fixed, self._sk_ref_free):
            bg.addButton(r)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._sk_ref_file)
        self._sk_ref_ainf_val = QDoubleSpinBox()
        self._sk_ref_ainf_val.setRange(-10, 10)
        self._sk_ref_ainf_val.setDecimals(4)
        self._sk_ref_ainf_val.setValue(0.0)
        self._sk_ref_ainf_val.setFixedWidth(90)
        self._sk_ref_ainf_val.setEnabled(False)
        mode_row.addWidget(self._sk_ref_fixed)
        mode_row.addWidget(self._sk_ref_ainf_val)
        mode_row.addWidget(self._sk_ref_free)
        mode_row.addStretch()
        card.add_layout(mode_row)

        self._sk_ref_fixed.toggled.connect(
            lambda on: self._sk_ref_ainf_val.setEnabled(on))
        self._sk_ref_file.toggled.connect(self._sk_mark_stale)
        self._sk_ref_fixed.toggled.connect(self._sk_mark_stale)
        self._sk_ref_free.toggled.connect(self._sk_mark_stale)
        self._sk_ref_ainf_val.valueChanged.connect(self._sk_mark_stale)

        # Reference file picker (only shown when "Use reference file" selected)
        self._sk_ref_file_row = QWidget()
        rfl = QHBoxLayout(self._sk_ref_file_row)
        rfl.setContentsMargins(0, 0, 0, 0)
        rfl.addWidget(QLabel("Reference file:"))
        self._sk_ref_path_edit = QLineEdit()
        self._sk_ref_path_edit.setPlaceholderText("Select reference CSV…")
        self._sk_ref_path_edit.setReadOnly(True)
        rfl.addWidget(self._sk_ref_path_edit)
        btn_ref = QPushButton("Browse")
        btn_ref.setFixedWidth(72)
        btn_ref.clicked.connect(self._sk_browse_ref)
        rfl.addWidget(btn_ref)
        card.add_widget(self._sk_ref_file_row)

        self._sk_ref_file.toggled.connect(self._sk_ref_file_row.setVisible)

        return card

    def _sk_browse_ref(self):
        start = str(self._raw_path) if self._raw_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference spectrum CSV", start, "CSV files (*.csv)")
        if not path:
            return
        self._sk_ref_path_edit.setText(path)
        try:
            self._ref_scans = load_reference_spectrum(path)
            print(f"Reference spectrum loaded: {Path(path).name} "
                  f"({len(self._ref_scans)} scan(s))")
            self._sk_mark_stale()
        except Exception as exc:
            self._sk_ref_path_edit.setText(f"Error: {exc}")

    # ── Stage 4: Scan window ───────────────────────────────────────────────

    def _build_sk_card4(self) -> StageCard:
        card = StageCard("Stage 4 — Scan window")

        start_row, _ = _make_label_row(
            "Start scan:",
            "Start scan",
            "First scan index to include in the fit (0 = first scan). "
            "Equivalent time shown in parentheses.",
        )
        self._sk_scan_start = QSpinBox()
        self._sk_scan_start.setRange(0, 9999)
        self._sk_scan_start.setValue(0)
        self._sk_scan_start.setFixedWidth(80)
        self._sk_start_time_lbl = QLabel("= 0 s")
        self._sk_start_time_lbl.setStyleSheet("color:#888; font-size:9pt;")
        start_row.addWidget(self._sk_scan_start)
        start_row.addWidget(self._sk_start_time_lbl)
        start_row.addStretch()
        card.add_layout(start_row)
        self._sk_scan_start.valueChanged.connect(self._sk_update_time_labels)
        self._sk_scan_start.valueChanged.connect(self._sk_mark_stale)

        end_row, _ = _make_label_row(
            "End scan:",
            "End scan",
            "Last scan index to include. Set to 0 to use all scans.",
        )
        self._sk_scan_end = QSpinBox()
        self._sk_scan_end.setRange(0, 9999)
        self._sk_scan_end.setValue(0)
        self._sk_scan_end.setFixedWidth(80)
        self._sk_end_time_lbl = QLabel("= last")
        self._sk_end_time_lbl.setStyleSheet("color:#888; font-size:9pt;")
        end_row.addWidget(self._sk_scan_end)
        end_row.addWidget(self._sk_end_time_lbl)
        end_row.addStretch()
        card.add_layout(end_row)
        self._sk_scan_end.valueChanged.connect(self._sk_update_time_labels)
        self._sk_scan_end.valueChanged.connect(self._sk_mark_stale)

        return card

    def _sk_update_time_labels(self):
        interval_s = self._sk_get_interval_s()
        s = self._sk_scan_start.value()
        e = self._sk_scan_end.value()
        self._sk_start_time_lbl.setText(f"= {s * interval_s:.0f} s")
        self._sk_end_time_lbl.setText(
            f"= {e * interval_s:.0f} s" if e > 0 else "= last scan")

    def _sk_get_interval_s(self) -> float:
        v = self._sk_interval.value()
        return v * 60.0 if self._sk_unit_min.isChecked() else v

    # ── Stage 5: Fit parameters ────────────────────────────────────────────

    def _build_sk_card5(self) -> StageCard:
        card = StageCard("Stage 5 — Fit parameters")

        sw_row = QHBoxLayout()
        sw_row.addWidget(QLabel("Photoswitch direction:"))
        sw_row.addWidget(InfoButton(
            "Photoswitch direction",
            "Negative photochromic photoswitch (build-up): irradiation DECREASES absorbance; "
            "thermal relaxation causes absorbance to BUILD UP.\n"
            "Positive photochromic photoswitch (decay): irradiation INCREASES absorbance; "
            "thermal relaxation causes absorbance to DECAY.",
        ))
        self._sk_sw_neg = QRadioButton("Negative photochromic photoswitch (build-up)")
        self._sk_sw_neg.setObjectName("pref_rb")
        self._sk_sw_pos = QRadioButton("Positive photochromic photoswitch (decay)")
        self._sk_sw_pos.setObjectName("pref_rb")
        self._sk_sw_neg.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._sk_sw_neg)
        bg.addButton(self._sk_sw_pos)
        sw_row.addWidget(self._sk_sw_neg)
        sw_row.addWidget(self._sk_sw_pos)
        sw_row.addStretch()
        card.add_layout(sw_row)
        self._sk_sw_neg.toggled.connect(self._sk_mark_stale)

        card.add_widget(_sep())

        iqr_row, _ = _make_label_row(
            "Outlier IQR factor:",
            "Outlier IQR factor",
            "Points with |residual| > factor×IQR are removed before the "
            "final fit. Default 50 is lenient for scanning kinetics.",
        )
        self._sk_iqr = QDoubleSpinBox()
        self._sk_iqr.setRange(0.1, 1000)
        self._sk_iqr.setDecimals(1)
        self._sk_iqr.setValue(50.0)
        self._sk_iqr.setFixedWidth(80)
        iqr_row.addWidget(self._sk_iqr)
        iqr_row.addStretch()
        card.add_layout(iqr_row)
        self._sk_iqr.valueChanged.connect(self._sk_mark_stale)

        card.add_widget(_sep())

        temp_row, _ = _make_label_row("Temperature (°C):", "Temperature",
                                      "Stored in master CSV for thermal analysis.")
        self._sk_temp = QDoubleSpinBox()
        self._sk_temp.setRange(-100, 500)
        self._sk_temp.setDecimals(1)
        self._sk_temp.setValue(25.0)
        self._sk_temp.setFixedWidth(80)
        temp_row.addWidget(self._sk_temp)
        temp_row.addStretch()
        card.add_layout(temp_row)
        self._sk_temp.valueChanged.connect(self._sk_mark_stale)

        return card

    # ── Stage 6: Results ───────────────────────────────────────────────────

    def _build_sk_card6(self) -> StageCard:
        card = StageCard("Stage 6 — Fit & Results")

        run_row = QHBoxLayout()
        self._sk_run_btn = QPushButton("▶  Run Fit")
        self._sk_run_btn.setObjectName("run_btn")
        self._sk_run_btn.setFixedHeight(34)
        self._sk_run_btn.clicked.connect(self._sk_run_fit)
        run_row.addWidget(self._sk_run_btn)
        run_row.addStretch()
        card.add_layout(run_row)

        self._sk_stale_banner = QLabel(
            "⚠  Parameters changed — re-run the fit to update results.")
        self._sk_stale_banner.setObjectName("stale_banner")
        self._sk_stale_banner.setVisible(False)
        card.add_widget(self._sk_stale_banner)

        self._sk_plot_area   = QWidget()
        self._sk_plot_layout = QVBoxLayout(self._sk_plot_area)
        self._sk_plot_layout.setContentsMargins(0, 0, 0, 0)
        card.add_widget(self._sk_plot_area)

        card.add_widget(_sep())
        self._sk_master = MasterCsvTable()
        card.add_widget(self._sk_master)

        return card

    # ── Stage 1 actions ────────────────────────────────────────────────────

    def _sk_browse_file(self):
        start = str(self._raw_path) if self._raw_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select scanning kinetics CSV", start, "CSV files (*.csv)")
        if not path:
            return
        self._sk_file_edit.setText(path)
        self._sk_load_file(Path(path))

    def _sk_load_file(self, path: Path):
        try:
            self._scans   = load_scanning_kinetics_csv(path)
            self._n_scans = len(self._scans)
        except Exception as exc:
            self._card1.set_status(ERROR)
            self._sk_scan_count_lbl.setText(f"Error: {exc}")
            self._sk_scan_count_lbl.setObjectName("error_label")
            return

        self._sk_scan_count_lbl.setText(f"Loaded {self._n_scans} scans.")
        self._card1.set_status(DONE)

        # Set scan window max
        self._sk_scan_end.blockSignals(True)
        self._sk_scan_end.setRange(0, self._n_scans - 1)
        self._sk_scan_end.setValue(0)   # 0 = last
        self._sk_scan_end.blockSignals(False)
        self._sk_scan_start.setRange(0, self._n_scans - 1)
        self._sk_update_time_labels()

        for card in (self._card2, self._card3, self._card4,
                     self._card5, self._card6):
            card.set_card_enabled(True)
            card.set_status(READY)

        self._fit_done = False
        self._sk_draw_waterfall()
        print(f"Loaded {self._n_scans} scans from {path.name}")

    def _sk_draw_waterfall(self):
        if not self._scans:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        cmap = plt.get_cmap("coolwarm")
        for i, (wl, ab) in enumerate(self._scans):
            color = cmap(i / max(len(self._scans) - 1, 1))
            ax.plot(wl, ab, color=color, linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title("Spectral waterfall  (blue→red: early→late)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._sk_raw_plot.set_figure(fig)
        plt.close(fig)

    # ── Stage 6 actions ────────────────────────────────────────────────────

    def _sk_mark_stale(self):
        if self._fit_done:
            self._card6.set_status(STALE)
            self._sk_stale_banner.setVisible(True)

    def _sk_run_fit(self):
        if not self._scans:
            return

        wl_list   = [sb.value() for sb in self._wavelength_inputs]
        tol       = self._sk_tol.value()
        interval_s = self._sk_get_interval_s()
        scan_start = self._sk_scan_start.value()
        scan_end   = self._sk_scan_end.value() or None
        switch     = "negative" if self._sk_sw_neg.isChecked() else "positive"
        iqr        = self._sk_iqr.value()
        temp       = self._sk_temp.value()

        if self._sk_ref_file.isChecked():
            a_inf_mode = "reference"
            ref_scans  = self._ref_scans or []
        elif self._sk_ref_fixed.isChecked():
            a_inf_mode = "fixed"
            ref_scans  = []
        else:
            a_inf_mode = "free"
            ref_scans  = []
        a_inf_value = self._sk_ref_ainf_val.value() if self._sk_ref_fixed.isChecked() else None

        filepath = Path(self._sk_file_edit.text())

        self._sk_run_btn.setEnabled(False)
        self._card6.set_status(WAITING)

        def _fit():
            return run_scanning_fit(
                filepath=filepath,
                target_wavelengths=wl_list,
                wavelength_tolerance=tol,
                time_interval_s=interval_s,
                scan_start=scan_start,
                scan_end=scan_end,
                switch=switch,
                a_inf_mode=a_inf_mode,
                a_inf_value=a_inf_value,
                reference_scans=ref_scans if ref_scans else None,
                iqr_factor=iqr,
                temperature_c=temp,
            )

        self._worker = Worker(_fit)
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._sk_on_fit_done)
        self._worker.error_signal.connect(self._sk_on_fit_error)
        self._worker.finished_signal.connect(
            lambda: self._sk_run_btn.setEnabled(True))
        self._worker.start()

    @pyqtSlot(object)
    def _sk_on_fit_done(self, results: list[FitResult]):
        self._fit_done = True
        self._card6.set_status(DONE)
        self._sk_stale_banner.setVisible(False)

        while self._sk_plot_layout.count():
            item = self._sk_plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        file_stem = Path(self._sk_file_edit.text()).stem
        result_entries = []

        for r in results:
            if not r.success:
                lbl = QLabel(f"  {r.label}: {r.error_msg}")
                lbl.setObjectName("error_label")
                self._sk_plot_layout.addWidget(lbl)
                continue

            fig, _ = plot_half_life_with_linear(
                r.time_full, r.abs_full,
                start_idx=r.start_idx, end_idx=r.end_idx,
                time_sel=r.time_clean, absorbance_sel=r.abs_clean,
                time_outliers=r.time_outliers, absorbance_outliers=r.abs_outliers,
                fitted_curve=r.fitted_curve, r_squared=r.r2,
                popt=r.popt, t_half=r.t_half, switch=r.switch,
                title=f"{file_stem} | {r.label} | T={r.temperature_c:.0f}°C",
                show=False,
            )
            pw = PlotWidget(
                info_title="Fit plot",
                info_text=(
                    "Left: exponential fit. Right: linearised transform.\n"
                    "A straight right panel confirms first-order kinetics."
                ),
                min_height=300,
            )
            pw.set_save_dir(self._output_path / "half_life" / "results" / "plots"
                            if self._output_path else None)
            pw.set_default_filename(
                f"{file_stem}_{r.label}_{r.temperature_c:.0f}C.png"
                .replace(" ", "_").replace("/", "-"))
            pw.set_figure(fig)
            plt.close(fig)
            self._sk_plot_layout.addWidget(pw)

            result_entries.append({
                "File":          Path(self._sk_file_edit.text()).name,
                "Wavelength":    r.label,
                "Type":          "Scanning Kinetics",
                "Temperature_C": r.temperature_c,
                "Switch":        r.switch,
                "A0":            r.popt[0] if r.popt is not None else None,
                "A_inf":         r.popt[1] if (r.popt is not None and r.switch == "negative") else None,
                "k":             r.popt[-1] if r.popt is not None else None,
                "Half_life_s":   r.t_half,
                "R2":            r.r2,
            })

        if result_entries:
            self._sk_master.add_pending(result_entries)

    @pyqtSlot(str)
    def _sk_on_fit_error(self, msg: str):
        self._card6.set_status(ERROR)
        self.log_signal.emit(f"Fit error: {msg}", "ERROR")

    def apply_prefs(self, prefs):
        """Populate preference widgets from a HalfLifeScanningPrefs object."""
        if prefs.switch == "positive":
            self._sk_sw_pos.setChecked(True)
        else:
            self._sk_sw_neg.setChecked(True)
        self._sk_iqr.setValue(prefs.iqr_factor)
        self._sk_temp.setValue(prefs.temperature_c)

    def collect_prefs(self, prefs):
        """Write current widget values into a HalfLifeScanningPrefs object."""
        prefs.switch        = "positive" if self._sk_sw_pos.isChecked() else "negative"
        prefs.iqr_factor    = self._sk_iqr.value()
        prefs.temperature_c = self._sk_temp.value()

    def set_raw_path(self, path: Path):
        self._raw_path = path

    def set_output_path(self, path: Path):
        self._output_path = path
        self._sk_master.set_output_path(path)
        plots_dir = path / "half_life" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self._sk_raw_plot.set_save_dir(plots_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Publication Composer panel
# ═══════════════════════════════════════════════════════════════════════════

class _RawFileEntry(QWidget):
    """One browseable raw-data + segments pair for the raw subplot."""

    removed = pyqtSignal(object)
    changed = pyqtSignal()

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self._idx = idx
        self.raw_path: "Path | None" = None
        self.seg_path: "Path | None" = None
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        row.addWidget(QLabel(f"File {self._idx + 1}:"))
        self._raw_edit = QLineEdit()
        self._raw_edit.setReadOnly(True)
        self._raw_edit.setPlaceholderText("raw_data.csv…")
        row.addWidget(self._raw_edit)
        btn_raw = QPushButton("Browse…")
        btn_raw.setFixedWidth(72)
        btn_raw.clicked.connect(self._browse_raw)
        row.addWidget(btn_raw)
        row.addWidget(QLabel("Segments:"))
        self._seg_edit = QLineEdit()
        self._seg_edit.setReadOnly(True)
        self._seg_edit.setPlaceholderText("segments.csv (optional)…")
        row.addWidget(self._seg_edit)
        btn_seg = QPushButton("Browse…")
        btn_seg.setFixedWidth(72)
        btn_seg.clicked.connect(self._browse_seg)
        row.addWidget(btn_seg)
        btn_rm = QPushButton("✕")
        btn_rm.setObjectName("delete_row_btn")
        btn_rm.setFixedWidth(28)
        btn_rm.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(btn_rm)

    def _browse_raw(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raw_data.csv", "", "CSV files (*.csv)")
        if path:
            self.raw_path = Path(path)
            self._raw_edit.setText(Path(path).name)
            self.changed.emit()

    def _browse_seg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select segments.csv", "", "CSV files (*.csv)")
        if path:
            self.seg_path = Path(path)
            self._seg_edit.setText(Path(path).name)
            self.changed.emit()

    def auto_fill_from_folder(self, folder: Path):
        raw = folder / "raw_data.csv"
        seg = folder / "segments.csv"
        if raw.exists():
            self.raw_path = raw
            self._raw_edit.setText(raw.name)
        if seg.exists():
            self.seg_path = seg
            self._seg_edit.setText(seg.name)
        self.changed.emit()


class PublicationPanel(QWidget):
    """
    Five-stage pipeline for assembling the 4-panel publication figure.

    Subplot layout:
      [Raw traces + segments] | [Segment 1 – linearised + fit]
      [Segment 2             ] | [Segment 3                   ]
    """

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pub_folder: "Path | None" = None
        self._raw_entries: list         = []   # list of _RawFileEntry
        self._seg_folders: list         = [None, None, None]
        self._seg_edits:   list         = []   # QLineEdit × 3
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
        lay = QVBoxLayout(container)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(0)
        scroll.setWidget(container)

        self._card1 = self._build_card1_pub()
        self._card2 = self._build_card2_pub()
        self._card3 = self._build_card3_pub()
        self._card4 = self._build_card4_pub()
        self._card5 = self._build_card5_pub()

        for c in (self._card1, self._card2, self._card3, self._card4, self._card5):
            lay.addWidget(c)
        lay.addStretch()

        for c in (self._card2, self._card3, self._card4, self._card5):
            c.set_card_enabled(False)

    # ── Stage 1: session folder ────────────────────────────────────────────

    def _build_card1_pub(self) -> StageCard:
        card = StageCard("Stage 1 — Publication folder")
        card.set_status(READY)
        row = QHBoxLayout()
        row.addWidget(QLabel("Folder:"))
        self._pub_folder_edit = QLineEdit()
        self._pub_folder_edit.setReadOnly(True)
        self._pub_folder_edit.setPlaceholderText(
            "…/half_life/results/publication/")
        row.addWidget(self._pub_folder_edit)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._browse_pub_folder)
        row.addWidget(btn)
        card.add_layout(row)
        self._pub_folder_lbl = QLabel("")
        self._pub_folder_lbl.setObjectName("detected_label")
        card.add_widget(self._pub_folder_lbl)
        return card

    # ── Stage 2: raw subplot ───────────────────────────────────────────────

    def _build_card2_pub(self) -> StageCard:
        card = StageCard("Stage 2 — Raw subplot (Subplot 1)")
        card.add_widget(InfoButton(
            "Raw subplot",
            "Add one or more raw kinetics files to overlay in the top-left subplot.\n"
            "Each entry needs a raw_data.csv and optionally a segments.csv\n"
            "(both produced by 'Save raw + segments for publication').",
        ))
        self._raw_list_widget = QWidget()
        self._raw_list_layout = QVBoxLayout(self._raw_list_widget)
        self._raw_list_layout.setContentsMargins(0, 0, 0, 0)
        self._raw_list_layout.setSpacing(2)
        card.add_widget(self._raw_list_widget)
        btn_add = QPushButton("+ Add raw file")
        btn_add.setFixedWidth(120)
        btn_add.clicked.connect(self._add_raw_entry)
        card.add_widget(btn_add)
        return card

    # ── Stage 3: segment subplots ──────────────────────────────────────────

    def _build_card3_pub(self) -> StageCard:
        card = StageCard("Stage 3 — Segment subplots (Subplots 2–4)")
        self._seg_edits = []
        for i in range(3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Subplot {i + 2}  — Segment {i + 1} folder:"))
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText(f"publication/segment_{i + 1}/")
            row.addWidget(edit)
            btn = QPushButton("Browse…")
            btn.setFixedWidth(72)
            btn.clicked.connect(lambda _, n=i: self._browse_seg_folder(n))
            row.addWidget(btn)
            card.add_layout(row)
            self._seg_edits.append(edit)
        return card

    # ── Stage 4: preview ──────────────────────────────────────────────────

    def _build_card4_pub(self) -> StageCard:
        card = StageCard("Stage 4 — Preview")
        btn_prev = QPushButton("▶  Generate preview")
        btn_prev.setObjectName("run_btn")
        btn_prev.setFixedHeight(34)
        btn_prev.clicked.connect(self._generate_preview)
        card.add_widget(btn_prev)
        self._pub_plot = PlotWidget(
            info_title="Publication figure",
            info_text=(
                "Four-panel publication figure.\n"
                "Top-left: raw traces with segment shading.\n"
                "Other panels: linearised fits per segment."
            ),
            min_height=500,
        )
        card.add_widget(self._pub_plot)
        return card

    # ── Stage 5: export ───────────────────────────────────────────────────

    def _build_card5_pub(self) -> StageCard:
        card = StageCard("Stage 5 — Export")
        row = QHBoxLayout()
        row.addWidget(QLabel("Output file:"))
        self._pub_out_edit = QLineEdit()
        self._pub_out_edit.setPlaceholderText("publication_data.xlsx")
        row.addWidget(self._pub_out_edit)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(72)
        btn_browse.clicked.connect(self._browse_out_file)
        row.addWidget(btn_browse)
        card.add_layout(row)
        btn_exp = QPushButton("💾  Export Excel")
        btn_exp.setObjectName("accent")
        btn_exp.setFixedHeight(34)
        btn_exp.clicked.connect(self._export_excel)
        card.add_widget(btn_exp)
        self._pub_exp_lbl = QLabel("")
        self._pub_exp_lbl.setObjectName("detected_label")
        card.add_widget(self._pub_exp_lbl)
        return card

    # ── Stage 1 actions ───────────────────────────────────────────────────

    def _browse_pub_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select publication folder", "")
        if not folder:
            return
        self._pub_folder = Path(folder)
        self._pub_folder_edit.setText(str(self._pub_folder))
        self._auto_populate()
        for c in (self._card2, self._card3, self._card4, self._card5):
            c.set_card_enabled(True)
            c.set_status(READY)

    def _auto_populate(self):
        """Auto-fill from the publication folder structure."""
        if self._pub_folder is None:
            return
        found = []
        raw_dir = self._pub_folder / "raw"
        if raw_dir.exists():
            found.append("raw/")
            # Auto-add one raw entry
            if not self._raw_entries:
                self._add_raw_entry()
            self._raw_entries[0].auto_fill_from_folder(raw_dir)
        for i in range(3):
            seg_dir = self._pub_folder / f"segment_{i + 1}"
            if seg_dir.exists():
                found.append(f"segment_{i + 1}/")
                self._seg_folders[i] = seg_dir
                self._seg_edits[i].setText(str(seg_dir))
        # Set default output path
        default_out = self._pub_folder / "publication_data.xlsx"
        self._pub_out_edit.setText(str(default_out))
        self._pub_folder_lbl.setText(
            f"Found: {', '.join(found) if found else 'nothing yet'}")

    # ── Stage 2 actions ───────────────────────────────────────────────────

    def _add_raw_entry(self):
        entry = _RawFileEntry(len(self._raw_entries), self)
        entry.removed.connect(self._remove_raw_entry)
        self._raw_entries.append(entry)
        self._raw_list_layout.addWidget(entry)

    def _remove_raw_entry(self, entry):
        self._raw_entries.remove(entry)
        self._raw_list_layout.removeWidget(entry)
        entry.deleteLater()
        # Re-index labels
        for i, e in enumerate(self._raw_entries):
            e._idx = i

    # ── Stage 3 actions ───────────────────────────────────────────────────

    def _browse_seg_folder(self, idx: int):
        start = str(self._pub_folder) if self._pub_folder else ""
        folder = QFileDialog.getExistingDirectory(
            self, f"Select segment {idx + 1} folder", start)
        if not folder:
            return
        self._seg_folders[idx] = Path(folder)
        self._seg_edits[idx].setText(folder)

    # ── Stage 4 actions ───────────────────────────────────────────────────

    def _generate_preview(self):
        try:
            fig = self._build_figure()
            self._pub_plot.set_figure(fig)
            plt.close(fig)
            self._card5.set_card_enabled(True)
            self._card5.set_status(READY)
        except Exception as exc:
            self._card4.set_status(ERROR)
            print(f"Preview error: {exc}")
            import traceback; traceback.print_exc()

    def _build_figure(self) -> "plt.Figure":
        import pandas as pd
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.patch.set_facecolor("white")
        ax_raw  = axes[0, 0]
        ax_segs = [axes[0, 1], axes[1, 0], axes[1, 1]]

        # ── Raw subplot ───────────────────────────────────────────────────
        ci = 0
        for entry in self._raw_entries:
            if entry.raw_path is None or not entry.raw_path.exists():
                continue
            df   = pd.read_csv(entry.raw_path)
            time = df["time_s"].values
            chs  = [c for c in df.columns if c != "time_s"]
            for ch in chs:
                color = _CHANNEL_COLORS[ci % len(_CHANNEL_COLORS)]
                ci += 1
                ax_raw.plot(time, df[ch].values, "o", color=color,
                            markersize=2, alpha=0.7, label=ch)
            if entry.seg_path and entry.seg_path.exists():
                seg_df = pd.read_csv(entry.seg_path)
                for _, srow in seg_df.iterrows():
                    ax_raw.axvspan(srow["t_start_s"], srow["t_end_s"],
                                   color="#e8a020", alpha=0.12)
                    ax_raw.text(
                        (srow["t_start_s"] + srow["t_end_s"]) / 2, 1.0,
                        str(int(srow["segment"])),
                        transform=ax_raw.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color="#b06010")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Absorbance")
        ax_raw.set_title("Raw kinetic traces")
        if ci:
            ax_raw.legend(fontsize=7)
        ax_raw.grid(True, alpha=0.3)

        # ── Segment subplots ──────────────────────────────────────────────
        for ax, folder, seg_lbl in zip(
                ax_segs,
                self._seg_folders,
                ["Segment 1", "Segment 2", "Segment 3"]):
            ax.set_title(seg_lbl)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ln(A \u2212 A\u221e)")
            ax.grid(True, alpha=0.3)
            if folder is None or not Path(folder).exists():
                continue
            folder = Path(folder)
            data_f   = folder / "data_points.csv"
            fit_f    = folder / "fit_line.csv"
            params_f = folder / "fit_params.csv"
            if not data_f.exists() or not params_f.exists():
                continue
            data_df   = pd.read_csv(data_f)
            fit_df    = pd.read_csv(fit_f)  if fit_f.exists()    else None
            params_df = pd.read_csv(params_f)
            channels  = params_df["channel"].tolist()
            annot_lines = []
            for j, ch in enumerate(channels):
                color = _CHANNEL_COLORS[j % len(_CHANNEL_COLORS)]
                ln_col = f"{ch}_ln_A"
                if ln_col in data_df.columns:
                    ax.scatter(data_df["time_s"], data_df[ln_col],
                               color=color, s=10, alpha=0.75,
                               label=ch, zorder=3)
                fit_col = f"{ch}_fit"
                if fit_df is not None and fit_col in fit_df.columns:
                    ax.plot(fit_df["time_s"], fit_df[fit_col],
                            color=color, linewidth=1.5)
                row_p = params_df[params_df["channel"] == ch]
                if not row_p.empty:
                    k_val     = float(row_p["k_s-1"].iloc[0])
                    a_inf_val = float(row_p["A_inf"].iloc[0])
                    annot_lines.append(
                        f"{ch}:  k = {k_val:.3e} s\u207b\u00b9,  "
                        f"A\u221e = {a_inf_val:.4f}")
            if annot_lines:
                ax.text(0.97, 0.97, "\n".join(annot_lines),
                        transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white", edgecolor="#cccccc",
                                  alpha=0.92))
            if channels:
                ax.legend(fontsize=7)

        plt.tight_layout()
        return fig

    # ── Stage 5 actions ───────────────────────────────────────────────────

    def _browse_out_file(self):
        start = self._pub_out_edit.text() or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save publication data", start,
            "Excel files (*.xlsx)")
        if path:
            self._pub_out_edit.setText(path)

    def _export_excel(self):
        import pandas as pd
        out_text = self._pub_out_edit.text().strip()
        if not out_text:
            self._pub_exp_lbl.setText("Set an output file first.")
            return
        out_path = Path(out_text)
        if not out_path.suffix:
            out_path = out_path.with_suffix(".xlsx")
        try:
            writer = pd.ExcelWriter(out_path, engine="openpyxl")
            # Raw sheet
            raw_frames = []
            for entry in self._raw_entries:
                if entry.raw_path and entry.raw_path.exists():
                    df = pd.read_csv(entry.raw_path)
                    prefix = entry.raw_path.parent.name
                    df = df.rename(columns={
                        c: (f"{prefix}_{c}" if c != "time_s" else f"{prefix}_time_s")
                        for c in df.columns})
                    raw_frames.append(df)
            if raw_frames:
                import numpy as np
                max_len = max(len(f) for f in raw_frames)
                padded = {}
                for df in raw_frames:
                    for col in df.columns:
                        arr = df[col].values
                        if len(arr) < max_len:
                            arr = np.concatenate(
                                [arr, np.full(max_len - len(arr), np.nan)])
                        padded[col] = arr
                pd.DataFrame(padded).to_excel(
                    writer, sheet_name="Raw", index=False)
            # Segments sheet
            for entry in self._raw_entries:
                if entry.seg_path and entry.seg_path.exists():
                    pd.read_csv(entry.seg_path).to_excel(
                        writer, sheet_name="Segments", index=False)
                    break
            # Segment data sheets
            for i, folder in enumerate(self._seg_folders, 1):
                if folder is None:
                    continue
                folder = Path(folder)
                for fname, sheet in [
                    ("data_points.csv", f"Segment_{i}_data"),
                    ("fit_line.csv",    f"Segment_{i}_fit"),
                    ("fit_params.csv",  f"Segment_{i}_params"),
                ]:
                    fpath = folder / fname
                    if fpath.exists():
                        pd.read_csv(fpath).to_excel(
                            writer, sheet_name=sheet, index=False)
            writer.close()
            self._pub_exp_lbl.setText(f"Exported → {out_path.name}")
            self._card5.set_status(DONE)
            print(f"Publication Excel → {out_path}")
        except Exception as exc:
            self._pub_exp_lbl.setText(f"Error: {exc}")
            self._card5.set_status(ERROR)

    def set_output_path(self, path: Path):
        """Called by HalfLifeTab when the project output path changes."""
        default = path / "half_life" / "results" / "publication"
        if default.exists() and not self._pub_folder:
            self._pub_folder = default
            self._pub_folder_edit.setText(str(default))
            self._auto_populate()
            for c in (self._card2, self._card3, self._card4, self._card5):
                c.set_card_enabled(True)
                c.set_status(READY)


# ═══════════════════════════════════════════════════════════════════════════
# Half-Life Tab  (top-level)
# ═══════════════════════════════════════════════════════════════════════════

class HalfLifeTab(QWidget):
    def __init__(self, folder_header=None, parent=None):
        super().__init__(parent)
        self._folder_header = folder_header
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        from PyQt6.QtWidgets import QTabWidget
        inner = QTabWidget()
        inner.setObjectName("inner_tabs")

        self._kinetics_panel    = KineticsPanel()
        self._scanning_panel    = ScanningKineticsPanel()
        self._publication_panel = PublicationPanel()
        inner.addTab(self._kinetics_panel,    "Kinetics")
        inner.addTab(self._scanning_panel,    "Scanning Kinetics")
        inner.addTab(self._publication_panel, "Publication")

        root.addWidget(inner)

    def set_raw_path(self, path: Path):
        self._kinetics_panel.set_raw_path(path)
        self._scanning_panel.set_raw_path(path)

    def set_output_path(self, path: Path):
        self._kinetics_panel.set_output_path(path)
        self._scanning_panel.set_output_path(path)
        self._publication_panel.set_output_path(path)

    def apply_prefs(self, project_prefs):
        """Apply loaded ProjectPrefs to both panels."""
        self._kinetics_panel.apply_prefs(project_prefs.half_life_kinetics)
        self._scanning_panel.apply_prefs(project_prefs.half_life_scanning)

    def collect_prefs(self, project_prefs):
        """Collect current panel settings into a ProjectPrefs object."""
        self._kinetics_panel.collect_prefs(project_prefs.half_life_kinetics)
        self._scanning_panel.collect_prefs(project_prefs.half_life_scanning)
