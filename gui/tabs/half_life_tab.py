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
        self._k_t_end.valueChanged.connect(self._k_update_raw_plot)
        self._k_t_end.valueChanged.connect(self._k_mark_stale)
        row2.addWidget(self._k_t_end)
        row2.addStretch()
        card.add_layout(row2)

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

        if t_end > t_start:
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
            pw.set_save_dir(self._output_path)
            pw.set_default_filename(
                f"{file_stem}_{r.label}_{r.temperature_c:.0f}C.png"
                .replace(" ", "_").replace("/", "-"))
            pw.set_figure(fig)
            plt.close(fig)
            self._k_plot_layout.addWidget(pw)

            # Add result to master table
            switch_val = r.switch
            popt = r.popt
            result_entry = {
                "File":          Path(self._k_file_edit.text()).name,
                "Wavelength":    r.label,
                "Type":          "Kinetics",
                "Temperature_C": r.temperature_c,
                "Switch":        switch_val,
                "A0":            popt[0] if popt is not None else None,
                "A_inf":         popt[1] if (popt is not None and switch_val == "negative") else None,
                "k":             popt[-1] if popt is not None else None,
                "Half_life_s":   r.t_half,
                "R2":            r.r2,
            }
            self._k_master.add_pending(result_entry)

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
        plots_dir = path / "plots" / "half_life"
        plots_dir.mkdir(parents=True, exist_ok=True)
        (path / "data" / "half_life" / "results").mkdir(parents=True, exist_ok=True)
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
            pw.set_save_dir(self._output_path)
            pw.set_default_filename(
                f"{file_stem}_{r.label}_{r.temperature_c:.0f}C.png"
                .replace(" ", "_").replace("/", "-"))
            pw.set_figure(fig)
            plt.close(fig)
            self._sk_plot_layout.addWidget(pw)

            result_entry = {
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
            }
            self._sk_master.add_pending(result_entry)

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
        plots_dir = path / "plots" / "half_life"
        plots_dir.mkdir(parents=True, exist_ok=True)
        (path / "data" / "half_life" / "results").mkdir(parents=True, exist_ok=True)
        self._sk_raw_plot.set_save_dir(plots_dir)


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

        # ── Mode selector bar ──────────────────────────────────────────────
        mode_bar = QWidget()
        mode_bar.setObjectName("mode_bar")
        mode_bar.setFixedHeight(44)
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(16, 0, 16, 0)
        mode_layout.setSpacing(24)

        mode_layout.addWidget(QLabel("Measurement type:"))

        self._rb_kinetics  = QRadioButton("Kinetics")
        self._rb_scanning  = QRadioButton("Scanning Kinetics")
        self._rb_kinetics.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_kinetics)
        bg.addButton(self._rb_scanning)
        mode_layout.addWidget(self._rb_kinetics)
        mode_layout.addWidget(self._rb_scanning)
        mode_layout.addStretch()

        root.addWidget(mode_bar)

        # ── Stacked panels ─────────────────────────────────────────────────
        from PyQt6.QtWidgets import QStackedWidget
        self._stack = QStackedWidget()

        self._kinetics_panel  = KineticsPanel()
        self._scanning_panel  = ScanningKineticsPanel()
        self._stack.addWidget(self._kinetics_panel)
        self._stack.addWidget(self._scanning_panel)

        root.addWidget(self._stack)

        self._rb_kinetics.toggled.connect(
            lambda on: self._stack.setCurrentIndex(0 if on else 1))

    def set_raw_path(self, path: Path):
        self._kinetics_panel.set_raw_path(path)
        self._scanning_panel.set_raw_path(path)

    def set_output_path(self, path: Path):
        self._kinetics_panel.set_output_path(path)
        self._scanning_panel.set_output_path(path)

    def apply_prefs(self, project_prefs):
        """Apply loaded ProjectPrefs to both panels."""
        self._kinetics_panel.apply_prefs(project_prefs.half_life_kinetics)
        self._scanning_panel.apply_prefs(project_prefs.half_life_scanning)

    def collect_prefs(self, project_prefs):
        """Collect current panel settings into a ProjectPrefs object."""
        self._kinetics_panel.collect_prefs(project_prefs.half_life_kinetics)
        self._scanning_panel.collect_prefs(project_prefs.half_life_scanning)
