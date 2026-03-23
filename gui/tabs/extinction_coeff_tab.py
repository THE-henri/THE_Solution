"""
Extinction Coefficients tab.

Stage 1 – Preparations   : per-file table (select files manually; weight, MW, volume)
Stage 2 – Parameters      : compound name, path length, solvent, temperature
Stage 3 – Run & Results   : background calculation + plot + save CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QAbstractItemView, QFrame,
)

from core.optics import calculate_extinction_coefficients_integer_wavelengths
from core.plotting import plot_extinction_coefficients
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.widgets.info_button import InfoButton
from gui.worker import Worker


class ExtinctionCoeffTab(QWidget):
    """Tab for calculating molar extinction coefficients from UV-Vis data."""

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path: Optional[Path] = None
        self._df_result: Optional[pd.DataFrame] = None
        self._worker: Optional[Worker] = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

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

        # ── Stage 1 — Preparations ────────────────────────────────────────
        self._stage1 = StageCard("Stage 1 — Preparations")
        self._stage1.add_info_button(
            "Extinction Coefficients — File Format",
            "Select one or more Cary 60 CSV files (column-pair format).\n\n"
            "Each file may contain multiple replicate scans. Replicates within "
            "a file are averaged to give one preparation value; preparations are "
            "then averaged for the final ε spectrum.\n\n"
            "Enter the dissolved mass (mg), molar mass (g/mol), and volume (mL) "
            "for each file to calculate the exact concentration."
        )

        hint = QLabel(
            "Add one row per CSV file. Each file may contain multiple replicate scans "
            "(Cary 60 column-pair format); replicates are averaged within each "
            "preparation, then preparations are averaged for the final result."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint)

        ctrl = QHBoxLayout()
        self._select_btn = QPushButton("Select files…")
        self._select_btn.setFixedWidth(120)
        self._select_btn.clicked.connect(self._select_files)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.setFixedWidth(90)
        self._clear_btn.clicked.connect(self._clear_table)
        ctrl.addWidget(self._select_btn)
        ctrl.addWidget(self._clear_btn)
        ctrl.addStretch()
        self._stage1.add_layout(ctrl)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["File", "Weight (mg)", "MW (g/mol)", "Volume (mL)", ""])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 52)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setMinimumHeight(140)
        self._table.setMaximumHeight(260)
        self._table.itemChanged.connect(self._on_table_changed)
        self._stage1.add_widget(self._table)

        self._stage1.set_status(WAITING)
        layout.addWidget(self._stage1)

        # ── Stage 2 — Parameters ──────────────────────────────────────────
        self._stage2 = StageCard("Stage 2 — Parameters")
        self._stage2.add_info_button(
            "Sample Parameters",
            "Path length (cm): optical path of the cuvette (typically 1 cm).\n\n"
            "Temperature (°C): recorded in the output CSV for traceability; "
            "does not affect the ε calculation.\n\n"
            "Solvent and compound name are stored in the output CSV header "
            "for use in downstream tabs (Spectra, QY)."
        )

        row1 = QHBoxLayout()
        self._compound_edit = self._text_field(
            row1, "Compound name", "", pref=True,
            info_title="Compound name",
            info_text=(
                "Stored in the output CSV — used for plot titles and file naming.\n"
                "Does not affect any calculation."
            ),
        )
        self._solvent_edit = self._text_field(
            row1, "Solvent", "acetonitrile", pref=True,
            info_title="Solvent",
            info_text=(
                "Solvent name stored in the output CSV for traceability.\n"
                "Solvent identity can shift absorption peaks; recording it ensures\n"
                "results remain interpretable when revisited later."
            ),
        )
        row1.addStretch()
        self._stage2.add_layout(row1)

        row2 = QHBoxLayout()
        self._path_length_spin = self._dspin(
            row2, "Path length (cm)", 0.001, 100.0, 1.0, 0.1, pref=True,
            info_title="Path length (cm)",
            info_text=(
                "Optical path length of the cuvette in centimetres (typically 1 cm).\n"
                "Used in Beer–Lambert: A = ε · c · l\n"
                "Ensure this matches the actual cuvette used during measurement."
            ),
        )
        self._temperature_spin = self._dspin(
            row2, "Temperature (°C)", -196.0, 400.0, 25.0, 1.0, pref=True,
            info_title="Temperature (°C)",
            info_text=(
                "Measurement temperature — stored in the output CSV for\n"
                "traceability. Does not affect the extinction coefficient calculation\n"
                "but is important for temperature-sensitive samples."
            ),
        )
        row2.addStretch()
        self._stage2.add_layout(row2)

        for sig in (
            self._compound_edit.textChanged,
            self._solvent_edit.textChanged,
            self._path_length_spin.valueChanged,
            self._temperature_spin.valueChanged,
        ):
            sig.connect(self._mark_stale)

        self._stage2.set_status(READY)
        layout.addWidget(self._stage2)

        # ── Stage 3 — Run & Results ───────────────────────────────────────
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(
            "Extinction Coefficient Calculation",
            "ε(λ) is calculated via Beer–Lambert: ε = A / (c · l)\n\n"
            "where A is absorbance, c is concentration (mol/L), and l is path "
            "length (cm). Units: M⁻¹ cm⁻¹.\n\n"
            "The result CSV contains wavelength, mean ε, and standard deviation "
            "across preparations. This file can be loaded directly by the "
            "Spectra and Quantum Yield tabs."
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
            info_title="Extinction Coefficients",
            info_text=(
                "Molar extinction coefficients ε (M⁻¹ cm⁻¹) calculated via "
                "Beer-Lambert law: ε = A / (c × l).\n"
                "Each preparation is plotted separately; "
                "the black line is the overall mean ± std across preparations."
            ),
            min_height=340,
        )
        self._stage3.add_widget(self._plot)

        save_row = QHBoxLayout()
        self._save_csv_btn = QPushButton("Save CSV")
        self._save_csv_btn.setEnabled(False)
        self._save_csv_btn.clicked.connect(self._save_csv)
        save_row.addWidget(self._save_csv_btn)
        save_row.addStretch()
        self._stage3.add_layout(save_row)

        self._stage3.set_status(WAITING)
        layout.addWidget(self._stage3)
        layout.addStretch()

    # ── UI helpers ────────────────────────────────────────────────────────

    def _text_field(self, parent_layout: QHBoxLayout,
                    label: str, default: str, pref: bool = False,
                    info_title: str = "", info_text: str = "") -> QLineEdit:
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl_row = QHBoxLayout()
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        lbl_row.addWidget(lbl)
        if info_title:
            lbl_row.addWidget(InfoButton(info_title, info_text))
        lbl_row.addStretch()
        col.addLayout(lbl_row)
        edit = QLineEdit(default)
        edit.setMinimumWidth(160)
        col.addWidget(edit)
        parent_layout.addLayout(col)
        return edit

    def _dspin(self, parent_layout: QHBoxLayout, label: str,
               lo: float, hi: float, val: float, step: float,
               pref: bool = False,
               info_title: str = "", info_text: str = "") -> QDoubleSpinBox:
        col = QVBoxLayout()
        col.setSpacing(4)
        lbl_row = QHBoxLayout()
        lbl = QLabel(label)
        if pref:
            lbl.setObjectName("pref_label")
        lbl_row.addWidget(lbl)
        if info_title:
            lbl_row.addWidget(InfoButton(info_title, info_text))
        lbl_row.addStretch()
        col.addLayout(lbl_row)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setMinimumWidth(120)
        col.addWidget(spin)
        parent_layout.addLayout(col)
        return spin

    # ── Public slots ──────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        plots_dir = path / "extinction_coefficients" / "results" / "plots"
        self._plot.set_save_dir(plots_dir)

    # ── Table management ──────────────────────────────────────────────────

    def _select_files(self):
        start = str(self._output_path or Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select UV-Vis CSV files", start, "CSV files (*.csv)")
        existing = self._table_paths()
        for p in paths:
            path = Path(p)
            if str(path) not in existing:
                self._add_row(path)
        self._update_stage1_status()

    def _clear_table(self):
        self._table.setRowCount(0)
        self._update_stage1_status()
        self._mark_stale()

    def _table_paths(self) -> set[str]:
        result: set[str] = set()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item:
                result.add(item.data(Qt.ItemDataRole.UserRole))
        return result

    def _add_row(self, filepath: Path):
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)

        file_item = QTableWidgetItem(filepath.name)
        file_item.setData(Qt.ItemDataRole.UserRole, str(filepath))
        file_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._table.setItem(row, 0, file_item)

        for col in (1, 2, 3):
            num_item = QTableWidgetItem("")
            num_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, col, num_item)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(44)
        del_btn.clicked.connect(self._delete_row_by_sender)
        self._table.setCellWidget(row, 4, del_btn)
        self._table.blockSignals(False)

    def _delete_row_by_sender(self):
        btn = self.sender()
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, 4) is btn:
                self._table.removeRow(row)
                break
        self._update_stage1_status()
        self._mark_stale()

    def _on_table_changed(self):
        self._update_stage1_status()
        self._mark_stale()

    def _update_stage1_status(self):
        if self._table.rowCount() == 0:
            self._stage1.set_status(WAITING)
            self._run_btn.setEnabled(False)
            return
        for row in range(self._table.rowCount()):
            for col in (1, 2, 3):
                item = self._table.item(row, col)
                if item is None or item.text().strip() == "":
                    self._stage1.set_status(WAITING)
                    self._run_btn.setEnabled(False)
                    return
        self._stage1.set_status(READY)
        self._run_btn.setEnabled(True)

    def _mark_stale(self):
        if self._df_result is not None:
            self._stage3.set_status(STALE)

    # ── Run ───────────────────────────────────────────────────────────────

    def _collect_measurements(self) -> tuple[list, list[str]]:
        measurements, errors = [], []
        for row in range(self._table.rowCount()):
            path_str = self._table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            try:
                weight_mg = float(self._table.item(row, 1).text())
                mw_gmol   = float(self._table.item(row, 2).text())
                volume_ml = float(self._table.item(row, 3).text())
            except (ValueError, AttributeError):
                errors.append(f"Row {row + 1}: invalid number")
                continue
            measurements.append({
                "csv_file":  path_str,
                "weight_mg": weight_mg,
                "MW_gmol":   mw_gmol,
                "volume_mL": volume_ml,
            })
        return measurements, errors

    def _run(self):
        measurements, errors = self._collect_measurements()
        if errors:
            self._status_lbl.setText("; ".join(errors))
            self._stage3.set_status(ERROR)
            return
        if not measurements:
            self._status_lbl.setText("No files to process.")
            return

        compound = self._compound_edit.text().strip() or "Unknown"
        path_len = self._path_length_spin.value()
        solvent  = self._solvent_edit.text().strip() or "unknown"
        temp_c   = self._temperature_spin.value()

        self._run_btn.setEnabled(False)
        self._save_csv_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._stage3.set_status(WAITING)

        self._worker = Worker(
            calculate_extinction_coefficients_integer_wavelengths,
            measurements=measurements,
            path_length_cm=path_len,
            solvent=solvent,
            temperature=temp_c,
            compound_name=compound,
        )
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, df: pd.DataFrame):
        self._df_result = df
        fig, _ = plot_extinction_coefficients(df, show=False)
        compound = self._compound_edit.text().strip() or "compound"
        temp_c   = int(self._temperature_spin.value())
        self._plot.set_default_filename(f"{compound}_EC_{temp_c}C.png")
        self._plot.set_figure(fig)
        wl_min = int(df["Wavelength (nm)"].min())
        wl_max = int(df["Wavelength (nm)"].max())
        n_prep = sum(1 for c in df.columns
                     if c.startswith("Prep") and c.endswith("_Mean"))
        self._status_lbl.setText(
            f"Done — {n_prep} preparation(s), {wl_min}–{wl_max} nm.")
        self._stage3.set_status(DONE)
        self._save_csv_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._stage3.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_finished(self):
        self._run_btn.setEnabled(True)

    # ── Save CSV ──────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._df_result is None:
            return
        compound = self._compound_edit.text().strip() or "compound"
        temp_c   = int(self._temperature_spin.value())
        ts = self._df_result["Date"].iloc[0].replace(":", "-").replace(" ", "_")
        default_name = f"{compound}_EC_{temp_c}C_{ts}.csv"

        if self._output_path:
            out_dir = self._output_path / "extinction_coefficients" / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path.home()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(out_dir / default_name), "CSV (*.csv)")
        if path:
            self._df_result.to_csv(path, index=False)
            print(f"[EC] CSV saved → {path}")
            self.log_signal.emit(f"[EC] CSV saved → {path}", "INFO")

    # ── Preferences ───────────────────────────────────────────────────────

    def apply_prefs(self, prefs):
        ec = prefs.extinction_coeff
        self._compound_edit.blockSignals(True)
        self._solvent_edit.blockSignals(True)
        self._compound_edit.setText(ec.compound_name)
        self._solvent_edit.setText(ec.solvent)
        self._compound_edit.blockSignals(False)
        self._solvent_edit.blockSignals(False)
        self._path_length_spin.setValue(ec.path_length_cm)
        self._temperature_spin.setValue(ec.temperature_c)

    def collect_prefs(self, prefs):
        prefs.extinction_coeff.compound_name  = self._compound_edit.text()
        prefs.extinction_coeff.path_length_cm = self._path_length_spin.value()
        prefs.extinction_coeff.solvent        = self._solvent_edit.text()
        prefs.extinction_coeff.temperature_c  = self._temperature_spin.value()
