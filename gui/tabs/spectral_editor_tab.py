"""
Spectral Editor tab — interactive data editing for kinetic and scanning kinetic data.

Two sub-tabs:
  Kinetic          — multi-channel kinetic CSV (Cary multi-wavelength format)
  Scanning Kinetic — Cary 60 multi-scan CSV files

Operations are applied to all currently loaded files simultaneously.
Files added later start from their own original state.
Each file tracks a history stack for per-file undo and a full reset to original.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QFrame,
    QGroupBox, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QTabWidget,
    QSplitter, QMessageBox, QSizePolicy,
)

from gui.tabs.spectral_editor_core import (
    EditorFile, KineticData, ScanData,
    delete_channels, add_offset_manual_kinetic, add_offset_from_spectrum_kinetic,
    shift_time, rescale_time,
    combine_kinetic_side_by_side, combine_kinetic_concatenate, save_kinetic_csv,
    delete_scans, add_offset_scanning, add_offset_from_spectrum_scanning,
    align_at_wavelength, normalize_at_wavelength,
    combine_scanning, save_scanning_csv,
    parse_index_range, load_single_spectrum,
)
from gui.tabs.qy_core import load_kinetic_csv, load_spectra_csv
from gui.widgets.plot_widget import PlotWidget


# ── File-row widget ───────────────────────────────────────────────────────────

class _FileRow(QFrame):
    """One row in the file list showing filename, state summary and action buttons."""

    def __init__(self, ef: EditorFile,
                 on_undo, on_reset, on_remove, parent=None):
        super().__init__(parent)
        self._ef = ef
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { border-radius: 4px; padding: 2px; }")

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(6)

        self._lbl = QLabel()
        self._lbl.setStyleSheet("font-size: 9pt;")
        row.addWidget(self._lbl, stretch=1)

        self._undo_btn = QPushButton("Undo ↩")
        self._undo_btn.setFixedWidth(70)
        self._undo_btn.setToolTip("Undo last operation for this file")
        self._undo_btn.clicked.connect(on_undo)
        row.addWidget(self._undo_btn)

        reset_btn = QPushButton("Reset ⟳")
        reset_btn.setFixedWidth(70)
        reset_btn.setToolTip("Reset this file to original state")
        reset_btn.clicked.connect(on_reset)
        row.addWidget(reset_btn)

        remove_btn = QPushButton("Remove ✕")
        remove_btn.setFixedWidth(80)
        remove_btn.setToolTip("Remove this file from the editor")
        remove_btn.clicked.connect(on_remove)
        row.addWidget(remove_btn)

        self.refresh()

    def refresh(self):
        name = self._ef.path.name
        summary = self._ef.summary()
        hist = len(self._ef.history)
        hist_txt = f"  [{hist} edit{'s' if hist != 1 else ''}]" if hist else ""
        self._lbl.setText(f"{name} — {summary}{hist_txt}")
        self._undo_btn.setEnabled(self._ef.can_undo)


# ── Kinetic editor ────────────────────────────────────────────────────────────

class _KineticEditorWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files:    list[EditorFile] = []
        self._rows:     list[_FileRow]   = []
        self._out_dir:  Optional[Path]   = None
        self._spec_path: Optional[Path]  = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ── Controls (scroll area) ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        self._build_files_group(vbox)
        self._build_delete_group(vbox)
        self._build_offset_group(vbox)
        self._build_time_group(vbox)
        self._build_combine_group(vbox)
        self._build_save_group(vbox)
        vbox.addStretch()
        scroll.setWidget(container)
        splitter.addWidget(scroll)

        # ── Plot ──────────────────────────────────────────────────────────
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(3)
        refresh_btn = QPushButton("Update Preview")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self._refresh_plot)
        btn_row = QHBoxLayout()
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        plot_layout.addLayout(btn_row)
        self._plot = PlotWidget("Kinetic Preview",
                                "All channels of all loaded files in their current state.")
        self._plot.setMinimumHeight(260)
        plot_layout.addWidget(self._plot)
        splitter.addWidget(plot_panel)
        splitter.setSizes([460, 300])

    def _build_files_group(self, parent):
        grp = QGroupBox("Files")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        add_btn = QPushButton("+ Add kinetic CSV files…")
        add_btn.setFixedWidth(200)
        add_btn.clicked.connect(self._add_files)
        lay.addWidget(add_btn)

        self._file_rows_layout = QVBoxLayout()
        self._file_rows_layout.setSpacing(3)
        lay.addLayout(self._file_rows_layout)
        parent.addWidget(grp)

    def _build_delete_group(self, parent):
        grp = QGroupBox("Delete Channels")
        lay = QVBoxLayout(grp)
        hint = QLabel("Check channels to remove from all loaded files:")
        hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(hint)

        self._chan_list = QListWidget()
        self._chan_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._chan_list.setMaximumHeight(100)
        lay.addWidget(self._chan_list)

        btn = QPushButton("Apply Delete")
        btn.setFixedWidth(110)
        btn.clicked.connect(self._apply_delete_channels)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_offset_group(self, parent):
        grp = QGroupBox("Offset")
        lay = QVBoxLayout(grp)

        # Source
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._off_src_bg = QButtonGroup(self)
        self._off_manual_rb = QRadioButton("Manual value")
        self._off_spec_rb   = QRadioButton("Initial spectrum file")
        self._off_manual_rb.setChecked(True)
        self._off_src_bg.addButton(self._off_manual_rb, 0)
        self._off_src_bg.addButton(self._off_spec_rb,   1)
        src_row.addWidget(self._off_manual_rb)
        src_row.addWidget(self._off_spec_rb)
        src_row.addStretch()
        lay.addLayout(src_row)

        # Reference point
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference point:"))
        self._off_ref_bg   = QButtonGroup(self)
        self._off_t0_rb    = QRadioButton("t = 0")
        self._off_plat_rb  = QRadioButton("Plateau mean:")
        self._off_t0_rb.setChecked(True)
        self._off_ref_bg.addButton(self._off_t0_rb,   0)
        self._off_ref_bg.addButton(self._off_plat_rb, 1)
        ref_row.addWidget(self._off_t0_rb)
        ref_row.addWidget(self._off_plat_rb)
        self._off_plat_t0 = QDoubleSpinBox()
        self._off_plat_t0.setRange(-1e6, 1e6)
        self._off_plat_t0.setSuffix(" s")
        self._off_plat_t0.setFixedWidth(90)
        self._off_plat_t1 = QDoubleSpinBox()
        self._off_plat_t1.setRange(-1e6, 1e6)
        self._off_plat_t1.setSuffix(" s")
        self._off_plat_t1.setValue(30.0)
        self._off_plat_t1.setFixedWidth(90)
        ref_row.addWidget(self._off_plat_t0)
        ref_row.addWidget(QLabel("–"))
        ref_row.addWidget(self._off_plat_t1)
        ref_row.addStretch()
        lay.addLayout(ref_row)

        # Manual value
        self._off_manual_row = QHBoxLayout()
        self._off_manual_row.addWidget(QLabel("Offset value:"))
        self._off_val = QDoubleSpinBox()
        self._off_val.setRange(-1e6, 1e6)
        self._off_val.setDecimals(4)
        self._off_val.setFixedWidth(110)
        self._off_manual_row.addWidget(self._off_val)
        self._off_manual_row.addStretch()
        lay.addLayout(self._off_manual_row)

        # Spectrum file
        self._off_spec_row = QHBoxLayout()
        self._off_spec_lbl = QLabel("No file selected")
        self._off_spec_lbl.setStyleSheet("color:#888; font-size:8pt;")
        spec_btn = QPushButton("Browse…")
        spec_btn.setFixedWidth(70)
        spec_btn.clicked.connect(self._browse_offset_spectrum)
        self._off_spec_row.addWidget(QLabel("Spectrum:"))
        self._off_spec_row.addWidget(self._off_spec_lbl, stretch=1)
        self._off_spec_row.addWidget(spec_btn)
        lay.addLayout(self._off_spec_row)

        self._off_src_bg.buttonClicked.connect(self._update_offset_ui)
        self._update_offset_ui()

        btn = QPushButton("Apply Offset")
        btn.setFixedWidth(110)
        btn.clicked.connect(self._apply_offset)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_time_group(self, parent):
        grp = QGroupBox("Time Axis")
        lay = QVBoxLayout(grp)

        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Shift all times by:"))
        self._time_shift = QDoubleSpinBox()
        self._time_shift.setRange(-1e7, 1e7)
        self._time_shift.setSuffix(" s")
        self._time_shift.setFixedWidth(110)
        shift_btn = QPushButton("Apply Shift")
        shift_btn.setFixedWidth(100)
        shift_btn.clicked.connect(self._apply_shift)
        shift_row.addWidget(self._time_shift)
        shift_row.addWidget(shift_btn)
        shift_row.addStretch()
        lay.addLayout(shift_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Rescale time by factor:"))
        self._time_scale = QDoubleSpinBox()
        self._time_scale.setRange(-1e6, 1e6)
        self._time_scale.setValue(1.0)
        self._time_scale.setDecimals(4)
        self._time_scale.setFixedWidth(110)
        scale_btn = QPushButton("Apply Rescale")
        scale_btn.setFixedWidth(100)
        scale_btn.clicked.connect(self._apply_rescale)
        scale_row.addWidget(self._time_scale)
        scale_row.addWidget(scale_btn)
        scale_row.addStretch()
        lay.addLayout(scale_row)

        parent.addWidget(grp)

    def _build_combine_group(self, parent):
        grp = QGroupBox("Combine All Loaded Files")
        lay = QVBoxLayout(grp)

        mode_row = QHBoxLayout()
        self._comb_mode_bg  = QButtonGroup(self)
        self._comb_side_rb  = QRadioButton("Side by side (merge channels)")
        self._comb_cat_rb   = QRadioButton("Concatenate (time series)")
        self._comb_side_rb.setChecked(True)
        self._comb_mode_bg.addButton(self._comb_side_rb, 0)
        self._comb_mode_bg.addButton(self._comb_cat_rb,  1)
        mode_row.addWidget(self._comb_side_rb)
        mode_row.addWidget(self._comb_cat_rb)
        mode_row.addStretch()
        lay.addLayout(mode_row)

        join_row = QHBoxLayout()
        self._comb_join_bg    = QButtonGroup(self)
        self._comb_join_auto  = QRadioButton("Auto join")
        self._comb_join_man   = QRadioButton("Join at:")
        self._comb_join_auto.setChecked(True)
        self._comb_join_bg.addButton(self._comb_join_auto, 0)
        self._comb_join_bg.addButton(self._comb_join_man,  1)
        self._comb_join_t = QDoubleSpinBox()
        self._comb_join_t.setRange(0, 1e9)
        self._comb_join_t.setSuffix(" s")
        self._comb_join_t.setFixedWidth(100)
        join_row.addWidget(self._comb_join_auto)
        join_row.addWidget(self._comb_join_man)
        join_row.addWidget(self._comb_join_t)
        join_row.addStretch()
        lay.addLayout(join_row)

        self._comb_mode_bg.buttonClicked.connect(self._update_combine_ui)
        self._update_combine_ui()

        hint = QLabel("The combined result replaces all loaded files as a single entry.")
        hint.setStyleSheet("color:#888; font-size:8pt;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btn = QPushButton("Combine All Files")
        btn.setFixedWidth(130)
        btn.clicked.connect(self._apply_combine)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_save_group(self, parent):
        grp = QGroupBox("Save")
        lay = QVBoxLayout(grp)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self._save_dir_lbl = QLabel("(same as each file's source)")
        self._save_dir_lbl.setStyleSheet("color:#555; font-size:8pt;")
        dir_row.addWidget(self._save_dir_lbl, stretch=1)
        dir_browse_btn = QPushButton("Browse…")
        dir_browse_btn.setFixedWidth(70)
        dir_browse_btn.clicked.connect(self._browse_save_dir)
        dir_row.addWidget(dir_browse_btn)
        lay.addLayout(dir_row)

        name_hint = QLabel(
            "Filename (editable; for multiple files a Save dialog opens per file):")
        name_hint.setStyleSheet("color:#888; font-size:8pt;")
        name_hint.setWordWrap(True)
        lay.addWidget(name_hint)

        name_row = QHBoxLayout()
        self._save_name_edit = QLineEdit()
        self._save_name_edit.setPlaceholderText("e.g. my_experiment_edited.csv")
        name_row.addWidget(self._save_name_edit, stretch=1)
        lay.addLayout(name_row)

        save_btn = QPushButton("Save All Files")
        save_btn.setFixedWidth(120)
        save_btn.clicked.connect(self._save_all)
        lay.addWidget(save_btn)
        parent.addWidget(grp)

    # ── File management ───────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select kinetic CSV files", "",
            "CSV files (*.csv *.txt);;All files (*)")
        if not paths:
            return
        for p_str in paths:
            path = Path(p_str)
            try:
                data = load_kinetic_csv(path)
                if not data:
                    raise ValueError("No channels loaded.")
                ef = EditorFile(path=path, data_type="kinetic", original=data)
                self._files.append(ef)
            except Exception as e:
                QMessageBox.warning(self, "Load error",
                                    f"Could not load {path.name}:\n{e}")
        self._refresh_all()

    def _on_undo(self, ef: EditorFile):
        if not ef.undo():
            QMessageBox.information(self, "Undo", "Nothing to undo for this file.")
        self._refresh_all()

    def _on_reset(self, ef: EditorFile):
        ef.reset()
        self._refresh_all()

    def _on_remove(self, ef: EditorFile):
        self._files.remove(ef)
        self._refresh_all()

    # ── Operation slots ───────────────────────────────────────────────────

    def _apply_delete_channels(self):
        if not self._files:
            return
        labels = []
        for i in range(self._chan_list.count()):
            item = self._chan_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                labels.append(item.text())
        if not labels:
            QMessageBox.information(self, "Delete", "No channels checked.")
            return
        for ef in self._files:
            ef.push_and_apply(delete_channels(ef.current, labels))
        self._refresh_all()

    def _apply_offset(self):
        if not self._files:
            return
        ref = "plateau" if self._off_plat_rb.isChecked() else "t0"
        plateau = (self._off_plat_t0.value(), self._off_plat_t1.value()) if ref == "plateau" else None
        try:
            if self._off_manual_rb.isChecked():
                val = self._off_val.value()
                for ef in self._files:
                    ef.push_and_apply(
                        add_offset_manual_kinetic(ef.current, val, ref, plateau))
            else:
                if self._spec_path is None:
                    QMessageBox.warning(self, "Offset", "No spectrum file selected.")
                    return
                spec_wl, spec_ab = load_single_spectrum(self._spec_path)
                for ef in self._files:
                    ef.push_and_apply(
                        add_offset_from_spectrum_kinetic(ef.current, spec_wl, spec_ab, ref, plateau))
        except Exception as e:
            QMessageBox.warning(self, "Offset error", str(e))
            return
        self._refresh_all()

    def _apply_shift(self):
        if not self._files:
            return
        shift = self._time_shift.value()
        for ef in self._files:
            ef.push_and_apply(shift_time(ef.current, shift))
        self._refresh_all()

    def _apply_rescale(self):
        if not self._files:
            return
        factor = self._time_scale.value()
        try:
            for ef in self._files:
                ef.push_and_apply(rescale_time(ef.current, factor))
        except ValueError as e:
            QMessageBox.warning(self, "Rescale error", str(e))
            return
        self._refresh_all()

    def _apply_combine(self):
        if len(self._files) < 2:
            QMessageBox.information(self, "Combine", "Load at least 2 files to combine.")
            return
        try:
            if self._comb_side_rb.isChecked():
                combined = combine_kinetic_side_by_side([ef.current for ef in self._files])
            else:
                join_mode = "auto" if self._comb_join_auto.isChecked() else "manual"
                join_time = self._comb_join_t.value() if join_mode == "manual" else None
                combined  = combine_kinetic_concatenate(
                    [ef.current for ef in self._files], join_mode, join_time)
        except Exception as e:
            QMessageBox.warning(self, "Combine error", str(e))
            return

        first_path = self._files[0].path
        stem       = "_".join(ef.path.stem for ef in self._files[:3])
        if len(self._files) > 3:
            stem += f"_and_{len(self._files)-3}_more"
        comb_path = first_path.parent / f"{stem}_combined.csv"
        ef_new = EditorFile(path=comb_path, data_type="kinetic",
                            original=copy.deepcopy(combined), current=combined)
        self._files = [ef_new]
        self._refresh_all()

    def _save_all(self):
        if not self._files:
            return
        saved, failed = [], []
        if len(self._files) == 1:
            ef       = self._files[0]
            out_dir  = self._out_dir or ef.path.parent
            name     = self._save_name_edit.text().strip() or ef.suggested_output_name()
            if not name.lower().endswith(".csv"):
                name += ".csv"
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save file", str(out_dir / name),
                "CSV files (*.csv);;All files (*)")
            if out_path:
                try:
                    save_kinetic_csv(ef.current, Path(out_path))
                    saved.append(Path(out_path).name)
                except Exception as e:
                    failed.append(str(e))
        else:
            for ef in self._files:
                out_dir  = self._out_dir or ef.path.parent
                name     = ef.suggested_output_name()
                out_path, _ = QFileDialog.getSaveFileName(
                    self, f"Save — {ef.path.name}", str(out_dir / name),
                    "CSV files (*.csv);;All files (*)")
                if out_path:
                    try:
                        save_kinetic_csv(ef.current, Path(out_path))
                        saved.append(Path(out_path).name)
                    except Exception as e:
                        failed.append(f"{ef.path.name}: {e}")
        if saved or failed:
            msg = ""
            if saved:
                msg += "Saved:\n" + "\n".join(f"  {s}" for s in saved)
            if failed:
                msg += "\n\nFailed:\n" + "\n".join(f"  {f}" for f in failed)
            QMessageBox.information(self, "Save", msg.strip())

    def _refresh_save_name(self):
        if len(self._files) == 1:
            self._save_name_edit.setText(self._files[0].suggested_output_name())
            self._save_name_edit.setEnabled(True)
        elif self._files:
            self._save_name_edit.setPlaceholderText(
                "Multiple files — a Save dialog opens per file")
            self._save_name_edit.clear()
            self._save_name_edit.setEnabled(False)
        else:
            self._save_name_edit.clear()
            self._save_name_edit.setEnabled(True)

    # ── Browse helpers ────────────────────────────────────────────────────

    def _browse_offset_spectrum(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select initial spectrum", "",
            "CSV files (*.csv *.txt);;All files (*)")
        if p:
            self._spec_path = Path(p)
            self._off_spec_lbl.setText(self._spec_path.name)
            self._off_spec_lbl.setStyleSheet("color:#333; font-size:8pt;")

    def _browse_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir = Path(d)
            self._save_dir_lbl.setText(str(self._out_dir))

    # ── UI state helpers ──────────────────────────────────────────────────

    def _update_offset_ui(self):
        manual = self._off_manual_rb.isChecked()
        # Show/hide relevant row by enabling/disabling widgets
        for i in range(self._off_manual_row.count()):
            w = self._off_manual_row.itemAt(i).widget()
            if w:
                w.setVisible(manual)
        for i in range(self._off_spec_row.count()):
            w = self._off_spec_row.itemAt(i).widget()
            if w:
                w.setVisible(not manual)

    def _update_combine_ui(self):
        cat = self._comb_cat_rb.isChecked()
        self._comb_join_auto.setEnabled(cat)
        self._comb_join_man.setEnabled(cat)
        self._comb_join_t.setEnabled(cat)

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh_all(self):
        self._refresh_file_list()
        self._refresh_channel_list()
        self._refresh_save_name()
        self._refresh_plot()

    def _refresh_file_list(self):
        # Clear existing rows
        while self._file_rows_layout.count():
            item = self._file_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for ef in self._files:
            row = _FileRow(
                ef,
                on_undo=lambda _checked=False, e=ef: self._on_undo(e),
                on_reset=lambda _checked=False, e=ef: self._on_reset(e),
                on_remove=lambda _checked=False, e=ef: self._on_remove(e),
            )
            self._rows.append(row)
            self._file_rows_layout.addWidget(row)

    def _refresh_channel_list(self):
        self._chan_list.clear()
        seen = set()
        for ef in self._files:
            for label in ef.current:
                if label not in seen:
                    seen.add(label)
                    item = QListWidgetItem(label)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self._chan_list.addItem(item)

    def _refresh_plot(self):
        if not self._files:
            self._plot.clear()
            return
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_idx = 0
        for ef in self._files:
            for label, (t, a) in ef.current.items():
                ax.plot(t, a, label=f"{ef.path.stem} / {label}",
                        color=colors[c_idx % len(colors)], lw=1.2)
                c_idx += 1
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absorbance")
        ax.legend(fontsize=7, ncol=2)
        self._plot.set_figure(fig)


# ── Scanning editor ───────────────────────────────────────────────────────────

class _ScanningEditorWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files:    list[EditorFile] = []
        self._rows:     list[_FileRow]   = []
        self._out_dir:  Optional[Path]   = None
        self._spec_path: Optional[Path]  = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        self._build_files_group(vbox)
        self._build_delete_group(vbox)
        self._build_offset_group(vbox)
        self._build_align_group(vbox)
        self._build_normalize_group(vbox)
        self._build_combine_group(vbox)
        self._build_save_group(vbox)
        vbox.addStretch()
        scroll.setWidget(container)
        splitter.addWidget(scroll)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(3)
        refresh_btn = QPushButton("Update Preview")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self._refresh_plot)
        btn_row = QHBoxLayout()
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        plot_layout.addLayout(btn_row)
        self._plot = PlotWidget("Scanning Preview",
                                "All scans of all loaded files. Colour: blue (first) → red (last).")
        self._plot.setMinimumHeight(260)
        plot_layout.addWidget(self._plot)
        splitter.addWidget(plot_panel)
        splitter.setSizes([460, 300])

    def _build_files_group(self, parent):
        grp = QGroupBox("Files")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        add_btn = QPushButton("+ Add scanning CSV files…")
        add_btn.setFixedWidth(200)
        add_btn.clicked.connect(self._add_files)
        lay.addWidget(add_btn)

        self._file_rows_layout = QVBoxLayout()
        self._file_rows_layout.setSpacing(3)
        lay.addLayout(self._file_rows_layout)
        parent.addWidget(grp)

    def _build_delete_group(self, parent):
        grp = QGroupBox("Delete Scans")
        lay = QVBoxLayout(grp)

        list_hint = QLabel("Toggle scans to mark for deletion:")
        list_hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(list_hint)

        self._del_scan_list = QListWidget()
        self._del_scan_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._del_scan_list.setMaximumHeight(150)
        lay.addWidget(self._del_scan_list)

        sel_row = QHBoxLayout()
        sel_all_btn = QPushButton("Select all")
        sel_all_btn.setFixedWidth(80)
        sel_all_btn.clicked.connect(self._del_select_all)
        sel_none_btn = QPushButton("Select none")
        sel_none_btn.setFixedWidth(80)
        sel_none_btn.clicked.connect(self._del_select_none)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(sel_none_btn)
        sel_row.addStretch()
        lay.addLayout(sel_row)

        range_hint = QLabel("Additional selection by index range (e.g. 0, 2, 5-10):")
        range_hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(range_hint)

        row = QHBoxLayout()
        self._del_idx_edit = QLineEdit()
        self._del_idx_edit.setPlaceholderText("e.g. 0, 2, 5-10  (combined with list above)")
        row.addWidget(self._del_idx_edit)
        btn = QPushButton("Apply Delete")
        btn.setFixedWidth(110)
        btn.clicked.connect(self._apply_delete_scans)
        row.addWidget(btn)
        lay.addLayout(row)
        parent.addWidget(grp)

    def _build_offset_group(self, parent):
        grp = QGroupBox("Offset")
        lay = QVBoxLayout(grp)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._off_src_bg    = QButtonGroup(self)
        self._off_manual_rb = QRadioButton("Manual value")
        self._off_spec_rb   = QRadioButton("Initial spectrum file")
        self._off_manual_rb.setChecked(True)
        self._off_src_bg.addButton(self._off_manual_rb, 0)
        self._off_src_bg.addButton(self._off_spec_rb,   1)
        src_row.addWidget(self._off_manual_rb)
        src_row.addWidget(self._off_spec_rb)
        src_row.addStretch()
        lay.addLayout(src_row)

        self._off_manual_row = QHBoxLayout()
        self._off_manual_row.addWidget(QLabel("Offset value:"))
        self._off_val = QDoubleSpinBox()
        self._off_val.setRange(-1e6, 1e6)
        self._off_val.setDecimals(4)
        self._off_val.setFixedWidth(110)
        self._off_manual_row.addWidget(self._off_val)
        self._off_manual_row.addStretch()
        lay.addLayout(self._off_manual_row)

        self._off_spec_row = QHBoxLayout()
        self._off_spec_lbl = QLabel("No file selected")
        self._off_spec_lbl.setStyleSheet("color:#888; font-size:8pt;")
        spec_btn = QPushButton("Browse…")
        spec_btn.setFixedWidth(70)
        spec_btn.clicked.connect(self._browse_offset_spectrum)
        self._off_spec_row.addWidget(QLabel("Spectrum:"))
        self._off_spec_row.addWidget(self._off_spec_lbl, stretch=1)
        self._off_spec_row.addWidget(spec_btn)
        lay.addLayout(self._off_spec_row)

        self._off_src_bg.buttonClicked.connect(self._update_offset_ui)
        self._update_offset_ui()

        btn = QPushButton("Apply Offset")
        btn.setFixedWidth(110)
        btn.clicked.connect(self._apply_offset)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_align_group(self, parent):
        grp = QGroupBox("Align at Wavelength")
        lay = QVBoxLayout(grp)
        hint = QLabel(
            "Shift each scan by a scalar so all spectra share the same "
            "value at the chosen wavelength.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Wavelength:"))
        self._align_wl = QDoubleSpinBox()
        self._align_wl.setRange(190, 1100)
        self._align_wl.setValue(400.0)
        self._align_wl.setSuffix(" nm")
        self._align_wl.setFixedWidth(100)
        row.addWidget(self._align_wl)

        row.addWidget(QLabel("Target:"))
        self._align_bg   = QButtonGroup(self)
        self._align_first = QRadioButton("First scan's value")
        self._align_zero  = QRadioButton("Zero")
        self._align_first.setChecked(True)
        self._align_bg.addButton(self._align_first, 0)
        self._align_bg.addButton(self._align_zero,  1)
        row.addWidget(self._align_first)
        row.addWidget(self._align_zero)
        row.addStretch()
        lay.addLayout(row)

        btn = QPushButton("Apply Align")
        btn.setFixedWidth(100)
        btn.clicked.connect(self._apply_align)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_normalize_group(self, parent):
        grp = QGroupBox("Normalize at Wavelength")
        lay = QVBoxLayout(grp)
        hint = QLabel(
            "Divide each selected scan by its absorbance at the chosen wavelength.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(hint)

        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Wavelength:"))
        self._norm_wl = QDoubleSpinBox()
        self._norm_wl.setRange(190, 1100)
        self._norm_wl.setValue(400.0)
        self._norm_wl.setSuffix(" nm")
        self._norm_wl.setFixedWidth(100)
        wl_row.addWidget(self._norm_wl)
        wl_row.addStretch()
        lay.addLayout(wl_row)

        sel_row = QHBoxLayout()
        self._norm_all_chk = QCheckBox("Apply to all scans")
        self._norm_all_chk.setChecked(True)
        self._norm_all_chk.stateChanged.connect(self._update_norm_ui)
        sel_row.addWidget(self._norm_all_chk)
        sel_row.addWidget(QLabel("  or indices:"))
        self._norm_idx_edit = QLineEdit()
        self._norm_idx_edit.setPlaceholderText("e.g. 0-5, 10")
        self._norm_idx_edit.setEnabled(False)
        sel_row.addWidget(self._norm_idx_edit, stretch=1)
        lay.addLayout(sel_row)

        btn = QPushButton("Apply Normalize")
        btn.setFixedWidth(130)
        btn.clicked.connect(self._apply_normalize)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_combine_group(self, parent):
        grp = QGroupBox("Combine All Loaded Files")
        lay = QVBoxLayout(grp)
        hint = QLabel(
            "Appends all scan lists in load order. "
            "The combined result replaces all loaded files.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:8pt;")
        lay.addWidget(hint)

        btn = QPushButton("Combine All Files")
        btn.setFixedWidth(130)
        btn.clicked.connect(self._apply_combine)
        lay.addWidget(btn)
        parent.addWidget(grp)

    def _build_save_group(self, parent):
        grp = QGroupBox("Save")
        lay = QVBoxLayout(grp)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self._save_dir_lbl = QLabel("(same as each file's source)")
        self._save_dir_lbl.setStyleSheet("color:#555; font-size:8pt;")
        dir_row.addWidget(self._save_dir_lbl, stretch=1)
        dir_browse_btn = QPushButton("Browse…")
        dir_browse_btn.setFixedWidth(70)
        dir_browse_btn.clicked.connect(self._browse_save_dir)
        dir_row.addWidget(dir_browse_btn)
        lay.addLayout(dir_row)

        name_hint = QLabel(
            "Filename (editable; for multiple files a Save dialog opens per file):")
        name_hint.setStyleSheet("color:#888; font-size:8pt;")
        name_hint.setWordWrap(True)
        lay.addWidget(name_hint)

        name_row = QHBoxLayout()
        self._save_name_edit = QLineEdit()
        self._save_name_edit.setPlaceholderText("e.g. my_experiment_edited.csv")
        name_row.addWidget(self._save_name_edit, stretch=1)
        lay.addLayout(name_row)

        save_btn = QPushButton("Save All Files")
        save_btn.setFixedWidth(120)
        save_btn.clicked.connect(self._save_all)
        lay.addWidget(save_btn)
        parent.addWidget(grp)

    # ── File management ───────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select scanning kinetic CSV files", "",
            "CSV files (*.csv *.txt);;All files (*)")
        if not paths:
            return
        for p_str in paths:
            path = Path(p_str)
            try:
                scans = load_spectra_csv(path)
                if not scans:
                    raise ValueError("No scans loaded.")
                ef = EditorFile(path=path, data_type="scanning", original=scans)
                self._files.append(ef)
            except Exception as e:
                QMessageBox.warning(self, "Load error",
                                    f"Could not load {path.name}:\n{e}")
        self._refresh_all()

    def _on_undo(self, ef: EditorFile):
        if not ef.undo():
            QMessageBox.information(self, "Undo", "Nothing to undo for this file.")
        self._refresh_all()

    def _on_reset(self, ef: EditorFile):
        ef.reset()
        self._refresh_all()

    def _on_remove(self, ef: EditorFile):
        self._files.remove(ef)
        self._refresh_all()

    # ── Operation slots ───────────────────────────────────────────────────

    def _apply_delete_scans(self):
        if not self._files:
            return
        # Collect checked indices from the toggle list
        list_indices = set()
        for i in range(self._del_scan_list.count()):
            item = self._del_scan_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                list_indices.add(i)
        # Also parse the text range input
        text = self._del_idx_edit.text().strip()
        for ef in self._files:
            n = len(ef.current)
            text_indices = set(parse_index_range(text, n - 1)) if text else set()
            indices = sorted(list_indices | text_indices)
            indices = [i for i in indices if i < n]
            if indices:
                ef.push_and_apply(delete_scans(ef.current, indices))
        self._del_select_none()
        self._del_idx_edit.clear()
        self._refresh_all()

    def _del_select_all(self):
        for i in range(self._del_scan_list.count()):
            self._del_scan_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _del_select_none(self):
        for i in range(self._del_scan_list.count()):
            self._del_scan_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _apply_offset(self):
        if not self._files:
            return
        try:
            if self._off_manual_rb.isChecked():
                val = self._off_val.value()
                for ef in self._files:
                    ef.push_and_apply(add_offset_scanning(ef.current, val))
            else:
                if self._spec_path is None:
                    QMessageBox.warning(self, "Offset", "No spectrum file selected.")
                    return
                ref_wl, ref_ab = load_single_spectrum(self._spec_path)
                for ef in self._files:
                    ef.push_and_apply(
                        add_offset_from_spectrum_scanning(ef.current, ref_wl, ref_ab))
        except Exception as e:
            QMessageBox.warning(self, "Offset error", str(e))
            return
        self._refresh_all()

    def _apply_align(self):
        if not self._files:
            return
        wl_ref = self._align_wl.value()
        target = "zero" if self._align_zero.isChecked() else "first"
        try:
            for ef in self._files:
                ef.push_and_apply(align_at_wavelength(ef.current, wl_ref, target))
        except Exception as e:
            QMessageBox.warning(self, "Align error", str(e))
            return
        self._refresh_all()

    def _apply_normalize(self):
        if not self._files:
            return
        wl_norm = self._norm_wl.value()
        all_scans = self._norm_all_chk.isChecked()
        try:
            for ef in self._files:
                indices = None if all_scans else parse_index_range(
                    self._norm_idx_edit.text(), len(ef.current) - 1)
                ef.push_and_apply(normalize_at_wavelength(ef.current, wl_norm, indices))
        except Exception as e:
            QMessageBox.warning(self, "Normalize error", str(e))
            return
        self._refresh_all()

    def _apply_combine(self):
        if len(self._files) < 2:
            QMessageBox.information(self, "Combine", "Load at least 2 files to combine.")
            return
        combined  = combine_scanning([ef.current for ef in self._files])
        first_path = self._files[0].path
        stem       = "_".join(ef.path.stem for ef in self._files[:3])
        if len(self._files) > 3:
            stem += f"_and_{len(self._files)-3}_more"
        comb_path = first_path.parent / f"{stem}_combined.csv"
        ef_new = EditorFile(path=comb_path, data_type="scanning",
                            original=copy.deepcopy(combined), current=combined)
        self._files = [ef_new]
        self._refresh_all()

    def _save_all(self):
        if not self._files:
            return
        saved, failed = [], []
        if len(self._files) == 1:
            ef       = self._files[0]
            out_dir  = self._out_dir or ef.path.parent
            name     = self._save_name_edit.text().strip() or ef.suggested_output_name()
            if not name.lower().endswith(".csv"):
                name += ".csv"
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save file", str(out_dir / name),
                "CSV files (*.csv);;All files (*)")
            if out_path:
                try:
                    save_scanning_csv(ef.current, Path(out_path))
                    saved.append(Path(out_path).name)
                except Exception as e:
                    failed.append(str(e))
        else:
            for ef in self._files:
                out_dir  = self._out_dir or ef.path.parent
                name     = ef.suggested_output_name()
                out_path, _ = QFileDialog.getSaveFileName(
                    self, f"Save — {ef.path.name}", str(out_dir / name),
                    "CSV files (*.csv);;All files (*)")
                if out_path:
                    try:
                        save_scanning_csv(ef.current, Path(out_path))
                        saved.append(Path(out_path).name)
                    except Exception as e:
                        failed.append(f"{ef.path.name}: {e}")
        if saved or failed:
            msg = ""
            if saved:
                msg += "Saved:\n" + "\n".join(f"  {s}" for s in saved)
            if failed:
                msg += "\n\nFailed:\n" + "\n".join(f"  {f}" for f in failed)
            QMessageBox.information(self, "Save", msg.strip())

    def _refresh_save_name(self):
        if len(self._files) == 1:
            self._save_name_edit.setText(self._files[0].suggested_output_name())
            self._save_name_edit.setEnabled(True)
        elif self._files:
            self._save_name_edit.setPlaceholderText(
                "Multiple files — a Save dialog opens per file")
            self._save_name_edit.clear()
            self._save_name_edit.setEnabled(False)
        else:
            self._save_name_edit.clear()
            self._save_name_edit.setEnabled(True)

    # ── Browse helpers ────────────────────────────────────────────────────

    def _browse_offset_spectrum(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select reference spectrum", "",
            "CSV files (*.csv *.txt);;All files (*)")
        if p:
            self._spec_path = Path(p)
            self._off_spec_lbl.setText(self._spec_path.name)
            self._off_spec_lbl.setStyleSheet("color:#333; font-size:8pt;")

    def _browse_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir = Path(d)
            self._save_dir_lbl.setText(str(self._out_dir))

    # ── UI state helpers ──────────────────────────────────────────────────

    def _update_offset_ui(self):
        manual = self._off_manual_rb.isChecked()
        for i in range(self._off_manual_row.count()):
            w = self._off_manual_row.itemAt(i).widget()
            if w:
                w.setVisible(manual)
        for i in range(self._off_spec_row.count()):
            w = self._off_spec_row.itemAt(i).widget()
            if w:
                w.setVisible(not manual)

    def _update_norm_ui(self):
        self._norm_idx_edit.setEnabled(not self._norm_all_chk.isChecked())

    # ── Refresh ───────────────────────────────────────────────────────────

    def _refresh_all(self):
        self._refresh_file_list()
        self._refresh_scan_list()
        self._refresh_save_name()
        self._refresh_plot()

    def _refresh_scan_list(self):
        self._del_scan_list.clear()
        if not self._files:
            return
        max_scans = max(len(ef.current) for ef in self._files)
        for i in range(max_scans):
            # Build a label showing wavelength range from the first file that has this scan
            wl_info = ""
            for ef in self._files:
                if i < len(ef.current):
                    wl, _ = ef.current[i]
                    wl_info = f"  λ: {wl.min():.0f}–{wl.max():.0f} nm"
                    break
            item = QListWidgetItem(f"Scan {i}{wl_info}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._del_scan_list.addItem(item)

    def _refresh_file_list(self):
        while self._file_rows_layout.count():
            item = self._file_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()
        for ef in self._files:
            row = _FileRow(
                ef,
                on_undo=lambda _checked=False, e=ef: self._on_undo(e),
                on_reset=lambda _checked=False, e=ef: self._on_reset(e),
                on_remove=lambda _checked=False, e=ef: self._on_remove(e),
            )
            self._rows.append(row)
            self._file_rows_layout.addWidget(row)

    def _refresh_plot(self):
        if not self._files:
            self._plot.clear()
            return
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        cmap = cm.get_cmap("coolwarm")
        for ef in self._files:
            scans = ef.current
            n = len(scans)
            for i, (wl, ab) in enumerate(scans):
                color = cmap(i / max(n - 1, 1))
                ax.plot(wl, ab, color=color, lw=0.8, alpha=0.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        self._plot.set_figure(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SpectralEditorTab — outer container
# ══════════════════════════════════════════════════════════════════════════════

class SpectralEditorTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.addTab(_KineticEditorWidget(),  "Kinetic")
        tabs.addTab(_ScanningEditorWidget(), "Scanning Kinetic")
        layout.addWidget(tabs)
