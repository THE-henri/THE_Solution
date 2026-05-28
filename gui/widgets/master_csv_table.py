from pathlib import Path
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QFileDialog, QHeaderView, QAbstractItemView,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# ── Half-life defaults (kept for backward compatibility) ─────────────────────

COLUMNS = [
    "File", "Segment", "Wavelength", "Type", "Temperature_C",
    "Switch", "A0", "A_inf", "k", "Half_life_s", "R2",
]

_HEADER_LABELS = [
    "File", "Segment", "Wavelength", "Type", "T (°C)",
    "Switch", "A₀", "A∞", "k (s⁻¹)", "t½ (s)", "R²",
]

_FLOAT_COLS = {"A0", "A_inf", "k", "Half_life_s", "R2"}
_FLOAT_FMT  = {"A0": ".4f", "A_inf": ".4f", "k": ".6f",
               "Half_life_s": ".2f", "R2": ".6f"}

# ── Row colours (shared) ─────────────────────────────────────────────────────

_BG_LOADED   = QColor("#1e1e2e")
_BG_PENDING  = QColor("#0d3050")
_BG_APPENDED = QColor("#1a3520")

_FG_LOADED   = QColor("#ffffff")
_FG_PENDING  = QColor("#ff4444")
_FG_APPENDED = QColor("#44ff88")


class MasterCsvTable(QWidget):
    """
    Generic append-and-save table for master CSV files.

    Three visual states:
      • Loaded   (dark)  — rows read from an existing CSV on disk
      • Pending  (blue)  — new calculation results, not yet committed
      • Appended (green) — committed; included in the next Save

    Workflow
    --------
    1. Run a calculation  → pending rows replace any previous pending rows.
    2. Re-run             → previous pending rows discarded, new ones appear.
    3. Click Append       → pending rows turn green and are included in Save.
    4. Click Save         → loaded + appended rows written to disk.

    Parameters
    ----------
    columns        : column keys for the CSV (defaults to half-life COLUMNS)
    header_labels  : display headers matching columns
    float_cols     : set of column keys to format as floats
    float_fmt      : dict of column key → format string
    csv_subdir     : subdirectory under output_path where the CSV lives
    csv_filename   : CSV filename
    title_text     : label shown above the table
    sort_column    : column to sort by on save/reload (or None)
    """

    def __init__(self,
                 columns=None,
                 header_labels=None,
                 float_cols=None,
                 float_fmt=None,
                 csv_subdir="half_life/results",
                 csv_filename="half_life_master.csv",
                 title_text=None,
                 sort_column=None,
                 pre_save_hook=None,
                 parent=None):
        super().__init__(parent)
        self._columns       = list(columns       or COLUMNS)
        self._header_labels = list(header_labels or _HEADER_LABELS)
        self._float_cols    = set(float_cols     or _FLOAT_COLS)
        self._float_fmt     = dict(float_fmt     or _FLOAT_FMT)
        self._pre_save_hook = pre_save_hook
        self._csv_subdir    = csv_subdir
        self._csv_filename  = csv_filename
        self._title_text    = title_text or f"Master CSV  —  {csv_filename}"
        self._sort_column   = sort_column
        self._output_path:  Path | None = None
        self._n_committed:  int         = 0
        self._pending:      list[dict]  = []
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel(self._title_text)
        lbl.setStyleSheet("color:#888; font-size:9pt;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        btn_reload = QPushButton("🔄 Reload")
        btn_reload.setObjectName("plot_tool_btn")
        btn_reload.setFixedHeight(24)
        btn_reload.clicked.connect(self.reload)
        hdr.addWidget(btn_reload)
        root.addLayout(hdr)

        self._table = QTableWidget(0, len(self._columns) + 1)
        self._table.setHorizontalHeaderLabels(self._header_labels + [""])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.setMinimumHeight(160)
        root.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_append = QPushButton("↓ Append new results")
        self._btn_append.setObjectName("plot_tool_btn")
        self._btn_append.setFixedHeight(28)
        self._btn_append.setEnabled(False)
        self._btn_append.clicked.connect(self._append_pending)
        btn_row.addWidget(self._btn_append)

        self._btn_save = QPushButton("💾 Save master CSV")
        self._btn_save.setObjectName("accent")
        self._btn_save.setFixedHeight(28)
        self._btn_save.clicked.connect(self.save)
        btn_row.addWidget(self._btn_save)
        root.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#3cb371; font-size:9pt;")
        root.addWidget(self._status_lbl)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        self.reload()

    def reload(self):
        """Re-read the master CSV from output_path and repopulate the table."""
        self._table.setRowCount(0)
        self._n_committed = 0
        self._pending.clear()
        self._btn_append.setEnabled(False)

        if self._output_path is None:
            return
        csv = self._output_path / self._csv_subdir / self._csv_filename
        if not csv.exists():
            self._set_status("No master CSV found yet.", ok=None)
            return
        try:
            df = pd.read_csv(csv)
            if self._sort_column and self._sort_column in df.columns:
                df = df.sort_values(self._sort_column, ignore_index=True)
            for _, row in df.iterrows():
                self._add_row({c: row.get(c, "") for c in self._columns},
                              _BG_LOADED, _FG_LOADED)
            self._n_committed = self._table.rowCount()
            self._set_status(f"Loaded {len(df)} rows.", ok=True)
        except Exception as exc:
            self._set_status(f"Error reading CSV: {exc}", ok=False)

    def add_pending(self, result_dicts):
        """
        Show new result rows as pending (blue). Re-running replaces old pending rows.
        Accepts a single dict or a list of dicts.
        """
        if isinstance(result_dicts, dict):
            result_dicts = [result_dicts]

        # Normalise key used by scanning kinetics
        for d in result_dicts:
            if "Wavelength_nm" in d and "Wavelength" not in d:
                d["Wavelength"] = d.pop("Wavelength_nm")

        while self._table.rowCount() > self._n_committed:
            self._table.removeRow(self._table.rowCount() - 1)
        self._pending.clear()

        for d in result_dicts:
            self._pending.append(d)
            self._add_row(d, _BG_PENDING, _FG_PENDING)
        self._btn_append.setEnabled(True)
        self._set_status(
            f"{len(self._pending)} new result(s) pending — click Append to commit.",
            ok=None)

    def save(self):
        """Write committed rows to disk. Pending rows are excluded."""
        if self._output_path is None:
            QMessageBox.warning(self, "No output folder",
                                "Please set an output folder first.")
            return
        rows = self._collect_committed_rows()
        if not rows:
            self._set_status("Nothing to save — append results first.", ok=None)
            return
        df = pd.DataFrame(rows, columns=self._columns)
        if self._sort_column and self._sort_column in df.columns:
            df[self._sort_column] = pd.to_numeric(
                df[self._sort_column], errors="coerce")
            df = df.sort_values(self._sort_column, ignore_index=True)
        if self._pre_save_hook is not None:
            df = self._pre_save_hook(df)

        out_dir = self._output_path / self._csv_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self._csv_filename
        df.to_csv(out_path, index=False)
        self._set_status(f"Saved {len(df)} rows → {out_path.name}", ok=True)
        print(f"Master CSV saved → {out_path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append_pending(self):
        if not self._pending:
            return
        n_pending = len(self._pending)
        start = self._n_committed
        end   = self._table.rowCount()
        for r in range(start, end):
            self._set_row_color(r, _BG_APPENDED, _FG_APPENDED)
        self._n_committed = end
        self._pending.clear()
        self._btn_append.setEnabled(False)
        self._set_status(
            f"{n_pending} result(s) appended. Click Save to write to disk.",
            ok=None)

    def _fmt(self, col: str, val) -> str:
        if pd.isna(val):
            return "—"
        if col in self._float_cols:
            try:
                return format(float(val), self._float_fmt.get(col, ".4f"))
            except (TypeError, ValueError):
                pass
        return str(val)

    def _add_row(self, data: dict, bg: QColor, fg: QColor):
        r = self._table.rowCount()
        self._table.insertRow(r)
        for c, col in enumerate(self._columns):
            item = QTableWidgetItem(self._fmt(col, data.get(col, "")))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(bg)
            item.setForeground(fg)
            self._table.setItem(r, c, item)
        btn = QPushButton("✕")
        btn.setObjectName("delete_row_btn")
        btn.clicked.connect(lambda _, row=r: self._delete_row(row))
        self._table.setCellWidget(r, len(self._columns), btn)

    def _set_row_color(self, row: int, bg: QColor, fg: QColor):
        for c in range(len(self._columns)):
            item = self._table.item(row, c)
            if item:
                item.setBackground(bg)
                item.setForeground(fg)

    def _delete_row(self, row: int):
        if row < self._n_committed:
            self._n_committed -= 1
        else:
            pending_idx = row - self._n_committed
            if 0 <= pending_idx < len(self._pending):
                self._pending.pop(pending_idx)
            if not self._pending:
                self._btn_append.setEnabled(False)
        self._table.removeRow(row)
        for r in range(row, self._table.rowCount()):
            btn = QPushButton("✕")
            btn.setObjectName("delete_row_btn")
            btn.clicked.connect(lambda _, rr=r: self._delete_row(rr))
            self._table.setCellWidget(r, len(self._columns), btn)

    def _collect_committed_rows(self) -> list[dict]:
        rows = []
        for r in range(self._n_committed):
            row = {}
            for c, col in enumerate(self._columns):
                item = self._table.item(r, c)
                row[col] = item.text() if item else ""
            rows.append(row)
        return rows

    def _set_status(self, text: str, ok):
        color = {True: "#3cb371", False: "#e84d4d", None: "#e8a020"}[ok]
        self._status_lbl.setStyleSheet(f"color:{color}; font-size:9pt;")
        self._status_lbl.setText(text)
