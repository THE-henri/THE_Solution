"""
Thermal tab — two sub-panels sharing the same stage structure:

  A. Arrhenius  (ln k  vs 1/T)
  B. Eyring     (ln k/T vs 1/T)

Both read from half_life_master.csv, which can be auto-filled from the
output results folder or selected manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QFrame, QTabWidget,
)

from gui.tabs.thermal_core import (
    ArrheniusResult, EyringResult,
    run_arrhenius, plot_arrhenius,
    run_eyring, plot_eyring,
)
from gui.widgets.stage_card import StageCard, WAITING, READY, DONE, STALE, ERROR
from gui.widgets.plot_widget import PlotWidget
from gui.widgets.info_button import InfoButton
from gui.worker import Worker


# ══════════════════════════════════════════════════════════════════════════════
# Shared base panel — Arrhenius and Eyring have identical stage layout
# ══════════════════════════════════════════════════════════════════════════════

class _ThermalPanel(QWidget):
    """
    Base class for Arrhenius and Eyring panels.

    Subclasses must implement:
        _run_analysis(master_csv, compound, weighted) -> result
        _plot_result(result) -> plt.Figure
        _result_to_row(result) -> dict          (for the save CSV)
        _result_filename(result) -> str         (default save filename)
        _info_title / _info_text                (PlotWidget tooltip)
    """

    log_signal = pyqtSignal(str, str)

    _info_title: str = "Thermal Analysis"
    _info_text:  str = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path:  Optional[Path] = None
        self._master_csv:   Optional[Path] = None
        self._result        = None
        self._worker:       Optional[Worker] = None
        self._compound_name: str = "Unknown"
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

        # ── Stage 1 — Data source ──────────────────────────────────────────
        self._stage1 = StageCard("Stage 1 — Data Source")
        self._stage1.add_info_button(
            "Data Source",
            "Load half_life_master.csv produced by the Half-Life tab.\n\n"
            "Use 'Auto-fill from results folder' to find the file automatically "
            "in the current output folder, or browse to any location.\n\n"
            "The file must contain columns: Temperature_C and k (s⁻¹). "
            "Multiple k values at the same temperature are averaged."
        )

        hint = QLabel(
            "Load half_life_master.csv produced by the Half-Life tab.\n"
            "Use \"Load from results\" to auto-find the file in the current "
            "output folder, or browse to select it manually."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(hint)

        btn_row = QHBoxLayout()
        self._load_results_btn = QPushButton("Load from results")
        self._load_results_btn.setFixedWidth(140)
        self._load_results_btn.clicked.connect(self._load_from_results)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setFixedWidth(90)
        self._browse_btn.clicked.connect(self._browse_master_csv)
        btn_row.addWidget(self._load_results_btn)
        btn_row.addWidget(self._browse_btn)
        btn_row.addStretch()
        self._stage1.add_layout(btn_row)

        self._csv_path_lbl = QLabel("(no file loaded)")
        self._csv_path_lbl.setStyleSheet("color:#888; font-size:9pt;")
        self._stage1.add_widget(self._csv_path_lbl)

        self._data_summary_lbl = QLabel("")
        self._data_summary_lbl.setStyleSheet("color:#5b8dee; font-size:9pt;")
        self._stage1.add_widget(self._data_summary_lbl)

        self._stage1.set_status(WAITING)
        layout.addWidget(self._stage1)

        # ── Stage 2 — Parameters ───────────────────────────────────────────
        self._stage2 = StageCard("Stage 2 — Parameters")
        self._stage2.add_info_button(
            "Analysis Parameters",
            "Compound name is taken from the project header.\n\n"
            "Weighted fit: when multiple k values exist at the same temperature "
            "(replicates), use inverse-variance weighting in the linear regression. "
            "Recommended when n > 1 per temperature point."
        )

        cmp_row = QHBoxLayout()
        _cmp_lbl = QLabel("Compound:")
        _cmp_lbl.setObjectName("pref_label")
        cmp_row.addWidget(_cmp_lbl)
        self._compound_name_lbl = QLabel(self._compound_name)
        self._compound_name_lbl.setStyleSheet("color:#5b8dee; font-size:9pt;")
        cmp_row.addWidget(self._compound_name_lbl)
        cmp_row.addStretch()
        self._stage2.add_layout(cmp_row)

        _wt_row = QHBoxLayout()
        self._weighted_chk = QCheckBox("Weighted fit  (uses k SEM when n > 1)")
        self._weighted_chk.setObjectName("pref_cb")
        self._weighted_chk.setChecked(True)
        _wt_row.addWidget(self._weighted_chk)
        _wt_row.addWidget(InfoButton(
            "Weighted fit",
            "When enabled and multiple k values exist at a temperature, each\n"
            "data point is weighted by 1/SEM\u00b2 before fitting.\n\n"
            "Recommended when temperature replicates have different uncertainties.\n"
            "Turn off to treat all points equally (ordinary least squares).",
        ))
        _wt_row.addStretch()
        self._stage2.add_layout(_wt_row)

        self._weighted_chk.stateChanged.connect(self._mark_stale)

        self._stage2.set_status(READY)
        layout.addWidget(self._stage2)

        # ── Stage 3 — Run & Results ────────────────────────────────────────
        self._stage3 = StageCard("Stage 3 — Run & Results")
        self._stage3.add_info_button(self._info_title, self._info_text)

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
            info_title=self._info_title,
            info_text=self._info_text,
            min_height=360,
        )
        self._stage3.add_widget(self._plot)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save results CSV")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_csv)
        self._save_pub_btn = QPushButton("Save publication data")
        self._save_pub_btn.setEnabled(False)
        self._save_pub_btn.setToolTip(
            "Save plot_data.csv (linearised data + fit line) and\n"
            "k_table.csv (k ± std, t½ in s / min / h / days) for each temperature.")
        self._save_pub_btn.clicked.connect(self._save_pub_data)
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._save_pub_btn)
        save_row.addStretch()
        self._stage3.add_layout(save_row)

        self._stage3.set_status(WAITING)
        layout.addWidget(self._stage3)
        layout.addStretch()

    # ── Public slots ───────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._output_path = path
        self._plot.set_save_dir(self._output_subdir() / "plots")
        # Try auto-loading the master CSV immediately
        self._load_from_results(silent=True)

    # ── Stage 1 — file loading ─────────────────────────────────────────────

    def _load_from_results(self, silent: bool = False):
        """Look for half_life_master.csv in the output results folder."""
        if self._output_path is None:
            if not silent:
                self._csv_path_lbl.setText(
                    "No output folder set. Select one in the folder header.")
            return
        candidate = (self._output_path / "half_life" / "results"
                     / "half_life_master.csv")
        if candidate.exists():
            self._load_csv(candidate)
        elif not silent:
            self._csv_path_lbl.setText(
                f"Not found: {candidate}")
            self._stage1.set_status(ERROR)

    def _browse_master_csv(self):
        start = str(self._output_path or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select half_life_master.csv", start, "CSV files (*.csv)")
        if path:
            self._load_csv(Path(path))

    def _load_csv(self, path: Path):
        try:
            df = pd.read_csv(path)
            required = {"Temperature_C", "k"}
            if not required.issubset(df.columns):
                self._csv_path_lbl.setText(
                    f"Missing columns {required - set(df.columns)} in {path.name}")
                self._stage1.set_status(ERROR)
                return
            temps = sorted(df["Temperature_C"].dropna().unique())
            n_temps = len(temps)
            n_rows  = len(df)
            self._master_csv = path
            self._csv_path_lbl.setText(str(path))
            self._data_summary_lbl.setText(
                f"{n_rows} rows  ·  {n_temps} temperatures: "
                + ", ".join(f"{t:.1f} °C" for t in temps)
            )
            if n_temps >= 2:
                self._stage1.set_status(READY)
                self._run_btn.setEnabled(True)
            else:
                self._stage1.set_status(ERROR)
                self._data_summary_lbl.setText(
                    self._data_summary_lbl.text()
                    + "  ← need ≥ 2 temperatures")
                self._run_btn.setEnabled(False)
            self._mark_stale()
        except Exception as exc:
            self._csv_path_lbl.setText(f"Error reading {path.name}: {exc}")
            self._stage1.set_status(ERROR)

    def _mark_stale(self):
        if self._result is not None:
            self._stage3.set_status(STALE)

    # ── Run ────────────────────────────────────────────────────────────────

    def _run(self):
        if self._master_csv is None:
            return
        compound = self._compound_name or "Unknown"
        weighted = self._weighted_chk.isChecked()

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._save_pub_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._stage3.set_status(WAITING)

        master_csv = self._master_csv

        self._worker = Worker(
            self._run_analysis,
            master_csv=master_csv,
            compound_name=compound,
            weighted_fit=weighted,
        )
        self._worker.log_signal.connect(self.log_signal)
        self._worker.result_signal.connect(self._on_result)
        self._worker.error_signal.connect(self._on_error)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_result(self, result):
        self._result = result
        fig = self._plot_result(result)
        self._plot.set_default_filename(self._result_filename(result))
        self._plot.set_figure(fig)
        self._stage3.set_status(DONE)
        t25 = result.t_half_25C_s
        self._status_lbl.setText(
            f"Done — {result.n_temperatures} temperatures  "
            f"({'weighted' if result.weighted else 'unweighted'})  "
            f"R² = {result.r2:.4f}  |  "
            f"t½(25°C) = {t25:.1f} s / {t25/60:.2f} min / {t25/3600:.3f} h")
        self._save_btn.setEnabled(True)
        self._save_pub_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._stage3.set_status(ERROR)
        self._status_lbl.setText(f"Error: {msg}")

    def _on_finished(self):
        self._run_btn.setEnabled(True)

    # ── Save CSV ───────────────────────────────────────────────────────────

    def _save_csv(self):
        if self._result is None:
            return
        default_name = self._result_filename(self._result).replace(".png", ".csv")
        out_dir = self._output_subdir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results CSV",
            str(out_dir / default_name), "CSV (*.csv)")
        if path:
            pd.DataFrame([self._result_to_row(self._result)]).to_csv(
                path, index=False)
            print(f"[Thermal] CSV saved → {path}")
            self.log_signal.emit(f"[Thermal] CSV saved → {path}", "INFO")

    def _save_pub_data(self):
        """Save publication CSVs: plot_data.csv and k_table.csv."""
        import numpy as np
        if self._result is None:
            return
        out_dir = self._output_subdir() / "publication"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self._result
        compound = result.compound

        # ── 1. plot_data.csv — linearised data points + fit line ──────────
        plot_df = self._pub_plot_df(result)
        plot_path = out_dir / f"{compound}_{self._pub_type()}_plot_data.csv"
        plot_df.to_csv(plot_path, index=False)

        # ── 2. k_table.csv — k stats + half-lives per temperature ─────────
        ln2 = np.log(2)
        rows = []
        for T_C, k_mean, k_std, n in zip(
                result.T_C_list, result.k_mean_list,
                result.k_std_list, result.n_list):
            t_half_s   = ln2 / k_mean
            # Propagate std:  σ(t½) = (ln2 / k²) · σ_k = t½ · (σ_k / k)
            if k_std is not None and k_mean > 0:
                t_half_std_s = t_half_s * (k_std / k_mean)
            else:
                t_half_std_s = None
            rows.append({
                "Temperature_C":    T_C,
                "n_replicates":     n,
                "k_mean_s-1":       k_mean,
                "k_std_s-1":        k_std if k_std is not None else "",
                "t_half_s":         t_half_s,
                "t_half_std_s":     t_half_std_s if t_half_std_s is not None else "",
                "t_half_min":       t_half_s / 60.0,
                "t_half_std_min":   t_half_std_s / 60.0 if t_half_std_s is not None else "",
                "t_half_h":         t_half_s / 3600.0,
                "t_half_std_h":     t_half_std_s / 3600.0 if t_half_std_s is not None else "",
                "t_half_days":      t_half_s / 86400.0,
                "t_half_std_days":  t_half_std_s / 86400.0 if t_half_std_s is not None else "",
            })
        ktable_path = out_dir / f"{compound}_k_table.csv"
        pd.DataFrame(rows).to_csv(ktable_path, index=False)

        msg = f"Publication data saved → {out_dir}"
        print(f"[Thermal] {msg}")
        self.log_signal.emit(msg, "INFO")

    # ── Subclass hooks ─────────────────────────────────────────────────────
    # Override in ArrheniusPanel / EyringPanel

    def _run_analysis(self, master_csv, compound_name, weighted_fit):
        raise NotImplementedError

    def _plot_result(self, result):
        raise NotImplementedError

    def _result_to_row(self, result) -> dict:
        raise NotImplementedError

    def _result_filename(self, result) -> str:
        raise NotImplementedError

    def _output_subdir(self) -> Path:
        raise NotImplementedError

    def _pub_plot_df(self, result) -> "pd.DataFrame":
        raise NotImplementedError

    def _pub_type(self) -> str:
        raise NotImplementedError

    # ── Preferences ────────────────────────────────────────────────────────

    def set_compound_name(self, name: str):
        self._compound_name = name
        self._compound_name_lbl.setText(name)
        self._mark_stale()

    def _apply_thermal_prefs(self, weighted: bool):
        self._weighted_chk.setChecked(weighted)

    def _collect_thermal_prefs(self) -> bool:
        return self._weighted_chk.isChecked()


# ══════════════════════════════════════════════════════════════════════════════
# Arrhenius sub-panel
# ══════════════════════════════════════════════════════════════════════════════

class _ArrheniusPanel(_ThermalPanel):
    _info_title = "Arrhenius Plot"
    _info_text  = (
        "ln(k) vs 1/T linear fit.\n"
        "Slope = −Ea/R  →  Ea = activation energy\n"
        "Intercept = ln(A)  →  A = pre-exponential factor"
    )

    def _run_analysis(self, master_csv, compound_name, weighted_fit):
        return run_arrhenius(master_csv, compound_name, weighted_fit)

    def _plot_result(self, result):
        return plot_arrhenius(result)

    def _result_to_row(self, result: ArrheniusResult) -> dict:
        t25 = result.t_half_25C_s
        return {
            "Compound":           result.compound,
            "Ea_kJmol":           result.Ea_kJmol,
            "Ea_std_kJmol":       result.Ea_std_kJmol,
            "A_s":                result.A_s,
            "A_std_s":            result.A_std_s,
            "R2_Arrhenius":       result.r2,
            "n_temperatures":     result.n_temperatures,
            "weighted":           result.weighted,
            "k_25C_s-1":          result.k_25C,
            "t_half_25C_s":       t25,
            "t_half_25C_min":     t25 / 60.0,
            "t_half_25C_h":       t25 / 3600.0,
        }

    def _result_filename(self, result: ArrheniusResult) -> str:
        return f"{result.compound}_Arrhenius.png"

    def _output_subdir(self) -> Path:
        base = self._output_path or Path.home()
        return base / "arrhenius" / "results"

    def _pub_type(self) -> str:
        return "Arrhenius"

    def _pub_plot_df(self, result: ArrheniusResult) -> "pd.DataFrame":
        import numpy as np
        # Data points
        yerr = result.sigma_y.tolist() if result.sigma_y is not None \
               else [""] * len(result.x_data)
        data_rows = pd.DataFrame({
            "inv_T_1000K":  result.x_data,
            "ln_k":         result.ln_k,
            "ln_k_err":     yerr,
        })
        # Fit line (200 points, NaN-pad shorter columns)
        fit_rows = pd.DataFrame({
            "fit_inv_T_1000K": result.x_fit,
            "fit_ln_k":        result.y_fit,
        })
        return pd.concat([data_rows, fit_rows], axis=1)

    def apply_prefs(self, prefs):
        self._apply_thermal_prefs(prefs.thermal.weighted_fit)

    def collect_prefs(self, prefs):
        prefs.thermal.weighted_fit = self._collect_thermal_prefs()


# ══════════════════════════════════════════════════════════════════════════════
# Eyring sub-panel
# ══════════════════════════════════════════════════════════════════════════════

class _EyringPanel(_ThermalPanel):
    _info_title = "Eyring Plot"
    _info_text  = (
        "ln(k/T) vs 1/T linear fit (transition-state theory).\n"
        "Slope = −ΔH‡/R  →  ΔH‡ = activation enthalpy\n"
        "Intercept = ΔS‡/R + ln(kB/h)  →  ΔS‡ = activation entropy"
    )

    def _run_analysis(self, master_csv, compound_name, weighted_fit):
        return run_eyring(master_csv, compound_name, weighted_fit)

    def _plot_result(self, result):
        return plot_eyring(result)

    def _result_to_row(self, result: EyringResult) -> dict:
        t25 = result.t_half_25C_s
        return {
            "Compound":           result.compound,
            "dH_kJmol":           result.dH_kJmol,
            "dH_std_kJmol":       result.dH_std_kJmol,
            "dS_JmolK":           result.dS_JmolK,
            "dS_std_JmolK":       result.dS_std_JmolK,
            "R2_Eyring":          result.r2,
            "n_temperatures":     result.n_temperatures,
            "weighted":           result.weighted,
            "k_25C_s-1":          result.k_25C,
            "t_half_25C_s":       t25,
            "t_half_25C_min":     t25 / 60.0,
            "t_half_25C_h":       t25 / 3600.0,
        }

    def _result_filename(self, result: EyringResult) -> str:
        return f"{result.compound}_Eyring.png"

    def _output_subdir(self) -> Path:
        base = self._output_path or Path.home()
        return base / "eyring" / "results"

    def _pub_type(self) -> str:
        return "Eyring"

    def _pub_plot_df(self, result: EyringResult) -> "pd.DataFrame":
        import numpy as np
        yerr = result.sigma_y.tolist() if result.sigma_y is not None \
               else [""] * len(result.x_data)
        data_rows = pd.DataFrame({
            "inv_T_1000K":  result.x_data,
            "ln_kT":        result.ln_kT,
            "ln_kT_err":    yerr,
        })
        fit_rows = pd.DataFrame({
            "fit_inv_T_1000K": result.x_fit,
            "fit_ln_kT":       result.y_fit,
        })
        return pd.concat([data_rows, fit_rows], axis=1)

    def apply_prefs(self, prefs):
        self._apply_thermal_prefs(prefs.thermal.weighted_fit)

    def collect_prefs(self, prefs):
        prefs.thermal.weighted_fit = self._collect_thermal_prefs()


# ══════════════════════════════════════════════════════════════════════════════
# Top-level ThermalTab
# ══════════════════════════════════════════════════════════════════════════════

class ThermalTab(QWidget):
    """Thermal tab: Arrhenius + Eyring analyses from half_life_master.csv."""

    log_signal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        inner = QTabWidget()
        inner.setObjectName("inner_tabs")

        self._arrhenius_panel = _ArrheniusPanel()
        self._eyring_panel    = _EyringPanel()
        inner.addTab(self._arrhenius_panel, "Arrhenius")
        inner.addTab(self._eyring_panel,    "Eyring")
        layout.addWidget(inner)

        self._arrhenius_panel.log_signal.connect(self.log_signal)
        self._eyring_panel.log_signal.connect(self.log_signal)

    # ── Public slots ───────────────────────────────────────────────────────

    def set_output_path(self, path: Path):
        self._arrhenius_panel.set_output_path(path)
        self._eyring_panel.set_output_path(path)

    def set_compound_name(self, name: str):
        self._arrhenius_panel.set_compound_name(name)
        self._eyring_panel.set_compound_name(name)

    def apply_prefs(self, prefs):
        self._arrhenius_panel.apply_prefs(prefs)
        self._eyring_panel.apply_prefs(prefs)

    def collect_prefs(self, prefs):
        self._arrhenius_panel.collect_prefs(prefs)
        # Eyring collect_prefs writes the same fields — calling both is safe
        self._eyring_panel.collect_prefs(prefs)
