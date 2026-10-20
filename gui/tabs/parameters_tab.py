"""
Parameters tab — read-only overview of all preference variables.

Lists every field of ProjectPrefs per section with three values:
  Current     – what the tabs hold right now (collected live via collect_prefs)
  Loaded      – what the last loaded/saved preferences file contained
  Default     – the dataclass default
A row whose current value differs from the loaded one is highlighted, which
usually means the value was changed by hand after loading — or that the tab
does not restore it correctly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QHeaderView,
)
from PyQt6.QtGui import QColor, QBrush, QFont

from gui.project_prefs import ProjectPrefs


# Section key in ProjectPrefs → heading, in program-tab order.
# ProjectMetadata is not collected by any tab yet and is therefore left out.
_SECTIONS = [
    ("extinction_coeff",     "Ext. Coeff."),
    ("spectra",              "Spectra"),
    ("half_life_kinetics",   "Half-Life — Kinetics"),
    ("half_life_scanning",   "Half-Life — Scanning Kinetics"),
    ("thermal",              "Thermal"),
    ("actinometer",          "Actinometer — Chemical"),
    ("led_characterisation", "Actinometer — LED"),
    ("quantum_yield",        "Quantum Yield"),
]

# Trailing name parts that are units → shown as "(unit)"
_UNITS = {
    ("mol", "s"): "mol/s", ("M", "cm"): "M⁻¹ cm⁻¹",
    ("nm",): "nm", ("s",): "s", ("mL",): "mL", ("cm",): "cm",
    ("C",): "°C", ("c",): "°C",
}

_ACRONYMS = {"pss": "PSS", "sb": "S_B", "fb": "f_B", "iqr": "IQR", "qy": "QY",
             "led": "LED", "th": "th"}

_MISMATCH = QColor("#f7768e")
_CHANGED  = QColor("#e08830")   # same orange as the preference labels in the tabs


def _label(name: str) -> str:
    """'irradiation_wavelength_nm' → 'Irradiation wavelength (nm)'."""
    parts = name.split("_")
    unit = ""
    for suffix, u in _UNITS.items():
        if len(parts) > len(suffix) and tuple(parts[-len(suffix):]) == suffix:
            parts, unit = parts[:-len(suffix)], u
            break
    words = [_ACRONYMS.get(p.lower(), p) for p in parts]
    text = " ".join(words)
    text = text[0].upper() + text[1:]
    return f"{text} ({unit})" if unit else text


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value) if value else "—"
    if value == "":
        return "(empty)"
    return str(value)


class ParametersTab(QWidget):
    """
    Parameters
    ----------
    collect_fn : callable returning a ProjectPrefs filled from all tabs.
    """

    def __init__(self, collect_fn: Callable[[], ProjectPrefs], parent=None):
        super().__init__(parent)
        self._collect_fn = collect_fn
        self._loaded: Optional[ProjectPrefs] = None
        self._loaded_from: Optional[Path] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        top = QHBoxLayout()
        self._source_lbl = QLabel()
        self._source_lbl.setObjectName("info_label")
        top.addWidget(self._source_lbl)
        top.addStretch()
        btn = QPushButton("Refresh")
        btn.clicked.connect(self.refresh)
        top.addWidget(btn)
        layout.addLayout(top)

        legend = QLabel(
            "<span style='color:#e08830'>■</span> differs from default &nbsp;&nbsp; "
            "<span style='color:#f7768e'>■</span> current value differs from the "
            "loaded file (changed by hand — or not restored correctly)")
        legend.setObjectName("info_label")
        layout.addWidget(legend)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Parameter", "Current", "Loaded file", "Default"])
        self._tree.setUniformRowHeights(True)
        hdr = self._tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in (1, 2, 3):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._tree)

    # ── Public API (called by MainWindow) ───────────────────────────────

    def set_loaded(self, prefs: ProjectPrefs, path: Path):
        """Remember the prefs of the file that was just loaded or saved."""
        self._loaded, self._loaded_from = prefs, path
        if self.isVisible():
            self.refresh()

    def refresh(self):
        current = asdict(self._collect_fn())
        default = asdict(ProjectPrefs())
        loaded = asdict(self._loaded) if self._loaded is not None else None

        self._source_lbl.setText(
            f"Preferences file: {self._loaded_from}" if self._loaded_from
            else "No preferences file loaded or saved in this session.")

        bold = QFont()
        bold.setBold(True)
        collapsed = {self._tree.topLevelItem(i).text(0)
                    for i in range(self._tree.topLevelItemCount())
                    if not self._tree.topLevelItem(i).isExpanded()}
        self._tree.clear()

        for key, heading in _SECTIONS:
            grp = QTreeWidgetItem(self._tree, [heading])
            grp.setFont(0, bold)
            for name, cur in current[key].items():
                dflt = default[key][name]
                ld = loaded[key].get(name) if loaded is not None else None
                row = QTreeWidgetItem(grp, [
                    _label(name), _fmt(cur),
                    _fmt(ld) if loaded is not None else "",
                    _fmt(dflt)])
                row.setToolTip(0, f"{key}.{name}")
                if loaded is not None and cur != ld:
                    color = _MISMATCH
                elif cur != dflt:
                    color = _CHANGED
                else:
                    continue
                for col in range(4):
                    row.setForeground(col, QBrush(color))
            grp.setExpanded(heading not in collapsed)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
