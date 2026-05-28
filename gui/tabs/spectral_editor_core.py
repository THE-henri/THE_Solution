"""
Core functions for the Spectral Editor tab.
Pure data manipulation — no GUI dependencies.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ── Type aliases ──────────────────────────────────────────────────────────────

KineticData = dict[str, tuple[np.ndarray, np.ndarray]]   # label → (time, abs)
ScanData    = list[tuple[np.ndarray, np.ndarray]]         # [(wl, ab), ...]


# ── File state container ──────────────────────────────────────────────────────

@dataclass
class EditorFile:
    """Tracks original, history, and current working state for one loaded file."""
    path:      Path
    data_type: str                                        # "kinetic" | "scanning"
    original:  Union[KineticData, ScanData]
    history:   list = field(default_factory=list)
    current:   Union[KineticData, ScanData, None] = field(default=None)

    def __post_init__(self):
        if self.current is None:
            self.current = copy.deepcopy(self.original)

    def push_and_apply(self, new_data: Union[KineticData, ScanData]) -> None:
        self.history.append(copy.deepcopy(self.current))
        self.current = new_data

    def undo(self) -> bool:
        if self.history:
            self.current = self.history.pop()
            return True
        return False

    def reset(self) -> None:
        self.history.clear()
        self.current = copy.deepcopy(self.original)

    @property
    def can_undo(self) -> bool:
        return bool(self.history)

    def summary(self) -> str:
        if self.data_type == "kinetic":
            n = len(self.current)
            return f"{n} channel{'s' if n != 1 else ''}"
        n = len(self.current)
        return f"{n} scan{'s' if n != 1 else ''}"

    def suggested_output_name(self) -> str:
        today = date.today().strftime("%Y-%m-%d")
        return f"{self.path.stem}_edited_{today}.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_to_wl(label: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", label, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)(?:\D*)$", label)
    return float(m.group(1)) if m else None


def _interp_abs(wl_source: np.ndarray, ab_source: np.ndarray,
                wl_target: np.ndarray) -> np.ndarray:
    order = np.argsort(wl_source)
    return np.interp(wl_target, wl_source[order], ab_source[order])


def _ref_value_kinetic(t: np.ndarray, a: np.ndarray,
                        ref: str, plateau_range: Optional[tuple]) -> float:
    if ref == "plateau" and plateau_range is not None:
        t0, t1 = plateau_range
        mask = (t >= t0) & (t <= t1)
        if not mask.any():
            raise ValueError(f"Plateau [{t0:.2f}, {t1:.2f}] s has no data points.")
        return float(a[mask].mean())
    return float(a[0])


def parse_index_range(text: str, max_idx: int) -> list[int]:
    """
    Parse a string like "0, 1, 5-10, 20" into a list of integer indices.
    Indices out of [0, max_idx] are silently ignored.
    """
    indices = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if m:
            for i in range(int(m.group(1)), int(m.group(2)) + 1):
                if 0 <= i <= max_idx:
                    indices.add(i)
        elif re.match(r"^\d+$", part):
            i = int(part)
            if 0 <= i <= max_idx:
                indices.add(i)
    return sorted(indices)


def load_single_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one spectrum from a file.
    Tries Cary 60 multi-scan format first (takes first scan),
    then falls back to two-column (wavelength, absorbance) CSV.
    """
    from gui.tabs.qy_core import load_spectra_csv
    try:
        scans = load_spectra_csv(path)
        if scans:
            return scans[0]
    except Exception:
        pass
    df = pd.read_csv(path, header=None, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] >= 2:
        wl = df.iloc[:, 0].values
        ab = df.iloc[:, 1].values
        order = np.argsort(wl)
        return wl[order], ab[order]
    raise ValueError(f"Cannot parse spectrum from {path.name}.")


# ── Kinetic operations ────────────────────────────────────────────────────────

def delete_channels(data: KineticData, labels: list[str]) -> KineticData:
    drop = set(labels)
    return {k: (t.copy(), a.copy()) for k, (t, a) in data.items() if k not in drop}


def add_offset_manual_kinetic(
    data:          KineticData,
    offset:        float,
    ref:           str            = "t0",
    plateau_range: Optional[tuple] = None,
) -> KineticData:
    """Shift each channel so its reference point becomes `offset`."""
    result: KineticData = {}
    for label, (t, a) in data.items():
        ref_val = _ref_value_kinetic(t, a, ref, plateau_range)
        result[label] = (t.copy(), a - ref_val + offset)
    return result


def add_offset_from_spectrum_kinetic(
    data:          KineticData,
    spec_wl:       np.ndarray,
    spec_ab:       np.ndarray,
    ref:           str            = "t0",
    plateau_range: Optional[tuple] = None,
    tol_nm:        float           = 2.0,
) -> KineticData:
    """Per-channel: shift trace so reference point matches the spectrum value at that wavelength."""
    result: KineticData = {}
    for label, (t, a) in data.items():
        wl = _label_to_wl(label)
        if wl is None:
            result[label] = (t.copy(), a.copy())
            continue
        mask = np.abs(spec_wl - wl) <= tol_nm
        if not mask.any():
            result[label] = (t.copy(), a.copy())
            continue
        target  = float(spec_ab[mask].mean())
        ref_val = _ref_value_kinetic(t, a, ref, plateau_range)
        result[label] = (t.copy(), a - ref_val + target)
    return result


def shift_time(data: KineticData, shift_s: float) -> KineticData:
    return {k: (t + shift_s, a.copy()) for k, (t, a) in data.items()}


def rescale_time(data: KineticData, factor: float) -> KineticData:
    if factor == 0.0:
        raise ValueError("Time rescale factor must not be zero.")
    return {k: (t * factor, a.copy()) for k, (t, a) in data.items()}


def combine_kinetic_side_by_side(datasets: list[KineticData]) -> KineticData:
    """Merge channels from multiple datasets; disambiguate duplicate labels."""
    result: KineticData = {}
    for data in datasets:
        for label, (t, a) in data.items():
            key, n = label, 1
            while key in result:
                key = f"{label}_{n}"
                n += 1
            result[key] = (t.copy(), a.copy())
    return result


def combine_kinetic_concatenate(
    datasets:  list[KineticData],
    join_mode: str            = "auto",
    join_time: Optional[float] = None,
) -> KineticData:
    """
    Concatenate time series from multiple datasets.
    join_mode='auto'  — second file starts immediately after first ends.
    join_mode='manual' — second file starts at join_time (s).
    Only channels present in the first dataset are kept.
    """
    if not datasets:
        return {}
    all_labels = list(datasets[0].keys())
    result: KineticData = {}
    for label in all_labels:
        parts_t: list[np.ndarray] = []
        parts_a: list[np.ndarray] = []
        t_end = 0.0
        for i, data in enumerate(datasets):
            if label not in data:
                continue
            t, a = data[label]
            if not parts_t:
                parts_t.append(t.copy())
                parts_a.append(a.copy())
                t_end = (float(t[-1]) if join_mode == "auto"
                         else (join_time if join_time is not None else float(t[-1])))
            else:
                t_new = t - t[0] + t_end
                parts_t.append(t_new)
                parts_a.append(a.copy())
                t_end = float(t_new[-1])
        if parts_t:
            result[label] = (np.concatenate(parts_t), np.concatenate(parts_a))
    return result


def save_kinetic_csv(data: KineticData, path: Path) -> None:
    """Write multi-channel kinetic CSV re-loadable by load_kinetic_csv."""
    labels = list(data.keys())
    arrays = [data[l] for l in labels]
    max_len = max((len(t) for t, _ in arrays), default=0)

    header_labels: list = []
    header_units:  list = []
    for l in labels:
        header_labels += [l, ""]
        header_units  += ["Time (sec)", "Abs"]

    rows = [header_labels, header_units]
    for i in range(max_len):
        row: list = []
        for t, a in arrays:
            if i < len(t):
                row += [f"{t[i]:.6g}", f"{a[i]:.8g}"]
            else:
                row += ["", ""]
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, header=False)


# ── Scanning operations ───────────────────────────────────────────────────────

def delete_scans(scans: ScanData, indices: list[int]) -> ScanData:
    drop = set(indices)
    return [(wl.copy(), ab.copy()) for i, (wl, ab) in enumerate(scans)
            if i not in drop]


def add_offset_scanning(scans: ScanData, offset: float) -> ScanData:
    return [(wl.copy(), ab + offset) for wl, ab in scans]


def add_offset_from_spectrum_scanning(
    scans:  ScanData,
    ref_wl: np.ndarray,
    ref_ab: np.ndarray,
) -> ScanData:
    """
    Align the first scan to the reference spectrum; apply the same
    per-wavelength delta to all subsequent scans.
    """
    if not scans:
        return []
    wl0, ab0 = scans[0]
    ref_on_grid = _interp_abs(ref_wl, ref_ab, wl0)
    delta = ref_on_grid - ab0
    result: ScanData = []
    for wl, ab in scans:
        if len(wl) == len(wl0) and np.allclose(wl, wl0, atol=0.01):
            result.append((wl.copy(), ab + delta))
        else:
            delta_i = np.interp(wl, wl0, delta)
            result.append((wl.copy(), ab + delta_i))
    return result


def align_at_wavelength(
    scans:  ScanData,
    wl_ref: float,
    target: str   = "first",   # "first" | "zero"
    tol_nm: float = 2.0,
) -> ScanData:
    """
    Scalar-shift each scan so all have the same absorbance at wl_ref.
    target='first' → match the first scan's value at wl_ref.
    target='zero'  → set the value at wl_ref to 0 for every scan.
    """
    if not scans:
        return []

    def _val(wl: np.ndarray, ab: np.ndarray) -> Optional[float]:
        mask = np.abs(wl - wl_ref) <= tol_nm
        return float(ab[mask].mean()) if mask.any() else None

    ref_val = 0.0 if target == "zero" else _val(*scans[0])
    if ref_val is None:
        raise ValueError(
            f"No data within {tol_nm:.1f} nm of {wl_ref:.1f} nm in the first scan.")

    result: ScanData = []
    for wl, ab in scans:
        v = _val(wl, ab)
        delta = (ref_val - v) if v is not None else 0.0
        result.append((wl.copy(), ab + delta))
    return result


def normalize_at_wavelength(
    scans:   ScanData,
    wl_norm: float,
    indices: Optional[list[int]] = None,
    tol_nm:  float = 2.0,
) -> ScanData:
    """Divide each selected scan by its absorbance at wl_norm."""
    apply_to = set(indices) if indices is not None else set(range(len(scans)))
    result: ScanData = []
    for i, (wl, ab) in enumerate(scans):
        if i not in apply_to:
            result.append((wl.copy(), ab.copy()))
            continue
        mask = np.abs(wl - wl_norm) <= tol_nm
        if not mask.any():
            result.append((wl.copy(), ab.copy()))
            continue
        norm_val = float(ab[mask].mean())
        if abs(norm_val) < 1e-10:
            result.append((wl.copy(), ab.copy()))
            continue
        result.append((wl.copy(), ab / norm_val))
    return result


def combine_scanning(scan_lists: list[ScanData]) -> ScanData:
    result: ScanData = []
    for scans in scan_lists:
        result.extend((wl.copy(), ab.copy()) for wl, ab in scans)
    return result


def save_scanning_csv(scans: ScanData, path: Path) -> None:
    """Write Cary 60 multi-scan format re-loadable by load_spectra_csv."""
    if not scans:
        pd.DataFrame().to_csv(path, index=False, header=False)
        return

    max_len = max(len(wl) for wl, _ in scans)
    n_scans = len(scans)

    header0: list = []
    header1: list = []
    for i in range(n_scans):
        header0 += [f"Scan_{i:04d}", ""]
        header1 += ["Wavelength (nm)", "Abs"]

    rows = [header0, header1]
    for i in range(max_len):
        row: list = []
        for wl, ab in scans:
            if i < len(wl):
                row += [f"{wl[i]:.4f}", f"{ab[i]:.8g}"]
            else:
                row += ["", ""]
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False, header=False)
