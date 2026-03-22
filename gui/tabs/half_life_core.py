"""
Pure data-loading and fitting functions for the Half-Life tab.

No top-level side-effects — safe to import at any time.
All functions return plain Python / numpy / pandas objects so they
can be called from both the GUI and the existing CLI scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from core.kinetics import fit_half_life


# ── Time-unit detection ───────────────────────────────────────────────────

def detect_time_unit(filepath: str | Path) -> str:
    """
    Inspect row 1 of a Cary-style CSV for 'Time (min)' or 'Time (sec)'.

    Returns 'min' or 'sec' (default 'sec' if header is unrecognised).
    """
    try:
        raw = pd.read_csv(filepath, header=None, nrows=2)
        header_val = str(raw.iloc[1, 0]).strip().lower()
        if "min" in header_val:
            return "min"
    except Exception:
        pass
    return "sec"


# ── Kinetics CSV loader ────────────────────────────────────────────────────

def load_kinetics_csv(filepath: str | Path, convert_to_seconds: bool = True
                      ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load a multi-wavelength kinetics CSV.

    Format
    ------
    Row 0 : channel labels  (e.g. '45C_672') in every even column
    Row 1 : 'Time (sec)'/'Time (min)', 'Abs' repeated per channel
    Row 2+: time / absorbance pairs

    Returns
    -------
    dict {label: (time_s, absorbance)}
    """
    MIN_VALID = 10
    filepath = Path(filepath)
    raw  = pd.read_csv(filepath, header=None)
    unit = detect_time_unit(filepath)

    label_row = raw.iloc[0]
    # Drop metadata rows: keep only rows where col 0 (first channel's time) is numeric.
    # Cary 60 inserts periodic metadata blocks (e.g. "Wavelengths (nm)  409.0") whose
    # values in cols 2-3 look numeric and would otherwise create spurious data points.
    _all_data  = raw.iloc[2:].reset_index(drop=True)
    _col0_num  = pd.to_numeric(_all_data.iloc[:, 0], errors="coerce")
    data       = _all_data[_col0_num.notna()].reset_index(drop=True)
    n_cols     = label_row.shape[0]

    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i in range(0, n_cols - 1, 2):
        label = str(label_row.iloc[i]).strip()
        if not label or label.lower() == "nan":
            continue

        t_col  = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        ab_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid  = t_col.notna() & ab_col.notna()

        if valid.sum() < MIN_VALID:
            print(f"  Skipping channel '{label}': only {valid.sum()} valid points.")
            continue

        t  = t_col[valid].values.astype(float)
        ab = ab_col[valid].values.astype(float)

        if convert_to_seconds and unit == "min":
            t = t * 60.0

        channels[label] = (t, ab)

    return channels


# ── Scanning kinetics CSV loader ──────────────────────────────────────────

def load_scanning_kinetics_csv(filepath: str | Path
                               ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load a scanning-kinetics CSV (one full spectrum per scan).

    Format
    ------
    Row 0 : scan labels in every even column
    Row 1 : 'Wavelength (nm)', 'Abs' repeated per scan
    Row 2+: wavelength / absorbance pairs

    Returns
    -------
    list of (wavelength_array, absorbance_array), one per scan in order.
    """
    MIN_VALID = 5
    raw    = pd.read_csv(filepath, header=None)
    data   = raw.iloc[2:].reset_index(drop=True)
    n_cols = data.shape[1]
    scans  = []

    for i in range(0, n_cols - 1, 2):
        wl_col  = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        ab_col  = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid   = wl_col.notna() & ab_col.notna()
        if valid.sum() < MIN_VALID:
            continue
        scans.append((wl_col[valid].values.astype(float),
                      ab_col[valid].values.astype(float)))

    return scans


def extract_trace_at_wavelength(scans: list, target_nm: float,
                                tolerance_nm: float) -> np.ndarray:
    """Mean absorbance near target_nm for each scan (NaN if no match)."""
    trace = []
    for wl, ab in scans:
        mask = (wl >= target_nm - tolerance_nm) & (wl <= target_nm + tolerance_nm)
        trace.append(float(ab[mask].mean()) if mask.any() else np.nan)
    return np.array(trace)


def load_reference_spectrum(filepath: str | Path
                            ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load a reference spectrum file (same format as scanning kinetics)."""
    return load_scanning_kinetics_csv(filepath)


def extract_a_inf_from_reference(scans: list, target_nm: float,
                                 tolerance_nm: float) -> float:
    """Mean absorbance at target_nm across all reference scans."""
    vals   = extract_trace_at_wavelength(scans, target_nm, tolerance_nm)
    valid  = vals[~np.isnan(vals)]
    if len(valid) == 0:
        raise ValueError(f"No data near {target_nm} nm in reference spectrum.")
    return float(np.mean(valid))


# ── Time-window auto-detection ───────────────────────────────────────────

def detect_time_window(
    time:       np.ndarray,
    absorbance: np.ndarray,
    switch:     str   = "negative",
    n_plateau:  int   = 10,
    threshold:  float = 5.0,
    min_consec: int   = 3,
) -> tuple[float | None, float]:
    """
    Detect where a kinetic trace starts deviating from an initial plateau
    in the direction expected for the given switch type.

    switch='negative' → negative photochromic (build-up): looks for an increase.
    switch='positive' → positive photochromic (decay): looks for a decrease.

    Returns (t_start, t_end); t_start is None if no deviation was found.
    """
    n = len(time)
    n_plat = min(n_plateau, n // 3)
    if n_plat < 2:
        return None, float(time[-1])

    plat      = absorbance[:n_plat]
    plat_mean = float(plat.mean())
    plat_std  = float(plat.std(ddof=1))
    min_std   = max(abs(plat_mean) * 0.0005, 1e-5)
    plat_std  = max(plat_std, min_std)
    thresh    = threshold * plat_std

    # For build-up (negative photochromic, switch="negative") we look for
    # points ABOVE the plateau; for decay (positive, switch="positive") BELOW.
    def _exceeds(val: float) -> bool:
        dev = val - plat_mean
        if abs(dev) <= thresh:
            return False
        if switch == "negative":
            return dev > 0   # building above plateau (negative photochromic)
        else:
            return dev < 0   # decaying below plateau (positive photochromic)

    consec    = 0
    idx_onset = None
    for i in range(n_plat, n):
        if _exceeds(absorbance[i]):
            consec += 1
            if consec >= min_consec and idx_onset is None:
                idx_onset = i - min_consec + 1
        else:
            consec = 0

    if idx_onset is None:
        return None, float(time[-1])
    return float(time[idx_onset]), float(time[-1])


def find_thermal_segments(
    time:                np.ndarray,
    absorbance:          np.ndarray,
    switch:              str   = "negative",
    smooth_window:       int   = 7,
    min_prominence_frac: float = 0.10,
    stability_n:         int   = 8,
    stability_frac:      float = 0.02,
    min_seg_points:      int   = 8,
) -> list[tuple[float, float]]:
    """
    Find all thermal-recovery segments in a kinetic trace.

    Strategy
    --------
    An irradiation event produces a sharp change in the *wrong* direction
    (rise for decay / fall for build-up).  Those events leave a local peak
    (decay) or valley (build-up) in the smoothed signal.  Each such
    extremum marks the START of a new recovery segment.

    If no extrema are found the whole trace is returned as one segment.
    Segment ends when the signal stabilises (point-to-point changes are
    small for `stability_n` consecutive steps).

    Returns
    -------
    list of (t_start, t_end) — at least one entry.
    """
    from scipy.signal import find_peaks

    n = len(time)
    if n < min_seg_points * 2:
        return [(float(time[0]), float(time[-1]))]

    # ── Smooth ─────────────────────────────────────────────────────────
    w      = max(1, min(smooth_window, n // 10))
    kernel = np.ones(w) / w
    pad    = w // 2
    padded = np.pad(absorbance.astype(float), pad, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")[:n]

    total_range = float(np.ptp(smooth))
    if total_range < 1e-8:
        return [(float(time[0]), float(time[-1]))]

    min_prom         = min_prominence_frac * total_range
    stability_thresh = stability_frac * total_range

    # ── Find irradiation-event extrema ─────────────────────────────────
    # Convention (matches core/kinetics.py fit_half_life):
    #   switch="negative" → NEGATIVE photochromic (build-up):
    #       irradiation pushed signal DOWN → extremum = valley → find_peaks(-smooth)
    #   switch="positive" → POSITIVE photochromic (decay):
    #       irradiation pushed signal UP   → extremum = peak  → find_peaks(smooth)
    if switch == "negative":
        peak_idxs, _ = find_peaks(-smooth, prominence=min_prom)   # find valleys
    else:
        peak_idxs, _ = find_peaks(smooth,  prominence=min_prom)   # find peaks

    # ── Filter: only keep extrema where recovery actually follows ───────
    # Use a generous look-ahead (n//5) and a low threshold (3 % of range).
    # negative (build-up): post-valley mean must be clearly ABOVE the valley.
    # positive (decay):    post-peak  mean must be clearly BELOW the peak.
    look_fwd = max(min_seg_points * 3, n // 5)
    valid_peaks: list[int] = []
    for idx in peak_idxs:
        post_end = min(idx + look_fwd, n)
        if post_end - idx < 4:
            continue
        post_mean = float(np.mean(smooth[idx + 2: post_end]))
        if switch == "negative":   # build-up: signal should rise after valley
            if post_mean - smooth[idx] > 0.03 * total_range:
                valid_peaks.append(idx)
        else:                      # decay: signal should fall after peak
            if smooth[idx] - post_mean > 0.03 * total_range:
                valid_peaks.append(idx)

    starts: list[int] = valid_peaks if valid_peaks else [0]

    # ── Plateau-exit helper ─────────────────────────────────────────────
    # Compute a plateau reference from the very first few points of the
    # trace.  Using mean ± threshold instead of per-step diffs makes the
    # exit detection immune to isolated noise spikes inside the plateau.
    init_n    = min(stability_n * 3, n // 4)
    init_mean = float(np.mean(smooth[:init_n]))
    init_std  = max(float(np.std(smooth[:init_n])), 1e-9)
    # Require 5 % of total range OR 4 σ of plateau noise to declare "left plateau".
    plateau_exit_thresh = max(0.05 * total_range, 4.0 * init_std)

    def _find_plateau_exit(from_idx: int, to_idx: int) -> int:
        """First index whose smooth value has clearly left the initial plateau."""
        for s in range(from_idx, to_idx):
            if abs(smooth[s] - init_mean) > plateau_exit_thresh:
                return s
        return to_idx

    # ── Build segments ──────────────────────────────────────────────────
    segments: list[tuple[float, float]] = []

    for seg_i, start_idx in enumerate(starts):
        hard_end = (max(start_idx + 1, starts[seg_i + 1] - 1)
                    if seg_i + 1 < len(starts) else n - 1)

        actual_start = start_idx

        # Check local flatness of the first window at start_idx.
        win = smooth[start_idx: min(start_idx + stability_n, hard_end)]
        is_locally_flat = (len(win) >= 2 and
                           np.all(np.abs(np.diff(win)) < stability_thresh))

        # Is start_idx close to the initial (pre-irradiation) plateau level?
        near_init = abs(smooth[start_idx] - init_mean) < plateau_exit_thresh

        # Is start_idx already at the recovery-starting extremum?
        # negative (build-up): extremum = valley (global minimum)
        # positive (decay):    extremum = peak  (global maximum)
        if switch == "negative":
            at_extreme = (smooth[start_idx] - float(np.min(smooth))
                          < 0.05 * total_range)
        else:
            at_extreme = (float(np.max(smooth)) - smooth[start_idx]
                          < 0.05 * total_range)

        if is_locally_flat:
            if near_init and not at_extreme:
                # ── Case 1: flat pre-irradiation plateau ────────────────
                # Skip plateau, then find the irradiation-event extremum.
                transition = _find_plateau_exit(start_idx, hard_end)
                if transition < hard_end:
                    search_end = min(
                        transition + max(stability_n * 8, n // 4), hard_end)
                    if switch == "negative":   # build-up: extremum = valley
                        actual_start = transition + int(
                            np.argmin(smooth[transition:search_end]))
                    else:                      # decay: extremum = peak
                        actual_start = transition + int(
                            np.argmax(smooth[transition:search_end]))

            elif at_extreme:
                # ── Case 2: flat at the recovery extremum ───────────────
                # File starts already-irradiated (e.g. Z-form plateau at
                # near-zero abs for a negative photochromic compound).
                # Skip the flat extremum region and find where the thermal
                # recovery actually begins.
                extreme_val = float(smooth[start_idx])
                for r in range(start_idx + stability_n, hard_end):
                    if switch == "negative":   # build-up: wait for rise
                        deviation = smooth[r] - extreme_val
                    else:                      # decay: wait for fall
                        deviation = extreme_val - smooth[r]
                    if deviation > plateau_exit_thresh:
                        # Step back slightly so the fit includes the onset
                        actual_start = max(start_idx, r - stability_n // 2)
                        break

        # Walk forward from actual_start until signal stabilises.
        # Two conditions must both hold to avoid stopping on a slow trend:
        #   1. No individual step exceeds stability_thresh (no noise jumps).
        #   2. The net drift across the full window is also < stability_thresh
        #      (rules out slow monotonic recovery being mistaken for a plateau).
        end_idx = hard_end
        for j in range(actual_start + min_seg_points,
                       hard_end - stability_n + 2):
            chunk = np.diff(smooth[j: j + stability_n])
            if np.all(np.abs(chunk) < stability_thresh):
                window_drift = abs(float(smooth[j + stability_n - 1])
                                   - float(smooth[j]))
                if window_drift < stability_thresh:
                    end_idx = j + stability_n - 1
                    break

        if end_idx - actual_start >= min_seg_points:
            segments.append((float(time[actual_start]), float(time[end_idx])))

    return segments if segments else [(float(time[0]), float(time[-1]))]


# ── Outlier removal ───────────────────────────────────────────────────────

def remove_outliers(time: np.ndarray, absorbance: np.ndarray,
                    fitted_curve: np.ndarray,
                    iqr_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exclude points where |residual| > iqr_factor × IQR.

    Returns (time_clean, abs_clean, inlier_mask).
    """
    residuals   = absorbance - fitted_curve
    q1, q3      = np.percentile(residuals, [25, 75])
    iqr         = q3 - q1
    inlier_mask = np.abs(residuals) <= iqr_factor * iqr
    return time[inlier_mask], absorbance[inlier_mask], inlier_mask


# ── Fit result dataclass ──────────────────────────────────────────────────

@dataclass
class FitResult:
    label:           str
    switch:          str
    popt:            Optional[tuple]
    t_half:          Optional[float]
    fitted_curve:    Optional[np.ndarray]
    r2:              Optional[float]
    time_full:       np.ndarray
    abs_full:        np.ndarray
    time_sel:        np.ndarray
    abs_sel:         np.ndarray
    time_clean:      np.ndarray
    abs_clean:       np.ndarray
    time_outliers:   np.ndarray
    abs_outliers:    np.ndarray
    start_idx:       int
    end_idx:         int
    temperature_c:   float
    n_excluded:      int
    success:         bool
    error_msg:       str = ""


# ── Main fitting function ─────────────────────────────────────────────────

def run_half_life_fit(
    label:          str,
    time:           np.ndarray,
    absorbance:     np.ndarray,
    t_start_s:      float,
    t_end_s:        float,
    switch:         str,
    a_inf_mode:     str,          # "free" | "fixed" | "reference"
    a_inf_value:    float | None,
    iqr_factor:     float,
    temperature_c:  float,
) -> FitResult:
    """
    Full fit pipeline for one kinetic trace:
      1. Select time window by time values
      2. Initial fit
      3. Outlier removal
      4. Final fit on cleaned data

    Parameters
    ----------
    a_inf_mode : 'free'      – fit A∞ as free parameter
                 'fixed'     – fix A∞ to a_inf_value
                 'reference' – a_inf_value already extracted from reference
    """
    # ── Time window → indices ─────────────────────────────────────────
    start_idx = int(np.searchsorted(time, t_start_s))
    end_idx   = int(np.searchsorted(time, t_end_s, side="right")) - 1
    end_idx   = min(end_idx, len(time) - 1)
    start_idx = max(start_idx, 0)

    time_sel = time[start_idx:end_idx + 1]
    abs_sel  = absorbance[start_idx:end_idx + 1]

    def _blank(msg: str) -> FitResult:
        return FitResult(
            label=label, switch=switch, popt=None, t_half=None,
            fitted_curve=None, r2=None,
            time_full=time, abs_full=absorbance,
            time_sel=time_sel, abs_sel=abs_sel,
            time_clean=time_sel, abs_clean=abs_sel,
            time_outliers=np.array([]), abs_outliers=np.array([]),
            start_idx=start_idx, end_idx=end_idx,
            temperature_c=temperature_c, n_excluded=0,
            success=False, error_msg=msg,
        )

    if len(time_sel) < 3:
        return _blank("Too few points in selection window.")

    # ── Determine A∞ ─────────────────────────────────────────────────
    if a_inf_mode == "free":
        a_inf_manual = None
    else:
        a_inf_manual = a_inf_value  # fixed or pre-extracted from reference

    # ── Initial fit ───────────────────────────────────────────────────
    raw_result = fit_half_life(time_sel, abs_sel,
                               switch=switch, A_inf_manual=a_inf_manual)
    # Defensive: handle 3- or 4-value return
    if len(raw_result) == 3:
        raw_result = (*raw_result, None)
    popt_init, t_half_init, curve_init, _ = raw_result

    if popt_init is None or curve_init is None:
        return _blank("Initial fit failed.")

    # ── Outlier removal ───────────────────────────────────────────────
    t_clean, ab_clean, inlier_mask = remove_outliers(
        time_sel, abs_sel, curve_init, iqr_factor)
    t_out  = time_sel[~inlier_mask]
    ab_out = abs_sel[~inlier_mask]
    n_excl = int((~inlier_mask).sum())

    print(f"  {label}: outlier removal ({iqr_factor}×IQR) → "
          f"{n_excl}/{len(inlier_mask)} points excluded "
          f"({100*n_excl/len(inlier_mask):.1f}%)")

    if len(t_clean) < 3:
        return _blank("Too few points after outlier removal.")

    # ── Final fit ─────────────────────────────────────────────────────
    raw_result2 = fit_half_life(t_clean, ab_clean,
                                switch=switch, A_inf_manual=a_inf_manual)
    if len(raw_result2) == 3:
        raw_result2 = (*raw_result2, None)
    popt, t_half, fitted_curve, r2 = raw_result2

    if popt is None:
        return _blank("Final fit failed after outlier removal.")

    print(f"  {label}: t½ = {t_half:.2f} s   R² = {r2:.6f}")

    return FitResult(
        label=label, switch=switch, popt=popt, t_half=t_half,
        fitted_curve=fitted_curve, r2=r2,
        time_full=time, abs_full=absorbance,
        time_sel=time_sel, abs_sel=abs_sel,
        time_clean=t_clean, abs_clean=ab_clean,
        time_outliers=t_out, abs_outliers=ab_out,
        start_idx=start_idx, end_idx=end_idx,
        temperature_c=temperature_c, n_excluded=n_excl,
        success=True,
    )


# ── Scanning-kinetics batch fit ───────────────────────────────────────────

def run_scanning_fit(
    filepath:       str | Path,
    target_wavelengths: list[float],
    wavelength_tolerance: float,
    time_interval_s: float,
    scan_start:     int,
    scan_end:       int | None,
    switch:         str,
    a_inf_mode:     str,
    a_inf_value:    float | None,
    reference_scans: list | None,
    iqr_factor:     float,
    temperature_c:  float,
) -> list[FitResult]:
    """
    Fit all target wavelengths from one scanning-kinetics file.
    Returns a list of FitResult (one per wavelength).
    """
    scans = load_scanning_kinetics_csv(filepath)
    n_scans = len(scans)
    if n_scans == 0:
        raise ValueError("No valid scans found.")

    end_idx_scan = (n_scans - 1) if scan_end is None else min(scan_end, n_scans - 1)
    time_full = np.arange(n_scans) * time_interval_s

    results = []
    for target_nm in target_wavelengths:
        print(f"\n  Wavelength: {target_nm} nm")

        abs_full = extract_trace_at_wavelength(scans, target_nm, wavelength_tolerance)
        valid_mask = ~np.isnan(abs_full)
        if valid_mask.sum() < 3:
            print(f"  Too few valid scans at {target_nm} nm — skipping.")
            continue

        t_v  = time_full[valid_mask]
        ab_v = abs_full[valid_mask]

        # Slice to selected scan range (convert scan index → array index)
        sel_mask = (np.arange(len(t_v)) >= scan_start) & \
                   (np.arange(len(t_v)) <= end_idx_scan)
        t_sel  = t_v[sel_mask]
        ab_sel = ab_v[sel_mask]

        # Determine A∞
        if a_inf_mode == "reference" and reference_scans is not None:
            try:
                a_inf_val = extract_a_inf_from_reference(
                    reference_scans, target_nm, wavelength_tolerance)
                print(f"  A∞ from reference: {a_inf_val:.6f}")
            except ValueError as exc:
                print(f"  {exc} — skipping.")
                continue
            a_inf_eff = a_inf_val
        elif a_inf_mode == "fixed":
            a_inf_eff = a_inf_value
        else:
            a_inf_eff = None

        # Re-use the same fit pipeline
        result = run_half_life_fit(
            label=f"{target_nm} nm",
            time=t_v,
            absorbance=ab_v,
            t_start_s=t_sel[0] if len(t_sel) else 0.0,
            t_end_s=t_sel[-1] if len(t_sel) else time_full[-1],
            switch=switch,
            a_inf_mode="fixed" if a_inf_eff is not None else "free",
            a_inf_value=a_inf_eff,
            iqr_factor=iqr_factor,
            temperature_c=temperature_c,
        )
        result.label = f"{Path(filepath).stem} | {target_nm} nm"
        results.append(result)

    return results
