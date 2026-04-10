"""
Core functions for the Actinometer tab.

Chemical actinometry  – extracted from workflows/actinometer_analysis.py
LED characterisation  – extracted from workflows/quantum_yield.py (LED block)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter


# ── Physical constants ────────────────────────────────────────────────────────

H_PLANCK = 6.626070e-34   # J s
C_LIGHT  = 299792458.0    # m s⁻¹
NA       = 6.022141e+23   # mol⁻¹


# ── Actinometer library ───────────────────────────────────────────────────────

ACTINOMETERS: dict[int, dict] = {
    1: {
        "name":                "Actinometer 1",
        "wavelength_range_nm": (450, 580),
        "epsilon_ref_nm":      515,           # wavelength at which ε is specified
        "epsilon_ref_M_cm":    1.0e4,         # ε at epsilon_ref_nm [L mol⁻¹ cm⁻¹]
        "QY_func":             lambda lam: 10 ** (-0.796 + 133 / lam),
    },
    2: {
        "name":                "Actinometer 2",
        "wavelength_range_nm": (480, 620),
        "epsilon_ref_nm":      562,           # verify reference wavelength for this compound
        "epsilon_ref_M_cm":    1.09e4,
        "QY_func":             lambda lam: 10 ** (-2.67 + 526 / lam),
    },
}


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ActinometerResult:
    file:                    str
    actinometer_name:        str
    irradiation_nm:          float
    QY:                      float
    epsilon_M_cm:            float
    volume_mL:               float
    path_length_cm:          float
    photon_flux_mol_s:       float
    photon_flux_std_mol_s:   float
    r2:                      float
    intercept:               float
    t_valid:                 np.ndarray
    y_vals:                  np.ndarray
    y_pred:                  np.ndarray
    abs_valid:               np.ndarray
    success:                 bool
    error_msg:               str = ""


@dataclass
class LEDResult:
    wl_arr:             np.ndarray        # wavelength grid [nm], threshold-clipped
    N_arr:              np.ndarray        # spectral photon flux [mol s⁻¹ nm⁻¹]
    N_mol_s:            float             # ∫ N_arr dλ [mol s⁻¹]
    N_std_mol_s:        float             # 1-σ uncertainty [mol s⁻¹]
    lam_eff:            Optional[float]   # flux-weighted centroid [nm] (scalar mode only)
    integration_mode:   str               # "scalar" | "full"
    P_before_mW:        float
    P_after_mW:         Optional[float]
    P_used_mW:          float
    # full-range arrays for verification plot (before threshold cut)
    wl_full:            np.ndarray
    N_arr_full:         np.ndarray        # N(λ) on full grid [mol s⁻¹ nm⁻¹]
    em_int_pre_smooth:  np.ndarray        # averaged but unsmoothed emission
    em_int_raw_b:       np.ndarray        # raw before emission
    em_int_raw_a:       Optional[np.ndarray]  # raw after emission (on before grid)
    # power time series
    pwr_time_b:         np.ndarray        # time axis for before power [s]
    pwr_vals_b:         np.ndarray        # power values before [mW]
    pwr_time_a:         Optional[np.ndarray]  # time axis for after power [s]
    pwr_vals_a:         Optional[np.ndarray]  # power values after [mW]


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_spectra_csv(filepath) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load a multi-scan Cary 60 CSV (column pairs: Wavelength, Abs).
    Skips the first two header rows; ignores non-numeric trailing metadata.
    Returns list of (wavelength_array, absorbance_array), one per scan.
    """
    MIN_VALID = 5
    raw  = pd.read_csv(filepath, header=None)
    data = raw.iloc[2:].reset_index(drop=True)
    scans = []
    for i in range(0, data.shape[1] - 1, 2):
        wl = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        ab = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid = wl.notna() & ab.notna()
        if valid.sum() < MIN_VALID:
            continue
        scans.append((wl[valid].values, ab[valid].values))
    return scans


def extract_absorbance(wl: np.ndarray, ab: np.ndarray,
                       target_nm: float, tol_nm: float) -> float:
    """Mean absorbance within [target_nm ± tol_nm]; NaN if no match."""
    mask = (wl >= target_nm - tol_nm) & (wl <= target_nm + tol_nm)
    return float(ab[mask].mean()) if mask.any() else np.nan


def load_kinetic_csv(filepath: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load a multi-channel kinetic CSV (Cary 60 kinetics format).
    Row 0  : channel labels, e.g. "25C_672nm" every two columns.
    Row 1  : column headers ("Time (sec)", "Abs", …).
    Rows 2+: time / absorbance data pairs.
    Returns {label: (time_array, abs_array)}.
    """
    MIN_VALID = 5
    raw       = pd.read_csv(filepath, header=None)
    label_row = raw.iloc[0]
    data      = raw.iloc[2:].reset_index(drop=True)
    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(0, data.shape[1] - 1, 2):
        label = str(label_row.iloc[i]).strip()
        t_col = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        a_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid = t_col.notna() & a_col.notna()
        if valid.sum() < MIN_VALID:
            continue
        channels[label] = (t_col[valid].values, a_col[valid].values)
    return channels


def _parse_wl_from_label(label: str) -> Optional[float]:
    """
    Extract wavelength (nm) from a kinetic channel label.

    Handles formats such as:
      '25C_672nm'      → 672.0   (number followed by 'nm')
      'Sample 1_500'   → 500.0   (bare number at end of label, after _ or space)
      '532.5nm'        → 532.5
    Returns None if no wavelength can be parsed.
    """
    # Priority 1: explicit 'nm' suffix
    m = re.search(r'(\d+(?:\.\d+)?)\s*nm', label, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Priority 2: bare number at end of label (after _ or whitespace).
    # Require ≥ 200 to avoid treating channel indices like "Sample 1" as 1 nm.
    m = re.search(r'[_\s](\d+(?:\.\d+)?)\s*$', label)
    if m:
        val = float(m.group(1))
        if val >= 200:
            return val
    return None


def _load_initial_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the first scan from a Cary 60 CSV and return (wl, ab) sorted by wl.
    Used as the Beer-Lambert reference for kinetic-mode LED actinometry.
    """
    scans = load_spectra_csv(path)
    if not scans:
        raise ValueError(f"No valid scans found in initial spectrum file: {path.name}")
    wl, ab   = scans[0]
    order    = np.argsort(wl)
    return wl[order], ab[order]


# ── Chemical actinometry ──────────────────────────────────────────────────────

def _linear(x, slope, intercept):
    return slope * x + intercept


def run_actinometry_file(
    filepath:                   Path,
    actinometer_choice:         int,
    irradiation_wavelength_nm:  float,
    irradiation_time_s:         float,
    volume_mL:                  float,
    path_length_cm:             float,
    scans_per_group:            int,
    wavelength_tolerance_nm:    float,
) -> ActinometerResult:
    """
    Process one actinometry CSV file and return an ActinometerResult.
    Mirrors the per-file logic in workflows/actinometer_analysis.py.
    """
    act = ACTINOMETERS[actinometer_choice]
    wl_min, wl_max = act["wavelength_range_nm"]

    if not (wl_min <= irradiation_wavelength_nm <= wl_max):
        raise ValueError(
            f"{act['name']} is valid {wl_min}–{wl_max} nm; "
            f"λ_irr = {irradiation_wavelength_nm} nm is out of range."
        )

    QY = act["QY_func"](irradiation_wavelength_nm)

    scans = load_spectra_csv(filepath)
    n_scans = len(scans)
    print(f"  {n_scans} scans loaded from {filepath.name}")

    if n_scans < scans_per_group:
        raise ValueError(
            f"Only {n_scans} scans found; need ≥ {scans_per_group} per group."
        )

    n_groups = n_scans // scans_per_group
    print(f"  {n_groups} groups × {scans_per_group} scans/group")

    # ε at irradiation wavelength from middle scan of first group (index 1)
    lam_ref  = act["epsilon_ref_nm"]
    mid_scan = scans[1]
    A_ref = extract_absorbance(mid_scan[0], mid_scan[1], lam_ref,
                               wavelength_tolerance_nm)
    A_irr = extract_absorbance(mid_scan[0], mid_scan[1],
                               irradiation_wavelength_nm, wavelength_tolerance_nm)

    if np.isnan(A_ref) or A_ref <= 0 or np.isnan(A_irr):
        raise ValueError(
            f"Cannot compute ε: A({lam_ref:.0f}) = {A_ref:.4f}, "
            f"A_irr = {A_irr:.4f}"
        )

    epsilon    = act["epsilon_ref_M_cm"] * A_irr / A_ref
    V_m3       = volume_mL * 1e-6
    epsilon_SI = epsilon * 0.1           # L mol⁻¹ cm⁻¹ → m² mol⁻¹
    l_m        = path_length_cm * 1e-2   # cm → m
    prefactor  = -V_m3 / (epsilon_SI * QY * l_m)

    print(f"  ε_irr = {epsilon:.4e} L mol⁻¹ cm⁻¹  "
          f"(ε_ref({lam_ref:.0f}nm) = {act['epsilon_ref_M_cm']:.3e}, "
          f"A_irr/A_ref = {A_irr:.4f}/{A_ref:.4f})")

    # Group-average absorbances
    absorbances = []
    for g in range(n_groups):
        group_abs = [
            extract_absorbance(
                scans[g * scans_per_group + s][0],
                scans[g * scans_per_group + s][1],
                irradiation_wavelength_nm, wavelength_tolerance_nm,
            )
            for s in range(scans_per_group)
        ]
        valid_abs = [v for v in group_abs if not np.isnan(v)]
        absorbances.append(np.mean(valid_abs) if valid_abs else np.nan)

    time_axis = np.arange(n_groups) * irradiation_time_s

    # Rate function: y_i = prefactor × [log10(10^A_i − 1) − log10(10^A_0 − 1)]
    A_0 = absorbances[0]
    if np.isnan(A_0) or (10 ** A_0 - 1) <= 0:
        raise ValueError(f"A_0 = {A_0} — cannot compute log10(10^A_0 − 1).")

    log_ref = np.log10(10 ** A_0 - 1)
    t_valid, y_vals, abs_valid = [], [], []

    for i, A_i in enumerate(absorbances):
        if np.isnan(A_i):
            continue
        val = 10 ** A_i - 1
        if val <= 0:
            print(f"  Group {i} (t = {time_axis[i]:.0f} s): "
                  f"10^A − 1 ≤ 0 — skipping.")
            continue
        y_vals.append(prefactor * (np.log10(val) - log_ref))
        t_valid.append(time_axis[i])
        abs_valid.append(A_i)

    t_valid   = np.array(t_valid)
    y_vals    = np.array(y_vals)
    abs_valid = np.array(abs_valid)

    if len(t_valid) < 2:
        raise ValueError("Not enough valid points for a linear fit (< 2).")

    popt, pcov         = curve_fit(_linear, t_valid, y_vals)
    slope, intercept   = popt
    slope_std, _       = np.sqrt(np.diag(pcov))
    photon_flux        = slope
    photon_flux_std    = slope_std

    y_pred  = _linear(t_valid, slope, intercept)
    ss_res  = np.sum((y_vals - y_pred) ** 2)
    ss_tot  = np.sum((y_vals - y_vals.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"  Photon flux : {photon_flux:.4e} ± {photon_flux_std:.4e} mol s⁻¹  "
          f"(R² = {r2:.4f})")

    return ActinometerResult(
        file=filepath.name,
        actinometer_name=act["name"],
        irradiation_nm=irradiation_wavelength_nm,
        QY=QY,
        epsilon_M_cm=epsilon,
        volume_mL=volume_mL,
        path_length_cm=path_length_cm,
        photon_flux_mol_s=photon_flux,
        photon_flux_std_mol_s=photon_flux_std,
        r2=r2,
        intercept=intercept,
        t_valid=t_valid,
        y_vals=y_vals,
        y_pred=y_pred,
        abs_valid=abs_valid,
        success=True,
    )


def plot_actinometry_result(result: ActinometerResult) -> plt.Figure:
    """Return a matplotlib Figure for one actinometry result."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(result.t_valid, result.y_vals,
            "o", color="black", markerfacecolor="none",
            markeredgewidth=1.2, label="Data")

    t_fit = np.linspace(0, result.t_valid.max() * 1.05, 200)
    ax.plot(t_fit, _linear(t_fit, result.photon_flux_mol_s, result.intercept),
            "--", color="red", linewidth=2, label="Linear fit")

    annotation = (
        r"$y = \frac{-V}{\varepsilon\,\Phi\,l}"
        r"\left[\log_{10}(10^A-1)-\log_{10}(10^{A_0}-1)\right]$" + "\n"
        f"N = {result.photon_flux_mol_s:.4e} ± "
        f"{result.photon_flux_std_mol_s:.4e} mol s⁻¹\n"
        f"R² = {result.r2:.4f}"
    )
    ax.text(0.03, 0.97, annotation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel("Irradiation time (s)")
    ax.set_ylabel("Rate function (mol)")
    ax.set_title(
        f"{result.actinometer_name}  |  "
        f"λ_irr = {result.irradiation_nm:.0f} nm  |  "
        f"{result.file}"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


# ── LED characterisation ──────────────────────────────────────────────────────

def run_led_characterization(
    emission_before_path:   Path,
    emission_after_path:    Optional[Path],
    power_before_path:      Path,
    power_after_path:       Optional[Path],
    power_use:              str,      # "before" | "after" | "average"
    emission_threshold:     float,
    smoothing_enabled:      bool,
    smoothing_window:       int,
    smoothing_order:        int,
    integration_mode:       str,      # "scalar" | "full"
    photon_flux_std_manual: float,
) -> LEDResult:
    """
    Compute the LED spectral photon flux density N(λ) [mol s⁻¹ nm⁻¹].
    Mirrors the LED block in workflows/quantum_yield.py (lines 449–668).
    """
    # ── Load emission spectrum ─────────────────────────────────────────────
    em_df_b = pd.read_csv(emission_before_path, comment="#")
    em_wl   = em_df_b["wavelength_nm"].values.astype(float)
    em_int  = em_df_b["intensity_au"].values.astype(float)
    order   = np.argsort(em_wl)
    em_wl   = em_wl[order]
    em_int  = em_int[order]
    em_int_raw_b = em_int.copy()

    em_int_raw_a: Optional[np.ndarray] = None
    if emission_after_path is not None:
        em_df_a   = pd.read_csv(emission_after_path, comment="#")
        em_wl_a   = em_df_a["wavelength_nm"].values.astype(float)
        em_int_a  = em_df_a["intensity_au"].values.astype(float)
        ord_a     = np.argsort(em_wl_a)
        em_int_ai = np.interp(em_wl, em_wl_a[ord_a], em_int_a[ord_a])
        em_int_raw_a = em_int_ai.copy()
        em_int    = (em_int + em_int_ai) / 2.0
        print(f"  Emission   : averaged before + after")

    em_int_pre_smooth = em_int.copy()

    if smoothing_enabled:
        em_int = savgol_filter(em_int, smoothing_window, smoothing_order)
        print(f"  Smoothing  : SG window={smoothing_window}, order={smoothing_order}")

    # Full-range arrays for verification plot (before threshold cut)
    em_wl_full  = em_wl.copy()
    em_int_full = em_int.copy()

    em_peak = float(em_int.max())
    em_keep = em_int >= emission_threshold * em_peak
    em_wl   = em_wl[em_keep]
    em_int  = np.clip(em_int[em_keep], 0.0, None)
    print(f"  Threshold  : {emission_threshold} × peak  "
          f"→ {em_keep.sum()}/{len(em_keep)} points kept  "
          f"({em_wl[0]:.0f}–{em_wl[-1]:.0f} nm)")

    # ── Load power file(s) ─────────────────────────────────────────────────
    pwr_df_b   = pd.read_csv(power_before_path, comment="#")
    P_before   = float(pwr_df_b["power_mW"].mean())
    pwr_time_b = pwr_df_b["time_s"].values.astype(float)
    pwr_vals_b = pwr_df_b["power_mW"].values.astype(float)
    P_after: Optional[float] = None
    pwr_time_a: Optional[np.ndarray] = None
    pwr_vals_a: Optional[np.ndarray] = None

    if power_after_path is not None and power_after_path.exists():
        pwr_df_a = pd.read_csv(power_after_path, comment="#")
        P_after  = float(pwr_df_a["power_mW"].mean())
        pwr_time_a = pwr_df_a["time_s"].values.astype(float)
        pwr_vals_a = pwr_df_a["power_mW"].values.astype(float)

    if power_use == "before":
        P_used = P_before
    elif power_use == "after":
        if P_after is None:
            raise ValueError("power_use = 'after' but no after-power file loaded.")
        P_used = P_after
    elif power_use == "average":
        P_used = (P_before + P_after) / 2.0 if P_after is not None else P_before
    else:
        raise ValueError(f"Unknown power_use: '{power_use}'")

    P_W = P_used * 1e-3
    print(f"  P before   : {P_before:.4f} mW")
    if P_after is not None:
        drift = (P_after - P_before) / P_before * 100.0
        print(f"  P after    : {P_after:.4f} mW  (drift = {drift:+.2f} %)")
    print(f"  P used     : {P_used:.4f} mW  ({power_use})")

    # ── Compute N(λ) ──────────────────────────────────────────────────────
    em_wl_m  = em_wl * 1e-9                          # nm → m
    E_ph     = H_PLANCK * C_LIGHT / em_wl_m          # J photon⁻¹
    em_norm  = em_int / np.trapezoid(em_int, em_wl)  # normalised shape [nm⁻¹]
    N_arr    = em_norm * P_W / E_ph / NA              # mol s⁻¹ nm⁻¹
    N_mol_s  = float(np.trapezoid(N_arr, em_wl))

    # Full-grid N(λ) for verification plot
    em_norm_full = em_int_full / np.trapezoid(em_int_full, em_wl_full)
    E_ph_full    = H_PLANCK * C_LIGHT / (em_wl_full * 1e-9)
    N_arr_full   = em_norm_full * P_W / E_ph_full / NA

    lam_eff: Optional[float] = None
    if integration_mode == "scalar":
        lam_eff = float(np.trapezoid(em_wl * N_arr, em_wl) / N_mol_s)
        print(f"  Mode       : scalar  (λ_eff = {lam_eff:.1f} nm)")
    else:
        print(f"  Mode       : full spectral integration")

    print(f"  N_total    : {N_mol_s:.4e} mol s⁻¹  "
          f"over {em_wl[0]:.0f}–{em_wl[-1]:.0f} nm")

    # ── Uncertainty ───────────────────────────────────────────────────────
    if photon_flux_std_manual > 0.0:
        N_std = photon_flux_std_manual
        print(f"  N_std      : {N_std:.4e} mol s⁻¹  (manual)")
    elif P_after is not None:
        N_std = N_mol_s * abs(P_after - P_before) / (2.0 * P_used)
        print(f"  N_std      : {N_std:.4e} mol s⁻¹  "
              f"(auto from power drift)")
    else:
        N_std = 0.0
        print(f"  N_std      : 0  (no after-power file)")

    return LEDResult(
        wl_arr=em_wl,
        N_arr=N_arr,
        N_mol_s=N_mol_s,
        N_std_mol_s=N_std,
        lam_eff=lam_eff,
        integration_mode=integration_mode,
        P_before_mW=P_before,
        P_after_mW=P_after,
        P_used_mW=P_used,
        wl_full=em_wl_full,
        N_arr_full=N_arr_full,
        em_int_pre_smooth=em_int_pre_smooth,
        em_int_raw_b=em_int_raw_b,
        em_int_raw_a=em_int_raw_a,
        pwr_time_b=pwr_time_b,
        pwr_vals_b=pwr_vals_b,
        pwr_time_a=pwr_time_a,
        pwr_vals_a=pwr_vals_a,
    )


def plot_led_result(result: LEDResult) -> plt.Figure:
    """
    Two-panel verification plot for LED characterisation:
      Top    – emission spectra (raw, smoothed/averaged, threshold boundary)
      Bottom – spectral photon flux density N(λ)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    # ── Top: emission ──────────────────────────────────────────────────────
    wl_full  = result.wl_full
    raw_b_n  = result.em_int_raw_b / result.em_int_raw_b.max()
    ax1.plot(wl_full, raw_b_n,
             color="#4a90d9", linewidth=1, alpha=0.6, label="Raw (before)")
    if result.em_int_raw_a is not None:
        raw_a_n = result.em_int_raw_a / result.em_int_raw_a.max()
        ax1.plot(wl_full, raw_a_n,
                 color="#e87040", linewidth=1, alpha=0.6, label="Raw (after)")
    pre_n = result.em_int_pre_smooth / result.em_int_pre_smooth.max()
    ax1.plot(wl_full, pre_n,
             color="black", linewidth=1.5, label="Used (avg ± smoothed)")
    # Threshold line
    ax1.axhline(
        result.wl_arr[0] and  # just to reference the object – draw threshold
        (result.N_arr[0] / result.N_arr.max()),
        color="grey", linestyle=":", linewidth=1,
    )
    # Mark clipped region
    ax1.axvline(result.wl_arr[0],  color="grey", linestyle="--", linewidth=0.8)
    ax1.axvline(result.wl_arr[-1], color="grey", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Normalised intensity")
    ax1.set_title("LED Emission Spectrum")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Bottom: N(λ) ───────────────────────────────────────────────────────
    N_pm = result.N_arr * 1e12   # mol s⁻¹ nm⁻¹ → pmol s⁻¹ nm⁻¹
    ax2.plot(result.wl_arr, N_pm, color="#3cb371", linewidth=2, label="N(λ)")
    ax2.fill_between(result.wl_arr, N_pm, alpha=0.2, color="#3cb371")

    if result.lam_eff is not None:
        ax2.axvline(result.lam_eff, color="red", linestyle="--", linewidth=1.2,
                    label=f"λ_eff = {result.lam_eff:.1f} nm")

    summary = (
        f"N_total = {result.N_mol_s:.4e} mol s⁻¹\n"
        f"N_std   = {result.N_std_mol_s:.4e} mol s⁻¹\n"
        f"P_used  = {result.P_used_mW:.4f} mW\n"
        f"Mode    : {result.integration_mode}"
    )
    if result.lam_eff is not None:
        summary += f"\nλ_eff   = {result.lam_eff:.1f} nm"
    ax2.text(0.97, 0.97, summary,
             transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("N(λ)  (pmol s⁻¹ nm⁻¹)")
    ax2.set_title("Spectral Photon Flux Density")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_led_diagnostic(
    emission_before_path: Path,
    emission_after_path:  Optional[Path],
    power_before_path:    Path,
    power_after_path:     Optional[Path],
) -> plt.Figure:
    """
    Stage-1 control figure: two side-by-side panels loaded straight from the
    raw CSVs, with no processing applied.

      Left  – emission spectra (raw intensity_au, not normalised)
      Right – power time series (before + after overlayed; drift % in title)
    """
    fig, (ax_em, ax_pw) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: raw emission ─────────────────────────────────────────────────
    em_df_b = pd.read_csv(emission_before_path, comment="#")
    wl_b    = em_df_b["wavelength_nm"].values.astype(float)
    int_b   = em_df_b["intensity_au"].values.astype(float)
    ax_em.plot(wl_b, int_b, color="#4a90d9", linewidth=1.2, label="Before")

    if emission_after_path is not None and emission_after_path.exists():
        em_df_a = pd.read_csv(emission_after_path, comment="#")
        wl_a    = em_df_a["wavelength_nm"].values.astype(float)
        int_a   = em_df_a["intensity_au"].values.astype(float)
        ax_em.plot(wl_a, int_a, color="#e87040", linewidth=1.2, label="After")

    ax_em.set_xlabel("Wavelength (nm)")
    ax_em.set_ylabel("Intensity (a.u.)")
    ax_em.set_title("Emission spectra (raw, not normalised)")
    ax_em.legend(fontsize=8)
    ax_em.grid(True, alpha=0.3)

    # ── Right: power time series ───────────────────────────────────────────
    pwr_df_b = pd.read_csv(power_before_path, comment="#")
    t_b      = pwr_df_b["time_s"].values.astype(float)
    p_b      = pwr_df_b["power_mW"].values.astype(float)
    P_b_mean = float(p_b.mean())
    ax_pw.plot(t_b, p_b, color="#3a7ebf", linewidth=1.2, alpha=0.85, label="Before")
    ax_pw.axhline(P_b_mean, color="#3a7ebf", linewidth=1.0, linestyle="--",
                  label=f"Mean = {P_b_mean:.4f} mW")

    drift_title = ""
    if power_after_path is not None and power_after_path.exists():
        pwr_df_a = pd.read_csv(power_after_path, comment="#")
        t_a      = pwr_df_a["time_s"].values.astype(float)
        p_a      = pwr_df_a["power_mW"].values.astype(float)
        P_a_mean = float(p_a.mean())
        ax_pw.plot(t_a, p_a, color="#e87d37", linewidth=1.2, alpha=0.85, label="After")
        ax_pw.axhline(P_a_mean, color="#e87d37", linewidth=1.0, linestyle="--",
                      label=f"Mean = {P_a_mean:.4f} mW")
        drift_pct  = (P_a_mean - P_b_mean) / P_b_mean * 100.0
        drift_title = f"  (drift = {drift_pct:+.2f} %)"

    ax_pw.set_xlabel("Time (s)")
    ax_pw.set_ylabel("Power (mW)")
    ax_pw.set_title(f"Power time series{drift_title}")
    ax_pw.legend(fontsize=8)
    ax_pw.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── LED chemical actinometry ──────────────────────────────────────────────────

@dataclass
class LEDActinometerResult:
    """Result of one actinometry CSV processed under LED irradiation."""
    file:                    str
    actinometer_name:        str
    integration_mode:        str            # "scalar" | "spectral"
    lam_eff:                 float          # flux-weighted LED centroid [nm]
    lam_mon:                 float          # actual monitoring wavelength [nm]
    epsilon_eff_M_cm:        float          # ε at λ_mon [L mol⁻¹ cm⁻¹]
    QY_eff:                  float          # effective QY (ε-weighted for spectral mode)
    volume_mL:               float
    path_length_cm:          float
    photon_flux_mol_s:       float          # N_chem from linear fit slope [mol s⁻¹]
    photon_flux_std_mol_s:   float
    r2:                      float
    intercept:               float
    t_valid:                 np.ndarray
    y_vals:                  np.ndarray
    y_pred:                  np.ndarray
    abs_valid:               np.ndarray
    N_LED_mol_s:             Optional[float] = None
    N_LED_std_mol_s:         Optional[float] = None
    success:                 bool = True
    error_msg:               str = ""
    channel:                 str = ""       # kinetic channel label (empty for scanning)
    flux_fraction:           float = 1.0   # fraction of LED flux in actinometer valid range
    fit_slope:               float = 0.0   # slope of ln(A) vs t = −k_obs (dilute mode only)
    fit_method:              str   = "ODE" # "ODE" | "dilute"
    t_smooth:                Optional[np.ndarray] = None   # fine time grid for ODE plot curve
    y_smooth:                Optional[np.ndarray] = None   # ODE solution on t_smooth (absorbance)


def run_led_actinometry_file(
    filepath:                Path,
    actinometer_choice:      int,
    led_wl_arr:              np.ndarray,
    led_N_arr:               np.ndarray,
    lam_eff:                 float,
    integration_mode:        str,           # "scalar" | "spectral"
    data_type:               str,           # "scanning" | "kinetic"
    irradiation_time_s:      float,
    volume_mL:               float,
    path_length_cm:          float,
    scans_per_group:         int,
    wavelength_tolerance_nm: float,
    skip_groups:             int            = 0,
    initial_spectrum_path:   Optional[Path] = None,
    fit_time_start_s:        Optional[float] = None,
    fit_time_end_s:          Optional[float] = None,
    N_LED_mol_s:             Optional[float] = None,
    N_LED_std_mol_s:         Optional[float] = None,
    monitoring_wavelength_nm: float           = 0.0,
) -> list["LEDActinometerResult"]:
    """
    Process one actinometry CSV measured under LED irradiation.

    Returns a list of LEDActinometerResult — one entry per monitoring channel:
      • scanning data  → one result (monitored at λ_eff)
      • kinetic data   → one result per channel with a parseable wavelength label

    Integration modes
    -----------------
    scalar   : treats LED as monochromatic at λ_eff.
               ε = ε(λ_mon) from Beer-Lambert reference.
               QY = QY(λ_eff).
               Only the kinetic channel closest to λ_eff is used.

    spectral : integrates over the full LED emission spectrum.
               ε(λ_mon) used for Beer-Lambert monitoring.
               QY_eff = ∫ f(λ)·QY(λ)·[ε(λ)/ε(λ_mon)] dλ  (dimensionless,
               captures both spectral QY variation and how each irradiation
               wavelength drives bleaching relative to the monitoring wavelength).
               All kinetic channels with parseable wavelengths are processed.
    """
    act    = ACTINOMETERS[actinometer_choice]
    wl_min, wl_max = act["wavelength_range_nm"]

    # ── Step 1: Beer-Lambert reference spectrum ───────────────────────────
    if data_type == "scanning":
        scans   = load_spectra_csv(filepath)
        n_scans = len(scans)
        print(f"  {n_scans} scans loaded from {filepath.name}")
        if n_scans < scans_per_group:
            raise ValueError(
                f"Only {n_scans} scans; need ≥ {scans_per_group} per group.")
        n_groups   = n_scans // scans_per_group
        ref_wl, ref_ab = scans[1]
        idx = np.argsort(ref_wl)
        ref_wl, ref_ab = ref_wl[idx], ref_ab[idx]

    elif data_type == "kinetic":
        if initial_spectrum_path is None:
            raise ValueError(
                "Kinetic mode requires an initial spectrum file for Beer-Lambert "
                "scaling.")
        ref_wl, ref_ab = _load_initial_spectrum(initial_spectrum_path)
        print(f"  Kinetic mode: Beer-Lambert reference from "
              f"{initial_spectrum_path.name}")

    else:
        raise ValueError(f"Unknown data_type: '{data_type}'")

    # ── Step 2: A_ref ─────────────────────────────────────────────────────
    lam_ref = act["epsilon_ref_nm"]
    A_ref   = extract_absorbance(ref_wl, ref_ab, lam_ref, wavelength_tolerance_nm)
    if np.isnan(A_ref) or A_ref <= 0:
        ref_name = (initial_spectrum_path.name if data_type == "kinetic"
                    and initial_spectrum_path else filepath.name)
        raise ValueError(
            f"Cannot extract A({lam_ref:.0f} nm) = {A_ref} from the Beer-Lambert "
            f"reference spectrum ('{ref_name}').\n"
            f"For kinetic mode this must be a Cary 60 full-spectrum scan CSV, "
            f"not the kinetic data file itself.")

    # ── Step 3: Shared spectral arrays (spectral mode only) ───────────────
    N_total = float(np.trapezoid(led_N_arr, led_wl_arr))
    f_arr   = led_N_arr / N_total

    wl_v = f_v = f_v_norm = eps_v = QY_v = None   # populated below for spectral mode
    # Spectral pre-computed values (used in per-channel loop)
    QY_eff_spectral = eps_eff_spectral_M_cm = driving_integral_spectral = flux_f_spectral = None
    if integration_mode == "spectral":
        A_interp    = np.interp(led_wl_arr, ref_wl, ref_ab,
                                left=np.nan, right=np.nan)
        epsilon_arr = act["epsilon_ref_M_cm"] * A_interp / A_ref
        QY_arr      = np.where(
            (led_wl_arr >= wl_min) & (led_wl_arr <= wl_max),
            np.array([act["QY_func"](lam) for lam in led_wl_arr]),
            np.nan,
        )
        valid = (np.isfinite(epsilon_arr) & np.isfinite(QY_arr)
                 & (epsilon_arr > 0))
        if valid.sum() < 5:
            raise ValueError(
                f"Only {valid.sum()} valid spectral points for flux-weighted "
                f"integration (need ≥ 5). Check LED emission overlap with "
                f"the actinometer's valid range ({wl_min}–{wl_max} nm).")
        wl_v      = led_wl_arr[valid]
        f_v       = f_arr[valid]
        eps_v     = epsilon_arr[valid]
        eps_v_SI  = eps_v * 0.1            # L/(mol·cm) → m²/mol, for ODE
        QY_v      = QY_arr[valid]
        flux_f_spectral = float(np.trapezoid(f_v, wl_v))
        f_v_norm        = f_v / np.trapezoid(f_v, wl_v)

        # QY_eff: pure flux-weighted QY (no ε ratio) — averaged over valid range
        QY_eff_spectral      = float(np.trapezoid(f_v_norm * QY_v, wl_v))
        # ε_eff: flux-weighted ε averaged over valid range — for display
        eps_eff_spectral_M_cm = float(np.trapezoid(f_v_norm * eps_v, wl_v))
        # Driving integral ∫ f(λ)·QY(λ)·ε(λ) dλ using original (non-renormalised) f
        # Automatically includes flux_fraction: photons outside valid range have QY=0
        driving_integral_spectral = float(np.trapezoid(f_v * QY_v * eps_v, wl_v)) * 0.1  # SI: m²/mol

        excluded_pct = (1.0 - flux_f_spectral) * 100.0
        print(f"  Mode      : spectral  ({valid.sum()} pts, "
              f"{wl_v[0]:.0f}–{wl_v[-1]:.0f} nm, "
              f"{flux_f_spectral*100:.1f}% of LED flux included, "
              f"{excluded_pct:.1f}% excluded outside actinometer range)")
        print(f"  QY_eff    : {QY_eff_spectral:.4f}  (flux-weighted over valid range)")
        print(f"  ε_eff     : {eps_eff_spectral_M_cm:.4e} L mol⁻¹ cm⁻¹  (flux-weighted over valid range)")

    # ── Step 4: Build per-channel work list ───────────────────────────────
    # Each entry: (lam_mon, channel_label, time_array, absorbances_list)
    channels_to_process: list = []

    if data_type == "scanning":
        if skip_groups >= n_groups:
            raise ValueError(
                f"skip_groups = {skip_groups} ≥ n_groups = {n_groups}.")
        if skip_groups > 0:
            print(f"  Skipping first {skip_groups} group(s).")
        absorbances = []
        for g in range(skip_groups, n_groups):
            group_abs = [
                extract_absorbance(
                    scans[g * scans_per_group + j][0],
                    scans[g * scans_per_group + j][1],
                    lam_eff, wavelength_tolerance_nm,
                )
                for j in range(scans_per_group)
                if g * scans_per_group + j < n_scans
            ]
            valid_abs = [v for v in group_abs if not np.isnan(v)]
            absorbances.append(np.mean(valid_abs) if valid_abs else np.nan)
        time_axis = np.arange(len(absorbances)) * irradiation_time_s
        channels_to_process.append((lam_eff, "", time_axis, absorbances))

    else:  # kinetic
        all_channels = load_kinetic_csv(filepath)
        if not all_channels:
            raise ValueError(f"No valid channels found in {filepath.name}.")

        if integration_mode == "scalar":
            # Scalar: only the channel closest to λ_eff
            best_label: Optional[str] = None
            best_dist  = float("inf")
            best_wl    = lam_eff
            for label in all_channels:
                wl_p = _parse_wl_from_label(label)
                if wl_p is None and monitoring_wavelength_nm > 0:
                    wl_p = monitoring_wavelength_nm
                if wl_p is not None and abs(wl_p - lam_eff) < best_dist:
                    best_dist  = abs(wl_p - lam_eff)
                    best_label = label
                    best_wl    = wl_p
            if best_label is None:
                raise ValueError(
                    f"Cannot parse wavelengths from channel labels in "
                    f"{filepath.name}. Labels: {list(all_channels.keys())}.\n"
                    f"Set 'Monitoring wavelength' in the kinetic parameters to specify it manually.")
            selected = {best_label: (best_wl, all_channels[best_label])}
            print(f"  Scalar: channel '{best_label}'  "
                  f"(λ_mon = {best_wl:.0f} nm, "
                  f"Δλ = {best_dist:.1f} nm from λ_eff = {lam_eff:.1f} nm)")
        else:
            # Spectral: all channels with parseable wavelengths (or monitoring_wavelength_nm fallback)
            selected = {}
            for label, data in all_channels.items():
                wl_p = _parse_wl_from_label(label)
                if wl_p is None:
                    if monitoring_wavelength_nm > 0:
                        wl_p = monitoring_wavelength_nm
                        print(f"  Channel '{label}': no wavelength in label — "
                              f"using manual λ_mon = {monitoring_wavelength_nm:.0f} nm")
                    else:
                        print(f"  Skipping '{label}': cannot parse wavelength "
                              f"(set 'Monitoring wavelength' to specify manually).")
                        continue
                selected[label] = (wl_p, data)
            if not selected:
                raise ValueError(
                    f"No channels with parseable wavelengths in {filepath.name}. "
                    f"Labels: {list(all_channels.keys())}.\n"
                    f"Set 'Monitoring wavelength' in the kinetic parameters to specify it manually.")

        for label, (lam_mon_ch, (t_raw, a_raw)) in selected.items():
            mask = np.ones(len(t_raw), dtype=bool)
            if fit_time_start_s is not None:
                mask &= t_raw >= fit_time_start_s
            if fit_time_end_s is not None:
                mask &= t_raw <= fit_time_end_s
            t_win = t_raw[mask]
            a_win = a_raw[mask]
            if len(t_win) < 2:
                print(f"  Skipping '{label}': too few points in time window.")
                continue
            # Offset correction: align kinetic start to initial spectrum at λ_mon
            A_init = extract_absorbance(ref_wl, ref_ab, lam_mon_ch,
                                        wavelength_tolerance_nm)
            if not np.isnan(A_init):
                n_bl        = min(5, len(a_win))
                A_kin_start = float(a_win[:n_bl].mean())
                offset      = A_init - A_kin_start
                if abs(offset) > 1e-4:
                    print(f"  Offset [{label}]: {offset:+.4f} AU  "
                          f"(A_init = {A_init:.4f}, "
                          f"A_kin_t0 = {A_kin_start:.4f})")
                a_win = a_win + offset
            else:
                print(f"  WARNING [{label}]: Cannot extract A at "
                      f"λ={lam_mon_ch:.0f} nm — no offset correction.")
            time_axis = t_win - t_win[0]
            channels_to_process.append(
                (lam_mon_ch, label, time_axis, list(a_win)))
            print(f"  Channel   : '{label}'  (λ_mon = {lam_mon_ch:.0f} nm)")

    if not channels_to_process:
        raise ValueError(f"No processable channels for {filepath.name}.")

    # ── Step 5: Per-channel QY_eff, ε_eff, driving integral, fit ────────────
    # Formula: A_mon(t) = A_mon(0) · exp(−k·t)  [dilute limit]
    #   → ln(A_mon) = ln(A_mon(0)) − k·t   (linear fit, slope = −k)
    #   → N_total = k·V / (ln(10)·l·∫f(λ)·QY(λ)·ε(λ)dλ)
    # The driving integral uses the original (non-renormalised) LED spectrum f(λ),
    # so photons outside the actinometer's valid range (QY=0) contribute nothing
    # and the flux fraction is automatically accounted for.
    results: list[LEDActinometerResult] = []

    V_m3 = volume_mL * 1e-6
    l_m  = path_length_cm * 1e-2

    for lam_mon, channel_label, time_axis, absorbances in channels_to_process:
        # ε at monitoring wavelength — needed only for the offset correction
        A_lam_mon = extract_absorbance(ref_wl, ref_ab, lam_mon,
                                       wavelength_tolerance_nm)
        if np.isnan(A_lam_mon) or A_lam_mon <= 0:
            print(f"  WARNING: Cannot extract A at λ_mon = {lam_mon:.0f} nm "
                  f"from reference spectrum — skipping channel.")
            continue
        epsilon_mon = act["epsilon_ref_M_cm"] * A_lam_mon / A_ref

        if integration_mode == "scalar":
            QY_eff = act["QY_func"](lam_eff)
            if not (wl_min <= lam_eff <= wl_max):
                print(f"  WARNING: λ_eff = {lam_eff:.1f} nm outside valid range "
                      f"({wl_min}–{wl_max} nm) for {act['name']}.")
            # ε at the irradiation wavelength drives the photochemistry
            A_lam_eff    = extract_absorbance(ref_wl, ref_ab, lam_eff, wavelength_tolerance_nm)
            epsilon_irr  = (act["epsilon_ref_M_cm"] * A_lam_eff / A_ref
                            if not np.isnan(A_lam_eff) and A_lam_eff > 0 else epsilon_mon)
            # driving integral: QY(λ_eff) × ε(λ_eff)  [SI: m²/mol]
            driving_integral = QY_eff * epsilon_irr * 0.1
            eps_eff_display  = epsilon_irr
            flux_fraction    = 1.0
            print(f"  Mode      : scalar  (λ_eff = {lam_eff:.1f} nm, "
                  f"λ_mon = {lam_mon:.0f} nm)")
            print(f"  ε(λ_eff)  : {epsilon_irr:.4e} L mol⁻¹ cm⁻¹  (driving)")
            print(f"  ε(λ_mon)  : {epsilon_mon:.4e} L mol⁻¹ cm⁻¹  (monitoring, offset correction only)")
            print(f"  QY(λ_eff) : {QY_eff:.4f}")

        else:  # spectral
            # Use pre-computed spectral integrals from Step 3
            QY_eff           = QY_eff_spectral
            eps_eff_display  = eps_eff_spectral_M_cm
            driving_integral = driving_integral_spectral   # already in SI (m²/mol)
            flux_fraction    = flux_f_spectral
            print(f"  Channel λ_mon = {lam_mon:.0f} nm  |  "
                  f"QY_eff = {QY_eff:.4f}  |  "
                  f"ε_eff = {eps_eff_display:.4e} L mol⁻¹ cm⁻¹  |  "
                  f"flux fraction = {flux_fraction:.3f}")

        # ── Build valid data arrays ──────────────────────────────────────────
        A_0 = absorbances[0]
        if np.isnan(A_0) or A_0 <= 0:
            lbl = channel_label or "scan"
            print(f"  WARNING [{lbl}]: A_0 = {A_0:.4f} — cannot start fit. Skipping.")
            continue

        t_valid, abs_valid = [], []
        for i, A_i in enumerate(absorbances):
            if np.isnan(A_i) or A_i <= 0:
                if not np.isnan(A_i):
                    print(f"  t = {time_axis[i]:.0f} s: A ≤ 0 — skipping point.")
                continue
            t_valid.append(time_axis[i])
            abs_valid.append(A_i)

        t_valid   = np.array(t_valid)
        abs_valid = np.array(abs_valid)

        if len(t_valid) < 2:
            print(f"  Not enough valid points for channel '{channel_label or 'scan'}'.")
            continue

        # ── EXACT ODE FIT ────────────────────────────────────────────────────
        # Exact rate law (no dilute approximation):
        #   d[C]/dt = -(N/V) · ∫ f(λ)·QY(λ)·(1 − 10^{−ε(λ)·l·[C]}) dλ
        # where (1 − 10^{−ε(λ)·l·[C]}) is the exact fraction of photons absorbed.
        # [C](t) is solved numerically; N_total is the single free parameter.
        # A_mon(t) = ε_mon · l · [C](t)  for comparison with measured absorbance.

        epsilon_mon_SI = epsilon_mon * 0.1    # L/(mol·cm) → m²/mol

        # Initial concentration from first absorbance point
        C0 = A_0 / (epsilon_mon_SI * l_m)

        # Build the ODE integrand function for this channel
        if integration_mode == "scalar":
            _eps_irr_SI = epsilon_irr * 0.1
            _QY_s       = QY_eff
            def _integrand(C, _e=_eps_irr_SI, _q=_QY_s):
                Av = _e * l_m * C
                return _q * (1.0 - 10.0 ** (-float(np.clip(Av, 0, 500))))
        else:  # spectral
            _ev = eps_v_SI   # vector, m²/mol
            _fv = f_v
            _qv = QY_v
            _wv = wl_v
            def _integrand(C, _e=_ev, _f=_fv, _q=_qv, _w=_wv):
                Av   = _e * l_m * C
                frac = 1.0 - 10.0 ** (-np.clip(Av, 0, 500))
                return float(np.trapezoid(_f * _q * frac, _w))

        def _ode(t, C_vec, N):
            return [-(N / V_m3) * _integrand(C_vec[0])]

        def _run_ode(N_trial, t_eval):
            sol = solve_ivp(_ode, [t_eval[0], t_eval[-1]], [C0],
                            t_eval=t_eval, args=(N_trial,),
                            method='RK45', rtol=1e-6, atol=1e-12)
            return sol.y[0] if sol.success else None

        def _A_pred(N_trial, t_eval=t_valid):
            C = _run_ode(N_trial, t_eval)
            return epsilon_mon_SI * l_m * C if C is not None else None

        def _ssr(log_N):
            A = _A_pred(np.exp(log_N))
            return float(np.sum((A - abs_valid) ** 2)) if A is not None else 1e30

        # Initial estimate from dilute approximation (fast, for bounds)
        k_lin_init  = max(-np.polyfit(t_valid, np.log(abs_valid), 1)[0], 1e-10)
        N_init      = k_lin_init * V_m3 / (np.log(10) * l_m * driving_integral)
        N_init      = max(N_init, 1e-15)

        opt = minimize_scalar(_ssr,
                              bounds=(np.log(N_init * 1e-3), np.log(N_init * 1e3)),
                              method='bounded',
                              options={'xatol': 1e-12})
        N_chem = float(np.exp(opt.x))

        # Uncertainty via numerical Jacobian: σ_N = σ_res / ‖dA/dN‖
        h_N = N_chem * 1e-4
        A_p = _A_pred(N_chem + h_N)
        A_m = _A_pred(N_chem - h_N)
        if A_p is not None and A_m is not None:
            J          = (A_p - A_m) / (2.0 * h_N)
            sigma_res  = np.sqrt(opt.fun / max(len(t_valid) - 1, 1))
            JtJ        = float(np.dot(J, J))
            N_chem_std = (sigma_res / np.sqrt(JtJ)) if JtJ > 0 else np.nan
        else:
            N_chem_std = np.nan

        # Fit quality (R² on absorbance)
        A_fitted = _A_pred(N_chem)
        if A_fitted is None:
            print(f"  ODE solution failed for channel '{channel_label or 'scan'}'.")
            continue
        ss_res = float(np.sum((abs_valid - A_fitted) ** 2))
        ss_tot = float(np.sum((abs_valid - abs_valid.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Smooth ODE curve for plotting (300-point fine grid)
        t_smooth = np.linspace(t_valid[0], t_valid[-1] * 1.02, 300)
        C_smooth = _run_ode(N_chem, t_smooth)
        y_smooth = epsilon_mon_SI * l_m * C_smooth if C_smooth is not None else A_fitted

        lbl = f"'{channel_label}'" if channel_label else "scan"
        print(f"  [{lbl}] N_chem : {N_chem:.4e} ± {N_chem_std:.4e} mol s⁻¹  "
              f"(R² = {r2:.4f},  method = ODE exact)")
        if N_LED_mol_s is not None:
            dev = (N_chem - N_LED_mol_s) / N_LED_mol_s * 100.0
            print(f"  vs N_LED  : {N_LED_mol_s:.4e} mol s⁻¹  "
                  f"(deviation = {dev:+.2f} %)")

        results.append(LEDActinometerResult(
            file=filepath.name,
            actinometer_name=act["name"],
            integration_mode=integration_mode,
            lam_eff=lam_eff,
            lam_mon=lam_mon,
            epsilon_eff_M_cm=eps_eff_display,
            QY_eff=QY_eff,
            volume_mL=volume_mL,
            path_length_cm=path_length_cm,
            photon_flux_mol_s=N_chem,
            photon_flux_std_mol_s=N_chem_std,
            r2=r2,
            intercept=0.0,
            t_valid=t_valid,
            y_vals=abs_valid,
            y_pred=A_fitted,
            abs_valid=abs_valid,
            N_LED_mol_s=N_LED_mol_s,
            N_LED_std_mol_s=N_LED_std_mol_s,
            success=True,
            channel=channel_label,
            flux_fraction=flux_fraction,
            fit_slope=0.0,
            fit_method="ODE",
            t_smooth=t_smooth,
            y_smooth=y_smooth,
        ))

        # ── DILUTE APPROXIMATION (kept for reference, not used) ───────────────
        # Valid only when A << 0.1 at all irradiated wavelengths.
        # Linearises (1−10^{−A}) ≈ A·ln10, giving first-order exponential decay.
        #   ln(A_mon) = ln(A_0) − k·t   (linear fit, slope = −k)
        #   N = k·V / (ln10·l·∫f(λ)·QY(λ)·ε(λ)dλ)
        #
        # y_vals_d = np.log(abs_valid)
        # popt, pcov = curve_fit(_linear, t_valid, y_vals_d)
        # fit_slope_d, intercept_d = popt
        # slope_std_d, _ = np.sqrt(np.diag(pcov))
        # k_obs_d    = -fit_slope_d
        # N_chem_d   = k_obs_d * V_m3 / (np.log(10) * l_m * driving_integral)
        # N_chem_std_d = slope_std_d * V_m3 / (np.log(10) * l_m * driving_integral)
        # y_pred_d   = _linear(t_valid, fit_slope_d, intercept_d)
        # r2_d       = 1 - np.sum((y_vals_d-y_pred_d)**2)/np.sum((y_vals_d-y_vals_d.mean())**2)
        # ─────────────────────────────────────────────────────────────────────

    if not results:
        raise ValueError(
            f"No valid results produced for {filepath.name}.")

    return results


def plot_led_actinometry_result(result: "LEDActinometerResult") -> plt.Figure:
    """
    Two-panel figure for LED actinometry:
      Left  – rate function vs irradiation time with linear fit and parameters
      Right – bar chart comparing N_chem vs N_LED (shown only if N_LED available)
    """
    has_ref = result.N_LED_mol_s is not None
    fig, axes = plt.subplots(
        1, 2 if has_ref else 1,
        figsize=(12 if has_ref else 7, 5),
    )
    ax1 = axes[0] if has_ref else axes
    ax2 = axes[1] if has_ref else None

    # ── Left: absorbance vs time with fit ─────────────────────────────────
    is_ode = getattr(result, "fit_method", "ODE") == "ODE"

    ax1.plot(result.t_valid, result.y_vals,
             "o", color="black", markerfacecolor="none",
             markeredgewidth=1.2, label="Data")

    if is_ode and result.t_smooth is not None:
        ax1.plot(result.t_smooth, result.y_smooth,
                 "-", color="red", linewidth=2, label="ODE fit (exact)")
    else:
        # Dilute approximation: reconstruct linear fit on ln(A) scale — replot as A
        t_fit = np.linspace(result.t_valid[0], result.t_valid.max() * 1.05, 200)
        ax1.plot(t_fit, np.exp(_linear(t_fit, result.fit_slope, result.intercept)),
                 "--", color="red", linewidth=2, label="Linear fit (dilute)")

    mode_lbl    = "scalar" if result.integration_mode == "scalar" else "spectral"
    QY_lbl      = "QY(λ_eff)" if result.integration_mode == "scalar" else "QY_eff"
    lam_mon_str = f"{result.lam_mon:.1f}" if hasattr(result, "lam_mon") else f"{result.lam_eff:.1f}"
    eps_lbl     = "ε(λ_eff)" if result.integration_mode == "scalar" else "ε_eff (flux-wtd)"
    fit_lbl     = "ODE (exact)" if is_ode else "dilute approx."
    flux_str    = (f"Flux in valid range: {result.flux_fraction*100:.1f}%\n"
                   if result.integration_mode == "spectral" else "")
    annotation = (
        f"Fit: {fit_lbl}  |  Mode: {mode_lbl}\n"
        f"λ_eff = {result.lam_eff:.1f} nm,  λ_mon = {lam_mon_str} nm\n"
        f"{flux_str}"
        f"{eps_lbl} = {result.epsilon_eff_M_cm:.3e} L mol⁻¹ cm⁻¹\n"
        f"{QY_lbl} = {result.QY_eff:.4f}\n"
        f"N_chem = {result.photon_flux_mol_s:.4e} ± "
        f"{result.photon_flux_std_mol_s:.4e} mol s⁻¹\n"
        f"R² = {result.r2:.4f}"
    )
    ax1.text(0.03, 0.97, annotation,
             transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax1.set_xlabel("Irradiation time (s)")
    ax1.set_ylabel("Absorbance")
    ch_str = f"  |  {result.channel}" if getattr(result, "channel", "") else ""
    ax1.set_title(f"{result.actinometer_name}  |  {result.file}{ch_str}")
    ax1.legend()
    ax1.grid(True)

    # ── Right: comparison bar chart ────────────────────────────────────────
    if ax2 is not None and result.N_LED_mol_s is not None:
        labels = ["N_chem", "N_LED"]
        values = [result.photon_flux_mol_s, result.N_LED_mol_s]
        errs   = [
            result.photon_flux_std_mol_s,
            result.N_LED_std_mol_s if result.N_LED_std_mol_s else 0.0,
        ]
        colors = ["#4a90d9", "#e87040"]

        bars = ax2.bar(labels, values, color=colors, alpha=0.75,
                       yerr=errs, capsize=6,
                       error_kw={"linewidth": 1.5, "capthick": 1.5})

        dev = (result.photon_flux_mol_s - result.N_LED_mol_s) / result.N_LED_mol_s * 100.0
        for bar, val, err in zip(bars, values, errs):
            ax2.text(bar.get_x() + bar.get_width() / 2.0,
                     val + max(err, val * 0.02) * 1.15,
                     f"{val:.3e}", ha="center", va="bottom", fontsize=9)

        ax2.set_ylabel("Photon flux (mol s⁻¹)")
        ax2.set_title(f"N_chem vs N_LED  (deviation = {dev:+.2f} %)")
        ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    return fig
