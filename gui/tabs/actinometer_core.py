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
from scipy.optimize import curve_fit
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
    # Priority 2: bare number at end of label (after _ or whitespace)
    m = re.search(r'[_\s](\d+(?:\.\d+)?)\s*$', label)
    if m:
        return float(m.group(1))
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
    lam_eff:                 float          # flux-weighted centroid used as monitor λ [nm]
    epsilon_eff_M_cm:        float          # effective ε [L mol⁻¹ cm⁻¹]
    QY_eff:                  float          # effective quantum yield
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
    N_LED_mol_s:             Optional[float] = None   # reference from LED panel
    N_LED_std_mol_s:         Optional[float] = None
    success:                 bool = True
    error_msg:               str = ""


def run_led_actinometry_file(
    filepath:                Path,
    actinometer_choice:      int,
    led_wl_arr:              np.ndarray,    # LED emission grid [nm], threshold-clipped
    led_N_arr:               np.ndarray,    # spectral photon flux [mol s⁻¹ nm⁻¹]
    lam_eff:                 float,         # flux-weighted centroid [nm]
    integration_mode:        str,           # "scalar" | "spectral"
    data_type:               str,           # "scanning" | "kinetic"
    irradiation_time_s:      float,         # scanning: interval per group; kinetic: unused
    volume_mL:               float,
    path_length_cm:          float,
    scans_per_group:         int,           # scanning only
    wavelength_tolerance_nm: float,
    skip_groups:             int            = 0,     # scanning: skip first N groups (irr. delay)
    initial_spectrum_path:   Optional[Path] = None,  # kinetic: required (Beer-Lambert ref)
    fit_time_start_s:        Optional[float] = None, # kinetic: first time point to include (s)
    fit_time_end_s:          Optional[float] = None, # kinetic: last time point to include (s)
    N_LED_mol_s:             Optional[float] = None,
    N_LED_std_mol_s:         Optional[float] = None,
) -> "LEDActinometerResult":
    """
    Process one actinometry CSV measured under LED irradiation.

    Data types
    ----------
    scanning : Cary 60 multi-scan UV-Vis CSV (full spectrum at each time point).
               Beer-Lambert reference = mid scan of the first scan group.
               Time axis  = group_index × irradiation_time_s.

    kinetic  : Cary 60 kinetics CSV (fixed wavelength, time series).
               Beer-Lambert reference = first scan of the initial_spectrum_path
               file (required).  The channel closest to λ_eff is used.
               Optional offset correction aligns the kinetic trace to the
               initial spectrum at λ_eff.
               Time axis  = directly from the CSV, filtered to [fit_time_start_s,
               fit_time_end_s] and reset to zero at the first selected point.

    Integration modes (both data types)
    ------------------------------------
    scalar   : ε(λ_eff) and QY(λ_eff) from the actinometer formula; equivalent
               to standard chemical actinometry at the effective wavelength.
    spectral : flux-weighted averages across the full LED emission band:
                 ε_eff  = ∫ f(λ)·ε(λ)  dλ
                 QY_eff = ∫ f(λ)·QY(λ) dλ
               where f(λ) = N(λ)/N_total (normalised LED spectral shape) and
               ε(λ) = ε_ref × A(λ)/A(λ_ref) (Beer-Lambert scaling).

    In both modes the standard rate-function formula is applied with the
    effective ε and QY, monitoring absorbance at λ_eff.
    """
    act = ACTINOMETERS[actinometer_choice]
    wl_min, wl_max = act["wavelength_range_nm"]

    # ── Step 1: Beer-Lambert reference spectrum (ref_wl, ref_ab) ──────────
    if data_type == "scanning":
        scans   = load_spectra_csv(filepath)
        n_scans = len(scans)
        print(f"  {n_scans} scans loaded from {filepath.name}")
        if n_scans < scans_per_group:
            raise ValueError(
                f"Only {n_scans} scans; need ≥ {scans_per_group} per group.")
        n_groups   = n_scans // scans_per_group
        ref_wl, ref_ab = scans[1]          # mid scan of first group
        idx = np.argsort(ref_wl)
        ref_wl, ref_ab = ref_wl[idx], ref_ab[idx]

    elif data_type == "kinetic":
        if initial_spectrum_path is None:
            raise ValueError(
                "Kinetic mode requires an initial spectrum file for Beer-Lambert "
                "scaling (ε_eff / QY_eff calculation).")
        ref_wl, ref_ab = _load_initial_spectrum(initial_spectrum_path)
        print(f"  Kinetic mode: Beer-Lambert reference from "
              f"{initial_spectrum_path.name}")

    else:
        raise ValueError(f"Unknown data_type: '{data_type}'")

    # ── Step 2: A_ref from reference spectrum ─────────────────────────────
    lam_ref = act["epsilon_ref_nm"]
    A_ref   = extract_absorbance(ref_wl, ref_ab, lam_ref, wavelength_tolerance_nm)
    if np.isnan(A_ref) or A_ref <= 0:
        raise ValueError(
            f"Cannot extract A({lam_ref:.0f} nm) = {A_ref} from the Beer-Lambert "
            f"reference spectrum. Check that the spectrum covers {lam_ref:.0f} nm "
            f"and the tolerance ({wavelength_tolerance_nm} nm) is appropriate.")

    # ── Step 3: Effective ε and QY ─────────────────────────────────────────
    N_total = float(np.trapezoid(led_N_arr, led_wl_arr))
    f_arr   = led_N_arr / N_total   # normalised spectral shape [nm⁻¹]

    if integration_mode == "scalar":
        A_lam = extract_absorbance(ref_wl, ref_ab, lam_eff, wavelength_tolerance_nm)
        if np.isnan(A_lam):
            raise ValueError(
                f"Cannot extract A at λ_eff = {lam_eff:.1f} nm from the "
                f"reference spectrum. Increase λ tolerance or check the spectrum.")
        epsilon_eff = act["epsilon_ref_M_cm"] * A_lam / A_ref
        QY_eff      = act["QY_func"](lam_eff)
        if not (wl_min <= lam_eff <= wl_max):
            print(f"  WARNING: λ_eff = {lam_eff:.1f} nm is outside the valid "
                  f"range for {act['name']} ({wl_min}–{wl_max} nm).")
        print(f"  Mode      : scalar  (λ_eff = {lam_eff:.1f} nm)")
        print(f"  A({lam_ref:.0f})   : {A_ref:.4f}  A(λ_eff) : {A_lam:.4f}  "
              f"ratio A(λ_eff)/A({lam_ref:.0f}) = {A_lam/A_ref:.3f}")
        print(f"  ε_ref({lam_ref:.0f}): {act['epsilon_ref_M_cm']:.4e} L mol⁻¹ cm⁻¹")
        print(f"  ε(λ_eff)  : {epsilon_eff:.4e} L mol⁻¹ cm⁻¹  "
              f"[= ε_ref × {A_lam/A_ref:.3f}]")
        print(f"  QY(λ_eff) : {QY_eff:.4f}")

    elif integration_mode == "spectral":
        A_interp    = np.interp(led_wl_arr, ref_wl, ref_ab,
                                left=np.nan, right=np.nan)
        epsilon_arr = act["epsilon_ref_M_cm"] * A_interp / A_ref
        QY_arr      = np.where(
            (led_wl_arr >= wl_min) & (led_wl_arr <= wl_max),
            np.array([act["QY_func"](lam) for lam in led_wl_arr]),
            np.nan,
        )
        valid = np.isfinite(epsilon_arr) & np.isfinite(QY_arr) & (epsilon_arr > 0)
        if valid.sum() < 5:
            raise ValueError(
                f"Only {valid.sum()} valid spectral points for flux-weighted "
                f"integration (need ≥ 5). Check that the LED emission overlaps "
                f"the actinometer's valid range ({wl_min}–{wl_max} nm) and "
                f"that the reference spectrum covers that region.")
        wl_v, f_v, eps_v, QY_v = (
            led_wl_arr[valid], f_arr[valid],
            epsilon_arr[valid], QY_arr[valid],
        )
        flux_fraction = float(np.trapezoid(f_v, wl_v))
        f_v_norm      = f_v / np.trapezoid(f_v, wl_v)
        epsilon_eff   = float(np.trapezoid(f_v_norm * eps_v, wl_v))
        QY_eff        = float(np.trapezoid(f_v_norm * QY_v, wl_v))
        print(f"  Mode      : spectral  ({valid.sum()} pts, "
              f"{wl_v[0]:.0f}–{wl_v[-1]:.0f} nm, "
              f"{flux_fraction*100:.1f}% of LED flux)")
        print(f"  ε_eff     : {epsilon_eff:.4e} L mol⁻¹ cm⁻¹")
        print(f"  QY_eff    : {QY_eff:.4f}")
    else:
        raise ValueError(f"Unknown integration_mode: '{integration_mode}'")

    # ── Step 4: Prefactor ─────────────────────────────────────────────────
    V_m3       = volume_mL * 1e-6
    epsilon_SI = epsilon_eff * 0.1        # L mol⁻¹ cm⁻¹ → m² mol⁻¹
    l_m        = path_length_cm * 1e-2
    prefactor  = -V_m3 / (epsilon_SI * QY_eff * l_m)

    # ── Step 5: Build (time_axis, absorbances) ────────────────────────────
    if data_type == "scanning":
        if skip_groups >= n_groups:
            raise ValueError(
                f"skip_groups = {skip_groups} ≥ n_groups = {n_groups}. "
                f"Nothing left to fit.")
        if skip_groups > 0:
            print(f"  Skipping first {skip_groups} group(s) (irradiation delay).")
        absorbances = []
        for g in range(skip_groups, n_groups):
            group_abs = [
                extract_absorbance(
                    scans[g * scans_per_group + s][0],
                    scans[g * scans_per_group + s][1],
                    lam_eff, wavelength_tolerance_nm,
                )
                for s in range(scans_per_group)
            ]
            valid_abs = [v for v in group_abs if not np.isnan(v)]
            absorbances.append(np.mean(valid_abs) if valid_abs else np.nan)
        time_axis = np.arange(len(absorbances)) * irradiation_time_s

    else:  # kinetic
        channels = load_kinetic_csv(filepath)
        if not channels:
            raise ValueError(f"No valid channels found in {filepath.name}.")

        # Find the channel whose label wavelength is closest to λ_eff
        best_label: Optional[str] = None
        best_dist = float("inf")
        best_wl_found = lam_eff
        for label in channels:
            wl_parsed = _parse_wl_from_label(label)
            if wl_parsed is not None and abs(wl_parsed - lam_eff) < best_dist:
                best_dist      = abs(wl_parsed - lam_eff)
                best_label     = label
                best_wl_found  = wl_parsed
        if best_label is None:
            raise ValueError(
                f"Cannot parse wavelengths from channel labels in {filepath.name}. "
                f"Labels found: {list(channels.keys())}. "
                f"Expected format: e.g. '25C_672nm'.")
        print(f"  Channel   : '{best_label}'  "
              f"(λ = {best_wl_found:.0f} nm, "
              f"Δλ = {best_dist:.1f} nm from λ_eff = {lam_eff:.1f} nm)")

        t_raw, a_raw = channels[best_label]

        # Time window filter
        mask = np.ones(len(t_raw), dtype=bool)
        if fit_time_start_s is not None:
            mask &= t_raw >= fit_time_start_s
        if fit_time_end_s is not None:
            mask &= t_raw <= fit_time_end_s
        t_win = t_raw[mask]
        a_win = a_raw[mask]
        if len(t_win) < 2:
            raise ValueError(
                f"Fewer than 2 data points in the selected time window "
                f"[{fit_time_start_s}, {fit_time_end_s}] s.")

        # Offset correction: align kinetic start to initial spectrum at λ_eff
        A_init_lam = extract_absorbance(ref_wl, ref_ab, lam_eff, wavelength_tolerance_nm)
        if not np.isnan(A_init_lam):
            n_bl         = min(5, len(a_win))
            A_kin_start  = float(a_win[:n_bl].mean())
            offset       = A_init_lam - A_kin_start
            if abs(offset) > 1e-4:
                print(f"  Offset    : {offset:+.4f} AU  "
                      f"(A_init = {A_init_lam:.4f}, "
                      f"A_kin_t0 = {A_kin_start:.4f})")
            a_win = a_win + offset
        else:
            print(f"  WARNING: Cannot extract A at λ_eff from initial spectrum — "
                  f"no offset correction applied.")

        # Reset time so first selected point = t=0
        time_axis   = t_win - t_win[0]
        absorbances = list(a_win)

    # ── Step 6: Rate function and linear fit ──────────────────────────────
    A_0 = absorbances[0]
    if np.isnan(A_0) or (10 ** A_0 - 1) <= 0:
        raise ValueError(
            f"A_0 = {A_0:.4f} — cannot compute log10(10^A − 1). "
            f"Check that the actinometer absorbance at λ_eff = {lam_eff:.1f} nm "
            f"is positive and not too close to zero.")

    log_ref = np.log10(10 ** A_0 - 1)
    t_valid, y_vals, abs_valid = [], [], []

    for i, A_i in enumerate(absorbances):
        if np.isnan(A_i):
            continue
        val = 10 ** A_i - 1
        if val <= 0:
            print(f"  t = {time_axis[i]:.0f} s: 10^A − 1 ≤ 0 — skipping.")
            continue
        y_vals.append(prefactor * (np.log10(val) - log_ref))
        t_valid.append(time_axis[i])
        abs_valid.append(A_i)

    t_valid   = np.array(t_valid)
    y_vals    = np.array(y_vals)
    abs_valid = np.array(abs_valid)

    if len(t_valid) < 2:
        raise ValueError("Not enough valid points for a linear fit (< 2).")

    popt, pcov       = curve_fit(_linear, t_valid, y_vals)
    slope, intercept = popt
    slope_std, _     = np.sqrt(np.diag(pcov))

    y_pred = _linear(t_valid, slope, intercept)
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"  N_chem    : {slope:.4e} ± {slope_std:.4e} mol s⁻¹  "
          f"(R² = {r2:.4f})")
    if N_LED_mol_s is not None:
        dev = (slope - N_LED_mol_s) / N_LED_mol_s * 100.0
        print(f"  vs N_LED  : {N_LED_mol_s:.4e} mol s⁻¹  "
              f"(deviation = {dev:+.2f} %)")

    return LEDActinometerResult(
        file=filepath.name,
        actinometer_name=act["name"],
        integration_mode=integration_mode,
        lam_eff=lam_eff,
        epsilon_eff_M_cm=epsilon_eff,
        QY_eff=QY_eff,
        volume_mL=volume_mL,
        path_length_cm=path_length_cm,
        photon_flux_mol_s=slope,
        photon_flux_std_mol_s=slope_std,
        r2=r2,
        intercept=intercept,
        t_valid=t_valid,
        y_vals=y_vals,
        y_pred=y_pred,
        abs_valid=abs_valid,
        N_LED_mol_s=N_LED_mol_s,
        N_LED_std_mol_s=N_LED_std_mol_s,
        success=True,
    )


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

    # ── Left: rate function ────────────────────────────────────────────────
    ax1.plot(result.t_valid, result.y_vals,
             "o", color="black", markerfacecolor="none",
             markeredgewidth=1.2, label="Data")

    t_fit = np.linspace(0, result.t_valid.max() * 1.05, 200)
    ax1.plot(t_fit, _linear(t_fit, result.photon_flux_mol_s, result.intercept),
             "--", color="red", linewidth=2, label="Linear fit")

    mode_lbl = "scalar" if result.integration_mode == "scalar" else "spectral"
    eps_lbl  = "ε(λ_eff)" if result.integration_mode == "scalar" else "ε_eff"
    QY_lbl   = "QY(λ_eff)" if result.integration_mode == "scalar" else "QY_eff"
    annotation = (
        f"Mode: {mode_lbl}  (λ_eff = {result.lam_eff:.1f} nm)\n"
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
    ax1.set_ylabel("Rate function (mol)")
    ax1.set_title(f"{result.actinometer_name}  |  {result.file}")
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
