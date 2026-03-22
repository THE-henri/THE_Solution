"""
Core functions for the Actinometer tab.

Chemical actinometry  – extracted from workflows/actinometer_analysis.py
LED characterisation  – extracted from workflows/quantum_yield.py (LED block)
"""

from __future__ import annotations

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
        "epsilon_562_M_cm":    1.0e4,
        "QY_func":             lambda lam: 10 ** (-0.796 + 133 / lam),
    },
    2: {
        "name":                "Actinometer 2",
        "wavelength_range_nm": (480, 620),
        "epsilon_562_M_cm":    1.09e4,
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
    mid_scan = scans[1]
    A_562 = extract_absorbance(mid_scan[0], mid_scan[1], 562,
                               wavelength_tolerance_nm)
    A_irr = extract_absorbance(mid_scan[0], mid_scan[1],
                               irradiation_wavelength_nm, wavelength_tolerance_nm)

    if np.isnan(A_562) or A_562 <= 0 or np.isnan(A_irr):
        raise ValueError(
            f"Cannot compute ε: A_562 = {A_562:.4f}, "
            f"A_irr = {A_irr:.4f}"
        )

    epsilon    = act["epsilon_562_M_cm"] * A_irr / A_562
    V_m3       = volume_mL * 1e-6
    epsilon_SI = epsilon * 0.1           # L mol⁻¹ cm⁻¹ → m² mol⁻¹
    l_m        = path_length_cm * 1e-2   # cm → m
    prefactor  = -V_m3 / (epsilon_SI * QY * l_m)

    print(f"  ε_irr = {epsilon:.4e} L mol⁻¹ cm⁻¹  "
          f"(ε_562 = {act['epsilon_562_M_cm']:.3e}, "
          f"A_irr/A_562 = {A_irr:.4f}/{A_562:.4f})")

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
    P_after: Optional[float] = None

    if power_after_path is not None and power_after_path.exists():
        pwr_df_a = pd.read_csv(power_after_path, comment="#")
        P_after  = float(pwr_df_a["power_mW"].mean())

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
        raw_a_n = result.em_int_raw_a / result.em_int_raw_b.max()
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
