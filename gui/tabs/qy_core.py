"""
Core functions for the Quantum Yield tab.

Extracted from workflows/quantum_yield.py.  All interactive prompts removed;
logic replaced by structured inputs / outputs for GUI use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters


# ── Physical constants ────────────────────────────────────────────────────────

h_PLANCK = 6.626070e-34   # J s
C_LIGHT  = 299792458.0    # m s⁻¹
NA       = 6.022141e+23   # mol⁻¹
R_GAS    = 8.314462       # J mol⁻¹ K⁻¹
kB       = 1.380649e-23   # J K⁻¹
h_ey     = 6.626070e-34   # J s (Eyring)


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_spectra_csv(filepath: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load a Cary 60 multi-scan CSV. Returns list of (wl, ab)."""
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
    Load multi-wavelength kinetic CSV.
    Row 0: channel labels (every 2 columns, e.g. "25C_672nm")
    Row 1: "Time (sec)", "Abs" per channel
    Returns dict {label: (time_array, abs_array)}.
    """
    MIN_VALID = 5
    raw = pd.read_csv(filepath, header=None)
    label_row = raw.iloc[0]
    data = raw.iloc[2:].reset_index(drop=True)
    channels = {}
    for i in range(0, data.shape[1] - 1, 2):
        label = str(label_row.iloc[i]).strip()
        t_col = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        a_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid = t_col.notna() & a_col.notna()
        if valid.sum() < MIN_VALID:
            continue
        channels[label] = (t_col[valid].values, a_col[valid].values)
    return channels


def load_epsilon_from_csv(csv_path: Path,
                          target_wavelengths: list[float],
                          column: Optional[str] = None) -> np.ndarray:
    """
    Load ε from an EC or spectra-calculation CSV.
    Accepts "Wavelength (nm)" or "Wavelength_nm" column.
    Interpolates to each target wavelength.
    """
    df = pd.read_csv(csv_path, comment="#")
    wl_col = None
    for _try in ("Wavelength (nm)", "Wavelength_nm"):
        if _try in df.columns:
            wl_col = _try
            break
    if wl_col is None:
        raise ValueError(
            f"No wavelength column found in {csv_path}. "
            f"Available: {list(df.columns)}"
        )
    col = column or "Mean"
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {csv_path}. "
            f"Available: {list(df.columns)}"
        )
    wl_arr  = df[wl_col].values.astype(float)
    eps_arr = df[col].values.astype(float)
    order   = np.argsort(wl_arr)
    return np.interp(np.array(target_wavelengths, dtype=float),
                     wl_arr[order], eps_arr[order])


# ── Irradiation start detection ───────────────────────────────────────────────

def detect_irr_start(
    time_s:     np.ndarray,
    abs_data:   np.ndarray,
    n_plateau:  int   = 20,
    threshold:  float = 5.0,
    min_consec: int   = 3,
) -> tuple:
    """
    Detect the irradiation start in a kinetic absorbance trace.
    Returns (t_fit_start, t_irr_onset, j_ref, plat_mean, plat_std, idx_fit).
    """
    n_pts = len(time_s)
    j_ref = int(np.argmax(np.ptp(abs_data, axis=0)))
    trace = abs_data[:, j_ref]

    n_plat    = min(n_plateau, n_pts // 2)
    plat      = trace[:n_plat]
    plat_mean = float(plat.mean())
    plat_std  = float(plat.std(ddof=1)) if n_plat > 1 else 0.0
    min_std   = max(abs(plat_mean) * 0.0005, 1e-5)
    plat_std  = max(plat_std, min_std)
    thresh_abs = threshold * plat_std

    consec    = 0
    idx_onset = None
    for i in range(n_plat, n_pts):
        if abs(trace[i] - plat_mean) > thresh_abs:
            consec += 1
            if consec >= min_consec and idx_onset is None:
                idx_onset = i - min_consec + 1
        else:
            consec = 0

    if idx_onset is None:
        return None, None, j_ref, plat_mean, plat_std, None

    idx_fit = max(idx_onset - 1, 0)
    return (float(time_s[idx_fit]), float(time_s[idx_onset]),
            j_ref, plat_mean, plat_std, idx_fit)


# ── Unit conversion ───────────────────────────────────────────────────────────

def uW_to_mol_s(power_uW: float, wavelength_nm: float) -> float:
    """Convert optical power in µW at a given wavelength to photon flux in mol s⁻¹."""
    lambda_m = wavelength_nm * 1e-9
    return (power_uW * 1e-6) / (h_PLANCK * C_LIGHT / lambda_m) / NA


# ── ODE functions ─────────────────────────────────────────────────────────────

def rate_equations(y, _t,
                   QY_AB, QY_BA, eps_A_irr, eps_B_irr,
                   N_mol_s_val, V_L, k_th_val, l_cm):
    """General two-species photoisomerisation ODE (monochromator / scalar LED)."""
    A, B  = y
    A_tot = (A * eps_A_irr + B * eps_B_irr) * l_cm
    if A_tot < 1e-10:
        factor = np.log(10.0)
    else:
        factor = (1.0 - 10.0 ** (-A_tot)) / A_tot
    rate = (N_mol_s_val / V_L) * l_cm * factor * (
        QY_BA * B * eps_B_irr - QY_AB * A * eps_A_irr)
    dAdt = rate + k_th_val * B
    return [dAdt, -dAdt]


def rate_equations_led(y, _t,
                       QY_AB, QY_BA, led_wl_arr, led_N_arr,
                       eps_A_led_arr, eps_B_led_arr, V_L, k_th_val, l_cm):
    """LED full-integration rate equation."""
    A, B = y
    A_tot_arr  = (A * eps_A_led_arr + B * eps_B_led_arr) * l_cm
    factor_arr = np.where(A_tot_arr < 1e-10,
                          np.log(10.0),
                          (1.0 - 10.0 ** (-A_tot_arr)) / A_tot_arr)
    rate_arr   = (led_N_arr / V_L) * l_cm * factor_arr * (
        QY_BA * B * eps_B_led_arr - QY_AB * A * eps_A_led_arr)
    dAdt = float(np.trapezoid(rate_arr, led_wl_arr)) + k_th_val * B
    return [dAdt, -dAdt]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_absorbance(params, time_s, conc_A_0, conc_B_0,
                        eps_A_irr, eps_B_irr,
                        eps_A_mon, eps_B_mon,
                        N_mol_s_val, V_L, k_th_val, l_cm):
    """Integrate the ODE and return simulated absorbance (n_time, n_mon_wl)."""
    QY_AB    = params["QY_AB"].value
    QY_BA    = params["QY_BA"].value
    ode_args = (QY_AB, QY_BA, eps_A_irr, eps_B_irr, N_mol_s_val, V_L, k_th_val, l_cm)
    if time_s.ndim == 2:
        all_t = np.unique(time_s.ravel())
        if all_t[0] > 1e-12:
            all_t = np.concatenate([[0.0], all_t])
        sol_all = odeint(rate_equations, [conc_A_0, conc_B_0], all_t,
                         args=ode_args, mxstep=5000)
        n_time, n_ch = time_s.shape
        abs_sim = np.zeros((n_time, n_ch))
        for j in range(n_ch):
            cA_j = np.interp(time_s[:, j], all_t, sol_all[:, 0])
            cB_j = np.interp(time_s[:, j], all_t, sol_all[:, 1])
            abs_sim[:, j] = (cA_j * eps_A_mon[j] + cB_j * eps_B_mon[j]) * l_cm
        return abs_sim
    else:
        conc = odeint(rate_equations, [conc_A_0, conc_B_0], time_s,
                      args=ode_args, mxstep=5000)
        return (np.outer(conc[:, 0], eps_A_mon)
                + np.outer(conc[:, 1], eps_B_mon)) * l_cm


def simulate_absorbance_led(params, time_s, conc_A_0, conc_B_0,
                            led_wl_arr, led_N_arr,
                            eps_A_led_arr, eps_B_led_arr,
                            eps_A_mon, eps_B_mon,
                            V_L, k_th_val, l_cm):
    """LED full-integration ODE simulation."""
    QY_AB    = params["QY_AB"].value
    QY_BA    = params["QY_BA"].value
    ode_args = (QY_AB, QY_BA, led_wl_arr, led_N_arr,
                eps_A_led_arr, eps_B_led_arr, V_L, k_th_val, l_cm)
    if time_s.ndim == 2:
        all_t = np.unique(time_s.ravel())
        if all_t[0] > 1e-12:
            all_t = np.concatenate([[0.0], all_t])
        sol_all = odeint(rate_equations_led, [conc_A_0, conc_B_0], all_t,
                         args=ode_args, mxstep=5000)
        n_time, n_ch = time_s.shape
        abs_sim = np.zeros((n_time, n_ch))
        for j in range(n_ch):
            cA_j = np.interp(time_s[:, j], all_t, sol_all[:, 0])
            cB_j = np.interp(time_s[:, j], all_t, sol_all[:, 1])
            abs_sim[:, j] = (cA_j * eps_A_mon[j] + cB_j * eps_B_mon[j]) * l_cm
        return abs_sim
    else:
        conc = odeint(rate_equations_led, [conc_A_0, conc_B_0], time_s,
                      args=ode_args, mxstep=5000)
        return (np.outer(conc[:, 0], eps_A_mon)
                + np.outer(conc[:, 1], eps_B_mon)) * l_cm


# ── Fitting ───────────────────────────────────────────────────────────────────

def _residuals(params, time_s, abs_exp,
               conc_A_0, conc_B_0,
               eps_A_irr, eps_B_irr,
               eps_A_mon, eps_B_mon,
               N_mol_s_val, V_L, k_th_val, l_cm):
    try:
        sim = simulate_absorbance(params, time_s, conc_A_0, conc_B_0,
                                  eps_A_irr, eps_B_irr, eps_A_mon, eps_B_mon,
                                  N_mol_s_val, V_L, k_th_val, l_cm)
        return (sim - abs_exp).flatten()
    except Exception:
        return np.full(abs_exp.size, 1e6)


def _residuals_led(params, time_s, abs_exp,
                   conc_A_0, conc_B_0,
                   led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
                   eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm):
    try:
        sim = simulate_absorbance_led(params, time_s, conc_A_0, conc_B_0,
                                      led_wl_arr, led_N_arr,
                                      eps_A_led_arr, eps_B_led_arr,
                                      eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm)
        return (sim - abs_exp).flatten()
    except Exception:
        return np.full(abs_exp.size, 1e6)


def run_fit(p_init, time_s, abs_exp,
            conc_A_0, conc_B_0,
            eps_A_irr, eps_B_irr,
            eps_A_mon, eps_B_mon,
            N_val, V_L, k_th_val, l_cm):
    return minimize(_residuals, p_init,
                    args=(time_s, abs_exp, conc_A_0, conc_B_0,
                          eps_A_irr, eps_B_irr, eps_A_mon, eps_B_mon,
                          N_val, V_L, k_th_val, l_cm),
                    method="leastsq")


def run_fit_led(p_init, time_s, abs_exp,
                conc_A_0, conc_B_0,
                led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
                eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm):
    return minimize(_residuals_led, p_init,
                    args=(time_s, abs_exp, conc_A_0, conc_B_0,
                          led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
                          eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm),
                    method="leastsq")


# ── Initial slopes estimate ───────────────────────────────────────────────────

def initial_slopes_QY(time_s_ode, abs_data_fit, n_pts,
                      eps_A_irr_v, eps_A_mon_arr, eps_B_mon_arr,
                      N_v, V_L_v, l_v, conc_A_0_v):
    """
    Estimate QY_AB from the initial slope at each monitoring wavelength.
    Returns (QY_estimates, slopes).
    """
    A0_irr        = conc_A_0_v * eps_A_irr_v * l_v
    absorbed_frac = ((1.0 - 10.0 ** (-A0_irr))
                     if A0_irr > 1e-10 else A0_irr * np.log(10.0))
    n    = min(n_pts, len(time_s_ode))
    t_pts = time_s_ode[:n]
    QY_estimates, slopes = [], []
    for j in range(len(eps_A_mon_arr)):
        slope = np.polyfit(t_pts, abs_data_fit[:n, j], 1)[0]
        slopes.append(slope)
        eps_diff = float(eps_A_mon_arr[j]) - float(eps_B_mon_arr[j])
        if abs(eps_diff) < 1.0 or absorbed_frac < 1e-12:
            QY_estimates.append(np.nan)
        else:
            QY_estimates.append(
                -slope * V_L_v / (eps_diff * l_v * N_v * absorbed_frac))
    return QY_estimates, slopes


# ── PSS algebraic ─────────────────────────────────────────────────────────────

def pss_algebraic(k_th_val, conc_B_pss, volume_L, N_val, A_abs_pss):
    denom = N_val * (1.0 - 10.0 ** (-A_abs_pss))
    if denom <= 0:
        raise ValueError(
            f"PSS denominator ≤ 0: N={N_val:.4e}, A_PSS={A_abs_pss:.4f}")
    return k_th_val * conc_B_pss * volume_L / denom


# ── Parameter container ───────────────────────────────────────────────────────

@dataclass
class QYParams:
    # ── Experiment ──────────────────────────────────────────────────────────
    case:              str   = "A_only"   # "A_only" | "AB_both" | "A_thermal_PSS"
    data_type:         str   = "kinetic"  # "kinetic" | "scanning"
    compound_name:     str   = ""
    temperature_C:     float = 25.0
    solvent:           str   = ""

    # ── Photon flux ──────────────────────────────────────────────────────────
    # "manual_mol_s" | "manual_uW" | "actinometry" | "led_spectrum"
    photon_flux_source:       str            = "manual_mol_s"
    photon_flux_mol_s:        float          = 1.0e-9
    photon_flux_uW:           float          = 0.0
    photon_flux_std_mol_s:    float          = 0.0
    irradiation_wavelength_nm: float         = 530.0
    actinometry_csv:          Optional[Path] = None
    actinometry_filter_nm:    Optional[float] = None  # None = last row
    led_spectrum_csv:         Optional[Path] = None   # from Actinometer tab
    led_integration_mode:     str            = "scalar"  # "scalar" | "full"

    # ── Extinction coefficients ───────────────────────────────────────────────
    # "manual" | "csv"
    epsilon_source_A:  str            = "manual"
    epsilon_source_B:  str            = "manual"
    epsilon_A_irr:     float          = 10000.0
    epsilon_B_irr:     float          = 0.0
    epsilon_A_csv:     Optional[Path] = None
    epsilon_B_csv:     Optional[Path] = None
    epsilon_A_col:     str            = "Mean"
    epsilon_B_col:     str            = "Mean"

    # ── Thermal back-reaction ────────────────────────────────────────────────
    # "none" | "manual" | "half_life_master" | "eyring" | "arrhenius"
    k_th_source:       str            = "none"
    k_th_manual:       float          = 0.0
    k_th_manual_std:   float          = 0.0
    k_th_temperature_C: float         = 25.0
    k_th_csv:          Optional[Path] = None

    # ── Optical ──────────────────────────────────────────────────────────────
    path_length_cm:   float = 1.0
    volume_mL:        float = 2.0

    # ── Data / wavelengths ───────────────────────────────────────────────────
    monitoring_wavelengths: Optional[list] = None   # None = auto from kinetic headers
    wavelength_tolerance_nm: float         = 2.0

    # ── Scanning-specific ────────────────────────────────────────────────────
    delta_t_s:        float = 12.0
    scans_per_group:  int   = 1
    first_cycle_off:  bool  = False

    # ── Baseline correction ──────────────────────────────────────────────────
    # "none" | "first_point" | "plateau" | "file"
    baseline_correction:         str            = "first_point"
    baseline_file:               Optional[Path] = None
    offset_plateau_duration_s:   Optional[float] = 20.0
    baseline_plateau_start_s:    Optional[float] = None
    baseline_plateau_end_s:      Optional[float] = None

    # ── Fit window ───────────────────────────────────────────────────────────
    fit_time_start_s:       Optional[float] = None
    fit_time_end_s:         Optional[float] = None
    auto_detect_irr_start:  bool  = True
    auto_detect_n_plateau:  int   = 20
    auto_detect_threshold:  float = 5.0
    auto_detect_min_consec: int   = 3

    # ── Initial conditions ───────────────────────────────────────────────────
    initial_conc_source:   str            = "absorbance"
    initial_conc_A_manual: Optional[float] = None
    initial_conc_B_manual: float           = 0.0

    # ── PSS (A_thermal_PSS) ──────────────────────────────────────────────────
    # "manual_fraction" | "manual_absorbance"
    pss_source:               str            = "manual_fraction"
    pss_fraction_B_manual:    Optional[float] = None
    pss_A_abs_pss_manual:     Optional[float] = None

    # ── Fitting ──────────────────────────────────────────────────────────────
    QY_AB_init:           float = 0.1
    QY_BA_init:           float = 0.05
    QY_bounds_lo:         float = 1e-6
    QY_bounds_hi:         float = 1.0
    QY_unconstrained:     bool  = False
    QY_AB_reference:      Optional[float] = None
    n_initial_slopes_pts: int   = 8


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class QYFileResult:
    file_name:       str
    compound:        str
    case:            str
    mon_wls:         list[float]
    N_mol_s:         float
    N_std_mol_s:     float
    k_th:            float
    eps_A_irr:       float
    eps_B_irr:       float
    conc_A_0:        float
    V_L:             float
    path_length_cm:  float
    temperature_C:   float
    solvent:         str
    # per-wavelength fit results
    QY_AB_per_wl:       list[float]
    QY_BA_per_wl:       list[float]
    stderr_AB_per_wl:   list[float]
    sigma_I0_AB_per_wl: list[float]
    sigma_total_per_wl: list[float]
    r2_per_wl:          list[float]
    # aggregated
    QY_AB:           float
    QY_BA:           float
    QY_AB_sigma_fit: float
    QY_AB_sigma_total: float
    r2:              float
    method:          str
    # initial slopes
    QY_slopes:       list[float]
    # arrays for plotting
    time_s:          np.ndarray
    abs_data:        np.ndarray     # (n_time, n_mon)
    abs_fit_per_wl:  list[np.ndarray]   # per-wl ODE curve, same length as display window
    abs_fit_lo_per_wl: list[np.ndarray]
    abs_fit_hi_per_wl: list[np.ndarray]
    residuals_2d:    np.ndarray     # (n_fit, n_mon)
    time_s_fit:      np.ndarray
    fit_mask:        np.ndarray
    t_display_per_wl: list[np.ndarray]  # absolute times for display


# ── Photon flux loading ───────────────────────────────────────────────────────

def load_photon_flux(params: QYParams):
    """
    Resolve photon flux from the chosen source.

    Returns
    -------
    N_mol_s     : float
    N_std_mol_s : float
    led_wl_arr  : ndarray or None   — LED wavelengths (full integration)
    led_N_arr   : ndarray or None   — spectral flux density mol/s/nm
    """
    led_wl_arr = None
    led_N_arr  = None

    if params.photon_flux_source == "manual_mol_s":
        N_mol_s     = params.photon_flux_mol_s
        N_std_mol_s = params.photon_flux_std_mol_s
        print(f"  Photon flux (manual): {N_mol_s:.4e} ± {N_std_mol_s:.4e} mol s⁻¹")

    elif params.photon_flux_source == "manual_uW":
        N_mol_s     = uW_to_mol_s(params.photon_flux_uW,
                                   params.irradiation_wavelength_nm)
        N_std_mol_s = params.photon_flux_std_mol_s
        print(f"  Photon flux from {params.photon_flux_uW:.4g} µW "
              f"@ {params.irradiation_wavelength_nm} nm: "
              f"N = {N_mol_s:.4e} mol s⁻¹")

    elif params.photon_flux_source == "actinometry":
        if params.actinometry_csv is None or not params.actinometry_csv.exists():
            raise FileNotFoundError(
                "Actinometry CSV not set or not found. "
                "Select photon_flux_master.csv in Stage 1.")
        df = pd.read_csv(params.actinometry_csv)
        if params.actinometry_filter_nm is not None:
            col = "Irradiation_nm"
            if col in df.columns:
                df = df[df[col] == params.actinometry_filter_nm]
        if df.empty:
            raise ValueError(
                f"No rows in {params.actinometry_csv.name} after applying "
                f"irradiation wavelength filter {params.actinometry_filter_nm}.")
        row         = df.iloc[-1]
        N_mol_s     = float(row["Photon_flux_mol_s"])
        N_std_mol_s = (float(row["Photon_flux_std_mol_s"])
                       if "Photon_flux_std_mol_s" in row.index else 0.0)
        print(f"  Photon flux from actinometry ({params.actinometry_csv.name}): "
              f"N = {N_mol_s:.4e} ± {N_std_mol_s:.4e} mol s⁻¹")

    elif params.photon_flux_source == "led_spectrum":
        if params.led_spectrum_csv is None or not params.led_spectrum_csv.exists():
            raise FileNotFoundError(
                "LED spectrum CSV not set or not found. "
                "Select the spectrum saved by the Actinometer / LED Characterisation panel.")
        df = pd.read_csv(params.led_spectrum_csv, comment="#")
        led_wl_arr = df["wavelength_nm"].values.astype(float)
        led_N_arr  = df["N_mol_s_per_nm"].values.astype(float)
        order = np.argsort(led_wl_arr)
        led_wl_arr = led_wl_arr[order]
        led_N_arr  = led_N_arr[order]
        N_mol_s = float(np.trapezoid(led_N_arr, led_wl_arr))
        N_std_mol_s = params.photon_flux_std_mol_s
        # If N_total is stored in row 0 of the CSV, use that
        if "N_total_mol_s" in df.columns:
            N_mol_s = float(df["N_total_mol_s"].iloc[0])
        if "N_std_mol_s" in df.columns and N_std_mol_s == 0.0:
            N_std_mol_s = float(df["N_std_mol_s"].iloc[0])
        print(f"  Photon flux from LED spectrum ({params.led_spectrum_csv.name}): "
              f"N = {N_mol_s:.4e} mol s⁻¹  "
              f"[{len(led_wl_arr)} points, "
              f"{led_wl_arr[0]:.0f}–{led_wl_arr[-1]:.0f} nm]")
    else:
        raise ValueError(
            f"Unknown photon_flux_source: '{params.photon_flux_source}'")

    print(f"  N std = {N_std_mol_s:.4e} mol s⁻¹")
    return N_mol_s, N_std_mol_s, led_wl_arr, led_N_arr


# ── Extinction coefficient loading ────────────────────────────────────────────

def load_epsilon_at_wavelength(
    source:      str,
    csv_path:    Optional[Path],
    col:         str,
    wavelength:  float,
    manual_val:  float,
    label:       str,
) -> float:
    """Return ε at a single wavelength from the configured source."""
    if source == "manual":
        return manual_val
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            f"ε_{label} CSV not set or not found. "
            f"Browse to the CSV file in Stage 3.")
    return float(load_epsilon_from_csv(csv_path, [wavelength], col)[0])


def load_epsilon_at_wavelengths(
    source:      str,
    csv_path:    Optional[Path],
    col:         str,
    wavelengths: list[float],
    manual_val:  float,
) -> np.ndarray:
    """Return ε at multiple wavelengths."""
    if source == "manual":
        return np.full(len(wavelengths), manual_val)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            f"ε CSV not set or not found.")
    return load_epsilon_from_csv(csv_path, wavelengths, col)


# ── k_th loading ─────────────────────────────────────────────────────────────

def load_k_th(params: QYParams) -> tuple[float, float]:
    """Return (k_th, k_th_std) in s⁻¹ from the configured source."""
    if params.k_th_source == "none":
        return 0.0, 0.0

    if params.k_th_source == "manual":
        print(f"  k_th = {params.k_th_manual:.4e} ± "
              f"{params.k_th_manual_std:.4e} s⁻¹ (manual)")
        return params.k_th_manual, params.k_th_manual_std

    if params.k_th_source == "half_life_master":
        if params.k_th_csv is None or not params.k_th_csv.exists():
            raise FileNotFoundError(
                "half_life_master.csv not set. Browse in Stage 2.")
        df = pd.read_csv(params.k_th_csv)
        T  = params.k_th_temperature_C
        sub = df[df["Temperature_C"] == T]
        if sub.empty:
            avail = sorted(df["Temperature_C"].unique())
            raise ValueError(
                f"No rows at Temperature_C = {T} °C in half_life_master.csv. "
                f"Available: {avail}")
        k_vals   = sub["k"].dropna().values
        k_th     = float(k_vals.mean())
        k_th_std = (float(k_vals.std(ddof=1) / np.sqrt(len(k_vals)))
                    if len(k_vals) > 1 else 0.0)
        print(f"  k_th (half-life master, T={T} °C, n={len(k_vals)}): "
              f"{k_th:.4e} ± {k_th_std:.4e} s⁻¹")
        return k_th, k_th_std

    if params.k_th_source == "eyring":
        if params.k_th_csv is None or not params.k_th_csv.exists():
            raise FileNotFoundError("Eyring results CSV not set. Browse in Stage 2.")
        df  = pd.read_csv(params.k_th_csv)
        row = df.iloc[-1]
        dH  = float(row["dH_kJmol"]) * 1000
        dS  = float(row["dS_JmolK"])
        T_K = params.k_th_temperature_C + 273.15
        k_th = (kB * T_K / h_ey) * np.exp(-dH / (R_GAS * T_K) + dS / R_GAS)
        print(f"  k_th (Eyring, T={params.k_th_temperature_C} °C): {k_th:.4e} s⁻¹")
        return k_th, 0.0

    if params.k_th_source == "arrhenius":
        if params.k_th_csv is None or not params.k_th_csv.exists():
            raise FileNotFoundError("Arrhenius results CSV not set. Browse in Stage 2.")
        df  = pd.read_csv(params.k_th_csv)
        row = df.iloc[-1]
        Ea  = float(row["Ea_kJmol"]) * 1000
        A_f = float(row["A_s"])
        T_K = params.k_th_temperature_C + 273.15
        k_th = A_f * np.exp(-Ea / (R_GAS * T_K))
        print(f"  k_th (Arrhenius, T={params.k_th_temperature_C} °C): {k_th:.4e} s⁻¹")
        return k_th, 0.0

    raise ValueError(f"Unknown k_th_source: '{params.k_th_source}'")


# ── Main per-file computation ─────────────────────────────────────────────────

def run_qy_file(
    params:      QYParams,
    data_file:   Path,
    N_mol_s:     float,
    N_std_mol_s: float,
    k_th:        float,
    k_th_std:    float,
    eps_A_irr:   float,
    eps_B_irr:   float,
    led_wl_arr:  Optional[np.ndarray],
    led_N_arr:   Optional[np.ndarray],
    led_eps_A_arr: Optional[np.ndarray],
    led_eps_B_arr: Optional[np.ndarray],
) -> QYFileResult:
    """
    Run the full QY calculation for one experimental data file.
    Pre-resolved inputs: N, k_th, ε_irr, LED arrays.
    Returns QYFileResult.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {data_file.name}")
    V_L = params.volume_mL / 1000.0

    # ── Load data ─────────────────────────────────────────────────────────────
    time_s   = None
    abs_data = None
    kin_t2d  = None

    if params.data_type == "scanning":
        scans = load_spectra_csv(data_file)
        n_scans = len(scans)
        if n_scans < params.scans_per_group:
            raise ValueError(
                f"Only {n_scans} scans < scans_per_group={params.scans_per_group}.")
        n_groups = n_scans // params.scans_per_group
        time_s   = np.arange(n_groups, dtype=float) * params.delta_t_s
        # resolve monitoring wavelengths
        mon_wls = (list(params.monitoring_wavelengths)
                   if params.monitoring_wavelengths else [scans[0][0].mean()])
        if params.monitoring_wavelengths is None:
            raise ValueError(
                "Monitoring wavelengths must be specified for scanning data.")
        mon_wls  = [float(w) for w in params.monitoring_wavelengths]
        abs_data = np.full((n_groups, len(mon_wls)), np.nan)
        for g in range(n_groups):
            for s in range(params.scans_per_group):
                wl_s, ab_s = scans[g * params.scans_per_group + s]
                for j, wl_nm in enumerate(mon_wls):
                    val = extract_absorbance(wl_s, ab_s, wl_nm,
                                             params.wavelength_tolerance_nm)
                    if not np.isnan(val):
                        if np.isnan(abs_data[g, j]):
                            abs_data[g, j] = val
                        else:
                            abs_data[g, j] = (abs_data[g, j] + val) / 2
        print(f"  Scanning: {n_scans} scans, {n_groups} time points")

    elif params.data_type == "kinetic":
        channels = load_kinetic_csv(data_file)
        if not channels:
            raise ValueError(f"No channels loaded from {data_file.name}.")

        # Resolve monitoring wavelengths from channel labels
        if params.monitoring_wavelengths is None:
            selected = channels
            mon_wls = []
            for label in selected:
                m = (re.search(r"(\d+(?:\.\d+)?)\s*nm", label, re.IGNORECASE)
                     or re.search(r"(\d+(?:\.\d+)?)(?:\D*)$", label))
                mon_wls.append(float(m.group(1)) if m else float("nan"))
        else:
            mon_wls  = [float(w) for w in params.monitoring_wavelengths]
            selected = {}
            for label, (t_arr, a_arr) in channels.items():
                m = (re.search(r"(\d+(?:\.\d+)?)\s*nm", label, re.IGNORECASE)
                     or re.search(r"(\d+(?:\.\d+)?)(?:\D*)$", label))
                ch_wl = float(m.group(1)) if m else None
                if ch_wl is not None:
                    for wl_req in mon_wls:
                        if abs(ch_wl - wl_req) <= params.wavelength_tolerance_nm:
                            selected[label] = (t_arr, a_arr)
                            break
            if not selected:
                items   = list(channels.items())[:len(mon_wls)]
                selected = dict(items)

        ch_items = list(selected.items())
        min_len  = min(len(t) for t, _ in selected.values())
        t_arrays = [t[:min_len] for (_, (t, _)) in ch_items]
        a_arrays = [a[:min_len] for (_, (_, a)) in ch_items]
        time_s   = t_arrays[0]
        kin_t2d  = np.column_stack(t_arrays)
        abs_data = np.column_stack(a_arrays)
        mon_wls  = mon_wls[:abs_data.shape[1]]
        print(f"  Kinetic: {len(ch_items)} channel(s), {min_len} time points, "
              f"λ = {mon_wls}")
    else:
        raise ValueError(f"Unknown data_type: '{params.data_type}'")

    # ── Baseline correction ───────────────────────────────────────────────────
    baseline_values   = np.zeros(len(mon_wls))
    initial_spec_abs  = None

    if params.baseline_correction == "first_point":
        baseline_values = abs_data[0, :].copy()
        abs_data = abs_data - baseline_values[np.newaxis, :]
        print("  Baseline: first point subtracted.")

    elif params.baseline_correction == "plateau":
        t_lo = (params.baseline_plateau_start_s
                if params.baseline_plateau_start_s is not None else time_s[0])
        t_hi = (params.baseline_plateau_end_s
                if params.baseline_plateau_end_s is not None
                else (params.fit_time_start_s
                      if params.fit_time_start_s is not None else time_s[0]))
        mask_p = (time_s >= t_lo) & (time_s <= t_hi)
        if not mask_p.any():
            raise ValueError(
                f"Baseline plateau [{t_lo}, {t_hi}] s contains no data points.")
        baseline_values = abs_data[mask_p, :].mean(axis=0)
        abs_data = abs_data - baseline_values[np.newaxis, :]
        print(f"  Baseline: plateau mean over {t_lo:.1f}–{t_hi:.1f} s.")

    elif params.baseline_correction == "file":
        if params.baseline_file is None or not params.baseline_file.exists():
            raise FileNotFoundError(
                "Baseline correction = 'file' but no initial spectrum file set.")
        init_scans = load_spectra_csv(params.baseline_file)
        init_abs = np.zeros(len(mon_wls))
        init_n   = np.zeros(len(mon_wls), dtype=int)
        for wl_s, ab_s in init_scans:
            for j, wl_nm in enumerate(mon_wls):
                val = extract_absorbance(wl_s, ab_s, wl_nm,
                                          params.wavelength_tolerance_nm)
                if not np.isnan(val):
                    init_abs[j] += val
                    init_n[j]   += 1
        for j, wl_nm in enumerate(mon_wls):
            if init_n[j] == 0:
                raise ValueError(
                    f"No absorbance at {wl_nm} nm in baseline file.")
            init_abs[j] /= init_n[j]
        # Offset
        if params.offset_plateau_duration_s is not None:
            off_mask = time_s <= (time_s[0] + params.offset_plateau_duration_s)
            if not off_mask.any():
                off_mask[0] = True
        else:
            off_mask = np.zeros(len(time_s), dtype=bool)
            off_mask[0] = True
        kin_t0 = abs_data[off_mask, :].mean(axis=0)
        abs_data = abs_data + (init_abs - kin_t0)[np.newaxis, :]
        initial_spec_abs = init_abs
        print(f"  Baseline: file offset alignment using {params.baseline_file.name}.")

    # Drop NaN rows
    valid = ~np.any(np.isnan(abs_data), axis=1)
    if not np.all(valid):
        print(f"  Dropping {(~valid).sum()} NaN rows.")
    time_s   = time_s[valid]
    abs_data = abs_data[valid, :]
    if kin_t2d is not None:
        kin_t2d = kin_t2d[valid, :]

    if len(time_s) < 3:
        raise ValueError("Fewer than 3 valid time points after baseline correction.")

    # ── Auto-detect irradiation start ────────────────────────────────────────
    fit_start_eff = params.fit_time_start_s
    fit_end_eff   = params.fit_time_end_s

    if params.auto_detect_irr_start:
        t_fit, t_onset, _, _, _, _ = detect_irr_start(
            time_s, abs_data,
            params.auto_detect_n_plateau,
            params.auto_detect_threshold,
            params.auto_detect_min_consec,
        )
        if t_fit is not None:
            fit_start_eff = t_fit
            print(f"  Auto-detect: irr onset at t={t_onset:.2f} s, "
                  f"fit start at t={t_fit:.2f} s.")
        else:
            print("  Auto-detect: no irr start found; using manual fit_time_start_s.")

    # ── Fit window ────────────────────────────────────────────────────────────
    fit_mask = np.ones(len(time_s), dtype=bool)
    if fit_start_eff is not None:
        fit_mask &= (time_s >= fit_start_eff)
    if fit_end_eff is not None:
        fit_mask &= (time_s <= fit_end_eff)

    time_s_fit   = time_s[fit_mask]
    abs_data_fit = abs_data[fit_mask, :]
    if len(time_s_fit) < 3:
        raise ValueError(
            f"Only {len(time_s_fit)} points in fit window; need ≥ 3. "
            f"Adjust fit_time_start_s / fit_time_end_s.")

    time_s_ode = time_s_fit - time_s_fit[0]
    kin_t2d_fit = None
    kin_t2d_ode = None
    if kin_t2d is not None:
        kin_t2d_fit = kin_t2d[fit_mask, :]
        kin_t2d_ode = kin_t2d_fit - kin_t2d_fit[0, :]
    print(f"  Fit window: {time_s_fit[0]:.1f} – {time_s_fit[-1]:.1f} s "
          f"({len(time_s_fit)} pts)")

    # ── ε at monitoring wavelengths ───────────────────────────────────────────
    eps_A_mon = load_epsilon_at_wavelengths(
        params.epsilon_source_A, params.epsilon_A_csv,
        params.epsilon_A_col, mon_wls, params.epsilon_A_irr)
    eps_B_mon = load_epsilon_at_wavelengths(
        params.epsilon_source_B, params.epsilon_B_csv,
        params.epsilon_B_col, mon_wls, params.epsilon_B_irr)

    # LED ε at LED wavelengths (full integration only)
    use_led_full = (led_wl_arr is not None and led_N_arr is not None
                    and params.led_integration_mode == "full"
                    and led_eps_A_arr is not None and led_eps_B_arr is not None)

    # ── Initial conditions ────────────────────────────────────────────────────
    if params.initial_conc_source == "absorbance":
        if initial_spec_abs is not None:
            A0 = initial_spec_abs[0]
        else:
            A0 = abs_data_fit[0, 0] + baseline_values[0]
        conc_A_0 = A0 / (eps_A_mon[0] * params.path_length_cm)
        conc_B_0 = 0.0
        print(f"  [A]₀ = {conc_A_0:.4e} mol L⁻¹ (from absorbance)")
    elif params.initial_conc_source == "manual":
        if params.initial_conc_A_manual is None:
            raise ValueError("initial_conc_source = 'manual' but initial_conc_A_manual is None.")
        conc_A_0 = params.initial_conc_A_manual
        conc_B_0 = params.initial_conc_B_manual
        print(f"  [A]₀ = {conc_A_0:.4e} mol L⁻¹ (manual)")
    else:
        raise ValueError(f"Unknown initial_conc_source: '{params.initial_conc_source}'")

    # ── Build lmfit params ────────────────────────────────────────────────────
    def make_params():
        p = Parameters()
        if params.QY_unconstrained:
            p.add("QY_AB", value=params.QY_AB_init)
        else:
            p.add("QY_AB", value=params.QY_AB_init,
                  min=params.QY_bounds_lo, max=params.QY_bounds_hi)
        if params.case == "AB_both":
            if params.QY_unconstrained:
                p.add("QY_BA", value=params.QY_BA_init)
            else:
                p.add("QY_BA", value=params.QY_BA_init,
                      min=params.QY_bounds_lo, max=params.QY_bounds_hi)
        else:
            p.add("QY_BA", value=0.0, vary=False)
        return p

    # ── A_thermal_PSS: algebraic ──────────────────────────────────────────────
    if params.case == "A_thermal_PSS":
        if k_th == 0.0:
            raise ValueError("A_thermal_PSS requires k_th > 0.")

        A0_irr = conc_A_0 * eps_A_irr * params.path_length_cm
        if params.pss_source == "manual_fraction":
            f_B = params.pss_fraction_B_manual
            if f_B is None:
                raise ValueError("pss_fraction_B_manual is None.")
            pss_B_conc    = conc_A_0 * f_B
            pss_A_abs_val = A0_irr * (1.0 - f_B)
        elif params.pss_source == "manual_absorbance":
            if params.pss_A_abs_pss_manual is None:
                raise ValueError("pss_A_abs_pss_manual is None.")
            pss_A_abs_val = params.pss_A_abs_pss_manual
            pss_B_conc    = conc_A_0 - pss_A_abs_val / (eps_A_irr * params.path_length_cm)
        else:
            raise ValueError(f"Unknown pss_source: '{params.pss_source}'")

        QY_AB_nom = pss_algebraic(k_th, pss_B_conc, V_L, N_mol_s, pss_A_abs_val)
        # I₀ perturbation
        sigma_I0_AB = 0.0
        if N_std_mol_s > 0:
            QY_hi = pss_algebraic(k_th, pss_B_conc, V_L, N_mol_s + N_std_mol_s, pss_A_abs_val)
            QY_lo = pss_algebraic(k_th, pss_B_conc, V_L, N_mol_s - N_std_mol_s, pss_A_abs_val)
            sigma_I0_AB = (abs(QY_hi - QY_AB_nom) + abs(QY_AB_nom - QY_lo)) / 2.0
        sigma_kth_AB = QY_AB_nom * (k_th_std / k_th) if k_th_std > 0 and k_th > 0 else 0.0
        sigma_total = np.sqrt(sigma_I0_AB**2 + sigma_kth_AB**2)
        print(f"  Φ_AB (PSS algebraic) = {QY_AB_nom:.5f} ± {sigma_total:.5f}")

        # Minimal plot arrays (no time-series fit)
        empty_arr = np.array([])
        return QYFileResult(
            file_name=data_file.name, compound=params.compound_name,
            case=params.case, mon_wls=mon_wls,
            N_mol_s=N_mol_s, N_std_mol_s=N_std_mol_s,
            k_th=k_th, eps_A_irr=eps_A_irr, eps_B_irr=eps_B_irr,
            conc_A_0=conc_A_0, V_L=V_L, path_length_cm=params.path_length_cm,
            temperature_C=params.temperature_C, solvent=params.solvent,
            QY_AB_per_wl=[QY_AB_nom], QY_BA_per_wl=[0.0],
            stderr_AB_per_wl=[np.nan], sigma_I0_AB_per_wl=[sigma_I0_AB],
            sigma_total_per_wl=[sigma_total], r2_per_wl=[np.nan],
            QY_AB=QY_AB_nom, QY_BA=0.0,
            QY_AB_sigma_fit=np.nan, QY_AB_sigma_total=sigma_total,
            r2=np.nan, method="PSS_algebraic",
            QY_slopes=[np.nan],
            time_s=time_s, abs_data=abs_data, abs_fit_per_wl=[],
            abs_fit_lo_per_wl=[], abs_fit_hi_per_wl=[],
            residuals_2d=empty_arr.reshape(0, len(mon_wls)),
            time_s_fit=time_s_fit, fit_mask=fit_mask, t_display_per_wl=[time_s],
        )

    # ── ODE fitting (A_only or AB_both) ──────────────────────────────────────
    n_mon = len(mon_wls)
    QY_AB_per_wl, QY_BA_per_wl   = [], []
    stderr_AB_per_wl              = []
    sigma_I0_AB_per_wl            = []
    sigma_total_per_wl            = []
    r2_per_wl                     = []

    for j in range(n_mon):
        t_j   = (kin_t2d_ode[:, j] if kin_t2d_ode is not None else time_s_ode)
        abs_j = abs_data_fit[:, j:j+1]
        eA_j  = eps_A_mon[j:j+1]
        eB_j  = eps_B_mon[j:j+1]

        if use_led_full:
            res_j = run_fit_led(make_params(), t_j, abs_j,
                                conc_A_0, conc_B_0,
                                led_wl_arr, led_N_arr,
                                led_eps_A_arr, led_eps_B_arr,
                                eA_j, eB_j, V_L, k_th, params.path_length_cm)
        else:
            res_j = run_fit(make_params(), t_j, abs_j,
                            conc_A_0, conc_B_0,
                            eps_A_irr, eps_B_irr, eA_j, eB_j,
                            N_mol_s, V_L, k_th, params.path_length_cm)

        QY_AB_j = res_j.params["QY_AB"].value
        QY_BA_j = res_j.params["QY_BA"].value
        se_AB_j = res_j.params["QY_AB"].stderr or np.nan
        rr_j    = res_j.residual
        sst_j   = np.sum((abs_j.flatten() - abs_j.mean()) ** 2)
        r2_j    = 1.0 - np.sum(rr_j**2) / sst_j if sst_j > 0 else np.nan

        # I₀ perturbation
        si0_AB_j = 0.0
        if N_std_mol_s > 0:
            if use_led_full:
                sc = N_std_mol_s / N_mol_s
                rhi = run_fit_led(make_params(), t_j, abs_j, conc_A_0, conc_B_0,
                                  led_wl_arr, led_N_arr * (1 + sc),
                                  led_eps_A_arr, led_eps_B_arr,
                                  eA_j, eB_j, V_L, k_th, params.path_length_cm)
                rlo = run_fit_led(make_params(), t_j, abs_j, conc_A_0, conc_B_0,
                                  led_wl_arr, led_N_arr * (1 - sc),
                                  led_eps_A_arr, led_eps_B_arr,
                                  eA_j, eB_j, V_L, k_th, params.path_length_cm)
            else:
                rhi = run_fit(make_params(), t_j, abs_j, conc_A_0, conc_B_0,
                              eps_A_irr, eps_B_irr, eA_j, eB_j,
                              N_mol_s + N_std_mol_s, V_L, k_th, params.path_length_cm)
                rlo = run_fit(make_params(), t_j, abs_j, conc_A_0, conc_B_0,
                              eps_A_irr, eps_B_irr, eA_j, eB_j,
                              N_mol_s - N_std_mol_s, V_L, k_th, params.path_length_cm)
            si0_AB_j = (abs(rhi.params["QY_AB"].value - QY_AB_j)
                        + abs(QY_AB_j - rlo.params["QY_AB"].value)) / 2.0

        st_j = np.sqrt((se_AB_j if not np.isnan(se_AB_j) else 0.0)**2 + si0_AB_j**2)
        QY_AB_per_wl.append(QY_AB_j)
        QY_BA_per_wl.append(QY_BA_j)
        stderr_AB_per_wl.append(se_AB_j)
        sigma_I0_AB_per_wl.append(si0_AB_j)
        sigma_total_per_wl.append(st_j)
        r2_per_wl.append(r2_j)
        print(f"  λ={mon_wls[j]:.0f} nm: Φ_AB={QY_AB_j:.5f} "
              f"σ_total={st_j:.5f} R²={r2_j:.5f}")

    QY_AB_nom       = float(np.nanmean(QY_AB_per_wl))
    QY_BA_nom       = float(np.nanmean(QY_BA_per_wl)) if params.case == "AB_both" else 0.0
    stderr_AB_mean  = float(np.nanmean(stderr_AB_per_wl))
    sigma_total_AB  = float(np.nanmean(sigma_total_per_wl))
    r2_mean         = float(np.nanmean(r2_per_wl))
    method = ("ODE_lmfit_LED_full" if use_led_full else "ODE_lmfit")

    # ── Initial slopes ────────────────────────────────────────────────────────
    QY_slopes, _ = initial_slopes_QY(
        time_s_ode, abs_data_fit, params.n_initial_slopes_pts,
        eps_A_irr, eps_A_mon, eps_B_mon,
        N_mol_s, V_L, params.path_length_cm, conc_A_0)

    # ── Build display curves ──────────────────────────────────────────────────
    disp_mask = time_s >= time_s_fit[0]
    time_disp = time_s[disp_mask]
    fit_in_disp = fit_mask[disp_mask]

    def t_disp_ode(j):
        if kin_t2d is not None:
            td = kin_t2d[disp_mask, j]
            return td - kin_t2d_fit[0, j]
        return time_disp - time_s_fit[0]

    def sim_curve(p, t_ode, j):
        if use_led_full:
            return simulate_absorbance_led(
                p, t_ode, conc_A_0, conc_B_0,
                led_wl_arr, led_N_arr, led_eps_A_arr, led_eps_B_arr,
                eps_A_mon[j:j+1], eps_B_mon[j:j+1], V_L, k_th,
                params.path_length_cm)[:, 0]
        return simulate_absorbance(
            p, t_ode, conc_A_0, conc_B_0,
            eps_A_irr, eps_B_irr,
            eps_A_mon[j:j+1], eps_B_mon[j:j+1],
            N_mol_s, V_L, k_th, params.path_length_cm)[:, 0]

    abs_fit_per_wl    = []
    abs_fit_lo_per_wl = []
    abs_fit_hi_per_wl = []
    t_display_per_wl  = []

    for j in range(n_mon):
        td_ode = t_disp_ode(j)
        p_j  = make_params(); p_j["QY_AB"].set(value=QY_AB_per_wl[j], vary=False)
        if params.case == "AB_both":
            p_j["QY_BA"].set(value=QY_BA_per_wl[j], vary=False)

        lo_val = (QY_AB_per_wl[j] - sigma_total_per_wl[j]
                  if params.QY_unconstrained
                  else max(QY_AB_per_wl[j] - sigma_total_per_wl[j], params.QY_bounds_lo))
        hi_val = (QY_AB_per_wl[j] + sigma_total_per_wl[j]
                  if params.QY_unconstrained
                  else min(QY_AB_per_wl[j] + sigma_total_per_wl[j], params.QY_bounds_hi))
        p_hi = make_params(); p_hi["QY_AB"].set(value=hi_val, vary=False)
        p_lo = make_params(); p_lo["QY_AB"].set(value=lo_val, vary=False)

        abs_fit_per_wl.append(sim_curve(p_j,  td_ode, j))
        abs_fit_hi_per_wl.append(sim_curve(p_hi, td_ode, j))
        abs_fit_lo_per_wl.append(sim_curve(p_lo, td_ode, j))

        td_abs = kin_t2d[disp_mask, j] if kin_t2d is not None else time_disp
        t_display_per_wl.append(td_abs)

    # Residuals
    residuals_2d = np.column_stack([
        abs_fit_per_wl[j][fit_in_disp] - abs_data_fit[:, j]
        for j in range(n_mon)
    ])

    return QYFileResult(
        file_name=data_file.name, compound=params.compound_name,
        case=params.case, mon_wls=mon_wls,
        N_mol_s=N_mol_s, N_std_mol_s=N_std_mol_s,
        k_th=k_th, eps_A_irr=eps_A_irr, eps_B_irr=eps_B_irr,
        conc_A_0=conc_A_0, V_L=V_L, path_length_cm=params.path_length_cm,
        temperature_C=params.temperature_C, solvent=params.solvent,
        QY_AB_per_wl=QY_AB_per_wl, QY_BA_per_wl=QY_BA_per_wl,
        stderr_AB_per_wl=stderr_AB_per_wl,
        sigma_I0_AB_per_wl=sigma_I0_AB_per_wl,
        sigma_total_per_wl=sigma_total_per_wl,
        r2_per_wl=r2_per_wl,
        QY_AB=QY_AB_nom, QY_BA=QY_BA_nom,
        QY_AB_sigma_fit=stderr_AB_mean, QY_AB_sigma_total=sigma_total_AB,
        r2=r2_mean, method=method,
        QY_slopes=QY_slopes,
        time_s=time_s, abs_data=abs_data,
        abs_fit_per_wl=abs_fit_per_wl,
        abs_fit_lo_per_wl=abs_fit_lo_per_wl,
        abs_fit_hi_per_wl=abs_fit_hi_per_wl,
        residuals_2d=residuals_2d,
        time_s_fit=time_s_fit, fit_mask=fit_mask,
        t_display_per_wl=t_display_per_wl,
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_qy_result(result: QYFileResult) -> plt.Figure:
    """Main result plot: absorbance per wavelength + residuals + concentrations."""
    n_mon = len(result.mon_wls)

    if result.case == "A_thermal_PSS":
        # Simple bar chart for PSS algebraic
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Φ_AB (PSS)"], [result.QY_AB],
               yerr=[result.QY_AB_sigma_total],
               color="#c0392b", alpha=0.8, capsize=6)
        ax.set_ylabel("Quantum yield Φ_AB")
        ax.set_title(f"{result.compound}  |  {result.file_name}")
        ax.grid(True, axis="y", alpha=0.4)
        plt.tight_layout()
        return fig

    n_rows = n_mon + 2
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(8, 2.5 * n_rows),
        gridspec_kw={"height_ratios": [3] * n_mon + [1.5, 2.5]},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for j, wl_nm in enumerate(result.mon_wls):
        ax = axes[j]
        td_j = result.t_display_per_wl[j] if result.t_display_per_wl else result.time_s
        ax.plot(result.time_s, result.abs_data[:, j], "-", color="#3a7ebf",
                linewidth=1.8, label=f"Data  λ={wl_nm:.0f} nm")
        if len(result.abs_fit_per_wl) > j:
            ax.plot(td_j, result.abs_fit_per_wl[j], "--", color="red",
                    linewidth=1.6,
                    label=f"Fit  Φ_AB={result.QY_AB_per_wl[j]:.4f}  "
                          f"R²={result.r2_per_wl[j]:.4f}")
            ax.fill_between(td_j,
                            result.abs_fit_lo_per_wl[j],
                            result.abs_fit_hi_per_wl[j],
                            color="#e67e22", alpha=0.30, label="±σ_total")
        ax.axvspan(result.time_s_fit[0], result.time_s_fit[-1],
                   alpha=0.10, color="gray",
                   label="Fit window" if j == 0 else None)
        ax.set_ylabel("Absorbance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    # Annotation
    lines = []
    for j, wl_nm in enumerate(result.mon_wls):
        lines.append(
            f"λ={wl_nm:.0f} nm: Φ_AB={result.QY_AB_per_wl[j]:.4f}  "
            f"σ_total={result.sigma_total_per_wl[j]:.4f}  "
            f"R²={result.r2_per_wl[j]:.4f}")
    lines.append(f"Mean Φ_AB = {result.QY_AB:.4f}  σ_total = {result.QY_AB_sigma_total:.4f}")
    lines.append(f"R² (mean) = {result.r2:.4f}")
    axes[0].text(0.97, 0.97, "\n".join(lines),
                 transform=axes[0].transAxes, fontsize=8,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    axes[0].set_title(
        f"{result.compound}  |  {result.case}  |  {result.file_name}")

    # Residuals panel
    ax_res = axes[n_mon]
    if result.residuals_2d.size > 0:
        for j, wl_nm in enumerate(result.mon_wls):
            t_fit_j = result.time_s_fit
            ax_res.plot(t_fit_j, result.residuals_2d[:, j], linewidth=1.2,
                        label=f"{wl_nm:.0f} nm")
    ax_res.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_res.set_ylabel("Residual")
    ax_res.legend(fontsize=8)
    ax_res.grid(True, alpha=0.4)

    # Concentrations panel
    ax_conc = axes[n_mon + 1]
    for j, wl_nm in enumerate(result.mon_wls):
        A_obs = result.abs_data[:, j] / (
            (result.eps_A_irr or 1.0) * result.path_length_cm) * 1000
        B_obs = result.conc_A_0 * 1000 - A_obs
        ax_conc.plot(result.time_s, A_obs, ".", markersize=3, alpha=0.5,
                     label=f"[A] obs {wl_nm:.0f} nm")
        ax_conc.plot(result.time_s, B_obs, ".", markersize=3, alpha=0.5,
                     label=f"[B] obs {wl_nm:.0f} nm" if j == 0 else "_nolegend_")
    ax_conc.set_xlabel("Time (s)")
    ax_conc.set_ylabel("Concentration (mmol L⁻¹)")
    ax_conc.legend(fontsize=8)
    ax_conc.grid(True, alpha=0.4)

    return fig
