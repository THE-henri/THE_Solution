"""
Core functions for the Thermal tab.

Arrhenius analysis – extracted from workflows/arrhenius_analysis.py
Eyring analysis    – extracted from workflows/eyring_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ── Physical constants ────────────────────────────────────────────────────────

R        = 8.314462       # J mol⁻¹ K⁻¹
kB       = 1.380649e-23   # J K⁻¹
h        = 6.626070e-34   # J s
LN_KB_H  = np.log(kB / h) # ≈ 23.760


# ── Shared helpers ────────────────────────────────────────────────────────────

def _linear(x, slope, intercept):
    return slope * x + intercept


def load_half_life_master(filepath: Path) -> pd.DataFrame:
    """Load and basic-validate a half_life_master.csv file."""
    df = pd.read_csv(filepath)
    required = {"Temperature_C", "k"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"half_life_master.csv is missing columns: {missing}")
    return df


def group_by_temperature(
    df: pd.DataFrame,
) -> list[tuple[float, float, Optional[float], int]]:
    """
    Group half-life master rows by Temperature_C.

    Returns sorted list of (T_K, k_mean, k_sem_or_None, n).
    k_sem is None when only one measurement exists at that temperature.
    """
    entries = []
    for T_C, grp in df.groupby("Temperature_C"):
        k_vals = grp["k"].dropna().values
        if len(k_vals) == 0:
            continue
        k_mean = float(k_vals.mean())
        k_sem  = float(k_vals.std(ddof=1) / np.sqrt(len(k_vals))) \
            if len(k_vals) > 1 else None
        T_K = float(T_C) + 273.15
        entries.append((T_K, k_mean, k_sem, len(k_vals)))
    return sorted(entries)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ArrheniusResult:
    compound:       str
    Ea_kJmol:       float
    Ea_std_kJmol:   float
    A_s:            float
    A_std_s:        float
    r2:             float
    n_temperatures: int
    weighted:       bool
    # arrays for the plot (1000/T axis)
    x_data:         np.ndarray   # 1000 / T_K
    ln_k:           np.ndarray
    ln_k_pred:      np.ndarray
    sigma_y:        Optional[np.ndarray]
    T_C_list:       list[float]
    k_mean_list:    list[float]


@dataclass
class EyringResult:
    compound:       str
    dH_kJmol:       float
    dH_std_kJmol:   float
    dS_JmolK:       float
    dS_std_JmolK:   float
    r2:             float
    n_temperatures: int
    weighted:       bool
    # arrays for the plot (1000/T axis)
    x_data:         np.ndarray   # 1000 / T_K
    ln_kT:          np.ndarray
    ln_kT_pred:     np.ndarray
    sigma_y:        Optional[np.ndarray]
    T_C_list:       list[float]
    k_mean_list:    list[float]


# ── Arrhenius ─────────────────────────────────────────────────────────────────

def run_arrhenius(
    master_csv:    Path,
    compound_name: str,
    weighted_fit:  bool,
) -> ArrheniusResult:
    """
    Fit the Arrhenius equation to k(T) data in half_life_master.csv.
    Mirrors workflows/arrhenius_analysis.py.
    """
    df = load_half_life_master(master_csv)
    print(f"Loaded {len(df)} rows from {master_csv.name}")

    entries = group_by_temperature(df)
    if len(entries) < 2:
        raise ValueError(
            f"Arrhenius requires ≥ 2 temperatures; "
            f"found {len(entries)} in master CSV."
        )

    T_K_arr    = np.array([e[0] for e in entries])
    k_arr      = np.array([e[1] for e in entries])
    k_sem_list = [e[2] for e in entries]
    T_C_list   = [e[0] - 273.15 for e in entries]

    for T_C, k_m, k_s, n in zip(T_C_list, k_arr, k_sem_list,
                                  [e[3] for e in entries]):
        sem_str = f"± {k_s:.6f}" if k_s is not None else "(n = 1)"
        print(f"  T = {T_C:.1f} °C  →  k = {k_m:.6f} {sem_str} s⁻¹")

    inv_T = 1.0 / T_K_arr
    ln_k  = np.log(k_arr)

    use_weights = (weighted_fit
                   and all(s is not None and s > 0 for s in k_sem_list))
    if use_weights:
        sigma_y = np.array([s / k for s, k in zip(k_sem_list, k_arr)])
        popt, pcov = curve_fit(_linear, inv_T, ln_k,
                               sigma=sigma_y, absolute_sigma=True)
    else:
        sigma_y = None
        popt, pcov = curve_fit(_linear, inv_T, ln_k)

    slope, intercept      = popt
    slope_std, intcpt_std = np.sqrt(np.diag(pcov))

    Ea_J      = -slope * R
    Ea_std_J  = slope_std * R
    Ea_kJ     = Ea_J / 1000.0
    Ea_std_kJ = Ea_std_J / 1000.0
    A         = float(np.exp(intercept))
    A_std     = A * intcpt_std

    ln_k_pred = _linear(inv_T, slope, intercept)
    ss_res    = np.sum((ln_k - ln_k_pred) ** 2)
    ss_tot    = np.sum((ln_k - ln_k.mean()) ** 2)
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_data = 1000.0 * inv_T
    print(f"\nArrhenius ({len(entries)} temperatures, "
          f"{'weighted' if use_weights else 'unweighted'}):")
    print(f"  Ea = {Ea_kJ:.2f} ± {Ea_std_kJ:.2f} kJ mol⁻¹")
    print(f"  A  = {A:.4e} ± {A_std:.4e} s⁻¹")
    print(f"  R² = {r2:.6f}")

    return ArrheniusResult(
        compound=compound_name,
        Ea_kJmol=Ea_kJ,
        Ea_std_kJmol=Ea_std_kJ,
        A_s=A,
        A_std_s=A_std,
        r2=r2,
        n_temperatures=len(entries),
        weighted=use_weights,
        x_data=x_data,
        ln_k=ln_k,
        ln_k_pred=_linear(x_data / 1000.0, slope, intercept),
        sigma_y=sigma_y,
        T_C_list=T_C_list,
        k_mean_list=list(k_arr),
    )


def plot_arrhenius(result: ArrheniusResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    if result.sigma_y is not None:
        ax.errorbar(result.x_data, result.ln_k, yerr=result.sigma_y,
                    fmt="o", color="black", markerfacecolor="none",
                    markeredgewidth=1.2, capsize=4, elinewidth=1.2,
                    label="Data ± σ")
    else:
        ax.plot(result.x_data, result.ln_k,
                "o", color="black", markerfacecolor="none",
                markeredgewidth=1.2, label="Data")

    x_fit = np.linspace(result.x_data.min() * 0.998,
                        result.x_data.max() * 1.002, 200)
    # reconstruct slope & intercept from result for the fit line
    # slope = -Ea_J / R  ;  intercept = ln(A)
    slope_r    = -(result.Ea_kJmol * 1000.0) / R
    intercept_r = np.log(result.A_s)
    ax.plot(x_fit, _linear(x_fit / 1000.0, slope_r, intercept_r),
            "--", color="red", linewidth=2, label="Arrhenius fit")

    annotation = (
        r"$\ln(k) = \ln(A) - \frac{E_a}{R}\cdot\frac{1}{T}$" + "\n"
        f"$E_a$ = {result.Ea_kJmol:.2f} ± {result.Ea_std_kJmol:.2f}"
        r" kJ mol$^{-1}$" + "\n"
        f"$A$   = {result.A_s:.4e} ± {result.A_std_s:.4e} s$^{{-1}}$\n"
        f"$R^2$ = {result.r2:.4f}"
    )
    ax.text(0.97, 0.97, annotation,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$1000\,/\,T\ \mathrm{(K^{-1})}$")
    ax.set_ylabel(r"$\ln(k)$")
    ax.set_title(f"Arrhenius Plot — {result.compound}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


# ── Eyring ────────────────────────────────────────────────────────────────────

def run_eyring(
    master_csv:    Path,
    compound_name: str,
    weighted_fit:  bool,
) -> EyringResult:
    """
    Fit the Eyring equation to k(T) data in half_life_master.csv.
    Mirrors workflows/eyring_analysis.py.
    """
    df = load_half_life_master(master_csv)
    print(f"Loaded {len(df)} rows from {master_csv.name}")

    entries = group_by_temperature(df)
    if len(entries) < 2:
        raise ValueError(
            f"Eyring requires ≥ 2 temperatures; "
            f"found {len(entries)} in master CSV."
        )

    T_K_arr    = np.array([e[0] for e in entries])
    k_arr      = np.array([e[1] for e in entries])
    k_sem_list = [e[2] for e in entries]
    T_C_list   = [e[0] - 273.15 for e in entries]

    for T_C, k_m, k_s, n in zip(T_C_list, k_arr, k_sem_list,
                                  [e[3] for e in entries]):
        sem_str = f"± {k_s:.6f}" if k_s is not None else "(n = 1)"
        print(f"  T = {T_C:.1f} °C  →  k = {k_m:.6f} {sem_str} s⁻¹")

    inv_T = 1.0 / T_K_arr
    ln_kT = np.log(k_arr / T_K_arr)

    use_weights = (weighted_fit
                   and all(s is not None and s > 0 for s in k_sem_list))
    if use_weights:
        sigma_y = np.array([s / k for s, k in zip(k_sem_list, k_arr)])
        popt, pcov = curve_fit(_linear, inv_T, ln_kT,
                               sigma=sigma_y, absolute_sigma=True)
    else:
        sigma_y = None
        popt, pcov = curve_fit(_linear, inv_T, ln_kT)

    slope, intercept      = popt
    slope_std, intcpt_std = np.sqrt(np.diag(pcov))

    dH_kJ     = -slope * R / 1000.0
    dH_std_kJ = slope_std * R / 1000.0
    dS_J      = (intercept - LN_KB_H) * R
    dS_std_J  = intcpt_std * R

    ln_kT_pred = _linear(inv_T, slope, intercept)
    ss_res     = np.sum((ln_kT - ln_kT_pred) ** 2)
    ss_tot     = np.sum((ln_kT - ln_kT.mean()) ** 2)
    r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_data = 1000.0 * inv_T
    print(f"\nEyring ({len(entries)} temperatures, "
          f"{'weighted' if use_weights else 'unweighted'}):")
    print(f"  ΔH‡ = {dH_kJ:.2f} ± {dH_std_kJ:.2f} kJ mol⁻¹")
    print(f"  ΔS‡ = {dS_J:.2f} ± {dS_std_J:.2f} J mol⁻¹ K⁻¹")
    print(f"  R²  = {r2:.6f}")

    return EyringResult(
        compound=compound_name,
        dH_kJmol=dH_kJ,
        dH_std_kJmol=dH_std_kJ,
        dS_JmolK=dS_J,
        dS_std_JmolK=dS_std_J,
        r2=r2,
        n_temperatures=len(entries),
        weighted=use_weights,
        x_data=x_data,
        ln_kT=ln_kT,
        ln_kT_pred=_linear(x_data / 1000.0, slope, intercept),
        sigma_y=sigma_y,
        T_C_list=T_C_list,
        k_mean_list=list(k_arr),
    )


def plot_eyring(result: EyringResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    if result.sigma_y is not None:
        ax.errorbar(result.x_data, result.ln_kT, yerr=result.sigma_y,
                    fmt="o", color="black", markerfacecolor="none",
                    markeredgewidth=1.2, capsize=4, elinewidth=1.2,
                    label="Data ± σ")
    else:
        ax.plot(result.x_data, result.ln_kT,
                "o", color="black", markerfacecolor="none",
                markeredgewidth=1.2, label="Data")

    x_fit = np.linspace(result.x_data.min() * 0.998,
                        result.x_data.max() * 1.002, 200)
    slope_r     = -(result.dH_kJmol * 1000.0) / R
    intercept_r = result.dS_JmolK / R + LN_KB_H
    ax.plot(x_fit, _linear(x_fit / 1000.0, slope_r, intercept_r),
            "--", color="red", linewidth=2, label="Eyring fit")

    annotation = (
        r"$\ln\!\left(\frac{k}{T}\right)="
        r"-\frac{\Delta H^\ddagger}{R}\cdot\frac{1}{T}"
        r"+\frac{\Delta S^\ddagger}{R}+\ln\frac{k_\mathrm{B}}{h}$" + "\n"
        f"$\\Delta H^\\ddagger$ = {result.dH_kJmol:.2f} ± "
        f"{result.dH_std_kJmol:.2f} kJ mol$^{{-1}}$\n"
        f"$\\Delta S^\\ddagger$ = {result.dS_JmolK:.2f} ± "
        f"{result.dS_std_JmolK:.2f} J mol$^{{-1}}$ K$^{{-1}}$\n"
        f"$R^2$ = {result.r2:.4f}"
    )
    ax.text(0.97, 0.97, annotation,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$1000\,/\,T\ \mathrm{(K^{-1})}$")
    ax.set_ylabel(r"$\ln(k\,/\,T)$")
    ax.set_title(f"Eyring Plot — {result.compound}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
