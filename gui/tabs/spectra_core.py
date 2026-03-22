"""
Core functions for the Spectra tab.

Spectral extraction of species B from:
  negative     – reference-wavelength subtraction
  negative_pca – PCA non-negativity extrapolation (bleaching system)
  positive_pca – PCA non-negativity extrapolation (build-up system)
  positive_pss – known PSS conversion fraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_spectra_csv(filepath: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load a Cary 60 multi-scan CSV (column pairs: Wavelength, Abs).
    Skips the two header rows; ignores non-numeric trailing metadata.
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


def interpolate_to_grid(wl: np.ndarray, ab: np.ndarray,
                        grid: np.ndarray) -> np.ndarray:
    """Interpolate a single spectrum onto an integer-nm wavelength grid."""
    if wl[0] > wl[-1]:
        wl, ab = wl[::-1], ab[::-1]
    return np.interp(grid, wl, ab)


# ── Reference / alpha helpers ────────────────────────────────────────────────

def resolve_reference(
    reference_wavelength_nm,
    grid: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Convert reference_wavelength_nm (int/float, (min,max) tuple, or list)
    into grid indices and a description string.
    """
    ref = reference_wavelength_nm
    if isinstance(ref, (int, float)):
        idx  = np.array([np.argmin(np.abs(grid - ref))])
        desc = f"{ref} nm"
    elif isinstance(ref, tuple) and len(ref) == 2:
        mask = (grid >= ref[0]) & (grid <= ref[1])
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"No grid points in band {ref[0]}–{ref[1]} nm.")
        desc = f"band {ref[0]}–{ref[1]} nm ({len(idx)} points)"
    else:
        idx  = np.array([np.argmin(np.abs(grid - wl)) for wl in ref])
        desc = f"{list(ref)} nm ({len(idx)} points)"
    return idx, desc


def ref_absorbance(spectrum: np.ndarray, ref_indices: np.ndarray) -> float:
    return float(spectrum[ref_indices].mean())


def compute_alpha(
    S_i: np.ndarray,
    S_A: np.ndarray,
    ref_indices: np.ndarray,
    weighted: bool,
) -> float:
    s_i = S_i[ref_indices]
    s_a = S_A[ref_indices]
    if weighted:
        denom = float(np.dot(s_a, s_a))
        return float(np.dot(s_a, s_i) / denom) if denom > 0 else np.nan
    else:
        m_a = float(s_a.mean())
        return float(s_i.mean() / m_a) if m_a != 0 else np.nan


# ── File-list loaders (GUI versions — take file paths, not directories) ───────

def load_and_average_files(
    file_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load all scans from *file_paths*, average onto a common integer-nm grid.
    Returns (grid, S_A, n_scans).
    """
    all_scans: list[tuple[np.ndarray, np.ndarray]] = []
    wl_min, wl_max = np.inf, -np.inf

    for fp in file_paths:
        for wl, ab in load_spectra_csv(fp):
            wl_min = min(wl_min, float(wl.min()))
            wl_max = max(wl_max, float(wl.max()))
            all_scans.append((wl, ab))

    if not all_scans:
        raise ValueError("No valid scans found in initial spectrum files.")

    grid    = np.arange(int(np.ceil(wl_min)), int(np.floor(wl_max)) + 1)
    spectra = np.array([interpolate_to_grid(wl, ab, grid)
                        for wl, ab in all_scans])
    return grid, spectra.mean(axis=0), len(all_scans)


def load_irradiation_series_files(
    file_paths: list[Path],
    grid: np.ndarray,
) -> np.ndarray:
    """
    Load all scans from *file_paths* in file-name order, interpolated onto *grid*.
    Returns array of shape (n_spectra, n_wavelengths).
    """
    series = []
    for fp in sorted(file_paths):
        for wl, ab in load_spectra_csv(fp):
            series.append(interpolate_to_grid(wl, ab, grid))
    if not series:
        raise ValueError("No valid scans found in irradiation series files.")
    return np.array(series)


def load_pss_files(
    file_paths: list[Path],
    grid: np.ndarray,
) -> np.ndarray:
    """Average all PSS scans onto *grid*. Returns S_PSS (1-D)."""
    scans = []
    for fp in file_paths:
        for wl, ab in load_spectra_csv(fp):
            scans.append(interpolate_to_grid(wl, ab, grid))
    if not scans:
        raise ValueError("No valid scans found in PSS files.")
    return np.array(scans).mean(axis=0)


# ── Parameter and result containers ──────────────────────────────────────────

@dataclass
class SpectraParams:
    mode:                   str   = "negative"   # negative|negative_pca|positive_pca|positive_pss
    compound_name:          str   = ""
    path_length_cm:         float = 1.0
    concentration_mol_L:    Optional[float] = None   # None → absorbance output
    baseline_offset:        float = 0.0
    baseline_inset_nm:      int   = 50
    # negative mode
    reference_wavelength_nm: object = 500.0  # float, (lo, hi) tuple, or [list]
    reference_weighted:     bool  = True
    min_alpha:              float = 0.2
    max_alpha:              float = 0.6
    exclude_negative_SB:    bool  = True
    sb_tolerance_sigma:     float = 3.0
    # PCA modes
    n_bootstrap:            int   = 2000
    # positive_pss mode
    pss_fraction_B:         float = 0.85
    pss_fraction_B_error:   float = 0.02
    # spectrum selection (None = all, tuple (start,stop), or list)
    spectrum_indices:       object = None


@dataclass
class SpectraResult:
    compound:        str
    mode:            str
    grid:            np.ndarray
    S_A:             np.ndarray
    S_B:             np.ndarray
    S_B_std:         np.ndarray
    S_B_lo:          np.ndarray
    S_B_hi:          np.ndarray
    n_spectra_used:  int
    n_spectra_total: int
    # extra per-mode diagnostics
    alphas:          Optional[np.ndarray]   = None   # α for each used spectrum (neg/neg_pca/pos_pca)
    scale:           Optional[float]        = None   # PCA scale factor
    var_explained:   Optional[float]        = None   # PCA
    n_boot_rejected: int                    = 0
    # annotation
    meta_lines:      list[str]              = field(default_factory=list)
    # series data for plots
    series_used:     Optional[np.ndarray]   = None


# ── Main extraction ───────────────────────────────────────────────────────────

def run_spectra_extraction(
    params: SpectraParams,
    grid:   np.ndarray,
    S_A_in: np.ndarray,
    series_in: np.ndarray,
    S_PSS_in: Optional[np.ndarray] = None,
) -> SpectraResult:
    """
    Run spectral extraction from pre-loaded data.

    Parameters
    ----------
    params      : SpectraParams dataclass
    grid        : integer-nm wavelength grid (1-D)
    S_A_in      : initial (species A) spectrum interpolated onto grid
    series_in   : irradiation series, shape (n, len(grid))
    S_PSS_in    : PSS spectrum (positive_pss mode only)
    """
    # ── copy inputs + apply concentration factor ────────────────────────────
    S_A   = S_A_in.copy()
    series = series_in.copy()
    S_PSS  = S_PSS_in.copy() if S_PSS_in is not None else None

    if params.concentration_mol_L is not None:
        factor = 1.0 / (params.concentration_mol_L * params.path_length_cm)
        S_A    = S_A    * factor
        series  = series  * factor
        if S_PSS is not None:
            S_PSS = S_PSS * factor
        print(f"Converted to ε (factor = {factor:.4e} L mol⁻¹ cm⁻¹ / AU)")

    # ── baseline offset ─────────────────────────────────────────────────────
    if params.baseline_offset != 0.0:
        S_A    = S_A    + params.baseline_offset
        series  = series  + params.baseline_offset
        if S_PSS is not None:
            S_PSS = S_PSS + params.baseline_offset
        print(f"Baseline offset {params.baseline_offset:+.6f} applied.")

    # ── baseline noise ──────────────────────────────────────────────────────
    bl_mask       = grid >= (grid[-1] - params.baseline_inset_nm)
    baseline_noise = float(S_A[bl_mask].std()) if bl_mask.any() else 1e-6
    sb_tolerance   = params.sb_tolerance_sigma * baseline_noise
    print(f"Baseline noise σ = {baseline_noise:.6f}  →  tolerance = {sb_tolerance:.6f}")

    # ── spectrum index selection ────────────────────────────────────────────
    n_total = len(series)
    si = params.spectrum_indices
    if si is None:
        selected_idx = list(range(n_total))
    elif isinstance(si, tuple) and len(si) == 2:
        selected_idx = list(range(*slice(si[0], si[1]).indices(n_total)))
    else:
        selected_idx = [int(i) for i in si]

    print(f"Loaded {n_total} irradiation spectra; "
          f"{len(selected_idx)} selected by index filter.")

    # ── mode dispatch ────────────────────────────────────────────────────────
    mode = params.mode
    S_B = S_B_std = S_B_lo = S_B_hi = None
    alphas = None
    scale = var_explained = None
    n_boot_rejected = 0
    series_used = series[selected_idx]

    # ── NEGATIVE ─────────────────────────────────────────────────────────────
    if mode == "negative":
        ref_indices, ref_desc = resolve_reference(
            params.reference_wavelength_nm, grid)
        A0_ref = ref_absorbance(S_A, ref_indices)
        if A0_ref <= 0:
            raise ValueError(
                f"Initial absorbance at reference ({ref_desc}) ≤ 0.")

        alpha_lo = max(0.0, params.min_alpha)
        alpha_hi = min(1.0, params.max_alpha)

        used_idx = [i for i in selected_idx
                    if alpha_lo < ref_absorbance(series[i], ref_indices) / A0_ref < alpha_hi]
        alpha_excl = len(selected_idx) - len(used_idx)
        print(f"α filter [{alpha_lo:.2f}, {alpha_hi:.2f}]: "
              f"{len(used_idx)} used, {alpha_excl} excluded.")

        if len(used_idx) < 2:
            raise ValueError(
                "Fewer than 2 spectra pass the α filter. "
                "Widen the min_alpha / max_alpha range.")

        series_used = series[used_idx]
        B_estimates = []
        alphas_list = []
        n_neg_rejected = 0
        for S_i in series_used:
            a = compute_alpha(S_i, S_A, ref_indices, params.reference_weighted)
            if np.isnan(a) or a <= 0 or a >= 1:
                continue
            B_i = (S_i - a * S_A) / (1.0 - a)
            if params.exclude_negative_SB and np.any(B_i < -sb_tolerance):
                n_neg_rejected += 1
                continue
            B_estimates.append(B_i)
            alphas_list.append(a)

        if n_neg_rejected:
            print(f"  {n_neg_rejected} spectra rejected (S_B < −tolerance).")

        if len(B_estimates) < 2:
            raise ValueError("Fewer than 2 valid spectra after negative-SB filter.")

        B_arr    = np.array(B_estimates)
        S_B      = B_arr.mean(axis=0)
        S_B_std  = B_arr.std(axis=0, ddof=1)
        S_B_lo   = S_B - S_B_std
        S_B_hi   = S_B + S_B_std
        alphas   = np.array(alphas_list)
        print(f"Negative mode: used {len(B_estimates)} spectra, "
              f"α range {alphas.min():.3f}–{alphas.max():.3f}")

    # ── PCA (negative_pca / positive_pca) ─────────────────────────────────
    elif mode in ("negative_pca", "positive_pca"):
        D = series_used - S_A
        U, sv, Vt = np.linalg.svd(D, full_matrices=False)
        PC1    = Vt[0]
        scores = U[:, 0] * sv[0]
        if scores[-1] < scores[0]:
            PC1    = -PC1
            scores = -scores

        neg_mask    = PC1 < 0
        constrained = neg_mask & (S_A + sb_tolerance > 0)
        if constrained.any():
            scale_nn = float(
                np.min((S_A[constrained] + sb_tolerance) / (-PC1[constrained])))
        else:
            scale_nn = np.inf

        scale_lower = float(scores.max())
        if scale_nn >= scale_lower:
            scale = scale_nn
        else:
            print(f"Warning: non-negativity limit ({scale_nn:.4f}) < score max "
                  f"({scale_lower:.4f}). Falling back to minimum scale.")
            scale = scale_lower

        S_B = S_A + scale * PC1

        alphas_pca = 1.0 - scores / scale
        alphas     = alphas_pca

        D_approx = np.outer(scores, PC1)
        residuals = D - D_approx
        S_B_std   = residuals.std(axis=0, ddof=1)

        var_explained = float(sv[0] ** 2 / np.sum(sv ** 2))

        # Bootstrap CI
        boot_SB_raw, boot_scales_raw = [], []
        rng = np.random.default_rng(42)
        for _ in range(params.n_bootstrap):
            idx_b = rng.integers(0, len(D), len(D))
            D_b   = D[idx_b]
            _, _, Vt_b = np.linalg.svd(D_b, full_matrices=False)
            pc1_b  = Vt_b[0]
            if pc1_b @ PC1 < 0:
                pc1_b = -pc1_b
            constr_b = (pc1_b < 0) & (S_A + sb_tolerance > 0)
            sc_nn_b  = float(np.min(
                (S_A[constr_b] + sb_tolerance) / (-pc1_b[constr_b])
            )) if constr_b.any() else np.inf
            sc_lower_b = float((D_b @ pc1_b).max())
            sc_b = sc_nn_b if sc_nn_b >= sc_lower_b else sc_lower_b
            boot_SB_raw.append(S_A + sc_b * pc1_b)
            boot_scales_raw.append(sc_b)

        boot_pairs = [(sb, sc) for sb, sc in zip(boot_SB_raw, boot_scales_raw)
                      if not np.any(sb < -sb_tolerance)]
        n_boot_rejected = params.n_bootstrap - len(boot_pairs)
        if n_boot_rejected:
            print(f"Bootstrap: {n_boot_rejected}/{params.n_bootstrap} "
                  f"resamples rejected (negative S_B).")
        if len(boot_pairs) < 10:
            boot_pairs = list(zip(boot_SB_raw, boot_scales_raw))

        boot_SB_arr = np.array([p[0] for p in boot_pairs])
        S_B_lo = np.percentile(boot_SB_arr,  2.5, axis=0)
        S_B_hi = np.percentile(boot_SB_arr, 97.5, axis=0)

        print(f"PCA ({mode}): PC1 explains {var_explained*100:.1f}% of variance, "
              f"scale = {scale:.4f}, α range {alphas.min():.3f}–{alphas.max():.3f}")

    # ── POSITIVE PSS ─────────────────────────────────────────────────────────
    elif mode == "positive_pss":
        if S_PSS is None:
            raise ValueError(
                "positive_pss mode requires PSS spectrum files to be loaded.")
        f_B     = params.pss_fraction_B
        f_B_err = params.pss_fraction_B_error
        S_B     = (S_PSS - (1.0 - f_B) * S_A) / f_B
        S_B_lo  = (S_PSS - (1.0 - (f_B + f_B_err)) * S_A) / (f_B + f_B_err)
        S_B_hi  = (S_PSS - (1.0 - (f_B - f_B_err)) * S_A) / (f_B - f_B_err)
        S_B_std = (S_B_hi - S_B_lo) / 2.0
        print(f"PSS mode: f_B = {f_B} ± {f_B_err}")
    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    # ── Metadata lines ────────────────────────────────────────────────────────
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = [
        f"Date             : {ts}",
        f"Compound         : {params.compound_name}",
        f"Mode             : {mode}",
        f"Wavelength range : {grid[0]}–{grid[-1]} nm",
        f"Spectra used     : {len(series_used)} of {n_total} loaded",
    ]
    if params.baseline_offset != 0.0:
        meta.append(f"Baseline offset  : {params.baseline_offset:+.6f}")
    if mode == "negative" and alphas is not None:
        meta += [
            f"α window         : [{params.min_alpha:.2f}, {params.max_alpha:.2f}]",
            f"α range (used)   : {alphas.min():.3f}–{alphas.max():.3f}",
            f"Reference        : {params.reference_wavelength_nm} nm",
        ]
    elif mode in ("negative_pca", "positive_pca") and alphas is not None:
        meta += [
            f"PC1 variance     : {var_explained*100:.1f}%",
            f"Scale (nn extrap): {scale:.4f}",
            f"α range          : {alphas.min():.3f}–{alphas.max():.3f}",
            f"Bootstrap        : {params.n_bootstrap} ({n_boot_rejected} rejected)",
        ]
    elif mode == "positive_pss":
        meta += [
            f"f_B              : {params.pss_fraction_B} ± {params.pss_fraction_B_error}",
        ]

    return SpectraResult(
        compound        = params.compound_name,
        mode            = mode,
        grid            = grid,
        S_A             = S_A,
        S_B             = S_B,
        S_B_std         = S_B_std,
        S_B_lo          = S_B_lo,
        S_B_hi          = S_B_hi,
        n_spectra_used  = len(series_used),
        n_spectra_total = n_total,
        alphas          = alphas,
        scale           = scale,
        var_explained   = var_explained,
        n_boot_rejected = n_boot_rejected,
        meta_lines      = meta,
        series_used     = series_used,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_overview(
    grid:         np.ndarray,
    S_A:          np.ndarray,
    series:       np.ndarray,
    S_PSS:        Optional[np.ndarray],
    mode:         str,
    reference_wavelength_nm=None,
    baseline_inset_nm: int = 50,
    compound_name: str = "",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = mcm.coolwarm
    n    = len(series)
    for j, s in enumerate(series):
        c = cmap(j / max(n - 1, 1))
        ax.plot(grid, s, color=c, linewidth=0.7, alpha=0.6,
                label="Irradiation (blue=early, red=late)" if j == 0 else None)

    ax.plot(grid, S_A, color="steelblue", linewidth=2, label="Species A (initial)")
    if S_PSS is not None:
        ax.plot(grid, S_PSS, color="mediumseagreen", linewidth=2, label="PSS spectrum")

    # Reference overlay for negative mode
    if mode == "negative" and reference_wavelength_nm is not None:
        ref = reference_wavelength_nm
        if isinstance(ref, (int, float)):
            ax.axvline(ref, color="red", linewidth=1.5, linestyle="--",
                       label=f"Reference: {ref} nm")
        elif isinstance(ref, tuple) and len(ref) == 2:
            ax.axvspan(ref[0], ref[1], color="red", alpha=0.12,
                       label=f"Reference band: {ref[0]}–{ref[1]} nm")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    sm = mcm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, max(n - 1, 1)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Spectrum index (early → late)",
                 fraction=0.03, pad=0.01)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Data Overview — {compound_name} | {n} irradiation spectra")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Baseline inset
    inset_wl_min = grid[-1] - baseline_inset_nm
    inset_mask   = grid >= inset_wl_min
    ax_in = ax.inset_axes([0.60, 0.55, 0.35, 0.38])
    for j, s in enumerate(series):
        c = cmap(j / max(n - 1, 1))
        ax_in.plot(grid[inset_mask], s[inset_mask], color=c,
                   linewidth=0.7, alpha=0.6)
    ax_in.plot(grid[inset_mask], S_A[inset_mask],
               color="steelblue", linewidth=1.5)
    if S_PSS is not None:
        ax_in.plot(grid[inset_mask], S_PSS[inset_mask],
                   color="mediumseagreen", linewidth=1.5)
    ax_in.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_in.set_title(f"Last {baseline_inset_nm} nm", fontsize=8)
    ax_in.tick_params(labelsize=7)
    ax_in.grid(True, linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_extraction_result(result: SpectraResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    if result.series_used is not None:
        n_show = min(5, len(result.series_used))
        step   = max(1, len(result.series_used) // n_show)
        shown  = result.series_used[::step]
        for j, s in enumerate(shown):
            ax.plot(result.grid, s, color="grey", linewidth=0.8, alpha=0.4,
                    label=f"Irradiation ({len(shown)} of "
                          f"{len(result.series_used)} shown)" if j == 0 else None)

    ax.plot(result.grid, result.S_A, color="steelblue", linewidth=2,
            label="Species A (initial)")
    ax.plot(result.grid, result.S_B, color="darkorange", linewidth=2,
            label="Species B (extracted)")
    ax.fill_between(result.grid, result.S_B_lo, result.S_B_hi,
                    color="darkorange", alpha=0.25, label="±1σ / 95% CI")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Spectral Extraction — {result.compound} | mode: {result.mode}")
    ax.legend(fontsize=8)
    ax.grid(True)

    if result.meta_lines:
        ax.text(0.01, 0.99, "\n".join(result.meta_lines),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))

    plt.tight_layout()
    return fig
