from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Setup directories
# -------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "spectra_calculation" / "raw"
INITIAL_DIR    = RAW_DIR / "initial"
IRRAD_DIR      = RAW_DIR / "irradiation"
PSS_DIR        = RAW_DIR / "pss"
RESULTS_DIR    = BASE_DIR / "data" / "spectra_calculation" / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
EXTRACTED_DIR  = RESULTS_DIR / "extracted"

for folder in [INITIAL_DIR, IRRAD_DIR, PSS_DIR, RESULTS_DIR, PLOTS_DIR, EXTRACTED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
mode                    = "positive_pca"  # "negative" | "negative_pca" | "positive_pca" | "positive_pss"
compound_name           = "DAE_150"    # used in plot titles and saved output

# Mode: negative — reference where only species A absorbs.
# Single wavelength : reference_wavelength_nm = 672
# Band (mean over range) : reference_wavelength_nm = (660, 690)
# Specific wavelengths : reference_wavelength_nm = [650, 672, 685]
reference_wavelength_nm = (650, 680)

# Weighting of the reference band when estimating α (negative mode only).
# False : α = mean(S_i[band]) / mean(S_A[band])          — equal weight per wavelength
# True  : α = Σ(S_A·S_i) / Σ(S_A²)                      — least-squares fit weighted
#             by S_A(λ), so high-absorption wavelengths contribute more
reference_weighted      = True

# Mode: positive_pss — known fraction of B in the PSS spectrum
pss_fraction_B          = 0.85        # fraction of B at PSS
pss_fraction_B_error    = 0.02        # ± uncertainty in that fraction

# Spectrum selection — applied before extraction
# None  = use all loaded irradiation spectra
# (start, stop) = use series[start:stop]  (0-based, stop exclusive)
# [i, j, ...]   = use these specific indices
spectrum_indices        = None

# Negative mode: α = A_ref_i / A_ref_0 acceptance window.
#
# Large α (α → 1): barely converted — denominator (1−α) is tiny → noise amplified hugely.
# Small α (α → 0): well converted — denominator ≈ 1, noise minimal; but reference
#                  absorbance may approach the noise floor at very low α.
#
# Recommended: exclude early (high-α) spectra with max_alpha and optionally
# exclude noise-floor (very low-α) spectra with min_alpha.
min_alpha               = 0.2   # exclude spectra with α < this (noise floor guard)
max_alpha               = 0.6   # exclude spectra with α > this (noise amplification guard)

# Negative mode: discard any individual S_B_i estimate that contains negative
# absorbance values anywhere in the spectrum.  These arise from inconsistent
# spectra (e.g. UV band moving in the wrong direction) and would bias the mean.
exclude_negative_SB     = True

# How far below zero S_B is permitted to go before a spectrum/resample is rejected.
# Expressed as a multiple of the baseline noise (std of S_A in the long-wavelength
# baseline region, measured after any offset correction).
# Set to 0 for strict non-negativity; 3 (default) allows ±3σ baseline scatter.
sb_tolerance_sigma      = 3.0

# Shared
path_length_cm          = 1.0
concentration_mol_L     = None        # if set → output as ε (L mol⁻¹ cm⁻¹); else absorbance
wavelength_tolerance_nm = 1           # nm — tolerance when extracting reference absorbance
n_bootstrap             = 2000        # bootstrap iterations for PCA confidence interval

# Baseline inset: shows the last N nm of the spectrum in the overview plot
# to make any constant offset from zero immediately visible.
baseline_inset_nm       = 50          # width of the inset zoom window (nm)

# Diagnostic plot (negative mode): show all individual S_B_i estimates coloured by α.
# Set to False to skip it.
show_diagnostic         = True

# Convergence diagnostic: re-run extraction on growing subsets of the irradiation series
# and overlay the resulting S_B curves. Reveals whether more irradiation time changes
# the result — if curves converge, the extraction is stable; if they keep shifting,
# either conversion is incomplete or scale determination is unreliable.
# Each entry in convergence_fractions is the fraction of series_used to include,
# so [0.25, 0.5, 0.75, 1.0] means 4 runs: first 25%, 50%, 75%, and all spectra.
# Requires at least 3 spectra per subset; subsets with fewer are skipped.
show_convergence        = True
convergence_fractions   = [0.25, 0.5, 0.75, 1.0]

# -------------------------
# CSV loading helper (Cary 60 column-pair format)
# -------------------------
def load_spectra_csv(filepath):
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


def interpolate_to_grid(wl, ab, grid):
    """Interpolate a single spectrum onto an integer-nm wavelength grid."""
    if wl[0] > wl[-1]:          # descending — flip to ascending for np.interp
        wl, ab = wl[::-1], ab[::-1]
    return np.interp(grid, wl, ab)


def resolve_reference(reference_wavelength_nm, grid):
    """
    Convert reference_wavelength_nm (int/float, (min,max) tuple, or list)
    into an array of grid indices.  Returns (ref_indices, description_string).
    """
    ref = reference_wavelength_nm
    if isinstance(ref, (int, float)):
        idx = np.array([np.argmin(np.abs(grid - ref))])
        desc = f"{ref} nm"
    elif isinstance(ref, tuple) and len(ref) == 2:
        mask = (grid >= ref[0]) & (grid <= ref[1])
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"No grid points found in band {ref[0]}–{ref[1]} nm.")
        desc = f"band {ref[0]}–{ref[1]} nm ({len(idx)} points)"
    else:
        idx  = np.array([np.argmin(np.abs(grid - wl)) for wl in ref])
        desc = f"{list(ref)} nm ({len(idx)} points)"
    return idx, desc


def ref_absorbance(spectrum, ref_indices):
    """Unweighted mean absorbance at the reference indices (used for α filter threshold)."""
    return spectrum[ref_indices].mean()


def compute_alpha(S_i, S_A, ref_indices, weighted):
    """
    Estimate α = fraction of species A remaining from the reference band.

    weighted=False : α = mean(S_i[ref]) / mean(S_A[ref])
                     Equal weight per wavelength.
    weighted=True  : α = Σ(S_A[ref] · S_i[ref]) / Σ(S_A[ref]²)
                     Least-squares fit weighted by S_A — high-absorption
                     wavelengths contribute more (better SNR).
    """
    s_i = S_i[ref_indices]
    s_a = S_A[ref_indices]
    if weighted:
        denom = np.dot(s_a, s_a)
        return np.dot(s_a, s_i) / denom if denom > 0 else np.nan
    else:
        return s_i.mean() / s_a.mean()


def load_and_average(folder):
    """
    Load all CSVs from a folder, average all scans across all files
    onto a common grid. Returns (grid, mean_spectrum).
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}.")

    all_scans = []
    wl_min, wl_max = np.inf, -np.inf

    for f in csv_files:
        for wl, ab in load_spectra_csv(f):
            wl_min = min(wl_min, wl.min())
            wl_max = max(wl_max, wl.max())
            all_scans.append((wl, ab))

    grid = np.arange(int(np.ceil(wl_min)), int(np.floor(wl_max)) + 1)
    spectra = np.array([interpolate_to_grid(wl, ab, grid) for wl, ab in all_scans])
    return grid, spectra.mean(axis=0), spectra


def load_irradiation_series(folder, grid):
    """
    Load all scans from all CSVs in folder, interpolated onto grid.
    Returns array of shape (n_spectra, n_wavelengths).
    Preserves scan order (sorted by filename then column order).
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}.")

    series = []
    for f in csv_files:
        for wl, ab in load_spectra_csv(f):
            series.append(interpolate_to_grid(wl, ab, grid))

    return np.array(series)


# -------------------------
# Load initial spectrum
# -------------------------
print("Loading initial spectrum...")
grid, S_A, initial_scans = load_and_average(INITIAL_DIR)
print(f"  {len(initial_scans)} scan(s), wavelength range: {grid[0]}–{grid[-1]} nm")

# -------------------------
# Load irradiation series
# -------------------------
print("Loading irradiation series...")
series = load_irradiation_series(IRRAD_DIR, grid)
print(f"  {len(series)} spectra loaded.")

# -------------------------
# Optionally convert to ε
# -------------------------
if concentration_mol_L is not None:
    factor = 1.0 / (concentration_mol_L * path_length_cm)
    S_A   = S_A   * factor
    series = series * factor
    y_label = r"$\varepsilon$ (L mol$^{-1}$ cm$^{-1}$)"
else:
    y_label = "Absorbance"

# -------------------------
# Apply spectrum_indices selection
# -------------------------
n_total = len(series)
if spectrum_indices is None:
    selected_idx = list(range(n_total))
elif isinstance(spectrum_indices, tuple) and len(spectrum_indices) == 2:
    start, stop = spectrum_indices
    selected_idx = list(range(*slice(start, stop).indices(n_total)))
else:
    selected_idx = list(spectrum_indices)

range_excluded_idx = [i for i in range(n_total) if i not in selected_idx]

# ============================================================
# PREVIEW — two plots, each with a continue prompt
# ============================================================

# --- Load PSS for preview (if present) ---
pss_files_preview = sorted(PSS_DIR.glob("*.csv"))
S_PSS_preview = None
if pss_files_preview:
    pss_scans_preview = []
    for f in pss_files_preview:
        for wl, ab in load_spectra_csv(f):
            pss_scans_preview.append(interpolate_to_grid(wl, ab, grid))
    if pss_scans_preview:
        S_PSS_preview = np.array(pss_scans_preview).mean(axis=0)

def _add_ref_overlay(ax):
    """Add reference wavelength / band shading (negative mode only)."""
    if mode != "negative":
        return
    ref = reference_wavelength_nm
    if isinstance(ref, (int, float)):
        ax.axvline(ref, color="red", linewidth=1.5, linestyle="--",
                   label=f"Reference: {ref} nm")
    elif isinstance(ref, tuple) and len(ref) == 2:
        ax.axvspan(ref[0], ref[1], color="red", alpha=0.12,
                   label=f"Reference band: {ref[0]}–{ref[1]} nm")
        ax.axvline(ref[0], color="red", linewidth=0.8, linestyle=":")
        ax.axvline(ref[1], color="red", linewidth=0.8, linestyle=":")
    else:
        for k, wl in enumerate(ref):
            ax.axvline(wl, color="red", linewidth=1.2, linestyle="--",
                       label=f"Reference: {list(ref)} nm" if k == 0 else None)

# --- Plot 1: raw data overview with baseline inset ---
# Irradiation spectra are coloured early→late using a blue→red gradient
# so the direction of change is immediately visible.
fig1, ax1 = plt.subplots(figsize=(9, 5))
_cmap_seq = plt.cm.coolwarm
_n_irr    = len(series)
for j, s in enumerate(series):
    _c = _cmap_seq(j / max(_n_irr - 1, 1))
    ax1.plot(grid, s, color=_c, linewidth=0.7, alpha=0.6,
             label="Irradiation spectra (blue=early, red=late)" if j == 0 else None)
ax1.plot(grid, S_A, color="steelblue", linewidth=2, label="Species A (initial)")
if S_PSS_preview is not None:
    ax1.plot(grid, S_PSS_preview, color="mediumseagreen", linewidth=2, label="PSS spectrum")
_add_ref_overlay(ax1)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
# Colorbar to show irradiation progress
_sm1 = plt.cm.ScalarMappable(cmap=_cmap_seq, norm=plt.Normalize(vmin=0, vmax=_n_irr - 1))
_sm1.set_array([])
plt.colorbar(_sm1, ax=ax1, label="Spectrum index (early → late)", fraction=0.03, pad=0.01)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel(y_label)
ax1.set_title(f"Data Overview — {compound_name} | {_n_irr} irradiation spectra")
ax1.legend()
ax1.grid(True)

# Inset: zoom into the last baseline_inset_nm of the spectrum (long-wavelength end)
inset_wl_min = grid[-1] - baseline_inset_nm
inset_mask   = grid >= inset_wl_min
ax_in = ax1.inset_axes([0.60, 0.55, 0.35, 0.38])
for j, s in enumerate(series):
    _c = _cmap_seq(j / max(_n_irr - 1, 1))
    ax_in.plot(grid[inset_mask], s[inset_mask], color=_c, linewidth=0.7, alpha=0.6)
ax_in.plot(grid[inset_mask], S_A[inset_mask], color="steelblue", linewidth=1.5)
if S_PSS_preview is not None:
    ax_in.plot(grid[inset_mask], S_PSS_preview[inset_mask], color="mediumseagreen", linewidth=1.5)
ax_in.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
ax_in.set_title(f"Last {baseline_inset_nm} nm", fontsize=8)
ax_in.tick_params(labelsize=7)
ax_in.grid(True, linewidth=0.5)

plt.tight_layout()
print("  [Close the plot window to continue]")
plt.show()

if input("Data looks OK? Continue? (y/n): ").strip().lower() != "y":
    plt.close(fig1)
    raise SystemExit(0)
plt.close(fig1)

# --- Baseline offset ---
_offset_raw = input(
    "Apply baseline offset? Enter value to add to all spectra (0 = skip, e.g. -0.005): "
).strip()
try:
    baseline_offset = float(_offset_raw)
except ValueError:
    baseline_offset = 0.0
if baseline_offset != 0.0:
    S_A    = S_A    + baseline_offset
    series = series + baseline_offset
    if S_PSS_preview is not None:
        S_PSS_preview = S_PSS_preview + baseline_offset
    print(f"  Baseline offset {baseline_offset:+.6f} applied to all spectra.")

# Baseline noise: std of S_A in the long-wavelength tail (after any offset correction).
# Used as the reference for sb_tolerance_sigma — how far below zero S_B may go.
_bl_mask      = grid >= (grid[-1] - baseline_inset_nm)
baseline_noise = float(S_A[_bl_mask].std()) if _bl_mask.any() else 1e-6
sb_tolerance   = sb_tolerance_sigma * baseline_noise
print(f"  Baseline noise (σ of S_A in last {baseline_inset_nm} nm): {baseline_noise:.6f}  "
      f"→  S_B tolerance: {sb_tolerance:.6f}  ({sb_tolerance_sigma}σ)")

# -------------------------
# Apply alpha filter (negative mode only) — computed after any offset
# -------------------------
alpha_excluded_idx = []
used_idx           = selected_idx

if mode == "negative":
    alpha_lo    = max(0.0, min_alpha) if min_alpha is not None else 0.0
    alpha_hi    = min(1.0, max_alpha) if max_alpha is not None else 1.0
    ref_indices, ref_desc = resolve_reference(reference_wavelength_nm, grid)
    A0_ref      = ref_absorbance(S_A, ref_indices)
    if A0_ref > 0:
        alpha_excluded_idx = [i for i in selected_idx
                              if not (alpha_lo < ref_absorbance(series[i], ref_indices) / A0_ref < alpha_hi)]
        used_idx           = [i for i in selected_idx
                              if     (alpha_lo < ref_absorbance(series[i], ref_indices) / A0_ref < alpha_hi)]

series_used = series[used_idx]
if mode == "negative":
    print(f"  Spectrum selection: {len(used_idx)} of {n_total} spectra will be used "
          f"({len(range_excluded_idx)} excluded by range, "
          f"{len(alpha_excluded_idx)} excluded by α filter).")
else:
    print(f"  Spectrum selection: {len(used_idx)} of {n_total} spectra will be used "
          f"({len(range_excluded_idx)} excluded by range).")

# --- Plot 2: selection preview ---
fig2, ax2 = plt.subplots(figsize=(9, 5))

for j, i in enumerate(range_excluded_idx):
    ax2.plot(grid, series[i], color="salmon", linewidth=0.6, alpha=0.3,
             label="Excluded (range)" if j == 0 else None)
for j, i in enumerate(alpha_excluded_idx):
    ax2.plot(grid, series[i], color="darkorange", linewidth=0.7, alpha=0.4,
             label=f"Excluded (α outside [{min_alpha}, {max_alpha}])" if j == 0 else None)
for j, i in enumerate(used_idx):
    ax2.plot(grid, series[i], color="grey", linewidth=0.7, alpha=0.5,
             label=f"Used ({len(used_idx)} spectra)" if j == 0 else None)

ax2.plot(grid, S_A, color="steelblue", linewidth=2, label="Species A (initial)")
if S_PSS_preview is not None:
    ax2.plot(grid, S_PSS_preview, color="mediumseagreen", linewidth=2, label="PSS spectrum")
_add_ref_overlay(ax2)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel(y_label)
if mode == "negative":
    _title2 = (f"Selection Preview — {compound_name} | "
               f"{len(used_idx)} used / {len(alpha_excluded_idx)} α-excl. / "
               f"{len(range_excluded_idx)} range-excl.")
else:
    _title2 = (f"Selection Preview — {compound_name} | "
               f"{len(used_idx)} used / {len(range_excluded_idx)} range-excl.")
ax2.set_title(_title2)
ax2.legend()
ax2.grid(True)
plt.tight_layout()
print("  [Close the plot window to continue]")
plt.show()

if input("Selection looks OK? Continue with extraction? (y/n): ").strip().lower() != "y":
    plt.close(fig2)
    raise SystemExit(0)
plt.close(fig2)

# ============================================================
# MODE: NEGATIVE SWITCH
# ============================================================
if mode == "negative":
    print(f"\nMode: negative switch (reference: {ref_desc})")

    if A0_ref <= 0:
        raise ValueError(f"Initial absorbance at reference ({ref_desc}) is ≤ 0.")

    B_estimates = []
    alphas      = []

    # series_used already filtered by spectrum_indices + alpha window
    weight_desc = "weighted (least-squares)" if reference_weighted else "unweighted (mean)"
    print(f"  Reference α estimation: {weight_desc}")
    n_negative_rejected = 0
    for idx_i, S_i in enumerate(series_used):
        alpha_i = compute_alpha(S_i, S_A, ref_indices, reference_weighted)
        B_i = (S_i - alpha_i * S_A) / (1.0 - alpha_i)
        if exclude_negative_SB and np.any(B_i < -sb_tolerance):
            n_negative_rejected += 1
            print(f"  Spectrum {idx_i}: S_B_i below tolerance "
                  f"(min = {B_i.min():.4f} at {grid[np.argmin(B_i)]} nm, "
                  f"tolerance = {-sb_tolerance:.4f}) — excluded.")
            continue
        B_estimates.append(B_i)
        alphas.append(alpha_i)

    if n_negative_rejected:
        print(f"  {n_negative_rejected} spectrum/spectra rejected due to negative S_B values.")

    if len(B_estimates) < 2:
        raise ValueError("Fewer than 2 valid spectra for negative-switch extraction.")

    B_estimates = np.array(B_estimates)
    S_B      = B_estimates.mean(axis=0)
    S_B_std  = B_estimates.std(axis=0, ddof=1)
    S_B_lo   = S_B - S_B_std
    S_B_hi   = S_B + S_B_std

    print(f"  Used {len(B_estimates)} spectra  (α range: {min(alphas):.3f}–{max(alphas):.3f})")
    if np.any(S_B < 0):
        print("  Warning: extracted S_B still has negative values at some wavelengths.")

    # Most-converted spectrum (lowest α = most B)
    most_converted_spectrum = series_used[np.argmin(alphas)]

    # Diagnostic plot: all individual S_B_i estimates coloured by α
    if show_diagnostic:
        fig_diag, ax_diag = plt.subplots(figsize=(9, 5))
        cmap   = plt.cm.plasma
        norm   = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
        for B_i, a_i in zip(B_estimates, alphas):
            ax_diag.plot(grid, B_i, color=cmap(norm(a_i)), linewidth=0.8, alpha=0.6)
        ax_diag.plot(grid, most_converted_spectrum, color="forestgreen", linewidth=1.5,
                     linestyle="--", label=f"Most converted irradiation spectrum (α={min(alphas):.2f})")
        ax_diag.plot(grid, S_B, color="black", linewidth=2, label="Mean S_B")
        ax_diag.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_diag, label="α (1 = barely converted, 0 = fully converted)")
        ax_diag.set_xlabel("Wavelength (nm)")
        ax_diag.set_ylabel(y_label)
        ax_diag.set_title(f"Diagnostic: individual S_B estimates — {compound_name}")
        ax_diag.legend()
        ax_diag.grid(True)
        plt.tight_layout()
        print("  [Close the plot window to continue]")
        plt.show()
        plt.close(fig_diag)

# ============================================================
# MODE: PCA — shared logic for negative_pca and positive_pca
# ============================================================
elif mode in ("negative_pca", "positive_pca"):
    print(f"\nMode: {mode.replace('_', ' ')} — PCA extrapolation with non-negativity constraint")

    D = series_used - S_A    # difference spectra (n_spectra × n_wavelengths)

    # SVD: keep first component (rank-1 approximation for 2-component system)
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    PC1       = Vt[0]                  # spectral direction of A→B change (unit vector)
    scores    = U[:, 0] * sv[0]        # how far each spectrum has moved along PC1

    # Sign convention: scores must increase as conversion from A to B progresses.
    # negative_pca: A bleaches → D < 0 → SVD may give PC1 in either direction.
    # positive_pca: B grows   → D > 0 → same ambiguity.
    # Flip so that the last (most-irradiated) spectrum has the highest score.
    if scores[-1] < scores[0]:
        PC1    = -PC1
        scores = -scores

    # -------------------------
    # Scale determination — genuine extrapolation to pure B
    # -------------------------
    # SVD gives PC1 = (S_B − S_A) / ‖S_B − S_A‖ exactly (for a 2-component system).
    # The unknown is the scale ‖S_B − S_A‖.
    #
    # Physical constraints bound the scale from both sides:
    #
    #   Lower bound (α ≥ 0):  scale ≥ scores.max()
    #     If scale were smaller, the most-converted spectrum would have α < 0
    #     (more than 100% B), which is unphysical.
    #
    #   Upper bound (S_B ≥ −tolerance):  scale ≤ scale_nn
    #     Push scale as far as possible until S_B just touches −tolerance at the
    #     most constraining wavelength. This IS genuine extrapolation — it finds
    #     the furthest physically valid pure-B spectrum beyond all measured mixtures.
    #     Baseline-noise wavelengths (S_A + tolerance ≤ 0) are excluded from the
    #     constraint so they don't collapse the scale.
    #
    # Primary scale = scale_nn (upper bound, maximum extrapolation).
    # If scale_nn < scores.max(), the constraints are inconsistent (baseline issue);
    # fall back to scores.max() with a warning.

    neg_mask    = PC1 < 0
    constrained = neg_mask & (S_A + sb_tolerance > 0)
    scale_nn    = float(np.min((S_A[constrained] + sb_tolerance) / (-PC1[constrained]))) \
                  if constrained.any() else np.inf

    scale_lower = float(scores.max())   # minimum from α ≥ 0

    if scale_nn >= scale_lower:
        scale = scale_nn
        _constraint_wl = grid[constrained][np.argmin(
            (S_A[constrained] + sb_tolerance) / (-PC1[constrained])
        )]
        print(f"  Scale (non-negativity extrapolation): {scale:.4f}  "
              f"[constraining wavelength: {_constraint_wl} nm]")
    else:
        scale = scale_lower
        print(f"  Warning: non-negativity limit ({scale_nn:.4f}) < minimum scale from α≥0 "
              f"({scale_lower:.4f}). Constraints inconsistent — likely a baseline offset issue. "
              f"Falling back to minimum scale (no extrapolation beyond last spectrum).")

    scale_max = scale   # used in metadata / diagnostic
    S_B = S_A + scale * PC1

    # Most-converted spectrum — used for visualisation in diagnostic and final plot
    most_converted_spectrum = series_used[np.argmax(scores)]

    # α for each spectrum: α(t) = 1 − score(t) / scale
    # Values near 0 → near pure B; values near 1 → near pure A.
    # With genuine extrapolation (scale = scale_nn > scores.max()), α_min > 0
    # confirms the spectra never reached pure B, yet S_B is still correctly extracted.
    alphas_pca    = 1.0 - scores / scale
    alpha_min_pca = float(alphas_pca.min())
    alpha_max_pca = float(alphas_pca.max())
    print(f"  α range across spectra: {alpha_min_pca:.3f} – {alpha_max_pca:.3f} "
          f"(α=0 → pure B, α=1 → pure A)")
    if alpha_min_pca > 0.3:
        print(f"  Note: best conversion reached was {(1-alpha_min_pca)*100:.0f}% B. "
              f"Extrapolation is valid but relies more heavily on the non-negativity "
              f"constraint — wider CI expected. Better conversion improves confidence.")

    # Residuals from rank-1 approximation → pointwise noise estimate
    D_approx = np.outer(scores, PC1)
    residuals = D - D_approx
    S_B_std   = residuals.std(axis=0, ddof=1)

    # Bootstrap: resample difference spectra, re-estimate PC1 and scale_nn
    boot_SB_raw     = []
    boot_scales_raw = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx_b    = rng.integers(0, len(D), len(D))
        D_b      = D[idx_b]
        _, _, Vt_b = np.linalg.svd(D_b, full_matrices=False)
        pc1_b    = Vt_b[0]
        if pc1_b @ PC1 < 0:
            pc1_b = -pc1_b
        constr_b = (pc1_b < 0) & (S_A + sb_tolerance > 0)
        sc_nn_b  = float(np.min((S_A[constr_b] + sb_tolerance) / (-pc1_b[constr_b]))) \
                   if constr_b.any() else np.inf
        sc_lower_b = float((D_b @ pc1_b).max())
        sc_b     = sc_nn_b if sc_nn_b >= sc_lower_b else sc_lower_b
        boot_SB_raw.append(S_A + sc_b * pc1_b)
        boot_scales_raw.append(sc_b)

    # Discard resamples where S_B goes below −sb_tolerance
    boot_pairs = [(sb, sc) for sb, sc in zip(boot_SB_raw, boot_scales_raw)
                  if not np.any(sb < -sb_tolerance)]
    n_boot_rejected = n_bootstrap - len(boot_pairs)
    if n_boot_rejected:
        print(f"  Bootstrap: {n_boot_rejected} of {n_bootstrap} resamples had negative S_B and were excluded.")
    if len(boot_pairs) < 10:
        print(f"  Warning: too few valid bootstrap resamples ({len(boot_pairs)}) — using all {n_bootstrap} for CI.")
        boot_pairs = list(zip(boot_SB_raw, boot_scales_raw))
    boot_SB_arr     = np.array([p[0] for p in boot_pairs])
    boot_scales_arr = np.array([p[1] for p in boot_pairs])
    S_B_lo = np.percentile(boot_SB_arr,  2.5, axis=0)
    S_B_hi = np.percentile(boot_SB_arr, 97.5, axis=0)

    var_explained = sv[0]**2 / np.sum(sv**2)
    print(f"  PC1 explains {var_explained*100:.1f}% of variance in difference spectra.")
    print(f"  Scale factor: {scale:.4f}  (non-negativity limit: "
          f"{'∞' if scale_nn == np.inf else f'{scale_nn:.4f}'})")
    _sb_neg_mask = S_B < -sb_tolerance
    if _sb_neg_mask.any():
        neg_wl = grid[_sb_neg_mask]
        print(f"  Warning: S_B exceeds tolerance ({-sb_tolerance:.4f}) at {_sb_neg_mask.sum()} "
              f"wavelengths ({neg_wl[0]}–{neg_wl[-1]} nm, min = {S_B.min():.4f}). "
              f"Consider a baseline offset or increasing sb_tolerance_sigma.")
    elif np.any(S_B < 0):
        print(f"  S_B has minor negative values (within {sb_tolerance_sigma}σ tolerance, "
              f"min = {S_B.min():.4f}) — within baseline noise, accepted.")

    # Correlation of PC1 with S_A is diagnostic:
    #   negative_pca: PC1 ≈ −S_A means B absorbs very little compared to A. Extraction
    #                 is valid but S_B will be small. If B has distinct absorption bands,
    #                 using negative mode with a reference wavelength is more reliable.
    #   positive_pca: PC1 should be mostly positive (B grows); strong anti-correlation
    #                 would be unusual and may indicate data ordering problems.
    corr_pc1_sa = np.corrcoef(PC1, S_A)[0, 1]
    if mode == "negative_pca" and corr_pc1_sa < -0.90:
        print(f"  Note: PC1 is strongly anti-correlated with S_A (r = {corr_pc1_sa:.3f}). "
              f"The dominant change is A bleaching — B may have very low absorbance. "
              f"If B has distinct absorption features, negative mode with a reference "
              f"wavelength (where only A absorbs) will give a more reliable S_B.")
    elif mode == "positive_pca" and corr_pc1_sa < -0.95:
        print(f"  Note: PC1 is strongly anti-correlated with S_A (r = {corr_pc1_sa:.3f}). "
              f"For positive_pca this is unexpected — check that spectra are in "
              f"irradiation order and that absorbance is increasing as expected.")

    # -------------------------
    # Diagnostic plot (PCA modes)
    # -------------------------
    if show_diagnostic:
        fig_diag, (ax_sc, ax_boot) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: PC1 scores vs spectrum index (primary y-axis)
        # and derived α = 1 − score/scale (secondary y-axis).
        # Scores should increase monotonically for a clean photoreaction;
        # deviations indicate noise or mis-ordered spectra.
        ax_sc.plot(range(len(scores)), scores, "o-", color="steelblue",
                   markerfacecolor="none", markeredgewidth=1.2, linewidth=1.2,
                   label="PC1 score")
        ax_sc.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_sc.set_xlabel("Spectrum index")
        ax_sc.set_ylabel("PC1 score", color="steelblue")
        ax_sc.tick_params(axis="y", labelcolor="steelblue")

        ax_alpha = ax_sc.twinx()
        ax_alpha.plot(range(len(alphas_pca)), alphas_pca, "s--", color="darkorange",
                      markerfacecolor="none", markeredgewidth=1.0, linewidth=1.0,
                      label="α (fraction A remaining)")
        ax_alpha.axhline(0, color="darkorange", linewidth=0.6, linestyle=":", alpha=0.6)
        ax_alpha.axhline(1, color="darkorange", linewidth=0.6, linestyle=":", alpha=0.6)
        ax_alpha.set_ylim(-0.1, 1.3)
        ax_alpha.set_ylabel("α (fraction of A remaining)", color="darkorange")
        ax_alpha.tick_params(axis="y", labelcolor="darkorange")

        _corr_str = f"PC1 vs S_A: r = {corr_pc1_sa:.3f}"
        ax_sc.set_title(f"Conversion progress — {_corr_str}")
        ax_sc.grid(True)

        # Annotation boxes
        _annot_lines = []
        if alpha_min_pca > 0.3:
            _annot_lines += [f"Best conversion: {(1-alpha_min_pca)*100:.0f}% B  (α_min={alpha_min_pca:.2f})",
                             "Extrapolation relies on non-neg. constraint",
                             "Better conversion → narrower CI"]
        if mode == "negative_pca" and corr_pc1_sa < -0.90:
            _annot_lines += [f"PC1 ≈ −S_A  (r = {corr_pc1_sa:.2f})",
                             "B may have very low absorbance",
                             "Consider negative mode + ref. λ"]
        elif mode == "positive_pca" and corr_pc1_sa < -0.50:
            _annot_lines += [f"PC1 anti-corr. with S_A  (r = {corr_pc1_sa:.2f})",
                             "Check spectrum ordering"]
        if _annot_lines:
            ax_sc.text(0.02, 0.97, "\n".join(_annot_lines),
                       transform=ax_sc.transAxes, fontsize=8,
                       verticalalignment="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

        # Right panel: bootstrap S_B samples coloured by scale factor
        n_boot_show = min(100, len(boot_SB_arr))
        step_boot   = max(1, len(boot_SB_arr) // n_boot_show)
        shown_SB    = boot_SB_arr[::step_boot]
        shown_sc    = boot_scales_arr[::step_boot]
        cmap_b  = plt.cm.plasma
        norm_b  = plt.Normalize(vmin=shown_sc.min(), vmax=shown_sc.max())
        for sb_i, sc_i in zip(shown_SB, shown_sc):
            ax_boot.plot(grid, sb_i, color=cmap_b(norm_b(sc_i)), linewidth=0.6, alpha=0.4)
        ax_boot.plot(grid, most_converted_spectrum, color="forestgreen", linewidth=1.5,
                     linestyle="--",
                     label=f"Most converted spectrum (α={alpha_min_pca:.2f})")
        ax_boot.plot(grid, S_B, color="black", linewidth=2, label="S_B (extracted)")
        ax_boot.fill_between(grid, S_B_lo, S_B_hi, color="darkorange", alpha=0.25, label="95% CI")
        ax_boot.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
        sm_b.set_array([])
        plt.colorbar(sm_b, ax=ax_boot, label="Bootstrap scale factor")
        ax_boot.set_xlabel("Wavelength (nm)")
        ax_boot.set_ylabel(y_label)
        ax_boot.set_title(f"Bootstrap S_B ({n_boot_show} of {len(boot_SB_arr)} valid shown) — {compound_name}")
        ax_boot.legend()
        ax_boot.grid(True)

        plt.tight_layout()
        print("  [Close the plot window to continue]")
        plt.show()
        plt.close(fig_diag)

# ============================================================
# MODE: POSITIVE SWITCH — known PSS ratio
# ============================================================
elif mode == "positive_pss":
    print(f"\nMode: positive switch — PSS ratio (f_B = {pss_fraction_B} ± {pss_fraction_B_error})")

    print("Loading PSS spectrum...")
    pss_files = sorted(PSS_DIR.glob("*.csv"))
    if not pss_files:
        raise FileNotFoundError(f"No CSV files found in {PSS_DIR}.")
    pss_scans = []
    for f in pss_files:
        for wl, ab in load_spectra_csv(f):
            pss_scans.append(interpolate_to_grid(wl, ab, grid))
    S_PSS = np.array(pss_scans).mean(axis=0)
    print(f"  {len(pss_scans)} scan(s) averaged onto shared grid.")

    if concentration_mol_L is not None:
        S_PSS = S_PSS * (1.0 / (concentration_mol_L * path_length_cm))

    f_B = pss_fraction_B
    S_B = (S_PSS - (1.0 - f_B) * S_A) / f_B

    # Confidence interval from uncertainty in f_B
    S_B_lo = (S_PSS - (1.0 - (f_B + pss_fraction_B_error)) * S_A) / (f_B + pss_fraction_B_error)
    S_B_hi = (S_PSS - (1.0 - (f_B - pss_fraction_B_error)) * S_A) / (f_B - pss_fraction_B_error)
    S_B_std = (S_B_hi - S_B_lo) / 2.0

    if np.any(S_B < 0):
        print("  Warning: extracted S_B has negative values at some wavelengths.")

    most_converted_spectrum = None   # no irradiation series used in this mode

else:
    raise ValueError(f"Unknown mode '{mode}'. Use 'negative', 'negative_pca', 'positive_pca', or 'positive_pss'.")

# ============================================================
# CONVERGENCE DIAGNOSTIC
# Re-run extraction on growing subsets of series_used.
# Shows whether the extracted S_B stabilises as more spectra are added.
# ============================================================
if show_convergence and mode in ("negative_pca", "positive_pca", "negative"):
    print("\nConvergence diagnostic...")

    def _pca_extract_subset(sub, S_A_ref, sb_tol):
        """Run PCA extraction on sub (n×λ array). Returns (S_B, alpha_min, scale) or None."""
        if len(sub) < 3:
            return None
        D_sub = sub - S_A_ref
        _, _, Vt_s = np.linalg.svd(D_sub, full_matrices=False)
        pc1_s    = Vt_s[0]
        scores_s = D_sub @ pc1_s
        if scores_s[-1] < scores_s[0]:
            pc1_s    = -pc1_s
            scores_s = -scores_s
        # Same scale logic as main extraction: scale_nn as primary, scores.max() as floor
        constr_s  = (pc1_s < 0) & (S_A_ref + sb_tol > 0)
        sc_nn_s   = float(np.min((S_A_ref[constr_s] + sb_tol) / (-pc1_s[constr_s]))) \
                    if constr_s.any() else np.inf
        sc_lower_s = float(scores_s.max())
        scale_s    = sc_nn_s if sc_nn_s >= sc_lower_s else sc_lower_s
        SB_s       = S_A_ref + scale_s * pc1_s
        alpha_min_s = float(1.0 - sc_lower_s / scale_s) if scale_s > 0 else np.nan
        return SB_s, alpha_min_s, scale_s

    def _neg_extract_subset(sub, S_A_ref, ref_idx, weighted, sb_tol):
        """Run negative-switch extraction on sub. Returns (S_B_mean, alpha_min) or None."""
        estimates, alphas_s = [], []
        for sp in sub:
            a = compute_alpha(sp, S_A_ref, ref_idx, weighted)
            if np.isnan(a) or a <= 0 or a >= 1:
                continue
            Bi = (sp - a * S_A_ref) / (1.0 - a)
            if np.any(Bi < -sb_tol):
                continue
            estimates.append(Bi)
            alphas_s.append(a)
        if len(estimates) < 2:
            return None
        SB_s = np.array(estimates).mean(axis=0)
        return SB_s, float(min(alphas_s))

    conv_results = []   # list of (fraction, n_spectra, S_B, alpha_min)
    n_used = len(series_used)
    for frac in convergence_fractions:
        n_sub = max(3, int(round(frac * n_used)))
        sub   = series_used[:n_sub]
        if mode in ("negative_pca", "positive_pca"):
            res = _pca_extract_subset(sub, S_A, sb_tolerance)
        else:
            res = _neg_extract_subset(sub, S_A, ref_indices, reference_weighted, sb_tolerance)
        if res is None:
            print(f"  Fraction {frac:.0%}: only {n_sub} spectra — skipped.")
            continue
        SB_s, amin_s = res[0], res[1]
        conv_results.append((frac, n_sub, SB_s, amin_s))
        scale_str = f", scale={res[2]:.4f}" if mode in ("negative_pca", "positive_pca") else ""
        print(f"  Fraction {frac:.0%} ({n_sub} spectra): α_min = {amin_s:.3f}{scale_str}")

    if len(conv_results) >= 2:
        fig_conv, ax_conv = plt.subplots(figsize=(9, 5))
        _cmap_conv = plt.cm.viridis
        _norm_conv = plt.Normalize(vmin=0, vmax=1)
        for frac, n_sub, SB_s, amin_s in conv_results:
            _c = _cmap_conv(_norm_conv(frac))
            ax_conv.plot(grid, SB_s, color=_c, linewidth=1.4,
                         label=f"{frac:.0%} of spectra ({n_sub}), α_min={amin_s:.2f}")
        # Overlay the final S_B for reference
        ax_conv.plot(grid, S_B, color="black", linewidth=2.0, linestyle="--", label="Final S_B (all data)")
        ax_conv.plot(grid, S_A, color="steelblue", linewidth=1.5, linestyle=":", label="S_A")
        ax_conv.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        sm_conv = plt.cm.ScalarMappable(cmap=_cmap_conv, norm=_norm_conv)
        sm_conv.set_array([])
        plt.colorbar(sm_conv, ax=ax_conv, label="Fraction of irradiation series used",
                     fraction=0.03, pad=0.01)
        ax_conv.set_xlabel("Wavelength (nm)")
        ax_conv.set_ylabel(y_label)
        ax_conv.set_title(f"Convergence diagnostic — {compound_name} | mode: {mode}")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True)
        plt.tight_layout()
        print("  [Close the plot window to continue]")
        plt.show()
        plt.close(fig_conv)
    else:
        print("  Not enough valid subsets for convergence plot.")

# ============================================================
# BUILD METADATA (for figure annotation and CSV header)
# ============================================================
_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
meta = [
    f"Date           : {_ts}",
    f"Compound       : {compound_name}",
    f"Mode           : {mode}",
    f"Wavelength range: {grid[0]}–{grid[-1]} nm",
    f"Spectra used   : {len(series_used)} of {len(series)} loaded",
]
if baseline_offset != 0.0:
    meta.append(f"Baseline offset: {baseline_offset:+.6f}")
if mode == "negative":
    meta += [
        f"Reference      : {ref_desc}",
        f"Ref. weighting : {'weighted (least-squares)' if reference_weighted else 'unweighted (mean)'}",
        f"alpha window   : ({alpha_lo:.2f}, {alpha_hi:.2f})",
        f"alpha range    : {min(alphas):.3f}–{max(alphas):.3f}",
        f"excl. neg. S_B : {exclude_negative_SB} ({n_negative_rejected} rejected)",
    ]
elif mode in ("negative_pca", "positive_pca"):
    meta += [
        f"n_bootstrap    : {n_bootstrap} ({n_boot_rejected} rejected, negative S_B)",
        f"PC1 variance   : {var_explained*100:.1f}%",
        f"Scale (non-neg extrap): {scale_max:.4f}",
        f"alpha range    : {alpha_min_pca:.3f}–{alpha_max_pca:.3f}",
    ]
elif mode == "positive_pss":
    meta += [
        f"f_B            : {pss_fraction_B} ± {pss_fraction_B_error}",
    ]

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

# Irradiation spectra (subsampled for display only — no data is excluded)
n_show = min(5, len(series_used))
step   = max(1, len(series_used) // n_show)
shown  = series_used[::step]
for j, s in enumerate(shown):
    ax.plot(grid, s, color="grey", linewidth=0.8, alpha=0.4,
            label=f"Irradiation spectra ({len(shown)} of {len(series_used)} shown)" if j == 0 else None)

# Initial (A), most-converted irradiation spectrum, and extracted (B)
ax.plot(grid, S_A, color="steelblue", linewidth=2, label="Species A (initial)")
if most_converted_spectrum is not None:
    _mc_alpha = alpha_min_pca if mode in ("negative_pca", "positive_pca") else min(alphas)
    ax.plot(grid, most_converted_spectrum, color="forestgreen", linewidth=1.5,
            linestyle="--", label=f"Most converted spectrum (α={_mc_alpha:.2f})")
ax.plot(grid, S_B, color="darkorange", linewidth=2, label="Species B (extracted)")
ax.fill_between(grid, S_B_lo, S_B_hi, color="darkorange", alpha=0.25, label="±1σ confidence")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(y_label)
ax.set_title(f"Spectral Extraction — {compound_name} | mode: {mode}")
ax.legend()
ax.grid(True)
ax.text(0.01, 0.99, "\n".join(meta), transform=ax.transAxes,
        fontsize=7, verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75))
plt.tight_layout()
print("  [Close the plot window to continue]")
plt.show()

# -------------------------
# Save image
# -------------------------
file_stem = f"{compound_name}_{mode}_spectra"
if input("Save image? (y/n): ").strip().lower() == "y":
    img_path = PLOTS_DIR / f"{file_stem}.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"Image saved → {img_path.relative_to(BASE_DIR)}")

plt.close(fig)

# -------------------------
# Save extracted spectrum
# -------------------------
if input("Save extracted spectrum as CSV? (y/n): ").strip().lower() == "y":
    df_out = pd.DataFrame({
        "Wavelength_nm": grid,
        "Species_A":     S_A,
        "Species_B":     S_B,
        "Species_B_lo":  S_B_lo,
        "Species_B_hi":  S_B_hi,
        "Species_B_std": S_B_std,
    })
    out_path = EXTRACTED_DIR / f"{file_stem}.csv"
    with open(out_path, "w") as _f:
        for line in meta:
            _f.write(f"# {line}\n")
        df_out.to_csv(_f, index=False)
    print(f"Spectrum saved → {out_path.relative_to(BASE_DIR)}")
    print("  (CSV header contains # comment lines — read back with pd.read_csv(..., comment='#'))")
