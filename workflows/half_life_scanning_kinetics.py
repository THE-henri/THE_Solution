from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.kinetics import fit_half_life
from core.plotting import plot_half_life_with_linear
from core.io import append_half_life_result

# -------------------------
# Setup directories
# -------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "half_life" / "raw"
REFERENCE_DIR  = BASE_DIR / "data" / "half_life" / "reference"
RESULTS_DIR    = BASE_DIR / "data" / "half_life" / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
GRAPH_DATA_DIR = RESULTS_DIR / "graph_data"

for folder in [RAW_DIR, REFERENCE_DIR, RESULTS_DIR, PLOTS_DIR, GRAPH_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load CSV files
# -------------------------
csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}. Please add your scanning kinetics CSVs.")

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
target_wavelengths  = [359, 539, 672]   # nm — list of wavelengths to extract kinetic traces for
wavelength_tolerance = 1           # nm — match window around each target wavelength
time_interval_s      = 300          # seconds between successive scans
switch               = "negative"  # "positive" or "negative"
selection_indices    = (0, None)   # (start_scan, end_scan); None = last scan
temperature_value    = 25          # °C
A_inf_man             = None        # fallback asymptote — only used when use_reference_spectrum = False
outlier_iqr_factor    = 50.0        # points with |residual| > factor × IQR are excluded
use_reference_spectrum = True  # if True, A∞ at each wavelength is read from the reference folder


# -------------------------
# Helper: load scanning kinetics CSV
# -------------------------
def load_scanning_kinetics_csv(filepath):
    """
    Load a scanning kinetics CSV with the format:
      Row 0 : scan labels (e.g. 3-QY_12s_1) in every other column
      Row 1 : 'Wavelength (nm)', 'Abs' column headers repeated per scan
      Rows 2+: wavelength / absorbance data pairs per scan

    Returns
    -------
    list of (wavelength_array, absorbance_array) — one entry per scan,
    in the order they appear in the file.
    """
    MIN_VALID_POINTS = 5

    raw = pd.read_csv(filepath, header=None)
    data = raw.iloc[2:].reset_index(drop=True)   # skip the two header rows
    n_cols = data.shape[1]

    scans = []
    for i in range(0, n_cols - 1, 2):
        wl_col  = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        abs_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")

        valid = wl_col.notna() & abs_col.notna()
        if valid.sum() < MIN_VALID_POINTS:
            continue

        scans.append((wl_col[valid].values, abs_col[valid].values))

    return scans


# -------------------------
# Helper: extract kinetic trace at one wavelength
# -------------------------
def extract_trace(scans, target_nm, tolerance_nm):
    """
    For each scan return the mean absorbance within
    [target_nm − tolerance_nm, target_nm + tolerance_nm].
    Scans with no matching wavelengths yield NaN.
    """
    trace = []
    for wl, ab in scans:
        mask = (wl >= target_nm - tolerance_nm) & (wl <= target_nm + tolerance_nm)
        trace.append(ab[mask].mean() if mask.any() else np.nan)
    return np.array(trace)


# -------------------------
# Helper: outlier removal (identical logic to half_life_workflow)
# -------------------------
def remove_outliers(time, absorbance, fitted_curve, iqr_factor):
    """
    Remove points whose residual from an initial fit exceeds iqr_factor × IQR.
    Returns time_clean, absorbance_clean, inlier_mask.
    """
    residuals = absorbance - fitted_curve
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    inlier_mask = np.abs(residuals) <= iqr_factor * iqr
    return time[inlier_mask], absorbance[inlier_mask], inlier_mask


# -------------------------
# Helper: extract A∞ from a reference spectrum file
# -------------------------
def load_reference_a_inf(reference_file, target_nm, tolerance_nm):
    """
    Load a reference spectrum (same column-pair format as scanning kinetics).
    If multiple scans are present, their absorbances are averaged.
    Returns the mean absorbance at target_nm as the A∞ value for that wavelength.
    """
    scans = load_scanning_kinetics_csv(reference_file)
    if not scans:
        raise ValueError(f"No valid scans found in reference file {reference_file.name}.")
    values = extract_trace(scans, target_nm, tolerance_nm)
    valid  = values[~np.isnan(values)]
    if len(valid) == 0:
        raise ValueError(f"No data near {target_nm} nm in reference file {reference_file.name}.")
    return float(np.mean(valid))


# -------------------------
# Load reference file once (GUI will replace this with a file picker)
# -------------------------
reference_file = None
if use_reference_spectrum:
    ref_files = list(REFERENCE_DIR.glob("*.csv"))
    if not ref_files:
        raise FileNotFoundError(
            f"use_reference_spectrum is True but no CSV found in {REFERENCE_DIR}."
        )
    if len(ref_files) > 1:
        print(f"  Warning: {len(ref_files)} files found in reference folder — using {ref_files[0].name}.")
    reference_file = ref_files[0]
    print(f"Reference spectrum: {reference_file.name}")


# -------------------------
# Process each file
# -------------------------
for csv_file in csv_files:
    print(f"\n{'='*60}")
    print(f"Processing {csv_file.name}")
    print(f"{'='*60}")

    try:
        scans = load_scanning_kinetics_csv(csv_file)
    except Exception as e:
        print(f"  Could not read {csv_file.name}: {e}")
        continue

    if not scans:
        print(f"  No valid scans found in {csv_file.name}.")
        continue

    n_scans = len(scans)
    print(f"  {n_scans} scans loaded.")

    # Full time axis covering all scans
    time_full = np.arange(n_scans) * time_interval_s

    for target_nm in target_wavelengths:
        print(f"\n  Wavelength: {target_nm} nm")

        # --- Extract kinetic trace ---
        absorbance_full = extract_trace(scans, target_nm, wavelength_tolerance)

        # Drop NaN scans (wavelength not found in that scan)
        valid_mask = ~np.isnan(absorbance_full)
        if valid_mask.sum() < 3:
            print(f"  Too few valid scans at {target_nm} nm — skipping.")
            continue

        time_valid      = time_full[valid_mask]
        absorbance_valid = absorbance_full[valid_mask]

        # --- Apply selection window ---
        start_idx = selection_indices[0]
        end_idx   = selection_indices[1] if selection_indices[1] is not None else len(time_valid) - 1
        end_idx   = min(end_idx, len(time_valid) - 1)

        time_sel      = time_valid[start_idx:end_idx + 1]
        absorbance_sel = absorbance_valid[start_idx:end_idx + 1]

        if len(time_sel) < 3:
            print(f"  Too few points in selection — skipping.")
            continue

        # --- Determine A∞ for this wavelength ---
        if use_reference_spectrum:
            try:
                a_inf_effective = load_reference_a_inf(
                    reference_file, target_nm, wavelength_tolerance
                )
                print(f"  A∞ from reference spectrum: {a_inf_effective:.6f}")
            except ValueError as e:
                print(f"  Could not extract A∞ from reference: {e} — skipping.")
                continue
        else:
            a_inf_effective = A_inf_man   # may be None (fitted freely)

        # --- Initial fit (for outlier detection) ---
        popt, t_half, fitted_curve, r2 = fit_half_life(
            time_sel, absorbance_sel, switch=switch, A_inf_manual=a_inf_effective
        )
        if t_half is None:
            print(f"  Initial fitting failed — skipping.")
            continue

        # --- Outlier removal ---
        time_clean, absorbance_clean, inlier_mask = remove_outliers(
            time_sel, absorbance_sel, fitted_curve, outlier_iqr_factor
        )
        time_outliers      = time_sel[~inlier_mask]
        absorbance_outliers = absorbance_sel[~inlier_mask]

        n_total    = len(inlier_mask)
        n_excluded = int((~inlier_mask).sum())
        print(f"  Outlier removal ({outlier_iqr_factor}×IQR): "
              f"{n_excluded}/{n_total} points excluded ({100*n_excluded/n_total:.1f}%)")

        if len(time_clean) < 3:
            print(f"  Too few points after outlier removal — skipping.")
            continue

        # --- Final fit on cleaned data ---
        popt, t_half, fitted_curve, r2 = fit_half_life(
            time_clean, absorbance_clean, switch=switch, A_inf_manual=a_inf_effective
        )
        if t_half is None:
            print(f"  Fitting failed after outlier removal — skipping.")
            continue

        print(f"  Half-life: {t_half:.2f} s   R² = {r2:.6f}")

        # --- Plot ---
        file_stem = f"{csv_file.stem}_{target_nm}nm_T{temperature_value}C"

        fig, _ = plot_half_life_with_linear(
            time_valid, absorbance_valid,
            start_idx=start_idx, end_idx=end_idx,
            time_sel=time_clean, absorbance_sel=absorbance_clean,
            time_outliers=time_outliers, absorbance_outliers=absorbance_outliers,
            fitted_curve=fitted_curve, r_squared=r2,
            popt=popt, t_half=t_half, switch=switch,
            title=f"{csv_file.stem} | {target_nm} nm | T={temperature_value}°C",
            show=False,
        )
        plt.show()

        # --- Save image ---
        if input("  Save image? (y/n): ").strip().lower() == "y":
            img_path = PLOTS_DIR / f"{file_stem}.png"
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            print(f"  Image saved → {img_path.relative_to(BASE_DIR)}")

        plt.close(fig)

        # --- Save graph data ---
        if input("  Save graph data as CSV? (y/n): ").strip().lower() == "y":
            graph_df = pd.DataFrame({
                "Time_s":            np.concatenate([time_clean, time_outliers]),
                "Absorbance":        np.concatenate([absorbance_clean, absorbance_outliers]),
                "Fitted_Absorbance": np.concatenate([fitted_curve,
                                                     np.full(len(time_outliers), np.nan)]),
                "Data_type":         (["inlier"] * len(time_clean) +
                                      ["outlier"] * len(time_outliers)),
            }).sort_values("Time_s").reset_index(drop=True)

            data_path = GRAPH_DATA_DIR / f"{file_stem}_data.csv"
            graph_df.to_csv(data_path, index=False)
            print(f"  Graph data saved → {data_path.relative_to(BASE_DIR)}")

        # --- Save fit result to master CSV ---
        if input("  Save fit result to master CSV? (y/n): ").strip().lower() == "y":
            result_entry = {
                "File":          csv_file.name,
                "Wavelength_nm": target_nm,
                "Type":          "Scanning Kinetics",
                "Temperature_C": temperature_value,
                "Switch":        switch,
                "A0":            popt[0],
                "A_inf":         popt[1] if switch == "negative" else None,
                "k":             popt[-1],
                "Half_life_s":   t_half,
                "R2":            r2,
            }
            append_half_life_result(result_entry, RESULTS_DIR / "half_life_master.csv")
            print("  Result saved.")
