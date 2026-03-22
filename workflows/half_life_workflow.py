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
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "half_life" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "half_life" / "processed"
RESULTS_DIR = BASE_DIR / "data" / "half_life" / "results"

PLOTS_DIR      = RESULTS_DIR / "plots"
GRAPH_DATA_DIR = RESULTS_DIR / "graph_data"

for folder in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, GRAPH_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load CSV files
# -------------------------
csv_files = list(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}. Please add your half-life measurement CSVs.")

# -------------------------
# Workflow parameters
# -------------------------
switch = "negative"          # "positive" or "negative"
selection_indices = (78, 398)  # start_idx, end_idx) applied to each wavelength channel
temperature_value = 45    # will be GUI input later
A_inf_man = 0             # optionally fix the asymptote for negative switch
outlier_iqr_factor = 3.0     # points with |residual| > factor × IQR are excluded; will be GUI input later


def load_multi_wavelength_csv(filepath):
    """
    Load a multi-wavelength CSV with the format:
      Row 0 : wavelength group labels (e.g. 45C_672) in every other column
      Row 1 : 'Time (sec)', 'Abs' column headers repeated per wavelength
      Rows 2+: time/absorbance data pairs

    Metadata rows inserted by the Cary 60 (e.g. "Wavelengths (nm)  409.0")
    are stripped by keeping only rows where col 0 (first channel's time) is
    numeric. Channels with fewer than MIN_VALID_POINTS remaining data points
    are skipped entirely.

    Returns
    -------
    dict : {label: (time_array, abs_array)}
    """
    MIN_VALID_POINTS = 10

    raw = pd.read_csv(filepath, header=None)
    label_row = raw.iloc[0]
    # Drop metadata rows: keep only rows where col 0 (first channel's time) is numeric.
    # Cary 60 inserts periodic metadata blocks (e.g. "Wavelengths (nm)  409.0") whose
    # values in cols 2-3 look numeric and would otherwise create spurious data points.
    _all_data = raw.iloc[2:].reset_index(drop=True)
    _col0_num = pd.to_numeric(_all_data.iloc[:, 0], errors="coerce")
    data = _all_data[_col0_num.notna()].reset_index(drop=True)

    channels = {}
    n_cols = label_row.shape[0]

    for i in range(0, n_cols - 1, 2):
        label = str(label_row.iloc[i]).strip()
        if not label or label.lower() == "nan":
            continue

        time_col = pd.to_numeric(data.iloc[:, i], errors="coerce")
        abs_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")

        valid = time_col.notna() & abs_col.notna()
        n_valid = valid.sum()

        if n_valid < MIN_VALID_POINTS:
            print(f"  Skipping channel '{label}': only {n_valid} valid data points.")
            continue

        channels[label] = (time_col[valid].values, abs_col[valid].values)

    return channels


def remove_outliers(time, absorbance, fitted_curve, iqr_factor):
    """
    Remove points whose residual from an initial fit exceeds iqr_factor × IQR.

    Uses residuals rather than raw absorbance values so that the threshold
    adapts to the local scatter around the exponential trend, not the
    overall range of the decaying signal.

    Returns
    -------
    time_clean, absorbance_clean : arrays with outliers removed
    inlier_mask                  : boolean array (True = kept)
    """
    residuals = absorbance - fitted_curve
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    inlier_mask = np.abs(residuals) <= iqr_factor * iqr
    return time[inlier_mask], absorbance[inlier_mask], inlier_mask


# -------------------------
# Process each file
# -------------------------
for csv_file in csv_files:
    print(f"\n{'='*60}")
    print(f"Processing {csv_file.name}")
    print(f"{'='*60}")

    try:
        channels = load_multi_wavelength_csv(csv_file)
    except Exception as e:
        print(f"  Could not read {csv_file.name}: {e}")
        continue

    if not channels:
        print(f"  No valid wavelength channels found in {csv_file.name}.")
        continue

    for label, (time, absorbance) in channels.items():
        print(f"\n  Wavelength channel: {label}  ({len(time)} points)")

        # Apply selection window
        start_idx = selection_indices[0]
        end_idx = selection_indices[1] if selection_indices[1] is not None else len(time) - 1
        end_idx = min(end_idx, len(time) - 1)

        time_sel = time[start_idx:end_idx + 1]
        absorbance_sel = absorbance[start_idx:end_idx + 1]

        if len(time_sel) < 3:
            print(f"  Too few points in selection — skipping.")
            continue

        # Initial fit to establish the trend for outlier detection
        popt, t_half, fitted_curve, r2 = fit_half_life(
            time_sel, absorbance_sel, switch=switch, A_inf_manual=A_inf_man
        )
        if t_half is None:
            print(f"  Initial fitting failed — skipping.")
            continue

        # Remove outliers based on residuals from the initial fit
        time_clean, absorbance_clean, inlier_mask = remove_outliers(
            time_sel, absorbance_sel, fitted_curve, outlier_iqr_factor
        )
        time_outliers = time_sel[~inlier_mask]
        absorbance_outliers = absorbance_sel[~inlier_mask]
        n_total = len(inlier_mask)
        n_excluded = int((~inlier_mask).sum())
        pct_excluded = 100.0 * n_excluded / n_total
        print(f"  Outlier removal ({outlier_iqr_factor}×IQR): "
              f"{n_excluded}/{n_total} points excluded ({pct_excluded:.1f}%)")

        # Refit on cleaned data
        if len(time_clean) < 3:
            print(f"  Too few points after outlier removal — skipping.")
            continue

        popt, t_half, fitted_curve, r2 = fit_half_life(
            time_clean, absorbance_clean, switch=switch, A_inf_manual=A_inf_man
        )
        if t_half is None:
            print(f"  Fitting failed after outlier removal — skipping.")
            continue

        print(f"  Half-life: {t_half:.2f} s   R² = {r2:.6f}")

        # Shared filename stem for any outputs from this channel
        file_stem = f"{csv_file.stem}_{label}_T{temperature_value}C"

        # Plot: orange = inliers, red x = outliers, red dashed = fit
        # show=False so we control when the window appears and can still save fig afterwards
        fig, _ = plot_half_life_with_linear(
            time, absorbance,
            start_idx=start_idx, end_idx=end_idx,
            time_sel=time_clean, absorbance_sel=absorbance_clean,
            time_outliers=time_outliers, absorbance_outliers=absorbance_outliers,
            fitted_curve=fitted_curve, r_squared=r2,
            popt=popt, t_half=t_half, switch=switch,
            title=f"{csv_file.stem} | {label} | T={temperature_value}°C",
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
                "File": csv_file.name,
                "Wavelength": label,
                "Type": "Kinetics",
                "Temperature_C": temperature_value,
                "Switch": switch,
                "A0": popt[0],
                "A_inf": popt[1] if switch == "negative" else None,
                "k": popt[-1],
                "Half_life_s": t_half,
                "R2": r2,
            }
            append_half_life_result(result_entry, RESULTS_DIR / "half_life_master.csv")
            print("  Result saved.")
