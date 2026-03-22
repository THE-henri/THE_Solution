import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_extinction_coefficients_integer_wavelengths(
    measurements, path_length_cm, solvent="acetonitrile", temperature=25,
    compound_name=None
):
    """
    Calculate molar extinction coefficients (ε) from UV-Vis absorbance data.

    Each entry in `measurements` corresponds to one preparation (one CSV file).
    A CSV file may contain multiple replicate scans as column pairs
    (Wavelength, Abs, Wavelength, Abs, …); their ε values are averaged within
    each preparation, then all preparations are averaged for the final result.

    Parameters
    ----------
    measurements : list of dict
        Each dict must contain:
            - 'csv_file'  : path to the CSV file
            - 'weight_mg' : mass of compound weighed in mg
            - 'MW_gmol'   : molecular weight in g/mol
            - 'volume_mL' : total dissolution volume in mL
    path_length_cm : float
        Cuvette path length in cm.
    solvent : str
    temperature : float
        Temperature in °C.
    compound_name : str, optional
        Identifier stored in the output DataFrame for downstream use (e.g. QY).

    Returns
    -------
    df_eps : pandas.DataFrame
        Columns: Wavelength (nm), Prep1_Mean, Prep1_Std, …, Mean, Std,
                 Compound, Solvent, Temperature_C, Path_length_cm, Date
    """

    prep_eps_list = []

    # Determine global wavelength range safely
    min_wl, max_wl = np.inf, -np.inf
    for meas in measurements:
        csv_file = meas["csv_file"]
        try:
            # Read CSV and skip first row (instrument header)
            df = pd.read_csv(csv_file, skiprows=1, header=None)
            if df.empty:
                print(f"Warning: CSV {csv_file} is empty. Skipping.")
                continue

            # Wavelength columns: every 2nd column starting at 0
            wl_cols = df.columns[::2]
            wl_values_list = []
            for col in wl_cols:
                col_numeric = pd.to_numeric(df[col], errors="coerce")
                wl_values_list.append(col_numeric.dropna().values)

            # Flatten all wavelength values into 1D
            if wl_values_list:
                wl_all = np.concatenate(wl_values_list)
                if len(wl_all) == 0:
                    print(f"Warning: no numeric wavelength data in {csv_file}. Skipping.")
                    continue
                min_wl = min(min_wl, wl_all.min())
                max_wl = max(max_wl, wl_all.max())

        except Exception as e:
            print(f"Error reading {csv_file}: {e}. Skipping.")
            continue


    if min_wl == np.inf or max_wl == -np.inf:
        raise ValueError("No valid wavelength data found in any CSV.")

    wavelengths_int = np.arange(int(np.floor(min_wl)), int(np.ceil(max_wl)) + 1)

    # Process each preparation
    for meas in measurements:
        csv_file = meas["csv_file"]
        try:
            df = pd.read_csv(csv_file, skiprows=1, header=None)
            if df.empty:
                continue
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        weight_mg = meas.get("weight_mg")
        MW_gmol   = meas.get("MW_gmol")
        volume_mL = meas.get("volume_mL")

        instrument_eps = []

        for i in range(0, df.shape[1], 2):  # loop over Wavelength/Abs pairs
            try:
                wl = pd.to_numeric(df.iloc[:, i], errors="coerce").values
                abs_values = pd.to_numeric(df.iloc[:, i + 1], errors="coerce").values
            except IndexError:
                print(f"Skipping incomplete replicate pair in {csv_file} columns {i}-{i+1}")
                continue

            valid_idx = ~np.isnan(wl) & ~np.isnan(abs_values)
            if valid_idx.sum() == 0:
                continue

            wl, abs_values = wl[valid_idx], abs_values[valid_idx]
            sort_idx = np.argsort(wl)
            wl, abs_values = wl[sort_idx], abs_values[sort_idx]

            # Concentration in mol/L
            try:
                c_molL = (weight_mg * 1e-3) / MW_gmol / (volume_mL * 1e-3)
                eps = abs_values / (c_molL * path_length_cm)
            except Exception as e:
                print(f"Error calculating extinction coefficient for {csv_file}: {e}")
                continue

            # Interpolate to integer wavelengths
            eps_interp = np.interp(wavelengths_int, wl, eps)
            instrument_eps.append(eps_interp)

        if instrument_eps:
            instrument_eps_array = np.column_stack(instrument_eps)
            prep_mean = instrument_eps_array.mean(axis=1)
            prep_std = instrument_eps_array.std(axis=1)
            prep_eps_list.append((prep_mean, prep_std))

    if not prep_eps_list:
        raise ValueError("No valid extinction coefficient data processed.")

    # Combine all preparations
    all_means = np.column_stack([p[0] for p in prep_eps_list])
    all_stds = np.column_stack([p[1] for p in prep_eps_list])
    final_mean = all_means.mean(axis=1)
    final_std = all_means.std(axis=1)

    df_result = pd.DataFrame({"Wavelength (nm)": wavelengths_int})
    for idx, (prep_mean, prep_std) in enumerate(prep_eps_list):
        df_result[f"Prep{idx+1}_Mean"] = prep_mean
        df_result[f"Prep{idx+1}_Std"] = prep_std

    df_result["Mean"] = final_mean
    df_result["Std"] = final_std
    df_result["Compound"] = compound_name if compound_name is not None else ""
    df_result["Solvent"] = solvent
    df_result["Temperature_C"] = temperature
    df_result["Path_length_cm"] = path_length_cm
    df_result["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df_result





def plot_extinction_coefficients(df_result, show=True):
    """
    Plot extinction coefficients with error shading per preparation and overall.

    Parameters
    ----------
    df_result : pandas.DataFrame
        DataFrame returned by `calculate_extinction_coefficients_integer_wavelengths`.
        Expected columns: "Wavelength (nm)", Prep1_Mean, Prep1_Std, ..., Mean, Std
    show : bool
        If True, immediately shows the plot. Can be False for GUI integration.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    wavelengths = df_result["Wavelength (nm)"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each preparation with shading for triplicates (std)
    prep_cols = [col for col in df_result.columns if "_Mean" in col and "Prep" in col]
    for col in prep_cols:
        prep_name = col.split("_")[0]
        std_col = prep_name + "_Std"
        if std_col in df_result.columns:
            ax.fill_between(
                wavelengths,
                df_result[col] - df_result[std_col],
                df_result[col] + df_result[std_col],
                alpha=0.2
            )
        ax.plot(wavelengths, df_result[col], label=f"{prep_name}")

    # Plot overall mean ± std across preparations
    if "Mean" in df_result.columns and "Std" in df_result.columns:
        ax.fill_between(
            wavelengths,
            df_result["Mean"] - df_result["Std"],
            df_result["Mean"] + df_result["Std"],
            color="black",
            alpha=0.2,
            label="Overall ± Std"
        )
        ax.plot(wavelengths, df_result["Mean"], color="black", linewidth=2, label="Overall Mean")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction Coefficient (M^-1 cm^-1)")
    ax.set_title("Extinction Coefficients")
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig, ax
