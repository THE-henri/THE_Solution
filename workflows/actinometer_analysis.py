from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Setup directories
# -------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "actinometer" / "raw"
RESULTS_DIR    = BASE_DIR / "data" / "actinometer" / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
GRAPH_DATA_DIR = RESULTS_DIR / "graph_data"

for folder in [RAW_DIR, RESULTS_DIR, PLOTS_DIR, GRAPH_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Actinometer library
# -------------------------
# epsilon_562_M_cm   : known ε at 562 nm (L mol⁻¹ cm⁻¹) — used as reference for
#                      scaling to ε at irradiation wavelength via Beer-Lambert:
#                      ε_irr = (ε_562 / A_562) × A_irr
#                      A_562 and A_irr are taken from the middle scan of the first group
# QY_func            : function of irradiation wavelength (nm) → quantum yield;
#                      literature formula gives log10(QY), so 10** is applied
# wavelength_range_nm: valid irradiation range (nm) as (min, max)
ACTINOMETERS = {
    1: {
        "name":                "Actinometer 1",
        "wavelength_range_nm": (450, 580),
        "epsilon_562_M_cm":    1.0e4,                      # L mol⁻¹ cm⁻¹ at 562 nm
        "QY_func":             lambda lam: 10 ** (-0.796 + 133 / lam),  # literature gives log10(QY)
    },
    2: {
        "name":                "Actinometer 2",
        "wavelength_range_nm": (480, 620),
        "epsilon_562_M_cm":    1.09e4,                     # L mol⁻¹ cm⁻¹ at 562 nm
        "QY_func":             lambda lam: 10 ** (-2.67 + 526 / lam),  # literature gives log10(QY)
    },
}

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
actinometer_choice        = 2       # 1 or 2
irradiation_wavelength_nm = 579     # nm — must be within the actinometer's valid range
irradiation_time_s        = 60      # s  — duration of each irradiation interval
volume_mL                 = 2.0    # mL — cuvette volume
path_length_cm            = 1.0     # cm — optical path length
scans_per_group           = 3       # spectra recorded per measurement set
wavelength_tolerance_nm   = 1       # nm — match window when extracting absorbance

# -------------------------
# Retrieve and validate actinometer parameters
# -------------------------
act = ACTINOMETERS[actinometer_choice]

wl_min, wl_max = act["wavelength_range_nm"]
if not (wl_min <= irradiation_wavelength_nm <= wl_max):
    raise ValueError(
        f"{act['name']} is only valid from {wl_min} to {wl_max} nm; "
        f"irradiation_wavelength_nm = {irradiation_wavelength_nm} nm is out of range."
    )

QY = act["QY_func"](irradiation_wavelength_nm)

print(f"Actinometer        : {act['name']}")
print(f"λ_irr              : {irradiation_wavelength_nm} nm  (valid range: {wl_min}-{wl_max} nm)")
print(f"QY                 : {QY:.4f}")
print(f"ε_562 (reference)  : {act['epsilon_562_M_cm']:.3e} L mol-1 cm-1")
print(f"ε_irr              : calculated per file from middle scan of first group")

# -------------------------
# CSV loading helper (Cary 60 column-pair format)
# -------------------------
def load_spectra_csv(filepath):
    """
    Load a multi-scan Cary 60 CSV (column pairs: Wavelength, Abs).
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


def extract_absorbance(wl, ab, target_nm, tol_nm):
    """Mean absorbance within [target_nm ± tol_nm]; NaN if no match."""
    mask = (wl >= target_nm - tol_nm) & (wl <= target_nm + tol_nm)
    return float(ab[mask].mean()) if mask.any() else np.nan


# -------------------------
# Load CSV files
# -------------------------
csv_files = sorted(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}.")

# -------------------------
# Process each file
# -------------------------
def linear_model(x, slope, intercept):
    return slope * x + intercept

for csv_file in csv_files:
    print(f"\n{'='*60}")
    print(f"Processing {csv_file.name}")
    print(f"{'='*60}")

    try:
        scans = load_spectra_csv(csv_file)
    except Exception as e:
        print(f"  Could not read {csv_file.name}: {e}")
        continue

    n_scans = len(scans)
    print(f"  {n_scans} scans loaded.")

    if n_scans < scans_per_group:
        print(f"  Fewer than {scans_per_group} scans — skipping.")
        continue

    n_groups = n_scans // scans_per_group
    print(f"  {n_groups} measurement groups ({scans_per_group} scans each).")

    # -------------------------
    # ε at irradiation wavelength — from middle scan of first group (index 1)
    # ε_irr = (ε_562 / A_562) × A_irr   (Beer-Lambert scaling at same concentration)
    # -------------------------
    mid_scan = scans[1]
    A_562 = extract_absorbance(mid_scan[0], mid_scan[1], 562, wavelength_tolerance_nm)
    A_irr = extract_absorbance(mid_scan[0], mid_scan[1], irradiation_wavelength_nm, wavelength_tolerance_nm)

    if np.isnan(A_562) or A_562 <= 0 or np.isnan(A_irr):
        print(f"  Cannot compute ε: A_562={A_562}, A_irr={A_irr} — skipping.")
        continue

    epsilon   = act["epsilon_562_M_cm"] * A_irr / A_562
    V_m3      = volume_mL * 1e-6
    epsilon_SI = epsilon * 0.1          # L mol⁻¹ cm⁻¹ → m² mol⁻¹
    l_m        = path_length_cm * 1e-2  # cm → m
    prefactor  = -V_m3 / (epsilon_SI * QY * l_m)   # mol

    print(f"  A_562 = {A_562:.4f}   A_irr ({irradiation_wavelength_nm} nm) = {A_irr:.4f}")
    print(f"  ε_irr = {epsilon:.3e} L mol-1 cm-1  "
          f"(= ε_562 × A_irr / A_562 = {act['epsilon_562_M_cm']:.3e} × {A_irr:.4f} / {A_562:.4f})")
    print(f"  prefactor = {prefactor:.4e} mol")

    # Average scans within each group → one absorbance per group
    absorbances = []
    for g in range(n_groups):
        group_abs = [
            extract_absorbance(scans[g * scans_per_group + s][0],
                               scans[g * scans_per_group + s][1],
                               irradiation_wavelength_nm, wavelength_tolerance_nm)
            for s in range(scans_per_group)
        ]
        valid_abs = [v for v in group_abs if not np.isnan(v)]
        absorbances.append(np.mean(valid_abs) if valid_abs else np.nan)

    time_axis = np.arange(n_groups) * irradiation_time_s   # cumulative irradiation time (s)

    # -------------------------
    # Compute rate function
    # y_i = prefactor × [log10(10^A_i − 1) − log10(10^A_0 − 1)]
    # -------------------------
    A_0 = absorbances[0]
    if np.isnan(A_0) or (10**A_0 - 1) <= 0:
        print(f"  A_0 = {A_0} — cannot compute log10(10^A_0 − 1). Skipping.")
        continue

    log_ref = np.log10(10**A_0 - 1)

    t_valid   = []
    y_vals    = []
    abs_valid = []

    for i, A_i in enumerate(absorbances):
        if np.isnan(A_i):
            continue
        val = 10**A_i - 1
        if val <= 0:
            print(f"  Group {i} (t={time_axis[i]:.0f} s): 10^A − 1 ≤ 0 — skipping point.")
            continue
        y_vals.append(prefactor * (np.log10(val) - log_ref))
        t_valid.append(time_axis[i])
        abs_valid.append(A_i)

    t_valid   = np.array(t_valid)
    y_vals    = np.array(y_vals)
    abs_valid = np.array(abs_valid)

    if len(t_valid) < 2:
        print(f"  Not enough valid points — skipping.")
        continue

    # -------------------------
    # Linear fit — slope = photon flux (mol s⁻¹)
    # -------------------------
    popt, pcov      = curve_fit(linear_model, t_valid, y_vals)
    slope, intercept      = popt
    slope_std, _          = np.sqrt(np.diag(pcov))

    photon_flux     = slope
    photon_flux_std = slope_std

    y_pred = linear_model(t_valid, slope, intercept)
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"\n  Photon flux : {photon_flux:.4e} ± {photon_flux_std:.4e} mol s⁻¹")
    print(f"  R²          : {r2:.6f}")

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t_valid, y_vals, 'o', color='black', markerfacecolor='none',
            markeredgewidth=1.2, label='Data')

    t_fit = np.linspace(0, t_valid.max() * 1.02, 200)
    ax.plot(t_fit, linear_model(t_fit, slope, intercept),
            '--', color='red', linewidth=2, label='Linear fit')

    annotation = (
        r"$y = \frac{-V}{\varepsilon\,\Phi\,l}"
        r"\left[\log_{10}(10^{A}-1) - \log_{10}(10^{A_0}-1)\right]$" + "\n"
        f"Photon flux = {photon_flux:.4e} ± {photon_flux_std:.4e} mol s$^{{-1}}$\n"
        f"$R^2$ = {r2:.4f}"
    )
    ax.text(0.03, 0.97, annotation,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.set_xlabel("Irradiation time (s)")
    ax.set_ylabel("Rate function (mol)")
    ax.set_title(
        f"{act['name']} | λ$_{{irr}}$ = {irradiation_wavelength_nm} nm | {csv_file.stem}"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Save image ---
    file_stem = f"{csv_file.stem}_{act['name'].replace(' ', '_')}_{irradiation_wavelength_nm}nm"
    if input("  Save image? (y/n): ").strip().lower() == "y":
        img_path = PLOTS_DIR / f"{file_stem}.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        print(f"  Image saved → {img_path.relative_to(BASE_DIR)}")

    plt.close(fig)

    # --- Save graph data ---
    if input("  Save graph data as CSV? (y/n): ").strip().lower() == "y":
        df_graph = pd.DataFrame({
            "Time_s":         t_valid,
            "Absorbance":     abs_valid,
            "Rate_function":  y_vals,
            "Fitted_rate":    y_pred,
        })
        data_path = GRAPH_DATA_DIR / f"{file_stem}_data.csv"
        df_graph.to_csv(data_path, index=False)
        print(f"  Graph data saved → {data_path.relative_to(BASE_DIR)}")

    # --- Save photon flux result ---
    if input("  Save photon flux result? (y/n): ").strip().lower() == "y":
        result = {
            "File":                  csv_file.name,
            "Actinometer":           act["name"],
            "Irradiation_nm":        irradiation_wavelength_nm,
            "QY":                    QY,
            "Epsilon_M_cm":          epsilon,
            "Volume_mL":             volume_mL,
            "Path_length_cm":        path_length_cm,
            "Photon_flux_mol_s":     photon_flux,
            "Photon_flux_std_mol_s": photon_flux_std,
            "R2":                    r2,
        }
        master_path = RESULTS_DIR / "photon_flux_master.csv"
        df_new = pd.DataFrame([result])
        if master_path.exists():
            df_existing = pd.read_csv(master_path)
            pd.concat([df_existing, df_new], ignore_index=True).to_csv(master_path, index=False)
        else:
            df_new.to_csv(master_path, index=False)
        print(f"  Result saved → {master_path.relative_to(BASE_DIR)}")
