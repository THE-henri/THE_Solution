from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from core.optics import calculate_extinction_coefficients_integer_wavelengths
from core.plotting import plot_extinction_coefficients

# -------------------------
# Setup directories
# -------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "extinction_coefficients" / "raw"
RESULTS_DIR    = BASE_DIR / "data" / "extinction_coefficients" / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"

for folder in [RAW_DIR, RESULTS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
compound_name   = "AZA-SO2Me"    # used in plot title and saved output
path_length_cm  = 1.0            # cm
solvent         = "acetonitrile"
temperature_value = 25           # °C

# Parameters per preparation, keyed by exact filename (not full path).
# Files found in RAW_DIR that have no entry here are skipped with a warning.
# A CSV may contain multiple replicate scans (column pairs) — they are
# averaged automatically within each preparation.
#   weight_mg : mass weighed in mg
#   MW_gmol   : molecular weight in g/mol
#   volume_mL : total dissolution / dilution volume in mL
measurement_params = {
    "1-AZA-SO2Me_EC_25C.csv": {
        "weight_mg": 0.097,
        "MW_gmol":   339.4530,
        "volume_mL": 10.0,
    },
    # Add further preparations below:
    # "prep2.csv": {"weight_mg": 1.1, "MW_gmol": 339.4530, "volume_mL": 10.0},
}

# Discover files and pair with parameters
csv_files = sorted(RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}.")

measurements = []
for csv_file in csv_files:
    params = measurement_params.get(csv_file.name)
    if params is None:
        print(f"No parameters configured for '{csv_file.name}' — skipping.")
        continue
    measurements.append({"csv_file": str(csv_file), **params})

if not measurements:
    raise ValueError(
        "No files matched the configured parameters. "
        "Check that the filenames in measurement_params match the files in RAW_DIR."
    )

# -------------------------
# Calculate extinction coefficients
# -------------------------
try:
    df_eps = calculate_extinction_coefficients_integer_wavelengths(
        measurements=measurements,
        path_length_cm=path_length_cm,
        solvent=solvent,
        temperature=temperature_value,
        compound_name=compound_name,
    )
except Exception as e:
    raise RuntimeError(f"Extinction coefficient calculation failed: {e}") from e

print(f"Calculated extinction coefficients for {len(measurements)} preparation(s).")
print(f"Wavelength range: {df_eps['Wavelength (nm)'].min()}–{df_eps['Wavelength (nm)'].max()} nm")

# -------------------------
# Plot
# -------------------------
fig, ax = plot_extinction_coefficients(df_eps, show=False)
plt.show()

# -------------------------
# Save image
# -------------------------
if input("Save image? (y/n): ").strip().lower() == "y":
    img_path = PLOTS_DIR / f"{compound_name}_EC_{temperature_value}C.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"Image saved → {img_path.relative_to(BASE_DIR)}")

plt.close(fig)

# -------------------------
# Save CSV
# -------------------------
if input("Save extinction coefficient data as CSV? (y/n): ").strip().lower() == "y":
    timestamp = df_eps["Date"].iloc[0].replace(":", "-").replace(" ", "_")
    csv_path  = RESULTS_DIR / f"{compound_name}_EC_{temperature_value}C_{timestamp}.csv"
    df_eps.to_csv(csv_path, index=False)
    print(f"Data saved  → {csv_path.relative_to(BASE_DIR)}")
    print("(This file can be loaded in the QY workflow for downstream analysis.)")
