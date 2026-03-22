from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Physical constants
# -------------------------
R       = 8.314462      # J mol⁻¹ K⁻¹
kB      = 1.380649e-23  # J K⁻¹
h       = 6.626070e-34  # J s
LN_KB_H = np.log(kB / h)  # ≈ 23.760

# -------------------------
# Setup directories
# -------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "half_life" / "results"
EYRING_DIR  = BASE_DIR / "data" / "eyring" / "results"
PLOTS_DIR   = EYRING_DIR / "plots"

for folder in [EYRING_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
compound_name    = "AZA-SO2Me"   # used in plot titles and output filenames
weighted_fit     = True  # True = weighted by k std error; False = unweighted

# -------------------------
# Load master CSV
# -------------------------
master_csv = RESULTS_DIR / "half_life_master.csv"
if not master_csv.exists():
    raise FileNotFoundError(f"Master CSV not found: {master_csv}")

df = pd.read_csv(master_csv)
print(f"Loaded {len(df)} rows from {master_csv.name}")

# -------------------------
# Group by Temperature_C — collect all k values regardless of wavelength
# Average k across all measurements at each temperature;
# use std/sqrt(n) as the error when n > 1
# -------------------------
temp_entries = []   # [(T_K, k_mean, k_sem or None, n), ...]

for T_C, grp in df.groupby("Temperature_C"):
    k_vals = grp["k"].dropna().values
    if len(k_vals) == 0:
        continue
    k_mean = k_vals.mean()
    k_sem  = float(k_vals.std(ddof=1) / np.sqrt(len(k_vals))) if len(k_vals) > 1 else None
    T_K    = float(T_C) + 273.15
    temp_entries.append((T_K, k_mean, k_sem, len(k_vals)))
    print(f"  T = {T_C} °C  →  k = {k_mean:.6f} ± "
          f"{k_sem:.6f} s⁻¹  (n = {len(k_vals)})" if k_sem is not None
          else f"  T = {T_C} °C  →  k = {k_mean:.6f} s⁻¹  (n = 1)")

temp_entries.sort()   # sort by T_K

if len(temp_entries) < 2:
    raise ValueError(
        f"Eyring analysis requires ≥2 temperatures; only {len(temp_entries)} found in master CSV."
    )

T_K_arr    = np.array([e[0] for e in temp_entries])
k_arr      = np.array([e[1] for e in temp_entries])
k_sem_list = [e[2] for e in temp_entries]
n_list     = [e[3] for e in temp_entries]

inv_T = 1.0 / T_K_arr
ln_kT = np.log(k_arr / T_K_arr)

# -------------------------
# Eyring fit
# -------------------------
def linear_model(x, slope, intercept):
    return slope * x + intercept

use_weights = weighted_fit and all(s is not None and s > 0 for s in k_sem_list)
if use_weights:
    sigma_y = np.array([s / k for s, k in zip(k_sem_list, k_arr)])
    popt, pcov = curve_fit(linear_model, inv_T, ln_kT,
                           sigma=sigma_y, absolute_sigma=True)
else:
    popt, pcov = curve_fit(linear_model, inv_T, ln_kT)

slope, intercept      = popt
slope_std, intcpt_std = np.sqrt(np.diag(pcov))

dH_kJ     = -slope * R / 1000
dH_std_kJ = slope_std * R / 1000
dS_J      = (intercept - LN_KB_H) * R
dS_std_J  = intcpt_std * R

# R² of the linearised fit
y_pred = linear_model(inv_T, slope, intercept)
ss_res = np.sum((ln_kT - y_pred) ** 2)
ss_tot = np.sum((ln_kT - ln_kT.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print(f"\nEyring results ({len(temp_entries)} temperatures"
      f", {'weighted' if use_weights else 'unweighted'}):")
print(f"  ΔH‡ = {dH_kJ:.2f} ± {dH_std_kJ:.2f} kJ mol⁻¹")
print(f"  ΔS‡ = {dS_J:.2f} ± {dS_std_J:.2f} J mol⁻¹ K⁻¹")
print(f"  R²  = {r2:.6f}")

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7, 5))

x_data = 1000.0 * inv_T   # 1000/T for readability

if use_weights:
    ax.errorbar(x_data, ln_kT, yerr=sigma_y,
                fmt='o', color='black', markerfacecolor='none',
                markeredgewidth=1.2, capsize=4, elinewidth=1.2, label='Data ± σ')
else:
    ax.plot(x_data, ln_kT, 'o', color='black', markerfacecolor='none',
            markeredgewidth=1.2, label='Data')

x_fit = np.linspace(x_data.min() * 0.998, x_data.max() * 1.002, 200)
ax.plot(x_fit, linear_model(x_fit / 1000.0, slope, intercept),
        '--', color='red', linewidth=2, label='Eyring fit')

annotation = (
    r"$\ln\!\left(\frac{k}{T}\right) = -\frac{\Delta H^\ddagger}{R}\cdot\frac{1}{T}"
    r" + \frac{\Delta S^\ddagger}{R} + \ln\!\frac{k_\mathrm{B}}{h}$" + "\n"
    f"$\\Delta H^\\ddagger$ = {dH_kJ:.2f} ± {dH_std_kJ:.2f} kJ mol$^{{-1}}$\n"
    f"$\\Delta S^\\ddagger$ = {dS_J:.2f} ± {dS_std_J:.2f} J mol$^{{-1}}$ K$^{{-1}}$\n"
    f"$R^2$ = {r2:.4f}"
)
ax.text(0.97, 0.97, annotation,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

ax.set_xlabel(r"$1000 \, / \, T \ \mathrm{(K^{-1})}$")
ax.set_ylabel(r"$\ln(k \, / \, T)$")
ax.set_title(f"Eyring Plot — {compound_name}")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Save image
# -------------------------
if input("Save image? (y/n): ").strip().lower() == "y":
    img_path = PLOTS_DIR / f"{compound_name}_Eyring.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"Image saved → {img_path.relative_to(BASE_DIR)}")

plt.close(fig)

# -------------------------
# Save results CSV
# -------------------------
if input("Save Eyring results as CSV? (y/n): ").strip().lower() == "y":
    rows = [
        {
            "Compound":       compound_name,
            "dH_kJmol":       dH_kJ,
            "dH_std_kJmol":   dH_std_kJ,
            "dS_JmolK":       dS_J,
            "dS_std_JmolK":   dS_std_J,
            "R2_Eyring":      r2,
            "n_temperatures": len(temp_entries),
            "weighted":       use_weights,
        }
    ]
    df_out   = pd.DataFrame(rows)
    out_path = EYRING_DIR / f"{compound_name}_Eyring_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Results saved → {out_path.relative_to(BASE_DIR)}")
