from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Physical constants
# -------------------------
R = 8.314462      # J mol⁻¹ K⁻¹

# -------------------------
# Setup directories
# -------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
RESULTS_DIR    = BASE_DIR / "data" / "half_life" / "results"
ARRHENIUS_DIR  = BASE_DIR / "data" / "arrhenius" / "results"
PLOTS_DIR      = ARRHENIUS_DIR / "plots"

for folder in [ARRHENIUS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Workflow parameters          (all will become GUI inputs later)
# -------------------------
compound_name = "AZA-SO2Me"   # used in plot titles and output filenames
weighted_fit  = True          # True = weighted by k std error; False = unweighted

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
        f"Arrhenius analysis requires ≥2 temperatures; only {len(temp_entries)} found in master CSV."
    )

T_K_arr    = np.array([e[0] for e in temp_entries])
k_arr      = np.array([e[1] for e in temp_entries])
k_sem_list = [e[2] for e in temp_entries]

inv_T = 1.0 / T_K_arr
ln_k  = np.log(k_arr)

# -------------------------
# Arrhenius fit
# ln(k) = ln(A) - (Ea/R) * (1/T)
# slope = -Ea/R  →  Ea = -slope * R
# intercept = ln(A)  →  A = exp(intercept)
# -------------------------
def linear_model(x, slope, intercept):
    return slope * x + intercept

use_weights = weighted_fit and all(s is not None and s > 0 for s in k_sem_list)
if use_weights:
    sigma_y = np.array([s / k for s, k in zip(k_sem_list, k_arr)])
    popt, pcov = curve_fit(linear_model, inv_T, ln_k,
                           sigma=sigma_y, absolute_sigma=True)
else:
    popt, pcov = curve_fit(linear_model, inv_T, ln_k)

slope, intercept      = popt
slope_std, intcpt_std = np.sqrt(np.diag(pcov))

Ea_J      = -slope * R
Ea_std_J  = slope_std * R
Ea_kJ     = Ea_J / 1000
Ea_std_kJ = Ea_std_J / 1000

A         = np.exp(intercept)
A_std     = A * intcpt_std   # error propagation: σ_A = A · σ_{ln A}

# R² of the linearised fit
y_pred = linear_model(inv_T, slope, intercept)
ss_res = np.sum((ln_k - y_pred) ** 2)
ss_tot = np.sum((ln_k - ln_k.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print(f"\nArrhenius results ({len(temp_entries)} temperatures"
      f", {'weighted' if use_weights else 'unweighted'}):")
print(f"  Ea = {Ea_kJ:.2f} ± {Ea_std_kJ:.2f} kJ mol⁻¹")
print(f"  A  = {A:.4e} ± {A_std:.4e} s⁻¹")
print(f"  R² = {r2:.6f}")

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7, 5))

x_data = 1000.0 * inv_T   # 1000/T for readability

if use_weights:
    ax.errorbar(x_data, ln_k, yerr=sigma_y,
                fmt='o', color='black', markerfacecolor='none',
                markeredgewidth=1.2, capsize=4, elinewidth=1.2, label='Data ± σ')
else:
    ax.plot(x_data, ln_k, 'o', color='black', markerfacecolor='none',
            markeredgewidth=1.2, label='Data')

x_fit = np.linspace(x_data.min() * 0.998, x_data.max() * 1.002, 200)
ax.plot(x_fit, linear_model(x_fit / 1000.0, slope, intercept),
        '--', color='red', linewidth=2, label='Arrhenius fit')

annotation = (
    r"$\ln(k) = \ln(A) - \frac{E_a}{R}\cdot\frac{1}{T}$" + "\n"
    f"$E_a$ = {Ea_kJ:.2f} ± {Ea_std_kJ:.2f} kJ mol$^{{-1}}$\n"
    f"$A$ = {A:.4e} ± {A_std:.4e} s$^{{-1}}$\n"
    f"$R^2$ = {r2:.4f}"
)
ax.text(0.97, 0.97, annotation,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

ax.set_xlabel(r"$1000 \, / \, T \ \mathrm{(K^{-1})}$")
ax.set_ylabel(r"$\ln(k)$")
ax.set_title(f"Arrhenius Plot — {compound_name}")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Save image
# -------------------------
if input("Save image? (y/n): ").strip().lower() == "y":
    img_path = PLOTS_DIR / f"{compound_name}_Arrhenius.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"Image saved → {img_path.relative_to(BASE_DIR)}")

plt.close(fig)

# -------------------------
# Save results CSV
# -------------------------
if input("Save Arrhenius results as CSV? (y/n): ").strip().lower() == "y":
    df_out = pd.DataFrame([{
        "Compound":       compound_name,
        "Ea_kJmol":       Ea_kJ,
        "Ea_std_kJmol":   Ea_std_kJ,
        "A_s":            A,
        "A_std_s":        A_std,
        "R2_Arrhenius":   r2,
        "n_temperatures": len(temp_entries),
        "weighted":       use_weights,
    }])
    out_path = ARRHENIUS_DIR / f"{compound_name}_Arrhenius_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Results saved → {out_path.relative_to(BASE_DIR)}")
