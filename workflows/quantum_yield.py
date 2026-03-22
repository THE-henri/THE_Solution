from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import savgol_filter
from lmfit import minimize, Parameters

# -------------------------
# Physical constants
# -------------------------
h_PLANCK  = 6.626070e-34  # J s
C_LIGHT   = 299792458.0   # m s⁻¹
NA        = 6.022141e+23  # mol⁻¹

# -------------------------
# Setup directories
# -------------------------
BASE_DIR              = Path(__file__).resolve().parent.parent
QY_RAW_DIR            = BASE_DIR / "data" / "quantum_yield" / "raw"
QY_INITIAL_DIR        = BASE_DIR / "data" / "quantum_yield" / "initial"
QY_RESULTS_DIR        = BASE_DIR / "data" / "quantum_yield" / "results"
QY_PLOTS_DIR          = QY_RESULTS_DIR / "plots"
QY_GRAPH_DATA_DIR     = QY_RESULTS_DIR / "graph_data"

ACTINOMETRY_DIR       = BASE_DIR / "data" / "actinometer" / "results"
HALF_LIFE_RESULTS_DIR = BASE_DIR / "data" / "half_life" / "results"
EYRING_DIR            = BASE_DIR / "data" / "eyring" / "results"
ARRHENIUS_DIR         = BASE_DIR / "data" / "arrhenius" / "results"
EC_DIR                = BASE_DIR / "data" / "extinction_coefficients" / "results"
SPECTRA_EXTRACTED_DIR = BASE_DIR / "data" / "spectra_calculation" / "results" / "extracted"

LED_EMISSION_DIR      = BASE_DIR / "data" / "led" / "emission"
LED_POWER_DIR         = BASE_DIR / "data" / "led" / "power"

for folder in [QY_RAW_DIR, QY_INITIAL_DIR, QY_RESULTS_DIR, QY_PLOTS_DIR, QY_GRAPH_DATA_DIR,
               LED_EMISSION_DIR, LED_POWER_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================
# PARAMETERS                         (all will become GUI inputs later)
# ==============================

# ----- Experimental case -----
# "A_only"        : only species A absorbs at λ_irr; fit Φ_AB
# "AB_both"       : A and B both absorb at λ_irr; fit Φ_AB and Φ_BA simultaneously
# "A_thermal_PSS" : only A absorbs at λ_irr, fast thermal backreaction; PSS algebraic method
case                      = "A_only"

# ----- Data type -----
# "kinetic"  : time-series CSV (fixed wavelength; same format as half_life_workflow)
# "scanning" : Cary 60 CSV (full spectra at each time step)
data_type                 = "kinetic"

# ----- Irradiation source -----
# "monochromator" : single wavelength, constant photon flux
# "LED"           : irradiation band with emission spectrum — structure reserved, not yet implemented
irradiation_source        = "LED"

compound_name             = "AZA-SO2Me"    # used in plot titles and saved output
temperature_C             = 25.0           # °C — sample temperature (stored in results)
solvent                   = "MeCN"         # solvent description (stored in results)

# ----- Irradiation wavelength (monochromator only) -----
irradiation_wavelength_nm = 579           # nm

# ----- Monitoring wavelengths -----
# scanning : explicit list (nm); None = user prompted interactively from available wavelengths
# kinetic  : list filters/orders channels by wavelength; None = use all channels in the CSV
monitoring_wavelengths_nm = 673   # nm — e.g. [530] or [450, 530, 600]

# ----- Photon flux -----
# Source: "manual_mol_s"  → provide photon_flux_mol_s below
#         "manual_uW"     → provide photon_flux_uW and irradiation_wavelength_nm
#         "actinometry"   → load from ACTINOMETRY_DIR / "photon_flux_master.csv"
photon_flux_source        = "actinometry"
photon_flux_mol_s         = 1.0e-9         # mol s⁻¹ (used if manual_mol_s)
photon_flux_uW            = None           # µW     (used if manual_uW)
photon_flux_std_mol_s     = 0.0            # 1σ uncertainty (mol s⁻¹); 0 = skip I₀ perturbation

# For actinometry source: optionally filter photon_flux_master.csv by irradiation wavelength.
# A bare number (e.g. 530) filters by Irradiation_nm; a dict filters by arbitrary columns
# e.g. {"Irradiation_nm": 530, "Actinometer": "Actinometer 2"}.  None = no filter (last row used).
actinometry_filter        = 579

# ----- LED irradiation source -----
# Used when irradiation_source = "LED".
# led_emission_before_file : filename in data/led/emission/ for the emission spectrum (required)
# led_emission_after_file  : second emission scan (optional; averaged with before)
# led_power_before_file    : filename in data/led/power/ for OPM power before the experiment
# led_power_after_file     : OPM power after the experiment (optional; used for drift check)
# led_power_use            : "before" | "after" | "average" — which power reading to use
# led_emission_threshold   : discard spectral points below this fraction of the peak intensity
# led_smoothing_enabled    : apply Savitzky-Golay smoothing to the emission spectrum
# led_smoothing_window     : SG window length (odd integer, must be > led_smoothing_order)
# led_smoothing_order      : SG polynomial order
# led_integration_mode     : "scalar" — use flux-weighted λ_eff and total N (fast, approximate)
#                            "full"   — integrate rate equation over full LED emission spectrum
#                                       (requires ε_A(λ) and ε_B(λ) as full spectra from
#                                        epsilon_source = "ec_results" / "ec_csv" / "spectra_results")
led_emission_before_file  = "1-Emission_590nm_600SP_575LP_before.csv"
led_emission_after_file   = None
led_power_before_file     = "1-power_timeseries-before.csv"
led_power_after_file      = "1-power_timeseries-after.csv"
led_power_use             = "before"      # "before" | "after" | "average"
led_emission_threshold    = 0.005         # discard where intensity < this × peak
led_smoothing_enabled     = True
led_smoothing_window      = 11            # SG window length (odd, > led_smoothing_order)
led_smoothing_order       = 3             # SG polynomial order
led_integration_mode      = "full"        # "scalar" | "full"

# ----- Extinction coefficients -----
# ε is needed at (1) the irradiation wavelength (drives the ODE) and
#                (2) each monitoring wavelength (converts concentrations to absorbance).
#
# "manual"          : enter ε values directly below
# "ec_results"      : auto-discover the most recent CSV in data/extinction_coefficients/results/
#                     (output of extinction_coefficients.py); optionally pin a specific file
# "ec_csv"          : provide a full path to any CSV with "Wavelength (nm)" and ε columns
# "spectra_results" : load from data/spectra_calculation/results/extracted/
#                     (output of spectra_calculation.py); columns "Species_A" / "Species_B"
#
# NOTE on ε_B at monitoring wavelengths: even in case = "A_only", B may absorb at the
# monitoring wavelength. If ε_B_mon is set to 0 when it is non-zero, the simulated
# absorbance underestimates the true value as [B] builds up, causing a systematic error
# in Φ_AB proportional to ε_B(λ_mon)/ε_A(λ_mon). Use a wavelength where only A absorbs
# (ε_B_mon ≈ 0) or provide the actual ε_B values via any of the sources below.
#
# All CSV-based sources interpolate to the exact wavelengths needed.
epsilon_source_A          = "ec_results"     # Options: "manual" | "ec_results" | "ec_csv" | "spectra_results"
epsilon_source_B          = "manual"         # Options: "manual" | "ec_results" | "ec_csv" | "spectra_results"
                                             # Applies to irr wavelength (AB_both only) AND
                                             # monitoring wavelengths (all cases — set to non-manual
                                             # to account for B absorption at monitoring λ)

# --- ec_results source ---
# Set to None to use the most recently modified CSV in data/extinction_coefficients/results/.
# Set to a filename string (not full path) to pin a specific file, e.g. "AZA_EC_25C_...csv"
epsilon_A_ec_results_file = None           # None (most recent) or filename string
epsilon_B_ec_results_file = None           # None (most recent) or filename string

# --- ec_csv source (full path) ---
epsilon_A_csv_path        = None           # str or Path to EC CSV for species A
epsilon_B_csv_path        = None           # str or Path to EC CSV for species B

# --- spectra_results source ---
# Set to None to use the most recently modified CSV in data/spectra_calculation/results/extracted/.
# Set to a filename string to pin a specific file.
epsilon_A_spectra_file    = None           # None (most recent) or filename string
epsilon_B_spectra_file    = None           # None (most recent) or filename string
# Column names in the spectra_results CSV for each species (spectra_calculation.py output).
epsilon_spectra_column_A  = "Species_A"    # default column for ε_A
epsilon_spectra_column_B  = "Species_B"    # default column for ε_B

# Column name for ec_results / ec_csv sources.
# extinction_coefficients.py saves: Prep1_Mean, Prep2_Mean, …, Mean, Std
epsilon_csv_column        = "Mean"         # e.g. "Mean" | "Prep1_Mean" | custom column name

# --- Manual ε values (L mol⁻¹ cm⁻¹) ---
epsilon_A_irr             = 12000.0        # ε_A at irradiation_wavelength_nm
epsilon_B_irr             = 0.0            # ε_B at irradiation_wavelength_nm
                                           #   set 0 for "A_only" and "A_thermal_PSS"

# ε at monitoring wavelengths — used when epsilon_source = "manual".
# None  → reuse the irradiation-wavelength ε for all monitoring wavelengths.
# dict  → {wavelength_nm: ε_value}  e.g. {450: 8000, 600: 4500}
# (ignored when epsilon_source = "ec_results", "ec_csv", or "spectra_results";
#  those interpolate automatically to every monitoring wavelength)
epsilon_A_mon_manual      = None           # None or {wl_nm: ε, ...}
epsilon_B_mon_manual      = None           # None or {wl_nm: ε, ...}

# ----- Thermal back-reaction rate -----
# Source: "none"              → k_th = 0 (no thermal back-reaction)
#         "manual"            → k_th_manual below
#         "half_life_master"  → load from half_life_master.csv, filter by temperature
#         "eyring"            → load from Eyring results CSV, compute k at k_th_temperature_C
#         "arrhenius"         → load from Arrhenius results CSV, compute k at k_th_temperature_C
k_th_source               = "half_life_master"  # "none" | "manual" | "half_life_master" | "eyring" | "arrhenius"
k_th_manual               = 0.0           # s⁻¹
k_th_manual_std           = 0.0           # s⁻¹ (1σ uncertainty)
k_th_temperature_C        = 25.0          # °C — used for half_life_master / Eyring / Arrhenius

# ----- Scanning data parameters -----
delta_t_s                 = 12.0          # s — time between consecutive spectra (or spectrum groups)
first_cycle_off           = False         # True = first spectrum was taken without irradiation (t=0 reference)
wavelength_tolerance_nm   = 2             # nm — match window for monitoring wavelength extraction
scans_per_group           = 1             # scans averaged per time point (use 1 for single scans)

# ----- Optical -----
path_length_cm            = 1.0           # cm
volume_mL                 = 2.0           # mL

# ----- Initial conditions -----
# "absorbance" : [A]₀ = A₀ / (ε_A_mon[0] × l) from first data point, first monitoring wavelength
# "manual"     : user provides initial_conc_A_manual
initial_conc_source       = "absorbance"
initial_conc_A_manual     = None          # mol L⁻¹
initial_conc_B_manual     = 2.5E-8    # mol L⁻¹ (usually 0)

# ----- PSS state (A_thermal_PSS only) -----
# "reference_wavelength"  : derive from absorbance ratio at a wavelength where only A absorbs
# "manual_fraction"       : user provides fraction of B at PSS
# "manual_absorbance"     : user provides A(λ_irr) at PSS directly
pss_source                = "reference_wavelength"
pss_reference_wavelength  = 650           # nm — only A absorbs here (reference_wavelength source)
pss_fraction_B_manual     = None          # fraction of total in B form at PSS (manual_fraction)
pss_A_abs_pss_manual      = None          # absorbance at irr wavelength at PSS (manual_absorbance)

# ----- Baseline correction / offset alignment -----
# Applied after data loading, before the ODE fit.  Works for both kinetic and scanning data.
# "none"        : no correction
# "first_point" : subtract the absorbance at the first loaded time point (per monitoring λ),
#                 i.e. shift the whole trace down so it starts at zero.
# "plateau"     : subtract the mean absorbance over a selected pre-irradiation plateau window
#                 (same idea as first_point but averaged over multiple points).
# "file"        : OFFSET ALIGNMENT — load a Cary 60 CSV from data/quantum_yield/initial/ and
#                 compute the difference between the initial spectrum and the first point of the
#                 kinetic data at each monitoring wavelength.  That difference is then ADDED to
#                 every data point in the kinetic trace, shifting the series so that t=0 aligns
#                 with the initial spectrum.  Use this when the kinetic cuvette was repositioned
#                 or there was instrument drift between the initial measurement and the kinetic run.
baseline_correction       = "file"    # Options: "none" | "first_point" | "plateau" | "file"
baseline_plateau_start_s  = None      # s — start of plateau window (None = first time point)
baseline_plateau_end_s    = None      # s — end of plateau window   (None = fit_time_start_s)
initial_spectrum_file     = "15-initial.csv"  # filename in data/quantum_yield/initial/ (used when source = "file")
offset_plateau_duration_s = 20.0             # s — average this many seconds at the start of kinetics for offset; None = first point only

# ----- Data selection for fitting -----
# Irradiation typically starts after a delay (e.g. baseline period in kinetic mode).
# Define the time window to include in the ODE fit.
# None = use all data points; the ODE time axis is always reset so t=0 = first selected point.
fit_time_start_s          = 90     # s — include points with t ≥ this value (None = auto-detect)
fit_time_end_s            = 400    # s — include points with t ≤ this value

# ----- Irradiation start auto-detection -----
# Analyses the plateau at the start of the kinetic trace and finds the first point where
# the signal deviates significantly from the plateau, reporting that as the irradiation start.
# The detected time is used as fit_time_start_s (overrides the value above when enabled).
# If detection fails, the manually set fit_time_start_s is used as a fallback.
auto_detect_irr_start     = True       # True = auto-detect; False = use fit_time_start_s only
auto_detect_n_plateau     = 20         # number of initial points used to establish plateau statistics
auto_detect_threshold     = 5.0        # detection threshold in multiples of the plateau σ
auto_detect_min_consec    = 3          # minimum consecutive out-of-plateau points to confirm start

# ----- Fitting -----
QY_AB_init                = 0.4          # initial guess Φ_AB
QY_BA_init                = 0.1           # initial guess Φ_BA (AB_both only)
QY_bounds                 = (1e-6, 1.0)   # (lower, upper) bounds applied to all QY parameters

# Unconstrained fitting: remove all bounds from Φ_AB (and Φ_BA for AB_both).
# Physically meaningless values (Φ < 0 or Φ > 1) are then allowed.
# Use this to:
#   (a) diagnose bad fits — if the unconstrained Φ is negative or > 1, the data or
#       parameters (ε, N, baseline) are likely wrong.
#   (b) electron-transfer-assisted switching — photon-driven electron transfer can in
#       principle yield Φ > 1 (each photon triggers a cascade); the unconstrained fit
#       reveals whether the data are consistent with that picture.
# Default: False  (normal bounded fitting)
QY_unconstrained          = False

# ----- Reference quantum yield -----
# When set to a float, simulate the ODE with this fixed Φ_AB and overlay the resulting
# curve on every absorbance panel.  Useful for:
#   • verifying the workflow with reference compounds of known Φ
#   • adding a physically expected Φ as a visual benchmark
#   • comparing constrained / unconstrained results against a literature value
# Set to None to disable.
QY_AB_reference           = None          # e.g. 0.08 — reference Φ_AB to show as a dashed line
QY_BA_reference           = None          # reference Φ_BA (AB_both only); None = use QY_BA_init

# ----- Control plot -----
# 1. Initial slopes method: linear fit to first n points → analytical QY estimate ([B] ≈ 0 limit).
# 2. Full-range ODE fit (scanning only): re-fit on all time points when a selection was applied,
#    to show whether the selection window affects the result.
n_initial_slopes_points   = 8             # data points (from fit window start) used for slope
show_control_plot         = True

# ----- Output -----
show_diagnostic           = True


# ==============================
# HELPER FUNCTIONS
# ==============================

def load_spectra_csv(filepath):
    """
    Load a multi-scan Cary 60 CSV (column pairs: Wavelength, Abs).
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


def load_kinetic_csv(filepath):
    """
    Load multi-wavelength kinetic CSV (same format as half_life_workflow.py).
    Row 0: channel labels (every 2 columns, e.g. "45C_672nm")
    Row 1: "Time (sec)", "Abs" repeated per channel
    Rows 2+: time / absorbance data pairs
    Returns dict {label: (time_array, abs_array)}
    """
    MIN_VALID = 5
    raw = pd.read_csv(filepath, header=None)
    label_row = raw.iloc[0]
    data = raw.iloc[2:].reset_index(drop=True)
    channels = {}
    for i in range(0, data.shape[1] - 1, 2):
        label = str(label_row.iloc[i]).strip()
        t_col = pd.to_numeric(data.iloc[:, i],     errors="coerce")
        a_col = pd.to_numeric(data.iloc[:, i + 1], errors="coerce")
        valid = t_col.notna() & a_col.notna()
        if valid.sum() < MIN_VALID:
            continue
        channels[label] = (t_col[valid].values, a_col[valid].values)
    return channels


def load_epsilon_from_csv(csv_path, target_wavelengths, column=None):
    """
    Load ε values from an EC or spectra-calculation CSV.
    Accepts both "Wavelength (nm)" (EC workflow) and "Wavelength_nm" (spectra_calculation).
    Interpolates to each target wavelength.
    Returns array of ε values (same order as target_wavelengths).
    """
    df = pd.read_csv(csv_path, comment="#")
    for _try in ("Wavelength (nm)", "Wavelength_nm"):
        if _try in df.columns:
            wl_col = _try
            break
    else:
        raise ValueError(
            f"No wavelength column ('Wavelength (nm)' or 'Wavelength_nm') found in {csv_path}. "
            f"Available: {list(df.columns)}"
        )
    col = column or "Mean"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {csv_path}. "
                         f"Available: {list(df.columns)}")
    wl_arr  = df[wl_col].values.astype(float)
    eps_arr = df[col].values.astype(float)
    # Sort ascending
    order   = np.argsort(wl_arr)
    wl_arr  = wl_arr[order]
    eps_arr = eps_arr[order]
    return np.interp(np.array(target_wavelengths, dtype=float), wl_arr, eps_arr)


def detect_irr_start(time_s, abs_data, n_plateau, threshold, min_consec):
    """
    Detect the irradiation start in a kinetic absorbance trace.

    Strategy
    --------
    1. Use the first n_plateau points to estimate the plateau mean and σ.
    2. Scan forward from point n_plateau; a run of min_consec consecutive points
       all deviating by > threshold × σ from the plateau mean triggers detection.
    3. The first point of that confirming run is the detected irradiation onset.
    4. The fit start is set to the point BEFORE the onset (idx_onset - 1) — the
       last confirmed pure-A plateau point — so the ODE begins from [B] = 0.

    The channel with the largest peak-to-peak variation is used for detection.

    Returns
    -------
    t_fit_start : float or None — time of the last pre-irradiation point (fit start)
    t_irr_onset : float or None — time of the first detected changing point
    j_ref       : int           — channel index used for detection
    plat_mean   : float         — plateau mean (detected channel)
    plat_std    : float         — plateau std (detected channel)
    idx_fit     : int or None   — index into time_s of the fit start point
    """
    n_pts, _ = abs_data.shape

    # Pick the channel with the most total variation
    j_ref = int(np.argmax(np.ptp(abs_data, axis=0)))
    trace = abs_data[:, j_ref]

    n_plat = min(n_plateau, n_pts // 2)   # never consume more than half the data
    plat   = trace[:n_plat]
    plat_mean = float(plat.mean())
    plat_std  = float(plat.std(ddof=1)) if n_plat > 1 else 0.0

    # Floor the std so a near-zero plateau doesn't give a trivial threshold
    min_std  = max(abs(plat_mean) * 0.0005, 1e-5)
    plat_std = max(plat_std, min_std)

    thresh_abs = threshold * plat_std

    consec    = 0
    idx_onset = None
    for i in range(n_plat, n_pts):
        if abs(trace[i] - plat_mean) > thresh_abs:
            consec += 1
            if consec >= min_consec and idx_onset is None:
                idx_onset = i - min_consec + 1   # first point of the confirming run
        else:
            consec = 0

    if idx_onset is None:
        return None, None, j_ref, plat_mean, plat_std, None

    # Fit starts one point before the onset — last pure-A plateau point
    idx_fit = max(idx_onset - 1, 0)

    return (float(time_s[idx_fit]), float(time_s[idx_onset]),
            j_ref, plat_mean, plat_std, idx_fit)


def uW_to_mol_s(power_uW, wavelength_nm):
    """Convert optical power in µW at a given wavelength to photon flux in mol s⁻¹."""
    lambda_m = wavelength_nm * 1e-9
    return (power_uW * 1e-6) / (h_PLANCK * C_LIGHT / lambda_m) / NA


# ==============================
# CHECKPOINT 1 — PHOTON FLUX
# ==============================
print("\n" + "=" * 60)
print("CHECKPOINT 1 — Photon flux")
print("=" * 60)

N_mol_s     = None
N_std_mol_s = photon_flux_std_mol_s

# _led_wl_arr / _led_N_arr : set in LED block; used later by full-integration ODE
_led_wl_arr = None   # wavelength grid (nm)
_led_N_arr  = None   # spectral photon flux density (mol s⁻¹ nm⁻¹)
_lam_eff    = None   # flux-weighted centroid wavelength (scalar LED mode)

if irradiation_source == "LED":
    # ---- Load emission spectrum ----
    _em_path_b = LED_EMISSION_DIR / led_emission_before_file
    if not _em_path_b.exists():
        raise FileNotFoundError(f"LED emission file not found: {_em_path_b}")
    _em_df_b  = pd.read_csv(_em_path_b, comment="#")
    _em_wl    = _em_df_b["wavelength_nm"].values.astype(float)
    _em_int   = _em_df_b["intensity_au"].values.astype(float)
    _em_ord   = np.argsort(_em_wl)
    _em_wl    = _em_wl[_em_ord]
    _em_int   = _em_int[_em_ord]
    _em_int_raw_b = _em_int.copy()   # raw before — kept for verification plot

    _em_int_raw_a = None   # raw after interpolated onto before grid (for plot)
    if led_emission_after_file is not None:
        _em_path_a  = LED_EMISSION_DIR / led_emission_after_file
        _em_df_a    = pd.read_csv(_em_path_a, comment="#")
        _em_wl_a    = _em_df_a["wavelength_nm"].values.astype(float)
        _em_int_a   = _em_df_a["intensity_au"].values.astype(float)
        _em_ord_a   = np.argsort(_em_wl_a)
        _em_int_a_i = np.interp(_em_wl, _em_wl_a[_em_ord_a], _em_int_a[_em_ord_a])
        _em_int_raw_a = _em_int_a_i.copy()   # raw after — kept for verification plot
        _em_int     = (_em_int + _em_int_a_i) / 2.0

    # ---- Optional Savitzky-Golay smoothing ----
    _em_int_pre_smooth = _em_int.copy()   # averaged but unsmoothed — for verification plot
    if led_smoothing_enabled:
        _em_int = savgol_filter(_em_int, led_smoothing_window, led_smoothing_order)

    # ---- Threshold cut and clip ----
    # Save full-range smoothed curve for the verification plot (before cut)
    _em_wl_full  = _em_wl.copy()
    _em_int_full = _em_int.copy()
    _em_peak = float(_em_int.max())
    _em_keep = _em_int >= led_emission_threshold * _em_peak
    _em_wl   = _em_wl[_em_keep]
    _em_int  = np.clip(_em_int[_em_keep], 0.0, None)

    # ---- Load power files ----
    _pwr_path_b  = LED_POWER_DIR / led_power_before_file
    if not _pwr_path_b.exists():
        raise FileNotFoundError(f"LED power file not found: {_pwr_path_b}")
    _pwr_df_b    = pd.read_csv(_pwr_path_b, comment="#")
    _P_before_mW = float(_pwr_df_b["power_mW"].mean())

    _pwr_df_a   = None
    _P_after_mW = None
    if led_power_after_file is not None:
        _pwr_path_a = LED_POWER_DIR / led_power_after_file
        if _pwr_path_a.exists():
            _pwr_df_a   = pd.read_csv(_pwr_path_a, comment="#")
            _P_after_mW = float(_pwr_df_a["power_mW"].mean())

    if led_power_use == "before":
        _P_mW = _P_before_mW
    elif led_power_use == "after":
        if _P_after_mW is None:
            raise ValueError("led_power_use = 'after' but no after-power file loaded.")
        _P_mW = _P_after_mW
    elif led_power_use == "average":
        _P_mW = (_P_before_mW + _P_after_mW) / 2.0 if _P_after_mW is not None else _P_before_mW
    else:
        raise ValueError(f"Unknown led_power_use: '{led_power_use}'")

    _P_W = _P_mW * 1e-3   # mW → W

    print(f"  Source    : LED")
    print(f"  P before  : {_P_before_mW:.4f} mW  ({led_power_before_file})")
    if _P_after_mW is not None:
        _drift_pct = (_P_after_mW - _P_before_mW) / _P_before_mW * 100.0
        print(f"  P after   : {_P_after_mW:.4f} mW  ({led_power_after_file})  "
              f"drift = {_drift_pct:+.2f}%")
    print(f"  P used    : {_P_mW:.4f} mW  (led_power_use = '{led_power_use}')")

    # ---- Verification plot 1: emission spectrum ----
    _n_em_rows = 1 + (1 if led_smoothing_enabled else 0)
    _fig_em_raw, _axes_em = plt.subplots(
        _n_em_rows, 1,
        figsize=(7, 3.5 * _n_em_rows),
        constrained_layout=True,
        squeeze=False,
    )
    _ax_em_top = _axes_em[0, 0]

    # Raw spectra
    _ax_em_top.plot(_em_wl_full, _em_int_raw_b, color="#888888", linewidth=1.0,
                    alpha=0.7, label=f"Raw before  ({led_emission_before_file})")
    if _em_int_raw_a is not None:
        _ax_em_top.plot(_em_wl_full, _em_int_raw_a, color="#5599cc", linewidth=1.0,
                        alpha=0.7, label=f"Raw after  ({led_emission_after_file})")
    # Processed (smoothed if enabled, then cut)
    _ax_em_top.plot(_em_wl_full, _em_int_full, color="#e84d0e", linewidth=1.8,
                    label="Smoothed" if led_smoothing_enabled else "Processed")
    # Mark threshold cut boundaries
    _em_thr_val = led_emission_threshold * float(_em_int_full.max())
    _ax_em_top.axhline(_em_thr_val, color="navy", linewidth=1.0, linestyle=":",
                       label=f"Threshold ({led_emission_threshold*100:.1f}% of peak)")
    _ax_em_top.axvspan(_em_wl[0], _em_wl[-1], alpha=0.10, color="green",
                       label=f"Kept region ({_em_wl[0]:.0f}–{_em_wl[-1]:.0f} nm)")
    _ax_em_top.set_ylabel("Intensity (a.u.)")
    _ax_em_top.set_title("LED emission spectrum — raw & processed")
    _ax_em_top.legend(fontsize=8)
    _ax_em_top.grid(True, alpha=0.4)

    # Optional smoothing detail panel
    if led_smoothing_enabled:
        _ax_em_bot = _axes_em[1, 0]
        _ax_em_bot.plot(_em_wl_full, _em_int_pre_smooth, color="#888888", linewidth=1.0,
                        alpha=0.8, label="Before smoothing (averaged)")
        _ax_em_bot.plot(_em_wl_full, _em_int_full, color="#e84d0e", linewidth=1.8,
                        label=f"After SG smoothing  "
                              f"(window={led_smoothing_window}, order={led_smoothing_order})")
        _ax_em_bot.set_ylabel("Intensity (a.u.)")
        _ax_em_bot.set_title("Smoothing detail")
        _ax_em_bot.legend(fontsize=8)
        _ax_em_bot.grid(True, alpha=0.4)

    _axes_em[-1, 0].set_xlabel("Wavelength (nm)")
    plt.show()
    if input("  Emission spectrum OK? (y/n): ").strip().lower() != "y":
        plt.close(_fig_em_raw)
        raise SystemExit("Aborted at LED emission verification.")
    plt.close(_fig_em_raw)

    # ---- Verification plot 2: power time series ----
    _n_pwr_panels = 1 + (1 if _pwr_df_a is not None else 0)
    _fig_pwr, _axes_pwr = plt.subplots(
        _n_pwr_panels, 1,
        figsize=(7, 3.0 * _n_pwr_panels),
        constrained_layout=True,
        squeeze=False,
    )

    _ax_pwr_b = _axes_pwr[0, 0]
    _t_b = _pwr_df_b["time_s"].values
    _p_b = _pwr_df_b["power_mW"].values
    _ax_pwr_b.plot(_t_b, _p_b, color="#3a7ebf", linewidth=1.2, alpha=0.8,
                   label=led_power_before_file)
    _ax_pwr_b.axhline(_P_before_mW, color="red", linewidth=1.2, linestyle="--",
                      label=f"Mean = {_P_before_mW:.4f} mW")
    _ax_pwr_b.set_ylabel("Power (mW)")
    _ax_pwr_b.set_title("LED power — before")
    _ax_pwr_b.legend(fontsize=8)
    _ax_pwr_b.grid(True, alpha=0.4)

    if _pwr_df_a is not None:
        _ax_pwr_a = _axes_pwr[1, 0]
        _t_a = _pwr_df_a["time_s"].values
        _p_a = _pwr_df_a["power_mW"].values
        _ax_pwr_a.plot(_t_a, _p_a, color="#e87d37", linewidth=1.2, alpha=0.8,
                       label=led_power_after_file)
        _ax_pwr_a.axhline(_P_after_mW, color="red", linewidth=1.2, linestyle="--",
                          label=f"Mean = {_P_after_mW:.4f} mW")
        _ax_pwr_a.set_ylabel("Power (mW)")
        _ax_pwr_a.set_title(f"LED power — after  (drift = {_drift_pct:+.2f}%)")
        _ax_pwr_a.legend(fontsize=8)
        _ax_pwr_a.grid(True, alpha=0.4)

    _axes_pwr[-1, 0].set_xlabel("Time (s)")
    plt.show()
    if input("  Power time series OK? (y/n): ").strip().lower() != "y":
        plt.close(_fig_pwr)
        raise SystemExit("Aborted at LED power verification.")
    plt.close(_fig_pwr)

    # ---- Compute spectral photon flux density N(λ) [mol s⁻¹ nm⁻¹] ----
    # N(λ) = normalised_shape(λ) × P_W / E_photon(λ) / NA
    _em_wl_m   = _em_wl * 1e-9                         # nm → m
    _E_ph      = h_PLANCK * C_LIGHT / _em_wl_m         # J/photon at each λ
    _em_norm   = _em_int / np.trapezoid(_em_int, _em_wl)   # normalised shape (nm⁻¹)
    _led_N_arr  = _em_norm * _P_W / _E_ph / NA          # mol s⁻¹ nm⁻¹
    _led_wl_arr = _em_wl.copy()
    N_mol_s     = float(np.trapezoid(_led_N_arr, _led_wl_arr))   # total mol s⁻¹

    if led_integration_mode == "scalar":
        _lam_eff = float(np.trapezoid(_led_wl_arr * _led_N_arr, _led_wl_arr) / N_mol_s)
        irradiation_wavelength_nm = _lam_eff   # override for downstream ε lookups
        print(f"  Mode      : scalar  (λ_eff = {_lam_eff:.1f} nm, flux-weighted centroid)")
        print(f"  N_total   : {N_mol_s:.4e} mol s⁻¹")
    elif led_integration_mode == "full":
        print(f"  Mode      : full spectral integration")
        print(f"  LED range : {_led_wl_arr[0]:.0f} – {_led_wl_arr[-1]:.0f} nm  "
              f"({len(_led_wl_arr)} points)")
        print(f"  N_total   : {N_mol_s:.4e} mol s⁻¹")
    else:
        raise ValueError(f"Unknown led_integration_mode: '{led_integration_mode}'")

    # I₀ uncertainty: use half the absolute power drift as 1σ if both power files
    # are available and photon_flux_std_mol_s has not been set manually (= 0).
    # σ_N / N = σ_P / P  →  σ_N = N × |P_after − P_before| / (2 × P_used)
    if photon_flux_std_mol_s > 0.0:
        N_std_mol_s = photon_flux_std_mol_s
        print(f"  N std     : {N_std_mol_s:.4e} mol s⁻¹  (manual)")
    elif _P_after_mW is not None:
        N_std_mol_s = N_mol_s * abs(_P_after_mW - _P_before_mW) / (2.0 * _P_mW)
        print(f"  N std     : {N_std_mol_s:.4e} mol s⁻¹  "
              f"(auto from power drift: |{_P_after_mW:.4f} − {_P_before_mW:.4f}| / 2 / {_P_mW:.4f} mW)")
    else:
        N_std_mol_s = 0.0
        print(f"  N std     : 0  (no after-power file; set photon_flux_std_mol_s manually if needed)")

    # ---- Plot LED emission spectrum ----
    _fig_em, _ax_em = plt.subplots(figsize=(7, 4))
    _ax_em.plot(_led_wl_arr, _led_N_arr * 1e12, color="#e84d0e", linewidth=1.5,
                label=f"N(λ)  |  P = {_P_mW:.3f} mW")
    if led_integration_mode == "scalar" and _lam_eff is not None:
        _ax_em.axvline(_lam_eff, color="navy", linewidth=1.4, linestyle="--",
                       label=f"λ_eff = {_lam_eff:.1f} nm")
    _ax_em.set_xlabel("Wavelength (nm)")
    _ax_em.set_ylabel("Spectral photon flux (pmol s$^{-1}$ nm$^{-1}$)")
    _ax_em.set_title(f"LED emission — {led_emission_before_file}\n"
                     f"N_total = {N_mol_s:.4e} mol s⁻¹")
    _ax_em.legend(fontsize=9)
    _ax_em.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    if input("  LED emission correct? (y/n): ").strip().lower() != "y":
        plt.close(_fig_em)
        raise SystemExit("Aborted at LED emission checkpoint.")
    plt.close(_fig_em)

else:
    # ---- Monochromator / actinometry photon flux sources ----
    if photon_flux_source == "manual_mol_s":
        N_mol_s = photon_flux_mol_s
        print(f"  Source    : manual (mol s⁻¹)")
        print(f"  N         : {N_mol_s:.4e} mol s⁻¹")

    elif photon_flux_source == "manual_uW":
        if photon_flux_uW is None:
            raise ValueError("photon_flux_source = 'manual_uW' but photon_flux_uW is None.")
        N_mol_s = uW_to_mol_s(photon_flux_uW, irradiation_wavelength_nm)
        print(f"  Source    : manual (µW)")
        print(f"  Power     : {photon_flux_uW:.4g} µW  @ {irradiation_wavelength_nm} nm")
        print(f"  N         : {N_mol_s:.4e} mol s⁻¹")
        if N_std_mol_s == 0.0:
            print("  Note: photon_flux_std_mol_s = 0 — no I₀ uncertainty propagation.")

    elif photon_flux_source == "actinometry":
        _act_csv = ACTINOMETRY_DIR / "photon_flux_master.csv"
        if not _act_csv.exists():
            raise FileNotFoundError(f"Actinometry master CSV not found: {_act_csv}")
        _df_act = pd.read_csv(_act_csv)
        if actinometry_filter is not None:
            if isinstance(actinometry_filter, dict):
                _filter_dict = actinometry_filter
            else:
                # bare number → shorthand for {"Irradiation_nm": value}
                _filter_dict = {"Irradiation_nm": actinometry_filter}
            for col, val in _filter_dict.items():
                if col in _df_act.columns:
                    _df_act = _df_act[_df_act[col] == val]
        if _df_act.empty:
            raise ValueError(
                f"No rows in {_act_csv} after applying filter {actinometry_filter}."
            )
        # Use the last row (most recent measurement)
        _row    = _df_act.iloc[-1]
        N_mol_s = float(_row["Photon_flux_mol_s"])
        if N_std_mol_s == 0.0 and "Photon_flux_std_mol_s" in _row.index:
            N_std_mol_s = float(_row["Photon_flux_std_mol_s"])
        print(f"  Source    : actinometry master CSV ({_act_csv.name})")
        print(f"  Row used  : {_row.get('File', 'unknown')}")
        print(f"  N         : {N_mol_s:.4e} mol s⁻¹")
        print(f"  N std     : {N_std_mol_s:.4e} mol s⁻¹")

    else:
        raise ValueError(f"Unknown photon_flux_source: '{photon_flux_source}'")

print(f"  N std     : {N_std_mol_s:.4e} mol s⁻¹  "
      f"({'I₀ perturbation enabled' if N_std_mol_s > 0 else 'no I₀ perturbation'})")

if input("\n  Confirm photon flux? (y/n): ").strip().lower() != "y":
    raise SystemExit("Aborted at photon flux checkpoint.")


# ==============================
# CHECKPOINT 2 — EXTINCTION COEFFICIENTS
# ==============================
print("\n" + "=" * 60)
print("CHECKPOINT 2 — Extinction coefficients")
print("=" * 60)

# We need ε at:
#   (a) irradiation wavelength  → drives the ODE
#   (b) each monitoring wavelength → maps concentrations to observable absorbance
# monitoring_wavelengths_nm may be None here; resolved later from data.
# For now, collect ε at irr wavelength only; monitoring ε are collected after
# monitoring wavelengths are confirmed in Checkpoint 4.

_eps_A_irr = None
_eps_B_irr = None

def _resolve_ec_csv_path(source, csv_path, ec_results_file, label,
                         spectra_file=None):
    """
    Return the resolved Path to the epsilon CSV for a given source.
    - "ec_results"      : auto-discover from EC_DIR; optionally pin by filename.
    - "ec_csv"          : use csv_path directly.
    - "spectra_results" : auto-discover from SPECTRA_EXTRACTED_DIR; optionally pin.
    """
    if source == "ec_results":
        candidates = sorted(EC_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in {EC_DIR}. "
                "Run extinction_coefficients.py and save the results first."
            )
        if ec_results_file is not None:
            match = next((p for p in candidates if p.name == ec_results_file), None)
            if match is None:
                raise FileNotFoundError(
                    f"File '{ec_results_file}' not found in {EC_DIR}.\n"
                    f"Available files: {[p.name for p in candidates]}"
                )
            return match
        return candidates[-1]
    elif source == "ec_csv":
        if csv_path is None:
            raise ValueError(
                f"epsilon_source_{label} = 'ec_csv' but epsilon_{label}_csv_path is None."
            )
        resolved = Path(csv_path)
        if not resolved.exists():
            raise FileNotFoundError(f"EC CSV not found: {resolved}")
        return resolved
    elif source == "spectra_results":
        candidates = sorted(SPECTRA_EXTRACTED_DIR.glob("*.csv"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(
                f"No CSV files found in {SPECTRA_EXTRACTED_DIR}. "
                "Run spectra_calculation.py and save the extracted spectrum first."
            )
        if spectra_file is not None:
            match = next((p for p in candidates if p.name == spectra_file), None)
            if match is None:
                raise FileNotFoundError(
                    f"File '{spectra_file}' not found in {SPECTRA_EXTRACTED_DIR}.\n"
                    f"Available files: {[p.name for p in candidates]}"
                )
            return match
        return candidates[-1]
    else:
        raise ValueError(f"Unknown epsilon source: '{source}'")


def _get_epsilon(source, csv_path, ec_results_file, label, irr_nm,
                 spectra_file=None, spectra_col=None):
    """
    Return (eps_irr, resolved_path) for the given source.
    Returns None for "manual" (caller fills irr eps from the parameter block).
    """
    if source == "manual":
        return None
    resolved = _resolve_ec_csv_path(source, csv_path, ec_results_file, label,
                                    spectra_file=spectra_file)
    col = spectra_col if source == "spectra_results" else epsilon_csv_column
    eps_irr_arr = load_epsilon_from_csv(resolved, [irr_nm], col)
    return float(eps_irr_arr[0]), resolved


if epsilon_source_A == "manual":
    _eps_A_irr   = epsilon_A_irr
    _eps_A_path  = None
    print(f"  ε_A source : manual")
else:
    _eps_A_irr, _eps_A_path = _get_epsilon(
        epsilon_source_A, epsilon_A_csv_path, epsilon_A_ec_results_file,
        "A", irradiation_wavelength_nm,
        spectra_file=epsilon_A_spectra_file, spectra_col=epsilon_spectra_column_A)
    print(f"  ε_A source : {epsilon_source_A}  ({_eps_A_path.name})")

# ε_B at irradiation wavelength: only drives the ODE for AB_both.
# ε_B at monitoring wavelengths: resolved for ALL cases so that B's absorption
# at the monitoring wavelength can be accounted for even in A_only / A_thermal_PSS.
if case == "AB_both":
    if epsilon_source_B == "manual":
        _eps_B_irr  = epsilon_B_irr
        _eps_B_path = None
        print(f"  ε_B source : manual")
    else:
        _eps_B_irr, _eps_B_path = _get_epsilon(
            epsilon_source_B, epsilon_B_csv_path, epsilon_B_ec_results_file,
            "B", irradiation_wavelength_nm,
            spectra_file=epsilon_B_spectra_file, spectra_col=epsilon_spectra_column_B)
        print(f"  ε_B source : {epsilon_source_B}  ({_eps_B_path.name})")
else:
    # ε_B at irr wavelength = 0 by definition for A_only / A_thermal_PSS.
    # But ε_B at monitoring wavelengths may still be non-zero — resolve the path here
    # so _resolve_mon_epsilon can interpolate to monitoring wavelengths later.
    _eps_B_irr = epsilon_B_irr   # typically 0
    if epsilon_source_B == "manual":
        _eps_B_path = None
    else:
        _, _eps_B_path = _get_epsilon(
            epsilon_source_B, epsilon_B_csv_path, epsilon_B_ec_results_file,
            "B", irradiation_wavelength_nm,
            spectra_file=epsilon_B_spectra_file, spectra_col=epsilon_spectra_column_B)
        print(f"  ε_B source (monitoring λ) : {epsilon_source_B}  ({_eps_B_path.name})")

if irradiation_source == "LED" and led_integration_mode == "full":
    print(f"\n  ε_A at {irradiation_wavelength_nm} nm  : {_eps_A_irr:.4e} L mol⁻¹ cm⁻¹"
          f"  (intermediate — will be replaced by flux-weighted value below)")
else:
    print(f"\n  ε_A at {irradiation_wavelength_nm} nm  : {_eps_A_irr:.4e} L mol⁻¹ cm⁻¹")
    print(f"  ε_B at {irradiation_wavelength_nm} nm  : {_eps_B_irr:.4e} L mol⁻¹ cm⁻¹")

# ---- LED full-integration: load ε_A(λ) and ε_B(λ) at LED wavelengths ----
# For scalar mode these are not needed (single _eps_A_irr at λ_eff is sufficient).
_led_eps_A_arr = None   # ε_A at each LED wavelength (L mol⁻¹ cm⁻¹)
_led_eps_B_arr = None   # ε_B at each LED wavelength (L mol⁻¹ cm⁻¹)

if irradiation_source == "LED" and led_integration_mode == "full":
    if _led_wl_arr is None:
        raise ValueError("LED wavelength array not set — check CHECKPOINT 1.")
    _led_wl_list = list(_led_wl_arr)

    if epsilon_source_A == "manual":
        _led_eps_A_arr = np.full(len(_led_wl_arr), _eps_A_irr)
        print(f"  LED ε_A  : uniform {_eps_A_irr:.4e} L mol⁻¹ cm⁻¹  (manual — all LED λ)")
    else:
        _col_A = epsilon_spectra_column_A if epsilon_source_A == "spectra_results" else epsilon_csv_column
        _led_eps_A_arr = load_epsilon_from_csv(_eps_A_path, _led_wl_list, _col_A)
        print(f"  LED ε_A  : loaded at {len(_led_wl_arr)} wavelengths from {_eps_A_path.name}")

    if epsilon_source_B == "manual":
        _led_eps_B_arr = np.full(len(_led_wl_arr), _eps_B_irr)
        print(f"  LED ε_B  : uniform {_eps_B_irr:.4e} L mol⁻¹ cm⁻¹  (manual — all LED λ)")
    else:
        _col_B = epsilon_spectra_column_B if epsilon_source_B == "spectra_results" else epsilon_csv_column
        _led_eps_B_arr = load_epsilon_from_csv(_eps_B_path, _led_wl_list, _col_B)
        print(f"  LED ε_B  : loaded at {len(_led_wl_arr)} wavelengths from {_eps_B_path.name}")

    # Flux-weighted effective ε_A for initial_slopes_QY and result_dict
    _eps_A_irr = float(np.trapezoid(_led_eps_A_arr * _led_N_arr, _led_wl_arr) / N_mol_s)
    _eps_B_irr = float(np.trapezoid(_led_eps_B_arr * _led_N_arr, _led_wl_arr) / N_mol_s)
    print(f"  LED flux-weighted ε_A = {_eps_A_irr:.4e} L mol⁻¹ cm⁻¹")
    print(f"  LED flux-weighted ε_B = {_eps_B_irr:.4e} L mol⁻¹ cm⁻¹")

# ---- LED overlay plot: N(λ) vs ε(λ) ----
if irradiation_source == "LED" and _led_wl_arr is not None and _eps_A_path is not None:
    # Load full ε_A spectrum from the CSV (all wavelengths in the file)
    def _load_full_ec_spectrum(path, col):
        _df_ec = pd.read_csv(path, comment="#")
        for _try in ("Wavelength (nm)", "Wavelength_nm"):
            if _try in _df_ec.columns:
                _wl_ec = _df_ec[_try].values.astype(float)
                break
        else:
            raise ValueError(f"No wavelength column found in {path}")
        _ep_ec = _df_ec[col].values.astype(float)
        _ord   = np.argsort(_wl_ec)
        return _wl_ec[_ord], _ep_ec[_ord]

    _col_A_ov = epsilon_spectra_column_A if epsilon_source_A == "spectra_results" else epsilon_csv_column
    _ov_wl_A, _ov_eps_A_full = _load_full_ec_spectrum(_eps_A_path, _col_A_ov)

    _ov_wl_B, _ov_eps_B_full = None, None
    if _eps_B_path is not None:
        _col_B_ov = epsilon_spectra_column_B if epsilon_source_B == "spectra_results" else epsilon_csv_column
        _ov_wl_B, _ov_eps_B_full = _load_full_ec_spectrum(_eps_B_path, _col_B_ov)

    _fig_ov, (_ax_left, _ax_right_N) = plt.subplots(1, 2, figsize=(14, 4))

    # ── Left panel: full ε spectrum + N(λ) shape ─────────────────────────────
    _ax_left_N = _ax_left.twinx()

    _ax_left.plot(_ov_wl_A, _ov_eps_A_full, color="#1a5276", linewidth=1.8,
                  label=r"$\varepsilon_A(\lambda)$")
    if _ov_wl_B is not None and np.any(_ov_eps_B_full > 0):
        _ax_left.plot(_ov_wl_B, _ov_eps_B_full, color="#1e8449", linewidth=1.8,
                      linestyle="--", label=r"$\varepsilon_B(\lambda)$")

    # N(λ) on right axis of left panel
    _lln1 = _ax_left_N.fill_between(_led_wl_arr, _led_N_arr * 1e12,
                                     color="#e84d0e", alpha=0.25, label="N(λ)  LED spectrum")
    _ax_left_N.plot(_led_wl_arr, _led_N_arr * 1e12, color="#e84d0e", linewidth=1.5)
    _ax_left_N.set_ylabel("Spectral photon flux (pmol s$^{-1}$ nm$^{-1}$)", color="#e84d0e")
    _ax_left_N.tick_params(axis="y", colors="#e84d0e")

    # LED wavelength range shading removed — N(λ) itself marks the range
    _led_wl_min, _led_wl_max = float(_led_wl_arr.min()), float(_led_wl_arr.max())

    _ax_left.set_xlabel("Wavelength (nm)")
    _ax_left.set_ylabel("ε (L mol⁻¹ cm⁻¹)", color="#1a5276")
    _ax_left.tick_params(axis="y", colors="#1a5276")
    _ax_left.set_title("Extinction coefficient — full spectrum")

    # combined legend
    _lhandles, _llabels = _ax_left.get_legend_handles_labels()
    _lhandles.append(_lln1)
    _llabels.append("N(λ)  LED spectrum")
    _ax_left.legend(_lhandles, _llabels, fontsize=8)
    _ax_left.grid(True, alpha=0.3)

    # ── Right panel: N(λ) fill + ε interpolated to LED range ─────────────────
    _ax_right_e = _ax_right_N.twinx()

    # N(λ) on left axis
    _rln1 = _ax_right_N.fill_between(_led_wl_arr, _led_N_arr * 1e12,
                                      color="#e84d0e", alpha=0.35, label="N(λ)  LED spectrum")
    _ax_right_N.plot(_led_wl_arr, _led_N_arr * 1e12, color="#e84d0e", linewidth=1.5)

    # ε_A interpolated onto LED wavelength grid
    _eps_A_on_led = np.interp(_led_wl_arr, _ov_wl_A, _ov_eps_A_full)
    _rln2, = _ax_right_e.plot(_led_wl_arr, _eps_A_on_led, color="#1a5276", linewidth=1.8,
                               label=r"$\varepsilon_A(\lambda)$  (LED range)")

    # ε_B interpolated onto LED wavelength grid (if non-trivial)
    _rln3 = None
    if _ov_wl_B is not None and np.any(_ov_eps_B_full > 0):
        _eps_B_on_led = np.interp(_led_wl_arr, _ov_wl_B, _ov_eps_B_full)
        _rln3, = _ax_right_e.plot(_led_wl_arr, _eps_B_on_led, color="#1e8449", linewidth=1.8,
                                   linestyle="--", label=r"$\varepsilon_B(\lambda)$  (LED range)")

    # Reference ε marker — label depends on mode
    _ov_eps_label = (f"ε_A flux-weighted = {_eps_A_irr:.0f} L mol⁻¹ cm⁻¹  (proxy)"
                     if led_integration_mode == "full"
                     else f"ε_A at λ_eff = {_eps_A_irr:.0f} L mol⁻¹ cm⁻¹  (used in ODE)")
    _ax_right_e.axhline(_eps_A_irr, color="#1a5276", linewidth=0.9, linestyle=":",
                         alpha=0.7, label=_ov_eps_label)

    # λ_eff marker (scalar mode)
    if led_integration_mode == "scalar" and _lam_eff is not None:
        _ax_right_N.axvline(_lam_eff, color="navy", linewidth=1.2, linestyle="--",
                             label=f"λ_eff = {_lam_eff:.1f} nm")

    _ax_right_N.set_xlabel("Wavelength (nm)")
    _ax_right_N.set_ylabel("Spectral photon flux (pmol s$^{-1}$ nm$^{-1}$)", color="#e84d0e")
    _ax_right_e.set_ylabel("ε (L mol⁻¹ cm⁻¹)", color="#1a5276")
    _ax_right_N.tick_params(axis="y", colors="#e84d0e")
    _ax_right_e.tick_params(axis="y", colors="#1a5276")

    _rhandles = [_rln1, _rln2] + ([_rln3] if _rln3 is not None else [])
    _rlabels  = [h.get_label() if hasattr(h, "get_label") else "N(λ)" for h in _rhandles]
    _ax_right_N.legend(_rhandles, _rlabels, fontsize=8, loc="upper right")

    _ax_right_N.set_title(
        f"N(λ) vs ε — LED range  |  N_total = {N_mol_s:.4e} mol s⁻¹\n"
        f"{'ε_A flux-weighted' if led_integration_mode == 'full' else 'ε_A at λ_eff'}"
        f" = {_eps_A_irr:.0f} L mol⁻¹ cm⁻¹  ({led_integration_mode} mode)"
    )
    _ax_right_N.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(_fig_ov)

if input("\n  Confirm extinction coefficients? (y/n): ").strip().lower() != "y":
    raise SystemExit("Aborted at extinction coefficient checkpoint.")


# ==============================
# CHECKPOINT 3 — THERMAL BACK-REACTION RATE
# ==============================
print("\n" + "=" * 60)
print("CHECKPOINT 3 — Thermal back-reaction rate (k_th)")
print("=" * 60)

R_GAS = 8.314462  # J mol⁻¹ K⁻¹
kB    = 1.380649e-23
h_ey  = 6.626070e-34

_k_th     = 0.0
_k_th_std = 0.0

if k_th_source == "none":
    _k_th     = 0.0
    _k_th_std = 0.0
    print("  k_th = 0  (no thermal back-reaction)")

elif k_th_source == "manual":
    _k_th     = k_th_manual
    _k_th_std = k_th_manual_std
    print(f"  Source : manual")
    print(f"  k_th   = {_k_th:.4e} ± {_k_th_std:.4e} s⁻¹")

elif k_th_source == "half_life_master":
    _hl_csv = HALF_LIFE_RESULTS_DIR / "half_life_master.csv"
    if not _hl_csv.exists():
        raise FileNotFoundError(f"Half-life master CSV not found: {_hl_csv}")
    _df_hl  = pd.read_csv(_hl_csv)
    _df_T   = _df_hl[_df_hl["Temperature_C"] == k_th_temperature_C]
    if _df_T.empty:
        raise ValueError(
            f"No rows at Temperature_C = {k_th_temperature_C} in {_hl_csv}. "
            f"Available temperatures: {sorted(_df_hl['Temperature_C'].unique())}"
        )
    _k_vals = _df_T["k"].dropna().values
    _k_th   = float(_k_vals.mean())
    _k_th_std = float(_k_vals.std(ddof=1) / np.sqrt(len(_k_vals))) if len(_k_vals) > 1 else 0.0
    print(f"  Source : half_life_master  (T = {k_th_temperature_C} °C, n = {len(_k_vals)})")
    print(f"  k_th   = {_k_th:.4e} ± {_k_th_std:.4e} s⁻¹")

elif k_th_source == "eyring":
    _ey_files = sorted(EYRING_DIR.glob("*_Eyring_results.csv"))
    if not _ey_files:
        raise FileNotFoundError(f"No Eyring results CSV found in {EYRING_DIR}.")
    _df_ey = pd.read_csv(_ey_files[-1])  # use most recent
    _row_ey = _df_ey.iloc[-1]
    _dH  = float(_row_ey["dH_kJmol"]) * 1000
    _dS  = float(_row_ey["dS_JmolK"])
    _T_K = k_th_temperature_C + 273.15
    _k_th = (kB * _T_K / h_ey) * np.exp(-_dH / (R_GAS * _T_K) + _dS / R_GAS)
    print(f"  Source  : Eyring  ({_ey_files[-1].name})")
    print(f"  ΔH‡ = {_dH/1000:.2f} kJ mol⁻¹,  ΔS‡ = {_dS:.2f} J mol⁻¹ K⁻¹")
    print(f"  T = {k_th_temperature_C} °C  →  k_th = {_k_th:.4e} s⁻¹")

elif k_th_source == "arrhenius":
    _ar_files = sorted(ARRHENIUS_DIR.glob("*_Arrhenius_results.csv"))
    if not _ar_files:
        raise FileNotFoundError(f"No Arrhenius results CSV found in {ARRHENIUS_DIR}.")
    _df_ar = pd.read_csv(_ar_files[-1])
    _row_ar = _df_ar.iloc[-1]
    _Ea  = float(_row_ar["Ea_kJmol"]) * 1000
    _A_f = float(_row_ar["A_s"])
    _T_K = k_th_temperature_C + 273.15
    _k_th = _A_f * np.exp(-_Ea / (R_GAS * _T_K))
    print(f"  Source  : Arrhenius  ({_ar_files[-1].name})")
    print(f"  Ea = {_Ea/1000:.2f} kJ mol⁻¹,  A = {_A_f:.4e} s⁻¹")
    print(f"  T = {k_th_temperature_C} °C  →  k_th = {_k_th:.4e} s⁻¹")

else:
    raise ValueError(f"Unknown k_th_source: '{k_th_source}'")

if input("\n  Confirm k_th? (y/n): ").strip().lower() != "y":
    raise SystemExit("Aborted at k_th checkpoint.")


# ==============================
# ODE AND FITTING FUNCTIONS
# ==============================

def initial_slopes_QY(time_s_ode, abs_data_fit, n_pts,
                      eps_A_irr_v, eps_A_mon_arr, eps_B_mon_arr,
                      N_v, V_L_v, l_v, conc_A_0_v):
    """
    Estimate QY_AB from the initial slope of absorbance at each monitoring wavelength.

    Derivation (at t=0, [B]=0):
      d(Abs_j)/dt|₀  =  −(ε_A,j − ε_B,j) · l · N/V · Φ_AB · (1 − 10^(−A₀_irr))

      ⟹  Φ_AB = −slope_j · V / [(ε_A,j − ε_B,j) · l · N · (1 − 10^(−A₀_irr))]

    Valid only when [B] ≈ 0, i.e., at small conversion.

    Parameters
    ----------
    time_s_ode   : 1-D array, ODE time axis (t[0] = 0)
    abs_data_fit : 2-D array (n_time, n_mon_wl)
    n_pts        : number of initial points to include in slope fit

    Returns
    -------
    QY_estimates : list of float (NaN if ε_diff ≈ 0)
    slopes       : list of float  (d(Abs)/dt per wavelength)
    """
    A0_irr        = conc_A_0_v * eps_A_irr_v * l_v
    absorbed_frac = (1.0 - 10.0 ** (-A0_irr)) if A0_irr > 1e-10 else A0_irr * np.log(10.0)
    n             = min(n_pts, len(time_s_ode))
    t_pts         = time_s_ode[:n]
    QY_estimates, slopes = [], []
    for j in range(len(eps_A_mon_arr)):
        slope = np.polyfit(t_pts, abs_data_fit[:n, j], 1)[0]
        slopes.append(slope)
        eps_diff = float(eps_A_mon_arr[j]) - float(eps_B_mon_arr[j])
        if abs(eps_diff) < 1.0 or absorbed_frac < 1e-12:
            QY_estimates.append(np.nan)
        else:
            QY_estimates.append(-slope * V_L_v / (eps_diff * l_v * N_v * absorbed_frac))
    return QY_estimates, slopes


def rate_equations(y, _t, QY_AB, QY_BA, eps_A_irr, eps_B_irr,  # noqa: ARG001 (_t required by odeint)
                   N_mol_s_val, V_L, k_th_val, l_cm):
    """
    General two-species photoisomerisation ODE.
    A_tot = (ε_A·[A] + ε_B·[B]) · l   (Beer-Lambert, dimensionless absorbance)
    d[A]/dt = (N/V) · l · (1 − 10^(−A_tot)) / A_tot
              · (Φ_BA·[B]·ε_B − Φ_AB·[A]·ε_A)  +  k_th·[B]

    Parameters
    ----------
    y   : [A, B]  in mol L⁻¹
    eps : L mol⁻¹ cm⁻¹
    N   : mol s⁻¹
    V_L : L
    l   : cm
    """
    A, B = y
    A_tot = (A * eps_A_irr + B * eps_B_irr) * l_cm
    if A_tot < 1e-10:
        factor = np.log(10.0)   # limit: (1 − 10^−x)/x → ln(10)  as x → 0
    else:
        factor = (1.0 - 10.0 ** (-A_tot)) / A_tot
    rate = (N_mol_s_val / V_L) * l_cm * factor * (QY_BA * B * eps_B_irr
                                                    - QY_AB * A * eps_A_irr)
    dAdt = rate + k_th_val * B
    return [dAdt, -dAdt]


def simulate_absorbance(params, time_s, conc_A_0, conc_B_0,
                        eps_A_irr, eps_B_irr,
                        eps_A_mon, eps_B_mon,
                        N_mol_s_val, V_L, k_th_val, l_cm):
    """
    Integrate the ODE and return simulated absorbance.

    Parameters
    ----------
    time_s   : 1-D array (shared time axis) or 2-D array (n_time × n_channels),
               where each column is that channel's own time axis, already reset to t=0.
    eps_A_mon, eps_B_mon : 1-D arrays of shape (n_mon_wl,)
        ε at each monitoring wavelength
    Returns
    -------
    abs_sim : ndarray, shape (n_time, n_mon_wl)
    """
    QY_AB    = params["QY_AB"].value
    QY_BA    = params["QY_BA"].value
    ode_args = (QY_AB, QY_BA, eps_A_irr, eps_B_irr, N_mol_s_val, V_L, k_th_val, l_cm)

    if time_s.ndim == 2:
        # Each column = one channel's time axis (already at t=0).
        # Solve ODE on the union of all time points, then interpolate per channel.
        all_t = np.unique(time_s.ravel())
        if all_t[0] > 1e-12:
            all_t = np.concatenate([[0.0], all_t])
        sol_all = odeint(rate_equations, [conc_A_0, conc_B_0], all_t,
                         args=ode_args, mxstep=5000)
        n_time, n_ch = time_s.shape
        abs_sim = np.zeros((n_time, n_ch))
        for j in range(n_ch):
            cA_j = np.interp(time_s[:, j], all_t, sol_all[:, 0])
            cB_j = np.interp(time_s[:, j], all_t, sol_all[:, 1])
            abs_sim[:, j] = (cA_j * eps_A_mon[j] + cB_j * eps_B_mon[j]) * l_cm
        return abs_sim
    else:
        conc  = odeint(rate_equations, [conc_A_0, conc_B_0], time_s,
                       args=ode_args, mxstep=5000)
        abs_sim = np.outer(conc[:, 0], eps_A_mon) + np.outer(conc[:, 1], eps_B_mon)
        abs_sim *= l_cm
        return abs_sim


def residuals_func(params, time_s, abs_exp,
                   conc_A_0, conc_B_0,
                   eps_A_irr, eps_B_irr,
                   eps_A_mon, eps_B_mon,
                   N_mol_s_val, V_L, k_th_val, l_cm):
    """Return flattened residuals (simulated − experimental)."""
    try:
        abs_sim = simulate_absorbance(
            params, time_s, conc_A_0, conc_B_0,
            eps_A_irr, eps_B_irr, eps_A_mon, eps_B_mon,
            N_mol_s_val, V_L, k_th_val, l_cm)
        return (abs_sim - abs_exp).flatten()
    except Exception:
        return np.full(abs_exp.size, 1e6)


def run_fit(params_init, time_s, abs_exp,
            conc_A_0, conc_B_0,
            eps_A_irr, eps_B_irr,
            eps_A_mon, eps_B_mon,
            N_val, V_L, k_th_val, l_cm):
    """Run lmfit.minimize and return MinimizerResult."""
    return minimize(
        residuals_func, params_init,
        args=(time_s, abs_exp, conc_A_0, conc_B_0,
              eps_A_irr, eps_B_irr, eps_A_mon, eps_B_mon,
              N_val, V_L, k_th_val, l_cm),
        method="leastsq",
    )


def rate_equations_led(y, _t,   # noqa: ARG001 (_t required by odeint)
                       QY_AB, QY_BA, led_wl_arr, led_N_arr,
                       eps_A_led_arr, eps_B_led_arr, V_L, k_th_val, l_cm):
    """
    LED full-integration rate equation.
    Integrates the photochemical rate over the LED emission spectrum using np.trapezoid.

    led_N_arr : mol s⁻¹ nm⁻¹ — spectral photon flux density at each LED wavelength
    """
    A, B      = y
    A_tot_arr = (A * eps_A_led_arr + B * eps_B_led_arr) * l_cm
    factor_arr = np.where(A_tot_arr < 1e-10,
                          np.log(10.0),
                          (1.0 - 10.0 ** (-A_tot_arr)) / A_tot_arr)
    rate_arr  = (led_N_arr / V_L) * l_cm * factor_arr * (
        QY_BA * B * eps_B_led_arr - QY_AB * A * eps_A_led_arr
    )
    dAdt = float(np.trapezoid(rate_arr, led_wl_arr)) + k_th_val * B
    return [dAdt, -dAdt]


def simulate_absorbance_led(params, time_s, conc_A_0, conc_B_0,
                            led_wl_arr, led_N_arr,
                            eps_A_led_arr, eps_B_led_arr,
                            eps_A_mon, eps_B_mon,
                            V_L, k_th_val, l_cm):
    """LED full-integration ODE simulation. Same return format as simulate_absorbance."""
    QY_AB    = params["QY_AB"].value
    QY_BA    = params["QY_BA"].value
    ode_args = (QY_AB, QY_BA, led_wl_arr, led_N_arr,
                eps_A_led_arr, eps_B_led_arr, V_L, k_th_val, l_cm)
    if time_s.ndim == 2:
        all_t = np.unique(time_s.ravel())
        if all_t[0] > 1e-12:
            all_t = np.concatenate([[0.0], all_t])
        sol_all = odeint(rate_equations_led, [conc_A_0, conc_B_0], all_t,
                         args=ode_args, mxstep=5000)
        n_time, n_ch = time_s.shape
        abs_sim = np.zeros((n_time, n_ch))
        for j in range(n_ch):
            cA_j = np.interp(time_s[:, j], all_t, sol_all[:, 0])
            cB_j = np.interp(time_s[:, j], all_t, sol_all[:, 1])
            abs_sim[:, j] = (cA_j * eps_A_mon[j] + cB_j * eps_B_mon[j]) * l_cm
        return abs_sim
    else:
        conc    = odeint(rate_equations_led, [conc_A_0, conc_B_0], time_s,
                         args=ode_args, mxstep=5000)
        abs_sim = np.outer(conc[:, 0], eps_A_mon) + np.outer(conc[:, 1], eps_B_mon)
        abs_sim *= l_cm
        return abs_sim


def residuals_func_led(params, time_s, abs_exp,
                       conc_A_0, conc_B_0,
                       led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
                       eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm):
    """Return flattened residuals for LED full-integration fit."""
    try:
        abs_sim = simulate_absorbance_led(
            params, time_s, conc_A_0, conc_B_0,
            led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
            eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm)
        return (abs_sim - abs_exp).flatten()
    except Exception:
        return np.full(abs_exp.size, 1e6)


def run_fit_led(params_init, time_s, abs_exp,
                conc_A_0, conc_B_0,
                led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
                eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm):
    """Run lmfit.minimize for LED full-integration mode. Returns MinimizerResult."""
    return minimize(
        residuals_func_led, params_init,
        args=(time_s, abs_exp, conc_A_0, conc_B_0,
              led_wl_arr, led_N_arr, eps_A_led_arr, eps_B_led_arr,
              eps_A_mon, eps_B_mon, V_L, k_th_val, l_cm),
        method="leastsq",
    )


def pss_algebraic(k_th_val, conc_B_pss, volume_L, N_val, A_abs_pss):
    """
    Algebraic quantum yield from the photostationary state:
      QY_AB = k_th · [B]_PSS · V / (N · (1 − 10^(−A_PSS(λ_irr))))
    """
    denom = N_val * (1.0 - 10.0 ** (-A_abs_pss))
    if denom <= 0:
        raise ValueError(f"PSS denominator ≤ 0: N={N_val:.4e}, A_PSS={A_abs_pss:.4f}")
    return k_th_val * conc_B_pss * volume_L / denom


# ==============================
# MAIN FILE LOOP
# ==============================
csv_files = sorted(QY_RAW_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {QY_RAW_DIR}.")

V_L = volume_mL / 1000.0   # mL → L

for csv_file in csv_files:
    print(f"\n{'=' * 60}")
    print(f"Processing: {csv_file.name}")
    print(f"{'=' * 60}")

    # ==============================
    # CHECKPOINT 4 — LOAD EXPERIMENTAL DATA
    # ==============================
    print("\n--- CHECKPOINT 4: Load experimental data ---")

    time_s    = None   # 1-D array (n_time,) — first channel's time axis (reference)
    abs_data  = None   # 2-D array (n_time, n_mon_wl) — filled after wavelength selection
    _kin_t2d  = None   # 2-D array (n_time, n_channels) — per-channel times (kinetic only)

    if data_type == "scanning":
        try:
            scans = load_spectra_csv(csv_file)
        except Exception as e:
            print(f"  ERROR reading {csv_file.name}: {e}. Skipping.")
            continue

        n_scans = len(scans)
        print(f"  {n_scans} scans loaded.")
        if n_scans < scans_per_group:
            print(f"  Fewer than {scans_per_group} scans — skipping.")
            continue

        n_groups = n_scans // scans_per_group
        print(f"  {n_groups} time points  ({scans_per_group} scan(s) averaged per time point).")

        # Build time axis
        time_s = np.arange(n_groups, dtype=float) * delta_t_s
        if first_cycle_off:
            print(f"  first_cycle_off = True: time[0] = 0 s (no irradiation).")

        # Show available wavelength range
        _wl_example = scans[0][0]
        print(f"  Wavelength range: {_wl_example.min():.0f} – {_wl_example.max():.0f} nm")

        # Resolve monitoring wavelengths interactively if not specified
        _mon_wls = monitoring_wavelengths_nm
        if _mon_wls is None:
            print("  monitoring_wavelengths_nm = None — enter wavelengths manually.")
            _input = input("  Enter monitoring wavelengths (nm), comma-separated: ").strip()
            _mon_wls = [float(x) for x in _input.split(",")]

        print(f"  Monitoring wavelengths: {_mon_wls} nm")

        # Average scans within each group, extract absorbance at monitoring wavelengths
        abs_data = np.full((n_groups, len(_mon_wls)), np.nan)
        for g in range(n_groups):
            group_abs = np.zeros(len(_mon_wls))
            n_valid   = np.zeros(len(_mon_wls), dtype=int)
            for s in range(scans_per_group):
                wl_s, ab_s = scans[g * scans_per_group + s]
                for j, wl_nm in enumerate(_mon_wls):
                    val = extract_absorbance(wl_s, ab_s, wl_nm, wavelength_tolerance_nm)
                    if not np.isnan(val):
                        group_abs[j] += val
                        n_valid[j]   += 1
            for j in range(len(_mon_wls)):
                if n_valid[j] > 0:
                    abs_data[g, j] = group_abs[j] / n_valid[j]

        # Check for NaN columns
        for j, wl_nm in enumerate(_mon_wls):
            n_nan = np.isnan(abs_data[:, j]).sum()
            if n_nan == n_groups:
                print(f"  WARNING: No valid absorbance found at {wl_nm} nm. "
                      f"Increase wavelength_tolerance_nm or check CSV.")
            elif n_nan > 0:
                print(f"  Note: {n_nan} NaN time points at {wl_nm} nm — will be excluded.")

    elif data_type == "kinetic":
        try:
            channels = load_kinetic_csv(csv_file)
        except Exception as e:
            print(f"  ERROR reading {csv_file.name}: {e}. Skipping.")
            continue

        if not channels:
            print("  No channels loaded — skipping.")
            continue

        print(f"  Channels found: {list(channels.keys())}")

        # Resolve monitoring wavelengths
        # Normalise: a bare number is treated as a single-element list
        _mon_wls = monitoring_wavelengths_nm
        if isinstance(_mon_wls, (int, float)):
            _mon_wls = [float(_mon_wls)]

        if _mon_wls is None:
            # Use all channels
            selected_channels = channels
            _mon_wls = []
            for label in selected_channels:
                # Try "530nm" style first, then fall back to last bare number in the label
                m = re.search(r"(\d+(?:\.\d+)?)\s*nm", label, re.IGNORECASE)
                if not m:
                    m = re.search(r"(\d+(?:\.\d+)?)(?:\D*)$", label)
                if m:
                    _mon_wls.append(float(m.group(1)))
                else:
                    _mon_wls.append(float("nan"))
            print(f"  Using all channels: {list(channels.keys())}")
        else:
            # Filter channels: match label by wavelength if parseable, else use order
            selected_channels = {}
            for label, (t_arr, a_arr) in channels.items():
                m = re.search(r"(\d+(?:\.\d+)?)\s*nm", label, re.IGNORECASE)
                if not m:
                    m = re.search(r"(\d+(?:\.\d+)?)(?:\D*)$", label)
                ch_wl = float(m.group(1)) if m else None
                if ch_wl is not None:
                    for wl_req in _mon_wls:
                        if abs(ch_wl - wl_req) <= wavelength_tolerance_nm:
                            selected_channels[label] = (t_arr, a_arr)
                            break
            if not selected_channels:
                # Fallback: use first len(_mon_wls) channels
                print("  Could not match channels by wavelength label — using first channels.")
                items = list(channels.items())[:len(_mon_wls)]
                selected_channels = dict(items)

        print(f"  Selected channels: {list(selected_channels.keys())}")

        # Channels are measured sequentially so each has its own time axis.
        # Truncate any longer channel to the shortest length (cut from the end).
        _ch_items    = list(selected_channels.items())
        _min_len     = min(len(t) for t, _ in selected_channels.values())
        _time_arrays = []
        _abs_arrays  = []
        for _lbl, (_t, _a) in _ch_items:
            if len(_t) > _min_len:
                print(f"  Note: channel '{_lbl}' had {len(_t)} points; "
                      f"truncated to {_min_len}.")
            _time_arrays.append(_t[:_min_len])
            _abs_arrays.append(_a[:_min_len])

        # time_s  = first channel's time (used for masking, display, plotting)
        # _kin_t2d = 2-D (n_time × n_channels) — each channel's actual time axis
        time_s   = _time_arrays[0]
        _kin_t2d = np.column_stack(_time_arrays)
        abs_data = np.column_stack(_abs_arrays)
        _mon_wls = _mon_wls[:abs_data.shape[1]]

    else:
        raise ValueError(f"Unknown data_type: '{data_type}'")

    # ==============================
    # BASELINE CORRECTION / OFFSET ALIGNMENT
    # ==============================
    # _baseline_values: per-channel amount subtracted from abs_data.
    #   "first_point" / "plateau": subtract instrumental background → trace starts near zero.
    #   "file": shift trace to align with initial spectrum → _baseline_values stays zero
    #            (the shift is already baked in; corrected abs_data is used directly for [A]₀).
    # _initial_spec_abs: initial spectrum values at monitoring wavelengths (file mode only).
    _baseline_values    = np.zeros(len(_mon_wls))
    _initial_spec_abs   = None    # set in "file" branch
    _init_scans         = None    # set in "file" branch; kept for initial spectrum plot
    _offset_t_lo        = None    # start of kinetic plateau used for offset (file mode)
    _offset_t_hi        = None    # end   of kinetic plateau used for offset (file mode)

    if baseline_correction == "first_point":
        _baseline_values = abs_data[0, :].copy()
        abs_data = abs_data - _baseline_values[np.newaxis, :]
        print("  Baseline correction applied: first time point subtracted.")
        for j, wl_nm in enumerate(_mon_wls):
            print(f"    {wl_nm:.1f} nm : baseline = {_baseline_values[j]:.5f} AU")

    elif baseline_correction == "plateau":
        _plat_lo = baseline_plateau_start_s if baseline_plateau_start_s is not None else time_s[0]
        _plat_hi = (baseline_plateau_end_s  if baseline_plateau_end_s   is not None
                    else (fit_time_start_s   if fit_time_start_s         is not None
                          else time_s[0]))
        _plat_mask = (time_s >= _plat_lo) & (time_s <= _plat_hi)
        if not _plat_mask.any():
            raise ValueError(
                f"Baseline plateau window [{_plat_lo}, {_plat_hi}] s contains no data points. "
                "Adjust baseline_plateau_start_s / baseline_plateau_end_s."
            )
        _baseline_values = abs_data[_plat_mask, :].mean(axis=0)
        abs_data = abs_data - _baseline_values[np.newaxis, :]
        print(f"  Baseline correction applied: plateau mean over "
              f"{_plat_lo:.1f} – {_plat_hi:.1f} s  ({_plat_mask.sum()} points).")
        for j, wl_nm in enumerate(_mon_wls):
            print(f"    {wl_nm:.1f} nm : baseline = {_baseline_values[j]:.5f} AU")

    elif baseline_correction == "file":
        if initial_spectrum_file is None:
            raise ValueError(
                "baseline_correction = 'file' requires initial_spectrum_file to be set."
            )
        _init_path = QY_INITIAL_DIR / initial_spectrum_file
        if not _init_path.exists():
            raise FileNotFoundError(
                f"Initial spectrum file not found: {_init_path}"
            )
        try:
            _init_scans = load_spectra_csv(_init_path)
        except Exception as e:
            raise RuntimeError(f"Could not load initial spectrum '{initial_spectrum_file}': {e}")

        # Average all scans in the initial file → single absorbance per monitoring wavelength
        _init_abs = np.zeros(len(_mon_wls))
        _init_n   = np.zeros(len(_mon_wls), dtype=int)
        for _wl_arr, _ab_arr in _init_scans:
            for j, wl_nm in enumerate(_mon_wls):
                val = extract_absorbance(_wl_arr, _ab_arr, wl_nm, wavelength_tolerance_nm)
                if not np.isnan(val):
                    _init_abs[j] += val
                    _init_n[j]   += 1

        for j, wl_nm in enumerate(_mon_wls):
            if _init_n[j] == 0:
                raise ValueError(
                    f"Initial spectrum '{initial_spectrum_file}' has no valid data "
                    f"at {wl_nm} nm (tolerance ±{wavelength_tolerance_nm} nm)."
                )
            _init_abs[j] /= _init_n[j]

        # Shift the kinetic trace so that it aligns with the initial spectrum.
        # Reference value from kinetics: mean over the first offset_plateau_duration_s seconds
        # (or first point if None / no points fall in window).
        if offset_plateau_duration_s is not None:
            _t0 = time_s[0]
            _off_mask = time_s <= (_t0 + offset_plateau_duration_s)
            if not _off_mask.any():
                _off_mask[0] = True   # safety fallback
        else:
            _off_mask = np.zeros(len(time_s), dtype=bool)
            _off_mask[0] = True

        _kinetic_t0       = abs_data[_off_mask, :].mean(axis=0)   # per-channel mean
        _kinetic_t0_std   = abs_data[_off_mask, :].std(axis=0, ddof=min(1, _off_mask.sum() - 1))
        _off_n            = int(_off_mask.sum())
        _off_t_lo         = float(time_s[_off_mask][0])
        _off_t_hi         = float(time_s[_off_mask][-1])

        _offset  = _init_abs - _kinetic_t0
        abs_data = abs_data + _offset[np.newaxis, :]

        # _baseline_values stays zero: the corrected data already represents the true absorbance,
        # so abs_data_fit[0, 0] is used directly for [A]₀ (no "add back" needed).
        _initial_spec_abs  = _init_abs   # stored for display and reference
        _offset_t_lo       = _off_t_lo   # kept for preview plot
        _offset_t_hi       = _off_t_hi

        print(f"  Offset alignment applied: trace shifted to match '{initial_spectrum_file}'.")
        print(f"  ({len(_init_scans)} scan(s) averaged for initial spectrum)")
        print(f"  Kinetic reference: mean over {_off_t_lo:.1f} – {_off_t_hi:.1f} s  ({_off_n} point(s))")
        for j, wl_nm in enumerate(_mon_wls):
            print(f"    {wl_nm:.1f} nm : initial = {_init_abs[j]:.5f} AU  "
                  f"| kinetic mean = {_kinetic_t0[j]:.5f} ± {_kinetic_t0_std[j]:.5f} AU  "
                  f"| offset applied = {_offset[j]:+.5f} AU")

    elif baseline_correction != "none":
        raise ValueError(f"Unknown baseline_correction: '{baseline_correction}'")

    # Drop rows where any monitoring wavelength is NaN
    valid_rows = ~np.any(np.isnan(abs_data), axis=1)
    if not np.all(valid_rows):
        print(f"  Dropping {(~valid_rows).sum()} time points with NaN absorbance.")
    time_s   = time_s[valid_rows]
    abs_data = abs_data[valid_rows, :]
    if _kin_t2d is not None:
        _kin_t2d = _kin_t2d[valid_rows, :]

    if len(time_s) < 3:
        print("  Not enough valid time points for fitting — skipping.")
        continue

    print(f"\n  Time range : {time_s[0]:.1f} – {time_s[-1]:.1f} s  ({len(time_s)} points)")
    for j, wl_nm in enumerate(_mon_wls):
        print(f"  λ = {wl_nm:.1f} nm  :  A₀ = {abs_data[0, j]:.4f},  "
              f"A_last = {abs_data[-1, j]:.4f}")

    # ==============================
    # IRRADIATION START AUTO-DETECTION
    # ==============================
    # Effective fit start/end — may be overridden by auto-detection below.
    _fit_time_start_eff = fit_time_start_s
    _fit_time_end_eff   = fit_time_end_s

    if auto_detect_irr_start:
        _det_t_fit, _det_t_onset, _det_ch, _det_mean, _det_std, _ = detect_irr_start(
            time_s, abs_data,
            auto_detect_n_plateau, auto_detect_threshold, auto_detect_min_consec
        )
        _det_wl = _mon_wls[_det_ch] if not np.isnan(_mon_wls[_det_ch]) else _det_ch
        if _det_t_fit is not None:
            print(f"\n  Auto-detection result:")
            print(f"    Channel used    : λ = {_det_wl:.0f} nm  (largest signal variation)")
            print(f"    Plateau stats   : mean = {_det_mean:.5f} AU,  "
                  f"σ = {_det_std:.5f} AU,  "
                  f"threshold = {auto_detect_threshold} × σ = {auto_detect_threshold * _det_std:.5f} AU")
            print(f"    Irradiation onset detected at  t = {_det_t_onset:.2f} s  "
                  f"(first point exceeding threshold)")
            print(f"    Fit start set to               t = {_det_t_fit:.2f} s  "
                  f"(point before onset — last pure-A plateau point)")
            _fit_time_start_eff = _det_t_fit
        else:
            print(f"\n  Auto-detection: no irradiation start found "
                  f"(signal never exceeded {auto_detect_threshold} × σ plateau for "
                  f"{auto_detect_min_consec} consecutive points).")
            if fit_time_start_s is not None:
                print(f"    Falling back to manual fit_time_start_s = {fit_time_start_s} s")
            else:
                print("    fit_time_start_s = None — all data will be used for fitting.")

    # --- Initial spectrum plot (file offset alignment) ---
    if baseline_correction == "file" and _initial_spec_abs is not None and _init_scans is not None:
        _fig_init, _ax_init = plt.subplots(figsize=(8, 4))
        # Plot every scan in the initial spectrum file
        for _si, (_wl_arr, _ab_arr) in enumerate(_init_scans):
            _sort_idx = np.argsort(_wl_arr)
            _ax_init.plot(_wl_arr[_sort_idx], _ab_arr[_sort_idx],
                          color="#888888", linewidth=0.9, alpha=0.6,
                          label="Initial spectrum scans" if _si == 0 else "_nolegend_")
        # Mark the monitoring wavelengths with the extracted values
        _cmap_init = plt.colormaps["tab10"].resampled(len(_mon_wls))
        for j, wl_nm in enumerate(_mon_wls):
            _col = _cmap_init(j)
            _ax_init.axvline(wl_nm, color=_col, linewidth=1.0, linestyle=":", alpha=0.7)
            _ax_init.plot(wl_nm, _initial_spec_abs[j], "o", color=_col,
                          markersize=8, markeredgewidth=1.5, markerfacecolor="white",
                          zorder=5,
                          label=f"λ = {wl_nm:.0f} nm  →  A = {_initial_spec_abs[j]:.4f}")
        _ax_init.set_xlabel("Wavelength (nm)")
        _ax_init.set_ylabel("Absorbance")
        _ax_init.set_title(
            f"Initial spectrum — '{initial_spectrum_file}'  "
            f"({len(_init_scans)} scan(s) averaged)\n"
            f"Markers show values used for offset correction"
        )
        _ax_init.legend(fontsize=8)
        _ax_init.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
        if input("  Initial spectrum correct? (y/n): ").strip().lower() != "y":
            plt.close(_fig_init)
            print("  Skipping — check initial_spectrum_file setting.")
            continue
        plt.close(_fig_init)

    # --- Checkpoint 4 preview plot ---
    _n_mon_preview = len(_mon_wls)
    _fig_prev, _axes_prev = plt.subplots(
        _n_mon_preview, 1,
        figsize=(8, 2.5 * _n_mon_preview),
        constrained_layout=True,
        squeeze=False,
    )
    for j, wl_nm in enumerate(_mon_wls):
        _ax = _axes_prev[j, 0]
        _t_plot = _kin_t2d[:, j] if _kin_t2d is not None else time_s
        _ax.plot(_t_plot, abs_data[:, j], "-", color="#3a7ebf", linewidth=1.5)
        # Fit window (uses effective start — auto-detected or manual)
        if _fit_time_start_eff is not None or _fit_time_end_eff is not None:
            _sel_lo = _fit_time_start_eff if _fit_time_start_eff is not None else time_s[0]
            _sel_hi = _fit_time_end_eff   if _fit_time_end_eff   is not None else time_s[-1]
            _ax.axvspan(_sel_lo, _sel_hi, alpha=0.12, color="gray", label="Fit window")
            _ax.axvline(_sel_lo, color="gray", linewidth=0.8, linestyle="--")
            _ax.axvline(_sel_hi, color="gray", linewidth=0.8, linestyle="--")
        # Plateau baseline window
        if baseline_correction == "plateau":
            _plat_lo_prev = baseline_plateau_start_s if baseline_plateau_start_s is not None else time_s[0]
            _plat_hi_prev = (baseline_plateau_end_s  if baseline_plateau_end_s   is not None
                             else (fit_time_start_s   if fit_time_start_s         is not None
                                   else time_s[0]))
            _ax.axvspan(_plat_lo_prev, _plat_hi_prev, alpha=0.18, color="green",
                        label="Plateau (baseline)")
            _ax.axhline(_baseline_values[j], color="green", linewidth=0.9,
                        linestyle=":", label=f"Baseline = {_baseline_values[j]:.4f}")
        # Initial spectrum reference line + offset plateau (file offset alignment)
        if baseline_correction == "file" and _initial_spec_abs is not None:
            _ax.axhline(_initial_spec_abs[j], color="green", linewidth=1.2,
                        linestyle="--",
                        label=f"Initial spectrum = {_initial_spec_abs[j]:.4f}")
            if _offset_t_lo is not None and _offset_t_hi is not None:
                _ax.axvspan(_offset_t_lo, _offset_t_hi, alpha=0.18, color="green",
                            label=f"Offset plateau ({_offset_t_lo:.0f}–{_offset_t_hi:.0f} s)")
        _ax.set_ylabel("Absorbance")
        _ax.set_xlabel("Time (s)")
        _ax.set_title(f"λ = {wl_nm:.0f} nm" if not np.isnan(wl_nm) else f"Channel {j+1}")
        _ax.grid(True, alpha=0.4)
        if any(_ax.get_legend_handles_labels()[0]):
            _ax.legend(fontsize=8)
    _axes_prev[0, 0].set_title(
        f"{csv_file.stem}  —  data preview\n"
        + (_axes_prev[0, 0].get_title())
    )
    plt.show()

    if input("\n  Confirm data? (y/n): ").strip().lower() != "y":
        plt.close(_fig_prev)
        print("  Skipping.")
        continue
    plt.close(_fig_prev)

    # ==============================
    # TIME SELECTION FOR FITTING
    # ==============================
    # Build fit_mask: only time points within the selected window are passed to the ODE.
    # time_s / abs_data (full range) are preserved for the control plot.
    fit_mask = np.ones(len(time_s), dtype=bool)
    if _fit_time_start_eff is not None:
        fit_mask &= (time_s >= _fit_time_start_eff)
    if _fit_time_end_eff is not None:
        fit_mask &= (time_s <= _fit_time_end_eff)

    time_s_fit   = time_s[fit_mask]
    abs_data_fit = abs_data[fit_mask, :]

    if len(time_s_fit) < 3:
        print(f"  WARNING: Only {len(time_s_fit)} point(s) in fitting window — need ≥ 3. "
              f"fit_time_start_eff={_fit_time_start_eff}, fit_time_end_eff={_fit_time_end_eff}. Skipping.")
        continue

    # ODE time axis: t=0 at the first selected point (= irradiation start).
    # For kinetic data each channel has its own time axis; use a 2-D array for the ODE.
    # For scanning (and single-channel kinetic) use the standard 1-D axis.
    time_s_ode = time_s_fit - time_s_fit[0]   # 1-D reference (first channel / scanning)
    if _kin_t2d is not None:
        _kin_t2d_fit = _kin_t2d[fit_mask, :]
        _kin_t2d_ode = _kin_t2d_fit - _kin_t2d_fit[0, :]   # reset each channel to t=0
    else:
        _kin_t2d_fit = None
        _kin_t2d_ode = None
    _n_excluded = (~fit_mask).sum()

    print(f"\n  Fitting window : {time_s_fit[0]:.1f} – {time_s_fit[-1]:.1f} s  "
          f"({len(time_s_fit)} / {len(time_s)} time points selected)")
    if _n_excluded:
        print(f"  ({_n_excluded} time point(s) outside window — shown in control plot, "
              f"excluded from ODE fit)")

    # ==============================
    # RESOLVE ε AT MONITORING WAVELENGTHS
    # ==============================
    # Now that _mon_wls is known, resolve ε at each monitoring wavelength.
    def _resolve_mon_epsilon(source, irr_eps, mon_manual_dict, resolved_path, mon_wls,
                             species_label="A"):
        """
        Return array of ε at each monitoring wavelength.
        For "manual":           use mon_manual_dict ({wl: ε}) or fall back to irr_eps.
        For "ec_results"/       load from the resolved CSV and interpolate.
            "ec_csv":
        For "spectra_results":  load from spectra CSV using the species-specific column.
        """
        if source == "manual":
            if mon_manual_dict is None:
                return np.full(len(mon_wls), irr_eps)
            return np.array([mon_manual_dict.get(wl, irr_eps) for wl in mon_wls])
        elif source in ("ec_results", "ec_csv"):
            return load_epsilon_from_csv(resolved_path, mon_wls, epsilon_csv_column)
        elif source == "spectra_results":
            col = epsilon_spectra_column_A if species_label == "A" else epsilon_spectra_column_B
            return load_epsilon_from_csv(resolved_path, mon_wls, col)
        else:
            raise ValueError(f"Unknown epsilon source: '{source}'")

    eps_A_mon = _resolve_mon_epsilon(epsilon_source_A, _eps_A_irr,
                                     epsilon_A_mon_manual, _eps_A_path, _mon_wls,
                                     species_label="A")
    eps_B_mon = _resolve_mon_epsilon(epsilon_source_B, _eps_B_irr,
                                     epsilon_B_mon_manual, _eps_B_path, _mon_wls,
                                     species_label="B")

    print(f"\n  ε_A at monitoring wavelengths:")
    for j, wl_nm in enumerate(_mon_wls):
        print(f"    {wl_nm:.1f} nm : {eps_A_mon[j]:.4e} L mol⁻¹ cm⁻¹")
    # Always print ε_B at monitoring wavelengths (relevant for all cases)
    print(f"  ε_B at monitoring wavelengths:")
    for j, wl_nm in enumerate(_mon_wls):
        print(f"    {wl_nm:.1f} nm : {eps_B_mon[j]:.4e} L mol⁻¹ cm⁻¹")

    # ==============================
    # CHECKPOINT 5 — INITIAL CONDITIONS
    # ==============================
    print("\n--- CHECKPOINT 5: Initial conditions ---")

    if initial_conc_source == "absorbance":
        if baseline_correction == "file" and _initial_spec_abs is not None:
            # Use the initial spectrum directly — it is the best measurement of pure A
            # (taken carefully before the kinetic run; not affected by fit-window selection).
            A0_mon0 = _initial_spec_abs[0]
            print(f"  [A]₀ from initial spectrum:  "
                  f"A₀({_mon_wls[0]:.1f} nm) = {A0_mon0:.4f}  "
                  f"(from '{initial_spectrum_file}')")
            if abs(abs_data_fit[0, 0] - A0_mon0) > 1e-4:
                print(f"  Note: absorbance at fit start (t = {time_s_fit[0]:.1f} s) = "
                      f"{abs_data_fit[0, 0]:.4f}  "
                      f"(difference = {abs_data_fit[0, 0] - A0_mon0:+.4f} — "
                      f"some conversion may have occurred before irradiation start)")
        else:
            # Recover the raw (pre-correction) absorbance at the first fit-window point.
            # After first_point / plateau correction abs_data_fit[0,0] may be near zero;
            # adding back _baseline_values[0] gives the true starting absorbance.
            A0_mon0_corrected = abs_data_fit[0, 0]
            A0_mon0           = A0_mon0_corrected + _baseline_values[0]
            if _baseline_values[0] != 0.0:
                print(f"  [A]₀ from absorbance (raw):  "
                      f"A₀({_mon_wls[0]:.1f} nm) = {A0_mon0_corrected:.4f} (corrected)"
                      f" + {_baseline_values[0]:.4f} (baseline) = {A0_mon0:.4f}")
            else:
                print(f"  [A]₀ from absorbance:  A₀({_mon_wls[0]:.1f} nm) = {A0_mon0:.4f}  "
                      f"(t = {time_s_fit[0]:.1f} s)")
        conc_A_0 = A0_mon0 / (eps_A_mon[0] * path_length_cm)
        conc_B_0 = 0.0
        print(f"  ε_A = {eps_A_mon[0]:.4e}  →  [A]₀ = {conc_A_0:.4e} mol L⁻¹")
    elif initial_conc_source == "manual":
        if initial_conc_A_manual is None:
            raise ValueError("initial_conc_source = 'manual' but initial_conc_A_manual is None.")
        conc_A_0 = initial_conc_A_manual
        conc_B_0 = initial_conc_B_manual
        print(f"  [A]₀ = {conc_A_0:.4e} mol L⁻¹  (manual)")
        print(f"  [B]₀ = {conc_B_0:.4e} mol L⁻¹  (manual)")
    else:
        raise ValueError(f"Unknown initial_conc_source: '{initial_conc_source}'")

    print(f"  [A]₀ = {conc_A_0:.4e} mol L⁻¹")
    print(f"  [B]₀ = {conc_B_0:.4e} mol L⁻¹")

    # --- End-of-trace concentrations from measured absorbance ---
    _n_tail = min(5, len(abs_data))
    _t_axis = _kin_t2d if _kin_t2d is not None else np.column_stack([time_s] * len(_mon_wls))
    print(f"\n  Last {_n_tail} points — concentrations derived from absorbance (obs):")
    _hdr = f"  {'t (s)':>10}  {'Abs':>10}  {'[A]_obs':>14}  {'[B]_obs':>14}  {'[A]+[B]':>14}  (mmol/L)"
    for j, wl_nm in enumerate(_mon_wls):
        print(f"\n    lambda = {wl_nm:.0f} nm   (eps_A = {eps_A_mon[j]:.0f}  eps_B = {eps_B_mon[j]:.0f}  L mol-1 cm-1)")
        print(_hdr)
        for _i in range(len(abs_data) - _n_tail, len(abs_data)):
            _t_j   = _t_axis[_i, j]
            _abs_j = abs_data[_i, j]
            _A     = _abs_j / (eps_A_mon[j] * path_length_cm) * 1000
            _B     = conc_A_0 * 1000 - _A
            _sum   = _A + _B
            print(f"  {_t_j:>10.2f}  {_abs_j:>10.5f}  {_A:>14.4e}  {_B:>14.4e}  {_sum:>14.4e}")
        _conv = (1 - abs_data[-1, j] / (conc_A_0 * eps_A_mon[j] * path_length_cm)) * 100
        print(f"    Conversion at last point: {_conv:.1f}%")

    if input("\n  Confirm initial conditions? (y/n): ").strip().lower() != "y":
        print("  Skipping.")
        continue

    # ==============================
    # CHECKPOINT 6 — PSS STATE (A_thermal_PSS only)
    # ==============================
    pss_A_abs_val = None   # absorbance at irr wavelength at PSS
    pss_B_conc    = None   # [B]_PSS in mol L⁻¹

    if case == "A_thermal_PSS":
        print("\n--- CHECKPOINT 6: PSS state ---")

        if _k_th == 0.0:
            raise ValueError(
                "case = 'A_thermal_PSS' requires k_th > 0. "
                "Set k_th_source and check k_th value."
            )

        # Absorbance at irr wavelength at t=0
        # For scanning/kinetic, extract from monitoring wavelengths if irr wl is monitored,
        # else the user must supply pss_A_abs_pss_manual.
        _irr_idx    = next((j for j, wl in enumerate(_mon_wls)
                            if abs(wl - irradiation_wavelength_nm) <= wavelength_tolerance_nm),
                           None)

        # Initial absorbance at irr wavelength (for computing [A]_0·ε_A·l)
        if _irr_idx is not None:
            A0_irr = abs_data[0, _irr_idx]
        else:
            # Compute from [A]_0 and ε_A_irr
            A0_irr = conc_A_0 * _eps_A_irr * path_length_cm

        if pss_source == "reference_wavelength":
            # Find PSS spectrum / data point = last time point
            if data_type == "scanning":
                # Extract absorbance at reference wavelength from first and last scans
                _ref_wl = pss_reference_wavelength
                _last_grp_idx = (n_groups - 1) * scans_per_group
                A_ref_0   = extract_absorbance(
                    scans[0][0], scans[0][1], _ref_wl, wavelength_tolerance_nm)
                # Last group: average
                A_ref_pss_vals = [
                    extract_absorbance(scans[_last_grp_idx + s][0],
                                       scans[_last_grp_idx + s][1],
                                       _ref_wl, wavelength_tolerance_nm)
                    for s in range(scans_per_group)
                ]
                A_ref_pss_vals = [v for v in A_ref_pss_vals if not np.isnan(v)]
                if not A_ref_pss_vals:
                    raise ValueError(f"No absorbance found at reference wavelength "
                                     f"{_ref_wl} nm in PSS spectrum.")
                A_ref_pss = float(np.mean(A_ref_pss_vals))
            else:
                raise ValueError(
                    "pss_source = 'reference_wavelength' with kinetic data: "
                    "extract reference absorbance from data externally and use "
                    "pss_source = 'manual_absorbance' or 'manual_fraction'."
                )

            if np.isnan(A_ref_0) or A_ref_0 <= 0:
                raise ValueError(
                    f"A_ref at {pss_reference_wavelength} nm at t=0 = {A_ref_0} — invalid. "
                    "Check reference wavelength."
                )
            pss_ratio     = A_ref_pss / A_ref_0   # = [A]_PSS / [A]_0
            pss_A_abs_val = A0_irr * pss_ratio
            pss_B_conc    = conc_A_0 * (1.0 - pss_ratio)
            print(f"  Source          : reference wavelength ({pss_reference_wavelength} nm)")
            print(f"  A_ref(t=0)      = {A_ref_0:.4f}")
            print(f"  A_ref(PSS)      = {A_ref_pss:.4f}")
            print(f"  [A]_PSS / [A]_0 = {pss_ratio:.4f}  →  f_B = {1.0 - pss_ratio:.4f}")

        elif pss_source == "manual_fraction":
            if pss_fraction_B_manual is None:
                raise ValueError("pss_source = 'manual_fraction' but pss_fraction_B_manual is None.")
            pss_B_conc    = conc_A_0 * pss_fraction_B_manual
            pss_A_abs_val = A0_irr * (1.0 - pss_fraction_B_manual)
            print(f"  Source          : manual fraction")
            print(f"  f_B_PSS         = {pss_fraction_B_manual:.4f}")

        elif pss_source == "manual_absorbance":
            if pss_A_abs_pss_manual is None:
                raise ValueError("pss_source = 'manual_absorbance' but pss_A_abs_pss_manual is None.")
            pss_A_abs_val = pss_A_abs_pss_manual
            conc_A_pss    = pss_A_abs_val / (_eps_A_irr * path_length_cm)
            pss_B_conc    = conc_A_0 - conc_A_pss
            print(f"  Source          : manual absorbance")
            print(f"  A(λ_irr) at PSS = {pss_A_abs_val:.4f}")

        else:
            raise ValueError(f"Unknown pss_source: '{pss_source}'")

        print(f"  A(λ_irr) at PSS = {pss_A_abs_val:.4f}")
        print(f"  [B]_PSS         = {pss_B_conc:.4e} mol L⁻¹")

        if pss_B_conc <= 0:
            raise ValueError(f"[B]_PSS ≤ 0 ({pss_B_conc:.4e}) — check PSS parameters.")

        if input("\n  Confirm PSS state? (y/n): ").strip().lower() != "y":
            print("  Skipping.")
            continue

    # ==============================
    # FIT / CALCULATE
    # ==============================
    print(f"\n--- {'Algebraic calculation' if case == 'A_thermal_PSS' else 'ODE fitting'} ---")

    file_stem = f"{csv_file.stem}_{compound_name}_{case}"

    if case == "A_thermal_PSS":
        # ---- Algebraic PSS calculation ----
        QY_AB_nom = pss_algebraic(_k_th, pss_B_conc, V_L, N_mol_s, pss_A_abs_val)

        # Uncertainty: I₀ perturbation
        if N_std_mol_s > 0:
            QY_hi = pss_algebraic(_k_th, pss_B_conc, V_L, N_mol_s + N_std_mol_s, pss_A_abs_val)
            QY_lo = pss_algebraic(_k_th, pss_B_conc, V_L, N_mol_s - N_std_mol_s, pss_A_abs_val)
            sigma_I0_AB = (abs(QY_hi - QY_AB_nom) + abs(QY_AB_nom - QY_lo)) / 2.0
        else:
            sigma_I0_AB = 0.0
            QY_hi = QY_lo = QY_AB_nom

        # Uncertainty: k_th propagation (σ_QY/QY = σ_k/k)
        if _k_th_std > 0 and _k_th > 0:
            sigma_kth_AB = QY_AB_nom * (_k_th_std / _k_th)
        else:
            sigma_kth_AB = 0.0

        sigma_total_AB = np.sqrt(sigma_I0_AB**2 + sigma_kth_AB**2)

        print(f"\n  Φ_AB (PSS)  = {QY_AB_nom:.5f}")
        print(f"  σ (I₀ pert) = {sigma_I0_AB:.5f}")
        print(f"  σ (k_th)    = {sigma_kth_AB:.5f}")
        print(f"  σ (total)   = {sigma_total_AB:.5f}")

        # Results dict for saving
        result_dict = {
            "File":                csv_file.name,
            "Compound":            compound_name,
            "Temperature_C":       temperature_C,
            "Solvent":             solvent,
            "Case":                case,
            "Irradiation_nm":      irradiation_wavelength_nm,
            "Monitoring_nm":       str(_mon_wls),
            "N_mol_s":             N_mol_s,
            "N_std_mol_s":         N_std_mol_s,
            "k_th_s":              _k_th,
            "k_th_std_s":          _k_th_std,
            "epsilon_A_irr":       _eps_A_irr,
            "epsilon_B_irr":       np.nan,
            "Volume_mL":           volume_mL,
            "Path_cm":             path_length_cm,
            "Fit_t_start_s":       np.nan,
            "Fit_t_end_s":         np.nan,
            "Fit_n_points":        np.nan,
            "Phi_AB":              QY_AB_nom,
            "Phi_AB_sigma_fit":    np.nan,
            "Phi_AB_sigma_I0":     sigma_I0_AB,
            "Phi_AB_sigma_kth":    sigma_kth_AB,
            "Phi_AB_sigma_total":  sigma_total_AB,
            "Phi_BA":              np.nan,
            "Phi_BA_sigma_fit":    np.nan,
            "Phi_BA_sigma_I0":     np.nan,
            "Phi_BA_sigma_total":  np.nan,
            "R2":                  np.nan,
            "Method":              "PSS_algebraic",
            "Date":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Plot: just report no time-series fit plot for algebraic case
        if show_diagnostic:
            print("\n  (No time-series fit plot for algebraic PSS method.)")

    else:
        # ---- ODE fitting (A_only or AB_both) ----

        # Build lmfit Parameters
        def make_params():
            p = Parameters()
            if QY_unconstrained:
                p.add("QY_AB", value=QY_AB_init)
            else:
                p.add("QY_AB", value=QY_AB_init,
                      min=QY_bounds[0], max=QY_bounds[1])
            if case == "AB_both":
                if QY_unconstrained:
                    p.add("QY_BA", value=QY_BA_init)
                else:
                    p.add("QY_BA", value=QY_BA_init,
                          min=QY_bounds[0], max=QY_bounds[1])
            else:
                p.add("QY_BA", value=0.0, vary=False)
            return p

        if QY_unconstrained:
            print("  NOTE: QY_unconstrained = True — bounds removed.  "
                  "Φ < 0 or Φ > 1 is physically possible in the fit.")

        # ---- Per-wavelength ODE fitting ----
        n_mon = len(_mon_wls)

        QY_AB_per_wl       = []
        QY_BA_per_wl       = []
        stderr_AB_per_wl   = []
        stderr_BA_per_wl   = []
        sigma_I0_AB_per_wl = []
        sigma_I0_BA_per_wl = []
        sigma_total_per_wl = []
        r2_per_wl          = []
        result_per_wl      = []

        _use_led_full = (irradiation_source == "LED" and
                         led_integration_mode == "full" and
                         _led_wl_arr is not None and _led_N_arr is not None and
                         _led_eps_A_arr is not None and _led_eps_B_arr is not None)

        for j in range(n_mon):
            # 1-D time axis for this channel
            if _kin_t2d_ode is not None:
                _t_j_fit = _kin_t2d_ode[:, j]
            else:
                _t_j_fit = time_s_ode

            _abs_j   = abs_data_fit[:, j:j+1]   # (n_time, 1)
            _eps_A_j = eps_A_mon[j:j+1]
            _eps_B_j = eps_B_mon[j:j+1]

            if _use_led_full:
                result_j = run_fit_led(make_params(), _t_j_fit, _abs_j,
                                       conc_A_0, conc_B_0,
                                       _led_wl_arr, _led_N_arr,
                                       _led_eps_A_arr, _led_eps_B_arr,
                                       _eps_A_j, _eps_B_j,
                                       V_L, _k_th, path_length_cm)
            else:
                result_j = run_fit(make_params(), _t_j_fit, _abs_j,
                                   conc_A_0, conc_B_0,
                                   _eps_A_irr, _eps_B_irr,
                                   _eps_A_j, _eps_B_j,
                                   N_mol_s, V_L, _k_th, path_length_cm)
            if not result_j.success:
                print(f"  WARNING λ={_mon_wls[j]:.0f} nm: fit may not have converged. "
                      f"{result_j.message}")

            QY_AB_j = result_j.params["QY_AB"].value
            QY_BA_j = result_j.params["QY_BA"].value
            se_AB_j = result_j.params["QY_AB"].stderr or np.nan
            se_BA_j = result_j.params["QY_BA"].stderr or np.nan

            _rr_j   = result_j.residual
            _sst_j  = np.sum((_abs_j.flatten() - _abs_j.mean()) ** 2)
            r2_j    = 1.0 - np.sum(_rr_j**2) / _sst_j if _sst_j > 0 else np.nan

            # I₀ perturbation for this channel
            _si0_AB_j = 0.0
            _si0_BA_j = 0.0
            if N_std_mol_s > 0:
                if _use_led_full:
                    _scale = N_std_mol_s / N_mol_s
                    _N_hi  = _led_N_arr * (1.0 + _scale)
                    _N_lo  = _led_N_arr * (1.0 - _scale)
                    _rhi_j = run_fit_led(make_params(), _t_j_fit, _abs_j,
                                         conc_A_0, conc_B_0,
                                         _led_wl_arr, _N_hi,
                                         _led_eps_A_arr, _led_eps_B_arr,
                                         _eps_A_j, _eps_B_j,
                                         V_L, _k_th, path_length_cm)
                    _rlo_j = run_fit_led(make_params(), _t_j_fit, _abs_j,
                                         conc_A_0, conc_B_0,
                                         _led_wl_arr, _N_lo,
                                         _led_eps_A_arr, _led_eps_B_arr,
                                         _eps_A_j, _eps_B_j,
                                         V_L, _k_th, path_length_cm)
                else:
                    _rhi_j = run_fit(make_params(), _t_j_fit, _abs_j,
                                     conc_A_0, conc_B_0,
                                     _eps_A_irr, _eps_B_irr, _eps_A_j, _eps_B_j,
                                     N_mol_s + N_std_mol_s, V_L, _k_th, path_length_cm)
                    _rlo_j = run_fit(make_params(), _t_j_fit, _abs_j,
                                     conc_A_0, conc_B_0,
                                     _eps_A_irr, _eps_B_irr, _eps_A_j, _eps_B_j,
                                     N_mol_s - N_std_mol_s, V_L, _k_th, path_length_cm)
                _si0_AB_j = (abs(_rhi_j.params["QY_AB"].value - QY_AB_j) +
                              abs(QY_AB_j - _rlo_j.params["QY_AB"].value)) / 2.0
                if case == "AB_both":
                    _si0_BA_j = (abs(_rhi_j.params["QY_BA"].value - QY_BA_j) +
                                  abs(QY_BA_j - _rlo_j.params["QY_BA"].value)) / 2.0

            _stot_j = np.sqrt((se_AB_j if not np.isnan(se_AB_j) else 0.0)**2 + _si0_AB_j**2)

            QY_AB_per_wl.append(QY_AB_j)
            QY_BA_per_wl.append(QY_BA_j)
            stderr_AB_per_wl.append(se_AB_j)
            stderr_BA_per_wl.append(se_BA_j)
            sigma_I0_AB_per_wl.append(_si0_AB_j)
            sigma_I0_BA_per_wl.append(_si0_BA_j)
            sigma_total_per_wl.append(_stot_j)
            r2_per_wl.append(r2_j)
            result_per_wl.append(result_j)

            print(f"  λ = {_mon_wls[j]:.0f} nm : Φ_AB = {QY_AB_j:.5f}  "
                  f"± {se_AB_j:.5f} (fit)  σ_I₀ = {_si0_AB_j:.5f}  "
                  f"σ_total = {_stot_j:.5f}  R² = {r2_j:.5f}")
            if case == "AB_both":
                print(f"           Φ_BA = {QY_BA_j:.5f}  ± {se_BA_j:.5f}")

        # ---- Aggregate across wavelengths ----
        QY_AB_nom    = float(np.nanmean(QY_AB_per_wl))
        QY_BA_nom    = float(np.nanmean(QY_BA_per_wl)) if case == "AB_both" else 0.0
        stderr_AB    = float(np.nanmean(stderr_AB_per_wl))
        stderr_BA    = float(np.nanmean(stderr_BA_per_wl)) if case == "AB_both" else np.nan
        sigma_I0_AB  = float(np.nanmean(sigma_I0_AB_per_wl))
        sigma_I0_BA  = float(np.nanmean(sigma_I0_BA_per_wl)) if case == "AB_both" else 0.0
        sigma_total_AB = float(np.nanmean(sigma_total_per_wl))
        sigma_total_BA = (
            float(np.nanmean([np.sqrt((se if not np.isnan(se) else 0.0)**2 + si**2)
                               for se, si in zip(stderr_BA_per_wl, sigma_I0_BA_per_wl)]))
            if case == "AB_both" else np.nan
        )
        r2 = float(np.nanmean(r2_per_wl))

        _valid_QY_AB  = [q for q in QY_AB_per_wl if not np.isnan(q)]
        _QY_AB_spread = float(np.std(_valid_QY_AB, ddof=1)) if len(_valid_QY_AB) > 1 else 0.0

        print(f"\n  Mean across {n_mon} wavelength(s):")
        print(f"  Φ_AB (mean)    = {QY_AB_nom:.5f}  ±  {_QY_AB_spread:.5f}  (spread across λ)")
        print(f"  Φ_AB σ_total   = {sigma_total_AB:.5f}  (mean of per-wavelength σ_total)")
        if case == "AB_both":
            _valid_QY_BA  = [q for q in QY_BA_per_wl if not np.isnan(q)]
            _QY_BA_spread = float(np.std(_valid_QY_BA, ddof=1)) if len(_valid_QY_BA) > 1 else 0.0
            print(f"  Φ_BA (mean)    = {QY_BA_nom:.5f}  ±  {_QY_BA_spread:.5f}  (spread across λ)")
        print(f"  R² (mean)      = {r2:.5f}")

        print(f"\n  --- Error components ---")
        print(f"  σ_fit  : lmfit Jacobian stderr — reflects how well the ODE curve fits the")
        print(f"           data scatter around the optimum.  Larger when data are noisy or")
        print(f"           the model is insensitive to Φ at the measured wavelengths.")
        print(f"  σ_I₀   : photon-flux sensitivity — the shift in Φ when I₀ is perturbed by")
        print(f"           ±σ_N ({N_std_mol_s:.2e} mol s⁻¹).  Zero when photon_flux_std_mol_s = 0.")
        print(f"           Larger when Φ is small (more sensitive to flux calibration).")
        print(f"  σ_total: √(σ_fit² + σ_I₀²) — independent sources combined in quadrature.")
        print(f"           This is the value shown as the orange uncertainty band in the plot.")

        result_dict = {
            "File":                  csv_file.name,
            "Compound":              compound_name,
            "Temperature_C":         temperature_C,
            "Solvent":               solvent,
            "Case":                  case,
            "Irradiation_nm":        irradiation_wavelength_nm,
            "Monitoring_nm":         str(_mon_wls),
            "N_mol_s":               N_mol_s,
            "N_std_mol_s":           N_std_mol_s,
            "k_th_s":                _k_th,
            "k_th_std_s":            _k_th_std,
            "epsilon_A_irr":         _eps_A_irr,
            "epsilon_B_irr":         _eps_B_irr,
            "Volume_mL":             volume_mL,
            "Path_cm":               path_length_cm,
            "Fit_t_start_s":         time_s_fit[0],
            "Fit_t_end_s":           time_s_fit[-1],
            "Fit_n_points":          len(time_s_fit),
            "Phi_AB":                QY_AB_nom,          # mean across wavelengths
            "Phi_AB_sigma_fit":      stderr_AB,
            "Phi_AB_sigma_I0":       sigma_I0_AB,
            "Phi_AB_sigma_kth":      np.nan,
            "Phi_AB_sigma_total":    sigma_total_AB,
            "Phi_BA":                QY_BA_nom if case == "AB_both" else np.nan,
            "Phi_BA_sigma_fit":      stderr_BA if case == "AB_both" else np.nan,
            "Phi_BA_sigma_I0":       sigma_I0_BA if case == "AB_both" else np.nan,
            "Phi_BA_sigma_total":    sigma_total_BA if case == "AB_both" else np.nan,
            "R2":                    r2,
            "Method":                ("ODE_lmfit_per_wl_LED_full" if _use_led_full else "ODE_lmfit_per_wl")
                                     + ("_unconstrained" if QY_unconstrained else ""),
            "Date":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Per-wavelength columns
        for j, wl_nm in enumerate(_mon_wls):
            result_dict[f"Phi_AB_{wl_nm:.0f}nm"]         = QY_AB_per_wl[j]
            result_dict[f"Phi_AB_sigma_total_{wl_nm:.0f}nm"] = sigma_total_per_wl[j]
            result_dict[f"R2_{wl_nm:.0f}nm"]             = r2_per_wl[j]
            if case == "AB_both":
                result_dict[f"Phi_BA_{wl_nm:.0f}nm"] = QY_BA_per_wl[j]

        # ==============================
        # INITIAL SLOPES CONTROL
        # ==============================
        # QY estimate from linear fit to the first n_initial_slopes_points of the fitting window.
        # Valid when [B] ≈ 0 (early times / small conversion).
        QY_slopes, _slopes = initial_slopes_QY(
            time_s_ode, abs_data_fit, n_initial_slopes_points,
            _eps_A_irr, eps_A_mon, eps_B_mon,
            N_mol_s, V_L, path_length_cm, conc_A_0)
        # time_s_ode is always 1-D (first channel or shared scanning axis) — correct for slopes

        print(f"\n  Initial slopes (first {min(n_initial_slopes_points, len(time_s_ode))} pts):")
        for j, wl_nm in enumerate(_mon_wls):
            if not np.isnan(QY_slopes[j]):
                print(f"    λ = {wl_nm:.0f} nm : slope = {_slopes[j]:.4e} s⁻¹  "
                      f"→  Φ_AB (initial slopes) = {QY_slopes[j]:.5f}")
            else:
                print(f"    λ = {wl_nm:.0f} nm : cannot compute (ε_A − ε_B ≈ 0)")

        _QY_slopes_valid = [q for q in QY_slopes if not np.isnan(q)]
        if _QY_slopes_valid:
            print(f"    Mean initial slopes Φ_AB = {np.mean(_QY_slopes_valid):.5f}  "
                  f"±  {np.std(_QY_slopes_valid, ddof=1):.5f}  (spread across wavelengths)"
                  if len(_QY_slopes_valid) > 1 else
                  f"    Initial slopes Φ_AB = {_QY_slopes_valid[0]:.5f}")

        # ==============================
        # FULL-RANGE ODE FIT (scanning only, when a selection was applied)
        # ==============================
        _do_full_range = (data_type == "scanning") and (_n_excluded > 0)
        result_full = None
        QY_AB_full  = np.nan
        r2_full     = np.nan

        abs_full_disp  = None   # full-range ODE simulation; used in control plot + graph data
        _c_A0_full     = abs_data[0, 0] / (eps_A_mon[0] * path_length_cm)

        if _do_full_range:
            print(f"\n  Running full-range ODE fit ({len(time_s)} points, "
                  f"t = {time_s[0]:.1f} – {time_s[-1]:.1f} s) ...")
            time_s_full_ode = time_s - time_s[0]
            if _use_led_full:
                result_full = run_fit_led(make_params(), time_s_full_ode, abs_data,
                                          _c_A0_full, conc_B_0,
                                          _led_wl_arr, _led_N_arr,
                                          _led_eps_A_arr, _led_eps_B_arr,
                                          eps_A_mon, eps_B_mon,
                                          V_L, _k_th, path_length_cm)
            else:
                result_full = run_fit(make_params(), time_s_full_ode, abs_data,
                                      _c_A0_full, conc_B_0,
                                      _eps_A_irr, _eps_B_irr,
                                      eps_A_mon, eps_B_mon,
                                      N_mol_s, V_L, _k_th, path_length_cm)
            QY_AB_full = result_full.params["QY_AB"].value
            _res_full  = result_full.residual
            _ss_res_f  = np.sum(_res_full ** 2)
            _ss_tot_f  = np.sum((abs_data.flatten() - abs_data.flatten().mean()) ** 2)
            r2_full    = 1.0 - _ss_res_f / _ss_tot_f if _ss_tot_f > 0 else np.nan
            print(f"    Full-range Φ_AB = {QY_AB_full:.5f}  "
                  f"±  {result_full.params['QY_AB'].stderr or np.nan:.5f}  "
                  f"  R² = {r2_full:.4f}")
            # Simulate the full-range ODE curve for plotting / saving
            _p_fr = make_params()
            _p_fr["QY_AB"].set(value=QY_AB_full, vary=False)
            abs_full_disp = simulate_absorbance_led(
                _p_fr, time_s_full_ode,
                _c_A0_full, conc_B_0,
                _led_wl_arr, _led_N_arr,
                _led_eps_A_arr, _led_eps_B_arr,
                eps_A_mon, eps_B_mon,
                V_L, _k_th, path_length_cm) if _use_led_full else \
                simulate_absorbance(
                _p_fr, time_s_full_ode,
                _c_A0_full, conc_B_0,
                _eps_A_irr, _eps_B_irr, eps_A_mon, eps_B_mon,
                N_mol_s, V_L, _k_th, path_length_cm)

        # ==============================
        # MAIN PLOT (selected window)
        # ==============================
        # Simulate ODE only for times >= irradiation start (t=0 in ODE frame).
        # Including pre-irradiation times would integrate backward with flux on → wrong.
        _disp_mask   = time_s >= time_s_fit[0]
        _time_s_disp = time_s[_disp_mask]
        _fit_in_disp = fit_mask[_disp_mask]

        # Per-wavelength display time axes (kinetic: each channel's own times)
        _t_disp_per_wl     = []   # absolute time for x-axis
        _t_disp_ode_per_wl = []   # ODE time (reset to 0 at fit start)
        for j in range(n_mon):
            if _kin_t2d is not None:
                _td_j     = _kin_t2d[_disp_mask, j]
                _td_ode_j = _td_j - _kin_t2d_fit[0, j]
            else:
                _td_j     = _time_s_disp
                _td_ode_j = _time_s_disp - time_s_fit[0]
            _t_disp_per_wl.append(_td_j)
            _t_disp_ode_per_wl.append(_td_ode_j)

        # Per-wavelength fit curves and ±σ bands
        abs_fit_per_wl    = []
        abs_fit_hi_per_wl = []
        abs_fit_lo_per_wl = []
        abs_fit_mean_per_wl = []   # curve using mean QY (reference)

        def _sim_curve(p, t_ode, j):
            """Simulate single-channel absorbance curve; branches on LED full mode."""
            if _use_led_full:
                return simulate_absorbance_led(
                    p, t_ode, conc_A_0, conc_B_0,
                    _led_wl_arr, _led_N_arr,
                    _led_eps_A_arr, _led_eps_B_arr,
                    eps_A_mon[j:j+1], eps_B_mon[j:j+1],
                    V_L, _k_th, path_length_cm)[:, 0]
            else:
                return simulate_absorbance(
                    p, t_ode, conc_A_0, conc_B_0,
                    _eps_A_irr, _eps_B_irr,
                    eps_A_mon[j:j+1], eps_B_mon[j:j+1],
                    N_mol_s, V_L, _k_th, path_length_cm)[:, 0]

        for j in range(n_mon):
            _p_j = make_params()
            _p_j["QY_AB"].set(value=QY_AB_per_wl[j], vary=False)
            if case == "AB_both":
                _p_j["QY_BA"].set(value=QY_BA_per_wl[j], vary=False)
            _curve_j = _sim_curve(_p_j, _t_disp_ode_per_wl[j], j)

            _p_hi = make_params()
            _p_lo = make_params()
            if QY_unconstrained:
                _p_hi["QY_AB"].set(value=QY_AB_per_wl[j] + sigma_total_per_wl[j], vary=False)
                _p_lo["QY_AB"].set(value=QY_AB_per_wl[j] - sigma_total_per_wl[j], vary=False)
            else:
                _p_hi["QY_AB"].set(
                    value=min(QY_AB_per_wl[j] + sigma_total_per_wl[j], QY_bounds[1]), vary=False)
                _p_lo["QY_AB"].set(
                    value=max(QY_AB_per_wl[j] - sigma_total_per_wl[j], QY_bounds[0]), vary=False)
            if case == "AB_both":
                _st_BA_j = np.sqrt(
                    (stderr_BA_per_wl[j] if not np.isnan(stderr_BA_per_wl[j]) else 0.0)**2
                    + sigma_I0_BA_per_wl[j]**2)
                if QY_unconstrained:
                    _p_hi["QY_BA"].set(value=QY_BA_per_wl[j] + _st_BA_j, vary=False)
                    _p_lo["QY_BA"].set(value=QY_BA_per_wl[j] - _st_BA_j, vary=False)
                else:
                    _p_hi["QY_BA"].set(value=min(QY_BA_per_wl[j] + _st_BA_j, QY_bounds[1]), vary=False)
                    _p_lo["QY_BA"].set(value=max(QY_BA_per_wl[j] - _st_BA_j, QY_bounds[0]), vary=False)

            _curve_hi   = _sim_curve(_p_hi,  _t_disp_ode_per_wl[j], j)
            _curve_lo   = _sim_curve(_p_lo,  _t_disp_ode_per_wl[j], j)

            _p_mean = make_params()
            _p_mean["QY_AB"].set(value=QY_AB_nom, vary=False)
            if case == "AB_both":
                _p_mean["QY_BA"].set(value=QY_BA_nom, vary=False)
            _curve_mean = _sim_curve(_p_mean, _t_disp_ode_per_wl[j], j)

            abs_fit_per_wl.append(_curve_j)
            abs_fit_hi_per_wl.append(_curve_hi)
            abs_fit_lo_per_wl.append(_curve_lo)
            abs_fit_mean_per_wl.append(_curve_mean)

        # Reference QY curve (fixed, user-supplied Φ_AB_reference)
        abs_fit_ref_per_wl = []
        if QY_AB_reference is not None:
            _ref_BA = QY_BA_reference if (case == "AB_both" and QY_BA_reference is not None) \
                      else QY_BA_nom
            for j in range(n_mon):
                _p_ref = Parameters()
                _p_ref.add("QY_AB", value=float(QY_AB_reference), vary=False)
                _p_ref.add("QY_BA", value=float(_ref_BA), vary=False)
                abs_fit_ref_per_wl.append(_sim_curve(_p_ref, _t_disp_ode_per_wl[j], j))

        # Residuals per wavelength (fitting window only)
        residuals_2d = np.column_stack([
            abs_fit_per_wl[j][_fit_in_disp] - abs_data_fit[:, j]
            for j in range(n_mon)
        ])

        # Full-length array (NaN before irradiation start) — used in graph data CSV
        abs_fit_disp_full = np.full((len(time_s), n_mon), np.nan)
        for j in range(n_mon):
            abs_fit_disp_full[_disp_mask, j] = abs_fit_per_wl[j]

        # Concentration display: use mean QY + first channel time axis
        _time_s_disp_ode = _t_disp_ode_per_wl[0]
        if _use_led_full:
            conc_display = odeint(
                rate_equations_led, [conc_A_0, conc_B_0], _time_s_disp_ode,
                args=(QY_AB_nom, QY_BA_nom,
                      _led_wl_arr, _led_N_arr,
                      _led_eps_A_arr, _led_eps_B_arr,
                      V_L, _k_th, path_length_cm),
                mxstep=5000,
            )
        else:
            conc_display = odeint(
                rate_equations, [conc_A_0, conc_B_0], _time_s_disp_ode,
                args=(QY_AB_nom, QY_BA_nom,
                      _eps_A_irr, _eps_B_irr,
                      N_mol_s, V_L, _k_th, path_length_cm),
                mxstep=5000,
            )

        n_mon = len(_mon_wls)
        n_rows_main = n_mon + 2
        fig_main, axes_main = plt.subplots(
            n_rows_main, 1,
            figsize=(8, 2.5 * n_rows_main),
            gridspec_kw={"height_ratios": [3] * n_mon + [1.5, 2.5]},
            constrained_layout=True)
        if n_rows_main == 1:
            axes_main = [axes_main]

        for j, wl_nm in enumerate(_mon_wls):
            ax = axes_main[j]
            # Full data — use per-channel time axis for kinetic data
            _t_data_j = _kin_t2d[:, j] if _kin_t2d is not None else time_s
            ax.plot(_t_data_j, abs_data[:, j], "-", color="#3a7ebf", linewidth=1.8,
                    label=f"Data  λ = {wl_nm:.0f} nm")
            # Per-wavelength ODE fit curve
            ax.plot(_t_disp_per_wl[j], abs_fit_per_wl[j], "--", color="red", linewidth=1.6,
                    label=f"Fit  Φ_AB={QY_AB_per_wl[j]:.4f}  R²={r2_per_wl[j]:.4f}")
            # Mean QY reference curve (only when more than one monitoring wavelength)
            if n_mon > 1:
                ax.plot(_t_disp_per_wl[j], abs_fit_mean_per_wl[j],
                        ":", color="purple", linewidth=1.4,
                        label=f"Mean Φ_AB={QY_AB_nom:.4f}")
            # ±σ_total uncertainty band (per-wavelength)
            ax.fill_between(_t_disp_per_wl[j],
                            abs_fit_lo_per_wl[j], abs_fit_hi_per_wl[j],
                            color="#e67e22", alpha=0.30, label="±σ_total")
            # Reference QY curve
            if abs_fit_ref_per_wl:
                ax.plot(_t_disp_per_wl[j], abs_fit_ref_per_wl[j],
                        "-.", color="#2ecc71", linewidth=1.6,
                        label=f"Reference Φ_AB={QY_AB_reference:.4f}")
            # Shade the fitting window
            ax.axvspan(time_s_fit[0], time_s_fit[-1],
                       alpha=0.12, color="gray", label="Fitting window" if j == 0 else None)
            ax.set_ylabel("Absorbance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

        # Annotation: per-wavelength values + mean
        _ann_lines = []
        if QY_unconstrained:
            _ann_lines.append("⚠ Unconstrained fit (Φ may be < 0 or > 1)")
        for j, wl_nm in enumerate(_mon_wls):
            _ann_lines.append(
                f"λ={wl_nm:.0f} nm: Φ_AB={QY_AB_per_wl[j]:.4f}  "
                f"σ_fit={stderr_AB_per_wl[j]:.4f}  "
                f"σ_total={sigma_total_per_wl[j]:.4f}  R²={r2_per_wl[j]:.4f}"
            )
            if case == "AB_both":
                _ann_lines.append(
                    f"         Φ_BA={QY_BA_per_wl[j]:.4f}  "
                    f"σ_fit={stderr_BA_per_wl[j]:.4f}"
                )
        _ann_lines.append(f"Mean Φ_AB = {QY_AB_nom:.4f}  σ_total = {sigma_total_AB:.4f}")
        if case == "AB_both":
            _ann_lines.append(f"Mean Φ_BA = {QY_BA_nom:.4f}  σ_total = {sigma_total_BA:.4f}")
        _ann_lines.append(f"R² (mean) = {r2:.4f}")
        if QY_AB_reference is not None:
            _ann_lines.append(f"Reference Φ_AB = {QY_AB_reference:.4f}")
        _ann = "\n".join(_ann_lines)
        axes_main[0].text(0.97, 0.97, _ann, transform=axes_main[0].transAxes,
                          fontsize=8, verticalalignment="top", horizontalalignment="right",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        _main_title = f"{compound_name}  |  {case}  |  {csv_file.stem}"
        if QY_unconstrained:
            _main_title += "  [unconstrained]"
        axes_main[0].set_title(_main_title)

        # Residual panel (fitting window only)
        # For kinetic: use per-channel fit-window time for each residual trace
        ax_res = axes_main[n_mon]
        for j, wl_nm in enumerate(_mon_wls):
            _t_fit_j = (_kin_t2d_fit[:, j] if _kin_t2d_fit is not None else time_s_fit)
            ax_res.plot(_t_fit_j, residuals_2d[:, j], linewidth=1.2,
                        label=f"{wl_nm:.0f} nm")
        ax_res.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_res.set_ylabel("Residual")
        ax_res.legend(fontsize=8)
        ax_res.grid(True, alpha=0.4)
        ax_res.set_xlim(axes_main[0].get_xlim())

        # Concentrations panel — ODE starts at irradiation time (mean QY, first channel time)
        ax_conc = axes_main[n_mon + 1]

        # Observed [A] from Beer-Lambert: [A]_obs = A_obs / (ε_A × l)
        # [B]_obs from mass balance: [B]_obs = [A]₀ − [A]_obs  (assumes [A]+[B] = const)
        # Both are approximations that ignore [B]'s contribution to absorbance.
        _cmap_conc = plt.colormaps["tab10"].resampled(n_mon)
        for j, wl_nm in enumerate(_mon_wls):
            _t_obs_j = _kin_t2d[:, j] if _kin_t2d is not None else time_s
            _A_obs_j = abs_data[:, j] / (eps_A_mon[j] * path_length_cm) * 1000  # mmol/L
            _B_obs_j = conc_A_0 * 1000 - _A_obs_j                               # mmol/L
            ax_conc.plot(_t_obs_j, _A_obs_j, ".", markersize=3, alpha=0.5,
                         color=_cmap_conc(j),
                         label=f"[A] obs  {wl_nm:.0f} nm")
            ax_conc.plot(_t_obs_j, _B_obs_j, ".", markersize=3, alpha=0.5,
                         color="#e87d37",
                         label=f"[B] obs  {wl_nm:.0f} nm" if j == 0 else "_nolegend_")

        # ODE model curves
        ax_conc.plot(_t_disp_per_wl[0], conc_display[:, 0] * 1000, "--",
                     color="green", linewidth=1.8, label="[A] fit")
        ax_conc.plot(_t_disp_per_wl[0], conc_display[:, 1] * 1000, "--",
                     color="red", linewidth=1.8, label="[B] fit")

        ax_conc.axvspan(time_s_fit[0], time_s_fit[-1], alpha=0.12, color="gray")
        ax_conc.set_xlabel("Time (s)")
        ax_conc.set_ylabel("Concentration (mmol L⁻¹)")
        ax_conc.set_xlim(axes_main[0].get_xlim())   # align x-axis with absorbance panels
        ax_conc.legend(fontsize=8)
        ax_conc.grid(True, alpha=0.4)

        plt.show()

        if input("  Save main plot? (y/n): ").strip().lower() == "y":
            img_path = QY_PLOTS_DIR / f"{file_stem}.png"
            fig_main.savefig(img_path, dpi=150, bbox_inches="tight")
            print(f"  Image saved → {img_path.relative_to(BASE_DIR)}")
        plt.close(fig_main)

        # ==============================
        # CONTROL PLOT
        # ==============================
        if show_control_plot:
            n_pts_slope = min(n_initial_slopes_points, len(time_s_ode))
            t_slope_end = time_s_fit[n_pts_slope - 1]   # absolute time of last slope point

            fig_ctrl, axes_ctrl = plt.subplots(
                n_mon + 1, 1,
                figsize=(9, 2.8 * (n_mon + 1)),
                gridspec_kw={"height_ratios": [3] * n_mon + [2]},
                constrained_layout=True)
            if n_mon + 1 == 1:
                axes_ctrl = [axes_ctrl]

            for j, wl_nm in enumerate(_mon_wls):
                ax = axes_ctrl[j]

                # All data — per-channel time axis
                _t_data_j = _kin_t2d[:, j] if _kin_t2d is not None else time_s
                ax.plot(_t_data_j, abs_data[:, j], "-", color="#3a7ebf", linewidth=1.8,
                        label=f"Data  λ = {wl_nm:.0f} nm")

                # Fitting window shading
                ax.axvspan(time_s_fit[0], time_s_fit[-1],
                           alpha=0.12, color="gray", zorder=0,
                           label="Fitting window" if j == 0 else None)

                # Per-wavelength ODE fit curve + ±σ band
                ax.plot(_t_disp_per_wl[j], abs_fit_per_wl[j], "--", color="red", linewidth=1.6,
                        label=f"Fit  Φ_AB={QY_AB_per_wl[j]:.4f}  R²={r2_per_wl[j]:.4f}")
                ax.fill_between(_t_disp_per_wl[j],
                                abs_fit_lo_per_wl[j], abs_fit_hi_per_wl[j],
                                color="#e67e22", alpha=0.30, label="±σ_total")

                # Mean QY reference (when multiple wavelengths)
                if n_mon > 1:
                    ax.plot(_t_disp_per_wl[j], abs_fit_mean_per_wl[j],
                            ":", color="purple", linewidth=1.4,
                            label=f"Mean Φ_AB={QY_AB_nom:.4f}")

                # Reference QY curve
                if abs_fit_ref_per_wl:
                    ax.plot(_t_disp_per_wl[j], abs_fit_ref_per_wl[j],
                            "-.", color="#2ecc71", linewidth=1.6,
                            label=f"Reference Φ_AB={QY_AB_reference:.4f}")

                # Full-range ODE fit (if computed)
                if abs_full_disp is not None:
                    ax.plot(_t_data_j, abs_full_disp[:, j],
                            "-.", color="darkorange", linewidth=1.4,
                            label=f"Full-range Φ_AB={QY_AB_full:.4f}")

                # Initial slopes line
                if not np.isnan(QY_slopes[j]) and n_pts_slope >= 2:
                    _t_sl  = np.array([time_s_fit[0], time_s_fit[0] + time_s_ode[n_pts_slope - 1]])
                    _a_sl0 = abs_data_fit[0, j]
                    _a_sl1 = _a_sl0 + _slopes[j] * time_s_ode[n_pts_slope - 1]
                    ax.plot(_t_sl, [_a_sl0, _a_sl1], ":", color="forestgreen", linewidth=2.0,
                            label=f"Initial slope  Φ_AB={QY_slopes[j]:.4f}")
                    ax.axvline(t_slope_end, color="forestgreen", linewidth=0.8,
                               linestyle=":", alpha=0.6)

                ax.set_ylabel("Absorbance")
                ax.legend(fontsize=8, loc="best")
                ax.grid(True, alpha=0.4)

            axes_ctrl[0].set_title(
                f"Control plot — {compound_name}  |  {case}  |  {csv_file.stem}"
            )

            # Summary panel: Φ_AB comparison bar chart
            ax_sum = axes_ctrl[n_mon]
            _methods    = []
            _QY_vals    = []
            _QY_errs    = []
            _bar_colors = []

            # Per-wavelength ODE bars
            for j, wl_nm in enumerate(_mon_wls):
                _methods.append(f"ODE\n{wl_nm:.0f} nm")
                _QY_vals.append(QY_AB_per_wl[j])
                _QY_errs.append(sigma_total_per_wl[j])
                _bar_colors.append("#c0392b")

            # Mean ODE bar
            if n_mon > 1:
                _methods.append("ODE\n(mean)")
                _QY_vals.append(QY_AB_nom)
                _QY_errs.append(sigma_total_AB)
                _bar_colors.append("#8e1a0e")

            if not np.isnan(QY_AB_full):
                _methods.append("ODE\n(full range)")
                _QY_vals.append(QY_AB_full)
                _QY_errs.append(result_full.params["QY_AB"].stderr or 0.0)
                _bar_colors.append("#e67e22")

            for j, wl_nm in enumerate(_mon_wls):
                if not np.isnan(QY_slopes[j]):
                    _methods.append(f"Slopes\n{wl_nm:.0f} nm")
                    _QY_vals.append(QY_slopes[j])
                    _QY_errs.append(0.0)
                    _bar_colors.append("#27ae60")

            # Reference QY — horizontal line + bar
            if QY_AB_reference is not None:
                _methods.append(f"Reference\n(Φ={QY_AB_reference:.4f})")
                _QY_vals.append(QY_AB_reference)
                _QY_errs.append(0.0)
                _bar_colors.append("#2ecc71")

            x_pos = np.arange(len(_methods))
            ax_sum.bar(x_pos, _QY_vals, yerr=_QY_errs,
                       color=_bar_colors, alpha=0.75, capsize=5,
                       error_kw={"elinewidth": 1.5, "ecolor": "black"})
            # Reference horizontal line across the whole chart
            if QY_AB_reference is not None:
                ax_sum.axhline(QY_AB_reference, color="#2ecc71", linewidth=1.2,
                               linestyle="--", alpha=0.8,
                               label=f"Φ_AB ref = {QY_AB_reference:.4f}")
                ax_sum.legend(fontsize=8)
            ax_sum.set_xticks(x_pos)
            ax_sum.set_xticklabels(_methods, fontsize=9)
            ax_sum.set_ylabel("Φ_AB estimate")
            _sum_title = "Φ_AB comparison across methods"
            if QY_unconstrained:
                _sum_title += "  [unconstrained fit]"
            ax_sum.set_title(_sum_title)
            ax_sum.grid(True, axis="y", alpha=0.4)
            # When unconstrained the y-axis may go below 0 — don't force bottom=0
            if not QY_unconstrained:
                ax_sum.set_ylim(bottom=0)

            axes_ctrl[-1].set_xlabel("Method")
            plt.show()

            if input("  Save control plot? (y/n): ").strip().lower() == "y":
                ctrl_path = QY_PLOTS_DIR / f"{file_stem}_control.png"
                fig_ctrl.savefig(ctrl_path, dpi=150, bbox_inches="tight")
                print(f"  Control plot saved → {ctrl_path.relative_to(BASE_DIR)}")
            plt.close(fig_ctrl)

        # --- Save graph data ---
        if input("  Save graph data as CSV? (y/n): ").strip().lower() == "y":
            rows = []
            for i in range(len(time_s)):
                row = {"Time_s": time_s[i],
                       "In_fit_window": bool(fit_mask[i])}
                for j, wl_nm in enumerate(_mon_wls):
                    # Store per-channel time if kinetic
                    if _kin_t2d is not None:
                        row[f"Time_{wl_nm:.0f}nm_s"] = _kin_t2d[i, j]
                    row[f"Abs_data_{wl_nm:.0f}nm"]  = abs_data[i, j]
                    row[f"Abs_fit_{wl_nm:.0f}nm"]   = abs_fit_disp_full[i, j]  # NaN before irr start
                    if abs_full_disp is not None:
                        row[f"Abs_full_{wl_nm:.0f}nm"] = abs_full_disp[i, j]
                # Concentrations — aligned to first channel time (mean QY)
                _t_ref_i = _kin_t2d[i, 0] if _kin_t2d is not None else time_s[i]
                _t_disp0 = _t_disp_per_wl[0]
                _disp_idx = int(np.searchsorted(_t_disp0, _t_ref_i))
                if _t_ref_i >= time_s_fit[0] and _disp_idx < len(conc_display):
                    row["Conc_A_molL"] = conc_display[_disp_idx, 0]
                    row["Conc_B_molL"] = conc_display[_disp_idx, 1]
                else:
                    row["Conc_A_molL"] = np.nan
                    row["Conc_B_molL"] = np.nan
                rows.append(row)
            # Append residuals (fitting window only)
            for i, orig_i in enumerate(np.where(fit_mask)[0]):
                for j, wl_nm in enumerate(_mon_wls):
                    rows[orig_i][f"Residual_{wl_nm:.0f}nm"] = residuals_2d[i, j]
            df_graph = pd.DataFrame(rows)
            data_path = QY_GRAPH_DATA_DIR / f"{file_stem}_data.csv"
            df_graph.to_csv(data_path, index=False)
            print(f"  Graph data saved → {data_path.relative_to(BASE_DIR)}")

    # ==============================
    # SAVE RESULT TO MASTER CSV
    # ==============================
    if input("  Save result to master CSV? (y/n): ").strip().lower() == "y":
        master_path = QY_RESULTS_DIR / "quantum_yield_master.csv"
        df_new = pd.DataFrame([result_dict])
        if master_path.exists():
            df_existing = pd.read_csv(master_path)
            pd.concat([df_existing, df_new], ignore_index=True).to_csv(master_path, index=False)
        else:
            df_new.to_csv(master_path, index=False)
        print(f"  Result saved → {master_path.relative_to(BASE_DIR)}")
