"""
Temporary conversion script — excel_control only.
Reads AZA-SO2Me_acidic.xlsx, extracts both kinetic traces, and writes
them as a two-channel kinetic CSV in the format expected by load_kinetic_csv().

Output format (load_kinetic_csv):
  Row 0:  channel labels  (one label every 2 columns)
  Row 1:  "Time (sec)"  "Abs"  repeated per channel
  Row 2+: time / absorbance pairs per channel; shorter channel is NaN-padded

Channels:
  Col 0-1 : 579 nm trace  (irradiation wavelength)
  Col 6-7 : 673 nm trace  (probe wavelength)
Both time axes are offset-corrected so first point = 0 s.
The two traces have independent time grids and may differ in length.

Delete or exclude this script after the control check is done.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "quantum_yield" / "excel_control" / "AZA-SO2Me_acidic.xlsx"
OUT_PATH   = BASE_DIR / "data" / "quantum_yield" / "raw" / "excel_control_AZA-SO2Me_acidic.csv"

# ── Load Excel ───────────────────────────────────────────────────────────────
df   = pd.read_excel(EXCEL_PATH, sheet_name="Tabelle1", header=None)
data = df.iloc[7:].copy()
data.columns = range(data.shape[1])

def extract_trace(col_t, col_a, label):
    t = pd.to_numeric(data[col_t], errors="coerce").values
    a = pd.to_numeric(data[col_a], errors="coerce").values
    valid = ~(np.isnan(t) | np.isnan(a))
    t, a = t[valid], a[valid]
    t = t - t[0]
    print(f"  {label}: {len(t)} points, t = {t[0]:.2f} to {t[-1]:.2f} s, "
          f"Abs {a[0]:.4f} to {a[-1]:.4f}")
    return t, a

print("Extracting traces:")
t_579, a_579 = extract_trace(0, 1, "579 nm")
t_673, a_673 = extract_trace(6, 7, "673 nm")

# ── Build two-channel CSV (NaN-pad the shorter channel) ──────────────────────
n = max(len(t_579), len(t_673))

def pad(arr, length):
    if len(arr) == length:
        return arr
    return np.concatenate([arr, np.full(length - len(arr), np.nan)])

t_579_p = pad(t_579, n)
a_579_p = pad(a_579, n)
t_673_p = pad(t_673, n)
a_673_p = pad(a_673, n)

rows = []
rows.append(["excel_control_579nm", "",  "excel_control_673nm", ""])
rows.append(["Time (sec)",          "Abs", "Time (sec)",         "Abs"])
for i in range(n):
    rows.append([t_579_p[i], a_579_p[i], t_673_p[i], a_673_p[i]])

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False, header=False)
print(f"\nWritten: {OUT_PATH.relative_to(BASE_DIR)}")
