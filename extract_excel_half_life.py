"""
Standalone script: extract kinetic / scanning-kinetic traces from an Excel file
and write them as CSV files compatible with the thermal/half-life analysis workflow.

Excel layout expected (per sheet)
----------------------------------
Row 0  : temperature value | type ("kinetic" / "scanning")  — repeated per trace pair
         Temperature may be a plain number (e.g. -40) or a string like "25C".
Row 1  : averaged initial value (A_inf)                     — skipped
Row 2  : "time (min)"/"time (sec)" | "Abs (XXX nm)"         — header (kinetic)
         "wavelength (nm)"         | "Abs"                  — header (scanning)
Row 3+ : numeric data (time or wavelength, absorbance)

Output structure
----------------
<output_dir>/
  <sheet_name>/
    {temp}C_{wavelength}_{n}.csv   (kinetic,  one CSV per trace)
    {temp}C_scan_{n}.csv           (scanning, one CSV per trace)

Each CSV contains exactly one trace in Cary-style format:
  Row 0 : channel label , ""
  Row 1 : "Time (min)" / "Time (sec)" / "Wavelength (nm)" , "Abs"
  Row 2+: numeric data

Usage
-----
    python extract_excel_half_life.py <excel_file> [output_folder]

If output_folder is omitted, CSVs are written next to the Excel file in a
subfolder called "extracted".
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_time_unit(header) -> str:
    h = str(header).lower()
    if "min" in h:
        return "Time (min)"
    return "Time (sec)"


def _parse_wavelength(header) -> str:
    """Extract wavelength integer from 'Abs (597 nm)' → '597'. Returns '?' if not found."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", str(header), re.IGNORECASE)
    if m:
        wl = m.group(1)
        return str(int(float(wl))) if float(wl) == int(float(wl)) else wl
    return "?"


def _temp_label(temp) -> str:
    """
    Format temperature as a filesystem-safe label.
    Plain numbers (int/float) and strings like '25C' are all handled.
    -40 → '-40C',  25 → '25C',  '25C' → '25C'
    """
    s = str(temp).strip()
    # Already has a unit suffix — return as-is (strip spaces only)
    if re.search(r"[a-zA-Z]", s):
        return s
    # Pure number
    try:
        t = float(s)
        return f"{int(t)}C" if t == int(t) else f"{t}C"
    except ValueError:
        return s


def _is_scanning(type_cell) -> bool:
    return "scan" in str(type_cell).lower()


def _safe_sheet_name(name: str) -> str:
    """Strip characters that are illegal in directory names on Windows."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


# ── per-sheet extraction ──────────────────────────────────────────────────────

def _extract_sheet(raw: pd.DataFrame, sheet_dir: Path, sheet_name: str) -> int:
    """
    Extract all traces from one sheet's DataFrame into sheet_dir.
    Returns the number of CSVs written.
    """
    if raw.shape[0] < 4 or raw.shape[1] < 2:
        print(f"  [{sheet_name}] Too few rows/cols — skipped.")
        return 0

    n_cols = raw.shape[1]
    sheet_dir.mkdir(parents=True, exist_ok=True)

    # Counter per (temp_label, wavelength_or_scan) key to handle duplicates
    counters: dict[str, int] = defaultdict(int)
    written = 0

    for i in range(0, n_cols - 1, 2):
        temp_raw = raw.iloc[0, i]
        type_raw = raw.iloc[0, i + 1]

        # Skip completely empty column pairs
        if pd.isna(temp_raw) and pd.isna(type_raw):
            continue

        time_hdr = raw.iloc[2, i]
        abs_hdr  = raw.iloc[2, i + 1]

        temp_str   = _temp_label(temp_raw)
        scanning   = _is_scanning(type_raw)
        time_unit  = _parse_time_unit(time_hdr)
        wavelength = _parse_wavelength(abs_hdr)

        # Data starts at row 3; drop rows where either column is non-numeric
        block = raw.iloc[3:, i:i + 2].copy()
        block.columns = ["x", "y"]
        block = block.apply(pd.to_numeric, errors="coerce").dropna()

        if block.empty:
            print(f"  [{sheet_name}] Col {i}-{i+1} ({temp_str}): no numeric data — skipped.")
            continue

        x_arr  = block["x"].values
        y_arr  = block["y"].values

        # Build a unique filename key and increment counter
        if scanning:
            key      = f"{temp_str}_scan"
            mode_tag = "scan"
        else:
            key      = f"{temp_str}_{wavelength}"
            mode_tag = wavelength

        counters[key] += 1
        n = counters[key]
        filename = sheet_dir / f"{temp_str}_{mode_tag}_{n}.csv"

        # Channel label inside the CSV (row 0)
        if scanning:
            channel_label = f"{temp_str}_{n}"
            x_header      = "Wavelength (nm)"
        else:
            channel_label = f"{temp_str}_{wavelength}"
            x_header      = time_unit

        _write_single_trace_csv(filename, channel_label, x_header, x_arr, y_arr)
        print(f"  [{sheet_name}] {filename.name}  ({len(x_arr)} points)")
        written += 1

    return written


def _write_single_trace_csv(
    path: Path,
    label: str,
    x_header: str,
    x_arr,
    y_arr,
) -> None:
    rows = [
        [label, ""],
        [x_header, "Abs"],
        *zip(x_arr, y_arr),
    ]
    pd.DataFrame(rows).to_csv(path, index=False, header=False)


# ── main entry ────────────────────────────────────────────────────────────────

def extract(excel_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(excel_path)

    sheets = xl.sheet_names
    print(f"\nFound {len(sheets)} sheet(s): {', '.join(sheets)}")

    total = 0
    for idx, sheet_name in enumerate(sheets, start=1):
        print(f"\n{'-' * 50}")
        print(f"Sheet {idx}/{len(sheets)}: {sheet_name!r}")
        raw = pd.read_excel(xl, sheet_name=sheet_name, header=None, dtype=object)
        print(f"  Size: {raw.shape[0]} rows x {raw.shape[1]} cols  "
              f"({raw.shape[1] // 2} trace column pair(s))")
        safe_name = _safe_sheet_name(sheet_name)
        sheet_dir = output_dir / safe_name
        print(f"  Output folder: {sheet_dir}")
        n = _extract_sheet(raw, sheet_dir, sheet_name)
        print(f"  -> {n} CSV(s) written for this sheet.")
        total += n

    print(f"\n{'=' * 50}")
    print(f"Total CSVs written: {total}")
    print(f"Output root:        {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    excel_path = Path(sys.argv[1]).resolve()
    if not excel_path.exists():
        print(f"Error: file not found: {excel_path}")
        sys.exit(1)

    output_dir = (
        Path(sys.argv[2]).resolve() if len(sys.argv) >= 3
        else excel_path.parent / "extracted"
    )

    print(f"Reading: {excel_path}")
    print(f"Output:  {output_dir}")
    extract(excel_path, output_dir)
    print("Done.")
