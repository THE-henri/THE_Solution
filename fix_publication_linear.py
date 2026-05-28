"""
fix_publication_linear.py
─────────────────────────
Retroactively replaces the exponential-model fit line saved in publication
segment folders with the correct linear regression (np.polyfit on the
already-cleaned log data in data_points.csv).

What it does per segment folder
────────────────────────────────
1. Reads  data_points.csv  →  {channel}_time_s  and  {channel}_ln_A
2. Runs np.polyfit  →  k_linear, t_half, R²   (matches individual plot logic)
3. Overwrites  fit_line.csv   with 300-point smooth polyfit line
4. Overwrites  fit_params.csv  with the linear k / t½ / R²

Cross-check
────────────
Loads half_life_master.csv (which already has the correct linear k values)
and compares the recomputed k against every master row at the same temperature.
Because channel labels differ between the two files, matching is done by
temperature only; the report shows all master k values at that temperature
so you can verify visually.

Note: A_inf is 0 in all these segments (positive switch or fixed A∞ = 0),
so no A_inf correction is needed — the ln_A column is already correct.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ── Project roots ─────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\StdUser\Documents\THE_Solution\data\Change_k_exp_to_linear_publication")
PROJECT_ROOTS = [
    BASE / "AZA-OMe",
    BASE / "AZA-tBu",
]


# ── Core maths ────────────────────────────────────────────────────────────────

def polyfit_channel(t: np.ndarray, ln_A: np.ndarray):
    """
    Linear regression on (t, ln_A), ignoring non-finite values.
    Returns (k, t_half_s, r2, coeffs) or raises ValueError.
    k = -slope  (first-order decay: ln_A = ln_A0 - k*t)
    """
    mask = np.isfinite(t) & np.isfinite(ln_A)
    t_v, y_v = t[mask], ln_A[mask]
    if len(t_v) < 2:
        raise ValueError("Too few valid data points for polyfit")

    coeffs = np.polyfit(t_v, y_v, 1)          # [slope, intercept]
    k      = -float(coeffs[0])
    if k <= 0:
        raise ValueError(f"Negative or zero k ({k:.6g}) — check data")

    t_half = np.log(2) / k

    y_pred = np.polyval(coeffs, t_v)
    ss_res = float(np.sum((y_v - y_pred) ** 2))
    ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return k, t_half, r2, coeffs


# ── Per-segment processing ─────────────────────────────────────────────────────

def process_segment(seg_dir: Path) -> list[dict]:
    """
    Rewrites fit_line.csv and fit_params.csv in seg_dir.
    Returns list of result dicts (one per channel) for the report.
    """
    data_f   = seg_dir / "data_points.csv"
    fit_f    = seg_dir / "fit_line.csv"
    params_f = seg_dir / "fit_params.csv"

    if not data_f.exists() or not params_f.exists():
        print(f"    [SKIP] {seg_dir.name}: missing data_points.csv or fit_params.csv")
        return []

    data_df   = pd.read_csv(data_f)
    params_df = pd.read_csv(params_f)

    new_params = []
    new_fit_frames = []
    results = []

    for _, row in params_df.iterrows():
        ch     = row["channel"]            # e.g. 375, 408, 362 …
        old_k  = float(row["k_s-1"])
        old_r2 = float(row["R2"])

        # Normalise channel to an integer string when possible so column
        # lookups like "375_time_s" work even if pandas read ch as 375.0
        try:
            ch_str = str(int(float(ch))) if float(ch) == int(float(ch)) else str(ch)
        except (ValueError, TypeError):
            ch_str = str(ch)

        t_col  = f"{ch_str}_time_s"
        ln_col = f"{ch_str}_ln_A"

        if t_col not in data_df.columns or ln_col not in data_df.columns:
            print(f"    [WARN] {seg_dir.name} / ch {ch_str}: columns not found — kept unchanged")
            new_params.append(row.to_dict())
            results.append(dict(channel=ch_str, old_k=old_k, new_k=None,
                                old_r2=old_r2, new_r2=None, status="MISSING COLUMNS"))
            continue

        t_arr  = data_df[t_col].astype(float).values
        ln_arr = data_df[ln_col].astype(float).values

        try:
            new_k, new_th, new_r2, coeffs = polyfit_channel(t_arr, ln_arr)
        except ValueError as exc:
            print(f"    [WARN] {seg_dir.name} / ch {ch_str}: {exc} — kept unchanged")
            new_params.append(row.to_dict())
            results.append(dict(channel=ch_str, old_k=old_k, new_k=None,
                                old_r2=old_r2, new_r2=None, status=str(exc)))
            continue

        # 300-point smooth fit line over the valid time range
        t_valid = t_arr[np.isfinite(t_arr) & np.isfinite(ln_arr)]
        t_dense = np.linspace(t_valid[0], t_valid[-1], 300)
        ln_fit  = np.polyval(coeffs, t_dense)

        new_fit_frames.append(pd.DataFrame({
            f"{ch_str}_time_s": t_dense,
            f"{ch_str}_fit":    ln_fit,
        }))
        new_params.append({
            "channel":  ch_str,
            "k_s-1":    new_k,
            "t_half_s": new_th,
            "A0":       row.get("A0", float("nan")),
            "A_inf":    row.get("A_inf", float("nan")),
            "R2":       new_r2,
        })
        results.append(dict(channel=ch_str, old_k=old_k, new_k=new_k,
                            old_r2=old_r2, new_r2=new_r2, status="OK"))

    # Overwrite files
    pd.DataFrame(new_params).to_csv(params_f, index=False)
    if new_fit_frames:
        pd.concat(new_fit_frames, axis=1).to_csv(fit_f, index=False)

    return results


# ── Temperature folder name → float ──────────────────────────────────────────

def parse_temp(name: str) -> float:
    """'25C' → 25.0,  '-36C' → -36.0"""
    try:
        return float(name.rstrip("Cc"))
    except ValueError:
        return float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []   # accumulated for final report

    for root in PROJECT_ROOTS:
        pub_root    = root / "half_life" / "results" / "publication"
        master_path = root / "half_life" / "results" / "half_life_master.csv"

        print(f"\n{'='*72}")
        print(f"Dataset : {root.name}")
        print(f"Pub dir : {pub_root}")

        if not pub_root.exists():
            print("  [ERROR] Publication folder not found — skipping.")
            continue

        # Load master CSV (for cross-check)
        master_by_temp: dict[float, list[float]] = {}
        if master_path.exists():
            mdf = pd.read_csv(master_path)
            mdf["Temperature_C"] = pd.to_numeric(mdf["Temperature_C"], errors="coerce")
            mdf["k"] = pd.to_numeric(mdf["k"], errors="coerce")
            for temp, grp in mdf.groupby("Temperature_C"):
                master_by_temp[float(temp)] = grp["k"].dropna().tolist()
            print(f"Master  : {len(mdf)} rows, "
                  f"{len(master_by_temp)} temperatures")
        else:
            print("  [WARN] Master CSV not found — cross-check skipped.")

        # Walk temperature subfolders
        for t_dir in sorted(pub_root.iterdir()):
            if not t_dir.is_dir():
                continue
            temp_c = parse_temp(t_dir.name)
            master_ks = master_by_temp.get(temp_c, [])

            seg_dirs = sorted(
                d for d in t_dir.iterdir()
                if d.is_dir() and d.name.startswith("segment_")
            )
            if not seg_dirs:
                continue

            print(f"\n  {t_dir.name}  ({temp_c} °C)  —  "
                  f"{len(seg_dirs)} segment folder(s)")

            for seg_dir in seg_dirs:
                results = process_segment(seg_dir)
                for r in results:
                    r["dataset"] = root.name
                    r["temp_c"]  = temp_c
                    r["segment"] = seg_dir.name
                    r["master_ks"] = master_ks
                    all_rows.append(r)
                    status = r["status"]
                    if r["new_k"] is not None:
                        print(f"    {seg_dir.name:<30}  ch {r['channel']:>4}  "
                              f"old k={r['old_k']:.6f}  new k={r['new_k']:.6f}  [{status}]")
                    else:
                        print(f"    {seg_dir.name:<30}  ch {r['channel']:>4}  [{status}]")

    # ── Final cross-check report ───────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("CROSS-CHECK AGAINST MASTER CSV")
    print("(master rows are grouped by temperature — match by proximity)")
    print(f"{'='*72}")

    prev_dataset = None
    prev_temp    = None

    for r in all_rows:
        if r["new_k"] is None:
            continue

        if r["dataset"] != prev_dataset:
            print(f"\n--- {r['dataset']} ---")
            prev_dataset = r["dataset"]
            prev_temp    = None

        if r["temp_c"] != prev_temp:
            mk_str = ", ".join(f"{v:.6f}" for v in r["master_ks"]) or "—"
            print(f"\n  T = {r['temp_c']:>6.1f} °C   master k values: [{mk_str}]")
            prev_temp = r["temp_c"]

        new_k = r["new_k"]
        if r["master_ks"]:
            closest    = min(r["master_ks"], key=lambda v: abs(v - new_k))
            rel_diff   = abs(new_k - closest) / closest if closest != 0 else float("nan")
            match_flag = "OK <1%" if rel_diff < 0.01 else f"DIFF {rel_diff*100:.1f}%"
        else:
            match_flag = "no master"

        print(f"    {r['segment']:<35} ch {r['channel']:>4}  "
              f"new k={new_k:.6f}  R2={r['new_r2']:.4f}  {match_flag}")

    print(f"\n{'='*72}")
    print("Done.  fit_params.csv and fit_line.csv have been overwritten.")


if __name__ == "__main__":
    main()
