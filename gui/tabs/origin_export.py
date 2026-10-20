"""
Export to OriginPro via OriginLab's ``originpro`` package.

Requires Origin 2021 or newer installed on the computer. Origin is started
hidden, the project is written, saved and Origin is closed again. If Origin
is not available, OriginUnavailable is raised with a message that can be
shown to the user as-is.

Graphs are built from the templates in gui/assets/origin_templates/ (styles,
axis titles, legend layout); data, legends and ranges are filled in here.
To change the look of an exported graph, edit the template in Origin and
re-import it with tools/import_origin_templates.py.

Generic helpers (connect, dialog, folders, worksheets) are at the top so
other tabs can reuse them; workflow-specific exports follow below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app_info import resource_path


class OriginUnavailable(Exception):
    """Origin is not installed or could not be started."""


def template(name: str) -> str:
    return str(resource_path(f"gui/assets/origin_templates/{name}.otpu"))


# ── Generic helpers ──────────────────────────────────────────────────────────

def connect():
    """Start a hidden Origin instance and return the originpro module."""
    try:
        import originpro as op
    except Exception as exc:
        raise OriginUnavailable(
            "The Origin interface (originpro package) could not be loaded.\n\n"
            f"Details: {exc}") from exc
    try:
        op.set_show(False)          # first call actually starts Origin
    except Exception as exc:
        raise OriginUnavailable(
            "Origin could not be started.\n\n"
            "The Origin export needs OriginPro 2021 or newer installed on this "
            "computer. Use the CSV export instead and import the files into "
            "Origin by hand.\n\n"
            f"Details: {exc}") from exc
    return op


def run_export(export: Callable, project_path: Path, append: bool, folder_name: str,
               *args) -> str:
    """
    Open/create the project, create *folder_name* in its root, call
    export(op, *args) inside that folder, save and close Origin.
    Returns the name of the created Origin folder (enumerated if it existed).
    """
    op = connect()
    try:
        if append:
            if not op.open(str(project_path)):
                raise RuntimeError(f"Origin could not open {project_path}")
        else:
            op.new()
            for book in list(op.pages("w")):   # drop the empty default Book1
                book.destroy()
        op.pe.cd("/")
        created = op.pe.mkdir(folder_name)       # "name1" if it already exists
        op.pe.cd(created)
        export(op, *args)
        op.pe.cd("/")
        # appending: save in place — "save as" onto the open file fails
        if not op.save("" if append else str(project_path)):
            raise RuntimeError(f"Origin could not save {project_path}")
        return created.strip("/").split("/")[-1]
    finally:
        op.exit()


def export_with_dialog(parent, default_name: str, start_dir: str,
                       export: Callable, *args, status_label=None) -> bool:
    """
    Ask new/existing project, folder name and file, then run the export with
    a wait cursor and user-readable error messages. Returns True on success.
    """
    import os
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

    box = QMessageBox(parent)
    box.setWindowTitle("Export to Origin")
    box.setText("Create a new Origin project, or add the data as a new folder "
                "to an existing project?")
    box.setInformativeText("If you add to an existing project, close it in Origin first.")
    btn_new = box.addButton("New project…", QMessageBox.ButtonRole.AcceptRole)
    btn_add = box.addButton("Add to existing…", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()
    if box.clickedButton() not in (btn_new, btn_add):
        return False
    append = box.clickedButton() is btn_add

    folder_name, ok = QInputDialog.getText(
        parent, "Export to Origin", "Name of the Origin folder:", text=default_name)
    folder_name = folder_name.strip()
    if not ok or not folder_name:
        return False

    if append:
        path, _ = QFileDialog.getOpenFileName(
            parent, "Select Origin project", start_dir, "Origin projects (*.opju *.opj)")
    else:
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save new Origin project",
            str(Path(start_dir) / f"{folder_name}.opju"), "Origin projects (*.opju)")
    if not path:
        return False

    def _status(text):
        if status_label is not None:
            status_label.setText(text)

    _status("Starting Origin… this can take up to a minute.")
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    QApplication.processEvents()
    try:
        folder = run_export(export, Path(path), append, folder_name, *args)
    except OriginUnavailable as exc:
        QApplication.restoreOverrideCursor()
        _status("Origin export not possible — see message.")
        QMessageBox.warning(parent, "Origin not available", str(exc))
        return False
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        _status(f"Origin export failed: {exc}")
        QMessageBox.critical(parent, "Origin export failed", str(exc))
        return False
    QApplication.restoreOverrideCursor()
    _status(f"Origin → {path}  (folder '{folder}')")
    print(f"Origin export → {path} (folder '{folder}')")
    if QMessageBox.question(
            parent, "Export to Origin",
            f"Saved to folder '{folder}' in\n{path}\n\nOpen the project in Origin now?"
    ) == QMessageBox.StandardButton.Yes:
        os.startfile(path)
    return True


def fill_sheet(wks, df: pd.DataFrame, axes: str, lnames=None, units=None, comments=None):
    """Write *df* into *wks* and set designations ('x', 'y', 'e', …) and column labels."""
    wks.from_df(df)
    if axes:
        wks.cols_axis(axes, repeat=False)
    for row, labels in (("L", lnames), ("U", units), ("C", comments)):
        if labels:
            wks.set_labels(labels, row)


def _wl(channel) -> str:
    """Channel label for legends: '672', '672nm' or 'Sample 1_672' → '672 nm'."""
    m = re.search(r"(?<!\d)(\d{3,4}(?:\.\d+)?)\s*(?:nm)?\s*$", str(channel))
    return f"{m.group(1)} nm" if m else str(channel)


def _lt(gp, script: str):
    """Run LabTalk with *gp*'s first layer active."""
    gp[0].lt_exec(script)


def _lt_str(op, gp, expr: str) -> str:
    op.lt_exec(f'win -a {gp.name}; string __s$ = {expr};')
    return op.get_lt_str("__s")


def _lt_num(op, gp, expr: str) -> float:
    op.lt_exec(f"win -a {gp.name};")
    return op.lt_float(expr)


def set_legend(gp, lines: list[str]):
    """Replace the legend text (one entry per list item)."""
    text = "%(CRLF)".join(lines).replace('"', "'")
    _lt(gp, f'legend.text$ = "{text}";')


def _num(v: float, digits: int = 6) -> str:
    return f"{v:.{digits}G}"


# ── Half-Life → Publication ──────────────────────────────────────────────────

def export_half_life_publication(op, raw_files, segment_slots):
    """
    raw_files      – (raw_data.csv, segments.csv or None) per raw entry
    segment_slots  – per segment subplot: list of (segment folder, active channels)
    """
    _raw_to_origin(op, raw_files)
    for i, entries in enumerate(segment_slots, 1):
        if entries:
            _segment_to_origin(op, i, entries)


def _raw_to_origin(op, raw_files):
    raw_files = [(r, s) for r, s in raw_files if r and Path(r).exists()]
    if not raw_files:
        return
    wb = op.new_book("w", lname="Raw data")
    sheets, legend, segments, t_min = [], [], [], []
    for n, (raw_path, seg_path) in enumerate(raw_files):
        raw_path = Path(raw_path)
        df = pd.read_csv(raw_path)
        chs = [c for c in df.columns if c != "time_s"]
        name = raw_path.stem.replace("raw_data", "").strip("_") or f"Raw{n + 1}"
        wks = wb[0] if n == 0 else wb.add_sheet(name)
        wks.name = name
        # Absorbance as measured (may or may not have been zeroed at the instrument)
        fill_sheet(wks, df[["time_s"] + chs], "x" + "y" * len(chs),
                   lnames=["Time"] + ["Abs"] * len(chs),
                   units=["s"] + [""] * len(chs),
                   comments=[""] + [_wl(c) for c in chs])
        sheets.append((wks, len(chs)))
        t_min.append(float(df["time_s"].min()))
        legend += [_wl(c) for c in chs]
        if seg_path and Path(seg_path).exists():
            seg_df = pd.read_csv(seg_path)
            fill_sheet(wb.add_sheet(f"Segments {name}"), seg_df, "")
            segments += list(zip(seg_df["t_start_s"], seg_df["t_end_s"]))

    n_ch = len(legend)
    raw_tpl = ("HalfLife_Raw_1ch" if n_ch == 1 else
               "HalfLife_Raw_4ch" if n_ch >= 4 else "HalfLife_Raw")   # 3-style template for 2–3
    gp = op.new_graph(lname="Raw kinetic traces", template=template(raw_tpl))
    gl = gp[0]
    for wks, n_ch in sheets:
        for j in range(n_ch):
            gl.add_plot(wks, coly=1 + j, colx=0, type="?")
    gl.rescale()
    if gl.axis("x").sfrom < 0 <= min(t_min):
        gl.set_xlim(begin=0)                          # time axis starts at 0
    set_legend(gp, [f"\\l({i}) {e}" for i, e in enumerate(legend, 1)])
    _set_segment_bands(op, gp, segments)


def _set_segment_bands(op, gp, segments):
    """Grey bands (reference-line fills, styled like the template's) at the segments."""
    segments = sorted({(round(float(a), 3), round(float(b), 3)) for a, b in segments})
    n_tpl = int(_lt_num(op, gp, "layer.x.reflines.count") or 0)
    src = next((i for i in range(1, n_tpl + 1)
                if _lt_num(op, gp, f"layer.x.refline{i}.filltonext") > 0), None)
    fc = _lt_num(op, gp, f"layer.x.refline{src}.fillcolor") if src else 19
    ft = _lt_num(op, gp, f"layer.x.refline{src}.filltrans") if src else 60
    # Lines 1 and 2 are the axis begin/end lines — keep them, without fill.
    # Segment k: start line (2 + k) fills to its end line (2 + n + k).
    n = len(segments)
    script = [f"layer.x.reflines.count = {2 + 2 * n};",
              "layer.x.refline1.filltonext = 0; layer.x.refline2.filltonext = 0;"]
    for k, (t0, t1) in enumerate(segments, 1):
        s, e = 2 + k, 2 + n + k
        script += [
            f"layer.x.refline{s}.value = {t0}; layer.x.refline{s}.filltonext = {e};",
            f"layer.x.refline{s}.fillcolor = {fc:g}; layer.x.refline{s}.filltrans = {ft:g};",
            f"layer.x.refline{e}.value = {t1}; layer.x.refline{e}.filltonext = 0;",
        ]
        for i in (s, e):
            script.append(f"layer.x.refline{i}.lineshow = 0; layer.x.refline{i}.labelshow = 0;")
    _lt(gp, "".join(script))


def _segment_to_origin(op, slot: int, entries):
    wb = op.new_book("w", lname=f"Segment {slot}")
    channels = []          # (label, wks, has_fit, params row)
    for folder, active in entries:
        folder = Path(folder)
        data_f, fit_f, params_f = (folder / "data_points.csv", folder / "fit_line.csv",
                                   folder / "fit_params.csv")
        if not data_f.exists() or not params_f.exists():
            continue
        data_df = pd.read_csv(data_f)
        fit_df = pd.read_csv(fit_f) if fit_f.exists() else None
        params_df = pd.read_csv(params_f)
        params_df["channel"] = params_df["channel"].astype(str)
        for ch in params_df["channel"]:
            if ch not in [str(a) for a in active]:
                continue
            t_col = f"{ch}_time_s" if f"{ch}_time_s" in data_df.columns else "time_s"
            ln_col = f"{ch}_ln_A"
            if ln_col not in data_df.columns or t_col not in data_df.columns:
                continue
            cols = {"t": data_df[t_col], "lnA": data_df[ln_col]}
            lnames, units, comments = ["Time", "ln|A − A∞|"], ["s", ""], ["", _wl(ch)]
            fit_col = f"{ch}_fit"
            has_fit = fit_df is not None and fit_col in fit_df.columns
            if has_fit:
                ft_col = f"{ch}_time_s" if f"{ch}_time_s" in fit_df.columns else "time_s"
                cols["t_fit"], cols["fit"] = fit_df[ft_col], fit_df[fit_col]
                lnames += ["Time", "Fit"]
                units += ["s", ""]
                comments += ["", f"fit ({_wl(ch)})"]
            df = pd.DataFrame({k: pd.Series(v.values) for k, v in cols.items()})
            wks = wb[0] if not channels else wb.add_sheet()
            wks.name = _wl(ch)[:31]
            fill_sheet(wks, df, "xy" * (len(cols) // 2),
                       lnames=lnames, units=units, comments=comments)
            row = params_df[params_df["channel"] == ch].iloc[0]
            channels.append((ch, wks, has_fit, row))
    if not channels:
        return
    fill_sheet(wb.add_sheet("Fit parameters"),
               pd.DataFrame([c[3] for c in channels]).reset_index(drop=True), "")

    n = len(channels)
    gp = op.new_graph(lname=f"Segment {slot}",
                      template=template(f"HalfLife_Segment_{min(n, 4)}ch"))
    gl = gp[0]
    for _, wks, _, _ in channels:                     # all data first …
        gl.add_plot(wks, coly=1, colx=0, type="?")
    fit_idx, k = {}, n
    for ch, wks, has_fit, _ in channels:              # … then all fits
        if has_fit:
            k += 1
            fit_idx[ch] = k
            gl.add_plot(wks, coly=3, colx=2, type="?")
    gl.rescale()

    # Legend in the template's layout: equation, then per channel
    # "data; fit" and a line with the fit parameters
    tpl_first = _lt_str(op, gp, "legend.text$").replace("\r", "").split("\n")[0]
    lines = [tpl_first] if "\\l(" not in tpl_first else []
    for i, (ch, _, has_fit, row) in enumerate(channels, 1):
        entry = f" \\l({i}) {_wl(ch)}"
        if has_fit:
            entry += f"; \\l({fit_idx[ch]}) fit ({_wl(ch)})"
        lines.append(entry)
        lines.append(
            f" \\i(k) = {_num(row['k_s-1'])} s\\+(-1), \\i(A)\\-(0) = {_num(row['A0'], 5)}, "
            f"\\i(A)\\-(∞) = {_num(row['A_inf'], 5)}, \\i(R)\\+(2) = {row['R2']:.5f}")
    set_legend(gp, lines)


# ── Spectra → ε of species A and B ───────────────────────────────────────────

_BAND_LABEL = {
    "negative":     "B ± 1σ",
    "negative_pca": "B 95% CI (bootstrap)",
    "positive_pca": "B 95% CI (bootstrap)",
    "positive_pss": "B range (f\\-(B) ± error)",
}


def export_spectra(op, result):
    """
    result – SpectraResult (grid, S_A, S_B, S_B_lo/hi/std, mode, is_epsilon, meta_lines)

    Workbook with A, B and B's uncertainty; graph from the Spectra_ExtCoeff
    template with B's uncertainty as a shaded band.
    """
    eps = result.is_epsilon
    unit = "L mol\\+(-1) cm\\+(-1)" if eps else ""
    quantity = "ε" if eps else "Abs"
    wb = op.new_book("w", lname="Spectra")
    wks = wb[0]
    wks.name = "Spectra"
    cols = {"wl": result.grid, "A": result.S_A}
    axes = "xy"
    lnames, comments = ["Wavelength", quantity], ["", "A (initial)"]
    has_b = result.S_B is not None
    if has_b:
        cols.update({"B": result.S_B, "B_lo": result.S_B_lo, "B_hi": result.S_B_hi,
                     "B_std": result.S_B_std})
        axes += "yyye"
        lnames += [quantity, f"{quantity} lower", f"{quantity} upper", f"{quantity} error"]
        comments += ["B (extracted)", "B lower bound", "B upper bound",
                     _BAND_LABEL.get(result.mode, "B uncertainty")]
    fill_sheet(wks, pd.DataFrame(cols), axes, lnames=lnames,
               units=["nm"] + [unit] * (len(cols) - 1), comments=comments)
    if result.meta_lines:
        fill_sheet(wb.add_sheet("Info"),
                   pd.DataFrame({"Info": [str(m) for m in result.meta_lines]}), "")

    gp = op.new_graph(lname="Spectra A and B", template=template("Spectra_ExtCoeff"))
    gl = gp[0]
    gl.add_plot(wks, coly=1, colx=0, type="?")
    cname = f"{result.compound} " if result.compound else ""   # header compound name
    legend = [f"\\l(1) {cname}(A)"]
    if has_b:
        p_b = gl.add_plot(wks, coly=2, colx=0, type="?")
        legend.append(f"\\l(2) {cname}(B)")
        if np.any(np.isfinite(result.S_B_lo)):
            p_lo = gl.add_plot(wks, coly=3, colx=0, type="l")
            p_hi = gl.add_plot(wks, coly=4, colx=0, type="l")
            for p in (p_lo, p_hi):
                p.color = p_b.color
                p.set_cmd("-w 0")                 # band only, no outline
            p_lo.set_fill_area(type=9)            # fill to next plot (upper bound)
            p_lo.set_cmd(f"-pfb color({','.join(str(c) for c in p_b.color)})")
            p_lo.set_cmd("-pft 70")               # fill transparency (%)
            legend.append(f"\\l(3) {_BAND_LABEL.get(result.mode, 'B uncertainty')}")
    gl.rescale()
    if not eps:
        gl.axis("y").title = "Absorbance"
    set_legend(gp, legend)
