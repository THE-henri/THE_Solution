"""
Import an Origin graph template into the program (gui/assets/origin_templates/).

    .venv/Scripts/python tools/import_origin_templates.py <source.otpu> <Name> <n_plots>

Example (after editing the raw half-life template in Origin):
    .venv/Scripts/python tools/import_origin_templates.py my_raw.otpu HalfLife_Raw 3

<Name> must be one of the names the exports use:
    HalfLife_Raw (2–3 channels), HalfLife_Raw_1ch, HalfLife_Raw_4ch,
    HalfLife_Segment_1ch … HalfLife_Segment_4ch, Spectra_ExtCoeff
<n_plots> is the number of plot styles the template defines, in plot order
(e.g. a 2-channel segment template: 2 data + 2 fits = 4).

The repo is public, so the template is cleaned before saving: neutral dummy
data is plotted into it (replacing references to the workbooks the original
graph was made from), and the window and file name are replaced by <Name>.
Check the result with a quick export before committing — the legend text of
the source template is kept, except that it is overwritten during export.

Needs OriginPro 2021+ (runs Origin visibly; saving templates fails when hidden).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import originpro as op

DEST = Path(__file__).resolve().parent.parent / "gui" / "assets" / "origin_templates"


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, name, n = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
    with tempfile.TemporaryDirectory() as tmp:
        tpl = Path(tmp) / f"{name}.otpu"      # neutral name → stored in the header
        shutil.copy(src, tpl)
        op.set_show(True)
        try:
            op.new()
            for book in list(op.pages("w")):
                book.destroy()
            wb = op.new_book("w", lname="Data")
            x = np.linspace(0, 1, 11)
            wks = wb[0]
            wks.from_df(pd.DataFrame({f"c{i}": x if i % 2 == 0 else x * (i // 2 + 1)
                                      for i in range(2 * n)}))
            wks.cols_axis("xy" * n)
            g = op.new_graph(template=str(tpl))
            for i in range(n):
                g[0].add_plot(wks, coly=2 * i + 1, colx=2 * i, type="?")
            op.lt_exec(f'win -r {g.name} {name}; win -a {name}; page.longname$ = "{name}";')
            op.lt_exec(f'template_saveas pg:=[{name}] template:="{name}" filepath:="{DEST}";')
        finally:
            op.exit()
    out = DEST / f"{name}.otpu"
    print(f"Saved {out}" if out.exists() else "Saving the template failed.")


if __name__ == "__main__":
    main()
