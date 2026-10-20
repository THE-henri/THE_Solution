# THE_Solution

Analysis program for UV/Vis spectroscopy data of photoswitchable compounds,
developed in the Hecht Lab: extinction coefficients, spectra of the
photoproduct, thermal half-lives, Arrhenius / Eyring analysis, chemical and
LED actinometry, and photoisomerisation quantum yields.

## How to cite

If you use THE_Solution for results in a publication or thesis, or if you
reuse, adapt or build on its code — by hand or with AI tools — please cite:

> H. Tertilt, *THE_Solution: Quantum Yield & Photochemistry Analysis*,
> version 1.0.0, Hecht Lab, 2026. https://github.com/THE-henri/THE_Solution

Please also cite the original methods the program implements. They are listed
in the in-app Handbook under **GENERAL → Citations & References** and in
[CITATION.cff](CITATION.cff).

## Running from source

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_gui.py
```

The Origin export (Half-Life → Publication, Spectra) needs OriginPro 2021 or
newer installed; everything else works without it.

## Building the Windows program

```
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python build_exe.py
```

The result is `dist/THE_Solution_<version>/`. Copy the whole folder — the
`.exe` needs the `_internal` folder next to it. Before a release, bump the
version in `app_info.py` and add release notes in the Handbook
(GENERAL → Version & Last Updated).

## License

MIT — see [LICENSE](LICENSE). Provided without warranty; you are responsible
for checking every result before you use it.
