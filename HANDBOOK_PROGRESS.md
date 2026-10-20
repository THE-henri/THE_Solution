# Handbook overhaul — progress tracker

Tracks the in-app Handbook rewrite (`gui/tabs/handbook_tab.py`) so it can be
picked up incrementally across sessions. The original plan is kept locally;
this file is enough to re-derive it.

Status markers: `[ ]` not started · `[~]` drafted, not yet reviewed with user ·
`[x]` done and verified with user in-app.

**Process change (2026-09-01):** in-app review is now batched at the end,
not done per-group. Keep drafting/auditing groups and marking them `[~]` as
finished; nothing moves to `[x]` until the user does one full in-app pass
over everything and gives feedback.

For groups that already had content before this project (SPECTRA, QUANTUM
YIELD, the LED part of ACTINOMETER), "written" and "audited against current
code" are tracked separately — old content isn't assumed correct.

---

## Navigation skeleton

- [x] All 8 groups registered in program-tab order, stub content in place — 2026-08-28

## 1. GENERAL *(front matter, new)*

Last worked on: 2026-10-20 — drafted for the v1.0 release. Version/date come
from `app_info.py`; DOI links open in the external browser.

- [~] About This Program — drafted (workflow table, folders, citing, contact)
- [~] Citations & References — how to cite THE_Solution (also in README /
  CITATION.cff), Methods (Volker/Steen/Crespi 2024 for LED QY; Sumi et al. 2014
  for Actinometer 1/2) and Software sections done
- [~] Version & Last Updated — drafted, incl. 1.0.0 release notes. Add a
  release-notes entry here for every new version

## 2. EXTINCTION COEFFICIENTS *(new)*

Last worked on: 2026-09-01 — all 3 sections drafted (verified against
`extinction_coeff_tab.py` and `core/optics.py::calculate_extinction_coefficients_integer_wavelengths`,
incl. the two-level-averaging detail that final `Std` is std-across-prep-means
and does not fold in within-prep replicate spread), not yet reviewed in-app

- [~] Overview
- [~] Input Files & Weighing Data
- [~] Calculation & Output

## 3. SPECTRA *(existing — audit only)*

Last worked on: —

- [ ] Spectra — Overview — audit against `spectra_tab.py`
- [ ] Spectra — Negative mode — audit against `workflows/spectra_calculation.py`
- [ ] Spectra — PCA modes — audit
- [ ] Spectra — positive_pss mode — audit

## 4. HALF-LIFE *(new, highest priority)*

Last worked on: 2026-08-28 — drafted, nav-collision bug fixed and confirmed working in-app

- [x] Overview
- [x] Loading Data & Channels
- [x] Fit Window & Outlier Removal (IQR)
- [x] Switch Convention & A∞
- [x] Exponential vs. Linear Fit (incl. A0 meaning, interval-averaging option)
- [x] Manual Segments
- [x] Scanning Kinetics
- [x] Publication Export & Master CSV structure
- [x] Common Pitfalls (temperature mislabeling, stale GUI fields)

## 5. THERMAL *(new)*

Last worked on: 2026-08-28 — first draft of all 5 sections written (formulas verified against `thermal_core.py`), not yet reviewed in-app

- [~] Overview (Arrhenius vs. Eyring)
- [~] Input: Grouping k(T) from the Half-Life master CSV
- [~] Arrhenius Fit & Output
- [~] Eyring Fit & Output
- [~] Common Pitfalls

## 6. ACTINOMETER *(Chemical new, LED existing — audit)*

Last worked on: 2026-09-01 — Chemical Actinometry drafted (verified against `actinometer_core.py::run_actinometry_file`, incl. correcting a wrong assumption from earlier research — it's a free-intercept fit, not through-origin, and the two built-in actinometers are generic "Actinometer 1/2" in code, not named compounds, so the section doesn't claim specific chemical identities). LED sections spot-checked (N(λ) formula, rate-equation structure) against current code — accurate, not rewritten. Input Files & Setup and QY Calculation Pipeline now also spot-checked: the ODE/units/formulas in both check out exactly against `actinometer_core.py` and `qy_core.py` (the two-process rework added new QY cases — A_only_thermal, AB_thermal, A_thermal_PSS — but didn't change this shared two-species rate-equation core, so the pipeline description is still accurate); two stale numbers fixed in the Key Parameters table — Volume (mL) typical value corrected 3.0→2.0 and Wavelength tolerance corrected 2.0→1.0 nm, to match `actinometer_tab.py`'s actual spinbox defaults.

- [~] Chemical Actinometry — Overview *(new)*
- [~] LED Actinometry — Overview — audited, accurate
- [~] LED Actinometry — Physics & ODE — audited, accurate
- [~] LED Actinometry — Input Files & Setup — spot-checked, 2 stale defaults corrected
- [~] LED QY — Calculation Pipeline — spot-checked, accurate, no changes needed

## 7. QUANTUM YIELD *(existing — full audit, not light)*

Reworked for two-process kinetics per recent commits ("Changes, especially to
QY for two processes") — treat every section as suspect until checked against
current `qy_core.py`/`qy_tab.py`.

Last worked on: —

- [ ] Quantum Yield — Overview
- [ ] Rate Equations (ODE model)
- [ ] LED Full-Spectrum Integration
- [ ] PSS Algebraic Method
- [ ] Photon Flux (N)
- [ ] Extinction Coefficients
- [ ] Offset Correction
- [ ] Fit Window & Irradiation Start
- [ ] Initial Slopes Estimate
- [ ] Initial Conditions — check the `plateau` averaging mode is documented (added this project)
- [ ] Thermal Back-Reaction (k_th)
- [ ] Fitting & Uncertainty
- [ ] Case: A_only
- [ ] Case: A_only_thermal
- [ ] Case: AB_both
- [ ] Case: AB_thermal
- [ ] Case: A_thermal_PSS (algebraic) — check new A0/eps_mon/conversion metadata fields are mentioned
- [ ] Exact Linearized Method (A_only)
- [ ] Data Types & File Formats

## 8. SPECTRAL EDITOR *(new)*

Last worked on: 2026-09-01 — all 3 sections drafted (verified against
`spectral_editor_tab.py` and `spectral_editor_core.py`, incl. surfacing
several silent-no-op behaviors as worth documenting: unmatched-label/no-tolerance-match
channels skipped silently in kinetic Offset-from-spectrum, concatenate combine
only keeping channels from the first file, index-based scan deletion having
no time/wavelength awareness, and near-zero normalization values being left
unmodified), not yet reviewed in-app

- [~] Overview
- [~] Kinetic Editor
- [~] Scanning Kinetic Editor

---

## Suggested order

In-app review is now a single batched pass at the end (see process-change note
above), so this list is about drafting/auditing order only — nothing here
implies anything has been checked in-app yet except where noted.

1. ~~Navigation skeleton~~ ✅
2. GENERAL — **deferred**, user wants to do at the end or in between
3. ~~HALF-LIFE~~ ✅ drafted (was verified in-app before the process change)
4. ~~THERMAL~~ ✅ drafted, awaiting the end-of-project in-app pass
5. ~~ACTINOMETER~~ ✅ fully drafted/audited, awaiting the end-of-project in-app pass
6. ~~EXTINCTION COEFFICIENTS~~ ✅ drafted, awaiting the end-of-project in-app pass
7. ~~SPECTRAL EDITOR~~ ✅ drafted, awaiting the end-of-project in-app pass
8. Final pass — **up next**: audit SPECTRA and QUANTUM YIELD against
   current code, then one full in-app review of every group with the user
