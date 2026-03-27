"""
Handbook tab — reference documentation for QY Tool methods and calculations.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QTextBrowser, QSplitter,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ── Colour palette (matches style.qss) ────────────────────────────────────────
_BG        = "#1e1e2e"
_BG2       = "#252535"
_BORDER    = "#3a3a50"
_TEXT      = "#e0e0e0"
_TEXT_DIM  = "#a0a0c0"
_ACCENT    = "#5b8dee"
_ACCENT2   = "#7aa2f7"
_GREEN     = "#9ece6a"
_ORANGE    = "#e0af68"
_RED       = "#f7768e"
_PURPLE    = "#bb9af7"
_CYAN      = "#7dcfff"

_CSS = f"""
    body {{
        background-color: {_BG};
        color: {_TEXT};
        font-family: "Segoe UI", "Arial", sans-serif;
        font-size: 10pt;
        margin: 18px 26px;
        line-height: 1.55;
    }}
    h1 {{
        color: {_ACCENT2};
        font-size: 15pt;
        font-weight: bold;
        margin-top: 0;
        margin-bottom: 6px;
        border-bottom: 1px solid {_BORDER};
        padding-bottom: 5px;
    }}
    h2 {{
        color: {_ACCENT};
        font-size: 11pt;
        font-weight: bold;
        margin-top: 18px;
        margin-bottom: 4px;
    }}
    h3 {{
        color: {_CYAN};
        font-size: 10pt;
        font-weight: bold;
        margin-top: 12px;
        margin-bottom: 3px;
    }}
    p  {{ margin: 5px 0 8px 0; }}
    ul {{ margin: 4px 0 8px 18px; padding: 0; }}
    li {{ margin-bottom: 3px; }}
    code {{
        background-color: {_BG2};
        color: {_GREEN};
        font-family: "Consolas", "Courier New", monospace;
        font-size: 9.5pt;
        padding: 1px 4px;
        border-radius: 3px;
    }}
    pre {{
        background-color: {_BG2};
        color: {_GREEN};
        font-family: "Consolas", "Courier New", monospace;
        font-size: 9.5pt;
        padding: 10px 14px;
        border-left: 3px solid {_ACCENT};
        margin: 8px 0 10px 0;
        white-space: pre-wrap;
    }}
    .eq {{
        background-color: {_BG2};
        color: {_ORANGE};
        font-family: "Consolas", "Courier New", monospace;
        font-size: 9.5pt;
        padding: 8px 14px;
        border-left: 3px solid {_ORANGE};
        margin: 8px 0 10px 0;
        display: block;
    }}
    .note {{
        background-color: #1a2040;
        border-left: 3px solid {_ACCENT};
        padding: 6px 12px;
        margin: 8px 0 10px 0;
        color: {_TEXT_DIM};
        font-size: 9.5pt;
    }}
    .warn {{
        background-color: #2a1a10;
        border-left: 3px solid {_ORANGE};
        padding: 6px 12px;
        margin: 8px 0 10px 0;
        color: {_ORANGE};
        font-size: 9.5pt;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 8px 0 12px 0;
    }}
    th {{
        background-color: {_BG2};
        color: {_ACCENT};
        padding: 5px 10px;
        border: 1px solid {_BORDER};
        text-align: left;
        font-size: 9.5pt;
    }}
    td {{
        padding: 4px 10px;
        border: 1px solid {_BORDER};
        font-size: 9.5pt;
        vertical-align: top;
    }}
    tr:nth-child(even) td {{ background-color: #232333; }}
    a {{ color: {_ACCENT2}; }}
"""


# ── Section content ────────────────────────────────────────────────────────────

_SECTIONS:     dict[str, str] = {}   # QY sections
_SECTIONS_LED: dict[str, str] = {}   # LED / Actinometry sections

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Quantum Yield — Overview"] = f"""
<h1>Quantum Yield Calculation</h1>

<p>
The quantum yield (Φ) of a photochemical reaction is the fraction of absorbed photons
that lead to a chemical transformation.  For a photoisomerisation A&nbsp;⇌&nbsp;B:
</p>

<pre class="eq">Φ_AB = (moles of A converted to B) / (moles of photons absorbed by A)</pre>

<p>
This tool fits Φ by integrating the coupled ODE that describes the time evolution
of concentrations [A](t) and [B](t) under irradiation, and comparing the simulated
absorbance curve against the measured kinetic trace.  The least-squares minimum
determines the best-fit quantum yield(s).
</p>

<h2>Experimental Cases</h2>

<table>
  <tr><th>Case</th><th>Description</th><th>Fitted parameters</th></tr>
  <tr>
    <td><code>A_only</code></td>
    <td>Only species A absorbs at the irradiation wavelength (ε_B ≈ 0 at λ_irr).
        The forward reaction A→B is driven by light; no photo-driven back-reaction.</td>
    <td>Φ_AB (Φ_BA = 0 fixed)</td>
  </tr>
  <tr>
    <td><code>AB_both</code></td>
    <td>Both A and B absorb at λ_irr.  The forward (A→B) and back (B→A)
        photo-reactions are fitted simultaneously.</td>
    <td>Φ_AB and Φ_BA</td>
  </tr>
  <tr>
    <td><code>A_thermal_PSS</code></td>
    <td>Only A absorbs; B thermally reverts to A (rate constant k_th).
        At the photostationary state (PSS) the net flux into and out of B is zero —
        this algebraic condition gives Φ_AB directly without ODE fitting.</td>
    <td>Φ_AB (PSS algebraic)</td>
  </tr>
</table>

<h2>Quick-Start Workflow</h2>
<ul>
  <li><b>Stage 1</b> — Select raw data file(s), data type, and photon flux source.</li>
  <li><b>Stage 2</b> — Set experimental parameters: sample info, baseline correction,
      fit window, and thermal back-reaction rate.</li>
  <li><b>Stage 3</b> — Provide extinction coefficients for species A (and B if needed).</li>
  <li><b>Stage 4</b> — Set initial guesses and bounds, then run the fit.
      Inspect the plot and residuals to validate the result.</li>
</ul>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Rate Equations (ODE model)"] = f"""
<h1>Rate Equations — ODE Model</h1>

<p>
The photochemical system is modelled as a reversible two-species isomerisation
A&nbsp;⇌&nbsp;B.  Concentrations [A] and [B] evolve according to:
</p>

<pre class="eq">d[A]/dt =  rate_photo + k_th·[B]
d[B]/dt = -rate_photo - k_th·[B]</pre>

<p>
where the photochemical rate is:
</p>

<pre class="eq">rate_photo = (N / V) · l · f(A_tot) · (Φ_BA·[B]·ε_B - Φ_AB·[A]·ε_A)</pre>

<p>
and <i>f</i>(A<sub>tot</sub>) is the Beer–Lambert absorption factor:
</p>

<pre class="eq">f(A_tot) = (1 − 10^(−A_tot)) / A_tot

A_tot = (ε_A·[A] + ε_B·[B]) · l</pre>

<p>
At very low absorbance (A_tot → 0) this reduces to f → ln(10) ≈ 2.303,
recovering the optically thin (differential) limit.
</p>

<h2>Symbol Definitions</h2>
<table>
  <tr><th>Symbol</th><th>Unit</th><th>Meaning</th></tr>
  <tr><td><code>[A], [B]</code></td><td>mol L⁻¹</td><td>Molar concentrations of species A and B</td></tr>
  <tr><td><code>N</code></td><td>mol s⁻¹</td><td>Incident photon flux (total photons per second entering the cuvette)</td></tr>
  <tr><td><code>V</code></td><td>L</td><td>Irradiated sample volume</td></tr>
  <tr><td><code>l</code></td><td>cm</td><td>Optical path length</td></tr>
  <tr><td><code>ε_A, ε_B</code></td><td>L mol⁻¹ cm⁻¹</td><td>Molar absorption coefficients of A and B at the irradiation wavelength</td></tr>
  <tr><td><code>Φ_AB</code></td><td>—</td><td>Quantum yield for A→B conversion</td></tr>
  <tr><td><code>Φ_BA</code></td><td>—</td><td>Quantum yield for B→A conversion (0 in <code>A_only</code> case)</td></tr>
  <tr><td><code>k_th</code></td><td>s⁻¹</td><td>First-order thermal back-reaction rate constant (B→A); 0 if absent</td></tr>
  <tr><td><code>A_tot</code></td><td>—</td><td>Total absorbance of the mixture at the irradiation wavelength</td></tr>
</table>

<h2>Simulated Absorbance</h2>

<p>
After integrating the ODE to obtain [A](t) and [B](t), the simulated absorbance
at each monitoring wavelength λ<sub>m</sub> is:
</p>

<pre class="eq">A_sim(t, λ_m) = (ε_A(λ_m)·[A](t) + ε_B(λ_m)·[B](t)) · l</pre>

<p>This is compared to the measured absorbance trace A_exp(t, λ<sub>m</sub>) to define
the least-squares residuals.</p>

<div class="note">
<b>Note on ε_B at monitoring wavelengths:</b>
Even in the <code>A_only</code> case, B may absorb at the monitoring wavelength.
Setting ε_B(λ_mon) = 0 when it is non-zero introduces a systematic error in Φ_AB
proportional to ε_B(λ_mon) / ε_A(λ_mon).
Use a wavelength where only A absorbs, or provide ε_B values explicitly.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["LED Full-Spectrum Integration"] = f"""
<h1>LED Full-Spectrum Integration</h1>

<p>
When the irradiation source is an LED (broadband), the single-wavelength ODE
can be replaced by a spectral integration over the full LED emission band.
The modified rate equation is:
</p>

<pre class="eq">d[A]/dt = ∫ (N(λ)/V) · l · f(A_tot(λ)) · (Φ_BA·[B]·ε_B(λ) − Φ_AB·[A]·ε_A(λ)) dλ
           + k_th·[B]</pre>

<p>
where <code>N(λ)</code> is the spectral photon flux density (mol s⁻¹ nm⁻¹) derived
from the LED emission spectrum and the measured optical power.
</p>

<h2>Deriving N(λ) from the LED Spectrum</h2>

<ol>
  <li>Load the LED emission spectrum I(λ) (normalised intensity vs. wavelength).</li>
  <li>Optionally apply Savitzky–Golay smoothing to reduce noise.</li>
  <li>Discard spectral tails below a fraction of the peak intensity (threshold).</li>
  <li>Measure the total optical power P (µW) with a power meter.</li>
  <li>Convert each spectral element to a photon flux density:</li>
</ol>

<pre class="eq">N(λ) = P · I(λ) / ∫ I(λ) dλ · λ / (h · c · N_A)    [mol s⁻¹ nm⁻¹]</pre>

<p>
where h is Planck's constant, c is the speed of light, and N_A is Avogadro's number.
</p>

<h2>Integration Modes</h2>
<table>
  <tr><th>Mode</th><th>Description</th><th>When to use</th></tr>
  <tr>
    <td><code>scalar</code></td>
    <td>Flux-weighted effective wavelength λ_eff and total N; reduces to the
        monochromator ODE at that single wavelength. Fast but approximate.</td>
    <td>Quick estimates; when ε(λ) variation across the LED band is small.</td>
  </tr>
  <tr>
    <td><code>full</code></td>
    <td>Spectrally resolved integration at every wavelength in the LED band.
        Requires ε_A(λ) and ε_B(λ) as full spectra (from EC or spectra results).</td>
    <td>High accuracy; when ε varies significantly across the LED emission band.</td>
  </tr>
</table>

<div class="note">
<b>Before / After power measurements:</b>
Record OPM power before and after the experiment.  A large drift (&gt;5 %) may
indicate lamp instability; consider averaging or using the more conservative value.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["PSS Algebraic Method"] = f"""
<h1>PSS Algebraic Method (<code>A_thermal_PSS</code>)</h1>

<p>
When species B undergoes fast thermal relaxation back to A (first-order rate k_th),
the system reaches a photostationary state (PSS) under continuous irradiation.
At PSS, the net rate of change of [B] is zero:
</p>

<pre class="eq">d[B]/dt = 0  ⟹  Φ_AB · N · (1 − 10^(−A_PSS)) / V = k_th · [B]_PSS · V</pre>

<p>
Rearranging to solve for Φ_AB:
</p>

<pre class="eq">Φ_AB = k_th · [B]_PSS · V / [ N · (1 − 10^(−A_PSS)) ]</pre>

<h2>Required Inputs</h2>
<table>
  <tr><th>Parameter</th><th>How to obtain</th></tr>
  <tr><td><code>k_th</code></td>
      <td>Thermal rate constant from half-life measurements (Half-Life tab),
          Eyring or Arrhenius analysis, or entered manually.</td></tr>
  <tr><td><code>[B]_PSS</code></td>
      <td>Concentration of B at PSS.  Derived from the PSS absorbance and
          ε_B, or from the fraction of B at PSS (via a reference wavelength
          where only A absorbs).</td></tr>
  <tr><td><code>A_PSS</code></td>
      <td>Total absorbance at the irradiation wavelength at PSS.
          Read directly from the kinetic trace at the plateau.</td></tr>
  <tr><td><code>N</code></td>
      <td>Incident photon flux (mol s⁻¹) — same sources as the ODE method.</td></tr>
  <tr><td><code>V</code></td>
      <td>Irradiated sample volume (L).</td></tr>
</table>

<h2>PSS Source Options</h2>
<table>
  <tr><th>Option</th><th>Description</th></tr>
  <tr><td><code>manual_fraction</code></td>
      <td>Enter the fraction f_B of total material in the B form at PSS directly.
          Then [B]_PSS = f_B · [A]_0 (assuming conservation of mass).</td></tr>
  <tr><td><code>manual_absorbance</code></td>
      <td>Enter A(λ_irr) at PSS directly from the kinetic trace plateau.</td></tr>
</table>

<div class="warn">
<b>Applicability:</b> The PSS method is valid only when the thermal back-reaction is fast
enough for the system to reach a true steady state under the irradiation conditions used.
If the kinetics are slow relative to the experiment duration, use the ODE method instead.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Photon Flux (N)"] = f"""
<h1>Photon Flux (N)</h1>

<p>
The incident photon flux N (mol s⁻¹) is the number of photons entering the cuvette
per unit time, expressed in molar units.  It is the key calibration quantity for
converting the observed kinetic rate into an absolute quantum yield.
</p>

<h2>Source Options</h2>

<table>
  <tr><th>Source</th><th>Input required</th><th>Notes</th></tr>
  <tr>
    <td><code>manual_mol_s</code></td>
    <td>N directly in mol s⁻¹</td>
    <td>Use when N was calculated externally.</td>
  </tr>
  <tr>
    <td><code>manual_uW</code></td>
    <td>Optical power P in µW and irradiation wavelength λ_irr</td>
    <td>Converted via: N = P·λ / (h·c·N_A).
        Valid for monochromatic (monochromator) sources only.</td>
  </tr>
  <tr>
    <td><code>actinometry</code></td>
    <td>Path to <code>photon_flux_master.csv</code> (output of Actinometer tab);
        optionally filter by irradiation wavelength</td>
    <td>Recommended when chemical actinometry was performed.
        The last matching row is used.</td>
  </tr>
  <tr>
    <td><code>led_spectrum</code></td>
    <td>LED emission CSV (from Actinometer tab) containing the normalised
        emission spectrum and total power</td>
    <td>Enables full spectral integration mode.</td>
  </tr>
</table>

<h2>Unit Conversion</h2>

<pre class="eq">N [mol s⁻¹] = P [µW] × 10⁻⁶ × λ [nm] × 10⁻⁹ / (h · c · N_A)

h  = 6.626 070 × 10⁻³⁴ J s
c  = 2.997 924 × 10⁸  m s⁻¹
N_A = 6.022 141 × 10²³ mol⁻¹</pre>

<h2>Uncertainty Propagation</h2>
<p>
If a 1σ standard deviation σ_N is provided for N, the tool performs a perturbation
analysis: the ODE is solved twice more at N ± σ_N, and the resulting spread in Φ_AB
is reported as <code>sigma_I0</code>.  This is combined in quadrature with the
fit standard error to give the total uncertainty σ_total.
</p>

<pre class="eq">σ_total = sqrt( σ_fit² + σ_I0² )</pre>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Extinction Coefficients"] = f"""
<h1>Extinction Coefficients (ε)</h1>

<p>
The molar absorption coefficient ε (L mol⁻¹ cm⁻¹) enters the ODE model at two points:
</p>
<ul>
  <li><b>Irradiation wavelength λ_irr</b> — controls how much light is absorbed by each species
      and therefore drives the photochemical rate.</li>
  <li><b>Monitoring wavelength(s) λ_mon</b> — converts simulated concentrations back into
      absorbance for comparison with the measured trace.</li>
</ul>

<h2>Source Options</h2>

<table>
  <tr><th>Source</th><th>Description</th></tr>
  <tr>
    <td><code>manual</code></td>
    <td>Enter ε values numerically.  For monitoring wavelengths a dict
        <code>{{λ_nm: ε}}</code> can be provided; if omitted, the irradiation-wavelength
        ε is reused for all monitoring wavelengths.</td>
  </tr>
  <tr>
    <td><code>ec_results</code></td>
    <td>Auto-loads the most recent CSV from the Ext. Coeff. results folder.
        A specific filename can be pinned.  The <code>Mean</code> column is used
        by default; other columns (e.g. <code>Prep1_Mean</code>) can be selected.</td>
  </tr>
  <tr>
    <td><code>ec_csv</code></td>
    <td>Full path to any CSV with <code>Wavelength (nm)</code> or
        <code>Wavelength_nm</code> and an ε column.  Values are interpolated
        to the exact wavelengths needed.</td>
  </tr>
  <tr>
    <td><code>spectra_results</code></td>
    <td>Loads from the Spectra Calculation results folder.  Column names
        <code>Species_A</code> / <code>Species_B</code> select the species;
        values are interpolated.</td>
  </tr>
</table>

<div class="note">
<b>Interpolation:</b> All CSV-based sources interpolate linearly to the requested
wavelengths.  Ensure the CSV covers the full wavelength range needed (both irradiation
and all monitoring wavelengths).
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Offset Correction"] = f"""
<h1>Offset Correction</h1>

<p>
Many spectrophotometers zero the detector automatically before a kinetic run
("autozero"), so the recorded absorbance starts at 0 even though the sample has a
real non-zero absorbance.  An <b>offset correction</b> compensates for this and
restores the absolute absorbance scale — which is required to compute the correct
initial concentration [A]₀.
</p>

<div class="warn">
<b>Important:</b> The offset correction must be applied consistently.
If <code>none</code> is selected, the raw absorbance trace is used directly to
derive [A]₀.  This is only valid when the instrument did <em>not</em> autozero
(i.e. the trace already reflects absolute AU).  Any autozero or offset shift in
the data makes the derived [A]₀ — and therefore Φ — wrong if no correction is applied.
</div>

<table>
  <tr><th>Method</th><th>When to use</th><th>What it does</th></tr>
  <tr>
    <td><code>none</code></td>
    <td>Data is already in absolute AU (no autozero performed)</td>
    <td>No correction applied.</td>
  </tr>
  <tr>
    <td><code>subtract_first_point</code></td>
    <td>Instrument autozero'd at t = 0; no pre-irradiation dark period</td>
    <td>Subtracts the absorbance at the first loaded time point from the entire
        trace.  The trace is shifted to start at zero; [A]₀ must then be obtained
        from an independent source (e.g. manual or <code>align_to_spectrum</code>).</td>
  </tr>
  <tr>
    <td><code>subtract_plateau</code></td>
    <td>Instrument autozero'd; a pre-irradiation plateau is recorded</td>
    <td>Subtracts the mean absorbance over a user-defined plateau window
        [t_start, t_end].  More robust than <code>subtract_first_point</code>
        against noise at t = 0.</td>
  </tr>
  <tr>
    <td><code>align_to_spectrum</code></td>
    <td>Kinetic trace is zeroed but an initial full-spectrum scan is available</td>
    <td><b>Recommended.</b>  Loads an initial spectrum file (Cary 60 CSV),
        extracts A(λ_mon) from it, and adds the difference
        (A_initial_spec − A_kinetic_t0) to every point of the kinetic trace.
        This rigidly shifts the kinetic trace onto the absolute absorbance scale
        established by the spectrum.  [A]₀ is then read from that corrected trace,
        giving the physically correct initial concentration.</td>
  </tr>
</table>

<div class="note">
<b>Plateau window parameters:</b> For <code>subtract_plateau</code>, the window is
defined by <code>baseline_plateau_start_s</code> and <code>baseline_plateau_end_s</code>.
For <code>align_to_spectrum</code>, <code>offset_plateau_duration_s</code> controls how
many seconds at the start of the kinetic trace are averaged to represent the pre-irradiation
level before the shift is applied.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Fit Window & Irradiation Start"] = f"""
<h1>Fit Window and Irradiation Start Detection</h1>

<p>
The ODE fit is performed only over the irradiation period.  Points before the
light is turned on (baseline region) and — optionally — points after a chosen
end time are excluded.
</p>

<h2>Manual Window</h2>
<p>
Set <code>fit_time_start_s</code> and <code>fit_time_end_s</code> explicitly.
The ODE time axis is reset so that t = 0 corresponds to the first included point.
</p>

<h2>Auto-Detection of Irradiation Start</h2>
<p>
When <code>auto_detect_irr_start = True</code>, the algorithm:
</p>
<ol>
  <li>Identifies the monitoring channel with the largest total change (reference channel).</li>
  <li>Computes the mean and standard deviation of the first <code>n_plateau</code> points.</li>
  <li>Scans forward and flags the first index where <code>|A(t) − mean| &gt; threshold × σ</code>
      for at least <code>min_consec</code> consecutive points.</li>
  <li>Uses that index as <code>fit_time_start_s</code>.</li>
</ol>

<table>
  <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
  <tr><td><code>n_plateau</code></td><td>20</td>
      <td>Number of initial points used to estimate plateau statistics.</td></tr>
  <tr><td><code>threshold</code></td><td>5.0</td>
      <td>Detection sensitivity in units of the plateau σ.  Lower values detect
          earlier (more sensitive); higher values require a larger change.</td></tr>
  <tr><td><code>min_consec</code></td><td>3</td>
      <td>Minimum consecutive out-of-plateau points to confirm irradiation start
          (reduces false positives from noise spikes).</td></tr>
</table>

<div class="note">
If auto-detection fails (e.g. very slow kinetics or noisy baseline), the manually
set <code>fit_time_start_s</code> is used as a fallback.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Initial Slopes Estimate"] = f"""
<h1>Initial Slopes Estimate</h1>

<p>
As a diagnostic, the tool also estimates Φ_AB from the initial slope of the
absorbance trace — the linear regime at the very start of irradiation where
[B] ≈ 0 and the back-reaction can be neglected.
</p>

<h2>Derivation</h2>

<p>
At t ≈ 0, [B] ≈ 0, so the ODE simplifies to:
</p>

<pre class="eq">d[A]/dt ≈ −Φ_AB · (N_abs / V)</pre>

<p>
where N_abs is the number of photons actually absorbed per second by A:
</p>

<pre class="eq">N_abs = N · (1 − 10^(−ε_A · [A]_0 · l))</pre>

<p>
The observed absorbance change at the monitoring wavelength follows:
</p>

<pre class="eq">dA_mon/dt ≈ (ε_A(λ_mon) − ε_B(λ_mon)) · l · d[A]/dt</pre>

<p>
Combining and rearranging:
</p>

<pre class="eq">Φ_AB = −(dA_mon/dt) · V / [ (ε_A(λ_mon) − ε_B(λ_mon)) · l · N_abs ]</pre>

<p>
The slope dA_mon/dt is estimated by a linear regression over the first
<code>n_initial_slopes_pts</code> data points of the fit window.
</p>

<div class="note">
<b>Use as a cross-check:</b> The initial slopes estimate should agree with the
full ODE fit within uncertainty.  Large disagreements may indicate baseline
offsets, incorrect ε values, or a non-negligible [B]_0.
</div>

<div class="note">
<b>LED mode:</b> When using LED full-integration, N_abs is computed by integrating
the spectrally resolved absorbed flux:
N_abs = ∫ N(λ) · (1 − 10^(−ε_A(λ) · [A]_0 · l)) dλ
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Initial Conditions"] = f"""
<h1>Initial Conditions</h1>

<p>
The ODE requires [A]₀ and [B]₀ at the start of the fit window (i.e. at the
moment the light is turned on).
</p>

<h2>Initial Concentration of A</h2>

<table>
  <tr><th>Source</th><th>Description</th></tr>
  <tr>
    <td><code>absorbance</code></td>
    <td>Derived from the first data point of the fit window at the first monitoring
        wavelength:<br>
        <code>[A]_0 = A_obs(t_start, λ_mon1) / (ε_A(λ_mon1) · l)</code><br>
        This is the default and requires ε_A at the monitoring wavelength.</td>
  </tr>
  <tr>
    <td><code>manual</code></td>
    <td>Enter [A]₀ in mol L⁻¹ directly.  Use this when the absorbance-based
        estimate is unreliable (e.g. very low absorbance, overlapping bands).</td>
  </tr>
</table>

<h2>Initial Concentration of B</h2>
<p>
[B]₀ is almost always zero (pure A at the start of irradiation).  Set a non-zero
value only if some B is already present at the beginning of the kinetic trace
(e.g. partial conversion before data acquisition started).
</p>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Thermal Back-Reaction (k_th)"] = f"""
<h1>Thermal Back-Reaction Rate k_th</h1>

<p>
The thermal rate constant k_th (s⁻¹) describes the first-order spontaneous conversion
B→A in the dark.  It appears as an additional decay term in the ODE:
</p>

<pre class="eq">d[A]/dt = (photochemical terms) + k_th · [B]
d[B]/dt = (photochemical terms) − k_th · [B]</pre>

<h2>Source Options</h2>

<table>
  <tr><th>Source</th><th>Description</th></tr>
  <tr>
    <td><code>none</code></td>
    <td>k_th = 0 (no thermal back-reaction).  Use for stable photoproducts.</td>
  </tr>
  <tr>
    <td><code>manual</code></td>
    <td>Enter k_th (and optionally its 1σ uncertainty) in s⁻¹ directly.</td>
  </tr>
  <tr>
    <td><code>half_life_master</code></td>
    <td>Loads k_th from <code>half_life_master.csv</code> (output of the Half-Life tab),
        filtered by temperature.  The matching row is used.</td>
  </tr>
  <tr>
    <td><code>eyring</code></td>
    <td>Loads Eyring fit parameters and computes k_th at the specified temperature
        using the Eyring equation.</td>
  </tr>
  <tr>
    <td><code>arrhenius</code></td>
    <td>Loads Arrhenius fit parameters and computes k_th at the specified temperature
        using the Arrhenius equation.</td>
  </tr>
</table>

<div class="note">
<b>Temperature:</b> When using half_life_master, Eyring, or Arrhenius sources,
the temperature used to look up or compute k_th is set by <code>k_th_temperature_C</code>,
which should match the sample temperature during the QY experiment.
</div>

<div class="warn">
<b>Impact on Φ:</b> An incorrect k_th shifts the apparent PSS position and thus
introduces a systematic error in Φ_AB.  For the PSS algebraic method the error is
directly proportional to k_th.  Always verify k_th at the experimental temperature.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Fitting & Uncertainty"] = f"""
<h1>Fitting and Uncertainty Estimation</h1>

<h2>Least-Squares Minimisation</h2>
<p>
The quantum yield(s) are determined by minimising the sum of squared residuals
between the simulated and measured absorbance traces, using the Levenberg–Marquardt
algorithm (via <code>lmfit</code>):
</p>

<pre class="eq">minimise  Σ_t Σ_λ [ A_sim(t, λ) − A_exp(t, λ) ]²
over      Φ_AB  (and Φ_BA for AB_both)</pre>

<p>
By default, quantum yields are bounded to [10⁻⁶, 1].  The upper bound of 1
can be removed (<code>QY_unconstrained = True</code>) to:
</p>
<ul>
  <li>Diagnose systematic errors (if the unconstrained fit yields Φ &lt; 0 or Φ &gt; 1,
      the data or parameters are likely wrong).</li>
  <li>Detect photon-driven cascades where Φ &gt; 1 is physically possible (e.g.
      electron-transfer-assisted switching).</li>
</ul>

<h2>Per-Wavelength Fitting</h2>
<p>
When multiple monitoring wavelengths are present, the fit is run independently
per wavelength, yielding a Φ_AB value for each.  The reported aggregate value
is the mean, weighted by the inverse variance (1 / σ_fit²):
</p>

<pre class="eq">Φ_AB = Σ (Φ_AB,i / σ_fit,i²) / Σ (1 / σ_fit,i²)</pre>

<h2>Uncertainty Components</h2>

<table>
  <tr><th>Component</th><th>Symbol</th><th>Source</th></tr>
  <tr>
    <td>Fit standard error</td>
    <td><code>sigma_fit</code></td>
    <td>Square root of the covariance diagonal from the least-squares fit.
        Reflects the precision of the curve match.</td>
  </tr>
  <tr>
    <td>Photon flux uncertainty</td>
    <td><code>sigma_I0</code></td>
    <td>Perturbation analysis: ODE solved at N ± σ_N; half the resulting
        Φ range.  Only computed when σ_N &gt; 0.</td>
  </tr>
  <tr>
    <td>Total uncertainty</td>
    <td><code>sigma_total</code></td>
    <td>Quadrature sum: √(σ_fit² + σ_I0²)</td>
  </tr>
</table>

<h2>Goodness of Fit</h2>
<p>
The coefficient of determination R² is computed per wavelength and aggregated:
</p>

<pre class="eq">R² = 1 − SS_res / SS_tot

SS_res = Σ_t (A_exp − A_sim)²
SS_tot = Σ_t (A_exp − mean(A_exp))²</pre>

<p>
R² values close to 1 indicate a good fit.  Low R² or structured residuals
suggest issues with the model assumptions, baseline, ε values, or photon flux.
</p>

<h2>Reference Curve</h2>
<p>
An optional reference quantum yield <code>QY_AB_reference</code> can be specified.
The ODE is then solved with that fixed Φ and the resulting curve overlaid as a
dashed line on the plot — useful for comparison with literature values or
verification experiments.
</p>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS_LED["LED Actinometry — How It Works"] = f"""
<h1>LED Actinometry — How It Works</h1>

<p>
LED actinometry measures the photon flux N (mol s⁻¹) of an LED source by
irradiating a chemical actinometer solution (e.g. potassium ferrioxalate) and
fitting the rate of photoproduct formation.  The result (N_chem) is compared to
the independently measured flux N_LED obtained from the LED emission spectrum
and optical power.
</p>

<h2>Step 1 — Beer–Lambert Reference Spectrum</h2>
<p>
A full absorption spectrum of the actinometer solution is loaded.  For scanning
kinetic data the mid-scan of the first group is used; for fixed-wavelength kinetic
data an independently measured initial spectrum must be supplied.
</p>
<p>
The absorbance at 562 nm, A(562), is extracted.  This serves as the anchor for
scaling ε to any other wavelength via Beer–Lambert:
</p>
<pre class="eq">ε(λ) = ε_562 · A(λ) / A(562)</pre>

<h2>Step 2 — Effective ε and QY (ε_eff, QY_eff)</h2>
<p>
Two integration modes are available:
</p>

<table>
  <tr><th>Mode</th><th>ε_eff definition</th><th>QY_eff definition</th></tr>
  <tr>
    <td><code>scalar</code></td>
    <td>ε at the flux-weighted effective wavelength λ_eff:<br>
        <code>ε_eff = ε_562 · A(λ_eff) / A(562)</code></td>
    <td><code>QY(λ_eff)</code> from the actinometer's wavelength-dependent
        quantum yield function</td>
  </tr>
  <tr>
    <td><code>spectral</code></td>
    <td>Flux-weighted spectral average over the full LED band:<br>
        <code>ε_eff = ∫ f(λ) · ε(λ) dλ / ∫ f(λ) dλ</code><br>
        where <code>f(λ) = N(λ) / N_total</code> is the normalised LED shape
        and <code>ε(λ) = ε_562 · A(λ)/A(562)</code></td>
    <td>Flux-weighted average QY:<br>
        <code>QY_eff = ∫ f(λ) · QY(λ) dλ / ∫ f(λ) dλ</code></td>
  </tr>
</table>

<div class="note">
<b>Physical meaning of ε_eff:</b>
In <em>scalar</em> mode, ε_eff is simply the molar absorption coefficient of the
actinometer at the representative wavelength λ_eff.  In <em>spectral</em> mode it
is the photon-flux-weighted mean of ε(λ) across the LED band — the effective
single-wavelength ε that would produce the same absorbed flux if the LED were
monochromatic.  Both definitions rely on the Beer–Lambert ratio
A(λ)/A(562) to extrapolate from the known reference ε_562.
</div>

<h2>Step 3 — Rate Function and Linear Fit</h2>
<p>
The actinometer ODE (one absorbing species, no back-reaction) can be integrated
analytically.  Define:
</p>
<pre class="eq">y(t) = −V / (QY_eff · ε_eff · l) · [log₁₀(10^A(t) − 1) − log₁₀(10^A₀ − 1)]</pre>
<p>
where A(t) is the measured absorbance at λ_eff (or the kinetic channel closest to it).
The derivation shows that y(t) is exactly linear in time:
</p>
<pre class="eq">y(t) = N_chem · t + intercept</pre>
<p>
The slope of a linear fit to the (t, y) data gives N_chem in mol s⁻¹ directly.
</p>

<div class="note">
<b>Why is y linear?</b>
The integral ∫ dA / (1 − 10^(−A)) = log₁₀(10^A − 1) + const, so the transformed
variable y absorbs the nonlinear Beer–Lambert correction factor exactly.  The result
is valid for <em>any</em> absorbance level, not just the optically thin limit.
</div>

<h2>Step 4 — N_chem vs N_LED Comparison</h2>
<p>
The result table shows N_chem (from the actinometer fit), N_LED (from the LED
characterisation panel), and the percentage deviation:
</p>
<pre class="eq">deviation (%) = (N_chem − N_LED) / N_LED × 100</pre>
<p>
A deviation of ±5 % is typical for careful measurements.  Larger deviations may
indicate:
</p>
<ul>
  <li>LED warm-up drift (measure power immediately before and after the experiment)</li>
  <li>Beam geometry mismatch between power-meter probe and cuvette cross-section</li>
  <li>Actinometer absorbance outside the recommended range for the chosen actinometer</li>
  <li>Inner-filter or photobleaching effects in the actinometer</li>
</ul>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS_LED["LED QY — Calculation Pipeline"] = f"""
<h1>LED Quantum Yield — Calculation Pipeline</h1>

<p>
When the photon source is an LED, the QY calculation can be run in two modes:
<em>scalar</em> (single effective wavelength) or <em>full spectral integration</em>.
This section describes both, explains the units throughout, and lists the most
common sources of systematic error.
</p>

<h2>1. Photon Flux Array N(λ)</h2>
<p>
The LED spectrum CSV (saved by the LED Characterisation panel) contains two columns:
</p>
<pre>wavelength_nm   N_mol_s_per_nm
   …                   …</pre>
<p>
<code>N_mol_s_per_nm</code> is the <b>spectral photon flux density</b> in mol s⁻¹ nm⁻¹.
It is computed from the measured LED emission shape I(λ) and total optical power P:
</p>
<pre class="eq">N(λ)  =  P · [I(λ) / ∫I(λ) dλ] · λ / (h · c · N_A)    [mol s⁻¹ nm⁻¹]

N_total  =  ∫ N(λ) dλ    [mol s⁻¹]</pre>
<p>
The first factor <code>I(λ)/∫I dλ</code> is the normalised emission shape (nm⁻¹).
The factor <code>λ/(h·c·N_A)</code> converts energy flux at wavelength λ into photon flux.
</p>

<div class="note">
<b>Assumption:</b> The emission spectrum I(λ) is proportional to the spectral power
density P(λ) (i.e. the spectrometer response is flat across the LED band).  For a
narrowband LED (&lt;30 nm FWHM) this approximation introduces &lt;1% error.
</div>

<h2>2. Scalar Mode</h2>
<p>
In scalar mode the LED is treated as monochromatic at λ_eff:
</p>
<pre class="eq">λ_eff = ∫ λ · N(λ) dλ / N_total    (flux-weighted mean wavelength)</pre>
<p>
The standard single-wavelength ODE is used with N = N_total and
ε_A = ε_A(λ_eff), ε_B = ε_B(λ_eff).  This is an approximation: it
is accurate only when ε(λ) is nearly constant across the LED emission band.
</p>

<h2>3. Full Spectral Integration Mode</h2>
<p>
In full integration mode the ODE integrates photon absorption at each wavelength
simultaneously:
</p>
<pre class="eq">d[A]/dt = ∫ (N(λ)/V) · l · f(A_tot(λ)) · (Φ_BA·[B]·ε_B(λ) − Φ_AB·[A]·ε_A(λ)) dλ
           + k_th · [B]

A_tot(λ) = (ε_A(λ)·[A] + ε_B(λ)·[B]) · l

f(A_tot) = (1 − 10^(−A_tot)) / A_tot    [absorption correction factor]</pre>
<p>
The integral is evaluated by <code>numpy.trapezoid</code> at each ODE time step.
ε_A(λ) and ε_B(λ) must be provided as full spectra covering the LED emission band
(loaded from the Extinction Coefficient or Spectra results).
</p>

<h2>4. Units Verification</h2>
<table>
  <tr><th>Quantity</th><th>Symbol</th><th>Unit</th></tr>
  <tr><td>Spectral photon flux density</td><td>N(λ)</td><td>mol s⁻¹ nm⁻¹</td></tr>
  <tr><td>Sample volume</td><td>V</td><td>L</td></tr>
  <tr><td>Path length</td><td>l</td><td>cm</td></tr>
  <tr><td>Molar absorption coeff.</td><td>ε</td><td>L mol⁻¹ cm⁻¹</td></tr>
  <tr><td>Concentration</td><td>[A], [B]</td><td>mol L⁻¹</td></tr>
  <tr><td>Rate (integrand per nm)</td><td>rate(λ) dλ</td><td>mol L⁻¹ s⁻¹ nm⁻¹ × nm</td></tr>
  <tr><td>d[A]/dt</td><td></td><td>mol L⁻¹ s⁻¹ ✓</td></tr>
</table>
<p>
Dimensional check of the integrand:
(mol s⁻¹ nm⁻¹ / L) × cm × (L mol⁻¹ cm⁻¹) × (mol L⁻¹) = mol s⁻¹ nm⁻¹ L⁻¹ → after ∫dλ: mol s⁻¹ L⁻¹ ✓
</p>

<h2>5. Initial Concentration [A]₀</h2>
<p>
[A]₀ is derived from the first absorbance data point at the first monitoring
wavelength:
</p>
<pre class="eq">[A]₀ = A_obs(t₀, λ_mon1) / (ε_A(λ_mon1) · l)</pre>
<p>
Note that ε here is at the <em>monitoring</em> wavelength, not the irradiation
wavelength.  This is correct: the monitored absorbance is converted to concentration
using the monitoring-wavelength ε.
</p>

<div class="warn">
<b>Critical: offset correction and [A]₀</b><br>
If the kinetic trace was zeroed by the spectrophotometer (autozero), A_obs(t₀) = 0
and the formula above gives [A]₀ = 0 — which is completely wrong.
Use <code>align_to_spectrum</code> offset correction to shift the kinetic trace to
the absolute absorbance scale before [A]₀ is computed.
</div>

<h2>6. ODE Fit and QY Extraction</h2>
<p>
The ODE is integrated by <code>scipy.integrate.odeint</code>.  The simulated
absorbance at each monitoring wavelength is:
</p>
<pre class="eq">A_sim(t, λ_mon) = (ε_A(λ_mon)·[A](t) + ε_B(λ_mon)·[B](t)) · l</pre>
<p>
<code>lmfit</code> (Levenberg–Marquardt) minimises the sum of squared residuals
between A_sim and A_exp over Φ_AB (and Φ_BA in the AB_both case).
</p>

<h2>7. Troubleshooting — Systematic Errors</h2>

<table>
  <tr><th>Symptom</th><th>Most likely cause</th><th>How to check</th></tr>
  <tr>
    <td>Φ 2–5× too high</td>
    <td><b>Sample volume V entered incorrectly.</b>
        If V is entered as 4× the true irradiated volume, the computed rate per
        unit volume is 4× too small, so the fitted Φ will be 4× too large to
        compensate.  The ODE rate is proportional to N/V; V cancels with [A]₀
        only partially, so errors in V propagate directly into Φ.</td>
    <td>Confirm the entered volume matches the actual irradiated cuvette volume,
        not the total cell volume.  For a standard 1 cm cuvette with 3 mL total
        but a 1 mL irradiated zone, use 1 mL.</td>
  </tr>
  <tr>
    <td>N_chem 10–20% from N_LED</td>
    <td>Power meter calibration, beam geometry, or LED drift.
        This is normal; the power meter absolute accuracy is typically ±5–15%.</td>
    <td>Use N_chem from actinometry (not N_LED) as input to the QY calculation
        for a self-consistent result.</td>
  </tr>
  <tr>
    <td>Φ depends strongly on monitoring wavelength</td>
    <td>Incorrect ε_B at monitoring wavelength; or the data is not well described
        by the two-species model (photoproduct accumulation, secondary reactions).</td>
    <td>Check ε_B at each λ_mon.  Run with <code>A_only</code> case and compare.</td>
  </tr>
  <tr>
    <td>[A]₀ = 0 or near zero</td>
    <td>Kinetic trace was zeroed by the spectrophotometer (autozero) but no
        offset correction was applied.</td>
    <td>Use <code>align_to_spectrum</code> correction with an initial spectrum file.</td>
  </tr>
  <tr>
    <td>Φ &gt; 1 (unconstrained fit)</td>
    <td>Systematic overestimation of Φ.  See Volume above.  Also check: Is N
        in mol s⁻¹ (not mol s⁻¹ nm⁻¹ total)?  Is ε in L mol⁻¹ cm⁻¹?</td>
    <td>Enable unconstrained fit to see the true optimum; diagnose the parameter
        causing the issue.</td>
  </tr>
</table>

<div class="note">
<b>Recommendation for LED experiments:</b>
Use the LED full-integration mode with N(λ) from the LED Characterisation panel
and ε(λ) full spectra from the Extinction Coefficient tab.  Set the photon flux
source to <code>actinometry</code> (using N_chem) rather than <code>led_spectrum</code>
(using N_LED) to absorb any power-meter calibration offset into the actinometry step,
where it can be validated against literature QY values for the actinometer.
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
_SECTIONS["Data Types & File Formats"] = f"""
<h1>Data Types and File Formats</h1>

<h2>Kinetic Data (<code>data_type = "kinetic"</code>)</h2>
<p>
Time-series CSV files produced by the spectrophotometer in multi-wavelength
kinetic mode (same format as the Half-Life workflow).
</p>

<p><b>Format:</b></p>
<pre>Row 0 :  channel labels     e.g.  25C_672nm  ,  (blank)  ,  25C_530nm  ,  (blank)
Row 1 :  column headers     Time (sec) , Abs , Time (sec) , Abs , …
Rows 2+: data               12.3 , 0.812 , 12.3 , 0.341 , …</pre>

<p>
Each pair of columns represents one monitoring wavelength.  The label in Row 0
is used to identify the wavelength; a numeric suffix (e.g. <code>672nm</code>)
is parsed automatically.  Channels not matching a requested monitoring wavelength
are ignored.
</p>

<h2>Scanning Data (<code>data_type = "scanning"</code>)</h2>
<p>
Full-spectrum CSV files from the Cary 60 spectrophotometer in time-series scan mode.
Each pair of columns is one spectrum (wavelength, absorbance).
</p>

<p><b>Key parameters:</b></p>
<table>
  <tr><th>Parameter</th><th>Description</th></tr>
  <tr><td><code>delta_t_s</code></td>
      <td>Time between consecutive spectra (or spectrum groups) in seconds.</td></tr>
  <tr><td><code>scans_per_group</code></td>
      <td>Number of scans averaged per time point.  Use &gt;1 when the
          spectrophotometer averages multiple scans per step.</td></tr>
  <tr><td><code>first_cycle_off</code></td>
      <td>Set True if the first spectrum was recorded with the light off
          (dark reference), so it is used as t = 0 rather than as an
          irradiation data point.</td></tr>
  <tr><td><code>wavelength_tolerance_nm</code></td>
      <td>±nm window used to extract absorbance at each monitoring wavelength
          from the spectrum.  The mean over the window is taken.</td></tr>
</table>

<h2>LED Emission and Power Files</h2>
<table>
  <tr><th>File</th><th>Location</th><th>Format</th></tr>
  <tr>
    <td>Emission spectrum</td>
    <td><code>data/led/emission/</code></td>
    <td>CSV with columns <code>Wavelength_nm</code> and <code>Intensity</code>
        (arbitrary normalised units).</td>
  </tr>
  <tr>
    <td>Power time-series</td>
    <td><code>data/led/power/</code></td>
    <td>CSV with columns <code>Time_s</code> and <code>Power_uW</code>.
        The mean power is used.</td>
  </tr>
</table>
"""


# ── Tab widget ─────────────────────────────────────────────────────────────────

class HandbookTab(QWidget):
    """
    Handbook tab with a sidebar navigation list and an HTML content browser.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        # Select first section
        self._nav.setCurrentRow(0)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Navigation list ────────────────────────────────────────────────
        self._nav = QListWidget()
        self._nav.setObjectName("handbook_nav")
        self._nav.setFixedWidth(220)
        font = QFont("Segoe UI", 9)
        self._nav.setFont(font)

        heading_font = QFont("Segoe UI", 8)
        heading_font.setBold(True)

        def _add_group(label: str, sections: dict):
            h = QListWidgetItem(f"  {label}")
            h.setFlags(Qt.ItemFlag.NoItemFlags)
            h.setFont(heading_font)
            self._nav.addItem(h)
            for title in sections:
                item = QListWidgetItem(f"   {title}")
                item.setData(Qt.ItemDataRole.UserRole, title)
                self._nav.addItem(item)

        _add_group("QUANTUM YIELD", _SECTIONS)
        _add_group("LED / ACTINOMETRY", _SECTIONS_LED)

        self._nav.currentItemChanged.connect(self._on_nav_changed)

        # ── Content browser ────────────────────────────────────────────────
        self._browser = QTextBrowser()
        self._browser.setObjectName("handbook_browser")
        self._browser.setOpenExternalLinks(False)
        self._browser.document().setDefaultStyleSheet(_CSS)

        splitter.addWidget(self._nav)
        splitter.addWidget(self._browser)
        splitter.setSizes([220, 900])

        root.addWidget(splitter)

        # ── Nav list extra styling ─────────────────────────────────────────
        self._nav.setStyleSheet(f"""
            QListWidget {{
                background: #181828;
                border: none;
                border-right: 1px solid {_BORDER};
                outline: none;
            }}
            QListWidget::item {{
                color: {_TEXT_DIM};
                padding: 5px 4px;
                border: none;
            }}
            QListWidget::item:selected {{
                background: #252545;
                color: {_ACCENT2};
                border-left: 3px solid {_ACCENT};
            }}
            QListWidget::item:hover:!selected {{
                background: #1e1e3a;
                color: {_TEXT};
            }}
            QListWidget::item:disabled {{
                color: #555577;
                font-size: 8pt;
                letter-spacing: 0.5px;
                padding-top: 10px;
            }}
        """)

        self._browser.setStyleSheet(f"""
            QTextBrowser {{
                background: {_BG};
                border: none;
                color: {_TEXT};
            }}
        """)

    def _on_nav_changed(self, current, _previous):
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        content = _SECTIONS.get(key) or _SECTIONS_LED.get(key)
        if content:
            html = f"<html><body>{content}</body></html>"
            self._browser.setHtml(html)
            self._browser.verticalScrollBar().setValue(0)
