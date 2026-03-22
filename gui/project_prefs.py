"""
Project-level preferences — saved as JSON inside the output folder.

A ``prefs.json`` file is written/read from the selected output folder so
that the same settings are restored automatically when a project is
re-opened.  Each group of preference variables corresponds to one
workflow panel.

Adding a new preference
-----------------------
1. Add a field to the relevant dataclass (give it a sensible default).
2. Call ``apply_prefs`` / ``collect_prefs`` in the matching GUI panel.
3. That's it — JSON round-trip is handled automatically via ``dataclasses.asdict``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

PREFS_FILENAME = "prefs.json"


# ── Per-panel preference groups ────────────────────────────────────────────

@dataclass
class ThermalPrefs:
    """Shared preference variables for Arrhenius and Eyring panels."""
    compound_name: str  = ""
    weighted_fit:  bool = True


@dataclass
class ActinometerPrefs:
    """Chemical-actinometry-panel preference variables."""
    actinometer_choice:        int   = 2
    irradiation_wavelength_nm: float = 579.0
    irradiation_time_s:        float = 60.0
    volume_mL:                 float = 2.0
    path_length_cm:            float = 1.0
    scans_per_group:           int   = 3
    wavelength_tolerance_nm:   float = 1.0


@dataclass
class LEDCharacterisationPrefs:
    """LED-characterisation-panel preference variables."""
    power_use:              str   = "before"   # "before" | "after" | "average"
    integration_mode:       str   = "full"     # "scalar" | "full"
    emission_threshold:     float = 0.005
    smoothing_enabled:      bool  = True
    smoothing_window:       int   = 11
    smoothing_order:        int   = 3
    photon_flux_std_manual: float = 0.0


@dataclass
class ExtinctionCoeffPrefs:
    """Extinction-coefficients-panel preference variables."""
    compound_name:  str   = ""
    path_length_cm: float = 1.0
    solvent:        str   = "acetonitrile"
    temperature_c:  float = 25.0


@dataclass
class HalfLifeKineticsPrefs:
    """Kinetics-panel preference variables."""
    switch:        str   = "negative"   # "negative" = positive photochromic (decay)
                                         # "positive" = negative photochromic (build-up)
    a_inf_mode:    str   = "free"        # "free" | "fixed" | "reference"
    a_inf_value:   float = 0.0
    iqr_factor:    float = 3.0
    temperature_c: float = 25.0


@dataclass
class SpectraPrefs:
    """Spectra-extraction-panel preference variables."""
    mode:                   str   = "negative"
    compound_name:          str   = ""
    path_length_cm:         float = 1.0
    reference_wavelength_nm: str  = ""     # stored as text so it supports bands/lists
    reference_weighted:     bool  = True
    min_alpha:              float = 0.2
    max_alpha:              float = 0.6
    exclude_negative_SB:    bool  = True
    sb_tolerance_sigma:     float = 3.0
    n_bootstrap:            int   = 2000
    pss_fraction_B:         float = 0.85
    pss_fraction_B_error:   float = 0.02


@dataclass
class HalfLifeScanningPrefs:
    """Scanning-kinetics-panel preference variables."""
    switch:        str   = "negative"
    a_inf_mode:    str   = "reference"
    a_inf_value:   float = 0.0
    iqr_factor:    float = 50.0
    temperature_c: float = 25.0


@dataclass
class QuantumYieldPrefs:
    """Quantum-yield-panel preference variables."""
    case:                      str   = "A_only"
    data_type:                 str   = "kinetic"
    compound_name:             str   = ""
    temperature_C:             float = 25.0
    solvent:                   str   = ""
    path_length_cm:            float = 1.0
    volume_mL:                 float = 2.0
    photon_flux_source:        str   = "manual_mol_s"
    photon_flux_mol_s:         float = 1.0e-9
    photon_flux_std_mol_s:     float = 0.0
    irradiation_wavelength_nm: float = 530.0
    delta_t_s:                 float = 12.0
    scans_per_group:           int   = 1
    k_th_source:               str   = "none"
    k_th_temperature_C:        float = 25.0
    epsilon_source_A:          str   = "manual"
    epsilon_A_irr:             float = 10000.0
    epsilon_source_B:          str   = "manual"
    epsilon_B_irr:             float = 0.0
    QY_AB_init:                float = 0.1
    QY_BA_init:                float = 0.05
    wavelength_tolerance_nm:   float = 2.0
    monitoring_wavelengths:    list  = field(default_factory=list)


# ── Project metadata ───────────────────────────────────────────────────────

@dataclass
class ProjectMetadata:
    """
    Free-text fields that travel with the project and can be used
    directly in publication captions / supplementary information.
    """
    compound:  str = ""
    solvent:   str = ""
    notes:     str = ""   # anything useful for the lab notebook / SI


# ── Top-level container ────────────────────────────────────────────────────

@dataclass
class ProjectPrefs:
    version:             int                    = 1
    metadata:            ProjectMetadata        = field(default_factory=ProjectMetadata)
    half_life_kinetics:  HalfLifeKineticsPrefs  = field(
        default_factory=HalfLifeKineticsPrefs)
    half_life_scanning:  HalfLifeScanningPrefs  = field(
        default_factory=HalfLifeScanningPrefs)
    thermal:               ThermalPrefs             = field(
        default_factory=ThermalPrefs)
    actinometer:           ActinometerPrefs        = field(
        default_factory=ActinometerPrefs)
    led_characterisation:  LEDCharacterisationPrefs = field(
        default_factory=LEDCharacterisationPrefs)
    extinction_coeff:      ExtinctionCoeffPrefs     = field(
        default_factory=ExtinctionCoeffPrefs)
    spectra:               SpectraPrefs             = field(
        default_factory=SpectraPrefs)
    quantum_yield:         QuantumYieldPrefs        = field(
        default_factory=QuantumYieldPrefs)

    # ── I/O ────────────────────────────────────────────────────────────

    def save(self, dest: Path) -> Path:
        """
        Write preferences to *dest*.

        *dest* must be the full file path (e.g. ``/project/prefs.json``).
        The parent directory must already exist.
        """
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return dest

    @classmethod
    def load_from_file(cls, path: Path) -> Optional["ProjectPrefs"]:
        """
        Load from an explicit file path.

        Returns ``None`` if the file doesn't exist or can't be parsed.
        """
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return cls._from_dict(data)
        except Exception as exc:
            print(f"[Preferences] Could not load {path}: {exc}")
            return None

    @classmethod
    def _from_dict(cls, data: dict) -> "ProjectPrefs":
        """Reconstruct a ProjectPrefs from a raw JSON dict."""
        def _get(d: dict, key: str, cls_):
            raw = d.get(key, {})
            # Only pass keys present in the dataclass to survive
            # forward/backward version differences.
            valid = {k: v for k, v in raw.items()
                     if k in cls_.__dataclass_fields__}
            return cls_(**valid)

        return cls(
            version               = data.get("version", 1),
            metadata              = _get(data, "metadata",             ProjectMetadata),
            half_life_kinetics    = _get(data, "half_life_kinetics",   HalfLifeKineticsPrefs),
            half_life_scanning    = _get(data, "half_life_scanning",   HalfLifeScanningPrefs),
            thermal               = _get(data, "thermal",              ThermalPrefs),
            actinometer           = _get(data, "actinometer",          ActinometerPrefs),
            led_characterisation  = _get(data, "led_characterisation", LEDCharacterisationPrefs),
            extinction_coeff      = _get(data, "extinction_coeff",     ExtinctionCoeffPrefs),
            spectra               = _get(data, "spectra",              SpectraPrefs),
            quantum_yield         = _get(data, "quantum_yield",        QuantumYieldPrefs),
        )
