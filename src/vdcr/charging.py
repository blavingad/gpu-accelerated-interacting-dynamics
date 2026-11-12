"""Analytical summed-RC surrogate for CIJ charge induction before breakup."""
from dataclasses import dataclass
import numpy as np

EPSILON_0 = 8.8541878128e-12

@dataclass(frozen=True)
class Ink:
    name: str
    resistivity_ohm_cm: float
    relative_permittivity: float = 19.0
    density_kg_m3: float = 906.0
    viscosity_pa_s: float = 3.5e-3
    surface_tension_n_m: float = 0.025

    @property
    def resistivity_ohm_m(self): return self.resistivity_ohm_cm * 1e-2
    @property
    def conductivity_s_m(self): return 1.0 / self.resistivity_ohm_m
    @property
    def dielectric_relaxation_s(self):
        return EPSILON_0 * self.relative_permittivity / self.conductivity_s_m

@dataclass(frozen=True)
class ChargingGeometry:
    frequency_hz: float = 76_800.0
    orifice_diameter_m: float = 70e-6
    wavelength_to_orifice_ratio: float = 4.5
    jet_diameter_m: float = 130e-6
    plate_gap_m: float = 0.6e-3
    nozzle_to_electrode_m: float = 2e-3
    electrode_length_m: float = 5.5e-3
    breakup_fraction_in_electrode: float = 0.5
    neck_length_wavelengths: float = 0.5
    neck_to_drop_radius: float = 0.10

    @property
    def wavelength_m(self): return self.wavelength_to_orifice_ratio * self.orifice_diameter_m
    @property
    def jet_speed_m_s(self): return self.frequency_hz * self.wavelength_m
    @property
    def period_s(self): return 1.0 / self.frequency_hz
    @property
    def breakup_distance_m(self):
        return self.nozzle_to_electrode_m + self.breakup_fraction_in_electrode * self.electrode_length_m
    @property
    def charging_residence_s(self):
        return self.breakup_fraction_in_electrode * self.electrode_length_m / self.jet_speed_m_s
    @property
    def drop_radius_m(self):
        r = self.jet_diameter_m / 2
        return (3 * r**2 * self.wavelength_m / 4) ** (1 / 3)
    @property
    def neck_length_m(self): return self.neck_length_wavelengths * self.wavelength_m
    @property
    def neck_radius_m(self): return self.neck_to_drop_radius * self.drop_radius_m

DEFAULT_INKS = (Ink("Rm", 1100.0), Ink("Rc", 1350.0), Ink("Rx", 1600.0))

def summed_thread_resistance(ink, lengths_m, radii_m):
    lengths, radii = np.asarray(lengths_m, float), np.asarray(radii_m, float)
    if lengths.shape != radii.shape or np.any(lengths < 0) or np.any(radii <= 0):
        raise ValueError("lengths and radii must have equal shapes and physical values")
    return float(np.sum(ink.resistivity_ohm_m * lengths / (np.pi * radii**2)))

def resistance_components(ink, geometry=ChargingGeometry()):
    bulk_length = geometry.breakup_distance_m - geometry.neck_length_m
    if bulk_length < 0: raise ValueError("neck exceeds connected-column length")
    bulk = summed_thread_resistance(ink, [bulk_length], [geometry.jet_diameter_m / 2])
    neck = summed_thread_resistance(ink, [geometry.neck_length_m], [geometry.neck_radius_m])
    return bulk, neck

def breakup_resistance(ink, geometry=ChargingGeometry()):
    return sum(resistance_components(ink, geometry))

def plate_capacitances(geometry=ChargingGeometry()):
    h, r = geometry.plate_gap_m / 2, geometry.jet_diameter_m / 2
    if h <= r: raise ValueError("jet must fit between plates")
    c = 2 * np.pi * EPSILON_0 * geometry.wavelength_m / np.arccosh(h / r)
    return float(c), float(c)

def analytical_step_charge(voltage_v, duration_s, resistance_ohm,
                           driven_capacitance_f, total_capacitance_f,
                           initial_charge_c=0.0):
    if duration_s < 0 or resistance_ohm <= 0 or total_capacitance_f <= 0:
        raise ValueError("duration must be non-negative and R/C positive")
    qeq = -driven_capacitance_f * voltage_v
    return float(qeq + (initial_charge_c - qeq) *
                 np.exp(-duration_s / (resistance_ohm * total_capacitance_f)))

def breakup_charge(voltage_v, ink, geometry=ChargingGeometry(), pulse_cycles=1.0):
    driven, grounded = plate_capacitances(geometry)
    return analytical_step_charge(voltage_v, pulse_cycles * geometry.period_s,
                                  breakup_resistance(ink, geometry),
                                  driven, driven + grounded)

def voltage_sweep(voltages_v, inks=DEFAULT_INKS, geometry=ChargingGeometry()):
    v = np.asarray(voltages_v, float)
    return {ink.name: np.array([breakup_charge(x, ink, geometry) for x in v]) * 1e12
            for ink in inks}
