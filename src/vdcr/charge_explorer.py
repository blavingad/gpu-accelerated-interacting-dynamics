"""Analytical charging equations shared by emission assignment and standalone experiments.
The forward trajectory integrator is unchanged.

The legacy-plane equations reproduce charging.py at matching geometry when
additional electronics, stray capacitance and waveform inputs are at defaults. It is
an engineering approximation, not a solved charging-electrode field map.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np
from .charging import EPSILON_0

@dataclass(frozen=True)
class ChargeParameters:
    frequency_hz: float = 76800.
    kinematics_mode: str = 'wavelength_ratio'
    orifice_diameter_m: float = 70e-6
    wavelength_ratio: float = 4.5
    jet_speed_m_s: float = 24.2
    jet_diameter_m: float = 130e-6
    capacitance_mode: str = 'effective'
    effective_coupling_capacitance_f: float = 10e-15
    geometry: str = 'legacy_planes'
    plate_gap_m: float = .6e-3
    sleeve_inner_radius_m: float = .3e-3
    electrode_length_m: float = 6e-3
    nozzle_to_electrode_m: float = 2.42e-3
    breakup_fraction: float = .5
    neck_length_wavelengths: float = .5
    neck_to_drop_radius: float = .1
    resistivity_ohm_cm: float = 1100.
    ink_relative_permittivity: float = 19.
    external_relative_permittivity: float = 1.
    density_kg_m3: float = 906.
    return_capacitance_ratio: float = 1.
    stray_capacitance_f: float = 0.
    exposure_mode: str = 'cycles'
    pulse_cycles: float = .5
    explicit_duration_s: float = 13e-6
    limit_to_residence: bool = True
    voltage_gain_v_per_dac: float = .25
    voltage_offset_v: float = 0.
    voltage_curvature_v: float = 0.
    voltage_rail_v: float = 300.
    amplifier_tau_s: float = 2e-6
    initial_voltage_v: float = 0.
    initial_charge_c: float = 0.
    dac_max: int = 1023


def derived(p=ChargeParameters()):
    values=asdict(p)
    for name,value in values.items():
        if isinstance(value,(int,float)) and not np.isfinite(value):raise ValueError(f'{name} must be finite')
    positive=['effective_coupling_capacitance_f','frequency_hz','orifice_diameter_m','wavelength_ratio','jet_speed_m_s','jet_diameter_m','plate_gap_m','sleeve_inner_radius_m','electrode_length_m','neck_to_drop_radius','resistivity_ohm_cm','ink_relative_permittivity','external_relative_permittivity','density_kg_m3','voltage_rail_v','dac_max']
    for name in positive:
        if values[name]<=0:raise ValueError(f'{name} must be positive')
    for name in ['nozzle_to_electrode_m','neck_length_wavelengths','return_capacitance_ratio','stray_capacitance_f','pulse_cycles','explicit_duration_s','amplifier_tau_s']:
        if values[name]<0:raise ValueError(f'{name} must be nonnegative')
    if not 0<p.breakup_fraction<=1:raise ValueError('Breakup fraction must be in (0,1]')
    if not 0<p.neck_to_drop_radius<=1:raise ValueError('Neck/drop radius ratio must be in (0,1]')
    if p.kinematics_mode=='wavelength_ratio':
        wavelength=p.wavelength_ratio*p.orifice_diameter_m;speed=p.frequency_hz*wavelength
    elif p.kinematics_mode=='speed':
        speed=p.jet_speed_m_s;wavelength=speed/p.frequency_hz
    else:raise ValueError('Unknown jet kinematics mode')
    radius=p.jet_diameter_m/2
    volume=np.pi*radius**2*wavelength
    drop_radius=(3*volume/(4*np.pi))**(1/3)
    breakup_distance=p.nozzle_to_electrode_m+p.breakup_fraction*p.electrode_length_m
    neck_length=p.neck_length_wavelengths*wavelength
    if neck_length>breakup_distance:raise ValueError('Neck exceeds nozzle-to-breakup length')
    neck_radius=p.neck_to_drop_radius*drop_radius
    resistivity=p.resistivity_ohm_cm*.01
    r_bulk=resistivity*(breakup_distance-neck_length)/(np.pi*radius**2)
    r_neck=resistivity*neck_length/(np.pi*neck_radius**2)
    resistance=r_bulk+r_neck
    if resistance<=0:raise ValueError('Connected liquid resistance must be positive')
    eps=EPSILON_0*p.external_relative_permittivity
    if p.geometry=='legacy_planes':
        if p.plate_gap_m/2<=radius:raise ValueError('Jet must fit between the charging plates')
        c_driven=2*np.pi*eps*wavelength/np.arccosh(p.plate_gap_m/(2*radius))
    elif p.geometry=='coaxial':
        if p.sleeve_inner_radius_m<=radius:raise ValueError('Sleeve bore must exceed the jet radius')
        c_driven=2*np.pi*eps*wavelength/np.log(p.sleeve_inner_radius_m/radius)
    else:raise ValueError('Unknown charging-electrode geometry')
    c_geometry=c_driven
    if p.capacitance_mode=='effective':c_driven=p.effective_coupling_capacitance_f
    elif p.capacitance_mode!='geometry':raise ValueError('Unknown capacitance mode')
    c_return=p.return_capacitance_ratio*c_driven
    c_total=c_driven+c_return+p.stray_capacitance_f
    residence=p.breakup_fraction*p.electrode_length_m/speed
    if p.exposure_mode=='cycles':requested=p.pulse_cycles/p.frequency_hz
    elif p.exposure_mode=='residence':requested=residence
    elif p.exposure_mode=='explicit':requested=p.explicit_duration_s
    else:raise ValueError('Unknown exposure mode')
    duration=min(requested,residence) if p.limit_to_residence else requested
    return dict(wavelength_m=wavelength,jet_speed_m_s=speed,period_s=1/p.frequency_hz,
        drop_diameter_m=2*drop_radius,drop_volume_m3=volume,drop_mass_kg=p.density_kg_m3*volume,
        volume_flow_m3_s=volume*p.frequency_hz,breakup_distance_m=breakup_distance,
        neck_radius_m=neck_radius,neck_length_m=neck_length,
        resistance_bulk_ohm=r_bulk,resistance_neck_ohm=r_neck,resistance_total_ohm=resistance,
        capacitance_geometry_f=c_geometry,capacitance_driven_f=c_driven,capacitance_return_f=c_return,capacitance_total_f=c_total,
        rc_time_s=resistance*c_total,dielectric_relaxation_s=EPSILON_0*p.ink_relative_permittivity*resistivity,
        residence_time_s=residence,requested_time_s=requested,charging_time_s=duration)


def command_voltage(dac,p=ChargeParameters()):
    d=np.asarray(dac,dtype=float)
    if np.any(~np.isfinite(d)) or np.any((d<0)|(d>p.dac_max)):raise ValueError('DAC must be finite and within the configured range')
    raw=p.voltage_offset_v+p.voltage_gain_v_per_dac*d+p.voltage_curvature_v*(d/p.dac_max)**2
    return np.clip(raw,-p.voltage_rail_v,p.voltage_rail_v)


def response(time_s,voltage_v,p=ChargeParameters(),properties=None):
    """Exact linear RC response to a step or first-order amplifier rise.

    A phenomenological waveform experiment, not a droplet-memory model. Initial
    electrode voltage and initial induced-charge state are independent inputs.
    """
    d=derived(p) if properties is None else properties
    t=np.asarray(time_s,dtype=float);v=np.asarray(voltage_v,dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t<0) or np.any(~np.isfinite(v)):raise ValueError('Finite voltage and nonnegative finite time are required')
    c=d['capacitance_driven_f'];tau=d['rc_time_s'];qinf=-c*v
    decay=np.exp(-t/tau)
    # Stable for a nearly instantaneous observation as well as full saturation.
    q=p.initial_charge_c*decay+qinf*(-np.expm1(-t/tau))
    if p.amplifier_tau_s==0:
        voltage=np.broadcast_arrays(t,v)[1]
    else:
        amp=p.amplifier_tau_s;decay_amp=np.exp(-t/amp)
        voltage=p.initial_voltage_v+(v-p.initial_voltage_v)*(-np.expm1(-t/amp))
        amplitude=c*(v-p.initial_voltage_v)
        if np.isclose(amp,tau,rtol=1e-7,atol=0):extra=amplitude*(t/tau)*decay
        else:extra=amplitude*amp/(amp-tau)*(decay_amp-decay)
        q=q+extra
    return dict(voltage_v=voltage,charge_c=q,equilibrium_charge_c=qinf)


def evaluate(dac,p=ChargeParameters()):
    d=derived(p);v=command_voltage(dac,p)
    result=response(d['charging_time_s'],v,p,d)
    return dict(dac=np.asarray(dac,dtype=float),command_voltage_v=v,**result,properties=d)


def flight_drop_consistent_preset():
    """A mass-consistent comparison, not a new measured charging calibration."""
    p=replace(ChargeParameters(),kinematics_mode='speed')
    wavelength=p.jet_speed_m_s/p.frequency_hz
    return replace(p,jet_diameter_m=float(np.sqrt(2/3*(130e-6)**3/wavelength)))
