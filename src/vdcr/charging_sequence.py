"""Continuous amplifier and provisional short-range charging induction.

The induction coefficients are inferred from an isolated neutral-guard reference
with ideal electronics. They are not a solved charging-electrode capacitance
matrix or an independent measurement of the actual charging process.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np
from .charge_explorer import derived,command_voltage,response

@dataclass(frozen=True)
class ChargingState:
    voltage_v: float = 0.
    previous_net_charges_pC: tuple = (0.,0.)  # newest first; includes guards
    last_printed_command_v: float = 0.
    guards_since_print: int = 1000000

    @classmethod
    def from_value(cls,value=None,initial_voltage_v=0.):
        if isinstance(value,cls):return value
        if value is None:return cls(voltage_v=float(initial_voltage_v))
        return cls(**{**value,'previous_net_charges_pC':tuple(value['previous_net_charges_pC'])})

def history_coefficients(raster_name):
    # Experimental common lag-1/lag-2 coupling, inferred from guard cancellation.
    # Applies to any forming drop; it does not add compensation to printed commands.
    return (.10,.02)

def run_sequence(dacs,printed,parameters,*,compensation_fractions=(0.,0.),
                 induction_coefficients=(0.,0.),initial_state=None,initial_voltage_v=0.,
                 target_printed_charges_pC=None,breakup_phase=.5):
    """Sample electrode + neighbour contributions at breakup for EVERY drop.

    Guard commands use the latest actual printed command: first guard k1*Vlast,
    second guard k2*Vlast. No printed-DAC reduction is applied. A missing print
    slot is an unprinted emitted drop and participates in this guard history.
    The two-neighbour equilibrium term is -sum(k_l*q_net[n-l]). It is filtered
    by the retained liquid RC, as a constant within each formation cycle.

    Optional targets invert the affine charge equation for printed commands.
    This preserves exact per-position average NET charges while recomputing
    physically consistent compensation commands and guard charges. Commands
    are continuous simulation DACs; infeasible values are errors, not clipped.
    """
    p=parameters;d=derived(p)
    base=np.asarray(dacs,float);active=np.asarray(printed,bool)
    if base.ndim!=1 or active.shape!=base.shape:raise ValueError('One-dimensional DAC sequence and matching printed mask required')
    command_voltage(base,p)  # validate DAC values, including unprinted zeros
    if np.any(base[~active]!=0):raise ValueError('Unprinted base DACs must be zero; compensation is generated separately')
    comp=np.asarray(compensation_fractions,float);coeff=np.asarray(induction_coefficients,float)
    for values in [comp,coeff]:
        if values.shape!=(2,) or np.any(~np.isfinite(values)) or np.any(values<0) or values.sum()>=1:raise ValueError('Two finite nonnegative history fractions with sum <1 required')
    if not np.isfinite(breakup_phase) or not 0<breakup_phase<=1:raise ValueError('Breakup phase must lie in (0,1]')
    state=ChargingState.from_value(initial_state,initial_voltage_v)
    history=np.asarray(state.previous_net_charges_pC,float)
    if history.shape!=(2,) or np.any(~np.isfinite(history)) or not np.isfinite(state.voltage_v) or not 0<=state.voltage_v<=p.voltage_rail_v:raise ValueError('Invalid charging state')
    if not np.isfinite(state.last_printed_command_v) or not 0<=state.last_printed_command_v<=p.voltage_rail_v or state.guards_since_print<0:raise ValueError('Invalid compensation history')
    targets=None if target_printed_charges_pC is None else np.asarray(target_printed_charges_pC,float)
    if targets is not None and (targets.shape!=base.shape or np.any(~np.isfinite(targets[active]))):raise ValueError('Charge targets must match the physical sequence')
    period=1/p.frequency_hz;sample=breakup_phase*period
    if sample>d['residence_time_s']:raise ValueError('Breakup sample exceeds electrode residence')
    liquid_fraction=-np.expm1(-sample/d['rc_time_s'])
    voltage=float(state.voltage_v);last=float(state.last_printed_command_v);age=int(state.guards_since_print)
    max_command=min(p.voltage_rail_v,float(command_voltage(p.dac_max,p)))
    if targets is not None and (p.voltage_offset_v!=0 or p.voltage_curvature_v!=0):raise ValueError('Mean-charge inversion currently requires linear zero-offset voltage commands')
    rows=[]
    for i,(dac,on) in enumerate(zip(base,active)):
        neighbour_equilibrium=-float(np.dot(coeff,history))
        neighbour_charge=neighbour_equilibrium*liquid_fraction
        local=replace(p,initial_voltage_v=voltage,initial_charge_c=0.)
        guard_fraction=0.
        if on:
            target=float(command_voltage(dac,p));age=0
            if targets is not None:
                q0=float(response(sample,0.,local,d)['charge_c'])*1e12
                q1=float(response(sample,1.,local,d)['charge_c'])*1e12
                target=(float(targets[i])-neighbour_charge-q0)/(q1-q0)
                if not np.isfinite(target) or target < -1e-9 or target>max_command+1e-9:
                    raise ValueError(f'Infeasible average charge at cycle {i+1}: electrode command {target:g} V outside 0..{max_command:g}')
                target=float(np.clip(target,0,max_command))  # numerical endpoint tolerance only
            last=target
        else:
            age+=1
            guard_fraction=float(comp[age-1]) if age<=len(comp) else 0.
            target=guard_fraction*last
        sample_response=response(sample,target,local,d)
        electrode=float(sample_response['charge_c'])*1e12
        net=electrode+neighbour_charge
        if on and targets is not None:
            if not np.isclose(net,targets[i],rtol=0,atol=2e-12):raise ValueError('Average-charge inversion residual exceeds tolerance')
            net=float(targets[i])
        end=float(target if p.amplifier_tau_s==0 else target+(voltage-target)*np.exp(-period/p.amplifier_tau_s))
        vmid=float(sample_response['voltage_v'])
        rows.append(dict(cycle=i+1,printed=bool(on),base_dac=float(dac),effective_dac=target/p.voltage_gain_v_per_dac,
            guard_compensation_fraction=guard_fraction,command_voltage_v=target,start_voltage_v=voltage,
            breakup_voltage_v=vmid,end_voltage_v=end,breakup_time_s=(i+breakup_phase)*period,
            raw_rc_charge_pC=electrode,electrode_induced_charge_pC=electrode,
            neighbour_equilibrium_charge_pC=neighbour_equilibrium,neighbour_induced_charge_pC=neighbour_charge,
            quasistatic_charge_pC=-d['capacitance_driven_f']*vmid*1e12,
            assigned_charge_pC=net,target_charge_pC=float(targets[i]) if on and targets is not None else np.nan))
        history=np.array([net,history[0]]);voltage=end
    final=ChargingState(voltage,tuple(map(float,history)),last,age)
    return dict(trace=rows,end_voltage_v=voltage,end_charging_state=asdict(final),initial_charging_state=asdict(state))
