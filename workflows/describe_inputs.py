"""Record physical parameters, numerical settings and input hashes."""
from pathlib import Path
from dataclasses import asdict
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from vdcr.framework_workload import Workload
from vdcr.electric_field import load_electric_field_grid
from _study_common import sha,write,environment,configuration

def main():
 dest=ROOT/'docs/parameters';dest.mkdir(parents=True,exist_ok=True)
 grid=load_electric_field_grid(ROOT/'inputs/electric_field.npz')
 geometry=dict(field_file='inputs/electric_field.npz',sha256=sha(ROOT/'inputs/electric_field.npz'),coordinate_units='m in NPZ and solver; mm in diagnostic tables',field_units='V/m',shape=list(grid.values.shape),origin_mm=(grid.origin*1000).tolist(),upper_bound_mm=(grid.upper_bound*1000).tolist(),spacing_mm=(grid.spacing*1000).tolist(),interpolation='Trilinear; zero field outside the stored grid',source_mesh_available_in_repository=False,field_poisson_solve_reproducible_from_this_repository=False)
 write(dest/'geometry.json',geometry);write(dest/'environment.json',environment());rows=[];controls=[];inventory=[]
 for case in ['low','high']:
  w=Workload(case);cfg=configuration(w.config);par=asdict(w.params)
  write(dest/f'{case}_physical_configuration.json',cfg);write(dest/f'{case}_charging_parameters.json',par)
  values=dict(droplet_diameter_um=w.config.particle_diameter_mm*1000,density_kg_m3=w.config.droplet_density_kg_per_m3,initial_mass_kg=w.config.initial_particle_mass,jet_speed_m_s=w.config.initial_z_velocity,frequency_hz=w.config.nozzle_frequency_hz,dt_s=w.config.dt,substrate_z_mm=w.config.coupled_substrate_z_mm,gutter_z_mm=43.770,speed_m_min=w.speed,capacitance_f=w.params.effective_coupling_capacitance_f,amplifier_tau_s=w.params.amplifier_tau_s,voltage_gain_v_per_DAC=w.params.voltage_gain_v_per_dac,flow_lpm=w.config.positive_air_flow_lpm)
  rows.append(dict(case=case,**values))
  controls.append(dict(case=case,contexts=len(w.pairs),current_only_contexts=sum(p==0 for p,c in w.pairs),known_keys=len(w.keys),binary_capacity=sum(1<<len(v) for v in w.defs.values()),blank_older_history=True,following_emitted=False,printed_command_min=230,printed_command_max=1023,breakup_phase=.5,guard_command_compensation=False,consecutive_print_command_compensation=False,neighbor_induction_coefficients=[.1,.02],waveform_state_continuous=True,phase_anchor='Blank/full simulation reference, not fitted to measured coordinates'))
  for name,category in [('physical_configuration.json','geometry/ink/numerical configuration'),('charge_parameters.json','charging/electronics calibration'),('printer_commands.csv','controller commands'),('charge_lookup.json','dependency representation'),('measured_coordinates.csv','experimental coordinate data'),('simulated_coordinates.csv','matched native simulation coordinates')]:
   path=f'inputs/cases/{case}/{name}';inventory.append(dict(path=path,category=category,sha256=sha(ROOT/path),included_in_manuscript_source_package=False))
 inventory.append(dict(path='inputs/electric_field.npz',category='industrial device field/geometry representation',sha256=geometry['sha256'],included_in_manuscript_source_package=False))
 pd.DataFrame(rows).to_csv(dest/'case_parameters.csv',index=False);write(dest/'controller_protocol.json',controls);write(dest/'input_inventory.json',dict(items=inventory))
 print('Parameter and input inventories written to docs/parameters.')
if __name__=='__main__':main()
