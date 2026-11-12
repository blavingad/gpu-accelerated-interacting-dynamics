"""Self-contained synthetic verification case; no industrial field, font or DAC table.

A single charged sphere moves at constant axial speed in a uniform X field.
Drag, gravity, wakes and pair forces are disabled, while the implemented
area-scaled mass-loss law is retained. Its continuous solution is analytic.
"""
from pathlib import Path
import os,sys,json,argparse
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'src').exists():sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from vdcr.config import SimulationConfig
from vdcr.jax_solver import prepare_integrator

def main(folder,device='cpu'):
 out=Path(folder);out.mkdir(exist_ok=True,parents=True)
 field=out/'uniform_field.npz';e=np.zeros((3,3,3,3),np.float32);e[...,0]=-3e5
 np.savez(field,E=e,origin=np.array([-.02,-.02,0],np.float32),spacing=np.array([.02,.02,.04],np.float32))
 records=[]
 for dt in [1e-6,.5e-6,.25e-6]:
  cfg=SimulationConfig(initial_position_mm=(0,0,.13),initial_y_velocity=0,initial_z_velocity=24.,particle_diameter_mm=.132,droplet_density_kg_per_m3=905.,particles_y_dim=1,droplet_charges_pC=np.array([-1.]),intended_print_mask=np.array([True]),sequential_injection=True,openfoam_field_path=str(field.resolve()),coupled_substrate_z_mm=64.,terminate_at_field_exit=False,max_simulation_time=.001,dt=dt,coulomb_interaction_enabled=False,coalescence_enabled=False,wake_initial_velocity_deficit_factor=0.,gravity_acceleration=0.,air_density=0.,positive_air_flow_lpm=0.,wall_velocity_y=0.,raster_repeat_count=1,raster_droplets_per_repeat=1)
  p=prepare_integrator(cfg,device,output_mode='endpoints');r,_,_=p.run();steps=int(r['steps_executed']);t=steps*dt
  k=cfg.mass_loss_rate_per_meter;alpha=k*cfg.initial_z_velocity/(3*cfg.initial_particle_mass)
  expected_mass=cfg.initial_particle_mass*(1-alpha*t)**3
  expected_x=(-1e-12)*(-3e5)/cfg.initial_particle_mass*t*t/(2*(1-alpha*t))
  observed_x=float(r['final_position'][0,0]);observed_mass=float(r['final_mass'][0])
  records.append(dict(dt_s=dt,native_steps=steps,time_s=t,analytic_x_mm=expected_x*1000,computed_x_mm=observed_x*1000,x_absolute_error_um=abs(expected_x-observed_x)*1e6,analytic_mass_kg=expected_mass,computed_mass_kg=observed_mass,mass_relative_error=abs(expected_mass-observed_mass)/expected_mass))
 f=pd.DataFrame(records);f.to_csv(out/'analytic_verification.csv',index=False)
 assert np.isfinite(f.select_dtypes('number')).all().all() and f.x_absolute_error_um.max()<2 and f.mass_relative_error.max()<1e-3
 (out/'README.md').write_text('''# Public synthetic uniform-field case

This case uses no industrial geometry, font, raster sequence or DAC table. It verifies the retained mass-loss/force/time-integration equations against an analytic event-free limit. It is numerical verification, not experimental validation.

Uniform field: E=(-300000,0,0) V/m. Charge: -1 pC. Diameter: 132 µm. Density: 905 kg/m³. Axial velocity: 24 m/s. Drag, gravity, wake and Coulomb interactions are disabled. The implemented area-scaled mass loss is retained.

With k the initial-mass loss coefficient per distance, alpha=k*vz/(3*m0), the continuous solution is m(t)=m0*(1-alpha*t)^3 and X(t)=q*Ex*t²/[2*m0*(1-alpha*t)]. Reported time is the actual number of discrete steps times dt. Single-precision integration and field interpolation are retained. No landing-time interpolation is needed for this event-free verification window.

Run `python run_public_uniform_field_benchmark.py --output output --device cpu`. Install compatible NumPy, pandas, JAX and jaxlib. Accelerator installation depends on the target platform. Exact package versions used internally are in environment.json.
''')
 import jax,platform
 (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,pandas=pd.__version__,device=device),indent=2)+'\n')
 print(f.to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'results/generated/public_synthetic'));p.add_argument('--device',default='cpu');a=p.parse_args();main(a.output,a.device)
