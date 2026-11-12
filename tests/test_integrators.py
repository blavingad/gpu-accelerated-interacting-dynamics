from pathlib import Path
import tempfile,unittest,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vdcr.config import SimulationConfig
from vdcr.jax_solver import prepare_integrator

class IntegratorTests(unittest.TestCase):
 def config(self,field,dt=1e-6):
  return SimulationConfig(initial_position_mm=(0,0,.13),initial_y_velocity=0,initial_z_velocity=24.,particle_diameter_mm=.132,droplet_density_kg_per_m3=905.,particles_y_dim=1,droplet_charges_pC=np.array([-1.]),intended_print_mask=np.array([True]),sequential_injection=True,openfoam_field_path=str(field),coupled_substrate_z_mm=64.,terminate_at_field_exit=False,max_simulation_time=.001,dt=dt,coulomb_interaction_enabled=False,coalescence_enabled=False,wake_initial_velocity_deficit_factor=0.,gravity_acceleration=0.,air_density=0.,positive_air_flow_lpm=0.,wall_velocity_y=0.,raster_repeat_count=1,raster_droplets_per_repeat=1)
 def test_analytic_mass_force_limit_and_stations(self):
  with tempfile.TemporaryDirectory() as folder:
   field=Path(folder)/'uniform.npz';E=np.zeros((3,3,3,3),np.float32);E[...,0]=-3e5
   np.savez(field,E=E,origin=np.array([-.02,-.02,0],np.float32),spacing=np.array([.02,.02,.04],np.float32))
   errors={}
   for method in ['euler','rk4']:
    cfg=self.config(field);p=prepare_integrator(cfg,'cpu',output_mode='endpoints',method=method,release_times_s=[0.],sample_stations_mm=[10.,20.])
    r,_,_=p.run();t=int(r['steps_executed'])*cfg.dt;alpha=cfg.mass_loss_rate_per_meter*24/(3*cfg.initial_particle_mass)
    expected=3e-7/cfg.initial_particle_mass*t*t/(2*(1-alpha*t));errors[method]=abs(float(r['final_position'][0,0])-expected)*1e6
    self.assertTrue(np.isfinite(r['station_time_s']).all());self.assertLess(np.max(abs(r['station_position_m'][0,:,2]-np.array([.01,.02]))),1e-8)
   self.assertLess(errors['rk4'],.02);self.assertLess(errors['rk4'],errors['euler']/5)
 def test_invalid_method(self):
  with self.assertRaises(ValueError):prepare_integrator(SimulationConfig(),'cpu',method='bad')
if __name__=='__main__':unittest.main()
