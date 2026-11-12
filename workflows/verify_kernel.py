"""Physical parity and dynamic-input checks against the preserved original solver."""
from pathlib import Path
import os,sys,json,importlib.util,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from vdcr.config import SimulationConfig
from vdcr.jax_solver import prepare_integrator,integrate
from vdcr.flight_reporting import crossings
spec=importlib.util.spec_from_file_location('vdcr._pre_reuse_solver',ROOT/'tests/fixtures/reference_solver.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
cfg_dict=json.loads((ROOT/'inputs/cases/low/physical_configuration.json').read_text())
cfg_dict['openfoam_field_path']=str(ROOT/'inputs/electric_field.npz')
cfg_dict.update(particles_y_dim=6,raster_repeat_count=1,raster_droplets_per_repeat=6,initial_position_mm=[0,0,42.9],coupled_substrate_z_mm=45.,max_simulation_time=.0002,droplet_charges_pC=np.array([-.6,-.8,-1.,-1.2,-1.4,-1.6]),intended_print_mask=np.array([True,False,True,True,False,True]))
cfg=SimulationConfig(**cfg_dict);cfg.validate()
start=time.time();old,_,_=reference.integrate(cfg,'cpu');full=prepare_integrator(cfg,'cpu');new,_,_=full.run()
for key in old:
 if old[key].dtype==bool or 'lineage' in key or 'merge_partner'==key:np.testing.assert_array_equal(new[key],old[key],err_msg=key)
 else:np.testing.assert_allclose(new[key],old[key],rtol=2e-5,atol=1e-10,equal_nan=True,err_msg=key)
endpoint=prepare_integrator(cfg,'cpu',output_mode='endpoints');lean,_,_=endpoint.run()
np.testing.assert_allclose(lean['final_landing_position'],old['final_landing_position'],rtol=0,atol=1e-8)
np.testing.assert_array_equal(lean['final_lineage'],old['final_lineage'])
for r in crossings(old,range(6)):
 np.testing.assert_allclose(lean['gutter_x_mm'][r['event_id']],r['x_gutter_mm'],rtol=1e-5,atol=1e-6,equal_nan=True)
for node in range(6):
 hit=np.flatnonzero(old['active_before'][:,node]&~old['active_after'][:,node])
 if len(hit):np.testing.assert_allclose(lean['landing_time_s'][node],(hit[-1]+1)*cfg.dt,atol=1e-10)
# Check dynamic inputs reuse the compiled executable.
charge2=cfg.droplet_charges_pC*1.07;mask2=np.logical_not(cfg.intended_print_mask)
second,_,_=endpoint.run(charge2,mask2)
assert not np.allclose(second['final_landing_position'][:,0],lean['final_landing_position'][:,0],rtol=0,atol=1e-8)
assert endpoint.kernel._cache_size()==1
q,m=endpoint.inputs(np.stack([cfg.droplet_charges_pC,charge2]),np.stack([cfg.intended_print_mask,mask2]))
import jax
batch=endpoint.batch_kernel(q,m);jax.block_until_ready(batch)
np.testing.assert_allclose(np.asarray(batch[0][5][0]),lean['final_landing_position'],rtol=0,atol=1e-8)
np.testing.assert_allclose(np.asarray(batch[0][5][1]),second['final_landing_position'],rtol=0,atol=1e-8)
result=dict(passed=True,reference_full_history_parity=True,endpoint_landing_lineage_and_gutter_parity=True,landing_time_parity=True,dynamic_charges_and_masks_tested=True,single_compilation_for_dynamic_inputs=True,batched_scalar_parity=True,runtime_s=time.time()-start)
(ROOT/'results/generated').mkdir(parents=True,exist_ok=True)
(ROOT/'results/generated/kernel_parity_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
