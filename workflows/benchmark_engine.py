"""Synchronized CPU/GPU execution benchmarks with explicit output and workload contracts."""
from pathlib import Path
import os,sys,time,json,argparse,hashlib
from dataclasses import asdict
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,jax
from vdcr.framework_workload import Workload,OUT
from vdcr.jax_solver import prepare_integrator
from vdcr.config import SimulationConfig

def sync(value):jax.block_until_ready(value);return value

def main(case='low',device='gpu',batch=8,repeat=3,full=False,output_dir=None,method='euler',synthetic=False,particles=32):
 if batch<1 or repeat<1 or particles<1:raise ValueError('batch, repeat and particles must be positive')
 out=Path(output_dir or OUT);out.mkdir(exist_ok=True,parents=True);model_sha=hashlib.sha256((ROOT/'src/vdcr/jax_solver.py').read_bytes()).hexdigest()
 if synthetic:
  case='synthetic';field=out/'synthetic_uniform_field.npz';E=np.zeros((3,3,3,3),np.float32);E[...,0]=-3e5
  if not field.exists():np.savez(field,E=E,origin=np.array([-.02,-.02,0],np.float32),spacing=np.array([.02,.02,.04],np.float32))
  else:
   with np.load(field) as f:np.testing.assert_array_equal(f['E'],E)
  cfg=SimulationConfig(initial_position_mm=(0,0,.13),initial_y_velocity=0,initial_z_velocity=24.,particle_diameter_mm=.132,droplet_density_kg_per_m3=905.,particles_y_dim=particles,droplet_charges_pC=-np.ones(particles),intended_print_mask=np.ones(particles,bool),sequential_injection=True,openfoam_field_path=str(field.resolve()),coupled_substrate_z_mm=64.,terminate_at_field_exit=False,max_simulation_time=.004,dt=1e-6,coulomb_interaction_enabled=True,coulomb_cutoff_distance_mm=5.,coalescence_enabled=False,positive_air_flow_lpm=0.,wall_velocity_y=0.,nozzle_frequency_hz=75000.,raster_repeat_count=1,raster_droplets_per_repeat=particles)
  charges=-np.random.default_rng(20261111).uniform(.5,1.5,size=(batch,particles));masks=np.ones_like(charges,bool)
  scope='Public synthetic interacting spheres in a uniform field; no industrial geometry, controller lookup or font; Coulomb/wake/drag enabled, coalescence disabled, fixed seeded charges'
 else:
  w=Workload(case);cfg=w.config;ids=np.array([w.pair_index[(65535,65535)]]+list(range(max(0,batch-1))),int)
  charges=w.charges(w.full_commands,ids);masks=w.eligible[ids];scope='Matched industrial paper workload; fixed physical configuration and capacity, dynamic waveform-derived charges and target masks'
 mode='full' if full else 'endpoints';started=time.perf_counter();p=prepare_integrator(cfg,device,output_mode=mode,method=method);q,m=p.inputs(charges,masks);prepare_s=time.perf_counter()-started
 if batch==1:q,m=q[0],m[0];kernel=p.kernel
 else:kernel=p.batch_kernel
 start=time.perf_counter();executable=kernel.lower(q,m).compile();compile_s=time.perf_counter()-start
 start=time.perf_counter();raw=sync(executable(q,m));first_s=time.perf_counter()-start
 times=[]
 for _ in range(repeat):
  start=time.perf_counter();raw=sync(executable(q,m));times.append(time.perf_counter()-start)
 start=time.perf_counter();host=jax.device_get(raw);transfer=time.perf_counter()-start
 nbytes=sum(v.nbytes for v in jax.tree.leaves(host));dest=out/f'benchmark_{case}_{device}_{method}_{mode}_b{batch}.json'
 record=dict(method=method,particle_precision=str(q.dtype),waveform_preparation_precision='float64 NumPy',case=case,synthetic=synthetic,particle_capacity=cfg.particles_y_dim,device=str(p.device),platform=device,mode=mode,batch=batch,native_steps=int(np.ceil(cfg.max_simulation_time/cfg.dt)),dt_s=cfg.dt,prepare_s=prepare_s,compile_s=compile_s,first_execution_s=first_s,warm_execution_s=times,warm_median_s=float(np.median(times)),warm_cases_per_s=batch/float(np.median(times)),device_to_host_s=transfer,output_bytes=nbytes,synchronization='jax.block_until_ready on complete result',scope=scope,python=sys.version.split()[0],jax=jax.__version__,model_sha256=model_sha,benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),charge_input_sha256=hashlib.sha256(charges.tobytes()).hexdigest(),mask_input_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),field_sha256=hashlib.sha256(Path(cfg.openfoam_field_path).read_bytes()).hexdigest(),final_landed_count=int(np.asarray(host[0][3]).sum()),final_mass_sum_kg=float(np.asarray(host[0][6]).sum()))
 if synthetic:record['synthetic_configuration']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(cfg).items()}
 dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir');p.add_argument('--method',choices=['euler','rk4'],default='euler');p.add_argument('--case',choices=['low','high'],default='low');p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--batch',type=int,default=8);p.add_argument('--repeat',type=int,default=3);p.add_argument('--full',action='store_true');p.add_argument('--synthetic',action='store_true');p.add_argument('--particles',type=int,default=32);a=p.parse_args();main(a.case,a.device,a.batch,a.repeat,a.full,a.output_dir,a.method,a.synthetic,a.particles)
