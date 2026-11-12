"""Controlled Euler/RK4 comparison with frozen releases and shared event handling."""
from pathlib import Path
from dataclasses import replace
import os,sys,json,time,argparse,hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,jax
from vdcr.framework_workload import Workload
from vdcr.jax_solver import prepare_integrator
from vdcr.config import SimulationConfig
from _study_common import STUDY,configuration,declare,environment,sources,inputs,write,freeze
DTS=[1.,.5,.25]
REFERENCE_DT=.125

def representatives(w):
 selected=[dict(label='isolated_least',previous_mask=0,current_mask=1),dict(label='full_blank_history',previous_mask=0,current_mask=65535),dict(label='full_previous_current',previous_mask=65535,current_mask=65535)]
 used={(r['previous_mask'],r['current_mask']) for r in selected}
 f=pd.read_csv(ROOT/f'results/reference/{w.case}_complete_pool/context_metrics.csv').sort_values('RMS_um',ascending=False)
 for r in f.itertuples():
  pair=(int(r.previous_mask),int(r.current_mask))
  if pair not in used:selected.append(dict(label='worst_frozen_context',previous_mask=pair[0],current_mask=pair[1]));break
 assert len(selected)==4
 return selected

def analytic(out,device='cpu'):
 dest=out/'analytic';dest.mkdir(parents=True,exist_ok=True);field=dest/'uniform_field.npz';E=np.zeros((3,3,3,3),np.float32);E[...,0]=-3e5
 np.savez(field,E=E,origin=np.array([-.02,-.02,0],np.float32),spacing=np.array([.02,.02,.04],np.float32));records=[]
 for method in ['euler','rk4']:
  for dt_us in DTS:
   dt=dt_us*1e-6;cfg=SimulationConfig(initial_position_mm=(0,0,.13),initial_y_velocity=0,initial_z_velocity=24.,particle_diameter_mm=.132,droplet_density_kg_per_m3=905.,particles_y_dim=1,droplet_charges_pC=np.array([-1.]),intended_print_mask=np.array([True]),sequential_injection=True,openfoam_field_path=str(field.resolve()),coupled_substrate_z_mm=64.,terminate_at_field_exit=False,max_simulation_time=np.nextafter(.001024,0),dt=dt,coulomb_interaction_enabled=False,coalescence_enabled=False,wake_initial_velocity_deficit_factor=0.,gravity_acceleration=0.,air_density=0.,positive_air_flow_lpm=0.,wall_velocity_y=0.,raster_repeat_count=1,raster_droplets_per_repeat=1)
   r,_,_=prepare_integrator(cfg,device,output_mode='endpoints',method=method,release_times_s=[0.]).run();t=int(r['steps_executed'])*dt;alpha=cfg.mass_loss_rate_per_meter*24/(3*cfg.initial_particle_mass)
   x=3e-7/cfg.initial_particle_mass*t*t/(2*(1-alpha*t));mass=cfg.initial_particle_mass*(1-alpha*t)**3
   records.append(dict(method=method,dt_us=dt_us,actual_time_s=t,x_error_um=abs(float(r['final_position'][0,0])-x)*1e6,mass_relative_error=abs(float(r['final_mass'][0])-mass)/mass))
 f=pd.DataFrame(records);f.to_csv(dest/'errors.csv',index=False)
 if not np.isfinite(f.select_dtypes('number')).all().all():raise RuntimeError('Nonfinite analytic verification')


def compare_case(out,case):
 paths=[p for p in (out/case).glob('*/*/positions.csv')]
 endpoints=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
 stations=pd.concat([pd.read_csv(p.parent/'stations.csv') for p in paths],ignore_index=True)
 keys=['context_index','drop'];ref=endpoints[(endpoints.method=='rk4')&(endpoints.dt_us==REFERENCE_DT)]
 er=ref[keys+['landing_mm','landing_time_s','landed','merged']].rename(columns={k:k+'_reference' for k in ['landing_mm','landing_time_s','landed','merged']})
 e=endpoints.merge(er,on=keys,validate='many_to_one')
 sr=stations[(stations.method=='rk4')&(stations.dt_us==REFERENCE_DT)]
 sk=['context_index','node_id','station_z_mm'];cols=['time_s','x_mm','y_mm','paper_horizontal_mm','singleton_final','requested_current']
 sr=sr[sk+cols].rename(columns={k:k+'_reference' for k in cols})
 s=stations.merge(sr,on=sk,validate='many_to_one');s['delta_x_um']=(s.x_mm-s.x_mm_reference)*1000;s['delta_horizontal_um']=(s.paper_horizontal_mm-s.paper_horizontal_mm_reference)*1000;s['delta_time_us']=(s.time_s-s.time_s_reference)*1e6
 s['comparable']=s.requested_current&s.requested_current_reference&s.singleton_final&s.singleton_final_reference&np.isfinite(s.delta_x_um)&np.isfinite(s.delta_horizontal_um)
 s.to_csv(out/case/'station_differences.csv',index=False)
 rows=[]
 for (method,dt_us),g in e.groupby(['method','dt_us']):
  valid=g.landed&g.landed_reference&~g.merged&~g.merged_reference&np.isfinite(g.landing_mm)&np.isfinite(g.landing_mm_reference)
  dx=(g.loc[valid,'landing_mm']-g.loc[valid,'landing_mm_reference'])*1000
  h=s[(s.method==method)&(s.dt_us==dt_us)&s.comparable]
  final=h[np.isclose(h.station_z_mm,h.station_z_mm.max())]
  finite=lambda a:float(np.sqrt(np.mean(np.asarray(a)**2))) if len(a) else None
  maximum=lambda a:float(np.max(abs(np.asarray(a)))) if len(a) else None
  rows.append(dict(case=case,method=method,dt_us=dt_us,requested_targets=len(g),scored_native_targets=int(valid.sum()),missing_targets=int((~g.landed).sum()),merged_targets=int(g.merged.sum()),event_flag_disagreements=int(((g.landed!=g.landed_reference)|(g.merged!=g.merged_reference)).sum()),native_endpoint_RMS_difference_um=finite(dx),native_endpoint_max_difference_um=maximum(dx),station_samples=len(h),station_RMS_deflection_difference_um=finite(h.delta_x_um),station_max_deflection_difference_um=maximum(h.delta_x_um),interpolated_endpoint_scored_targets=len(final),interpolated_endpoint_RMS_deflection_difference_um=finite(final.delta_x_um),interpolated_endpoint_max_deflection_difference_um=maximum(final.delta_x_um),interpolated_endpoint_RMS_horizontal_difference_um=finite(final.delta_horizontal_um),interpolated_endpoint_max_horizontal_difference_um=maximum(final.delta_horizontal_um),station_max_arrival_difference_us=maximum(h.delta_time_us)))
 pd.DataFrame(rows).to_csv(out/case/'summary.csv',index=False)
 return rows

def main(output=None,device='gpu'):
 out=Path(output or STUDY/'integrators')
 if (out/'completed.json').exists():freeze(out);print('Frozen integrator study already complete');return
 models={c:Workload(c) for c in ['low','high']};selection={c:representatives(w) for c,w in models.items()}
 for w in models.values():
  if w.config.positive_air_unsteady_scale!=0:raise ValueError('This controlled cross-dt study requires inactive stochastic carrier forcing')
 plan=dict(study='shared_force_euler_rk4_v1',cases=selection,methods=['euler','rk4'],timesteps_us=DTS,finest_reference=dict(method='rk4',dt_us=REFERENCE_DT,exact_truth=False),release_schedule='All refinements use the exact same release times rounded once to the native 1-us grid; charging and guards unchanged',events='Same grid-boundary releases, merge eligibility/pairing, collection and landing policy; no event root solver',sampling='Common fixed axial planes, linearly interpolated within each native step; samples spanning merges excluded; denominators retained',force_law='Identical field, pair/wake/drag/carrier closures; RK4 advances position, velocity and mass jointly; coefficients of inactive stochastic flow remain unused',device=device,precision='float32 particle state; NumPy float64 actuation preparation',sources=sources(['workflows/run_integrator_comparison.py']),selection_source_sha256={c:hashlib.sha256((ROOT/f'results/reference/{c}_complete_pool/context_metrics.csv').read_bytes()).hexdigest() for c in models},inputs={c:inputs(c) for c in models},conclusion_policy='No claim of fourth-order convergence for discontinuous switching, linear station interpolation or grid-resolved events')
 declare(out,plan)
 write(out/'environment.json',environment());timings=[];summaries=[]
 for case,w in models.items():
  chosen=selection[case];ids=np.array([w.pair_index[r['previous_mask'],r['current_mask']] for r in chosen]);q0=w.charges(w.full_commands,ids);m0=w.eligible[ids]
  releases=np.rint(np.arange(w.config.particles_y_dim)/(w.config.nozzle_frequency_hz*1e-6))*1e-6
  stations=np.array(sorted(set([5.,10.,15.,20.,25.,30.,35.,40.,43.770,48.,52.130,56.,60.,float(w.config.coupled_substrate_z_mm)])))
  write(out/case/'controls.json',dict(cases=chosen,release_times_s=releases,station_z_mm=stations,realized_charges_sha256=hashlib.sha256(q0.tobytes()).hexdigest(),configuration=configuration(w.config)))
  runs=[(method,dt_us) for dt_us in DTS for method in ['euler','rk4']]+[('rk4',REFERENCE_DT)]
  for method,dt_us in runs:
   dest=out/case/method/f'{dt_us:g}us';write(out/'status.json',dict(status='running',case=case,method=method,dt_us=dt_us))
   if (dest/'completed.json').exists():timings.append(json.loads((dest/'completed.json').read_text()));continue
   cfg=replace(w.config,dt=dt_us*1e-6);started=time.perf_counter();p=prepare_integrator(cfg,device,output_mode='endpoints',method=method,release_times_s=releases,sample_stations_mm=stations);q,m=p.inputs(q0,m0);prepare_s=time.perf_counter()-started
   t=time.perf_counter();executable=p.batch_kernel.lower(q,m).compile();compile_s=time.perf_counter()-t
   t=time.perf_counter();raw=executable(q,m);jax.block_until_ready(raw);execution_s=time.perf_counter()-t
   t=time.perf_counter();host=jax.device_get(raw);transfer_s=time.perf_counter()-t
   f=w.decode(host,ids);f['method']=method;f['dt_us']=dt_us;f['case_label']=f.context_index.map({int(i):r['label'] for i,r in zip(ids,chosen)});dest.mkdir(parents=True,exist_ok=True);f.to_csv(dest/'positions.csv',index=False)
   final,diag=host;rows=[]
   for b,index in enumerate(ids):
    for node in range(cfg.particles_y_dim):
     owners=np.flatnonzero(np.asarray(final[10][b,:,node]));owner=int(owners[0]) if len(owners) else -1
     singleton=bool(owner>=0 and np.asarray(final[10][b,owner]).sum()==1)
     for k,z in enumerate(stations):
      time_s=float(diag['station_time_s'][b,node,k]);pos=np.asarray(diag['station_position_m'][b,node,k])*1000;vel=np.asarray(diag['station_velocity_m_s'][b,node,k])
      rows.append(dict(case=case,context_index=int(index),case_label=chosen[b]['label'],node_id=node,logical_drop=int(w.logical[node]),stroke_index=node//w.per,requested_current=bool(m0[b,node]),singleton_final=singleton,method=method,dt_us=dt_us,station_z_mm=z,time_s=time_s,x_mm=float(pos[0]),y_mm=float(pos[1]),z_mm=float(pos[2]),paper_horizontal_mm=float(pos[1]-w.speed/60*1000*time_s),vx_m_s=float(vel[0]),vy_m_s=float(vel[1]),vz_m_s=float(vel[2]),mass_kg=float(diag['station_mass_kg'][b,node,k])))
   pd.DataFrame(rows).to_csv(dest/'stations.csv',index=False)
   record=dict(case=case,method=method,dt_us=dt_us,contexts=len(ids),prepare_s=prepare_s,compile_lowering_s=compile_s,first_execution_s=execution_s,transfer_s=transfer_s,particle_dtype=str(q.dtype),output_bytes=sum(v.nbytes for v in jax.tree.leaves(host)),timing_scope='Single synchronized execution after explicit compilation; not a warmed throughput benchmark',native_steps_executed=np.asarray(diag['steps_executed']).tolist())
   write(dest/'completed.json',record);timings.append(record);pd.DataFrame(timings).to_csv(out/'execution_costs.csv',index=False)
   print(case,method,dt_us,'us complete, execution',round(execution_s,2),'s',flush=True)
  summaries.extend(compare_case(out,case));pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
 analytic(out,'cpu')
 write(out/'completed.json',dict(status='complete',representative_contexts_per_case=4,continuous_methods=['velocity-first Euler','classical RK4'],common_event_policy=True,releases_frozen_across_steps=True,finest_solution_is_a_numerical_reference_not_exact_truth=True))
 write(out/'status.json',dict(status='complete'));freeze(out)
 print(pd.DataFrame(summaries).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--device',choices=['cpu','gpu'],default='gpu');a=p.parse_args();main(a.output,a.device)
