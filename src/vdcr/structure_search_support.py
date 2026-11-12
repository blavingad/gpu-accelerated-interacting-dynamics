"""Coupled-flight evaluation and joint charge optimization for firing-order search.
"""
from pathlib import Path
from types import SimpleNamespace
import sys,os,json,time,hashlib
import numpy as np,pandas as pd
from scipy.optimize import lsq_linear
from .config import SimulationConfig
from .charge_explorer import ChargeParameters
from .charging_sequence import run_sequence
from .flight_reporting import run_print_simulation
from .flight_context import landing_by_current_drop,current_coalesced_print_drops
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated/structure_search'
DEVICE='cpu'
_SETTINGS=json.loads((ROOT/'inputs/structure_search/model.json').read_text())
FREQUENCY=_SETTINGS['charge_parameters']['frequency_hz']
SPEED=_SETTINGS['physical_configuration']['wall_velocity_y']*60
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def status(**value):atomic(OUT/'status.json',value)
def publish():pass
class Pilot:
 def __init__(self):
  src=json.loads((ROOT/'inputs/structure_search/model.json').read_text())
  self.source=src;self.config=src['physical_configuration'];self.config['openfoam_field_path']=str(ROOT/'inputs/electric_field.npz')
  self.params=ChargeParameters(**src['charge_parameters'])
  self.target=np.array(src['target_mm']);self.qiso=np.array(src['isolated_charge_pC']);self.qfull=np.array(src['full_stroke_charge_pC'])
  self.capture=SimpleNamespace(last_case_id=None,wave_context={})
 def waveform(self,order,q):
  target=-np.tile(np.asarray(q)[np.array(order)-1],2)
  wave=run_sequence(np.full(32,400.),np.ones(32,bool),self.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),target_printed_charges_pC=target,breakup_phase=.5)
  u=np.array([r['effective_dac'] for r in wave['trace']])
  if np.any(u<230-1e-8) or np.any(u>1023+1e-8):raise ValueError('Command bounds')
  return wave,u
 def evaluate(self,name,order,q,label,forced=None):
  try:wave,u=self.waveform(order,q) if forced is None else (forced,np.array([r['effective_dac'] for r in forced['trace']]))
  except ValueError:return None,None
  actual=np.array([r['assigned_charge_pC'] for r in wave['trace']]);key=hashlib.sha256(actual.tobytes()+str(order).encode()).hexdigest()[:24];folder=OUT/name/'flights'/key;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'record.json').exists():return json.loads((folder/'record.json').read_text()),pd.read_csv(folder/'positions.csv')
  status(status='running',candidate=name,phase=label)
  cfg=SimulationConfig(**self.config);cfg.droplet_charges_pC=actual;cfg.intended_print_mask=np.r_[np.zeros(16,bool),np.ones(16,bool)];cfg.validate()
  traces=[dict(stroke=i//16,physical_cycle_in_stroke=i%16+1,**r) for i,r in enumerate(wave['trace'])]
  self.capture.wave_context=dict(raster='no_guard_'+name,mask=65535,previous_mask=65535,sequence=order,commands=u[16:].tolist(),event_charge_pC=actual,charging=traces)
  result=run_print_simulation(cfg,device=DEVICE,proximity_diagnostics=True,trajectory_path=folder/'trajectory.npz')
  logical=np.tile(order,2);stroke=np.repeat([0,1],16);lands=landing_by_current_drop(result,logical,stroke,cfg.intended_print_mask);merged=current_coalesced_print_drops(result,logical,stroke)
  gutter={int(logical[r['event_id']]):r['x_gutter_mm'] for r in result.gutter_crossings}
  frame=pd.DataFrame([dict(drop=i,firing_slot=order.index(i)+1,charge_magnitude_pC=float(-actual[16+order.index(i)]),continuous_DAC=float(u[16+order.index(i)]),target_mm=self.target[i-1],landing_mm=lands[i]['x_mm'],paper_y_mm=lands[i]['y_mm'],gutter_mm=gutter.get(i,np.nan),landed=lands[i]['landed'],merged=i in merged) for i in range(1,17)])
  valid=bool(frame.landed.all() and not frame.merged.any() and np.isfinite(frame.gutter_mm).all() and not result.coalescence_events);e=frame.landing_mm.to_numpy()-self.target;minimum=float(frame.gutter_mm.min()) if np.isfinite(frame.gutter_mm).all() else 0.
  yy=frame.paper_y_mm.to_numpy();centred=yy-np.nanmean(yy);x=np.arange(16)-7.5;shear=float(centred@x/(x@x));residual=centred-shear*x
  r=dict(candidate=name,sequence=order,label=label,key=key,valid=valid,safe=bool(valid and minimum>=.5),cost=float(np.mean(e*e)) if valid else 1e3,vertical_RMS_um=float(np.sqrt(np.mean(e*e))*1000) if valid else None,vertical_max_um=float(abs(e).max()*1000) if valid else None,horizontal_span_um=float(np.ptp(yy)*1000) if frame.landed.all() else None,horizontal_RMS_um=float(np.std(yy)*1000) if frame.landed.all() else None,shear_um_per_row=shear*1000,residual_after_global_shear_RMS_um=float(np.sqrt(np.mean(residual*residual))*1000),minimum_gutter_um=minimum*1000,merge_events=len(result.coalescence_events),missing=int((~frame.landed).sum()),charge=np.asarray(q).tolist(),DAC_current=u[16:].tolist(),trace_case_id=self.capture.last_case_id)
  frame['error_um']=e*1000;frame.to_csv(folder/'positions.csv',index=False);pd.DataFrame(traces).to_csv(folder/'charging.csv',index=False)
  pd.DataFrame([dict(time_ms=m.time*1000,x_mm=float(m.position[0]*1000),z_mm=float(m.position[2]*1000),lineage=' '.join(str(i+1) for i in m.lineage)) for m in result.coalescence_events],columns=['time_ms','x_mm','z_mm','lineage']).to_csv(folder/'merges.csv',index=False)
  atomic(folder/'record.json',r);print(name,label,'valid',valid,'RMS',r['vertical_RMS_um'],'horizontal span',r['horizontal_span_um'],'merges',r['merge_events'],flush=True);return r,frame
 def fit(self,name,order):
  dest=OUT/name
  if (dest/'completed.json').exists():return json.loads((dest/'completed.json').read_text())
  trials=[]
  for label,q in [('previous_full_fit_seed',self.qfull),('isolated_seed',self.qiso),('scaled_full_up',self.qfull*1.04),('scaled_full_down',self.qfull*.96)]:
   r,f=self.evaluate(name,order,q,label)
   if r:trials.append((r,f,np.array(q)))
  usable=[v for v in trials if v[0]['safe']]
  if not usable:
   atomic(dest/'completed.json',dict(candidate=name,sequence=order,status='no_nonmerging_seed_found',seeds=[r for r,f,q in trials],impossibility_proved=False));return json.loads((dest/'completed.json').read_text())
  best,frame,q=min(usable,key=lambda v:v[0]['cost']);J=None;history=[]
  for it in range(10):
   if best['vertical_RMS_um']<=3 and best['vertical_max_um']<=8:break
   if J is None or it in [4,8]:
    columns=[];failed=False
    for j in range(16):
     h=.005*q[j];plus=q.copy();minus=q.copy();plus[j]+=h;minus[j]-=h
     rp,fp=self.evaluate(name,order,plus,f'matrix{it}_drop{j+1}_plus');rm,fm=self.evaluate(name,order,minus,f'matrix{it}_drop{j+1}_minus')
     if rp and rm and rp['valid'] and rm['valid']:col=(fp.landing_mm.to_numpy()-fm.landing_mm.to_numpy())/(2*h)
     elif rp and rp['valid']:col=(fp.landing_mm.to_numpy()-frame.landing_mm.to_numpy())/h
     elif rm and rm['valid']:col=(frame.landing_mm.to_numpy()-fm.landing_mm.to_numpy())/h
     else:failed=True;break
     columns.append(col)
    if failed:break
    J=np.column_stack(columns);np.save(dest/f'jacobian_{it}.npy',J)
   error=frame.landing_mm.to_numpy()-self.target;delta=lsq_linear(np.vstack([J,1e-6*np.eye(16)]),np.r_[-error,np.zeros(16)],bounds=(np.maximum(.01-q,-.04*q),np.minimum(4.-q,.04*q)),tol=1e-10,max_iter=100).x
   improved=False
   for scale in [1.,.5,.25,.125]:
    trial=q+scale*delta;r,f=self.evaluate(name,order,trial,f'iteration{it}_step{scale}')
    if r and r['safe'] and r['cost']<best['cost']:
     dq=trial-q;dx=f.landing_mm.to_numpy()-frame.landing_mm.to_numpy()
     if dq@dq>1e-15:J+=np.outer(dx-J@dq,dq)/(dq@dq)
     q,best,frame=trial,r,f;improved=True;break
   history.append(dict(iteration=it,accepted=improved,vertical_RMS_um=best['vertical_RMS_um'],horizontal_span_um=best['horizontal_span_um']));atomic(dest/'history.json',history)
   if not improved:break
  wave,u=self.waveform(order,q);rounded=np.floor(u+.5).clip(230,1023).astype(int)
  # One fixed row DAC repeated across both strokes; aggregate both charging histories.
  row_commands={i:int(np.floor(np.mean([u[order.index(i)],u[16+order.index(i)]])+.5)) for i in range(1,17)}
  physical=np.tile([row_commands[i] for i in order],2)
  replay_wave=run_sequence(physical,np.ones(32,bool),self.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),breakup_phase=.5)
  replay,rf=self.evaluate(name,order,q,'fixed_integer_DAC_replay',forced=replay_wave)
  atomic(dest/'completed.json',dict(candidate=name,sequence=order,status='complete',best=best,integer_DAC_replay=replay,scope='two repeated full strokes only; no font or arbitrary histories',fixed_integer_row_commands=row_commands));publish();return json.loads((dest/'completed.json').read_text())
