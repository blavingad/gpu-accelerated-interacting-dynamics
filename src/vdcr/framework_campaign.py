"""Batched finite-history pool characterization with explicit evaluation/acceptance coverage."""
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd,jax
from .jax_solver import prepare_integrator
from .framework_workload import OUT
ROOT=Path(__file__).resolve().parents[2]
POLICY=json.loads((ROOT/'inputs/quality_policy.json').read_text())

def context_metrics(frame):
 rows=[]
 for index,g in frame.groupby('context_index',sort=True):
  e=(g.landing_mm-g.target_mm)*1000;finite=np.isfinite(e)
  valid=bool(g.landed.all() and not g.merged.any() and finite.all() and np.isfinite(g.gutter_mm).all())
  rms=float(np.sqrt(np.mean(e[finite]**2))) if finite.any() else np.nan
  mx=float(abs(e[finite]).max()) if finite.any() else np.nan
  gutter=float(g.gutter_mm.min()*1000) if np.isfinite(g.gutter_mm).all() else np.nan
  passed=valid and rms<=POLICY['RMS_limit_um'] and mx<=POLICY['maximum_dot_error_limit_um'] and gutter>=POLICY['minimum_gutter_deflection_um']
  rows.append(dict(context_index=int(index),previous_mask=int(g.previous_mask.iloc[0]),current_mask=int(g.current_mask.iloc[0]),dots=len(g),valid=valid,accepted=bool(passed),RMS_um=rms,max_error_um=mx,minimum_gutter_um=gutter,missing=int((~g.landed).sum()),merged=int(g.merged.sum()),strict_diagnostic_accepted=bool(valid and rms<=10 and mx<=25 and gutter>=500)))
 return pd.DataFrame(rows)

class Runner:
 def __init__(self,w,device='gpu',batch=64,integrator='euler'):
  self.w,self.batch,self.integrator=w,batch,integrator
  self.prepared=prepare_integrator(w.config,device,output_mode='endpoints',method=integrator)
  ids=np.repeat(w.pair_index[(65535,65535)],batch)
  q,m=self.prepared.inputs(w.charges(w.full_commands,ids),w.eligible[ids])
  start=time.perf_counter();self.executable=self.prepared.batch_kernel.lower(q,m).compile();self.compile_s=time.perf_counter()-start
 def run(self,table,indices=None,folder=None,label='fixed_table'):
  w=self.w;indices=np.arange(len(w.pairs)) if indices is None else np.asarray(indices,int)
  frames=[];times=[];start=time.perf_counter()
  if folder is not None:folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  for begin in range(0,len(indices),self.batch):
   ids=indices[begin:begin+self.batch];padded=np.pad(ids,(0,self.batch-len(ids)),constant_values=int(ids[-1]))
   t=time.perf_counter();q,m=self.prepared.inputs(w.charges(table,padded),w.eligible[padded]);input_s=time.perf_counter()-t
   t=time.perf_counter();raw=self.executable(q,m);jax.block_until_ready(raw);execute_s=time.perf_counter()-t
   t=time.perf_counter();host=jax.device_get(raw);transfer_s=time.perf_counter()-t
   f=w.decode(host,ids);frames.append(f)
   joined=pd.concat(frames,ignore_index=True);cm=context_metrics(joined)
   row=dict(evaluated_contexts=len(cm),accepted_contexts=int(cm.accepted.sum()),required_contexts=len(indices),evaluation_coverage=len(cm)/len(indices),acceptance_coverage=float(cm.accepted.sum()/len(indices)),wall_s=time.perf_counter()-start,upload_and_actuation_s=input_s,kernel_s=execute_s,download_s=transfer_s,padded_cases=self.batch-len(ids))
   times.append(row)
   if folder is not None:
    pd.DataFrame(times).to_csv(folder/'coverage_vs_time.csv',index=False)
    (folder/'status.json').write_text(json.dumps(dict(status='running',case=w.case,label=label,**row),indent=2)+'\n')
   print(w.case,label,len(cm),'/',len(indices),'accepted',int(cm.accepted.sum()),'batch_s',round(execute_s,3),flush=True)
  f=pd.concat(frames,ignore_index=True);cm=context_metrics(f)
  result=dict(integrator=self.integrator,case=w.case,label=label,contexts=len(cm),dots=len(f),accepted=int(cm.accepted.sum()),strict_diagnostic_accepted=int(cm.strict_diagnostic_accepted.sum()),evaluation_coverage=len(cm)/len(indices),acceptance_coverage=float(cm.accepted.mean()),pooled_vertical_RMS_um=float(np.sqrt(np.mean(((f.landing_mm-f.target_mm).dropna()*1000)**2))),worst_valid_context_RMS_um=float(cm.loc[cm.valid,'RMS_um'].max()) if cm.valid.any() else None,maximum_valid_dot_error_um=float(cm.loc[cm.valid,'max_error_um'].max()) if cm.valid.any() else None,missing=int(cm.missing.sum()),merged=int(cm.merged.sum()),minimum_gutter_um=float(cm.minimum_gutter_um.min()),wall_s=time.perf_counter()-start,compile_s=self.compile_s,batch=self.batch,device=str(self.prepared.device),policy=POLICY,table_sha256=hashlib.sha256(np.asarray(table,dtype=np.float64).tobytes()).hexdigest(),model_sha256=hashlib.sha256((ROOT/'src/vdcr/jax_solver.py').read_bytes()).hexdigest(),padding_is_not_counted_as_evaluated_contexts=True)
  if folder is not None:
   f.to_csv(folder/'positions.csv',index=False);cm.to_csv(folder/'context_metrics.csv',index=False)
   (folder/'completed.json').write_text(json.dumps(result,indent=2)+'\n');(folder/'status.json').write_text(json.dumps(dict(status='complete',**result),indent=2)+'\n')
  return f,cm,result
