"""Charge-first shared dependency fitting with sparse bounded least squares and secants.

Fits charge targets against geometric objectives. Controller deployment requires
command realization followed by integer-waveform replay.
"""
from pathlib import Path
from types import MethodType
import sys,os,json,argparse
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,jax
from scipy.sparse import coo_matrix,vstack,eye
from scipy.optimize import lsq_linear
from vdcr.framework_workload import Workload,OUT
from vdcr.framework_campaign import Runner
from vdcr.charging_sequence import run_sequence

def realized_charges(w,table,indices=None):
 ids=np.arange(len(w.pairs)) if indices is None else np.asarray(indices,int)
 result=[]
 for index in ids:
  keys=w.indices[index];on=keys>=0;targets=-np.asarray(table)[np.maximum(keys,0)]
  wave=run_sequence(np.where(on,400.,0.),on,w.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),target_printed_charges_pC=targets,breakup_phase=.5)
  commands=np.array([r['effective_dac'] for r in wave['trace']])
  if np.any(commands[on]<230-1e-7) or np.any(commands[on]>1023+1e-7):raise ValueError('Target charges violate printed-command bounds')
  result.append([r['assigned_charge_pC'] for r in wave['trace']])
 return np.asarray(result)

def main(case='low',epochs=8,batch=64,device='gpu',seed='reference'):
 if batch<32:raise ValueError('batch must be at least 32 for the isolated sensitivity probes')
 w=Workload(case);folder=OUT/('shared_charge_fit_'+case);folder.mkdir(parents=True,exist_ok=True)
 table=np.array([w.lookup['entries'][str(d)][a] for d,a in w.keys],float)
 # Keep baseline tables frozen in inputs; generated optimization never edits them.
 if seed=='isolated':
  per_row={i+1:table[w.key_index[i+1,w.address(0,1<<i,i+1)]] for i in range(16)}
  table=np.array([per_row[d] for d,a in w.keys])
 w.charges=MethodType(realized_charges,w);w.full_commands=table.copy()
 runner=Runner(w,device,batch)
 iso=np.array([w.pair_index[0,1<<i] for i in range(16)])
 plus=table*1.005;ids=np.r_[iso,iso];q=np.r_[w.charges(table,iso),w.charges(plus,iso)]
 q=np.pad(q,((0,batch-len(q)),(0,0)),mode='edge');mask=np.pad(w.eligible[ids],((0,batch-len(ids)),(0,0)),mode='edge')
 qa,ma=runner.prepared.inputs(q,mask);raw=runner.executable(qa,ma);jax.block_until_ready(raw);host=jax.device_get(raw)
 x=np.array([host[0][5][i,w.nodes[i%16],0]*1000 for i in range(32)])
 iso_q=np.array([table[w.key_index[i+1,w.address(0,1<<i,i+1)]] for i in range(16)])
 slopes=(x[16:]-x[:16])/(.005*iso_q)
 if not np.isfinite(slopes).all() or np.any(slopes<=0):raise ValueError('Invalid isolated sensitivities')
 states={};history=[]
 for epoch in range(epochs):
  frame,metrics,summary=runner.run(table,folder=folder/f'epoch_{epoch}',label='shared_net_charge')
  rr=[];cc=[];vv=[];errors=[];offset=0
  for index,g in frame.groupby('context_index',sort=True):
   index=int(index);variables=np.unique(w.indices[index][w.indices[index]>=0]);local={k:j for j,k in enumerate(variables)}
   g=g.sort_values('drop');x=g.landing_mm.to_numpy();target=g.target_mm.to_numpy();q=table[variables];J=np.zeros((len(g),len(variables)))
   for j,r in enumerate(g.itertuples()):J[j,local[r.key_index]]=slopes[r.drop-1]
   valid=bool(g.landed.all() and not g.merged.any() and np.isfinite(x).all())
   old=states.get(index)
   if old is not None:
    J=old['J'].copy();dq=q-old['q']
    if valid and old['valid'] and dq@dq>1e-14:J+=np.outer(x-old['x']-J@dq,dq)/(dq@dq)
   states[index]=dict(J=J,q=q.copy(),x=x.copy(),valid=valid)
   if not valid:raise RuntimeError('Invalid coupled flight; no fabricated residuals admitted')
   weight=1/np.sqrt(len(g));a,b=np.indices(J.shape);rr.extend((offset+a.ravel()).tolist());cc.extend(variables[b.ravel()].tolist());vv.extend((weight*J).ravel().tolist());errors.extend((weight*(x-target)).tolist());offset+=len(g)
  cost=float(np.sum(np.asarray(errors)**2)/len(w.pairs));history.append(dict(epoch=epoch,cost=cost,**summary));pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
  if summary['accepted']==len(w.pairs) or epoch==epochs-1:break
  J=coo_matrix((vv,(rr,cc)),shape=(offset,len(table))).tocsr()
  delta=lsq_linear(vstack([J,1e-5*eye(len(table),format='csr')]),np.r_[-np.asarray(errors),np.zeros(len(table))],bounds=(np.maximum(.01-table,-.06),np.minimum(3.5-table,.06)),tol=1e-9,lsmr_tol='auto',max_iter=100).x
  accepted=False
  for fraction in [1.,.5,.25,.125,.0625]:
   proposed=table+fraction*delta
   try:w.charges(proposed)
   except ValueError:continue
   pf,pm,ps=runner.run(proposed,folder=folder/f'epoch_{epoch}_trial_{fraction}',label='charge_line_search')
   objective=float(np.mean([np.mean(((g.landing_mm-g.target_mm).to_numpy())**2) for _,g in pf.groupby('context_index')]))
   if ps['missing']==0 and ps['merged']==0 and ps['minimum_gutter_um']>=500 and objective<cost:table=proposed;accepted=True;break
  if not accepted:break
 pd.DataFrame([dict(drop=d,address=a,charge_magnitude_pC=float(v)) for (d,a),v in zip(w.keys,table)]).to_csv(folder/'charge_lookup.csv',index=False)
 (folder/'status.json').write_text(json.dumps(dict(status='complete',units='net charge magnitude pC',printer_ready=False,controller_realization='fixed-command fitting and integer replay required',case=case,contexts=len(w.pairs)),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',choices=['reference','isolated'],default='reference');p.add_argument('--case',choices=['low','high'],default='low');p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch',type=int,default=64);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');a=p.parse_args();main(a.case,a.epochs,a.batch,a.device,a.seed)
