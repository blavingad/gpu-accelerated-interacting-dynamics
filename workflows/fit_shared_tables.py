"""Coverage-size experiment: identical isolated seed and bounded refit across font subsets.

Each fit starts from isolated-drop commands and targets the ideal geometry.
The deployed table supplies a reference comparison. Trial and coverage records
include the additional computational cost of larger development sets.
"""
from pathlib import Path
import os,sys,json,time,argparse
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,jax
from vdcr.framework_workload import Workload,OUT
from vdcr.framework_campaign import Runner,context_metrics

def isolated_seed(w,runner,folder):
 if (folder/'isolated_seed.csv').exists():
  f=pd.read_csv(folder/'isolated_seed.csv');return f.DAC.to_numpy(),f.slope_mm_per_DAC.to_numpy()
 ids=np.array([w.pair_index[(0,1<<j)] for j in range(16)])
 u=np.linspace(230,800,16);slopes=np.full(16,.008)
 for iteration in range(4):
  table=np.array([u[d-1] for d,a in w.keys]);plus=np.array([min(u[d-1]+4,1023) for d,a in w.keys])
  q0=w.charges(table,ids);q1=w.charges(plus,ids)
  allq=np.concatenate([q0,q1],axis=0);mask=np.concatenate([w.eligible[ids],w.eligible[ids]],axis=0)
  allq=np.pad(allq,((0,runner.batch-len(allq)),(0,0)),mode='edge');mask=np.pad(mask,((0,runner.batch-len(mask)),(0,0)),mode='edge')
  q,m=runner.prepared.inputs(allq,mask);raw=runner.executable(q,m);jax.block_until_ready(raw);host=jax.device_get(raw)
  x=np.array([float(host[0][5][i,w.nodes[i],0])*1000 for i in range(16)])
  xp=np.array([float(host[0][5][16+i,w.nodes[i],0])*1000 for i in range(16)])
  landed=np.array([bool(host[0][3][i,w.nodes[i]]) for i in range(16)])
  if not landed.all():raise RuntimeError('Isolated baseline did not land; cannot fabricate a slope')
  slopes=(xp-x)/np.minimum(4,1023-u)
  if np.any(slopes<=1e-6):raise RuntimeError('Nonpositive isolated slope')
  pd.DataFrame(dict(drop=np.arange(1,17),DAC=u,landing_mm=x,target_mm=w.target,slope_mm_per_DAC=slopes)).to_csv(folder/f'isolated_iteration_{iteration}.csv',index=False)
  u=np.clip(u+np.clip((w.target-x)/slopes,-80,80),230,1023)
 pd.DataFrame(dict(drop=np.arange(1,17),DAC=u,slope_mm_per_DAC=slopes)).to_csv(folder/'isolated_seed.csv',index=False)
 return u,slopes

def main(epochs=4,batch=64,case='low',device='gpu'):
 if batch<32:raise ValueError('batch must be at least 32 for isolated sensitivity probes')
 w=Workload(case);folder=OUT/('coverage_size_experiment' if case=='low' else 'coverage_size_experiment_high');folder.mkdir(parents=True,exist_ok=True)
 manifest=dict(case=case,levels=['digit_9','three_glyphs_AMO','all_printable_ASCII'],epochs=epochs,optimizer='simultaneous shared-table residual correction using isolated finite-difference slopes; bounded steps and full-model line search',maximum_update_DAC=12,seed='isolated-flight bounded Newton solves from nominal per-row DAC ramp; no full-font optimized entries used',target='ideal geometric rows, not observed paper data or full-table outputs',comparison_scope='coverage-size intervention with equal refit passes; costs vary with context count; not an equal-total-budget optimizer comparison',history='previous and current stroke interact; older history blank, no following emission',quality_policy=json.loads((ROOT/'inputs/quality_policy.json').read_text()))
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 runner=Runner(w,device,batch);seed,slopes=isolated_seed(w,runner,folder)
 results=[]
 levels=[('digit_9','9'),('three_glyphs_AMO','AMO'),('all_printable_ASCII',''.join(chr(i) for i in range(32,127)))]
 for label,characters in levels:
  dest=folder/label;dest.mkdir(exist_ok=True)
  if (dest/'completed.json').exists():results.append(json.loads((dest/'completed.json').read_text()));continue
  ids=w.selected_pairs(characters);training=set(map(int,ids));seen_keys=set(w.indices[ids][w.indices[ids]>=0].tolist())
  table=np.array([seed[d-1] for d,a in w.keys]);start=time.perf_counter();trace=[]
  for epoch in range(epochs):
   f,cm,rec=runner.run(table,ids,folder=dest/f'epoch_{epoch}_baseline',label=label+'_fit')
   e=f.landing_mm-f.target_mm
   correction=(-(e/np.array([slopes[int(d)-1] for d in f['drop']]))).groupby(f.key_index).mean()
   delta=np.zeros_like(table)
   for key,v in correction.items():
    if np.isfinite(v):delta[int(key)]=np.clip(v,-12,12)
   basecost=float(np.mean(e[np.isfinite(e)]**2))
   accepted=False
   for scale in [1.,.5,.25]:
    proposed=np.clip(table+scale*delta,230,1023)
    pf,pc,pr=runner.run(proposed,ids,folder=dest/f'epoch_{epoch}_step_{scale:g}',label=label+'_line_search')
    pe=pf.landing_mm-pf.target_mm;cost=float(np.mean(pe[np.isfinite(pe)]**2))
    ok=pr['minimum_gutter_um']>=500 and np.isfinite(cost) and cost<basecost and pr['missing']<=rec['missing'] and pr['merged']<=rec['merged']
    trace.append(dict(epoch=epoch,scale=scale,accepted=bool(ok),cost=cost,baseline_cost=basecost,contexts=len(ids),missing=pr['missing'],merged=pr['merged']))
    pd.DataFrame(trace).to_csv(dest/'optimization_trace.csv',index=False)
    if ok:table=proposed;accepted=True;break
   if not accepted:break
  fitting_wall=time.perf_counter()-start
  pd.DataFrame([dict(drop=d,address=a,DAC=float(v),seen_in_development=(i in seen_keys)) for i,((d,a),v) in enumerate(zip(w.keys,table))]).to_csv(dest/'candidate_commands.csv',index=False)
  allf,allcm,summary=runner.run(table,folder=dest/'all_required_contexts',label=label+'_frozen_evaluation')
  unseen_rows=0;unsupported_contexts=0
  for index in range(len(w.pairs)):
   keys=set(w.indices[index][w.indices[index]>=0].tolist());unseen_rows+=len(keys-seen_keys);unsupported_contexts+=bool(keys-seen_keys)
  allcm['used_for_development']=allcm.context_index.isin(training)
  allcm.to_csv(dest/'all_contexts_with_split.csv',index=False)
  held=allcm[~allcm.used_for_development]
  summary.update(level=label,development_contexts=len(ids),development_key_coverage=len(seen_keys)/len(w.keys),populated_development_keys=len(seen_keys),required_keys=len(w.keys),contexts_using_unseen_keys=unsupported_contexts,held_out_contexts=len(held),held_out_accepted=int(held.accepted.sum()),fitting_wall_s=fitting_wall,epochs_completed=len({r['epoch'] for r in trace}),unknown_key_policy='isolated per-row command retained; not nearest-neighbor imputation')
  (dest/'completed.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary)
  pd.DataFrame(results).to_csv(folder/'summary.csv',index=False)
 print(pd.DataFrame(results)[['level','development_contexts','development_key_coverage','accepted','contexts','worst_valid_context_RMS_um','maximum_valid_dot_error_um','fitting_wall_s']].to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--case',choices=['low','high'],default='low');p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch',type=int,default=64);a=p.parse_args();main(a.epochs,a.batch,a.case,a.device)
