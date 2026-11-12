"""Targeted event-step sensitivity, not a substitute for field/grid convergence."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dataclasses import replace
import numpy as np,pandas as pd,jax
from vdcr.framework_workload import Workload,OUT
from vdcr.jax_solver import prepare_integrator

def main():
 dest=OUT/'time_step_probe';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completed.json').exists():return
 w=Workload('high');pairs=[(0,26622),(8191,12480),(0,65535),(65535,65535)];ids=np.array([w.pair_index[p] for p in pairs]);rows=[]
 for dt in [1e-6,.5e-6,.25e-6]:
  config=replace(w.config,dt=dt);p=prepare_integrator(config,'gpu',output_mode='endpoints');q,m=p.inputs(w.charges(w.full_commands,ids),w.eligible[ids]);raw=p.batch_kernel(q,m);jax.block_until_ready(raw)
  f=w.decode(jax.device_get(raw),ids);f['dt_us']=dt*1e6;rows.append(f)
 frame=pd.concat(rows,ignore_index=True);frame.to_csv(dest/'positions.csv',index=False)
 key=['previous_mask','current_mask','drop'];ref=frame[frame.dt_us.eq(.25)][key+['landing_mm','landing_time_s']].rename(columns={'landing_mm':'reference_landing_mm','landing_time_s':'reference_landing_time_s'})
 joined=frame.merge(ref,on=key,validate='many_to_one');joined['difference_to_finest_um']=(joined.landing_mm-joined.reference_landing_mm)*1000
 summary=joined.groupby('dt_us').agg(dots=('drop','size'),maximum_difference_to_finest_um=('difference_to_finest_um',lambda v:float(abs(v).max())),RMS_difference_to_finest_um=('difference_to_finest_um',lambda v:float(np.sqrt(np.mean(v**2))))).reset_index()
 summary.to_csv(dest/'summary.csv',index=False)
 (dest/'completed.json').write_text(json.dumps(dict(contexts=len(pairs),scope='two event-sensitive contexts and two reference contexts; unchanged commands; release steps re-quantized at each dt; finest is a numerical reference, not exact truth',timesteps_us=[1.,.5,.25],summary=summary.to_dict('records')),indent=2)+'\n')
 print(summary.to_string(index=False))
if __name__=='__main__':main()
