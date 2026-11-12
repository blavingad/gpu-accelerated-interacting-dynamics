"""Run one declared context; optionally record every native force/state array."""
from pathlib import Path
from dataclasses import asdict
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from vdcr.framework_workload import Workload,OUT
from vdcr.framework_campaign import Runner
from vdcr.jax_solver import prepare_integrator

def main(case,previous,current,device,record):
 w=Workload(case);pair=(previous,current)
 if pair not in w.pair_index:raise ValueError('Context is outside the declared font/history pool; use the model API for a new experiment')
 i=w.pair_index[pair];folder=OUT/'single_context'/case/f'{previous:04X}_{current:04X}';folder.mkdir(parents=True,exist_ok=True)
 if not record:Runner(w,device,batch=1).run(w.full_commands,indices=[i],folder=folder,label='single_context')
 else:
  p=prepare_integrator(w.config,device,output_mode='full');data,platform,backend=p.run(w.charges(w.full_commands,[i])[0],w.eligible[i])
  np.savez_compressed(folder/'native_history.npz',**data)
  (folder/'recording.json').write_text(json.dumps(dict(case=case,previous_mask=previous,current_mask=current,dt_s=w.config.dt,platform=platform,device=backend,stored_keys=sorted(data),units='Native solver SI arrays; see model equations and key definitions',warning='Dense histories can exceed one GB per context'),indent=2)+'\n')
 print(folder)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['low','high'],default='high');p.add_argument('--previous',type=lambda v:int(v,0),default=0);p.add_argument('--current',type=lambda v:int(v,0),default=65535);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--record',action='store_true');a=p.parse_args();main(a.case,a.previous,a.current,a.device,a.record)
