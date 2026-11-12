from pathlib import Path
import os,sys,argparse,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vdcr.framework_workload import Workload,OUT
from vdcr.framework_campaign import Runner

def main(case,batch,device,smoke=False):
 w=Workload(case);dest=OUT/(f'{case}_smoke' if smoke else f'{case}_complete_pool');dest.mkdir(parents=True,exist_ok=True)
 (dest/'workload_manifest.json').write_text(json.dumps(w.manifest,indent=2)+'\n')
 if (dest/'completed.json').exists():print((dest/'completed.json').read_text());return
 r=Runner(w,device,batch)
 indices=[w.pair_index[(0,1)],w.pair_index[(0,65535)],w.pair_index[(65535,65535)]] if smoke else None
 r.run(w.full_commands,indices=indices,folder=dest,label='smoke_only' if smoke else 'unchanged_deployed_table')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--case',default='low');p.add_argument('--batch',type=int,default=64);p.add_argument('--device',default='gpu');a=p.parse_args();main(a.case,a.batch,a.device,a.smoke)
