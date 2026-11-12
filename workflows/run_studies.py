"""Run the integration, benchmark and manuscript workflows sequentially."""
from pathlib import Path
import sys,os,time,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
from _study_common import STUDY,write

def run(label,args):
 write(STUDY/'pipeline_status.json',dict(status='running',stage=label,pid=os.getpid()))
 print('STAGE',label,flush=True);subprocess.run([sys.executable,*args],cwd=ROOT,env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false'),check=True)
def main():
 start=time.time()
 if (STUDY/'publication.json').exists():
  from build_manuscript import check_frozen
  for name in ['ablation','integrators','benchmarks']:check_frozen(STUDY/name)
  print('Completed evidence verified; use build_manuscript.py to rebuild manuscript.');return
 while not (STUDY/'ablation/frozen.json').exists():
  write(STUDY/'pipeline_status.json',dict(status='running',stage='waiting_for_fixed_command_ablation',pid=os.getpid()))
  if time.time()-start>3600:raise RuntimeError('Ablation evidence did not finish; no manuscript publication performed')
  time.sleep(5)
 run('integrator_comparison',['workflows/run_integrator_comparison.py'])
 for device in ['cpu','gpu']:
  run('matched_benchmark_'+device,['workflows/benchmark_engine.py','--case','low','--device',device,'--batch','64','--repeat','2','--output-dir',str(STUDY/'benchmarks')])
 run('freeze_and_publish',['workflows/build_manuscript.py'])
 write(STUDY/'pipeline_status.json',dict(status='complete',stage='paper_and_package_ready',pid=os.getpid()))
if __name__=='__main__':
 try:main()
 except Exception as e:write(STUDY/'pipeline_status.json',dict(status='failed',error=repr(e),pid=os.getpid()));raise
