"""Traceability helpers shared by the predeclared paper studies."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,os,platform,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/numerical'

def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),default=serial,allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def write(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,default=serial)+'\n');tmp.replace(path)
def configuration(cfg):
 d=asdict(cfg);d['openfoam_field_path']='inputs/electric_field.npz';return json.loads(canonical(d))
def sources(extra=()):
 names=['src/vdcr/jax_solver.py','src/vdcr/config.py','src/vdcr/electric_field.py','src/vdcr/charging.py','src/vdcr/charging_sequence.py','src/vdcr/charge_explorer.py','src/vdcr/framework_workload.py','src/vdcr/framework_campaign.py','workflows/_study_common.py',*extra]
 return {n:sha(ROOT/n) for n in names}
def inputs(case):
 names=['inputs/electric_field.npz','inputs/quality_policy.json',f'inputs/cases/{case}/physical_configuration.json',f'inputs/cases/{case}/charge_parameters.json',f'inputs/cases/{case}/printer_commands.csv',f'inputs/cases/{case}/charge_lookup.json']
 return {n:sha(ROOT/n) for n in names}
def declare(folder,plan):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);fingerprint=digest(plan);path=folder/'study_plan.json'
 if path.exists():
  old=json.loads(path.read_text())
  if old['fingerprint']!=fingerprint:raise ValueError(f'Frozen study specification changed: {folder}; use a new output directory')
 else:write(path,dict(fingerprint=fingerprint,declared_utc=datetime.now(timezone.utc).isoformat(),**plan))
 return fingerprint
def environment():
 import jax,jaxlib,importlib.metadata
 commands={}
 for name,cmd in [('cpu',['lscpu']),('gpu',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])]:
  try:commands[name]=subprocess.check_output(cmd,text=True).strip()
  except (OSError,subprocess.CalledProcessError) as e:commands[name]=str(e)
 return dict(platform=platform.platform(),python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','jax','jaxlib']},jax_devices=[str(d) for d in jax.devices()],jax_x64_enabled=bool(jax.config.jax_enable_x64),particle_precision='float32',waveform_precision='NumPy float64 before conversion to particle float32',thread_policy='Installed backend defaults; no CPU thread pinning',environment={k:os.environ.get(k) for k in ['XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE','OMP_NUM_THREADS']},hardware=commands)
def freeze(folder):
 folder=Path(folder);files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ['frozen.json','status.json'] and '.tmp' not in p.name}
 path=folder/'frozen.json'
 if path.exists():
  if json.loads(path.read_text())['files']!=files:raise ValueError('Previously frozen numerical evidence changed')
 else:write(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files=files,interpretation='Numerical evidence frozen before manuscript conclusions are revised'))
 return files
