"""Declared previous/current pool and unchanged printer-table waveform realizations."""
from pathlib import Path
from dataclasses import replace,asdict
import json,hashlib
import numpy as np,pandas as pd
from .config import SimulationConfig
from .charge_explorer import ChargeParameters
from .charging_sequence import run_sequence
from . import videojet_font as font
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated'

class Workload:
 def __init__(self,case):
  self.case=case
  if case not in ('low','high'):raise ValueError(case)
  base=ROOT/'inputs/cases'/case
  self.lookup=json.loads((base/'charge_lookup.json').read_text())
  cfg=json.loads((base/'physical_configuration.json').read_text())
  cfg['openfoam_field_path']=str(ROOT/'inputs/electric_field.npz')
  params=json.loads((base/'charge_parameters.json').read_text())
  table_path=base/'printer_commands.csv'
  records=pd.read_csv(table_path);dac='dac' if 'dac' in records.columns else 'DAC'
  self.table={(int(r['drop']),int(r['address'])):int(r[dac]) for _,r in records.iterrows()}
  self.first,self.pitch=(1.8076272010803223,.3969) if case=='low' else (float(self.lookup['target_first_mm']),float(self.lookup['target_pitch_mm']))
  self.speed=50. if case=='low' else 100.
  self.seq=self.lookup['physical_sequence'];self.per=len(self.seq)
  self.defs={int(d):[(int(v['stroke_offset']),int(v['drop'])) for v in values] for d,values in self.lookup['dependencies'].items()}
  self.pairs=sorted(map(tuple,self.lookup['previous_current_pairs']))
  assert len(self.pairs)==644
  self.pair_index={pair:i for i,pair in enumerate(self.pairs)}
  self.keys=sorted({(d,self.address(p,c,d)) for p,c in self.pairs for d in range(1,17) if c>>(d-1)&1} | {(d,self.address(0,p,d)) for p,c in self.pairs for d in range(1,17) if p>>(d-1)&1})
  self.key_index={key:i for i,key in enumerate(self.keys)}
  self.full_commands=np.array([self.table[key] for key in self.keys],float)
  self.logical=np.tile([-1 if v=='guard' else int(v) for v in self.seq],2)
  self.target=self.first+np.arange(16)*self.pitch
  self.nodes=np.array([self.per+self.seq.index(d) for d in range(1,17)])
  self.indices=np.full((len(self.pairs),2*self.per),-1,int)
  self.eligible=np.zeros_like(self.indices,bool)
  for index,(p,c) in enumerate(self.pairs):
   for stroke,(prior,current) in enumerate([(0,p),(p,c)]):
    for slot,v in enumerate(self.seq):
     if v!='guard' and current>>(int(v)-1)&1:
      self.indices[index,stroke*self.per+slot]=self.key_index[int(v),self.address(prior,current,int(v))]
      if stroke==1:self.eligible[index,stroke*self.per+slot]=True
  self.params=replace(ChargeParameters(**params),frequency_hz=75000.)
  # The fixed zero-offset, unsaturated charging equations are linear in commands.
  # Form the exact response matrix; verify against the full sequential routine.
  assert self.params.voltage_offset_v==0 and self.params.voltage_curvature_v==0
  h=[]
  for node in range(2*self.per):
   u=np.zeros(2*self.per);u[node]=1.
   w=run_sequence(u,np.ones(2*self.per,bool),self.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),breakup_phase=.5)
   h.append([r['assigned_charge_pC'] for r in w['trace']])
  self.response=np.array(h).T
  cfg.update(nozzle_frequency_hz=75000.,wall_velocity_y=self.speed/60,particles_y_dim=2*self.per,raster_repeat_count=2,raster_droplets_per_repeat=self.per)
  cfg['droplet_charges_pC']=self.charges(self.full_commands,[0])[0]
  cfg['intended_print_mask']=self.eligible[0]
  self.config=SimulationConfig(**cfg);self.config.validate()
  residuals=[]
  for index in np.linspace(0,len(self.pairs)-1,8,dtype=int):
   u=self.events(self.full_commands,[index])[0]
   active=u!=0
   expected=run_sequence(u,active,self.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),breakup_phase=.5)
   actual=self.response@u
   residuals.append(float(np.max(abs(actual-np.array([r['assigned_charge_pC'] for r in expected['trace']])))))
  assert max(residuals)<1e-11
  self.manifest=dict(case=case,table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),nozzle_frequency_hz=75000.,speed_m_min=self.speed,contexts=len(self.pairs),current_only_contexts=sum(p==0 for p,c in self.pairs),nonzero_previous_contexts=sum(p!=0 for p,c in self.pairs),distinct_current_masks=len({c for p,c in self.pairs}),fitted_keys=len(self.keys),full_address_capacity=sum(1<<len(v) for v in self.defs.values()),charging_matrix_max_residual_pC=max(residuals),history='one preceding stroke emitted; older history blank; following stroke not emitted',guard_compensation_commands=False,consecutive_compensation_commands=False,parameters=asdict(self.params))
 def address(self,p,c,d):
  a=0
  for offset,i in self.defs[d]:a=(a<<1)|(((p if offset==-1 else c)>>(i-1))&1)
  return a
 def events(self,table,indices=None):
  select=self.indices if indices is None else self.indices[np.asarray(indices)]
  return np.where(select>=0,np.asarray(table)[np.maximum(select,0)],0.)
 def charges(self,table,indices=None):return self.events(table,indices)@self.response.T
 def selected_pairs(self,characters,baselines=True):
  pairs=set()
  for character in characters:
   cols=font.glyph_to_columns(character,vertical_flip=True,column_scale=1)
   sequence=[0]+cols+[0,0]
   pairs.update((int(p),int(c)) for p,c in zip(sequence,sequence[1:]) if c)
  masks={c for p,c in pairs}|{p for p,c in pairs if p}
  pairs|={(0,m) for m in masks}
  if baselines:pairs|={(0,1<<i) for i in range(16)}|{(0,65535)}
  assert pairs<=set(self.pairs)
  return np.array([self.pair_index[p] for p in sorted(pairs)],int)
 def decode(self,raw,indices):
  final,diag=raw
  arrays=[np.asarray(v) for v in final];diag={k:np.asarray(v) for k,v in diag.items()}
  rows=[]
  for b,index in enumerate(indices):
   p,c=self.pairs[index]
   for d,node in enumerate(self.nodes,1):
    if not c>>(d-1)&1:continue
    owners=np.flatnonzero(arrays[10][b,:,node]);owner=int(owners[0]) if len(owners) else -1
    landed=bool(owner>=0 and arrays[3][b,owner]);merged=bool(owner>=0 and arrays[10][b,owner].sum()>1)
    rows.append(dict(case=self.case,context_index=int(index),previous_mask=p,current_mask=c,drop=d,key_index=int(self.indices[index,node]),landed=landed,merged=merged,
      landing_mm=float(arrays[5][b,owner,0]*1000) if landed else np.nan,
      paper_horizontal_unregistered_mm=float(arrays[5][b,owner,1]*1000+self.speed/60*1000*self.per/75000) if landed else np.nan,
      target_mm=float(self.target[d-1]),gutter_mm=float(diag['gutter_x_mm'][b,node]),landing_time_s=float(diag['landing_time_s'][b,owner]) if landed else np.nan))
  return pd.DataFrame(rows)
