"""Full-context frozen-command in-flight Coulomb/wake ablation; no refitting."""
from pathlib import Path
from dataclasses import replace
import os,sys,json,argparse,hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from vdcr.framework_workload import Workload
from vdcr.framework_campaign import Runner
from _study_common import STUDY,configuration,declare,environment,sources,inputs,write,freeze
VARIANTS={'full':{},'no_coulomb':{'coulomb_interaction_enabled':False},'no_wakes':{'wake_initial_velocity_deficit_factor':0.},'neither':{'coulomb_interaction_enabled':False,'wake_initial_velocity_deficit_factor':0.}}

def measured_score(w,positions,case,variant):
 inp=ROOT/'inputs/cases'/case;m=pd.read_csv(inp/'measured_coordinates.csv');template=pd.read_csv(inp/'simulated_coordinates.csv')
 j=m.merge(template,on=['line','column_index','drop'],suffixes=('_measured','_template'),validate='one_to_one')
 ref=positions.rename(columns={'current_mask':'mask','landing_mm':'prediction_deflection_mm'})
 j=j.merge(ref[['previous_mask','mask','drop','prediction_deflection_mm','paper_horizontal_unregistered_mm','landed','merged']],left_on=['previous_mask_template','mask_template','drop'],right_on=['previous_mask','mask','drop'],how='left',validate='many_to_one')
 phase=template.horizontal_registration_mm.unique();assert len(phase)==1
 j['prediction_horizontal_mm']=j.column_origin_mm_template+j.paper_horizontal_unregistered_mm-float(phase[0])
 valid=j.landed.fillna(False)&~j.merged.fillna(True)&np.isfinite(j.prediction_deflection_mm)&np.isfinite(j.prediction_horizontal_mm)
 j['scoreable_singleton']=valid
 j['vertical_residual_um']=(j.landing_mm_measured-j.prediction_deflection_mm)*1000
 j['horizontal_residual_um']=(j.substrate_horizontal_mm_measured-j.prediction_horizontal_mm)*1000
 score=dict(case=case,variant=variant,requested_dots=len(j),scored_singletons=int(valid.sum()),missing_targets=int((~j.landed.fillna(False)).sum()),merged_targets=int(j.merged.fillna(False).sum()),horizontal_RMSE_um=float(np.sqrt(np.mean(j.loc[valid,'horizontal_residual_um']**2))) if valid.any() else None,vertical_RMSE_um=float(np.sqrt(np.mean(j.loc[valid,'vertical_residual_um']**2))) if valid.any() else None,phase_rule='Unchanged simulation-only phase from the frozen native paper reference; common across all ablations',registration_fitted_to_measurements=False)
 return j,score

def main(output=None,device='gpu',batch=64):
 out=Path(output or STUDY/'ablation')
 if (out/'completed.json').exists():freeze(out);print('Frozen ablation already complete');return
 models={c:Workload(c) for c in ['low','high']}
 plan=dict(study='fixed_command_flight_force_ablation_v1',cases=['low','high'],contexts_per_case=644,variants=VARIANTS,integrator='velocity-first Euler',timestep_s=1e-6,commands='Unchanged supplied printer integer lookup; no refitting',charging='Complete unchanged waveform and induction, including nonprinting cycles',device=device,batch=batch,precision='float32 particle integration; float64 waveform preparation',sources=sources(['workflows/run_physics_ablation.py']),inputs={c:inputs(c) for c in models},measured_inputs={c:hashlib.sha256((ROOT/f'inputs/cases/{c}/measured_coordinates.csv').read_bytes()).hexdigest() for c in models},metrics='Geometric error for finite lineage-mapped requested targets; missing/merged counts separately. Experimental RMSE only for identifiable landed singletons, with denominator reported.')
 declare(out,plan)
 assert all(w.config.dt==1e-6 and w.params.frequency_hz==75000 for w in models.values())
 write(out/'environment.json',environment());summaries=[];physical=[]
 for case,w in models.items():
  base=w.config;charge_sha=hashlib.sha256(w.charges(w.full_commands).tobytes()).hexdigest();basecfg=configuration(base)
  for variant,changes in VARIANTS.items():
   w.config=replace(base,**changes);cfg=configuration(w.config);diff={k for k in cfg if cfg[k]!=basecfg[k]}
   assert diff<=set(changes)
   assert hashlib.sha256(w.charges(w.full_commands).tobytes()).hexdigest()==charge_sha
   dest=out/case/variant;write(out/'status.json',dict(status='running',case=case,variant=variant))
   if not (dest/'completed.json').exists():
    write(dest/'configuration.json',cfg);write(dest/'control.json',dict(realized_charge_sha256=charge_sha,changed_fields=sorted(diff),commands_unchanged=True))
    runner=Runner(w,device,batch);runner.run(w.full_commands,folder=dest,label=variant)
   positions=pd.read_csv(dest/'positions.csv');result=json.loads((dest/'completed.json').read_text());assert result['contexts']==644
   summaries.append(dict(case=case,variant=variant,**{k:result[k] for k in ['contexts','dots','accepted','pooled_vertical_RMS_um','missing','merged','minimum_gutter_um','wall_s','compile_s']}))
   paired,score=measured_score(w,positions,case,variant);paired.to_csv(dest/'measured_pairs.csv',index=False);physical.append(score)
   pd.DataFrame(summaries).to_csv(out/'numerical_summary.csv',index=False);pd.DataFrame(physical).to_csv(out/'experimental_summary.csv',index=False)
  w.config=base
 write(out/'completed.json',dict(status='complete',evaluated_contexts=2*4*644,all_controller_charges_identical_across_variants=True,experimental_coordinates_refitted=False))
 write(out/'status.json',dict(status='complete'));freeze(out)
 print(pd.DataFrame(physical).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--batch',type=int,default=64);a=p.parse_args();main(a.output,a.device,a.batch)
