"""Update measured coordinates and rescore cached flights."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows')]
import numpy as np,pandas as pd
from assess_prints import main as assess
from run_physics_ablation import measured_score,VARIANTS
from vdcr.framework_workload import Workload
from _study_common import sha,write,freeze

KEYS=['line','column_index','drop']
COORDINATES=['landing_mm','substrate_horizontal_mm']

def main(source,output):
    source=Path(source);out=Path(output)
    if not out.is_absolute():out=ROOT/out
    out=out.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must be inside the repository.')
    if out.exists():raise ValueError('Choose an empty output directory.')
    incoming=pd.read_csv(source)
    high=ROOT/'inputs/cases/high';old=high/'measured_coordinates.csv'
    current=pd.read_csv(old);sim=pd.read_csv(high/'simulated_coordinates.csv')
    for frame in [incoming,current,sim]:
        if len(frame)!=1725 or frame.duplicated(KEYS).any():raise ValueError('Expected 1,725 unique measured identities.')
        if not np.isfinite(frame[COORDINATES].to_numpy()).all():raise ValueError('Nonfinite coordinates.')
    j=incoming.merge(sim,on=KEYS,suffixes=('_m','_s'),validate='one_to_one')
    if len(j)!=1725:raise ValueError('Measurement/simulation identities differ.')
    for col in ['previous_mask','mask','current_mask','address','reference_frequency_hz']:
        if not (j[col+'_m']==j[col+'_s']).all():raise ValueError('Mismatched context or operating metadata: '+col)
    lowpath=ROOT/'inputs/cases/low/measured_coordinates.csv';lowsha=sha(lowpath)
    archive=high/'history'/f'measurements_{sha(old)[:12]}.csv';archive.parent.mkdir(exist_ok=True)
    if archive.exists() and sha(archive)!=sha(old):raise ValueError('Measurement archive hash collision.')
    oldsha=sha(old);shutil.copyfile(old,archive);shutil.copyfile(source,old)
    out.mkdir(parents=True)
    assess(output=out,verify=False)
    physical=[]
    for case in ['low','high']:
        w=Workload(case)
        for variant in VARIANTS:
            positions=ROOT/f'results/numerical/ablation/{case}/{variant}/positions.csv'
            paired,score=measured_score(w,pd.read_csv(positions),case,variant)
            dest=out/'ablation'/case/variant;dest.mkdir(parents=True)
            paired.to_csv(dest/'measured_pairs.csv',index=False);physical.append(score)
    pd.DataFrame(physical).to_csv(out/'experimental_ablation_summary.csv',index=False)
    paired=pd.read_csv(out/'experimental_paired_dots_by_glyph.csv')
    rows=[]
    for case,g in paired.groupby('configuration'):
        for axis,col in [('horizontal','raw_horizontal_residual_um'),('deflection','raw_vertical_residual_um'),('radial','raw_radial_residual_um')]:
            e=g[col].to_numpy();rows.append(dict(configuration=case,axis=axis,dots=len(g),mean_um=float(e.mean()),RMS_um=float(np.sqrt(np.mean(e**2))),max_absolute_um=float(abs(e).max()),p95_absolute_um=float(np.percentile(abs(e),95))))
    pd.DataFrame(rows).to_csv(out/'directional_metrics.csv',index=False)
    oldpairs=pd.read_csv(ROOT/'results/reference/experimental_paired_dots_by_glyph.csv')
    a=paired[paired.configuration=='A'].sort_values(KEYS).reset_index(drop=True)
    b=oldpairs[oldpairs.configuration=='A'].sort_values(KEYS).reset_index(drop=True)
    for col in ['measured_horizontal_mm','measured_deflection_mm','simulated_horizontal_mm','simulated_deflection_mm']:
        np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-12)
    assert sha(lowpath)==lowsha
    provenance=dict(assessment=out.name,changed_case='high',coordinate_columns=COORDINATES,source_sha256=sha(source),previous_high_sha256=oldsha,previous_high_archive=str(archive.relative_to(ROOT)),low_measured_sha256=lowsha,low_unchanged=True,simulation_files={c:sha(ROOT/f'inputs/cases/{c}/simulated_coordinates.csv') for c in ['low','high']},ablation_predictions='Unchanged frozen results/numerical/ablation/{case}/{variant}/positions.csv',ablation_freeze_sha256=sha(ROOT/'results/numerical/ablation/frozen.json'),simulation_refitted=False,registration_fitted=False,physical_simulation_rerun=False,experimental_protocol='Previously supplied protocol: corresponding dot means over ten A-Z prints per raster',noncoordinate_fields='Sequence/controller/flight metadata retained from supplied schema; only designated coordinate columns are treated as experimental positions',assessment_source_sha256=sha(ROOT/'workflows/assess_prints.py'),update_source_sha256=sha(Path(__file__)))
    write(out/'provenance.json',provenance);freeze(out)
    write(ROOT/'inputs/experimental_assessment.json',dict(results_directory=str(out.relative_to(ROOT)),provenance_sha256=sha(out/'provenance.json'),frozen_manifest_sha256=sha(out/'frozen.json')))
    print('Current experimental assessment:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--high',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.high,a.output)
