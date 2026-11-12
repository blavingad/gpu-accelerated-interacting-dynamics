"""Recompute paper endpoint errors from frozen measured and native-simulation coordinates."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from vdcr import videojet_font as font
from vdcr.experimental_data import assessment_file

def main(native=False,pool_root=None,output=None,verify=True):
 out=Path(output) if output else ROOT/'results/generated'/('native_assessment' if native else '');out.mkdir(parents=True,exist_ok=True);parts=[];summary=[];glyph_rows=[]
 mapping={};offset=0
 for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
  width=len(font.glyph_to_columns(ch,vertical_flip=True,column_scale=1))
  mapping.update({c:ch for c in range(offset,offset+width)});offset+=width+2
 for case,label in [('low','A'),('high','B')]:
  inp=ROOT/'inputs/cases'/case;m=pd.read_csv(inp/'measured_coordinates.csv');s=pd.read_csv(inp/'simulated_coordinates.csv')
  if native:
   pool=pd.read_csv(Path(pool_root or ROOT/'results/generated')/f'{case}_complete_pool/positions.csv')
   anchor=pool[(pool.previous_mask==0)&(pool.current_mask==65535)]
   if len(anchor)!=16:raise ValueError('Missing blank/full phase anchor')
   phase=anchor.paper_horizontal_unregistered_mm.mean()
   ref=pool.rename(columns={'current_mask':'mask','landing_mm':'native_landing_mm'})[['previous_mask','mask','drop','native_landing_mm','paper_horizontal_unregistered_mm']]
   s=s.drop(columns=['paper_horizontal_unregistered_mm'],errors='ignore').merge(ref,on=['previous_mask','mask','drop'],how='left',validate='many_to_one')
   if s.native_landing_mm.isna().any():raise ValueError('Generated native pool misses a required measured context')
   s['landing_mm']=s.native_landing_mm
   s['substrate_horizontal_mm']=s.column_origin_mm+s.paper_horizontal_unregistered_mm-phase
  keys=['line','column_index','drop']
  assert not m.duplicated(keys).any() and not s.duplicated(keys).any()
  j=m.merge(s,on=keys,suffixes=('_measured','_simulated'),validate='one_to_one');assert len(j)==len(m)==len(s)==1725
  f=j[keys].copy();f['configuration']=label
  for axis,col in [('horizontal','substrate_horizontal_mm'),('deflection','landing_mm')]:
   for tag in ['measured','simulated']:f[tag+'_'+axis+'_mm']=j[col+'_'+tag]
  f['glyph']=f.column_index.map(mapping);assert f.glyph.notna().all()
  for axis,outaxis in [('horizontal','horizontal'),('deflection','vertical')]:f['raw_'+outaxis+'_residual_um']=(f['measured_'+axis+'_mm']-f['simulated_'+axis+'_mm'])*1000
  f['raw_radial_residual_um']=np.hypot(f.raw_horizontal_residual_um,f.raw_vertical_residual_um)
  def metrics(g):return dict(dots=len(g),horizontal_RMSE_um=float(np.sqrt(np.mean(g.raw_horizontal_residual_um**2))),vertical_RMSE_um=float(np.sqrt(np.mean(g.raw_vertical_residual_um**2))),radial_RMSE_um=float(np.sqrt(np.mean(g.raw_radial_residual_um**2))),radial_p95_um=float(np.percentile(g.raw_radial_residual_um,95)))
  summary.append(dict(configuration=label,case=case,**metrics(f)))
  for ch,g in f.groupby('glyph'):glyph_rows.append(dict(configuration=label,glyph=ch,**metrics(g)))
  parts.append(f)
 pd.concat(parts,ignore_index=True).to_csv(out/'experimental_paired_dots_by_glyph.csv',index=False)
 pd.DataFrame(summary).to_csv(out/'physical_coordinate_metrics.csv',index=False);pd.DataFrame(glyph_rows).to_csv(out/'experimental_per_glyph_metrics.csv',index=False)
 expected=pd.read_csv(assessment_file('experimental_paired_dots_by_glyph.csv')) if verify else pd.concat(parts).copy()
 actual=pd.concat(parts).merge(expected,on=['configuration',*keys],suffixes=('_new','_reference'),validate='one_to_one')
 for col in ['raw_horizontal_residual_um','raw_vertical_residual_um']:
  if not native:assert np.max(abs(actual[col+'_new']-actual[col+'_reference']))<1e-8
 print(pd.DataFrame(summary).to_string(index=False));print('Native-pool endpoints assessed with a simulation-only phase anchor.' if native else 'All paired residuals reproduce the frozen paper assessment.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--pool-root');p.add_argument('--output');a=p.parse_args();main(a.native,a.pool_root,a.output)
