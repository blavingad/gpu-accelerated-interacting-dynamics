"""Render manuscript figures from the reference results."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vdcr.paper_style import apply,BLUE,TEAL,ORANGE,SLATE
from vdcr.experimental_data import assessment_file
PAPER=ROOT/'paper';OUT=ROOT/'results/reference'
def read(p):return json.loads(Path(p).read_text())
def main():
 from vdcr.coordinate_print_comparison import build
 apply();build()
 figures=ROOT/'results/generated/figures';figures.mkdir(parents=True,exist_ok=True)
 apply()
 modelhash=hashlib.sha256((ROOT/'tests/fixtures/paper_euler_solver.py').read_bytes()).hexdigest()
 bench=pd.DataFrame([read(p) for p in sorted(OUT.glob('benchmark_*.json')) if read(p)['model_sha256']==modelhash])
 bench.to_csv(ROOT/'results/generated/benchmark_summary.csv',index=False)
 pools={case:read(OUT/f'{case}_complete_pool/completed.json') for case in ['low','high']}
 assert all(v['contexts']==644 and v['evaluation_coverage']==1 for v in pools.values())
 assert all(v['model_sha256']==modelhash for v in pools.values())
 covpath=OUT/'coverage_size_experiment/summary.csv'
 coverage=pd.read_csv(covpath) if covpath.exists() else pd.DataFrame()
 complete=len(coverage)==3 and (OUT/'public_synthetic/analytic_verification.csv').exists()
 if len(coverage):
  assert coverage.model_sha256.eq(modelhash).all()
  assert coverage.contexts.eq(644).all()
 if not complete:raise RuntimeError('Missing reference coverage or analytic-verification results')
 def save(fig,name):
  fig.savefig(figures/f'{name}.png',bbox_inches='tight');fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(13,2.3));ax.axis('off')
 labels=['Printer inputs\n+ requirements','Font + history\ncontext pool','New structure\n+ charge fitting','Shared dependencies\n+ integer replay','Controller export\n+ qualification']
 nodes=[]
 centers=np.linspace(.08,.92,len(labels))
 for x,t in zip(centers,labels):
  nodes.append(ax.text(x,.58,t,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.55',fc='#e5eff7',ec='#4c7b9d'),fontsize=9))
 fig.canvas.draw()
 for i in range(len(nodes)-1):
  ax.annotate('',xy=(centers[i+1],.58),xytext=(centers[i],.58),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#31556e',patchA=nodes[i].get_bbox_patch(),patchB=nodes[i+1].get_bbox_patch(),shrinkA=7,shrinkB=7))
 ax.text(.5,.05,'Version geometry, physics, commands, coverage, quality policy and provenance at every stage',ha='center',transform=ax.transAxes,color='#31556e',fontsize=9)
 save(fig,'digital_development_workflow')
 fig,axs=plt.subplots(1,2,figsize=(10,3.4))
 b=bench[bench['mode'].eq('endpoints') & bench['case'].eq('low')].sort_values(['batch','platform'])
 names=[f"{r.platform.upper()} / batch {r.batch}" for r in b.itertuples()]
 axs[0].barh(names,b.warm_cases_per_s,color=[BLUE if p=='gpu' else SLATE for p in b.platform]);axs[0].set_xlabel('Warmed contexts / second');axs[0].set_title('Matched physics; execution synchronized')
 for case,color in [('low',BLUE),('high',TEAL)]:
  f=pd.read_csv(OUT/f'{case}_complete_pool/coverage_vs_time.csv');axs[1].plot(f.wall_s+pools[case]['compile_s'],f.evaluated_contexts,label=f'{case.capitalize()} speed',color=color,marker='o',ms=3)
 axs[1].axhline(644,color='#9caab5',ls=':',lw=1);axs[1].set(xlabel='Elapsed seconds, including compilation',ylabel='Distinct contexts evaluated',title='Complete declared-pool assessment');axs[1].legend(frameon=False)
 fig.tight_layout();save(fig,'framework_throughput_coverage')
 fig,ax=plt.subplots(figsize=(10,3.5))
 g=pd.read_csv(assessment_file('experimental_per_glyph_metrics.csv'))
 metric='vertical_RMSE_um'
 casecol='configuration';glyphcol=next(c for c in g.columns if c in ['glyph','character'])
 letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ');x=np.arange(len(letters));width=.36
 for offset,case,color,label in [(-width/2,'A',BLUE,'Low speed'),(width/2,'B',TEAL,'High speed')]:
  q=g[g[casecol].eq(case)].set_index(glyphcol).reindex(letters)
  ax.bar(x+offset,q[metric],width=width,label=label,color=color)
 ax.set(xticks=x,xticklabels=letters,ylabel='Deflection RMSE [µm]',title='Per-letter agreement with the ten-print mean',ylim=(0,float(g[metric].max())*1.15))
 ax.set_xlabel('Printed character');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.45)
 ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False);fig.tight_layout();save(fig,'framework_glyph_assessment')
 search=OUT/'structure_search/evaluations.csv'
 if search.exists():
  f=pd.read_csv(search);ok=f['valid'].astype(str).str.lower().eq('true')
  fig,ax=plt.subplots(figsize=(7,3.5));ax.scatter(f.loc[ok,'horizontal_RMS_um'],f.loc[ok,'vertical_RMS_um'],s=9,alpha=.35,color=BLUE,label='Valid coupled evaluations');ax.set(xlabel='Horizontal RMS [µm]',ylabel='Vertical RMS [µm]',title='Archived coupled structural exploration');ax.set_ylim(bottom=0);ax.legend(loc='upper right',frameon=False,markerscale=2);fig.tight_layout();save(fig,'framework_structure_exploration')
 if len(coverage):
  fig,axs=plt.subplots(1,2,figsize=(9,3.4));labels=coverage.level.str.replace('_',' ')
  axs[0].bar(labels,coverage.development_key_coverage*100,color=BLUE);axs[0].set(ylabel='Required lookup keys covered [%]',ylim=(0,105));axs[1].bar(labels,coverage.maximum_valid_dot_error_um,color=TEAL);axs[1].set_ylabel('Maximum valid dot error on full pool [µm]')
  for ax in axs:ax.tick_params(axis='x',rotation=20)
  fig.suptitle('Common seed and refit passes; total simulation cost varies',fontsize=11,y=.98);fig.tight_layout(rect=(0,0,1,.88),w_pad=2.5);save(fig,'framework_coverage_intervention')
 cpu=bench[(bench['case']=='low')&(bench['platform']=='cpu')&(bench['mode']=='endpoints')&(bench['batch']==64)].iloc[0]
 gpu=bench[(bench['case']=='low')&(bench['platform']=='gpu')&(bench['mode']=='endpoints')&(bench['batch']==64)].iloc[0]
 speedup=cpu.warm_median_s/gpu.warm_median_s

 # Coordinate residuals without fitted alignment.
 f=pd.read_csv(assessment_file('experimental_paired_dots_by_glyph.csv'));fig,axs=plt.subplots(1,3,figsize=(10,3.3))
 for case,label,color in [('A','Low speed',BLUE),('B','High speed',TEAL)]:
  g=f[f.configuration.eq(case)]
  for ax,column in zip(axs,['raw_horizontal_residual_um','raw_vertical_residual_um','raw_radial_residual_um']):ax.hist(g[column],bins=40,density=True,histtype='step',lw=1.4,label=label,color=color)
 for ax,label in zip(axs,['Horizontal residual [µm]','Deflection residual [µm]','Radial residual [µm]']):ax.set_xlabel(label);ax.set_ylabel('Density')
 axs[0].legend(frameon=False);fig.tight_layout();save(fig,'experimental_residual_distributions')
 # Analytical closure response.
 fig,axs=plt.subplots(1,2,figsize=(10,3.4));L=np.linspace(2,100,300);R=1+L*np.tan(np.deg2rad(10));r=132.29171023896175e-6/2;U=24.192;Re=1.225*U*2*r/1.81e-5;Cd=24/Re*(1+.15*Re**.687)
 for rr in [0,.5,.9]:axs[0].plot(L,np.minimum(.25*.5*Cd/R**2,.12)*np.exp(-4.605*rr**2),label=f'radial / wake radius = {rr:g}')
 axs[0].set(xlabel='Trailing separation / source radius',ylabel='Wake-induced air speed / source speed');axs[0].legend(fontsize=8,frameon=False);axs[0].grid(alpha=.2)
 Q=np.linspace(0,5,100)
 for area in [5,10,20]:axs[1].plot(Q,.5*(Q*1e-3/60)/(area*1e-6),label=f'Effective area {area} mm²')
 axs[1].set(xlabel='Positive-air flow [L/min]',ylabel='Prescribed mean coflow [m/s]');axs[1].legend(fontsize=8,frameon=False);axs[1].grid(alpha=.2);fig.tight_layout();save(fig,'aerodynamic_closures')
 print('Paper figures regenerated from frozen evidence.')
if __name__=='__main__':main()
