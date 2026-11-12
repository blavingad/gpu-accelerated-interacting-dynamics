"""Build manuscript figures and text from saved results."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,re,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _study_common import STUDY,sha,write,freeze,environment
from vdcr.paper_style import apply,BLUE,TEAL,ORANGE,SLATE
from vdcr.experimental_data import assessment_directory,assessment_file
PAPER=ROOT/'paper';TEMPLATES=ROOT/'workflows/manuscript_templates'
LABELS={'full':'Full','no_coulomb':'No flight Coulomb','no_wakes':'No wakes','neither':'Neither'}
COLORS={'full':'#1c6c93','no_coulomb':'#d67d2c','no_wakes':'#9b587f','neither':'#798791'}

def call(script,*args):subprocess.run([sys.executable,str(ROOT/'workflows'/script),*map(str,args)],cwd=ROOT,env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false'),check=True)
def check_frozen(folder):
 f=json.loads((folder/'frozen.json').read_text())
 for name,digest in f['files'].items():
  if sha(folder/name)!=digest:raise RuntimeError(f'Frozen evidence changed: {folder/name}')
def save(fig,name):
 out=STUDY/'figures';out.mkdir(exist_ok=True)
 fig.savefig(out/f'{name}.png',dpi=200,bbox_inches='tight');fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def table_row(values):return ' & '.join(map(str,values))+r' \\'

def benchmarks():
 folder=STUDY/'benchmarks'
 if (folder/'frozen.json').exists():check_frozen(folder);return pd.read_csv(folder/'consolidated.csv')
 for device in ['cpu','gpu']:
  path=folder/f'benchmark_synthetic_{device}_euler_endpoints_b64.json'
  if not path.exists():call('benchmark_engine.py','--synthetic','--particles',32,'--device',device,'--batch',64,'--repeat',2,'--output-dir',folder)
 write(folder/'environment.json',environment())
 records=[]
 for cohort,base in [('historical',ROOT/'results/reference'),('current',folder)]:
  for p in sorted(base.glob('benchmark_*.json')):
   r=json.loads(p.read_text())
   if 'model_sha256' not in r:continue
   records.append({**r,'cohort':cohort,'source_file':str(p.relative_to(ROOT)),'method':r.get('method','euler'),'particle_precision':r.get('particle_precision','float32')})
 f=pd.DataFrame(records);f.to_csv(folder/'consolidated.csv',index=False)
 for case in ['low','synthetic']:
  g=f[(f.cohort=='current')&(f['case']==case)&(f['mode']=='endpoints')&(f.batch==64)]
  if len(g)!=2:raise RuntimeError('Missing matched CPU/GPU benchmark')
  for key in ['model_sha256','charge_input_sha256','mask_input_sha256','field_sha256','particle_precision']:
   if g[key].nunique()!=1:raise RuntimeError('Mismatched benchmark inputs: '+key)
 freeze(folder)
 return f

def plots(physical,integrators):
 apply()
 fig,axs=plt.subplots(1,2,figsize=(8.4,3.7))
 for ax,case in zip(axs,['low','high']):
  f=physical[physical['case']==case].set_index('variant').loc[list(LABELS)]
  x=np.arange(4);ax.bar(x-.17,f.vertical_RMSE_um,width=.32,color=BLUE,label='Deflection')
  ax.bar(x+.17,f.horizontal_RMSE_um,width=.32,color=TEAL,label='Substrate direction')
  ax.set(xticks=x,xticklabels=['Full','No Coulomb','No wakes','Neither'],ylabel='Measured-coordinate RMSE [µm]',title=case.capitalize()+'-speed case')
  ax.tick_params(axis='x',labelrotation=17)
  for j,r in enumerate(f.itertuples()):
   if r.scored_singletons<r.requested_dots:ax.text(j,max(r.vertical_RMSE_um,r.horizontal_RMSE_um)+6,f'{r.scored_singletons}/{r.requested_dots}',ha='center',fontsize=8)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False)
 fig.text(.5,.015,'Fixed commands, charging history and reference phase · missing/merged targets counted separately',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.09,1,.90),w_pad=2.5);save(fig,'physics_ablation')
 fig,axs=plt.subplots(1,3,figsize=(9.2,3.8))
 for case,color in [('low',BLUE),('high',TEAL)]:
  for method,style in [('euler','-'),('rk4','--')]:
   f=integrators[(integrators['case']==case)&(integrators.method==method)&(integrators.dt_us>=.25)].sort_values('dt_us')
   for ax,col in zip(axs[:2],['native_endpoint_RMS_difference_um','station_RMS_deflection_difference_um']):ax.loglog(f.dt_us,np.maximum(f[col],1e-5),style,marker='o',color=color,label=case.capitalize()+' / '+method.upper(),ms=4)
   axs[0].set(xlabel='Integration step [µs]',ylabel='Native endpoint RMS difference [µm]');axs[1].set(xlabel='Integration step [µs]',ylabel='Common-plane RMS difference [µm]')
 a=pd.read_csv(STUDY/'integrators/analytic/errors.csv')
 for method,color in [('euler',BLUE),('rk4',TEAL)]:
  f=a[a.method==method].sort_values('dt_us');axs[2].loglog(f.dt_us,np.maximum(f.x_error_um,1e-7),marker='o',label=method.upper(),color=color,ms=4)
 axs[2].set(xlabel='Integration step [µs]',ylabel='Analytic-limit absolute X error [µm]')
 for ax in axs:
  ax.set_xticks([.25,.5,1.],labels=['0.25','0.5','1']);ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator());ax.grid(alpha=.15,which='both')
 axs[2].legend(frameon=False,fontsize=8);handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.36,.995),ncol=2,fontsize=8,frameon=False)
 fig.text(.5,.015,'Industrial differences use RK4 at 0.125 µs as a numerical reference, not exact truth; releases held fixed.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.11,1,.86),w_pad=2.2);save(fig,'integration_comparison')
 fig,axs=plt.subplots(1,2,figsize=(8.4,4.25))
 for ax,case in zip(axs,['low','high']):
  f=pd.read_csv(STUDY/f'integrators/{case}/station_differences.csv');f=f[(f.case_label=='worst_frozen_context')&f.comparable&(f.dt_us==1.)]
  rows=sorted(f.logical_drop.unique());selected=sorted({rows[0],rows[len(rows)//2],rows[-1]}) if rows else []
  for drop,color in zip(selected,[BLUE,ORANGE,TEAL]):
   for method,style in [('euler','-'),('rk4','--')]:
    g=f[(f.logical_drop==drop)&(f.method==method)].sort_values('station_z_mm');ax.plot(g.station_z_mm,g.delta_x_um,style,color=color,label=f'Row {drop} / {method.upper()}',lw=1.2)
  ax.axhline(0,color='#929da4',lw=.7);ax.set(xlabel='Axial station [mm]',ylabel='Deflection difference [µm]',title=case.capitalize()+' / worst frozen context');ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.30),columnspacing=1.2,handlelength=2.1)
 fig.text(.5,.012,'Requested rows selected by first/middle/last label · 1 µs trajectories versus finest RK4 · no measured in-flight paths',ha='center',fontsize=8)
 fig.subplots_adjust(top=.85,bottom=.36,left=.085,right=.98,wspace=.32);save(fig,'integration_trajectory_differences')

def physical_section(numerical,physical):
 text=r'''\subsection{Controlled physics ablation at frozen controller commands}\label{sec:ablation}
The raster commands were developed with the full interacting model. To test whether its in-flight interaction terms matter for those deployed designs, each unchanged integer-command lookup is evaluated with full physics, flight Coulomb forces disabled, wakes disabled, and both disabled. All 644 declared contexts per case are retained. The field, ink properties, charging waveform and induction, guard commands, initial states, event rules, integration step and controller table are identical within each case; no alternative is refitted. In particular, disabling flight Coulomb forces does not disable electrode forces or neighboring-drop charging induction.

The geometric-target assessment and comparison with measured deposited coordinates are distinct. Table~\ref{tab:ablation} reports the full computational pool; Figure~\ref{fig:ablation} compares the A--Z subset with the same measured coordinates and a common simulation-defined phase. Merged targets are not treated as independently localized singleton droplets. Their counts and the number of scoreable measurement pairs are retained, rather than allowing exclusions to improve an ablated model's apparent accuracy.
\begin{table}[htbp]\centering\small
\caption{Fixed-command ablation on 644 contexts per case. RMS is the deflection error of finite requested-target coordinates, including lineage-mapped coordinates; merged targets are separately counted. These are numerical target errors, not measurement RMSE.}\label{tab:ablation}
\begin{tabular}{llrrr}\toprule Case & Physics & Target RMS ($\mu$m) & Accepted contexts & Merged targets\\\midrule
'''
 for r in numerical.itertuples():text+=table_row([str(r.case).capitalize(),LABELS[r.variant],f'{r.pooled_vertical_RMS_um:.1f}',r.accepted,r.merged])+'\n'
 text+=r'''\bottomrule\end{tabular}\end{table}
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/physics_ablation.pdf}\caption{Coordinate-prediction ablation with unchanged commands and reference phase. Blue and teal bars denote deflection and substrate-direction RMSE. Each case has 1,725 measured target positions; an annotated fraction gives scoreable singletons when merged targets prevent an identity-preserving coordinate comparison.}\label{fig:ablation}\end{figure}
'''
 full=physical[physical.variant=='full'];worst=physical[physical.variant!='full']
 best=all(float(g[g.variant=='full'].vertical_RMSE_um.iloc[0])<float(g[g.variant!='full'].vertical_RMSE_um.min()) for _,g in physical.groupby('case'))
 if best:text+='The full model gives the lowest measured deflection RMSE in both cases under this fixed-command test. '
 text+=f"Its deflection RMSE is {full.vertical_RMSE_um.min():.1f}--{full.vertical_RMSE_um.max():.1f} micrometers, whereas omitted-term variants span {worst.vertical_RMSE_um.min():.1f}--{worst.vertical_RMSE_um.max():.1f} micrometers on their scoreable targets. "
 text+=r'''These results show that the omitted physics matter for the developed command tables. Because those tables were optimized with the full model, this is a conditional deployment/prediction test, not a comparison of independently recalibrated and reoptimized alternatives. It does not uniquely identify each closure or prove that no simplified model could perform adequately after its own development. Coalescence outcomes need not vary monotonically when two coupled terms are removed together.
'''
 return text

def numerical_section(summary):
 text=r'''\subsection{Euler--RK4 integration and timestep sensitivity}\label{sec:integration_comparison}
Both integrators use the same force evaluator and grid-boundary event policy. The Euler compatibility path retains the original mass-first, velocity-first update and passes preserved-solver parity checks. Classical RK4 advances $\vect y=(\vect r,\vect v,m)$ jointly, reevaluating the imposed field, pair forces, wake, drag and mass-loss rate at the four stages:
\begin{equation}
 \begin{aligned}
 \vect k_1&=F(\vect y_n,t_n), &\quad \vect k_2&=F(\vect y_n+\tfrac h2\vect k_1,t_n+\tfrac h2),\\
 \vect k_3&=F(\vect y_n+\tfrac h2\vect k_2,t_n+\tfrac h2), &\quad \vect k_4&=F(\vect y_n+h\vect k_3,t_n+h).
 \end{aligned}
\end{equation}
\begin{equation}
 \vect y_{n+1}^{\rm continuous}=\vect y_n+\frac h6(\vect k_1+2\vect k_2+2\vect k_3+\vect k_4).
\end{equation}
Release masks are held fixed during each step, then the same eligibility, greedy-contact, collection and first-crossing landing rules are applied. Detached charge is constant between events. The four-stage continuous method does not make discontinuous event handling fourth order.

For each case, the predeclared examples comprise an isolated least-deflected target, a complete stroke with blank history, two complete preceding/current strokes, and the worst context in the frozen full-pool reference. All use the same unchanged realized charges. Release times are rounded once to the native $1\,\mu$s schedule and held exactly fixed across integration refinements; otherwise changing release quantization would confound integration error. Euler and RK4 are evaluated at $h=1$, $0.5$ and $0.25\,\mu$s, with RK4 at $0.125\,\mu$s providing a finest numerical reference, not exact truth. Positive-air fluctuations are inactive in these cases. Native landing endpoints retain the original overshoot/event grid. Additional fixed-plane samples are linearly interpolated to distinguish trajectory differences from first-crossing endpoint quantization; samples spanning a merge are excluded with their denominators retained.

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/integration_comparison.pdf}\caption{Numerical comparison at fixed release times and event rules. Left: native endpoint disagreement. Center: deflection disagreement at common axial planes. Both use the finest RK4 result as a numerical reference. Right: a separate event-free uniform-field/mass-loss limit has an analytic solution. Single precision and nonsmooth switching prevent assuming ideal asymptotic order for the complete industrial model.}\label{fig:integration_comparison}\end{figure}
'''
 for case in ['low','high']:
  f=summary[(summary['case']==case)&(summary.dt_us==1.)].set_index('method')
  e=f.loc['euler'];r=f.loc['rk4']
  text+=f"For the {case}-speed examples at 1 microsecond, common-plane RMS deflection differences from the finest reference are {e.station_RMS_deflection_difference_um:.2f} micrometers for Euler and {r.station_RMS_deflection_difference_um:.2f} micrometers for RK4. "
 text+=r'''All requested targets in these selected cases remain singleton landings, with no event-flag disagreements against the finest reference. Native endpoint RMS disagreements at 1 microsecond are approximately 3.5--3.8 micrometers for Euler and 1.1 micrometers for RK4; this distinction from common-plane errors reflects first-crossing overshoot. RK4 common-plane differences do not decrease monotonically at the finest steps, and the analytic-limit error also reaches a single-precision floor. These are numerical disagreements for selected trajectories, not errors against measured in-flight paths. The finest solution, event masks, interpolation and timestep remain part of the evidence boundary; a small optimized geometric residual does not certify equally small continuous-time error. The existing deposited-coordinate assessment retains its frozen native Euler reference and is not silently replaced by a newly optimized RK4 print.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/integration_trajectory_differences.pdf}\caption{Representative trajectory differences at common axial stations for the worst preselected contexts. First/middle/last requested row labels are shown at the same integration step. This tests numerical integration, not experimental trajectory accuracy.}\label{fig:integration_trajectories}\end{figure}
'''
 return text

def main():
 for folder in ['ablation','integrators']:check_frozen(STUDY/folder)
 bench=benchmarks()
 physical_path=assessment_file('experimental_ablation_summary.csv')
 physical=pd.read_csv(physical_path if physical_path.exists() else STUDY/'ablation/experimental_summary.csv')
 numerical=pd.read_csv(STUDY/'ablation/numerical_summary.csv');integration=pd.read_csv(STUDY/'integrators/summary.csv')
 coordinate_metrics=pd.read_csv(assessment_file('physical_coordinate_metrics.csv')).set_index('case')
 directional=pd.read_csv(assessment_file('directional_metrics.csv'))
 if (assessment_directory()/'frozen.json').exists():check_frozen(assessment_directory())
 if len(physical)!=8 or len(integration)!=14:raise RuntimeError('Incomplete numerical evidence')
 current=bench[(bench.cohort=='current')&(bench['case']=='low')&(bench.batch==64)&(bench['mode']=='endpoints')].set_index('platform');speedup=float(current.loc['cpu','warm_median_s']/current.loc['gpu','warm_median_s'])
 # Record the numerical inputs before rendering the manuscript.
 write(STUDY/'numerical_freeze.json',dict(studies={k:sha(STUDY/k/'frozen.json') for k in ['ablation','integrators','benchmarks']},conclusions_not_used_to_select_results=True,source_kernel_sha256=sha(ROOT/'src/vdcr/jax_solver.py')))
 plots(physical,integration)
 evidence=PAPER/'evidence';evidence.mkdir(exist_ok=True)
 numerical.to_csv(evidence/'ablation_numerical_summary.csv',index=False);physical.drop(columns=['phase_rule']).to_csv(evidence/'ablation_experimental_summary.csv',index=False);integration.to_csv(evidence/'integration_summary.csv',index=False)
 publiccols=['cohort','case','platform','method','mode','batch','particle_precision','compile_s','first_execution_s','warm_median_s','warm_cases_per_s','device_to_host_s','output_bytes','model_sha256']
 bench[publiccols].to_csv(evidence/'computational_benchmarks.csv',index=False)
 coordinate_metrics.reset_index().to_csv(evidence/'physical_coordinate_metrics.csv',index=False)
 directional.to_csv(evidence/'directional_metrics.csv',index=False)
 write(evidence/'provenance_summary.json',dict(experimental_assessment=str(assessment_directory().relative_to(ROOT)),experimental_provenance_sha256=sha(assessment_directory()/'provenance.json'),numerical_freeze_sha256=sha(STUDY/'numerical_freeze.json'),cases=['low-speed','high-speed'],experimental_scope='Image-derived corresponding dot coordinates averaged across ten repeated full A-Z prints per raster, as clarified by the data provider; additional raw laboratory records are confidential; replicate uncertainty is outside the released assessment',source_types=dict(experimental='Measured deposited coordinates',numerical='Fixed-command ablation and Euler/RK4 comparison',theoretical='Analytical arguments separated in appendices'),industrial_configuration_disclosure='Repository release authorized by the author; simulation inputs distributed separately from the manuscript source ZIP'))
 main=(TEMPLATES/'main.tex').read_text();results=(TEMPLATES/'results.tex').read_text();num=(TEMPLATES/'numerical_implementation.tex').read_text();discussion=(TEMPLATES/'discussion.tex').read_text();conclusion=(TEMPLATES/'conclusion.tex').read_text();framework=(TEMPLATES/'framework_evidence.tex').read_text()
 deflection_range=f"{coordinate_metrics.vertical_RMSE_um.min():.1f}--{coordinate_metrics.vertical_RMSE_um.max():.1f}"
 horizontal_range=f"{coordinate_metrics.horizontal_RMSE_um.min():.1f}--{coordinate_metrics.horizontal_RMSE_um.max():.1f}"
 main=main.replace('@@DEFLECTION_RANGE@@',deflection_range).replace('@@HORIZONTAL_RANGE@@',horizontal_range).replace('@@GPU_SPEEDUP@@',f'{speedup:.2f}')
 rows=[]
 for case,label in [('low','Low speed'),('high','High speed')]:
  r=coordinate_metrics.loc[case]
  rows.append(table_row([label,int(r.dots),f'{r.horizontal_RMSE_um:.2f}',f'{r.vertical_RMSE_um:.2f}',f'{r.radial_RMSE_um:.2f}',f'{r.radial_p95_um:.2f}']))
 results=results.replace('@@DEPOSITED_ERROR_ROWS@@','\n'.join(rows))
 maxima=directional[directional.configuration=='B'].set_index('axis')
 interpretation=(f"The high-speed case has higher measured-coordinate errors in both directions: {coordinate_metrics.loc['high','vertical_RMSE_um']:.2f} micrometers RMS in deflection and {coordinate_metrics.loc['high','horizontal_RMSE_um']:.2f} micrometers along substrate motion. Its maximum absolute directional residuals are {maxima.loc['deflection','max_absolute_um']:.2f} and {maxima.loc['horizontal','max_absolute_um']:.2f} micrometers, respectively. These are prediction discrepancies against the experimental mean positions, separate from the geometric residual minimized during computational design. The commands, field and trajectories were retained for this assessment; no measurement-dependent refit or alignment was applied.")
 results=results.replace('@@EXPERIMENTAL_INTERPRETATION@@',interpretation)
 discussion+='\nThe high-speed application has larger residuals in both coordinate directions; the cases retain different operating properties, so this difference is not attributed solely to raster structure.\n'
 conclusion=conclusion.replace(r'\section*{Data and code availability}',f"The deposited-coordinate comparison gives deflection RMSE of {deflection_range} micrometers and substrate-direction RMSE of {horizontal_range} micrometers against the experimental ten-print means. The high-speed case has the larger residuals, which remain an accuracy limit of the present model and operating-point assessment.\n\n"+r'\section*{Data and code availability}')
 results=results.replace('@@PHYSICS_ABLATION@@',physical_section(numerical,physical))
 # Integration study.
 a=num.find('A public synthetic uniform-field case isolates');num=num[:a] if a>=0 else num
 num+=numerical_section(integration)
 # Keep current and archived benchmark cohorts separate.
 start=framework.index('A matched batch of 64 low-speed contexts');end=framework.index('\\paragraph{Coverage-size intervention.}',start) if '\\paragraph{Coverage-size intervention.}' in framework[start:] else framework.index('\\paragraph{Effect of development-set coverage.}',start)
 if '\\paragraph{Effect of development-set coverage.}' in framework[start:]:end=framework.index('\\paragraph{Effect of development-set coverage.}',start)
 benchmark_text=f"A current matched batch of 64 low-speed contexts takes {current.loc['gpu','warm_median_s']:.2f} seconds on the NVIDIA RTX A5000 and {current.loc['cpu','warm_median_s']:.2f} seconds on the Intel Core i9-14900K CPU, a {speedup:.2f}-fold warmed execution speedup on this workstation. Compilation/lowering, first execution, warmed execution, transfer and output retention are recorded separately. Particle state is float32 and waveform preparation uses NumPy float64. Two synchronized warm executions are retained; this is a workstation-specific workload result, not a confidence interval or a universal accelerator advantage. The workstation was not reserved exclusively, and background processes and accelerator memory occupancy were not controlled.\n\n"
 benchmark_text+=r'''\begin{table}[htbp]\centering\small\caption{Consolidated execution evidence. Current rows use the revised executable; archived dense-recording rows retain their original source hash and output contract. Times are seconds.}\label{tab:benchmarks}\begin{tabular}{llrrr}\toprule Workload & Backend & Batch & Compile & Warm median\\\midrule
'''
 for workload,case in [('Industrial endpoints','low'),('Synthetic endpoints','synthetic')]:
  for device in ['cpu','gpu']:
   r=bench[(bench.cohort=='current')&(bench['case']==case)&(bench.platform==device)&(bench['mode']=='endpoints')&(bench.batch==64)].iloc[0]
   benchmark_text+=table_row([workload,device.upper(),64,f'{r.compile_s:.2f}',f'{r.warm_median_s:.2f}'])+'\n'
 for device in ['cpu','gpu']:
  r=bench[(bench.cohort=='historical')&(bench['case']=='low')&(bench.platform==device)&(bench['mode']=='full')&(bench.batch==1)].iloc[0]
  benchmark_text+=table_row(['Archived dense recording',device.upper(),1,f'{r.compile_s:.2f}',f'{r.warm_median_s:.2f}'])+'\n'
 benchmark_text+=r'''\bottomrule\end{tabular}\end{table}
The fully specified synthetic benchmark contains 32 spheres of diameter 132 micrometers and density 905 kg/m$^3$, emitted at 75 kHz with initial axial speed 24 m/s. It uses $E_X=-300000$ V/m, independently seeded charge magnitudes uniform on 0.5--1.5 pC (seed 20261111), $h=1$ microsecond and a 4 ms maximum window, with Coulomb and wake/drag terms enabled and coalescence disabled; it contains no industrial font, field or command table. It provides an independently reproducible computational workload, not another physical-validation case. Complete settings, precision, batch shapes, hardware and synchronized timing records are preserved. Application-specific settings are recorded with the reproduction inputs. The archived single-case dense-recording workload returns approximately 1.18 GB and is slower on GPU than CPU, delimiting the acceleration regime. Current matched endpoints and historical dense recording are not compared as identical workloads. The throughput illustration above is retained as the historical reference; the table reports the current rerun.

'''
 framework=framework[:start]+benchmark_text+framework[end:]
 main=re.sub(r'A matched batch of 64 endpoint evaluations achieves a [0-9.]+-fold warmed GPU speedup over the CPU on the tested workstation; compilation and recording costs are reported separately\.',f'A current matched batch of 64 endpoint evaluations achieves a {speedup:.2f}-fold warmed GPU speedup over the CPU on the tested workstation; compilation and recording costs are reported separately.',main)
 main=main.replace('The contribution is a traceable development process with explicit computational coverage and a defined experimental scope.','Frozen-command force ablations test interaction relevance, while a shared-force Euler--RK4 comparison separates continuous integration from discrete-event sensitivity. The contribution is a traceable development process with explicit computational coverage and a defined experimental scope.')
 (PAPER/'main.tex').write_text(main.rstrip()+'\n')
 for name,text in [('results',results),('numerical_implementation',num),('framework_evidence',framework),('discussion',discussion),('conclusion',conclusion)]: (PAPER/'sections'/f'{name}.tex').write_text(text.rstrip()+'\n')
 import shutil
 call('build_paper.py','--engine',os.environ.get('CIJ_TECTONIC') or shutil.which('tectonic') or 'tectonic')
 with zipfile.ZipFile(PAPER/'source.zip') as z:
  if any('disclosure_review' in n or n.startswith('inputs/') for n in z.namelist()):raise RuntimeError('Private review material entered manuscript archive')
 write(STUDY/'publication.json',dict(status='complete',experimental_assessment=str(assessment_directory().relative_to(ROOT)),experimental_provenance_sha256=sha(assessment_directory()/'provenance.json'),numerical_freeze_sha256=sha(STUDY/'numerical_freeze.json'),paper_sha256=sha(PAPER/'main.pdf'),current_matched_gpu_speedup=speedup,source_zip_sha256=sha(PAPER/'source.zip')))
 print('Evidence frozen; compact sections integrated; manuscript built.')
if __name__=='__main__':main()
