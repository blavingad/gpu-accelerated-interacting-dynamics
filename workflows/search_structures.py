"""Bounded large no-guard permutation search with direct waveform/DAC fitting.

A surrogate screens permutations only. Every reported physical result uses the
unchanged full solver. No claim of exhaustive search over 16! orders.
"""
from pathlib import Path
import sys, os, json, time, hashlib, fcntl, traceback, argparse
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from vdcr import structure_search_support as pilot
from vdcr.charging_sequence import run_sequence

from types import SimpleNamespace

OUT=ROOT/'results/generated/structure_search'
PILOT=ROOT/'inputs/structure_search/seeds'

def atomic(path,value):
    # Serialize nonfinite flight diagnostics as null.
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def score_horizontal(y):
    """Remove the common column phase while retaining shear in the score."""
    y=np.asarray(y,float);centered=y-y.mean(axis=-1,keepdims=True)
    x=np.arange(16)-7.5
    rms=np.sqrt(np.mean(centered**2,axis=-1));maximum=np.max(abs(centered),axis=-1)
    slope=centered@x/(x@x)
    rough=np.sqrt(np.mean(np.diff(centered,axis=-1)**2,axis=-1))
    return rms+.20*maximum+.25*rough+.35*abs(slope)*np.std(x)

def valid_order(order):return sorted(map(int,order))==list(range(1,17))

def waveform(p,order,commands):
    if not valid_order(order):raise ValueError('Expected a permutation of 1..16')
    commands=np.asarray(commands,float)
    if commands.shape!=(16,) or not np.isfinite(commands).all() or np.any(commands<230) or np.any(commands>1023):
        raise ValueError('Sixteen row DACs in 230..1023 required')
    physical=np.tile(commands[np.asarray(order)-1],2)
    return run_sequence(physical,np.ones(32,bool),p.params,compensation_fractions=(0.,0.),induction_coefficients=(.1,.02),breakup_phase=.5)

def quality(r):
    return bool(r and r.get('safe') and r.get('vertical_RMS_um') is not None and r['vertical_RMS_um']<=66.45 and r['vertical_max_um']<=132.9)

def predicted_orders(count,selected):
    files=sorted(PILOT.glob('*/completed.json'));offsets=[];seeds=[]
    v=pilot.SPEED/60;period=1/pilot.FREQUENCY
    for f in files:
        r=json.loads(f.read_text());order=r['sequence'];seeds.append(order)
        a=r['best'];frame=pd.read_csv(f.parent/'flights'/a['key']/'positions.csv').sort_values('drop')
        offsets.append(frame.paper_y_mm.to_numpy()+v*1000*(frame.firing_slot.to_numpy()-1)*period)
    offsets=np.asarray(offsets);rng=np.random.default_rng(20261102);pool={};bestseeds=np.asarray(seeds,dtype=np.int16)
    # Scores are predictions from four frozen empirical flight-offset models.
    def consider(orders):
        slots=np.argsort(orders,axis=1)
        predictions=offsets[:,None,:]-v*1000*slots[None,:,:]*period
        scores=score_horizontal(predictions)
        value=scores.mean(axis=0)+.15*scores.std(axis=0)
        chosen=np.argpartition(value,min(255,len(value)-1))[:min(256,len(value))]
        for j in chosen:
            key=tuple(map(int,orders[j]));val=float(value[j]);pool[key]=min(val,pool.get(key,np.inf))
    for start in range(0,count,10000):
        n=min(10000,count-start);orders=np.argsort(rng.random((n,16)),axis=1).astype(np.int16)+1
        # Include local neighbourhoods around promising seeds as well as global permutations.
        local=n//2;orders[:local]=bestseeds[rng.integers(0,len(bestseeds),local)]
        for _ in range(4):
            a=rng.integers(0,16,local);b=rng.integers(0,16,local);ii=np.arange(local)
            temp=orders[ii,a].copy();orders[ii,a]=orders[ii,b];orders[ii,b]=temp
        consider(orders)
        if len(pool)>16000:pool=dict(sorted(pool.items(),key=lambda kv:kv[1])[:8192])
    # Always include all original controls, rotations and reversals.
    extras=[]
    for seed in seeds:
        for k in range(16):
            q=seed[k:]+seed[:k];extras.extend([q,q[::-1]])
    consider(np.asarray(extras));ranked=sorted(pool.items(),key=lambda kv:kv[1])
    picked=[tuple(s) for s in seeds]
    for minimum_distance in [7,5,3,1]:
        for order,value in ranked:
            if len(picked)>=selected:break
            if order not in picked and all(sum(a!=b for a,b in zip(order,q))>=minimum_distance for q in picked):picked.append(order)
        if len(picked)>=selected:break
    rows=[dict(index=i,candidate='sequence_'+hashlib.sha256(bytes(order)).hexdigest()[:12],sequence=' '.join(map(str,order)),predicted_score_mm=pool.get(order)) for i,order in enumerate(picked[:selected])]
    pd.DataFrame(rows).to_csv(OUT/'selected_orders.csv',index=False)
    pd.DataFrame([dict(sequence=' '.join(map(str,o)),predicted_score_mm=s) for o,s in ranked[:8192]]).to_csv(OUT/'surrogate_leaderboard.csv',index=False)
    atomic(OUT/'surrogate_manifest.json',dict(sampled_candidates=count,unique_retained=len(pool),physical_orders_selected=len(rows),seed=20261102,model='ensemble of four frozen simulated full-stroke flight offsets; frozen offsets are approximate and do not validate charge or coalescence',exhaustive=False))
    return rows

class Search:
    def __init__(self,args):
        self.args=args;self.started=time.time();self.records=[];self.cache={}
        pilot.OUT=OUT;pilot.atomic=atomic
        pilot.status=lambda **v:atomic(OUT/'last_evaluation.json',dict(updated_unix=time.time(),**v))
        self.p=pilot.Pilot()
        self.p.capture=SimpleNamespace(last_case_id=None,wave_context={})
        old=json.loads((OUT/'evaluations.json').read_text()) if (OUT/'evaluations.json').exists() else []
        self.records=old;self.cache={r['evaluation_key']:r for r in old}
        self.qseed=np.mean([json.loads(f.read_text())['best']['charge'] for f in PILOT.glob('*/completed.json')],axis=0)
        self.status(phase='initialised')
    def status(self,**v):
        atomic(OUT/'status.json',dict(status='running',pid=os.getpid(),updated_unix=time.time(),elapsed_s=time.time()-self.started,physical_evaluations=len(self.records),screen_limit=self.args.physical,refine_limit=self.args.refine,**v))
    def deadline(self):
        if time.time()-self.started>self.args.hours*3600:raise TimeoutError('Configured wall-clock budget reached; all completed results retained')
    def evaluate(self,order,u,label):
        self.deadline();order=list(map(int,order));u=np.asarray(u,float)
        key=hashlib.sha256(bytes(order)+u.tobytes()).hexdigest()[:24]
        if key in self.cache:return self.cache[key],pd.read_csv(OUT/self.cache[key]['candidate']/'flights'/self.cache[key]['key']/'positions.csv')
        if np.any(u<230) or np.any(u>1023):return None,None
        wave=waveform(self.p,order,u)
        q=-np.asarray([r['assigned_charge_pC'] for r in wave['trace'][16:]])[np.argsort(order)]
        name='sequence_'+hashlib.sha256(bytes(order)).hexdigest()[:12]
        r,f=self.p.evaluate(name,order,q,label,forced=wave)
        if r is None:return None,None
        y=f.paper_y_mm.to_numpy();centered=y-y.mean()
        r.update(evaluation_key=key,row_DACs=u.tolist(),horizontal_max_um=float(np.max(abs(centered))*1000) if np.isfinite(centered).all() else None,horizontal_shape_score_um=float(score_horizontal(y)*1000) if np.isfinite(y).all() else None,height_mm=float(np.ptp(f.landing_mm)+.3968751307168854) if f.landed.all() else None,within_vertical_policy=quality(r))
        atomic(OUT/name/'flights'/r['key']/'record.json',r)
        self.records.append(r);self.cache[key]=r;atomic(OUT/'evaluations.json',self.records)
        rows=[{k:v for k,v in a.items() if not isinstance(v,(dict,list))} for a in self.records]
        pd.DataFrame(rows).to_csv(OUT/'evaluations.csv',index=False)
        return r,f
    def initial_commands(self,order):
        # Use warm CURRENT-stroke inverse commands, never average with cold-start commands.
        wave,u=self.p.waveform(order,self.qseed)
        return u[16:][np.argsort(order)]
    def fit(self,order,start):
        name='sequence_'+hashlib.sha256(bytes(order)).hexdigest()[:12];dest=OUT/name
        if (dest/'refinement.json').exists():return json.loads((dest/'refinement.json').read_text())
        best,f=self.evaluate(order,start,'refine_start');u=np.asarray(start,float);J=None;history=[]
        if not best or not best['safe']:
            return dict(candidate=name,status='no_safe_start')
        # First restore vertical geometry, then trade small vertical error for horizontal quality.
        for it in range(8):
            stage='vertical' if best['vertical_RMS_um']>10 or best['vertical_max_um']>25 else 'joint'
            self.status(phase='refinement',candidate=name,iteration=it,objective=stage)
            def residual(frame):
                ex=frame.landing_mm.to_numpy()-self.p.target;y=frame.paper_y_mm.to_numpy();ey=y-y.mean()
                return ex if stage=='vertical' else np.r_[4*ex,ey]
            error=residual(f)
            if J is None or it in [3,6] or J.shape[0]!=len(error):
                columns=[]
                for j in range(16):
                    h=max(2.,.005*u[j]);plus=u.copy();minus=u.copy();plus[j]=min(1023,u[j]+h);minus[j]=max(230,u[j]-h)
                    rp,fp=self.evaluate(order,plus,f'jac{it}_row{j+1}_plus');rm,fm=self.evaluate(order,minus,f'jac{it}_row{j+1}_minus')
                    if rp and rm and rp['safe'] and rm['safe']:col=(residual(fp)-residual(fm))/(plus[j]-minus[j])
                    elif rp and rp['safe'] and plus[j]>u[j]+1e-9:col=(residual(fp)-error)/(plus[j]-u[j])
                    elif rm and rm['safe'] and u[j]>minus[j]+1e-9:col=(error-residual(fm))/(u[j]-minus[j])
                    else:columns=[];break
                    columns.append(col)
                if not columns:break
                J=np.column_stack(columns);np.save(dest/f'dac_jacobian_{it}.npy',J)
            delta=lsq_linear(np.vstack([J,1e-5*np.eye(16)]),np.r_[-error,np.zeros(16)],bounds=(np.maximum(230-u,-.05*u),np.minimum(1023-u,.05*u)),tol=1e-9,max_iter=100).x
            improved=False
            for fraction in [1.,.5,.25,.125]:
                trial=u+fraction*delta;r,ff=self.evaluate(order,trial,f'fit{it}_{stage}_step{fraction}')
                if not r or not r['safe']:continue
                accept=r['vertical_RMS_um']<best['vertical_RMS_um'] if stage=='vertical' else quality(r) and np.sum(residual(ff)**2)<np.sum(error**2)
                if accept:
                    dq=trial-u;de=residual(ff)-error
                    if dq@dq>1e-15:J+=np.outer(de-J@dq,dq)/(dq@dq)
                    u,best,f=trial,r,ff;improved=True;break
            history.append(dict(iteration=it,objective=stage,improved=improved,vertical_RMS_um=best['vertical_RMS_um'],horizontal_RMS_um=best['horizontal_RMS_um']))
            atomic(dest/'refinement_history.json',history)
            if not improved:break
        # Replay exactly one INTEGER row command in both strokes. Compare nearby rounding choices.
        integer=np.floor(u+.5).astype(int);replay,rf=self.evaluate(order,integer,'integer_DAC_replay')
        result=dict(candidate=name,status='complete',sequence=order,best_continuous=best,integer_DAC_replay=replay,scope='two repeated full strokes; full font not yet validated')
        atomic(dest/'refinement.json',result);return result

def publish(execute=True):
    # Plotting/report generation is a separate workflow, never launched by optimization.
    if (OUT/'evaluations.csv').exists():
        f=pd.read_csv(OUT/'evaluations.csv')
        atomic(OUT/'progress_summary.json',dict(physical_evaluations=len(f),safe=int(f.safe.sum())))

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));parser.add_argument('--device',choices=['cpu','gpu'],default='cpu');parser.add_argument('--samples',type=int,default=1000000);parser.add_argument('--physical',type=int,default=512);parser.add_argument('--refine',type=int,default=24);parser.add_argument('--hours',type=float,default=12.);parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    if a.samples<1 or a.physical<1 or a.refine<0 or a.hours<=0:parser.error('samples/physical/hours must be positive; refine must be nonnegative')
    OUT=Path(a.output).resolve();pilot.DEVICE=a.device
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Worker already active')
        if (OUT/'selected_orders.csv').exists():orders=pd.read_csv(OUT/'selected_orders.csv').to_dict('records')
        else:orders=predicted_orders(a.samples,a.physical)
        seed=pilot.Pilot();atomic(OUT/'manifest.json',dict(arguments=vars(a),target_landing_mm=seed.target.tolist(),physical_configuration=seed.config,started_unix=time.time(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(pilot.__file__)]},guard_cycles=0,exhaustive=False))
        atomic(OUT/'status.json',dict(status='prepared',physical_orders=len(orders)));publish()
        if a.prepare_only:return
        search=Search(a)
        for i,row in enumerate(orders):
            search.status(phase='physical_screen',orders_completed=i,candidate=row['candidate'])
            order=list(map(int,row['sequence'].split()))
            try:u=search.initial_commands(order)
            except ValueError:continue
            search.evaluate(order,u,'screen_warm_inverse_seed')
            if (i+1)%64==0:publish()
        # Prefer vertically feasible orders; include a small number with promising timing needing restoration.
        ranked=sorted((r for r in search.records if r['safe']),key=lambda r:(not quality(r),r['horizontal_shape_score_um']+.25*r['vertical_RMS_um']))
        shortlist=[];seen=set()
        for r in (ranked if a.refine>0 else []):
            if r['candidate'] not in seen:shortlist.append(r);seen.add(r['candidate'])
            if len(shortlist)>=a.refine:break
        atomic(OUT/'shortlist.json',shortlist)
        refinements=[]
        for j,r in enumerate(shortlist):
            search.status(phase='refinement',refinements_completed=j,candidate=r['candidate'])
            refinements.append(search.fit(r['sequence'],r['row_DACs']));publish()
        atomic(OUT/'completed.json',dict(status='complete',physical_evaluations=len(search.records),refinements=refinements,global_optimum_proved=False))
        search.status(phase='finished',refinements_completed=len(refinements));s=json.loads((OUT/'status.json').read_text());s['status']='complete';atomic(OUT/'status.json',s);publish()

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic(OUT/'status.json',dict(status='budget_complete' if isinstance(e,TimeoutError) else 'failed',error=repr(e),updated_unix=time.time()));traceback.print_exc()
        try:publish()
        except Exception:traceback.print_exc()
        raise
