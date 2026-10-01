"""A preassigned NSW fixed-state SPO+ / squared-error comparison.

Only revision_v8/spo is written. Existing episodes, controllers and replays
remain read-only. A real mixed-integer primary oracle defines SPO+, while
deployment uses the existing three-stage tolerance-ordered policy.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, hashlib, json, time, datetime, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(1);pa.set_io_thread_count(1)
from revision_v4_forecast_models import LEVEL_COLS, Y, bound_paths
from revision_v5_control_solver import LexMPC

ROOT=Path(__file__).resolve().parents[1]
REGION='NSW1'; SCALE=100.; SEED=20261001
CONFIG={'region':REGION,'training_years':[2021,2022], 'validation_year':2023,
 'evaluation_start':'2024-01-01','evaluation_end_exclusive':'2026-09-01',
 'train_sample_size':2048,'selection_sample_size':2048,'sampling_seed':SEED,
 'sampling':'uniform without replacement within each chronological phase; identical rows for both losses',
 'features':LEVEL_COLS,'prediction_class':'raw + 100 times standardized-current-feature linear correction with intercept',
 'regularization_grid':[0.0001,0.01,1.0],'ridge_loss':'mean squared 12-horizon normalized price error / 2',
 'spo_loss':'exact SPO+ for cost q+A*p; q includes fixed wear',
 'initial_soc_mwh':1.0,'epochs':8,'batch_size':16,'adam_initial_step':0.01,
 'adam_step_schedule':'0.01 / sqrt(epoch+1)','final_weights':'mean of epoch 7 and 8 endpoints',
 'selection':'minimum mean fixed-state local reference-objective regret on common purged 2023 sample; ties prefer larger regularization',
 'no_evaluation_fit_or_selection':True,'deployment':'same capped forecasts and cold three-stage tolerance controller as R and C*',
 'boundary':'local fixed-state training, not sequential end-to-end training; design is exploratory after earlier evaluation inspection'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    Path(p).write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cost(p,h=8,kappa=5.,dt=.5):
    p=np.asarray(p,float);v=np.zeros(4*h+1)
    v[:h]=dt*(p[:h]+kappa);v[h:2*h]=dt*(-p[:h]+kappa)
    v[-1]=-p[h:].mean()
    return v
def sensitivity(x,h=8,dt=.5,tail=4):
    return np.r_[dt*(x[:h]-x[h:2*h]),np.full(tail,-x[-1]/tail)]

class PrimaryOracle(LexMPC):
    """Primary global oracle; no sequential secondary objectives in training."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.calls=0;self.binary_calls=0
    def optimize(self,p,soc=1.):
        h=self.h;c=cost(p,h,self.kappa,self.dt)
        self.solver.changeColBounds(3*h,float(soc),float(soc))
        for r in (self.primary_row,self.throughput_row,self.branch_row):
            self.solver.changeRowBounds(r,-np.inf,np.inf)
        ans=self._run(c)
        if ans is None:raise RuntimeError('Primary oracle infeasible')
        x,v,b=ans;self.calls+=1
        err=max(abs(np.diff(x[3*h:])-self.eta*self.dt*x[:h]+self.dt/self.eta*x[h:2*h]))
        if err>1e-6 or np.minimum(x[:h],x[h:2*h]).max()>1e-6:
            raise RuntimeError('Primary oracle physical violation')
        if v-b>2e-6:raise RuntimeError('Primary oracle objective gap')
        return x,v

def spo(p,y,truth_x,oracle):
    reflected=2*np.asarray(p)-np.asarray(y)
    reflected_x,z=oracle.optimize(reflected)
    loss=float(cost(reflected)@truth_x-z)
    g=2*(sensitivity(truth_x)-sensitivity(reflected_x))
    if loss< -1e-6:raise RuntimeError('SPO+ loss below zero')
    return max(0.,loss),g

def verify(out):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    rng=np.random.default_rng(SEED);oracle=PrimaryOracle();reference=LexMPC(relax_if_mode_feasible=False)
    cases=[np.full(12,a) for a in (-1000.,-50.,0.,50.,17500.)]
    cases += list(rng.normal(70,140,(25,12)))
    diffs=[];ineq=[];finite=[];upper=[]
    for y in cases:
        truth,v=oracle.optimize(y);ref=reference.solve(y,1.)
        diffs.append(abs(v+ref['primary_optimum_incumbent_aud']))
        p=y+rng.normal(0,80,12);loss,g=spo(p,y,truth,oracle)
        px,pv=oracle.optimize(p);regret=cost(y)@px-v;upper.append(float(regret-loss))
        delta=rng.normal(0,2,12);next_loss,_=spo(p+delta,y,truth,oracle)
        ineq.append(float(loss+g@delta-next_loss))
        for j in (0,7,8,11):
            eps=np.zeros(12);eps[j]=1e-4
            lp,_=spo(p+eps,y,truth,oracle);lm,_=spo(p-eps,y,truth,oracle)
            finite.append(abs((lp-lm)/2e-4-g[j]))
    flat=oracle.optimize(np.full(12,50.))[0]
    closed=float(cost(np.full(12,50.))@flat)
    assert np.max(diffs)<1e-6 and max(ineq)<1e-6 and max(upper)<1e-6
    assert max(finite)<1e-5 and abs(closed+50.)<1e-8
    assert abs(np.sum(flat[:16]))<1e-8
    # 2(q+A p_hat)-(q+A y)=q+A(2p_hat-y); fixed wear survives once.
    p=np.arange(12)*20.;y=np.arange(12)[::-1]*40.
    identity=float(np.max(abs(2*cost(p)-cost(y)-cost(2*p-y))))
    assert identity<1e-12
    report={'status':'PASS','before_formal_fit_utc':now(),'cases':len(cases),
      'max_primary_difference_vs_all_binary_controller_aud':max(diffs),
      'max_subgradient_inequality_violation_aud':max(0.,max(ineq)),
      'max_finite_difference_error':max(finite),'max_SPO_minus_SPOplus_aud':max(upper),
      'fixed_wear_reflection_identity_max_error':identity,
      'positive_flat_price_closed_form':'no flow, terminal 1 MWh at A$50; minimum cost -50',
      'positive_flat_cost':closed,'seconds':time.time()-start,
      'oracle_calls':oracle.calls,'source':'Elmachtoub and Grigas (2022), DOI10.1287/mnsc.2020.3922; arXiv1710.08005'}
    write(out/'oracle_validation.json',report);print(json.dumps(report),flush=True)
    return report

def load():
    p=Path(os.environ.get('AOOR_DATA_ROOT',str(ROOT)))/'results/revision_v5/history/NSW1_history_features_aligned.parquet'
    cols=list(dict.fromkeys(['target','curve_asof','cutoff_minutes',*LEVEL_COLS,*Y]))
    f=pd.read_parquet(p,columns=cols,filters=[('cutoff_minutes','==',60)]).sort_values('target').reset_index(drop=True)
    assert not f.target.duplicated().any() and np.isfinite(f[Y].to_numpy()).all()
    assert (f.curve_asof<=f.target-pd.Timedelta(minutes=60)).all()
    train=f.loc[f.target.ge('2021-01-01')&(f.target+pd.Timedelta(minutes=330)<pd.Timestamp('2023-01-01')-pd.Timedelta(minutes=60))].copy()
    val=f.loc[f.target.ge('2023-01-01')&(f.target+pd.Timedelta(minutes=330)<pd.Timestamp('2024-01-01')-pd.Timedelta(minutes=60))].copy()
    test=f.loc[f.target.ge('2024-01-01')&f.target.lt('2026-09-01')].copy()
    assert len(val)==17507 and len(test)==46741
    rng=np.random.default_rng(SEED)
    train=train.iloc[np.sort(rng.choice(len(train),CONFIG['train_sample_size'],replace=False))].reset_index(drop=True)
    val_sample=val.iloc[np.sort(rng.choice(len(val),CONFIG['selection_sample_size'],replace=False))].reset_index(drop=True)
    return p,train,val_sample,val,test

def design_x(frame,median,mean,sd):
    x=frame[LEVEL_COLS].to_numpy(float);x=np.where(np.isfinite(x),x,median)
    return np.c_[(x-mean)/sd,np.ones(len(frame))]

def fit(out):
    if not (out/'oracle_validation.json').exists():raise RuntimeError('Verify first')
    p,train,val,fullval,test=load();start=time.time()
    x=train[LEVEL_COLS].to_numpy(float);median=np.nanmedian(x,axis=0)
    x=np.where(np.isfinite(x),x,median);mean=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.
    x=design_x(train,median,mean,sd);vx=design_x(val,median,mean,sd)
    raw=train[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float);y=train[Y].to_numpy(float)
    vy=val[Y].to_numpy(float);vr=val[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float)
    source_scripts=['revision_v8_spo.py','revision_v5_control_solver.py','revision_v5_control_safe_policy.py','revision_v5_control_scaled_policy.py','revision_v5_control_enumerated_policy.py','revision_v4_forecast_models.py']
    freeze={'status':'frozen before formal fit','utc':now(),'config':CONFIG,'source_path':str(p),'source_sha256':sha(p),
      'script_sha256':{n:sha(ROOT/'scripts'/n) for n in source_scripts},'D_free_bytes':shutil.disk_usage(out.resolve()).free,
      'training_rows':len(train),'training_first_origin':str(train.target.min()),'training_last_origin':str(train.target.max()),
      'training_last_complete_label':str(train.target.max()+pd.Timedelta(minutes=330)),
      'validation_rows':len(val),'full_validation_rows':len(fullval),'evaluation_rows':len(test),
      'train_target_sha256':hashlib.sha256(train.target.to_numpy().tobytes()).hexdigest(),
      'validation_target_sha256':hashlib.sha256(val.target.to_numpy().tobytes()).hexdigest()}
    write(out/'design_freeze.json',freeze)
    train[['target','curve_asof']].to_csv(out/'training_origins.csv',index=False)
    val[['target','curve_asof']].to_csv(out/'selection_origins.csv',index=False)
    oracle=PrimaryOracle();truth=[];truev=[]
    for yy in y:
        xx,v=oracle.optimize(yy);truth.append(xx);truev.append(v)
    truth=np.array(truth);truev=np.array(truev)
    validation_truth=[];validation_value=[]
    for yy in vy:
        xx,v=oracle.optimize(yy);validation_truth.append(xx);validation_value.append(v)
    validation_value=np.array(validation_value)
    models={};rows=[];logs=[];penalty=np.ones(x.shape[1]);penalty[-1]=0
    for lam in CONFIG['regularization_grid']:
        ts=time.time();w=np.linalg.solve(x.T@x+len(x)*12*lam*np.diag(penalty),x.T@((y-raw)/SCALE))
        name=f'ridge_l{lam:g}';models[name]=w
        rows.append(select_score(name,lam,w,vx,vr,vy,val,validation_value,oracle,time.time()-ts))
    for lam in CONFIG['regularization_grid']:
        ts=time.time();w=np.zeros((x.shape[1],12));m=w.copy();v=w.copy();step=0;ends=[]
        rng=np.random.default_rng(SEED)
        for ep in range(CONFIG['epochs']):
            order=rng.permutation(len(x));losses=[]
            for lo in range(0,len(x),CONFIG['batch_size']):
                ix=order[lo:lo+CONFIG['batch_size']];grad=np.zeros_like(w)
                for i in ix:
                    loss,g=spo(raw[i]+SCALE*(x[i]@w),y[i],truth[i],oracle)
                    losses.append(loss/SCALE);grad+=np.outer(x[i],g)
                grad/=len(ix);grad+=lam*penalty[:,None]*w
                step+=1;m=.9*m+.1*grad;v=.999*v+.001*grad**2
                w-=(CONFIG['adam_initial_step']/np.sqrt(ep+1))*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            ends.append(w.copy())
            row={'lambda':lam,'epoch':ep+1,'online_mean_SPOplus_aud':np.mean(losses)*SCALE,
              'weight_norm':float(np.linalg.norm(w[:-1])),'elapsed_seconds':time.time()-ts}
            logs.append(row);print(json.dumps(row),flush=True)
        w=np.mean(ends[-2:],axis=0);name=f'spo_l{lam:g}';models[name]=w
        rows.append(select_score(name,lam,w,vx,vr,vy,val,validation_value,oracle,time.time()-ts))
    selected={family:min([r for r in rows if r['family']==family],key=lambda r:(r['local_reference_regret_aud'], -r['lambda']))['model'] for family in ('ridge','spo')}
    np.savez(out/'models.npz',median=median,mean=mean,sd=sd,**models)
    pd.DataFrame(rows).to_csv(out/'selection_candidates.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'training_curve.csv',index=False)
    write(out/'selection_complete.json',{'status':'complete','utc':now(),'selected':selected,'candidates':rows,'seconds':time.time()-start,
      'selection_uses_evaluation_outcomes':False,'training_is_local_fixed_state':True,'model_sha256':sha(out/'models.npz'),
      'design_freeze_sha256':sha(out/'design_freeze.json')})
    print(json.dumps({'selected':selected,'fit_seconds':time.time()-start}),flush=True)

def select_score(name,lam,w,vx,vr,vy,val,truev,oracle,seconds):
    pred=bound_paths(vr+SCALE*(vx@w),val).astype(float);regrets=[]
    for p,y,tv in zip(pred,vy,truev):
        xx,_=oracle.optimize(p);regrets.append(float(cost(y)@xx-tv))
    if min(regrets)<-1e-5:raise RuntimeError('Negative regret')
    return {'model':name,'family':name.split('_')[0],'lambda':lam,'local_reference_regret_aud':float(np.mean(regrets)),
      'price_mae':float(np.abs(pred-vy).mean()),'price_rmse':float(np.sqrt(np.mean((pred-vy)**2))),
      'fit_seconds':seconds,'validation_rows':len(vy)}

def controller(out):
    # Route rare numerical-case records into this experiment only; no old write.
    import revision_v5_control_solver as a, revision_v5_control_safe_policy as b
    import revision_v5_control_scaled_policy as c, revision_v5_control_enumerated_policy as d
    for mod in (a,b,c,d):mod.OUT=out/'numerical_cases'
    return d.EnumeratedSafeLexMPC()

def replay(out):
    _,_,_,val,test=load();sel=json.loads((out/'selection_complete.json').read_text())
    models=np.load(out/'models.npz');x=design_x(test,models['median'],models['mean'],models['sd'])
    raw=test[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float)
    for family,name in sel['selected'].items():
        path=out/f'{family}_evaluation_trajectory.parquet'
        if path.exists() and (out/f'{family}_evaluation_report.json').exists():continue
        pred=bound_paths(raw+SCALE*(x@models[name]),test).astype(float)
        pd.DataFrame(pred,columns=[f'p_{j+1:02d}' for j in range(12)]).assign(target=test.target.to_numpy()).to_parquet(out/f'{family}_evaluation_forecasts.parquet',index=False)
        model=controller(out);rows=[];soc=1.;start=time.time()
        for i,(p,rr) in enumerate(zip(pred,test.itertuples(index=False))):
            ans=model.solve(p,soc);price=rr.actual_00
            cash=.5*(ans['d']-ans['c'])*price;wear=.5*5*(ans['d']+ans['c'])
            rows.append({'target':rr.target,'actual_price':price,'soc_start':soc,'soc_end':ans['soc'],
              'charge_mw':ans['c'],'discharge_mw':ans['d'],'cashflow_aud':cash,'degradation_aud':wear,'net_aud':cash-wear,
              'primary_loss_aud':ans['policy_primary_loss_aud'],'primary_gap_aud':ans['primary_objective_gap_aud']})
            soc=ans['soc']
            if (i+1)%2000==0:
                write(out/f'{family}_progress.json',{'rows':i+1,'of':len(test),'seconds':time.time()-start,'utc':now()})
                print(json.dumps({'replay':family,'rows':i+1,'seconds':time.time()-start}),flush=True)
        f=pd.DataFrame(rows);f.to_parquet(path,index=False,compression='zstd')
        end=float(f.soc_end.iloc[-1]*f.actual_price.iloc[-1]-f.soc_start.iloc[0]*f.actual_price.iloc[0])
        report={'status':'complete','model':name,'intervals':len(f),'net_cash_aud':float(f.net_aud.sum()),'endpoint_mark_aud':end,
          'marked_value_aud':float(f.net_aud.sum()+end),'marked_value_aud_per_974_nominal_days':float((f.net_aud.sum()+end)/974),
          'seconds':time.time()-start,'forecast_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),
          'trajectory_sha256':sha(path),'controller_definition':model.definition(),'final_soc_mwh':soc,
          'max_primary_loss_aud':float(f.primary_loss_aud.max()),'max_primary_gap_aud':float(f.primary_gap_aud.max()),
          'forecast_price_mae':float(np.abs(pred-test[Y].to_numpy(float)).mean()),
          'forecast_price_rmse':float(np.sqrt(np.mean((pred-test[Y].to_numpy(float))**2))),
          'first_price_mae':float(np.abs(pred[:,0]-test.actual_00.to_numpy()).mean()),
          'terminal_mean_mae':float(np.abs(pred[:,8:].mean(1)-test[Y[8:]].to_numpy(float).mean(1)).mean()),
          'numerical_fallback_count':getattr(model,'numerical_fallback_count',0),
          'equivalent_unit_fallback_count':getattr(model,'equivalent_unit_fallback_count',0),
          'enumerated_fallback_count':getattr(model,'enumerated_fallback_count',0)}
        write(out/f'{family}_evaluation_report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['verify','fit','replay','all'],default='all')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results/revision_v8/spo');args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.mode in ('verify','all'):verify(out)
    if args.mode in ('fit','all'):fit(out)
    if args.mode in ('replay','all'):replay(out)
