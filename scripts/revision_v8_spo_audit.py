"""Read-only numerical evaluation of the frozen local SPO+ experiment."""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
from revision_v8_spo import ROOT,sha,write,load,design_x,PrimaryOracle,cost,SCALE
from revision_v4_forecast_models import Y,bound_paths

PLAN={'bootstrap_draws':5000,'block_lengths_days':[7,14,28], 'seed':20261008,
 'uncertainty_estimand':'mean paired realized net operating cash on common interior execution dates, excluding first and final boundary days',
 'local_regret_grid':'all 46741 common evaluation origins; fixed initial SOC1; primary global oracle selects a deterministic primary optimum, separate from deployment tie ordering',
 'contrasts':['SPO_minus_Ridge','SPO_minus_R','SPO_minus_Cstar','Ridge_minus_Cstar'],
 'tests':'two-sided null-centered circular block bootstrap; Holm across four contrasts separately per block length',
 'scope':'conditional on NSW asset, policies and chronological period; dependent-block assumptions, no guarantee under strong nonstationarity',
 'annual_descriptives':'2024,2025,2026 January–August, net operating cash, endpoint mark not split arbitrarily'}

def physics(f,prices,definition):
    needed=['target','actual_price','soc_start','soc_end','charge_mw','discharge_mw','cashflow_aud','degradation_aud','net_aud']
    if not set(needed)<=set(f):raise RuntimeError('Missing trajectory audit columns')
    assert np.array_equal(f.target.to_numpy(),prices.target.to_numpy())
    assert np.array_equal(f.actual_price.to_numpy(),prices.actual_00.to_numpy(float))
    c=f.charge_mw.to_numpy();d=f.discharge_mw.to_numpy();s=f.soc_start.to_numpy();sn=f.soc_end.to_numpy();p=f.actual_price.to_numpy()
    cash=.5*(d-c)*p;wear=.5*5*(c+d);net=cash-wear
    balance=sn-s-.91*.5*c+.5*d/.91
    err={'cash_max_error_aud':float(np.max(abs(cash-f.cashflow_aud.to_numpy()))),
      'wear_max_error_aud':float(np.max(abs(wear-f.degradation_aud.to_numpy()))),
      'net_max_error_aud':float(np.max(abs(net-f.net_aud.to_numpy()))),
      'soc_balance_max_error_mwh':float(np.max(abs(balance))),
      'soc_continuity_max_error_mwh':float(np.max(abs(s[1:]-sn[:-1]))),
      'simultaneous_modes_count':int(np.sum((c>1e-6)&(d>1e-6))),
      'max_bound_violation':float(max(0.,.2-s.min(),.2-sn.min(),s.max()-1.8,sn.max()-1.8,-c.min(),-d.min(),c.max()-1,d.max()-1))}
    assert max(err['cash_max_error_aud'],err['wear_max_error_aud'],err['net_max_error_aud'])<1e-7
    assert max(err['soc_balance_max_error_mwh'],err['soc_continuity_max_error_mwh'],err['max_bound_violation'])<1e-6
    assert err['simultaneous_modes_count']==0 and abs(s[0]-1.)<1e-12
    for key,value in [('h',8),('power_mw',1),('eta_c',.91),('eta_d',.91),('dt_hours',.5),('kappa_aud_per_grid_mwh',5),('smin_mwh',.2),('smax_mwh',1.8),('terminal_factor',1),('primary_tolerance_aud',1e-5),('throughput_tolerance_mwh',1e-7)]:
        assert key in definition and definition[key]==value
    endpoint=sn[-1]*p[-1]-s[0]*p[0]
    return err,float(net.sum()),float(endpoint)

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return out

def run(out):
    start=time.time();_,train,val,fullval,test=load();freeze=json.loads((out/'design_freeze.json').read_text())
    assert sha(out/'training_script_frozen.py')==freeze['script_sha256']['revision_v8_spo.py']
    for n,h in freeze['script_sha256'].items():assert sha(ROOT/'scripts'/n)==h,('code changed',n)
    models=np.load(out/'models.npz');sel=json.loads((out/'selection_complete.json').read_text())
    assert sha(out/'models.npz')==sel['model_sha256'];assert sha(out/'design_freeze.json')==sel['design_freeze_sha256']
    assert sel['selection_uses_evaluation_outcomes'] is False
    assert hashlib.sha256(train.target.to_numpy().tobytes()).hexdigest()==freeze['train_target_sha256']
    assert hashlib.sha256(val.target.to_numpy().tobytes()).hexdigest()==freeze['validation_target_sha256']
    assert train.target.max()+pd.Timedelta(minutes=330)<pd.Timestamp('2023-01-01')-pd.Timedelta(minutes=60)
    assert val.target.max()+pd.Timedelta(minutes=330)<pd.Timestamp('2024-01-01')-pd.Timedelta(minutes=60)
    # Independently recompute the normalized Ridge normal equations.
    xx=design_x(train,models['median'],models['mean'],models['sd'])
    yy=(train[Y].to_numpy(float)-train[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float))/SCALE
    reg=np.r_[np.ones(xx.shape[1]-1),0];ridge_error=[]
    for lam in freeze['config']['regularization_grid']:
        w=models[f'ridge_l{lam:g}'];gradient=xx.T@(xx@w-yy)/(len(xx)*12)+lam*reg[:,None]*w
        ridge_error.append(float(np.max(abs(gradient))))
    assert max(ridge_error)<1e-8
    vo=PrimaryOracle();vactual=val[Y].to_numpy(float)
    vtrue=np.array([vo.optimize(y)[1] for y in vactual]);vx=design_x(val,models['median'],models['mean'],models['sd'])
    vraw=val[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float)
    selection_errors=[]
    for candidate in sel['candidates']:
        pred=bound_paths(vraw+SCALE*(vx@models[candidate['model']]),val).astype(float)
        regret=[]
        for pp,yy,tv in zip(pred,vactual,vtrue):
            ww,_=vo.optimize(pp);regret.append(float(cost(yy)@ww-tv))
        selection_errors.append(abs(np.mean(regret)-candidate['local_reference_regret_aud']))
    assert max(selection_errors)<1e-8
    for family in ('ridge','spo'):
        winner=min([c for c in sel['candidates'] if c['family']==family],key=lambda c:(c['local_reference_regret_aud'],-c['lambda']))
        assert winner['model']==sel['selected'][family]
    source=ROOT/'results/revision_v5/history/NSW1_baseline_inputs.parquet'
    base=pd.read_parquet(source,filters=[('phase','==','evaluation')]).sort_values(['policy','target'])
    names=['R','Cstar','Ridge','SPO'];policies={'R':'raw','Cstar':'current_smooth_a050'}
    cps=json.loads((ROOT/'results/revision_v5/control/main/NSW1_current_only_selection.json').read_text())
    assert cps['selected_policy']==policies['Cstar']
    forecasts={};trajectories={};rows=[];source_records=[]
    for name in names:
        if name in policies:
            stem=f'NSW1_evaluation_c60_{policies[name]}'
            rp=ROOT/'results/revision_v5/control/main'/f'{stem}.json';tp=rp.with_suffix('.parquet')
            r=json.loads(rp.read_text());f=pd.read_parquet(tp);definition=r['policy_definition']
            assert r['solver_sha256']==freeze['script_sha256']['revision_v5_control_solver.py']
            assert r['safety_sha256']==freeze['script_sha256']['revision_v5_control_safe_policy.py']
            bp=base.loc[base.policy.eq(policies[name])].sort_values('target')
            assert np.array_equal(bp.target.to_numpy(),test.target.to_numpy())
            pred=bp[[f'p_{j+1:02d}' for j in range(12)]].to_numpy(float)
            assert hashlib.sha256(pred.tobytes()).hexdigest()==r['forecast_array_sha256']
            expected=r['net_value_aud'];seconds=r['seconds']
        else:
            family='ridge' if name=='Ridge' else 'spo';rp=out/f'{family}_evaluation_report.json';tp=out/f'{family}_evaluation_trajectory.parquet'
            r=json.loads(rp.read_text());f=pd.read_parquet(tp);definition=r['controller_definition']
            pf=pd.read_parquet(out/f'{family}_evaluation_forecasts.parquet');assert np.array_equal(pf.target.to_numpy(),test.target.to_numpy())
            pred=pf[[f'p_{j+1:02d}' for j in range(12)]].to_numpy(float)
            x=design_x(test,models['median'],models['mean'],models['sd'])
            recalc=bound_paths(test[[f'fcst_{j:02d}' for j in range(12)]].to_numpy(float)+SCALE*(x@models[sel['selected'][family]]),test).astype(float)
            assert np.array_equal(recalc,pred)
            assert sha(tp)==r['trajectory_sha256'];assert hashlib.sha256(pred.tobytes()).hexdigest()==r['forecast_sha256']
            assert r['max_primary_loss_aud']<=1e-5+2e-6 and r['max_primary_gap_aud']<2e-6
            expected=r['marked_value_aud'];seconds=r['seconds']
        err,cash,mark=physics(f,test,definition);assert abs(cash+mark-expected)<1e-6
        forecasts[name]=pred;trajectories[name]=f
        y=test[Y].to_numpy(float)
        rows.append({'policy':name,'origins':len(f),'net_cash_aud':cash,'endpoint_mark_aud':mark,'marked_value_aud':cash+mark,
         'marked_value_aud_per_974_nominal_days':(cash+mark)/974,'price_mae':float(np.abs(pred-y).mean()),
         'price_rmse':float(np.sqrt(np.mean((pred-y)**2))),'first_price_mae':float(np.abs(pred[:,0]-y[:,0]).mean()),
         'terminal_mean_mae':float(np.abs(pred[:,8:].mean(1)-y[:,8:].mean(1)).mean()),'replay_seconds':seconds})
        source_records.append({'policy':name,'trajectory_path':str(tp),'trajectory_sha256':sha(tp),'report_path':str(rp),'report_sha256':sha(rp),'physical_audit':err})
    # Full evaluation local primary-selector regret, exactly the training oracle.
    oracle=PrimaryOracle();local=[]
    for i,y in enumerate(test[Y].to_numpy(float)):
        truth,tv=oracle.optimize(y);rec={'target':test.target.iloc[i],'reference_min_cost_aud':tv}
        for name in names:
            w,_=oracle.optimize(forecasts[name][i]);regret=float(cost(y)@w-tv)
            if regret< -1e-5:raise RuntimeError('Local regret negative')
            rec[name]=max(0.,regret)
        local.append(rec)
        if (i+1)%5000==0:print(json.dumps({'audit_local_rows':i+1,'seconds':time.time()-start}),flush=True)
    local=pd.DataFrame(local);local.to_csv(out/'evaluation_local_regret_by_origin.csv',index=False)
    for row in rows:row['local_primary_selector_regret_aud_per_origin']=float(local[row['policy']].mean())
    pd.DataFrame(rows).to_csv(out/'evaluation_policy_summary.csv',index=False)
    daily=[];annual=[]
    for name,f in trajectories.items():
        days=(f.target-pd.Timedelta(minutes=30)).dt.floor('D')
        d=f.assign(day=days).groupby('day').net_aud.sum().rename(name);daily.append(d)
        for year,q in f.assign(execution_year=days.dt.year).groupby('execution_year'):
            if year==2023:continue
            annual.append({'policy':name,'execution_year':year,'origins':len(q),'net_cash_aud':float(q.net_aud.sum())})
    daily=pd.concat(daily,axis=1).sort_index();assert not daily.isna().any().any()
    interior=daily.iloc[1:-1];assert len(interior)==973
    daily.to_csv(out/'evaluation_daily_net_cash.csv');pd.DataFrame(annual).to_csv(out/'evaluation_annual_cash.csv',index=False)
    contrasts=np.column_stack([interior.SPO-interior.Ridge,interior.SPO-interior.R,interior.SPO-interior.Cstar,interior.Ridge-interior.Cstar])
    bootrows=[];n=len(interior);B=PLAN['bootstrap_draws'];rng=np.random.default_rng(PLAN['seed'])
    for L in PLAN['block_lengths_days']:
        indices=(rng.integers(0,n,size=(B,int(np.ceil(n/L)),1))+np.arange(L)[None,None,:])%n
        indices=indices.reshape(B,-1)[:,:n]
        # Chunk draws to avoid replicating a large trajectory tensor.
        bm=np.vstack([contrasts[ii].mean(1) for ii in np.array_split(indices,20)])
        observed=contrasts.mean(0);centered=bm-observed
        p=(1+(abs(centered)>=abs(observed)[None,:]).sum(0))/(B+1);hp=holm(p)
        for j,name in enumerate(PLAN['contrasts']):
            q=np.quantile(bm[:,j],[.025,.975])
            bootrows.append({'contrast':name,'block_days':L,'draws':B,'interior_days':n,
              'mean_net_cash_difference_aud_per_interior_day':observed[j],'ci95_low':q[0],'ci95_high':q[1],
              'two_sided_null_centered_p':p[j],'holm_four_contrasts_p':hp[j]})
    pd.DataFrame(bootrows).to_csv(out/'paired_daily_bootstrap.csv',index=False)
    # Independent literal resampling check for the first 20 draws of each block.
    rng2=np.random.default_rng(PLAN['seed']);maxcheck=0.
    for L in PLAN['block_lengths_days']:
        starts=rng2.integers(0,n,size=(B,int(np.ceil(n/L)),1))
        for r in range(20):
            index=[(int(a)+k)%n for a in starts[r,:,0] for k in range(L)][:n]
            direct=np.array([sum(contrasts[t,j] for t in index)/n for j in range(4)])
            vectorized=contrasts[np.array(index)].mean(0)
            maxcheck=max(maxcheck,float(np.max(abs(direct-vectorized))))
    assert maxcheck<1e-10
    report={'status':'PASS','selected':sel['selected'],'full_local_grid_origins':len(local),'policy_results':rows,
      'paired_daily_bootstrap':bootrows,'sources':source_records,'analysis_plan':PLAN,
      'independent_ridge_stationarity_max_error':max(ridge_error),'literal_bootstrap_max_error':maxcheck,
      'independent_validation_candidate_regret_max_error_aud':max(selection_errors),
      'analysis_seconds':time.time()-start,'training_seconds':sel['seconds'],'training_config':freeze['config'],
      'source_script_sha256':sha(ROOT/'scripts/revision_v8_spo.py'),'audit_script_sha256':sha(Path(__file__)),
      'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
      'boundary':'Primary-selector local regret is distinct from the deployed tolerance selector and from realized rolling value. No test-based hyperparameter selection. Conditional NSW comparison, not universal superiority.'}
    write(out/'evaluation_complete.json',report)
    print(json.dumps({'status':'PASS','policy_results':rows,'analysis_seconds':time.time()-start}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'results/revision_v8/spo');ap.add_argument('--plan-only',action='store_true');a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.plan_only:write(a.output_dir/'evaluation_analysis_plan.json',PLAN)
    else:
        assert json.loads((a.output_dir/'evaluation_analysis_plan.json').read_text())==PLAN
        run(a.output_dir)
