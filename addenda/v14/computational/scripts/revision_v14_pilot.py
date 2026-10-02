"""Locked exploratory rolling-sufficiency pilot, not a new-algorithm claim.

NSW January 2024 is selected chronologically, not by observed performance.
Training of coordinate-retention masks uses only 24 chronological 2023 origins.
The gate retains full IL streaming moments; storage savings are NOT assumed.
All outputs and numerical-failure receipts are confined to revision_v14.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, hashlib, json, time, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import revision_v5_control_solver as core
import revision_v5_control_safe_policy as safe
import revision_v5_control_scaled_policy as scaled
import revision_v5_control_enumerated_policy as enum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/revision_v14/pilot'
for module in (core, safe, scaled, enum):
    module.OUT = OUT/'numerical_failures'
P = [f'p_{j:02d}' for j in range(1,13)]
HISTORY = ROOT/'results/revision_v5/history/NSW1_inputs_aligned.parquet'
BASELINE = ROOT/'results/revision_v5/history/NSW1_baseline_inputs.parquet'
CURRENT = 'current_smooth_a050'
STOCKS = [.2, 1.8]
BINS = [.2, 1., 1.8]
EPS = .01

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False,
                                   default=str), encoding='utf-8')

def load(phase, policies):
    sources=[HISTORY] if phase=='validation' else [HISTORY, BASELINE]
    frames=[]
    for path in sources:
        f=pd.read_parquet(path, filters=[('cutoff_minutes','==',60),
                ('phase','==',phase),('policy','in',policies)],
                columns=['target','policy',*P,'actual_00'])
        frames.append(f)
    f=pd.concat(frames).drop_duplicates(['target','policy']).sort_values('target')
    if phase=='evaluation':
        f=f[(f.target>=pd.Timestamp('2024-01-01')) &
            (f.target<pd.Timestamp('2024-02-01'))]
    common=None
    result={}
    for policy in policies:
        part=f[f.policy==policy].sort_values('target').reset_index(drop=True)
        common=part.target if common is None else common
        if not part.target.equals(common): raise ValueError('unaligned origins')
        result[policy]=part
    return result

def objective(curve):
    return np.r_[np.asarray(curve,float)[:8],np.mean(curve[8:])]

def expand(v):
    return np.r_[v[:8],np.full(4,v[8])]

def compressed(current, il, mask):
    a=objective(current); b=objective(il)
    out=a.copy(); out[list(mask)]=b[list(mask)]
    return expand(out)

def primary(model, curve, stock, action=None):
    h=model.h; q=np.zeros(model.nv); x=np.asarray(curve,float)
    q[:h]=(x[:h]+model.kappa)*model.dt
    q[h:2*h]=(-x[:h]+model.kappa)*model.dt
    q[4*h]=-model.terminal_factor*float(x[h:].mean())
    model.solver.changeColBounds(3*h,float(stock),float(stock))
    for row in (model.primary_row,model.throughput_row,model.branch_row):
        model.solver.changeRowBounds(row,-np.inf,np.inf)
    if action is not None:
        model.solver.changeRowBounds(model.branch_row,float(action),float(action))
    result=model._run(q)
    if result is None: raise RuntimeError('fixed-action primary infeasible')
    return result

def certificate(model, curve, stock, action):
    t=time.perf_counter()
    _,v,b=primary(model,curve,stock)
    _,restricted,restricted_bound=primary(model,curve,stock,action)
    # Feasible constrained incumbent + unconstrained global objective bound.
    upper=max(0.,float(restricted-b))+1e-6
    return {'regret_upper_aud':upper,'reference_primary_gap_aud':max(0.,v-b),
            'fixed_action_gap_aud':max(0.,restricted-restricted_bound),
            'gate_seconds':time.perf_counter()-t}

def plan():
    if (OUT/'locked_design.json').exists(): return
    data=load('evaluation',['inverse_lead'])['inverse_lead']
    training=load('validation',[CURRENT,'inverse_lead'])
    ids=np.unique(np.linspace(0,len(training[CURRENT])-1,24,dtype=int))
    design={'stage':'exploratory feasibility pilot; all historical years previously viewed',
        'region':'NSW1','start':'2024-01-01','end_exclusive':'2024-02-01',
        'origins':len(data),'initial_soc_mwh':STOCKS,'mask_training_stock_mwh':BINS,
        'training_indices':ids.tolist(),
        'training_targets':training[CURRENT].target.iloc[ids].astype(str).tolist(),
        'mask_training_loss':'mean squared first-action deviation from IL',
        'budgets':[0,3,6,9],'closed_loop_budget':3,'primary_gate_epsilon_aud':EPS,
        'descriptive_gate_epsilons_aud':[.001,.01,.1,1.],
        'shared_state_origins':'128 equally spaced chronological indices; stocks .2,1,1.8',
        'mask_selection':'greedy coordinate addition, stable lowest-index ties; no evaluation prices',
        'reference':'inverse nominal-lead weighted history; stream exact moments',
        'strong_practical_comparator':CURRENT,
        'static_sufficiency_baseline':'8 trade coefficients and mean of4 continuation coefficients; exact structural map, not reproduction of Ye learner',
        'realized_value':'operating cash less kappa throughput, common endpoint actual-price mark',
        'cash_guarantee':'NONE; gate certifies only same-state window first-action objective regret',
        'actual_source':'six_valid_tradingprice_mean for January2024; not direct DISPATCHPRICE',
        'clock':'cutoff target minus60min; execution target minus30min; 12 half-hour leads (6h)',
        'resource_accounting':'gate retains full IL moment state, candidate current curve, lookup masks, working curves and all reference/fallback solves',
        'stop_rule':'do not scale new compression if streaming IL dominates online storage and solve overhead without meaningful guarantee',
        'sources':{str(p):{'bytes':p.stat().st_size,'sha256':digest(p)}
                   for p in (HISTORY,BASELINE)},
        'controller':enum.EnumeratedSafeLexMPC().definition()}
    write('locked_design.json',design)

def benchmark():
    plan(); data=load('evaluation',[CURRENT,'inverse_lead'])
    solver=enum.EnumeratedSafeLexMPC(); gate=core.LexMPC()
    rows=[]
    for i in range(48):
        stock=BINS[i%3]; curve=data['inverse_lead'][P].iloc[i].to_numpy(float)
        candidate=compressed(data[CURRENT][P].iloc[i].to_numpy(float),curve,[0,1,8])
        t=time.perf_counter(); a=solver.solve(candidate,stock); wall=time.perf_counter()-t
        cert=certificate(gate,curve,stock,a['u'])
        rows.append({'origin_index':i,'stock':stock,'solve_seconds':wall,**cert})
    report={'status':'PASS','cases':48,'mean_solve_seconds':np.mean([r['solve_seconds'] for r in rows]),
            'mean_gate_seconds':np.mean([r['gate_seconds'] for r in rows]),'cases_detail':rows}
    report['estimated_month_seconds_8policies_2stocks']=report['mean_solve_seconds']*1488*16+report['mean_gate_seconds']*1488*4
    write('benchmark.json',report); print(json.dumps({k:v for k,v in report.items() if k!='cases_detail'}),flush=True)

def train():
    plan(); design=json.loads((OUT/'locked_design.json').read_text())
    frames=load('validation',[CURRENT,'inverse_lead']); ids=design['training_indices']
    current=frames[CURRENT][P].to_numpy(float)[ids]
    il=frames['inverse_lead'][P].to_numpy(float)[ids]
    solver=enum.EnumeratedSafeLexMPC(); start=time.perf_counter()
    truth=np.array([[solver.solve(p,s)['u'] for p in il] for s in BINS])
    cache={}
    def losses(mask):
        key=tuple(sorted(mask))
        if key not in cache:
            actions=np.array([[solver.solve(compressed(a,b,key),s)['u']
                for a,b in zip(current,il)] for s in BINS])
            cache[key]=np.mean((actions-truth)**2,axis=1)
        return cache[key]
    selections={}
    for name,bin_index in [('global',None),*[(f'bin{j}',j) for j in range(3)]]:
        mask=[]; chain=[]
        for size in range(1,7):
            options=[]
            for j in range(9):
                if j in mask: continue
                l=losses(mask+[j]); score=float(l.mean() if bin_index is None else l[bin_index])
                options.append((score,j,l))
            score,j,l=min(options,key=lambda v:(v[0],v[1])); mask.append(j)
            chain.append({'budget':size,'retained_coordinates':sorted(mask),
                          'training_mse_by_stock':l.tolist(),'selection_loss':score})
        selections[name]=chain
    result={'status':'PASS','selection':selections,'seconds':time.perf_counter()-start,
            'unique_masks_evaluated':len(cache),'training_origins':len(ids),
            'training_only':'2023 strict validation origins; exploratory retrospective mask development'}
    write('trained_masks.json',result); print(json.dumps({k:v for k,v in result.items() if k!='selection'}),flush=True)

def mask_for(stock,budget=3,state=True):
    report=json.loads((OUT/'trained_masks.json').read_text())
    key=f'bin{int(np.argmin(abs(np.asarray(BINS)-stock)))}' if state else 'global'
    return report['selection'][key][budget-1]['retained_coordinates']

def replay(policy,initial):
    plan(); policies=['raw',CURRENT,'sparse_equal','full_equal','inverse_lead']
    frames=load('evaluation',policies); il=frames['inverse_lead'][P].to_numpy(float)
    curr=frames[CURRENT][P].to_numpy(float); prices=frames['inverse_lead'].actual_00.to_numpy(float)
    targets=frames['inverse_lead'].target
    solver=enum.EnumeratedSafeLexMPC(); gate=core.LexMPC(); stock=initial
    rows=[]; start=time.perf_counter(); gate_count=0; fallback_count=0
    direct=policy in policies
    for i in range(len(prices)):
        s0=stock; clock=time.perf_counter(); mask=[]; cert={}; fallback=False
        if direct:
            curve=frames[policy][P].iloc[i].to_numpy(float); sol=solver.solve(curve,s0)
        elif policy=='static9':
            curve=expand(objective(il[i])); sol=solver.solve(curve,s0)
        else:
            mask=mask_for(s0,3,state=policy!='global3_gate')
            curve=compressed(curr[i],il[i],mask); sol=solver.solve(curve,s0)
            if policy!='state3_no_gate':
                cert=certificate(gate,il[i],s0,sol['u']); gate_count+=1
                if cert['regret_upper_aud']>EPS:
                    sol=solver.solve(il[i],s0); fallback=True; fallback_count+=1
        elapsed=time.perf_counter()-clock; stock=sol['soc']
        cash=.5*(prices[i]*(sol['d']-sol['c'])-5*(sol['c']+sol['d']))
        rows.append({'origin_index':i,'target':targets.iloc[i], 'policy':policy,
            'initial_soc_mwh':initial,'soc_start_mwh':s0,'soc_end_mwh':stock,
            'charge_mw':sol['c'],'discharge_mw':sol['d'],'action_mw':sol['u'],
            'actual_price':prices[i],'net_aud':cash,'retained_coordinates':','.join(map(str,mask)),
            'fallback':fallback,'wall_seconds':elapsed,
            'policy_loss_upper_aud':sol['policy_optimum_loss_upper_aud'],
            'primary_gap_aud':sol['primary_objective_gap_aud'],**cert})
        if i%400==0: print(json.dumps({'policy':policy,'initial':initial,'completed':i,'origins':len(prices)}),flush=True)
    f=pd.DataFrame(rows);stem=f'{policy}_soc{initial:.1f}'
    f.to_parquet(OUT/f'{stem}.parquet',index=False)
    mark=stock*prices[-1]-initial*prices[0]
    report={'status':'PASS','policy':policy,'initial_soc_mwh':initial,'origins':len(f),
            'net_operating_aud':float(f.net_aud.sum()),'endpoint_mark_aud':float(mark),
            'marked_value_aud':float(f.net_aud.sum()+mark),'final_soc_mwh':stock,
            'gate_count':gate_count,'fallback_count':fallback_count,
            'fallback_share':fallback_count/gate_count if gate_count else None,
            'seconds':time.perf_counter()-start,'summed_origin_seconds':float(f.wall_seconds.sum()),
            'max_primary_loss_upper_aud':float(f.policy_loss_upper_aud.max()),
            'max_primary_gap_aud':float(f.primary_gap_aud.max()),
            'maximum_physical_balance_error_mwh':float(np.max(abs(f.soc_end_mwh-f.soc_start_mwh-.91*f.charge_mw*.5+f.discharge_mw*.5/.91))),
            'source':'previously evaluated chronological exploratory month; no significance/noninferiority inference'}
    write(f'{stem}.json',report); print(json.dumps(report),flush=True)

def shared():
    frames=load('evaluation',[CURRENT,'inverse_lead']); a=frames[CURRENT][P].to_numpy(float); b=frames['inverse_lead'][P].to_numpy(float)
    ids=np.unique(np.linspace(0,len(a)-1,128,dtype=int)); solver=enum.EnumeratedSafeLexMPC(); gate=core.LexMPC(); rows=[]
    for i in ids:
        for stock in BINS:
            truth=solver.solve(b[i],stock)
            for budget in (0,3,6,9):
                for state in ([True,False] if budget in (3,6) else [True]):
                    mask=[] if budget==0 else list(range(9)) if budget==9 else mask_for(stock,budget,state)
                    curve=compressed(a[i],b[i],mask); t=time.perf_counter(); sol=solver.solve(curve,stock); wall=time.perf_counter()-t
                    cert=certificate(gate,b[i],stock,sol['u'])
                    rows.append({'origin_index':int(i),'target':frames[CURRENT].target.iloc[i],
                        'stock_mwh':stock,'budget':budget,'state_dependent':state,
                        'action_difference_mw':sol['u']-truth['u'],'candidate_solve_seconds':wall,**cert})
    pd.DataFrame(rows).to_parquet(OUT/'shared_state_resource_curve.parquet',index=False)
    write('shared_state_complete.json',{'status':'PASS','origins':len(ids),'stocks':BINS,'cases':len(rows)})

def summarize():
    reports=[]
    for p in sorted(OUT.glob('*_soc*.json')): reports.append(json.loads(p.read_text()))
    f=pd.DataFrame(reports)
    for initial in STOCKS:
        ref=f[(f.policy==CURRENT)&(f.initial_soc_mwh==initial)].marked_value_aud.iloc[0]
        il=f[(f.policy=='inverse_lead')&(f.initial_soc_mwh==initial)].marked_value_aud.iloc[0]
        select=f.initial_soc_mwh==initial
        f.loc[select,'increment_vs_current_aud']=f.loc[select,'marked_value_aud']-ref
        f.loc[select,'regret_vs_il_aud']=il-f.loc[select,'marked_value_aud']
    f.to_csv(OUT/'closed_loop_summary.csv',index=False)
    shared_frame=pd.read_parquet(OUT/'shared_state_resource_curve.parquet'); summary=[]
    for (budget,state), g in shared_frame.groupby(['budget','state_dependent']):
        summary.append({'budget':int(budget),'state_dependent':bool(state),'cases':len(g),
            'material_action_change_share_gt_0_1mw':float((abs(g.action_difference_mw)>.1).mean()),
            'mean_candidate_seconds':float(g.candidate_solve_seconds.mean()),
            'mean_gate_seconds':float(g.gate_seconds.mean()),
            'certificate_acceptance':{str(e):float((g.regret_upper_aud<=e).mean()) for e in (.001,.01,.1,1.)},
            'regret_upper_median_aud':float(g.regret_upper_aud.median()),
            'regret_upper_p95_aud':float(g.regret_upper_aud.quantile(.95))})
    # IL exact online aggregation uses2 float64 moments per live target.
    # Gates require the SAME moments. Raw immutable audit archives are excluded
    # from both live-state byte counts and are never deleted.
    resource={'counting_unit':'one origin with12 live delivery targets; excludes common keys/allocator overhead symmetrically',
        'IL_moments_bytes':12*2*8,'state_gate_moments_bytes':12*2*8,
        'state_gate_masks_minimum_bytes':3*3,'global_gate_masks_minimum_bytes':3,
        'static_objective_output_bytes':9*8,'full_curve_output_bytes':12*8,
        'conclusion':'retention gate cannot reduce reference IL moment storage; additional gate solves measured above',
        'unmeasured':'transport/production systems and cold archive storage; no maintenance price claimed'}
    write('pilot_summary.json',{'status':'PASS','stage_decision':'STOP_NEW_COMPRESSION_ALGORITHM',
        'reason':'full IL moments remain necessary for this gate; no live-storage gain versus exact streaming IL. Local objective certificate is not cumulative cash guarantee.',
        'closed_loop':f.astype(object).where(pd.notna(f),None).to_dict(orient='records'),'shared_state':summary,'resource':resource,
        'next_manuscript':'narrow rolling-sufficiency conditions and failure-boundary protocol; no new superior compressor or independent holdout claim'})
    print(json.dumps({'status':'PASS','replays':len(f),'stage_decision':'STOP_NEW_COMPRESSION_ALGORITHM'}),flush=True)

def matrix():
    # Two independent single-thread processes; no large official matrix.
    policies=['raw',CURRENT,'sparse_equal','full_equal','inverse_lead','static9',
              'state3_no_gate','state3_gate','global3_gate']
    tasks=[(p,s) for p in policies for s in STOCKS]
    def run(task):
        policy,initial=task
        log=OUT/f'{policy}_soc{initial:.1f}.log'
        with log.open('w',encoding='utf-8') as handle:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                'replay','--policy',policy,'--initial',str(initial)],
                stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
        if process.returncode: raise RuntimeError(f'{task} failed; inspect {log}')
        print(json.dumps({'complete':policy,'initial':initial}),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,tasks))
    print(json.dumps({'status':'PASS','completed_replays':len(tasks)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','benchmark','train','replay','shared','summarize','matrix'])
    ap.add_argument('--policy');ap.add_argument('--initial',type=float)
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='replay': replay(args.policy,args.initial)
    else: globals()[args.mode]()
