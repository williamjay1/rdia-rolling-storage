"""Executed, small conditional-theory checks; never a NEM policy replay.

Freeze recipe and source hash with --plan-only before execution. Reads existing
NEM inputs only for an explicitly non-oracle tail-price surrogate diagnostic.
Every new output is confined to results/revision_v8/theory.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v8/theory/verified_run'
SEED = 20261008
TOL = 1e-8
LP_CALLS = 0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')

def csv_dump(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def solve(c, X, *, maximize=True, near=None):
    global LP_CALLS
    A, b, bounds = X
    kwargs = {}
    if near is not None:
        objective, threshold = near
        kwargs.update(A_ub=-np.asarray(objective)[None, :], b_ub=[-threshold])
    r = linprog(-c if maximize else c, A_eq=A, b_eq=b, bounds=bounds,
                method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                        'dual_feasibility_tolerance': 1e-9}, **kwargs)
    LP_CALLS += 1
    if not r.success:
        raise RuntimeError(r.message)
    return float(np.dot(c, r.x)), r.x

def finite_checks(rng):
    rows, inputs = [], []
    for k in range(500):
        n = int(rng.integers(2, 25))
        F, B, h, j = rng.normal(size=(4, n))
        G = F + h
        eps = float(rng.uniform(0, .6) * np.ptp(G))
        ix = int(rng.choice(np.flatnonzero(G >= G.max() - eps)))
        star = int(np.argmax(F)); omega = float(np.ptp(h))
        loss = float(F.max() - F[ix])
        one_sided = float(eps + h[ix] - h[star])
        common_distortion = float((F+h).max()-F.max()-((B+h).max()-B.max()))
        different_distortion = float((F+h).max()-F.max()-((B+j).max()-B.max()))
        interval_lower = float(h.min()-j.max()); interval_upper = float(h.max()-j.min())
        ref_set = F >= F.max()-eps
        pert_set = G >= G.max()-eps
        included_forward = bool(np.all(F[pert_set] >= F.max()-eps-omega-TOL))
        included_reverse = bool(np.all(G[ref_set] >= G.max()-eps-omega-TOL))
        assert loss <= eps+omega+TOL and loss <= one_sided+TOL
        assert abs(common_distortion) <= omega+TOL
        assert interval_lower-TOL <= different_distortion <= interval_upper+TOL
        assert included_forward and included_reverse
        # Scalar continuation errors have a sharper midpoint interval bound.
        s = rng.uniform(.2, 1.8, n); s[0] = .2; s[-1] = 1.8
        da, db = rng.uniform(-3, 3, 2)
        coef_dist = float((F+da*s).max()-F.max()-((B+db*s).max()-B.max()))
        center_bound = float(abs(da-db)*1.0+(abs(da)+abs(db))*.8)
        legacy_bound = float((abs(da)+abs(db))*1.8)
        assert abs(coef_dist) <= center_bound+TOL
        rows.append({'trial': k, 'plans': n, 'epsilon': eps, 'omega_h': omega,
                     'regret': loss, 'one_sided_bound': one_sided,
                     'oscillation_bound': eps+omega,
                     'forward_inclusion': included_forward,
                     'reverse_inclusion': included_reverse,
                     'common_distortion': common_distortion,
                     'different_distortion': different_distortion,
                     'different_interval_lower': interval_lower,
                     'different_interval_upper': interval_upper,
                     'coefficient_distortion': coef_dist,
                     'midpoint_bound': center_bound, 'legacy_bound': legacy_bound})
        inputs.append({'trial': k, 'F': F.tolist(), 'B': B.tolist(),
                       'h': h.tolist(), 'j': j.tolist(), 'chosen_index': ix,
                       's': s.tolist(), 'delta_A': float(da), 'delta_B': float(db)})
    csv_dump('finite_objective_checks.csv', rows); dump('finite_inputs.json', inputs)
    return {'trials': len(rows), 'failures': 0,
            'max_regret_minus_oscillation_bound': max(r['regret']-r['oscillation_bound'] for r in rows),
            'max_regret_minus_one_sided_bound': max(r['regret']-r['one_sided_bound'] for r in rows),
            'max_common_distortion_excess': max(abs(r['common_distortion'])-r['omega_h'] for r in rows),
            'max_midpoint_bound_excess': max(abs(r['coefficient_distortion'])-r['midpoint_bound'] for r in rows)}

def physical_set(E, P, horizon, s0, eta, step=.5):
    n = horizon; A = np.zeros((n, 3*n)); b = np.zeros(n); b[0] = s0
    for i in range(n):
        A[i, i] = -eta*step; A[i, n+i] = step/eta; A[i, 2*n+i] = 1
        if i: A[i, 2*n+i-1] = -1
    return (A, b, [(0., P)]*(2*n)+[(.1*E, .9*E)]*n)

def reachable_checks():
    rows = []
    for E in (1., 2., 4.):
        for n in (1, 2, 8):
            for fraction in (.1, .5, .9):
                for eta in (.85, .91, .95):
                    P, step, s0 = 1., .5, fraction*E
                    lo=max(.1*E, s0-n*P*step/eta)
                    hi=min(.9*E, s0+n*eta*P*step)
                    X=physical_set(E,P,n,s0,eta,step)
                    c=np.zeros(3*n); c[-1]=1
                    vmin,_=solve(c,X,maximize=False); vmax,_=solve(c,X)
                    # Independent exclusive, monotone feasible constructions.
                    low_state=high_state=s0; low_actions=[]; high_actions=[]
                    for _ in range(n):
                        d=min(P, max(0., (low_state-.1*E)*eta/step))
                        ch=min(P, max(0., (.9*E-high_state)/(eta*step)))
                        low_state-=d*step/eta; high_state+=eta*ch*step
                        low_actions.append([0.,d]); high_actions.append([ch,0.])
                    error=max(abs(vmin-lo),abs(vmax-hi),abs(low_state-lo),abs(high_state-hi))
                    assert error <= TOL
                    rows.append({'energy_mwh':E,'power_mw':P,'horizon_periods':n,
                                 'step_hours':step,'initial_soc_mwh':s0,'eta_c':eta,'eta_d':eta,
                                 'reachable_lower_mwh':lo,'reachable_upper_mwh':hi,
                                 'reachable_amplitude_mwh':hi-lo,'nominal_soc_amplitude_mwh':.8*E,
                                 'LP_lower_mwh':vmin,'LP_upper_mwh':vmax,
                                 'exclusive_monotone_lower_mwh':low_state,
                                 'exclusive_monotone_upper_mwh':high_state,'max_error':error,
                                 'exclusive_low_actions':json.dumps(low_actions),
                                 'exclusive_high_actions':json.dumps(high_actions)})
    csv_dump('reachable_terminal_checks.csv',rows)
    return {'cases':len(rows),'LP_extreme_solves':len(rows)*2,'failures':0,
            'max_formula_LP_or_exclusive_error':max(r['max_error'] for r in rows),
            'baseline_E2_h8_eta091_amplitude_mwh':1.6,
            'scope':'LP superset extrema are achieved by explicit exclusive monotone paths; therefore extrema also exact under mode exclusivity. No extra throughput/terminal/network constraints.'}

def ideal_set(n, s0=1.):
    A=np.zeros((n,2*n)); b=np.zeros(n); b[0]=s0
    for i in range(n):
        A[i,i]=.5; A[i,n+i]=1.
        if i:A[i,n+i-1]=-1.
    return (A,b,[(-1.,1.)]*n+[(.2,1.8)]*n)

def storage_checks(rng):
    rows, inputs=[] ,[]
    for k in range(60):
        n=1+k%3; X=ideal_set(n)
        F=np.r_[.5*rng.normal(0,10,n),np.zeros(n)]
        F[-1]=rng.normal(0,3)
        trade=np.r_[.5*rng.normal(0,2,n),np.zeros(n)]
        tail=np.zeros(2*n);tail[-1]=rng.normal(0,2)
        h=trade+tail; G=F+h
        Vf,xf=solve(F,X); Vg,xg=solve(G,X)
        maxh,_=solve(h,X);minh,_=solve(h,X,maximize=False);omega=maxh-minh
        mt,_=solve(trade,X);nt,_=solve(trade,X,maximize=False)
        ms,_=solve(tail,X);ns,_=solve(tail,X,maximize=False)
        eps=.01
        lowest_F,_=solve(F,X,maximize=False,near=(G,Vg-eps))
        lowest_G,_=solve(G,X,maximize=False,near=(F,Vf-eps))
        worst_forward=Vf-lowest_F;worst_reverse=Vg-lowest_G
        loss=Vf-float(F@xg); directional=float(h@(xg-xf))
        # Scalar first-action range: endpoints of a compact convex superlevel set.
        action=np.zeros(2*n);action[0]=1.
        lowG,_=solve(action,X,maximize=False,near=(G,Vg-eps))
        highG,_=solve(action,X,near=(G,Vg-eps))
        lowExpanded,_=solve(action,X,maximize=False,near=(F,Vf-eps-omega))
        highExpanded,_=solve(action,X,near=(F,Vf-eps-omega))
        assert loss <= omega+TOL and loss <= directional+TOL
        assert worst_forward <= eps+omega+TOL and worst_reverse <= eps+omega+TOL
        assert lowG >= lowExpanded-TOL and highG <= highExpanded+TOL
        assert omega <= (mt-nt)+(ms-ns)+TOL
        rows.append({'trial':k,'horizon':n,'regret':loss,'one_sided_bound':directional,
                     'joint_oscillation':omega,'sum_channel_oscillations':mt-nt+ms-ns,
                     'epsilon':eps,'worst_forward_regret':worst_forward,'worst_reverse_regret':worst_reverse,
                     'perturbed_action_low':lowG,'perturbed_action_high':highG,
                     'expanded_reference_low':lowExpanded,'expanded_reference_high':highExpanded})
        inputs.append({'trial':k,'horizon':n,'initial_soc':1.,'F':F.tolist(),
                       'trade_perturbation':trade.tolist(),'terminal_perturbation':tail.tolist()})
    csv_dump('storage_objective_checks.csv',rows);dump('storage_inputs.json',inputs)
    sharp=[]
    for n,s0 in ((1,.2),(8,1.)):
        X=physical_set(2.,1.,n,s0,.91); c=np.zeros(3*n);c[-1]=1.
        lo,_=solve(c,X,maximize=False);hi,_=solve(c,X)
        delta=1.;z=1e-6;F=-(delta-z)*c;G=z*c
        vf,xf=solve(F,X);vg,xg=solve(G,X)
        regret=vf-float(F@xg);bound=delta*(hi-lo)
        assert abs(regret/bound-(1-z))<=TOL
        sharp.append({'horizon':n,'initial_soc':s0,'eta':.91,'reachable_amplitude':hi-lo,
                      'delta':delta,'z':z,'regret':regret,'bound':bound,'ratio':regret/bound})
    dump('sharpness_checks.json',sharp)
    return {'trials':len(rows),'failures':0,'physical_terminal_sharpness_cases':sharp,
            'max_worst_set_regret_excess':max(max(r['worst_forward_regret'],r['worst_reverse_regret'])-r['epsilon']-r['joint_oscillation'] for r in rows),
            'scope':'Ideal convex storage with single net-power controls, efficiency one; not a market-calibrated replay or a replacement for the NEM exclusive-mode policy.'}

def quadratic_checks(rng):
    rows=[];inputs=[]
    for k in range(100):
        n=int(rng.integers(1,7)); q=rng.uniform(.5,4,n)
        center=rng.uniform(-.4,.4,n); b=rng.normal(0,.25,n)
        optimum=np.clip(center+b/q,-1,1)
        chosen=np.clip(optimum+rng.normal(0,.05,n),-1,1)
        f=lambda x: -.5*float(np.sum(q*(x-center)**2))
        g=lambda x:f(x)+float(b@x)
        eps=g(optimum)-g(chosen); mu=float(q.min());L=float(np.linalg.norm(b))
        regret=f(center)-f(chosen); bound=(np.sqrt(max(0,eps))+L/np.sqrt(2*mu))**2
        assert regret <= bound+TOL
        rows.append({'trial':k,'dimension':n,'mu':mu,'L':L,'epsilon':eps,
                     'reference_regret':regret,'quadratic_bound':bound,
                     'oscillation_bound':float(2*np.abs(b).sum()+eps)})
        inputs.append({'trial':k,'q':q.tolist(),'reference_center':center.tolist(),
                       'linear_perturbation':b.tolist(),'perturbed_optimum':optimum.tolist(),
                       'epsilon_optimal_choice':chosen.tolist()})
    sharp=[]
    mu,L=2.,.4
    for eps in (0.,.005,.25):
        xhat=L/mu+np.sqrt(2*eps/mu);regret=mu*xhat*xhat/2
        bound=(np.sqrt(eps)+L/np.sqrt(2*mu))**2
        assert abs(regret-bound)<TOL
        sharp.append({'mu':mu,'L':L,'epsilon':eps,'xstar':0.,'xhat':xhat,
                      'reference_regret':regret,'quadratic_bound':bound})
    csv_dump('strong_concavity_checks.csv',rows);dump('strong_concavity_inputs.json',inputs)
    dump('strong_concavity_sharpness.json',sharp)
    return {'trials':len(rows),'sharp_cases':sharp,'failures':0,
            'max_bound_excess':max(r['reference_regret']-r['quadratic_bound'] for r in rows),
            'scope':'Separate strongly concave box-quadratic controls; this assumption is absent from the article storage LP/MIP.'}

def linear_range(slope,eps):
    if slope>0:return [max(0.,1-eps/slope),1.]
    if slope<0:return [0.,min(1.,eps/abs(slope))]
    return [0.,1.]

def projection_controls(rng):
    controls=[]
    def add(name,inputs,findings):controls.append({'case':name,'inputs':inputs,'findings':findings})
    a=linear_range(-.005,.01);b=linear_range(.005,.01)
    assert a==b==[0.,1.]
    add('overlap_does_not_imply_selector_only',{'slopes':[-.005,.005],'epsilon':.01},
        {'ranges':[a,b],'unique_optimal_actions':[0.,1.],
         'reason':'Both ranges overlap completely although the distinct primary objectives uniquely prefer opposite actions.'})
    add('identical_flat_primary_selector_only',{'slopes':[0.,0.],'epsilon':.01},
        {'ranges':[[0.,1.],[0.,1.]],'chosen_secondary_actions':[0.,1.],
         'reason':'Identical flat objectives establish selector-only behavior; overlap alone did not establish it.'})
    outer=[linear_range(0.,2.01),linear_range(0.,2.01)]
    true=[linear_range(2.,.01),linear_range(-2.,.01)]
    assert true[0][0]>true[1][1]
    add('expanded_overlap_not_sufficient_for_true_overlap',{'observed_slopes':[0.,0.],
        'reference_slopes':[2.,-2.],'epsilon':.01,'oscillation_bounds':[2.,2.]},
        {'expanded_observed_ranges':outer,'reference_ranges':true,
         'reason':'Expanded observed envelopes overlap but the true near-optimal actions are disjoint.'})
    original=[linear_range(1.,.1),linear_range(-1.,.1)]
    inflated=[linear_range(1.,1.1),linear_range(-1.,1.1)]
    add('original_separation_not_robust_to_admitted_error',{'observed_slopes':[1.,-1.],
        'reference_slopes':[0.,0.],'epsilon':.1,'oscillation_bounds':[1.,1.]},
        {'original_observed_ranges':original,'expanded_observed_ranges':inflated,
         'reference_ranges':[[0.,1.],[0.,1.]]})
    robust=[linear_range(1.,.15),linear_range(-1.,.15)]
    tested_gaps=[]
    for _ in range(200):
        error=rng.uniform(-.05,.05,2)
        refA=linear_range(1.-error[0],.1);refB=linear_range(-1.-error[1],.1)
        gap=refA[0]-refB[1];tested_gaps.append(gap)
        assert gap>=robust[0][0]-robust[1][1]-TOL
    add('expanded_separation_sufficient_for_robust_response',{'observed_slopes':[1.,-1.],
        'epsilon':.1,'oscillation_bounds':[.05,.05],'error_pairs_drawn':200},
        {'expanded_ranges':robust,'certified_minimum_gap':.7,'minimum_actual_reference_gap':min(tested_gaps)})
    add('contracted_overlap_sufficient_for_common_action',{'observed_slopes':[0.,0.],
        'reference_slopes':[.02,-.02],'epsilon':.1,'oscillation_bounds':[.02,.02]},
        {'contracted_observed_ranges':[[0.,1.],[0.,1.]],
         'reference_ranges':[linear_range(.02,.1),linear_range(-.02,.1)],
         'reason':'Inner intervals overlap and lie inside both reference epsilon-optimal projections.'})
    actions=[0,1,2,3];FA=[0,-1,0,-1];FB=[-1,0,-1,0]
    add('nonconvex_envelope_overlap_is_inconclusive',{'actions':actions,'FA':FA,'FB':FB,'epsilon':0.},
        {'actual_optimal_action_sets':[[0,2],[1,3]],'outer_envelopes':[[0,2],[1,3]],
         'outer_overlap':[1,2],'actual_set_intersection':[]})
    # First-window objectives coincide; later update and realized reward vary.
    M=1000.;raw_cash=M;history_cash=0.
    add('single_window_bound_does_not_bound_dynamic_branch',
        {'initial_soc':1.,'raw_trading':[0.,2.],'history_trading':[0.,0.],
         'raw_terminal':[1.,0.],'history_terminal':[1.,1.],
         'realized_prices':[1.,M],'endpoint_mark':0.},
        {'first_window_oscillation':0.,'first_actions':[0.,0.],
         'raw_cash':raw_cash,'history_cash':history_cash,'branch_difference':-M,
         'reason':'Arbitrary M yields arbitrary later cash despite zero first-window perturbation; updating needs its own evaluation.'})
    dump('projection_and_dynamic_controls.json',controls)
    return {'controls':len(controls),'robust_error_pairs':200,'failures':0,
            'dynamic_counterexample_branch_difference':-M}

def observed_surrogates():
    rows=[];sources=[]
    for region in ('NSW1','QLD1','SA1','TAS1','VIC1'):
        frames={}
        for policy in ('raw','sparse_equal','inverse_lead'):
            fn=region+('_baseline_inputs.parquet' if policy!='inverse_lead' else '_inputs_aligned.parquet')
            path=ROOT/'results/revision_v5/history'/fn
            cols=['target']+[f'p_{h:02d}' for h in range(9,13)]+[f'actual_{h:02d}' for h in range(8,12)]
            f=pd.read_parquet(path,columns=cols,filters=[('policy','==',policy),('cutoff_minutes','==',60),
                                                       ('target','>=',pd.Timestamp('2024-01-01'))])
            f=f.sort_values('target').reset_index(drop=True)
            assert len(f)==46741 and not f.target.duplicated().any()
            frames[policy]=f
            if not any(s['path']==str(path.relative_to(ROOT)) for s in sources):
                sources.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
        R=frames['raw'];actual=R[[f'actual_{h:02d}' for h in range(8,12)]].to_numpy(float).mean(axis=1)
        raw=R[[f'p_{h:02d}' for h in range(9,13)]].to_numpy(float).mean(axis=1)
        eR=raw-actual
        for policy in ('sparse_equal','inverse_lead'):
            H=frames[policy];assert H.target.astype('datetime64[ns]').equals(R.target.astype('datetime64[ns]'))
            Ha=H[[f'actual_{h:02d}' for h in range(8,12)]].to_numpy(float).mean(axis=1)
            assert np.max(abs(Ha-actual))<TOL
            alt=H[[f'p_{h:02d}' for h in range(9,13)]].to_numpy(float).mean(axis=1)
            eH=alt-actual;diff=eH-eR
            assert np.max(abs(diff-(alt-raw)))<TOL
            rows.append({'region':region,'alternative':policy,'origins':len(R),
                         'first_target':str(R.target.iloc[0]),'last_target':str(R.target.iloc[-1]),
                         'surrogate_error_correlation':float(np.corrcoef(eR,eH)[0,1]),
                         'raw_surrogate_error_rmse':float(np.sqrt(np.mean(eR**2))),
                         'alternative_surrogate_error_rmse':float(np.sqrt(np.mean(eH**2))),
                         'pairwise_error_difference_rmse':float(np.sqrt(np.mean(diff**2))),
                         'pointwise_equal_fraction_tol_1e10':float(np.mean(abs(diff)<=1e-10)),
                         'max_abs_error_difference':float(np.max(abs(diff)))})
    csv_dump('observed_tail_price_surrogate.csv',rows);dump('observed_surrogate_sources.json',sources)
    return {'rows':len(rows),'origins_per_asset':46741,'assets':5,'sources':sources,
            'definition':'gamma=1, issued mean prices p09:p12 minus realized mean actual08:actual11 at the same origin.',
            'not_true_continuation_oracle':True,'no_hypothesis_test_of_common_true_error':True,
            'scope':'Pure observed tail-price surrogate diagnostic; future prices are used only for retrospective error reporting, never optimizer actions or fitting.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);design_path=OUT/'design.json'
    design={'seed':SEED,'script_sha256':sha(__file__),'finite_trials':500,
            'reachable_cases':81,'storage_trials':60,'quadratic_trials':100,
            'projection_controls':8,'robust_error_pairs':200,'tolerance':TOL,
            'observed_surrogates':True,'synthetic_not_observed_market':True,
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.plan_only:
        if design_path.exists():raise FileExistsError(design_path)
        dump('design.json',design);print(json.dumps({'status':'DESIGN_FROZEN','path':str(design_path)}));return
    if (OUT/'execution_receipt.json').exists():raise FileExistsError('Existing execution receipt must not be overwritten.')
    frozen=json.loads(design_path.read_text(encoding='utf-8'))
    assert frozen['script_sha256']==sha(__file__) and frozen['seed']==SEED
    start=time.perf_counter();rng=np.random.default_rng(SEED)
    sections={'finite':finite_checks(rng),'reachable':reachable_checks(),
              'storage':storage_checks(rng),'strong_concavity':quadratic_checks(rng),
              'projection_controls':projection_controls(rng),'observed_surrogates':observed_surrogates()}
    outputs=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
             for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='execution_receipt.json']
    receipt={'status':'PASS','completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
             'elapsed_seconds':time.perf_counter()-start,'python_version':platform.python_version(),
             'python_executable':sys.executable,'numpy_version':np.__version__,'scipy_version':scipy.__version__,
             'script_sha256':sha(__file__),'design_sha256':sha(design_path),'seed':SEED,
             'LP_calls_executed':LP_CALLS,'sections':sections,'output_manifest':outputs,
             'NEM_policy_solver_or_training_executed':False,'pre_v8_files_modified':False}
    dump('execution_receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds'],
                      'LP_calls_executed':LP_CALLS,'output':str(OUT)},ensure_ascii=False))

if __name__=='__main__':main()
