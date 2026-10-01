"""Additive, noncausal v13 action/certificate diagnostics; frozen v5 unchanged.

Full-sample material-threshold counts reuse saved epsilon=.01 outer ranges.
A chronology-only fixed subsample is selected and saved before new binary MILPs.
No manuscript, original input, original diagnostic or raw archive is overwritten.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, datetime as dt, hashlib, json, time, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from revision_v5_control_solver import LexMPC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v13/action'
REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
HISTORIES = ['sparse_equal', 'inverse_lead']
EPSILONS = [.001, .01, .1]
MATERIAL = [.05, .1, .2]
P = [f'p_{i:02d}' for i in range(1, 13)]
MANIFEST = {}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def source(path):
    path = Path(path)
    MANIFEST[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size,
                                                 'sha256': digest(path)}
    return path

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   default=str, allow_nan=False), encoding='utf-8')

def binary_model():
    return LexMPC(relax_if_mode_feasible=False)

def reset(model, soc, primary=None):
    model.solver.changeColBounds(3*model.h, float(soc), float(soc))
    for row in (model.primary_row, model.throughput_row, model.branch_row):
        model.solver.changeRowBounds(row, -np.inf, np.inf)
    if primary is not None:
        for j in model.primary_indices:
            model.solver.changeCoeff(model.primary_row, int(j), float(primary[j]))

def primary_cost(model, curve):
    h = model.h; curve = np.asarray(curve, float); q = np.zeros(model.nv)
    q[:h] = (curve[:h]+model.kappa)*model.dt
    q[h:2*h] = (-curve[:h]+model.kappa)*model.dt
    q[4*h] = -model.terminal_factor*curve[h:].mean()
    return q

def range_only(model, curve, soc, eps):
    """Binary primary solve then global min/max net-action dual-bound envelope."""
    q = primary_cost(model, curve); reset(model, soc, q)
    x, incumbent, bound = model._run(q)
    model.solver.changeRowBounds(model.primary_row, -np.inf, incumbent+eps)
    lx, li, lb = model._run(model.first_cost)
    ux, ui, ub = model._run(-model.first_cost)
    return {'range_lower_outer_mw': float(lb), 'range_upper_outer_mw': float(-ub),
            'range_lower_incumbent_mw': float(li),
            'range_upper_incumbent_mw': float(-ui),
            'primary_incumbent_reward_aud': float(-incumbent),
            'primary_gap_aud': max(0., float(incumbent-bound)),
            'lower_branch_gap_mw': max(0., float(li-lb)),
            'upper_branch_gap_mw': max(0., float(ui-ub)),
            'primary_epsilon_aud': float(eps)}

def counts(frame, threshold):
    changed = abs(frame.u_history_at_raw_soc-frame.u_raw_at_raw_soc)>threshold
    separation = np.maximum(frame.raw_range_lower_outer_mw-frame.history_range_upper_outer_mw,
                            frame.history_range_lower_outer_mw-frame.raw_range_upper_outer_mw)
    forced = separation>threshold
    overlap = (frame.raw_range_lower_outer_mw<=frame.history_range_upper_outer_mw+1e-7)&(
        frame.history_range_lower_outer_mw<=frame.raw_range_upper_outer_mw+1e-7)
    return {'origins': len(frame), 'action_changes': int(changed.sum()),
            'separated_among_changed': int((changed&forced).sum()),
            'separated_all_origins': int(forced.sum()),
            'overlap_among_changed': int((changed&overlap).sum()),
            'smaller_gap_nonoverlap_among_changed': int((changed&~forced&~overlap).sum()),
            'separated_share_among_changed': float((changed&forced).sum()/changed.sum()) if changed.any() else None}

def inputs(region, policy):
    family = 'baseline_inputs' if policy in ('raw','sparse_equal') else 'inputs_aligned'
    path = ROOT/f'results/revision_v5/history/{region}_{family}.parquet'
    if path.relative_to(ROOT).as_posix() not in MANIFEST: source(path)
    f = pd.read_parquet(path, columns=['target', *P], filters=[('policy','==',policy),
          ('cutoff_minutes','==',60), ('phase','==','evaluation')])
    f['target']=f.target.astype('datetime64[ns]')
    return f.sort_values('target').reset_index(drop=True)

def full_and_plan(n):
    frames = {}; rows = []; plan = []
    for region in REGIONS:
        for history in HISTORIES:
            p = source(ROOT/f'results/revision_v5/control/diagnostics/{region}_raw_{history}_common_state.parquet')
            f = pd.read_parquet(p).sort_values('sample_index').reset_index(drop=True)
            assert len(f)==1461
            assert np.allclose(f.raw_primary_epsilon_aud,.01)
            assert np.allclose(f.history_primary_epsilon_aud,.01)
            frames[region, history] = f
            ix = np.unique(np.round(np.linspace(0, len(f)-1, n)).astype(int))
            for pos in ix:
                q=f.iloc[pos]
                plan.append({'region':region, 'history':history, 'diagnostic_row':int(pos),
                             'sample_index':int(q.sample_index), 'target':q.target,
                             'raw_soc_mwh':float(q.raw_soc_mwh)})
            for t in MATERIAL:
                rows.append({'region':region,'history':history,'epsilon_aud':.01,
                             'material_threshold_mw':t, 'scope':'full1461_saved_ranges', **counts(f,t)})
    full=pd.DataFrame(rows)
    for history in HISTORIES:
        for t in MATERIAL:
            f=pd.concat([frames[r,history] for r in REGIONS],ignore_index=True)
            rows.append({'region':'five_assets','history':history,'epsilon_aud':.01,
                         'material_threshold_mw':t,'scope':'full7305_saved_ranges',**counts(f,t)})
    pd.DataFrame(rows).to_csv(OUT/'material_threshold_full_sample.csv', index=False)
    pd.DataFrame(plan).to_csv(OUT/'fixed_subsample_plan.csv',index=False)
    write('design.json', {'status':'fixed_before_new_MILP_outputs',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'selection':'round(linspace(0,1460,n)) on chronological existing stride32 diagnostic rows; no selection using price, action change or outcome',
        'n_per_region_per_history':n,'epsilons_aud':EPSILONS,'material_thresholds_mw':MATERIAL,
        'controller_primary_allowance_aud':1e-5,'epsilon_multipliers':[100,1000,10000],
        'projection_model':'all charge/discharge modes binary at all three new diagnostic MILP stages',
        'primary_envelope':'incumbent cost+epsilon; dual first-action bounds are numerical outer endpoints',
        'fresh_subsample_only':True,'original_full_sample_unmodified':True,
        'D_free_bytes':shutil.disk_usage(ROOT).free,'plan_sha256':digest(OUT/'fixed_subsample_plan.csv')})
    return frames, pd.DataFrame(plan)

def sensitivity(frames, plan):
    rows=[]; started=time.perf_counter(); model=binary_model()
    for region in REGIONS:
        raw=inputs(region,'raw')
        ps=plan[(plan.region==region)&(plan.history==HISTORIES[0])]
        raw_ranges={}
        for q in ps.itertuples(index=False):
            assert raw.target.iloc[q.sample_index]==q.target
            curve=raw.loc[q.sample_index,P].to_numpy(float)
            for eps in EPSILONS: raw_ranges[q.sample_index,eps]=range_only(model,curve,q.raw_soc_mwh,eps)
        for history in HISTORIES:
            historical=inputs(region,history); assert raw.target.equals(historical.target)
            for q in plan[(plan.region==region)&(plan.history==history)].itertuples(index=False):
                base=frames[region,history].iloc[q.diagnostic_row]
                curve=historical.loc[q.sample_index,P].to_numpy(float)
                for eps in EPSILONS:
                    rr=raw_ranges[q.sample_index,eps]; hr=range_only(model,curve,q.raw_soc_mwh,eps)
                    row={'region':region,'history':history,'target':q.target,
                         'sample_index':q.sample_index,'raw_soc_mwh':q.raw_soc_mwh,'epsilon_aud':eps,
                         'u_raw_at_raw_soc':float(base.u_raw_at_raw_soc),
                         'u_history_at_raw_soc':float(base.u_history_at_raw_soc)}
                    row.update({f'raw_{k}':v for k,v in rr.items()})
                    row.update({f'history_{k}':v for k,v in hr.items()})
                    if eps==.01:
                        row['max_baseline_endpoint_difference_mw']=max(abs(row[f'{a}_{b}']-base[f'{a}_{b}'])
                          for a in ('raw','history') for b in ('range_lower_outer_mw','range_upper_outer_mw'))
                    rows.append(row)
            print(json.dumps({'completed_region_history':f'{region}/{history}','rows':len(rows),
                              'seconds':time.perf_counter()-started}),flush=True)
            pd.DataFrame(rows).to_parquet(OUT/'epsilon_subsample_ranges_partial.parquet',index=False)
    f=pd.DataFrame(rows); f.to_parquet(OUT/'epsilon_subsample_ranges.parquet',index=False)
    summaries=[]
    for (region,history,eps),g in f.groupby(['region','history','epsilon_aud']):
        for t in MATERIAL:
            summaries.append({'region':region,'history':history,'epsilon_aud':eps,
                              'material_threshold_mw':t,'scope':'fresh_fixed_subsample',**counts(g,t)})
    for (history,eps),g in f.groupby(['history','epsilon_aud']):
        for t in MATERIAL:
            summaries.append({'region':'five_assets','history':history,'epsilon_aud':eps,
                              'material_threshold_mw':t,'scope':'fresh_fixed_subsample',**counts(g,t)})
    pd.DataFrame(summaries).to_csv(OUT/'epsilon_material_subsample_counts.csv',index=False)
    maxgap=max(float(f.raw_primary_gap_aud.max()),float(f.history_primary_gap_aud.max()))
    monotonic=[]
    for (_,__,___),g in f.groupby(['region','history','sample_index']):
        g=g.sort_values('epsilon_aud')
        for prefix in ('raw','history'):
            monotonic.extend(np.diff(g[f'{prefix}_range_lower_outer_mw'].to_numpy())>1e-6)
            monotonic.extend(np.diff(g[f'{prefix}_range_upper_outer_mw'].to_numpy()) < -1e-6)
    validation={'status':'PASS' if not any(monotonic) else 'REPAIR_REQUIRED',
        'rows':len(f),'range_calculations':len(REGIONS)*len(EPSILONS)*len(ps)*3,
        'all_binary':True,'max_primary_gap_aud':maxgap,
        'max_baseline_endpoint_difference_mw':float(f.max_baseline_endpoint_difference_mw.max()),
        'nested_epsilon_violations_gt1e-6':int(sum(monotonic)),
        'seconds':time.perf_counter()-started,'fresh_scope':'125 common states per history; not all1461 per region'}
    write('epsilon_sensitivity_validation.json',validation)
    return validation

def audit_witness(model, x, soc):
    h=model.h
    balance=x[3*h+1:]-x[3*h:4*h]-model.eta*model.dt*x[:h]+model.dt*x[h:2*h]/model.eta
    return {'balance_max_abs_mwh':float(abs(balance).max()),
            'initial_soc_abs_error_mwh':float(abs(x[3*h]-soc)),
            'max_simultaneous_charge_discharge_mw':float(np.minimum(x[:h],x[h:2*h]).max()),
            'soc_bound_violation_mwh':float(max(0, model.smin-x[3*h:].min(),x[3*h:].max()-model.smax)),
            'power_bound_violation_mw':float(max(0,-x[:2*h].min(),x[:2*h].max()-model.power))}

def sa_certificate():
    p=source(ROOT/'results/revision_v5/control/diagnostics/SA1_posthoc_20240923_1900_common_soc.json')
    saved=json.loads(p.read_text()); raw=np.asarray(saved['forecast_paths']['raw'])
    hist=np.asarray(saved['forecast_paths']['sparse_equal']); soc=float(saved['common_soc_mwh'])
    model=binary_model(); delta=hist-raw
    hc=np.zeros(model.nv); h=model.h
    hc[:h]=-model.dt*delta[:h];hc[h:2*h]=model.dt*delta[:h]
    hc[4*h]=model.terminal_factor*delta[h:].mean()
    reset(model,soc); lx,li,lb=model._run(hc)
    reset(model,soc); ux,ui,ub=model._run(-hc)
    maximum=-ui;maximum_outer=-ub;omega_outer=maximum_outer-lb
    ranges={name:{str(e):range_only(model,curve,soc,e) for e in EPSILONS}
            for name,curve in [('raw',raw),('sparse_equal',hist)]}
    widened=range_only(model,raw,soc,.01+omega_outer)
    witness_audits={'minimum':audit_witness(model,lx,soc),'maximum':audit_witness(model,ux,soc)}
    assert max(v for a in witness_audits.values() for v in a.values())<1e-6
    result={'status':'PASS','region':'SA1','target':saved['target'],'event_id':178,
            'event_step':8,'posthoc_illustrative':True,'common_soc_mwh':soc,
            'h_definition':'sum first8(dt*(S-R)*(d-c)) + mean(last4(S-R))*terminal_SOC; wear cancels',
            'h_min_incumbent_aud':float(li),'h_min_outer_lower_aud':float(lb),
            'h_max_incumbent_aud':float(maximum),'h_max_outer_upper_aud':float(maximum_outer),
            'h_min_gap_aud':max(0.,float(li-lb)),'h_max_gap_aud':max(0.,float(ui-ub)),
            'omega_incumbent_span_aud':float(maximum-li),'omega_outer_aud':float(omega_outer),
            'epsilon_s_aud':.01,'epsilon_s_plus_omega_outer_aud':float(.01+omega_outer),
            'own_objective_projection_ranges':ranges,'reference_projection_at_epsilon_plus_omega':widened,
            'witness_original_unit_audits':witness_audits,
            'minimum_witness':lx.tolist(),'maximum_witness':ux.tolist(),
            'interpretation':'Own-objective epsilon ranges support numerical action separation; the reference epsilon+omega projection is a distinct conservative inclusion check and need not separate. Separation is not inferred from omega alone.',
            'physical_constraints':model.definition()}
    write('SA1_event178_oscillation_certificate.json',result)
    return result

def event_features():
    rows=[];rules=[]
    for region in REGIONS:
        directory=ROOT/'results/revision_v5/events'
        events=pd.read_csv(source(directory/f'{region}_events.csv'))
        panel=pd.read_parquet(source(directory/f'{region}_event_panel.parquet'))
        definition=json.loads(source(directory/f'{region}_event_definition.json').read_text())
        trajectory=pd.read_parquet(source(ROOT/f'results/revision_v5/control/main/{region}_evaluation_c60_raw.parquet')).sort_values('target').reset_index(drop=True)
        eligible=np.flatnonzero(events.first_pos.to_numpy()+48<=len(panel))
        pick=eligible[np.unique(np.linspace(0,len(eligible)-1,min(20,len(eligible))).round().astype(int))]
        assert set(events.index[events.rolling_probe])==set(pick)
        assert len(pick)==20
        raw=inputs(region,'raw');sparse=inputs(region,'sparse_equal');inverse=inputs(region,'inverse_lead')
        for index in pick:
            event=events.loc[index];pos=int(event.first_pos);f=panel.iloc[pos]
            assert raw.target.iloc[pos]==f.target==trajectory.target.iloc[pos]
            rr=raw.loc[pos,P].to_numpy(float)
            rows.append({'region':region,'event_id':int(event.event_id),'first_pos':pos,
                'execution_start':f.target-pd.Timedelta(minutes=30),'target':f.target,
                'initial_soc_mwh':float(trajectory.soc_start.iloc[pos]),
                'current_first_forecast_aud_mwh':float(rr[0]),
                'current_forecast_peak_aud_mwh':float(f.forecast_peak),
                'current_forecast_range_aud_mwh':float(f.forecast_range),
                'revision_abs_mean_aud_mwh':float(f.revision_absolute_mean),
                'sparse_current_mean_abs_curve_difference_aud_mwh':float(abs(sparse.loc[pos,P].to_numpy(float)-rr).mean()),
                'inverse_current_mean_abs_curve_difference_aud_mwh':float(abs(inverse.loc[pos,P].to_numpy(float)-rr).mean()),
                'actual_first_price_posthoc_aud_mwh':float(trajectory.actual_price.iloc[pos]),
                'trigger_peak':bool(event.first_trigger_peak),'trigger_range':bool(event.first_trigger_range),
                'trigger_revision':bool(event.first_trigger_revision),'selection_uses_realized_profit':False})
        rules.append({'region':region,'thresholds':definition['thresholds'],
            'training_rows':definition['train_rows'],'events':len(events),'eligible_events':len(eligible),
            'selected_events':len(pick)})
    pd.DataFrame(rows).to_csv(OUT/'layerC_100_start_features.csv',index=False)
    write('layerC_selection_rule.json',{'status':'PASS_exact_saved_selection_reconstructed',
        'source_script_sha256':digest(source(ROOT/'scripts/revision_v5_events.py')),
        'trigger':'any of first8 current forecast max, first8 range, mean finite first8 absolute historical revision >= own region pre2023 95th percentile',
        'training_eligibility':'target+330min < 2023-01-01 minus60min',
        'grouping':'new event when gap between consecutive trigger timestamps >4hours; event extends through last trigger+4hours',
        'start':'first trigger of selected event',
        'eligible':'first_pos+48<=len(panel)',
        'selection':'chronological eligible event indices selected with round(linspace(0,nEligible-1,min(20,nEligible))) and unique',
        'branch_initial_state':'canonical raw inventory at event first_pos',
        'scope':'nonrandom forecast-excursion/revision diagnostics; no population failure rate or total policy attribution',
        'future_realization_used_for_selection':False,'regions':rules})

def delay_feasibility():
    """Inventory/re-eligibility counts, explicitly not a delayed-policy replay."""
    rows=[]
    for region in REGIONS:
        f=inputs(region,'raw');target=pd.to_datetime(f.target)
        p=ROOT/f'results/revision_v5/history/{region}_baseline_inputs.parquet'
        q=pd.read_parquet(p,columns=['target','query_time'],filters=[('policy','==','raw'),
            ('cutoff_minutes','==',60),('phase','==','evaluation')]).sort_values('target').reset_index(drop=True)
        original=pd.to_datetime(q.query_time);receipt=original+pd.Timedelta(minutes=30)
        rows.append({'region':region,'origins':len(f),
            'delayed_unavailable_at_original_T_minus60_count':int((receipt>target-pd.Timedelta(minutes=60)).sum()),
            'delayed_unavailable_at_execution_T_minus30_count':int((receipt>target-pd.Timedelta(minutes=30)).sum()),
            'scope':'eligibility only; delayed raw/sparse/IL must be reconstructed against unchanged T-60 information cutoff'})
    pd.DataFrame(rows).to_csv(OUT/'delay30_information_cutoff_feasibility.csv',index=False)
    import build_vintage_training_panel_crossregime as builder
    archives=[]
    for date in pd.date_range('2022-12-01','2026-08-01',freq='MS'):
        p=builder.month_archive('PREDISPATCHPRICE',date.year,date.month)
        archives.append({'month':date.strftime('%Y-%m'),'present':p is not None and p.is_file(),
                         'bytes':p.stat().st_size if p is not None and p.is_file() else None,
                         'path':str(p) if p is not None else None})
    write('delay30_resource_plan.json',{'status':'DATA_AVAILABLE_REBUILD_AND_REPLAY_REQUIRED',
        'hypothesis':'LASTCHANGED+30min <= original target-60min cutoff; nominal run start still <=original cutoff',
        'not_equivalent_to':'original v4 late-at-execution feasibility; shift of information cutoff to execution',
        'raw_months_including_validation_and_carry':archives,
        'compressed_raw_bytes':sum(r['bytes'] or 0 for r in archives),
        'D_free_bytes':shutil.disk_usage(ROOT).free,
        'source_reader':'revision_v5_history_inputs.read_raw; complete all-positive-lead intervention0 archives, no shortened lead-filtered cache',
        'correct_design':'reselect complete eligible coherent case at fixed cutoff; align all twelve targets and all available history; preserve actual labels and joint policy chronology; reselect current comparator using delayed2023 only or separately report fixed original comparator',
        'monthly_aggregate_cache_limitation':'saved paths are aggregates queried at30/60, not all raw vintage records; cannot infer a90min query by subtracting one stored average',
        'policy_origin_solves_for_raw_sparse_IL_five_regions_evaluation':46741*3*5,
        'full_delay_reconstruction_executed':False,'full_delayed_replay_executed':False,
        'remaining_boundary':'participant receipt unobserved; hypothetical delay is conservative sensitivity, not observed availability or causal evidence'})

def delayed_build():
    """Rebuild complete curves from read-only raw archives at ASOF<=T-90."""
    import revision_v5_history_inputs as reader
    from revision_v4_forecast_models import bound_paths
    folder=OUT/'delay30';folder.mkdir(exist_ok=True)
    monthly=folder/'monthly';monthly.mkdir(exist_ok=True)
    originals={}; selected={}
    for region in REGIONS:
        p=source(ROOT/f'results/revision_v5/history/{region}_baseline_inputs.parquet')
        f=pd.read_parquet(p, filters=[('policy','==','raw'),('cutoff_minutes','==',60),('phase','==','evaluation')])
        f['target']=f.target.astype('datetime64[ns]'); originals[region]=f.sort_values('target').reset_index(drop=True)
        choice=source(ROOT/f'results/revision_v5/control/main/{region}_current_only_selection.json')
        selected[region]=json.loads(choice.read_text())['selected_policy']
    write('delay30/design.json',{'status':'fixed_before_reconstruction',
          'original_cutoff':'T-60min','assumed_receipt':'LASTCHANGED+30min',
          'eligibility':'LASTCHANGED<=T-90min, nominal_issue<=T-60min',
          'current_processor_original2023_fixed':selected,
          'coherent_case':'latest complete same seq/run/asof case covering all12 targets; no stitched future case',
          'sparse':'equal current and available prior1/2/4/6 target-aligned versions, previous archival leadfilter20..645 retained',
          'inverse':'all available target-aligned intervention0 positive nominal-lead archived versions, weights1/(delivery-nominal_issue)',
          'support':'original canonical46741 chronology; if missing complete delayed curve explicit common idle guard, never future fill',
          'phase':'evaluation only; current processor weights frozen from original2023; no delayed re-selection claim'})
    previous, previnfo=reader.read_raw(2023,12); raw_sources=[previnfo]; audits=[]
    started=time.perf_counter()
    for date in pd.date_range('2024-01-01','2026-08-01',freq='MS'):
        tick=time.perf_counter();current,info=reader.read_raw(date.year,date.month);raw_sources.append(info)
        end=date+pd.offsets.MonthBegin(1)
        pred=pd.concat([previous,current],ignore_index=True)
        pred=pred.loc[pred.target.ge(date)&pred.target.lt(end+pd.Timedelta(hours=6))].copy()
        pred=pred.sort_values(['PREDISPATCHSEQNO','RUNNO','target','REGIONID','asof','runno_num']).drop_duplicates(reader.KEY+['asof'],keep='last')
        pred=pred.sort_values(['REGIONID','target','asof','PREDISPATCHSEQNO','runno_num']).drop_duplicates(['REGIONID','target','asof'],keep='last').reset_index(drop=True)
        assert ((pred['asof']-pred.nominal_issue).dt.total_seconds()/60>=-30).all(), 'Nominal availability requires per-query filtering'
        pred['lead_min']=(pred.target-pred['asof']).dt.total_seconds()/60
        pred['weight']=1/((pred.target-pred.nominal_issue).dt.total_seconds()/60)
        pred['wp']=pred.weight*pred.forecast
        group=pred.groupby(['REGIONID','target'],sort=False,observed=True)
        pred['wsum']=group.weight.cumsum();pred['wpsum']=group.wp.cumsum()
        pred['inverse_lead']=pred.wpsum/pred.wsum
        sparse=pred.loc[pred.lead_min.between(20,645)].copy()
        sg=sparse.groupby(['REGIONID','target'],sort=False,observed=True)
        sparse['sparse_equal']=pd.concat([sparse.forecast]+[sg.forecast.shift(j) for j in (1,2,4,6)],axis=1).mean(axis=1)
        keys=['REGIONID','target','PREDISPATCHSEQNO','RUNNO','asof']
        lookup=pred.set_index(keys).forecast; slookup=sparse.set_index(keys).sparse_equal
        for region in REGIONS:
            origin=originals[region].loc[lambda f:f.target.ge(date)&f.target.lt(end)].copy()
            if origin.empty:continue
            candidate=pred.loc[pred.REGIONID.eq(region)&pred.target.isin(origin.target)].copy()
            candidate=candidate.loc[(candidate['asof']<=candidate.target-pd.Timedelta(minutes=90))&
                                    (candidate.nominal_issue<=candidate.target-pd.Timedelta(minutes=60))]
            candidate=candidate.sort_values(['target','asof','PREDISPATCHSEQNO','runno_num']).reset_index(drop=True)
            arrays=[]; sparse_arrays=[]
            for j in range(12):
                ix=pd.MultiIndex.from_arrays([candidate.REGIONID,candidate.target+pd.Timedelta(minutes=30*j),
                    candidate.PREDISPATCHSEQNO,candidate.RUNNO,candidate['asof']],names=keys)
                arrays.append(lookup.reindex(ix).to_numpy(float));sparse_arrays.append(slookup.reindex(ix).to_numpy(float))
            matrix=np.column_stack(arrays);smatrix=np.column_stack(sparse_arrays)
            valid=np.isfinite(matrix).all(axis=1)&np.isfinite(smatrix).all(axis=1)
            take=candidate.loc[valid].drop_duplicates('target',keep='last').index.to_numpy()
            chosen=candidate.loc[take].set_index('target').reindex(origin.target)
            support=chosen['asof'].notna().to_numpy()
            # Explicitly fail to queue a silent shortened or future-filled sample.
            # A future implementation could retain missing origins as common idle.
            if not support.all():
                write(f'delay30/missing_{region}_{date:%Y%m}.json',{'status':'REQUIRES_EXPLICIT_COMMON_IDLE_IMPLEMENTATION',
                    'targets':origin.target.loc[~support].astype(str).tolist(),'no_replays_queued':True})
                raise RuntimeError('Missing delayed complete case; no silent chronology deletion')
            chosen_row=pd.Series(take,index=candidate.loc[take].target).reindex(origin.target).to_numpy(int)
            raw_curve=bound_paths(matrix[chosen_row],origin);sparse_curve=bound_paths(smatrix[chosen_row],origin)
            query=[]
            for j in range(12):
                query.append(pd.DataFrame({'origin_pos':np.arange(len(origin)),'step':j,'REGIONID':region,
                    'target':origin.target.to_numpy()+np.timedelta64(30*j,'m'),
                    'cutoff':origin.target.to_numpy()-np.timedelta64(90,'m')}))
            query=pd.concat(query,ignore_index=True)
            historical=pd.merge_asof(query.sort_values('cutoff'),pred.sort_values('asof'),
                left_on='cutoff',right_on='asof',by=['REGIONID','target'],direction='backward')
            assert historical.inverse_lead.notna().all()
            assert (historical['asof']<=historical.cutoff).all()
            assert (historical.nominal_issue<=historical.cutoff+pd.Timedelta(minutes=30)).all()
            inverse=historical.pivot(index='origin_pos',columns='step',values='inverse_lead').reindex(range(len(origin))).to_numpy()
            inverse=bound_paths(inverse,origin)
            processor=selected[region]
            if processor=='raw':cstar=raw_curve.copy()
            elif processor.startswith('current_shrink_'):
                alpha=.25 if processor.endswith('a025') else .5
                cstar=(1-alpha)*raw_curve+alpha*raw_curve.mean(axis=1,keepdims=True)
            elif processor.startswith('current_smooth_'):
                alpha=.25 if processor.endswith('a025') else .5
                sm=raw_curve.copy();sm[:,1:-1]=(raw_curve[:,:-2]+raw_curve[:,1:-1]+raw_curve[:,2:])/3
                sm[:,0]=(raw_curve[:,0]+raw_curve[:,1])/2;sm[:,-1]=(raw_curve[:,-2]+raw_curve[:,-1])/2
                cstar=(1-alpha)*raw_curve+alpha*sm
            else:raise ValueError(processor)
            pieces=[]
            for policy,x in [('raw',raw_curve),('sparse_equal',sparse_curve),('inverse_lead',inverse),('cstar_fixed',cstar)]:
                f=origin[['region','target','phase',*[f'actual_{j:02d}' for j in range(12)]]].copy()
                f['cutoff_minutes']=60;f['policy']=policy;f['query_time']=chosen['asof'].to_numpy()
                f['assumed_receipt_time']=chosen['asof'].to_numpy()+np.timedelta64(30,'m')
                f[P]=bound_paths(x,origin)
                pieces.append(f)
            path=monthly/f'{region}_{date:%Y%m}_delayed_inputs.parquet'
            pd.concat(pieces,ignore_index=True).to_parquet(path,index=False,compression='zstd')
            audits.append({'region':region,'month':date.strftime('%Y-%m'),'origins':len(origin),
                'missing_common_origins':0,'max_receipt_minus_information_cutoff_minutes':float(((chosen['asof'].to_numpy()+np.timedelta64(30,'m')-(origin.target.to_numpy()-np.timedelta64(60,'m')))/np.timedelta64(1,'m')).max()),
                'raw_original_max_forecast_change_aud_mwh':float(abs(raw_curve-origin[P].to_numpy()).max()),
                'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)})
        previous=current
        print(json.dumps({'delay_month_completed':date.strftime('%Y-%m'),'month_seconds':time.perf_counter()-tick,
                          'elapsed_seconds':time.perf_counter()-started}),flush=True)
    for region in REGIONS:
        pieces=[pd.read_parquet(p) for p in sorted(monthly.glob(f'{region}_*_delayed_inputs.parquet'))]
        full=pd.concat(pieces,ignore_index=True).sort_values(['policy','target']).reset_index(drop=True)
        for policy,g in full.groupby('policy'):
            assert len(g)==46741 and np.array_equal(g.target.to_numpy(dtype='datetime64[ns]'),originals[region].target.to_numpy(dtype='datetime64[ns]'))
            assert np.array_equal(g.actual_00.to_numpy(),originals[region].actual_00.to_numpy())
        full.to_parquet(folder/f'{region}_delayed_inputs.parquet',index=False,compression='zstd')
    write('delay30/input_reconstruction_complete.json',{'status':'PASS','regions':REGIONS,
        'origins_per_policy_per_region':46741,'policies':['raw','sparse_equal','inverse_lead','cstar_fixed'],
        'common_chronology_original_preserved':True,'missing_origins':0,'receipt_cutoff_violations':0,
        'nominal_issue_cutoff_violations':0,'fixed_original2023_processors':selected,
        'raw_sources':raw_sources,'months':audits,'seconds':time.perf_counter()-started,
        'all_raw_sources_read_only':True,'raw_reconstruction_executed':True,'controller_replays_executed':False})

def delay_replay(region,policy):
    import revision_v5_control_replays as replay
    for name in ('revision_v5_control_solver','revision_v5_control_safe_policy',
                 'revision_v5_control_scaled_policy','revision_v5_control_enumerated_policy'):
        module=__import__(name);module.OUT=OUT/'delay30/control'
    replay.OUT=OUT/'delay30/control'
    f=pd.read_parquet(OUT/f'delay30/{region}_delayed_inputs.parquet',filters=[('policy','==',policy)])
    f['target']=f.target.astype('datetime64[ns]')
    result=replay.replay(region,f,policy,scenario='delay30')
    print(json.dumps({'completed_delay_replay':f'{region}/{policy}','net_value_aud':result['net_value_aud']}),flush=True)

def delay_queue(workers):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import subprocess
    jobs=[(r,p) for r in REGIONS for p in ('raw','sparse_equal','inverse_lead','cstar_fixed')]
    logs=OUT/'delay30/logs';logs.mkdir(exist_ok=True)
    def run(job):
        region,policy=job
        with (logs/f'{region}_{policy}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)), '--mode','delay-replay',
                '--region',region,'--policy',policy],stdout=log,stderr=subprocess.STDOUT)
        return {'region':region,'policy':policy,'exit_code':result.returncode}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result)
            write('delay30/queue_status.json',{'jobs_completed':len(results),'total':20,'results':results})
            print(json.dumps({'delay_queue':result,'completed':len(results),'total':20}),flush=True)
    if any(r['exit_code'] for r in results):raise RuntimeError('Delayed replay failed; see per-job logs')
    table=[]
    for region in REGIONS:
        reports={policy:json.loads((OUT/f'delay30/control/delay30/{region}_evaluation_c60_{policy}.json').read_text())
                 for policy in ('raw','sparse_equal','inverse_lead','cstar_fixed')}
        r={'region':region}
        for policy,rep in reports.items():
            r[f'{policy}_marked_value_aud']=rep['net_value_aud']
            r[f'{policy}_aud_per_974_nominal_days']=rep['value_aud_per_day']
        for history in HISTORIES:
            for comparator in ('raw','cstar_fixed'):
                r[f'{history}_minus_{comparator}_aud']=reports[history]['net_value_aud']-reports[comparator]['net_value_aud']
        table.append(r)
    frame=pd.DataFrame(table); total={'region':'five_assets'}
    total.update({col:float(frame[col].sum()) for col in frame if col!='region'})
    pd.concat([frame,pd.DataFrame([total])],ignore_index=True).to_csv(OUT/'delay30/marked_value_contrasts.csv',index=False)
    write('delay30/full_replay_complete.json',{'status':'PASS','full_replays':20,
        'origins_per_replay':46741,'workers':workers,'fixed_original_2023_processors':True,
        'receipt_requirement':'LASTCHANGED+30min<=T-60min','reconstruction_receipt_sha256':digest(OUT/'delay30/input_reconstruction_complete.json'),
        'results':results,'reported_intervals':'none; point sensitivity only, not inferential replication'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample-per-region',type=int,default=25)
    p.add_argument('--mode',choices=['all','counts','epsilon','case','events','delay','delay-build','delay-replay','delay-queue'],default='all')
    p.add_argument('--region',choices=REGIONS);p.add_argument('--policy',choices=['raw','sparse_equal','inverse_lead','cstar_fixed'])
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    sys.dont_write_bytecode=True;start=time.perf_counter()
    if args.mode=='delay-build':delayed_build();return
    if args.mode=='delay-replay':delay_replay(args.region,args.policy);return
    if args.mode=='delay-queue':delay_queue(args.workers);return
    frames,plan=full_and_plan(args.sample_per_region)
    result={}
    if args.mode in ('all','epsilon'):result['epsilon']=sensitivity(frames,plan)
    if args.mode in ('all','case'):result['case']=sa_certificate()
    if args.mode in ('all','events'):event_features()
    if args.mode in ('all','delay'):delay_feasibility()
    source(Path(__file__))
    write('input_manifest.json',{'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                                'sources':MANIFEST,'seconds':time.perf_counter()-start,
                                'original_files_modified':False})
    print(json.dumps({'status':'completed','mode':args.mode,'seconds':time.perf_counter()-start}),flush=True)

if __name__=='__main__':main()
