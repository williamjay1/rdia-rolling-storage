"""Conditional finite-family statistics on frozen rolling-storage trajectories.

No optimiser, feature refit, policy reselection during evaluation, or old result
writer is called. Five assets always share the same resampled calendar indices.
The marked daily scores explicitly allocate the realised fixed boundary term;
they are not observations of daily mark-to-market cash. All historical analyses
are exploratory reanalyses, conditional on the saved controller and state paths.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import revision_v7_statistics as frozen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_v15/statistics"
REGIONS = list(frozen.REGIONS)
CURRENT = list(frozen.CANDIDATES)
POLICIES = CURRENT + ["sparse_equal", "full_equal", "inverse_lead"]
MODELS = ["R", "Cstar", "S", "F", "IL"]
HISTORY = ["S", "F", "IL"]
TIE_ORDER = np.asarray([0, 4, 3, 2, 1])  # raw first, then lexicographically descending


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.12g")


def mc_p(observed, simulations):
    hits = int(np.count_nonzero(np.asarray(simulations) >= observed))
    p = (hits + 1) / (len(simulations) + 1)
    return p, math.sqrt(p * (1-p) / (len(simulations)+1)), hits


def studentizer(mean, bootstrap):
    centered = bootstrap - mean
    se = np.sqrt(np.mean(centered ** 2, axis=0))
    if np.any(se <= 1e-14):
        raise ValueError("Non-identical comparison has zero bootstrap variance; no variance regularisation allowed")
    return centered, se


def spa(mean, bootstrap, n):
    """Hansen's consistent-recentered, studentised one-sided finite-family SPA.

    Uses the circular moving-block adaptation explicitly allowed by Hansen
    (2005). The long-run variance estimate is n times bootstrap mean-error RMS².
    The same consistent variance estimate is held fixed in bootstrap statistics.
    """
    err, se = studentizer(mean, bootstrap)
    t = mean / se
    threshold = math.sqrt(2 * math.log(math.log(n)))
    mu_c = np.where(t <= -threshold, mean, 0.0)
    observed = max(0.0, float(np.max(t)))
    shifts = {"lower": np.minimum(mean, 0), "consistent": mu_c,
              "upper": np.zeros_like(mean)}
    variants = {}
    for kind, shift in shifts.items():
        sim = np.maximum(0, np.max((err + shift) / se, axis=1))
        pv, mcse, hits = mc_p(observed, sim)
        variants[kind] = dict(p_value=pv, monte_carlo_se=mcse, exceedances=hits)
    assert variants["lower"]["p_value"] <= variants["consistent"]["p_value"] <= variants["upper"]["p_value"]
    return dict(statistic=observed, consistent_recenter_threshold=threshold,
                t_statistics=t.tolist(), standard_errors=se.tolist(),
                mu_consistent=mu_c.tolist(), long_run_sd=(math.sqrt(n)*se).tolist(),
                null="All three population mean fixed-program marked-score increments relative to fixed Cstar are nonpositive",
                **variants)


def romano_wolf(mean, bootstrap, labels):
    """One-sided studentised max-t bootstrap stepdown, common fixed studentizer."""
    err, se = studentizer(mean, bootstrap)
    t = mean / se
    z = err / se
    order = np.argsort(-t, kind="stable")
    adjusted = np.zeros(len(mean))
    rows = []
    cumulative = 0.0
    for rank, j in enumerate(order):
        remaining = order[rank:]
        step_p, mcse, hits = mc_p(float(t[j]), z[:, remaining].max(axis=1))
        cumulative = max(cumulative, step_p)
        individual, _, _ = mc_p(float(t[j]), z[:, j])
        adjusted[j] = cumulative
        rows.append(dict(label=labels[j], step_rank=rank+1, mean=float(mean[j]),
                         standard_error=float(se[j]), statistic=float(t[j]),
                         individual_one_sided_p=individual, step_p=step_p,
                         adjusted_one_sided_p=cumulative, step_monte_carlo_se=mcse,
                         step_exceedances=hits,
                         remaining_labels=[labels[k] for k in remaining]))
        assert cumulative >= individual
    assert np.all(np.diff(adjusted[order]) >= 0)
    return rows, adjusted, se, z


def model_confidence_set(mean_loss, boot_loss, labels):
    """Hansen--Lunde--Nason Tmax elimination with bootstrap-mean studentisation.

    Recomputes the active-set grand-mean centering and variance at each deletion.
    Higher loss is worse. Elimination p-values are monotonised, as in the
    authors' MCS algorithm and the documented arch implementation.
    """
    base_error = boot_loss - mean_loss
    active = list(range(len(labels)))
    adjusted = np.ones(len(labels))
    trace = []
    cumulative = 0.0
    while len(active) > 1:
        err = base_error[:, active]
        err = err - err.mean(axis=1, keepdims=True)
        observed = mean_loss[active] - mean_loss[active].mean()
        se = np.sqrt(np.mean(err ** 2, axis=0))
        if np.all(np.abs(observed) < 1e-13) and np.all(se < 1e-13):
            for j in active:
                adjusted[j] = 1.0
            trace.append(dict(active=[labels[j] for j in active], all_programs_identical=True,
                              step_p=1.0, eliminated=None))
            break
        if np.any(se <= 1e-14):
            raise ValueError("Nontrivial zero-variance MCS active set; stop rather than regularise")
        t = observed / se
        worst = int(np.argmax(t))
        sim = (err / se).max(axis=1)
        pv, mcse, hits = mc_p(float(t[worst]), sim)
        cumulative = max(cumulative, pv)
        active_before=[labels[j] for j in active]
        eliminated = active.pop(worst)
        adjusted[eliminated] = cumulative
        trace.append(dict(active=active_before,
                          eliminated=labels[eliminated], statistic=float(t[worst]),
                          step_p=pv, adjusted_elimination_p=cumulative,
                          monte_carlo_se=mcse, exceedances=hits,
                          active_standard_errors=se.tolist()))
    retained90 = [labels[i] for i in range(len(labels)) if adjusted[i] > .10]
    retained95 = [labels[i] for i in range(len(labels)) if adjusted[i] > .05]
    assert set(retained90) <= set(retained95)
    return dict(method="Tmax", models=labels, model_p_values=adjusted.tolist(),
                retained_90=retained90, retained_95=retained95, elimination_trace=trace,
                null="Equal expected negative portfolio marked scores among the current active set",
                interpretation="Retained membership means failure to exclude by this equal-predictive-ability elimination procedure, not equivalence or noninferiority")


def choose_current(values):
    return TIE_ORDER[np.argmax(np.asarray(values)[..., TIE_ORDER], axis=-1)]


def implementation_checks():
    rng = np.random.default_rng(471915)
    n, b = 210, 2400
    noise = rng.normal(size=(n, 3)) @ np.array([[1,.2,.1],[0,1,.25],[0,0,.75]])
    x = noise + np.array([.6, 0.0, -.8])
    boot, literal = frozen.bootstrap_means(x, b, 7, 718315)
    mean = x.mean(axis=0)
    a = spa(mean, boot, n)
    rw, adj, _, _ = romano_wolf(mean, boot, HISTORY)
    permutation = [2, 0, 1]
    a2 = spa(3.5*mean[permutation], 3.5*boot[:, permutation], n)
    rw2, adj2, _, _ = romano_wolf(3.5*mean[permutation], 3.5*boot[:, permutation],
                                 [HISTORY[i] for i in permutation])
    assert a["consistent"]["p_value"] == a2["consistent"]["p_value"]
    assert np.array_equal(adj[permutation], adj2)
    loss = np.column_stack([np.zeros(n), -.75+noise[:, 0], 1.5+noise[:, 1], noise[:, 2]])
    bl, _ = frozen.bootstrap_means(loss, b, 7, 718316)
    ml = loss.mean(axis=0)
    names = ["base", "better", "worse", "noise"]
    m = model_confidence_set(ml, bl, names)
    perm4 = [2,0,3,1]
    common_shift = np.linspace(-2, 3, b)
    m2 = model_confidence_set(4*ml[perm4]+19,
                             4*bl[:, perm4]+19+common_shift[:, None],
                             [names[i] for i in perm4])
    assert np.allclose(np.asarray(m["model_p_values"])[perm4], m2["model_p_values"], atol=0, rtol=0)
    assert "worse" not in m["retained_95"] and "better" in m["retained_90"]
    tied = model_confidence_set(np.zeros(3), np.zeros((b, 3)), HISTORY)
    assert tied["retained_90"] == HISTORY and tied["retained_95"] == HISTORY
    assert choose_current(np.zeros((5,5))).tolist() == [0]*5
    tie = np.zeros((1,5)); tie[0,1:] = 1
    assert choose_current(tie).tolist() == [4]
    assert spa(-np.ones(3), boot-mean-1, n)["consistent"]["p_value"] == 1.0
    return dict(status="PASS", tests=["literal circular-block arithmetic", "SPA lower <= consistent <= upper",
                "SPA/RW positive-scale and model-permutation invariance", "RW adjusted p >= marginal p",
                "RW step-order monotonicity", "MCS higher-loss direction",
                "MCS positive-scale, model-permutation and common-time-offset invariance",
                "MCS 95 percent set contains 90 percent set", "MCS identical-program handling",
                "exact raw-first / lexicographically greatest tie rule", "all-negative SPA has p=1"],
                literal_index_error=literal, spa_toy=a, romano_wolf_toy=rw, mcs_toy=m)


def marked_value(frame):
    return float(frame.net_aud.sum()+frame.soc_end.iloc[-1]*frame.actual_price.iloc[-1]
                 -frame.soc_start.iloc[0]*frame.actual_price.iloc[0])


def daily_cube(frames, names, phase, n_expected, complete_all=False):
    ref = frames[REGIONS[0]][names[0]]
    dates = pd.DatetimeIndex(ref.execution_day.unique()).sort_values()
    keep = dates if complete_all else dates[1:-1]
    assert len(keep) == n_expected, (phase, len(keep))
    cube = np.empty((len(keep), len(REGIONS), len(names)))
    totals = np.empty((len(REGIONS),len(names)))
    long = []
    maxerr = 0.0
    for ri, region in enumerate(REGIONS):
        for pi, policy in enumerate(names):
            f = frames[region][policy]
            assert f.target.equals(ref.target) and f.execution_day.equals(ref.execution_day)
            assert np.array_equal(f.actual_price.to_numpy(), frames[region][names[0]].actual_price.to_numpy())
            count = f.groupby("execution_day").size().reindex(keep)
            expected_count=ref.groupby("execution_day").size().reindex(keep)
            assert count.equals(expected_count)
            assert count.isin([47,48]).all(), (phase, region, policy, count[count.lt(47)].to_dict())
            if phase=="validation":assert count.eq(48).all()
            else:assert count.index[count.ne(48)].tolist()==[pd.Timestamp("2024-09-05")]
            grouped = f.groupby("execution_day").net_aud.sum().reindex(keep)
            codes = keep.get_indexer(f.execution_day)
            direct = np.bincount(codes[codes >= 0], weights=f.net_aud.to_numpy()[codes>=0], minlength=len(keep))
            err = float(np.max(np.abs(direct-grouped.to_numpy())))
            maxerr=max(maxerr,err); assert err<1e-8
            cube[:,ri,pi] = grouped.to_numpy()
            totals[ri,pi] = marked_value(f)
            for day, value,observed_intervals in zip(keep, grouped,count):
                long.append(dict(phase=phase, region=region, policy=policy, execution_day=day,
                                 operating_net_cash_aud=float(value), intervals=int(observed_intervals)))
    return cube, totals, keep, long, maxerr


def load_panels():
    frames = {"validation":{}, "evaluation":{}, "evaluation_2024_2025":{}}
    cstar=[]
    for region in REGIONS:
        print("AUDIT_LOAD", region, flush=True)
        frames["validation"][region]={}
        frames["evaluation"][region]={}
        frames["evaluation_2024_2025"][region]={}
        for p in CURRENT:
            f,_=frozen.load(region,p,"validation")
            deadline=pd.Timestamp("2024-01-01")-pd.Timedelta(minutes=60)
            mask=(f.target>=pd.Timestamp("2023-01-01"))&(f.target+pd.Timedelta(minutes=330)<deadline)
            assert mask.all(), "Saved strict-2023 validation must already implement the full-label purge"
            frames["validation"][region][p]=f
        for p in POLICIES:
            f,_=frozen.load(region,p,"evaluation")
            frames["evaluation"][region][p]=f
            mask=(f.execution_day>=pd.Timestamp("2024-01-01"))&(f.execution_day<pd.Timestamp("2026-01-01"))
            frames["evaluation_2024_2025"][region][p]=f.loc[mask].reset_index(drop=True)
        path=frozen.SOURCE/f"{region}_current_only_selection.json"
        frozen.record(path)
        cstar.append(json.loads(path.read_text(encoding="utf-8"))["selected_policy"])
    arrays={}; totals={}; dates={}; all_rows=[]; errors={}
    for phase,names,n,whole in (("validation",CURRENT,364,False),("evaluation",POLICIES,973,False),
                                ("evaluation_2024_2025",POLICIES,731,True)):
        arrays[phase],totals[phase],dates[phase],rows,errors[phase]=daily_cube(frames[phase],names,phase,n,whole)
        all_rows+=rows
    chosen=choose_current(totals["validation"])
    assert [CURRENT[i] for i in chosen] == cstar
    pd.DataFrame(all_rows).to_parquet(OUT/"daily_operating_cash_panel.parquet",index=False)
    save("panel_totals.json",dict(format="v15_derived_daily_cash_fixture_v1",regions=REGIONS,current=CURRENT,policies=POLICIES,
         totals={p:v.tolist() for p,v in totals.items()},frozen_choice=chosen.tolist(),
         daily_panel_sha256=sha(OUT/"daily_operating_cash_panel.parquet"),
         phase_days={p:len(v) for p,v in dates.items()},
         source_mode="Derived operating net cash and exact realised boundary totals of frozen audited paths; no raw MMS or XML",
         original_source_manifest=list(frozen.MANIFEST.values())))
    save("source_panel_audit.json",dict(status="PASS",regions=REGIONS,candidates=CURRENT,evaluation_policies=POLICIES,
         frozen_Cstar=dict(zip(REGIONS,cstar)), train_days=364,full_evaluation_interior_days=973,
         evaluation_2024_2025_days=731,maximum_direct_aggregation_errors=errors,
         training_full_label_purge="target + 330min < 2024-01-01 -60min",
         first_last_days={p:[str(v[0]),str(v[-1])] for p,v in dates.items()},
         daily_counts="Validation364 dates all48 actions; evaluation973 and2024-2025 subset731 observed execution dates each include2024-09-05 with47 synchronized saved actions, all other interior dates48; no imputation",
         evaluation_subpath="Execution-day 2024-01-01 through 2025-12-31 of previously saved state paths; no 2024/2025 restart or forecast refit"))
    return arrays,totals,chosen,dates


def load_daily_fixture(folder):
    """Use exact derived daily cash plus endpoint totals in an additive package.

    This branch repeats statistics, not interval-level market-data or controller
    audits. The original default branch audits all frozen interval trajectories.
    """
    folder=Path(folder).resolve()
    mp=folder/"panel_totals.json";dp=folder/"daily_operating_cash_panel.parquet"
    meta=json.loads(mp.read_text(encoding="utf-8"))
    assert meta["format"]=="v15_derived_daily_cash_fixture_v1"
    assert meta["regions"]==REGIONS and meta["current"]==CURRENT and meta["policies"]==POLICIES
    assert sha(dp)==meta["daily_panel_sha256"]
    frozen.record(mp);frozen.record(dp)
    panel=pd.read_parquet(dp)
    arrays={};dates={};totals={p:np.asarray(v,dtype=np.float64) for p,v in meta["totals"].items()}
    for phase in ("validation","evaluation","evaluation_2024_2025"):
        sub=panel.loc[panel.phase.eq(phase)]
        names=CURRENT if phase=="validation" else POLICIES
        dates[phase]=pd.DatetimeIndex(sub.execution_day.unique()).sort_values()
        assert len(dates[phase])==meta["phase_days"][phase]
        cube=np.empty((len(dates[phase]),5,len(names)))
        for ri,r in enumerate(REGIONS):
            for pi,p in enumerate(names):
                f=sub.loc[sub.region.eq(r)&sub.policy.eq(p)].set_index("execution_day").reindex(dates[phase])
                assert len(f)==len(dates[phase]) and f.operating_net_cash_aud.notna().all()
                assert f.intervals.isin([47,48]).all()
                cube[:,ri,pi]=f.operating_net_cash_aud.to_numpy()
        arrays[phase]=cube
    chosen=np.asarray(meta["frozen_choice"],dtype=int)
    assert np.array_equal(choose_current(totals["validation"]),chosen)
    panel.to_parquet(OUT/"daily_operating_cash_panel.parquet",index=False)
    save("panel_totals.json",meta)
    save("source_panel_audit.json",dict(status="PASS_DERIVED_FIXTURE_HASH_AND_ARRAY_AUDIT",regions=REGIONS,
         train_days=364,full_evaluation_interior_days=973,evaluation_2024_2025_days=731,
         frozen_Cstar=dict(zip(REGIONS,[CURRENT[i] for i in chosen])),
         original_interval_audit_reexecuted=False,
         scope="Fresh statistical calculation from exact archived derived daily cash and realised endpoint totals; no new interval, market-data or state-path audit"))
    return arrays,totals,chosen,dates


def gather_model(array, chosen):
    cur=array[..., np.arange(5), chosen] if array.ndim==3 else array[np.arange(5),chosen]
    # For a n x region x policy cube, choose region-specific current programs.
    if array.ndim==3:
        cur=array[:,np.arange(5),chosen].sum(axis=1)
        return np.column_stack([array[:,:,0].sum(axis=1),cur,
                                *[array[:,:,POLICIES.index(p)].sum(axis=1)
                                  for p in ("sparse_equal","full_equal","inverse_lead")]])
    return np.r_[array[:,0].sum(), cur.sum(),
                  [array[:,POLICIES.index(p)].sum() for p in ("sparse_equal","full_equal","inverse_lead")]]


def inference(arrays,totals,chosen,draws,seed):
    x=arrays["evaluation"]
    n,N=len(x),974
    boundary=totals["evaluation"]-x.sum(axis=0)
    y=(n*x+boundary[None,:,:])/N
    assert np.max(np.abs(y.mean(axis=0)-totals["evaluation"]/N))<1e-8
    models=gather_model(y,chosen)
    interior=gather_model(x,chosen)
    fullportfolio=gather_model(totals["evaluation"],chosen)
    assert np.max(np.abs(models.mean(axis=0)-fullportfolio/N))<1e-8
    pd.DataFrame(models,columns=MODELS).to_parquet(OUT/"portfolio_fixed_boundary_marked_daily_scores.parquet",index=False)
    save("marked_score_definition.json",dict(interior_days=n,nominal_days=N,regions=REGIONS,
         formula="Y[d,r,p]=(973*X[d,r,p]+B[r,p])/974; B[r,p]=V[r,p]-sum_d X[d,r,p]; portfolio Y=sum regions; loss=-Y",
         interpretation="Fixed-boundary accounting score, not observed daily mark-to-market cash",
         boundary_terms_aud=boundary.tolist(),boundary_portfolio_models_aud=gather_model(boundary,chosen).tolist(),
         full_marked_portfolio_values_aud=fullportfolio.tolist(),models=MODELS,
         bootstrap_conditioning="Partial boundary-day cash and realised initial/final SOC price marks held fixed; standard errors scaled973/974 relative to pure interior operating cash",
         scope="Named assets, frozen programs and realised SOC paths; no independent regional replication or prospective confirmation"))
    familyrows=[]; mcsrows=[]; costrows=[]; interiorrows=[]; details={}
    for length in (7,28):
        print("INFERENCE",length,flush=True)
        bm,error=frozen.bootstrap_means(models,draws,length,seed+length)
        mean=models.mean(axis=0)
        effect=mean[2:]-mean[1]
        be=bm[:,2:]-bm[:,1,None]
        sr=spa(effect,be,n)
        rw,adj,se,z=romano_wolf(effect,be,HISTORY)
        mcs=model_confidence_set(-mean,-bm,MODELS)
        details[str(length)]=dict(spa=sr,romano_wolf=rw,mcs=mcs,literal_index_error=error,
                                 seed=seed+length,draws=draws)
        maxq=float(np.quantile(z.max(axis=1),.95))
        for j,label in enumerate(HISTORY):
            lo,hi=np.quantile(be[:,j],[.025,.975])
            familyrows.append(dict(block_days=length,history_policy=label,portfolio_mean_daily_increment_aud=float(effect[j]),
                spa_family_p=sr["consistent"]["p_value"],spa_family_mcse=sr["consistent"]["monte_carlo_se"],
                spa_lower_p=sr["lower"]["p_value"],spa_upper_p=sr["upper"]["p_value"],
                rw_adjusted_one_sided_p=float(adj[j]),individual_one_sided_p=rw[[r["label"] for r in rw].index(label)]["individual_one_sided_p"],
                mcs90_retained=";".join(mcs["retained_90"]),mcs95_retained=";".join(mcs["retained_95"]),
                standard_error=float(se[j]),bootstrap_draws=draws))
            costrows.append(dict(block_days=length,history_policy=label,mean_daily_increment_aud=float(effect[j]),
                  daily_ci_lo=float(lo),daily_ci_hi=float(hi),annualized_365_mean_aud=float(365*effect[j]),
                  annualized_365_ci_lo=float(365*lo),annualized_365_ci_hi=float(365*hi),
                  annualized_per_mw_365_mean_aud=float(365*effect[j]/5),
                  annualized_per_mw_365_ci_lo=float(365*lo/5),annualized_per_mw_365_ci_hi=float(365*hi/5),
                  simultaneous_one_sided95_lower_daily_aud=float(effect[j]-maxq*se[j]),
                  scope="Signed five-asset stylized incremental budget rate; actual maintenance cost unobserved"))
        for j,label in enumerate(MODELS):
            mcsrows.append(dict(block_days=length,model=label,mcs_adjusted_elimination_p=mcs["model_p_values"][j],
                                retained90=label in mcs["retained_90"],retained95=label in mcs["retained_95"]))
        bi,error2=frozen.bootstrap_means(interior,draws,length,seed+length)
        ei=interior.mean(axis=0)[2:]-interior.mean(axis=0)[1]
        bei=bi[:,2:]-bi[:,1,None]
        sri=spa(ei,bei,n); rwi,adji,sei,_=romano_wolf(ei,bei,HISTORY)
        mcsi=model_confidence_set(-interior.mean(axis=0),-bi,MODELS)
        for j,label in enumerate(HISTORY):
            lo,hi=np.quantile(bei[:,j],[.025,.975])
            interiorrows.append(dict(block_days=length,policy=label,pure_interior_mean_aud=float(ei[j]),
                 ci_lo=float(lo),ci_hi=float(hi),spa_family_p=sri["consistent"]["p_value"],
                 rw_adjusted_p=float(adji[j]),mcs90=";".join(mcsi["retained_90"]),mcs95=";".join(mcsi["retained_95"])))
        details[str(length)]["pure_interior_cash_sensitivity"]=dict(spa=sri,romano_wolf=rwi,mcs=mcsi,
                                                                   literal_index_error=error2)
    csv("family_inference.csv",familyrows);csv("mcs_membership.csv",mcsrows)
    csv("break_even_incremental_budget.csv",costrows);csv("pure_interior_cash_sensitivity.csv",interiorrows)
    save("formal_inference.json",details)
    return familyrows,costrows,details


def quantiles(value):
    q=np.quantile(value,[.025,.5,.975])
    return dict(p025=float(q[0]),median=float(q[1]),p975=float(q[2]),mean=float(np.mean(value)))


def selection_uncertainty(arrays,totals,frozen_choice,draws,seed):
    train=arrays["validation"]; evalu=arrays["evaluation_2024_2025"]
    ntr,nev=len(train),len(evalu)
    btr=totals["validation"]-train.sum(axis=0)
    bev=totals["evaluation_2024_2025"]-evalu.sum(axis=0)
    frozen_eval=totals["evaluation_2024_2025"][np.arange(5),frozen_choice].sum()
    raw_eval=totals["evaluation_2024_2025"][:,0].sum()
    history_eval=totals["evaluation_2024_2025"][:,5:].sum(axis=0)
    probs=[]; rows=[]; details={}
    for block in (7,28):
        print("SELECTION_UNCERTAINTY",block,flush=True)
        bt,err1=frozen.bootstrap_means(train,draws,block,seed+200+block)
        scores=ntr*bt+btr
        chosen=choose_current(scores)
        joint=int(np.all(chosen==frozen_choice,axis=1).sum())
        for ri,region in enumerate(REGIONS):
            for pi,policy in enumerate(CURRENT):
                hits=int(np.count_nonzero(chosen[:,ri]==pi))
                probability=hits/draws
                probs.append(dict(block_days=block,region=region,candidate=policy,selected_count=hits,
                              selection_probability=probability,binomial_monte_carlo_se=math.sqrt(probability*(1-probability)/draws),
                              originally_selected=pi==frozen_choice[ri],draws=draws))
        actual_selected=totals["evaluation_2024_2025"][np.arange(5)[None,:],chosen].sum(axis=1)
        be,err2=frozen.bootstrap_means(evalu,draws,block,seed+400+block)
        es=nev*be+bev
        boot_selected=es[np.arange(draws)[:,None],np.arange(5)[None,:],chosen].sum(axis=1)
        boot_raw=es[:,:,0].sum(axis=1)
        boot_hist=es[:,:,5:].sum(axis=1)
        frozen_boot=es[:,np.arange(5),frozen_choice].sum(axis=1)
        metrics={"selection_only_selected_current_minus_frozen":(actual_selected-frozen_eval)/nev,
                 "selection_only_selected_current_minus_raw":(actual_selected-raw_eval)/nev,
                 "combined_selected_current_minus_raw":(boot_selected-boot_raw)/nev}
        for j,label in enumerate(HISTORY):
            metrics[f"selection_only_{label}_minus_reselected_current"]=(history_eval[j]-actual_selected)/nev
            metrics[f"combined_{label}_minus_reselected_current"]=(boot_hist[:,j]-boot_selected)/nev
            metrics[f"combined_{label}_minus_frozen_current"]=(boot_hist[:,j]-frozen_boot)/nev
        for label,value in metrics.items():
            rows.append(dict(block_days=block,metric=label,**quantiles(value),days=nev,draws=draws,
                     interpretation="Conditional sensitivity distribution, not unconditional selection-adjusted confidence interval"))
        pd.DataFrame({"draw":np.arange(draws),"block_days":block,
                      **{f"choice_{r}":[CURRENT[i] for i in chosen[:,ri]] for ri,r in enumerate(REGIONS)},
                      "fixed_path_selected_current_total_aud":actual_selected,
                      "combined_selected_current_total_aud":boot_selected,
                      "combined_raw_total_aud":boot_raw,
                      **{f"combined_{label}_total_aud":boot_hist[:,j] for j,label in enumerate(HISTORY)}}).to_parquet(
                            OUT/f"selection_draws_block{block}.parquet",index=False)
        details[str(block)]=dict(draws=draws,training_block_seed=seed+200+block,evaluation_block_seed=seed+400+block,
             training_literal_error=err1,evaluation_literal_error=err2,
             all_five_original_choices_count=joint,all_five_original_choices_probability=joint/draws,
             actual_frozen_current_2024_2025_marked_total_aud=float(frozen_eval),
             actual_raw_2024_2025_marked_total_aud=float(raw_eval),actual_history_2024_2025_totals_aud=history_eval.tolist())
    csv("current_selection_probabilities.csv",probs);csv("selection_sensitivity_distributions.csv",rows)
    save("selection_uncertainty.json",dict(blocks=details,train_days=ntr,evaluation_days=nev,
         training="Strict purged 2023 saved current-program daily paths; shared date resampling and region-specific selection",
         evaluation="Saved 2024-2025 state subpaths, candidate programs run unchanged; realized boundary marks held fixed",
         selection_only="Reselection across saved validation scores, then evaluate saved candidate 2024-2025 total values without resampling evaluation",
         combined="Independent phase-specific block draws across train and evaluation; within each phase all regions and programs share sampled dates",
         exclusions="No model or preprocessing refit, no closed-loop replay after resampling, no correction for earlier research choices, no claimed independent cross-year or unconditional selection inference"))
    return probs,rows,details


def tex_name(s):
    return {"R":"R", "Cstar":r"C^{*}", "S":"S", "F":"F", "IL":r"\mathrm{IL}"}[s]


def tex_set(s):
    return r"$"+", ".join(tex_name(k) for k in s.split(";"))+r"$"




def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--draws",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20261002)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--input-fixture",type=Path,help="Exact derived daily cash/totals folder; does not repeat interval-level audits")
    parser.add_argument("--output",type=Path,help="New output directory; must not exist")
    args=parser.parse_args()
    if args.output:
        OUT=args.output.resolve()
        if OUT.exists():raise FileExistsError("Explicit output must be a new directory")
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(locked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage="EXPLORATORY_REANALYSIS",
        primary_family="S/F/IL minus fixed strict-2023 Cstar",mcs_family=MODELS,
        loss="Negative five-asset fixed-boundary portfolio marked daily score",blocks_days=[7,28],
        draws=args.draws,seed=args.seed,region_resampling="Same dates for all five regions and programs",
        selection="Resample strict-2023 current candidate value paths; reselect region-specific current programs, evaluate saved2024-2025 trajectories; no refit or new replay",
        maintenance="Unobserved cost, signed incremental daily and365-day annualised budget with uncertainty",
        software="Independent finite-family SPA consistent recentering; one-sided studentised max-t RW stepdown; Tmax MCS",
        source_mode="Read-only saved trajectories; no optimiser or earlier-version writer called")
    if not (OUT/"locked_design.json").exists():save("locked_design.json",design)
    start=time.perf_counter()
    checks=implementation_checks();save("implementation_checks.json",checks)
    if args.check_only:
        print(json.dumps({"status":"PASS_IMPLEMENTATION_CHECKS","tests":len(checks["tests"])}),flush=True)
        return
    arrays,totals,chosen,dates=load_daily_fixture(args.input_fixture) if args.input_fixture else load_panels()
    family,costs,detail=inference(arrays,totals,chosen,args.draws,args.seed)
    probs,selection,seldetail=selection_uncertainty(arrays,totals,chosen,args.draws,args.seed)
    receipt=dict(status="PASS_EXECUTED_ALL_V15_STATISTICS",elapsed_seconds=time.perf_counter()-start,
         script_sha256=sha(__file__),locked_design_sha256=sha(OUT/"locked_design.json"),
         source_manifest=list(frozen.MANIFEST.values()),implementation_tests=len(checks["tests"]),
         bootstrap_draws_per_block=args.draws,block_lengths_days=[7,28],
         independent_region_samples=False,model_or_forecast_refit=False,closed_loop_resampling=False,
         source_mode="exact derived daily fixture" if args.input_fixture else "full frozen interval trajectory audit",
         scope="Conditional frozen-policy trajectory reanalysis, not independent holdout or global research-selection correction",
         output_manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="execution_receipt.json"])
    save("execution_receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("status","elapsed_seconds","implementation_tests","bootstrap_draws_per_block","block_lengths_days")}),flush=True)


if __name__=="__main__":main()
