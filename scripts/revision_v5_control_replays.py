"""Executed tolerance-ordered storage replays and validation-only selection."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"
import argparse,json,time,hashlib,math,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from revision_v5_control_solver import ROOT,OUT
from revision_v5_control_safe_policy import SafeLexMPC
from revision_v5_control_scaled_policy import ScaledSafeLexMPC
from revision_v5_control_enumerated_policy import EnumeratedSafeLexMPC as LexMPC

REGIONS=["NSW1","QLD1","SA1","TAS1","VIC1"]
HISTORY=ROOT/"results"/"revision_v5"/"history"
FIELDS=["soc_start","charge_mw","discharge_mw","soc_end","cashflow_aud","degradation_aud","net_aud",
        "primary_optimum_aud","primary_loss_aud","primary_loss_upper_aud","primary_gap_aud"]

def source_path(region,kind="baseline"):
    return HISTORY/(f"{region}_baseline_inputs.parquet" if kind=="baseline" else
                    f"{region}_incremental_inputs_aligned.parquet" if kind=="incremental" else f"{region}_inputs_aligned.parquet")

def load_inputs(region,kind="baseline",cutoff=60,phase="evaluation",policies=None):
    filters=[("cutoff_minutes","==",cutoff),("phase","==",phase)]
    if policies is not None:filters.append(("policy","in",list(policies)))
    columns=["region","target","phase","cutoff_minutes","policy","query_time"]
    columns += [f"p_{j:02d}" for j in range(1,13)]+["actual_00"]
    f=pd.read_parquet(source_path(region,kind),filters=filters,columns=columns)
    f["target"]=f["target"].astype("datetime64[ns]")
    if f.empty:raise ValueError(f"No inputs {region} {kind} {cutoff} {phase} {policies}")
    return f

def metric(frame,solver,region,policy,phase,forecast_hash):
    daily=frame.groupby("execution_day").net_aud.sum();d=daily.iloc[1:-1]
    # The policy continuation coefficient can vary; ex-post accounting remains
    # identical across every policy and boundary experiment.
    mark=frame.soc_end.iloc[-1]*frame.actual_price.iloc[-1]-frame.soc_start.iloc[0]*frame.actual_price.iloc[0]
    elapsed_days=float((frame.target.iloc[-1]-(frame.target.iloc[0]-pd.Timedelta(hours=solver.dt))).total_seconds()/86400)
    # Retain the declared 974 nominal calendar exposure days rather than
    # relabeling the 973.79-day observed interval span as exactly 974 days.
    days=974. if phase=="evaluation" else float(math.ceil(elapsed_days))
    c=frame.charge_mw.to_numpy();z=frame.discharge_mw.to_numpy();s0=frame.soc_start.to_numpy();s1=frame.soc_end.to_numpy()
    audit={"max_soc_balance_error_mwh":float(np.max(np.abs(s1-s0-solver.eta*c*solver.dt+z*solver.dt/solver.eta))),
           "max_continuity_error_mwh":float(np.max(np.abs(s0[1:]-s1[:-1]))),
           "mode_violations_gt1e-6":int(np.sum((c>1e-6)&(z>1e-6))),
           "bounds_violations":int(np.sum((s0<solver.smin-1e-6)|(s0>solver.smax+1e-6)|(s1<solver.smin-1e-6)|(s1>solver.smax+1e-6)|(c< -1e-6)|(z< -1e-6)|(c>solver.power+1e-6)|(z>solver.power+1e-6))),
           "max_primary_loss_aud":float(frame.primary_loss_aud.max()),
           "max_primary_loss_upper_aud":float(frame.primary_loss_upper_aud.max()),
           "max_primary_gap_aud":float(frame.primary_gap_aud.max())}
    return {"region":region,"policy":policy,"phase":phase,"intervals":len(frame),"nominal_days":days,"elapsed_observed_span_days":elapsed_days,
            "net_operating_aud":float(frame.net_aud.sum()),"endpoint_mark_aud":float(mark),
            "net_value_aud":float(frame.net_aud.sum()+mark),"value_aud_per_day":float((frame.net_aud.sum()+mark)/days),
            "throughput_mwh":float((c+z).sum()*solver.dt),"initial_soc_mwh":float(s0[0]),"final_soc_mwh":float(s1[-1]),
            "tail_interior_days":len(d),"daily_p05_aud":float(d.quantile(.05)),
            "losing_interior_days":int((d<0).sum()),"minimum_interior_day_aud":float(d.min()),
            "policy_definition":solver.definition(),"forecast_array_sha256":forecast_hash,"audit":audit}

def replay(region,paths,policy,phase="evaluation",cutoff=60,scenario="main",nominal_mwh=2.,terminal_factor=1.,rerun=False,eta=.91,kappa=5.):
    # eta and kappa are additive model-sensitivity switches. Their defaults
    # reproduce the frozen v5 policy exactly, so every previously audited
    # report keeps its identical policy definition and completed cache.
    folder=OUT/scenario;folder.mkdir(parents=True,exist_ok=True)
    stem=f"{region}_{phase}_c{cutoff}_{policy}"
    dest=folder/f"{stem}.parquet";report=folder/f"{stem}.json"
    paths=paths.sort_values("target").reset_index(drop=True)
    x=paths[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)
    hashx=hashlib.sha256(x.tobytes()).hexdigest()
    solver=LexMPC(smin=.1*nominal_mwh,smax=.9*nominal_mwh,terminal_factor=terminal_factor,eta=eta,kappa=kappa)
    solver_sha=hashlib.sha256((ROOT/"scripts"/"revision_v5_control_solver.py").read_bytes()).hexdigest()
    safety_sha=hashlib.sha256((ROOT/"scripts"/"revision_v5_control_safe_policy.py").read_bytes()).hexdigest()
    scaled_sha=hashlib.sha256((ROOT/"scripts"/"revision_v5_control_scaled_policy.py").read_bytes()).hexdigest()
    enum_sha=hashlib.sha256((ROOT/"scripts"/"revision_v5_control_enumerated_policy.py").read_bytes()).hexdigest()
    level_definitions={"safe":SafeLexMPC(smin=.1*nominal_mwh,smax=.9*nominal_mwh,terminal_factor=terminal_factor,eta=eta,kappa=kappa).definition(),
        "scaled":ScaledSafeLexMPC(smin=.1*nominal_mwh,smax=.9*nominal_mwh,terminal_factor=terminal_factor,eta=eta,kappa=kappa).definition(),"enumerated":solver.definition()}
    def accepted_delegation(prior):
        if (prior.get("forecast_array_sha256")!=hashx or prior.get("solver_sha256")!=solver_sha
            or prior.get("nominal_energy_mwh")!=nominal_mwh):return None
        if prior.get("policy_definition")==level_definitions["enumerated"]:
            if prior.get("safety_sha256")==safety_sha and prior.get("scaled_safety_sha256")==scaled_sha and prior.get("enumerated_safety_sha256")==enum_sha:return "enumerated"
        if prior.get("policy_definition")==level_definitions["scaled"]:
            if prior.get("safety_sha256")==safety_sha and prior.get("scaled_safety_sha256")==scaled_sha:return "scaled"
        if prior.get("policy_definition")==level_definitions["safe"]:
            if prior.get("safety_sha256")==safety_sha:return "safe"
        return None
    def upgrade_delegated(prior,level):
        prior.update(policy_definition=solver.definition(),scaled_safety_sha256=scaled_sha,enumerated_safety_sha256=enum_sha)
        prior.setdefault("equivalent_unit_fallback_count",0);prior.setdefault("equivalent_unit_fallbacks",[])
        prior.setdefault("enumerated_fallback_count",0);prior.setdefault("enumerated_fallbacks",[])
        if level!="enumerated":prior["completed_lower_layer_delegation"]=f"immutable final wrapper delegates unchanged completed {level} solutions; no failed {level} origin among this completed prefix"
        return prior
    if report.exists() and not rerun:
        prior=json.loads(report.read_text(encoding="utf-8"))
        level=accepted_delegation(prior)
        if level:
            cached=pd.read_parquet(dest,columns=["target","actual_price"])
            assert np.array_equal(cached.target.to_numpy(dtype="datetime64[ns]"),paths.target.to_numpy(dtype="datetime64[ns]")),"Cached chronology differs"
            assert np.array_equal(cached.actual_price.to_numpy(),paths.actual_00.to_numpy()),"Cached actual prices differ"
            if level!="enumerated":
                archive=OUT/"before_logged_fallback_wrapper";archive.mkdir(exist_ok=True)
                preserved=archive/(report.stem+f"_{level}_delegated.json")
                if not preserved.exists():preserved.write_bytes(report.read_bytes())
                prior=upgrade_delegated(prior,level);report.write_text(json.dumps(prior,indent=2),encoding="utf-8")
            return prior
        old_definition=dict(solver.definition());old_definition.pop("numerical_failure_fallback");old_definition.pop("equivalent_unit_failure_fallback");old_definition.pop("soc_elimination_failure_fallback")
        # A completed frozen-core run has encountered no exception (otherwise
        # it exports no trajectory). The wrapper is identical on every such
        # origin; retain the audited completed array and preserve its report.
        prior_definition=dict(prior["policy_definition"]);prior_definition.pop("numerical_failure_fallback",None);prior_definition.pop("equivalent_unit_failure_fallback",None);prior_definition.pop("soc_elimination_failure_fallback",None)
        safe_definition=SafeLexMPC(smin=.1*nominal_mwh,smax=.9*nominal_mwh,terminal_factor=terminal_factor,eta=eta,kappa=kappa).definition()
        if (prior["forecast_array_sha256"]==hashx and prior["policy_definition"]==safe_definition
            and prior.get("solver_sha256")==solver_sha and prior.get("safety_sha256")==safety_sha
            and prior.get("nominal_energy_mwh")==nominal_mwh):
            archive=OUT/"before_logged_fallback_wrapper";archive.mkdir(exist_ok=True)
            preserved=archive/(report.stem+"_safe_final.json")
            if not preserved.exists():preserved.write_bytes(report.read_bytes())
            prior.update(policy_definition=solver.definition(),scaled_safety_sha256=scaled_sha,
                equivalent_unit_fallback_count=0,equivalent_unit_fallbacks=[],
                completed_safe_cache_reused="nested wrapper delegates unchanged immutable Safe on every successful Safe origin; no failed Safe origin in completed run")
            report.write_text(json.dumps(prior,indent=2),encoding="utf-8")
        if (prior.get("numerical_fallback_count",0)>0 and prior["forecast_array_sha256"]==hashx
            and prior_definition==old_definition and prior.get("solver_sha256")==solver_sha
            and (prior.get("safety_sha256")!=safety_sha or (prior["policy_definition"]!=solver.definition() and prior["policy_definition"]!=safe_definition))):
            archive=OUT/"before_logged_fallback_wrapper"/prior.get("safety_sha256","core")[:12]
            archive.mkdir(parents=True,exist_ok=True)
            for original in (report,dest):
                preserved=archive/original.name
                if not preserved.exists():shutil.copy2(original,preserved)
            print(json.dumps({"refresh_older_executed_fallback_policy":str(report),"old_safety_sha":prior.get("safety_sha256"),"current_safety_sha":safety_sha}),flush=True)
            return replay(region,paths,policy,phase,cutoff,scenario,nominal_mwh,terminal_factor,rerun=True)
        if (prior.get("numerical_fallback_count",0)==0 and prior["forecast_array_sha256"]==hashx
            and prior_definition==old_definition and prior.get("solver_sha256")==solver_sha
            and (prior.get("safety_sha256")!=safety_sha or prior["policy_definition"]!=solver.definition())
            and prior.get("nominal_energy_mwh")==nominal_mwh):
            archive=OUT/"before_logged_fallback_wrapper";archive.mkdir(exist_ok=True)
            preserved=archive/(report.stem+"_"+prior.get("safety_sha256","core")[:12]+".json")
            if not preserved.exists():preserved.write_bytes(report.read_bytes())
            prior.update(policy_definition=solver.definition(),safety_sha256=safety_sha,
                         scaled_safety_sha256=scaled_sha,equivalent_unit_fallback_count=0,equivalent_unit_fallbacks=[],
                         enumerated_safety_sha256=enum_sha,enumerated_fallback_count=0,enumerated_fallbacks=[],
                         numerical_fallback_count=0,numerical_fallbacks=[],
                         completed_core_only_cache_reused="wrapper delegates unchanged core on every successful origin; no failed origin in completed run")
            report.write_text(json.dumps(prior,indent=2),encoding="utf-8")
        if (prior["forecast_array_sha256"]==hashx and prior["policy_definition"]==solver.definition()
            and prior.get("solver_sha256")==solver_sha and prior.get("safety_sha256")==safety_sha
            and prior.get("scaled_safety_sha256")==scaled_sha
            and prior.get("enumerated_safety_sha256")==enum_sha
            and prior.get("nominal_energy_mwh")==nominal_mwh):
            return prior
        raise RuntimeError(f"Refuse stale replay {report}")
    assert np.isfinite(x).all() and paths.target.is_monotonic_increasing and not paths.target.duplicated().any()
    soc=.5*nominal_mwh;actual=paths.actual_00.to_numpy(float);rows=[];start=time.time();begin=0
    progress=OUT/"progress"/scenario;progress.mkdir(parents=True,exist_ok=True)
    progress_json=progress/f"{stem}.json";progress_parquet=progress/f"{stem}.parquet"
    def checkpoint():
        d=pd.DataFrame(rows,columns=FIELDS)
        d["target"]=paths.target.iloc[:len(d)].to_numpy();d["actual_price"]=actual[:len(d)]
        d.to_parquet(progress_parquet,index=False)
        meta={"forecast_array_sha256":hashx,"solver_sha256":solver_sha,"safety_sha256":safety_sha,
              "scaled_safety_sha256":scaled_sha,"enumerated_safety_sha256":enum_sha,"policy_definition":solver.definition(),
              "nominal_energy_mwh":nominal_mwh,"completed_origins":len(rows),"next_soc":soc,
              "numerical_fallback_count":solver.fallback_count,"numerical_fallbacks":solver.fallback_records,
              "equivalent_unit_fallback_count":solver.scaled_fallback_count,"equivalent_unit_fallbacks":solver.scaled_fallback_records,
              "enumerated_fallback_count":solver.enumerated_fallback_count,"enumerated_fallbacks":solver.enumerated_fallback_records}
        progress_json.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    if progress_json.exists() and not rerun:
        prefix=json.loads(progress_json.read_text());level=accepted_delegation(prefix)
        if not level:raise RuntimeError(f"Refuse mismatched incomplete prefix {progress_json}")
        d=pd.read_parquet(progress_parquet);begin=len(d)
        assert begin==prefix["completed_origins"] and begin<=len(paths)
        assert np.array_equal(d.target.to_numpy(dtype="datetime64[ns]"),paths.target.iloc[:begin].to_numpy(dtype="datetime64[ns]"))
        assert np.array_equal(d.actual_price.to_numpy(),actual[:begin])
        rows=d[FIELDS].to_numpy(float).tolist();soc=float(prefix["next_soc"])
        for field,meta in [("fallback_count","numerical_fallback_count"),("fallback_records","numerical_fallbacks"),
                           ("scaled_fallback_count","equivalent_unit_fallback_count"),("scaled_fallback_records","equivalent_unit_fallbacks"),
                           ("enumerated_fallback_count","enumerated_fallback_count"),("enumerated_fallback_records","enumerated_fallbacks")]:setattr(solver,field,prefix.get(meta,[] if field.endswith("records") else 0))
        print(json.dumps({"resumed_contiguous_prefix":str(progress_json),"completed_origins":begin,"unchanged_delegated_layer":level}),flush=True)
    for i in range(begin,len(x)):
        p=x[i];price=actual[i]
        try:sol=solver.solve(p,soc)
        except RuntimeError:checkpoint();raise
        c=sol["c"];d=sol["d"];cash=(d-c)*price*solver.dt;wear=(c+d)*solver.kappa*solver.dt
        rows.append((soc,c,d,sol["soc"],cash,wear,cash-wear,sol["primary_optimum_incumbent_aud"],
                     sol["policy_primary_loss_aud"],sol["policy_optimum_loss_upper_aud"],sol["primary_objective_gap_aud"]))
        soc=sol["soc"]
        if (i+1)%10000==0:
            checkpoint();print(json.dumps({"region":region,"policy":policy,"phase":phase,"scenario":scenario,"done":i+1,"seconds":time.time()-start}),flush=True)
    frame=pd.DataFrame({"target":paths.target,"actual_price":actual})
    frame["execution_start"]=frame.target-pd.Timedelta(hours=solver.dt)
    frame["execution_day"]=frame.execution_start.dt.floor("D")
    frame[FIELDS]=np.asarray(rows,float);frame.insert(0,"region",region)
    rep=metric(frame,solver,region,policy,phase,hashx);rep.update(cutoff_minutes=cutoff,scenario=scenario,
            nominal_energy_mwh=nominal_mwh,eta_c=eta,eta_d=eta,kappa_aud_per_grid_mwh=kappa,seconds=time.time()-start,trajectory_path=str(dest),
            solver_sha256=solver_sha,safety_sha256=safety_sha,endpoint_mark_factor=1.,
            scaled_safety_sha256=scaled_sha,
            enumerated_safety_sha256=enum_sha,
            numerical_fallback_count=solver.fallback_count,numerical_fallbacks=solver.fallback_records,
            equivalent_unit_fallback_count=solver.scaled_fallback_count,equivalent_unit_fallbacks=solver.scaled_fallback_records,
            enumerated_fallback_count=solver.enumerated_fallback_count,enumerated_fallbacks=solver.enumerated_fallback_records,
            resumed_prefix_origins=begin)
    frame.to_parquet(dest,index=False);report.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(json.dumps({"region":region,"policy":policy,"phase":phase,"scenario":scenario,"completed":True,"value_aud":rep["net_value_aud"],"seconds":rep["seconds"],"audit":rep["audit"]}),flush=True)
    return rep

def select_current(region):
    folder=OUT/"main";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"{region}_current_only_selection.json"
    prior_selection=json.loads(dest.read_text()) if dest.exists() else None
    val=load_inputs(region,policies=["raw","current_shrink_a025","current_shrink_a050","current_smooth_a025","current_smooth_a050"],phase="validation")
    candidates=[]
    for name,g in val.groupby("policy",sort=False):
        candidates.append(replay(region,g,name,phase="validation"))
    chosen=max(candidates,key=lambda x:(x["net_value_aud"],x["policy"]=="raw",x["policy"]))
    if prior_selection is not None:assert prior_selection["selected_policy"]==chosen["policy"],"Changed frozen current-only selection"
    rep={"status":"completed","region":region,"selection_period":"strict2023 complete-label cutoff",
         "selection_objective":"full rolling energy value + matching endpoint mark",
         "selected_policy":chosen["policy"],"candidate_results":candidates,
         "tie_rule":"exact equal values prefer raw; otherwise lexicographically greatest policy name",
         "uses_evaluation_outcomes":False}
    dest.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(json.dumps({"region":region,"current_only_selected":chosen["policy"]}),flush=True)
    return rep

def select_incremental(region):
    folder=OUT/"main";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"{region}_incremental_selection.json"
    prior_selection=json.loads(dest.read_text()) if dest.exists() else None
    f=load_inputs(region,kind="incremental",phase="validation")
    candidates=[]
    for name,g in f.groupby("policy",sort=False):candidates.append(replay(region,g,name,phase="validation"))
    chosen={}
    for family in ["level","history"]:
        pool=[r for r in candidates if r["policy"]=="incremental_a000" or r["policy"].startswith(f"incremental_{family}_")]
        if len(pool)!=4:raise RuntimeError(f"Incomplete incremental pool {region}/{family}")
        winner=max(pool,key=lambda x:(x["net_value_aud"],x["policy"]=="incremental_a000",x["policy"]))
        chosen[family]=winner["policy"]
    if prior_selection is not None:assert prior_selection["selected_policies"]==chosen,"Changed frozen incremental selection"
    rep={"status":"completed","region":region,"selected_policies":chosen,
         "candidate_results":candidates,"selection_objective":"strict2023 rolling net energy value with common endpoint",
         "alpha_grid":[0,.25,.5,1],"shared_alpha_zero":"incremental_a000",
         "tie_rule":"exact equal value prefer alpha0, otherwise lexicographically greatest policy name","uses_evaluation_outcomes":False}
    dest.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    for name in sorted(set(chosen.values())):
        replay(region,load_inputs(region,kind="incremental",policies=[name]),name)
    print(json.dumps({"region":region,"incremental_selected":chosen}),flush=True)
    return rep

def main_region(region,kind="baseline",cutoff=60,mode="main"):
    if mode=="baseline":
        # Main raw and sparse trajectories are emitted before the validation grid.
        f=load_inputs(region,kind="baseline",policies=["raw","sparse_equal"])
        for p in ["raw","sparse_equal"]:replay(region,f[f.policy.eq(p)],p)
        sel=select_current(region);p=sel["selected_policy"]
        if p!="raw":replay(region,load_inputs(region,policies=[p]),p)
        return
    policies=["raw","full_equal","inverse_lead"] if cutoff==60 and kind=="full" else ["raw","sparse_equal","inverse_lead"]
    f=load_inputs(region,kind=kind,cutoff=cutoff,policies=policies)
    available=f.policy.unique().tolist()
    base=load_inputs(region,kind="baseline",policies=["raw"]).sort_values("target")
    for p,g in f.groupby("policy",sort=False):
        g=g.sort_values("target")
        assert np.array_equal(g.target.to_numpy(),base.target.to_numpy()),f"Different chronology {region}/{p}"
        assert np.array_equal(g.actual_00.to_numpy(),base.actual_00.to_numpy()),f"Different actual prices {region}/{p}"
    wanted=[p for p in policies if p in available]
    for p in wanted:replay(region,f[f.policy.eq(p)],p,cutoff=cutoff)

def boundary(region,nominal_mwh=None,terminal_factor=None,policies=("raw","sparse_equal")):
    f=load_inputs(region,kind="baseline",policies=policies)
    if nominal_mwh is not None:
        scenario=f"nominal{nominal_mwh:g}mwh";energy=nominal_mwh;factor=1.
    else:
        scenario=f"terminal{terminal_factor:g}";energy=2.;factor=terminal_factor
    for p in policies:replay(region,f[f.policy.eq(p)],p,scenario=scenario,nominal_mwh=energy,terminal_factor=factor)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--regions",nargs="+",default=REGIONS)
    ap.add_argument("--mode",choices=["baseline","main","boundary","incremental"],default="baseline")
    ap.add_argument("--kind",choices=["baseline","full","incremental"],default="baseline")
    ap.add_argument("--cutoff",type=int,default=60);ap.add_argument("--nominal-mwh",type=float)
    ap.add_argument("--terminal-factor",type=float)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for region in a.regions:
        if a.mode=="boundary":boundary(region,a.nominal_mwh,a.terminal_factor)
        elif a.mode=="incremental":select_incremental(region)
        else:main_region(region,a.kind,a.cutoff,a.mode)
