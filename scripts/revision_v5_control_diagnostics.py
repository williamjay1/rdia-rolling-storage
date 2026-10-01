"""Aligned raw-versus-history inputs, SOC, and near-optimal action ranges."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import argparse,json,time
import numpy as np
import pandas as pd
from revision_v5_control_solver import ROOT,OUT
from revision_v5_control_enumerated_policy import EnumeratedSafeLexMPC as LexMPC
from revision_v5_control_replays import load_inputs,replay,REGIONS

def near_optimal_range(model,path,soc,amount_aud=.01):
    """Outer numerical action interval for an epsilon-primary-optimal set.

    The primary incumbent plus epsilon is a superset of the true epsilon set
    whenever its certified minimization gap is nonzero. Min/max first-action
    dual bounds produce outer endpoints, never just incumbent endpoints.
    Throughput and tertiary preferences are removed for this diagnostic.
    """
    baseline=model.solve(path,soc);h=model.h
    model.solver.changeRowBounds(model.primary_row,-np.inf,
                                -baseline["primary_optimum_incumbent_aud"]+amount_aud)
    model.solver.changeRowBounds(model.throughput_row,-np.inf,np.inf)
    model.solver.changeRowBounds(model.branch_row,-np.inf,np.inf)
    lo=model._run(model.first_cost)
    hi=model._run(-model.first_cost)
    if lo is None or hi is None:raise RuntimeError("Primary-near-optimal range infeasible")
    lx,lp,lb=lo;hx,hp,hb=hi
    return {"range_lower_outer_mw":float(lb),"range_upper_outer_mw":float(-hb),
            "range_lower_incumbent_mw":float(lp),"range_upper_incumbent_mw":float(-hp),
            "lower_branch_gap_mw":max(0.,lp-lb),"upper_branch_gap_mw":max(0.,hp-hb),
            "primary_gap_aud":baseline["primary_objective_gap_aud"],
            "primary_incumbent_aud":baseline["primary_optimum_incumbent_aud"],
            "primary_epsilon_aud":amount_aud,"policy_u_mw":baseline["u"]}

def mixed_replays(region,history="sparse_equal"):
    f=load_inputs(region,policies=["raw",history]);a=f[f.policy.eq("raw")].sort_values("target").reset_index(drop=True)
    b=f[f.policy.eq(history)].sort_values("target").reset_index(drop=True)
    assert a.target.equals(b.target)
    for head,tail in [("raw",history),(history,"raw")]:
        name=f"trade_{head}__terminal_{tail}"
        out=a.copy()
        if head!="raw":out[[f"p_{j:02d}" for j in range(1,9)]]=b[[f"p_{j:02d}" for j in range(1,9)]]
        if tail!="raw":out[[f"p_{j:02d}" for j in range(9,13)]]=b[[f"p_{j:02d}" for j in range(9,13)]]
        replay(region,out,name,scenario="input_2x2")

def illustrative_sa_point():
    """Post-hoc illustration requested after the independently sampled events."""
    region="SA1";target=pd.Timestamp("2024-09-23 19:00:00");soc=1.8
    f=load_inputs(region,policies=["raw","sparse_equal"])
    subset=f[f.target.eq(target)];paths={}
    for policy,g in subset.groupby("policy"):
        if len(g)!=1:raise RuntimeError("Illustrative time not unique")
        paths[policy]=g[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)[0]
    model=LexMPC();cells={};ranges={}
    for head,tail in [("raw","raw"),("raw","sparse_equal"),("sparse_equal","raw"),("sparse_equal","sparse_equal")]:
        path=np.r_[paths[head][:8],paths[tail][8:]]
        sol=model.solve(path,soc)
        cells[f"trade_{head}__terminal_{tail}"]={k:float(sol[k]) for k in ["c","d","u","soc","primary_optimum_incumbent_aud","policy_optimum_loss_upper_aud"]}
    for name,path in paths.items():ranges[name]=near_optimal_range(model,path,soc,.01)
    rep={"status":"completed","region":region,"target":str(target),"common_soc_mwh":soc,
        "selection":"explicitly post-hoc illustration of SA event178 step8; excluded from fixed-stride and fixed-event inferential summaries",
        "actual_price":float(subset.actual_00.iloc[0]),"forecast_paths":{k:v.tolist() for k,v in paths.items()},
        "common_state_input_swap_cells":cells,"primary_near_optimal_ranges":ranges}
    folder=OUT/"diagnostics";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/"SA1_posthoc_20240923_1900_common_soc.json"
    dest.write_text(json.dumps(rep,indent=2),encoding="utf-8");print(json.dumps(rep),flush=True)

def common_state(region,stride=32,epsilon_aud=.01,history="sparse_equal"):
    folder=OUT/"diagnostics";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"{region}_raw_{history}_common_state.parquet"
    repdest=folder/f"{region}_raw_{history}_common_state.json"
    if repdest.exists():print(f"skip {repdest}",flush=True);return
    inputs=load_inputs(region,policies=["raw",history]);a=inputs[inputs.policy.eq("raw")].sort_values("target").reset_index(drop=True)
    b=inputs[inputs.policy.eq(history)].sort_values("target").reset_index(drop=True)
    ar=pd.read_parquet(OUT/"main"/f"{region}_evaluation_c60_raw.parquet")
    br=pd.read_parquet(OUT/"main"/f"{region}_evaluation_c60_{history}.parquet")
    ar["target"]=ar["target"].astype("datetime64[ns]");br["target"]=br["target"].astype("datetime64[ns]")
    assert a.target.equals(b.target) and a.target.equals(ar.target) and a.target.equals(br.target)
    ac=a[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float);bc=b[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)
    model=LexMPC();records=[];start=time.time()
    for i in range(0,len(a),stride):
        sr=float(ar.soc_start.iloc[i]);sh=float(br.soc_start.iloc[i])
        rr=model.solve(ac[i],sr);hr=model.solve(bc[i],sr);rh=model.solve(ac[i],sh);hh=model.solve(bc[i],sh)
        r=near_optimal_range(model,ac[i],sr,epsilon_aud);v=near_optimal_range(model,bc[i],sr,epsilon_aud)
        row={"target":a.target.iloc[i],"sample_index":i,"raw_soc_mwh":sr,"history_soc_mwh":sh,
             "u_raw_at_raw_soc":rr["u"],"u_history_at_raw_soc":hr["u"],
             "u_raw_at_history_soc":rh["u"],"u_history_at_history_soc":hh["u"],
             "actual_raw_u_mw":float(ar.discharge_mw.iloc[i]-ar.charge_mw.iloc[i]),
             "actual_history_u_mw":float(br.discharge_mw.iloc[i]-br.charge_mw.iloc[i])}
        row.update({f"raw_{k}":value for k,value in r.items()});row.update({f"history_{k}":value for k,value in v.items()})
        info=((hr["u"]-rr["u"])+(hh["u"]-rh["u"]))/2
        state=((rh["u"]-rr["u"])+(hh["u"]-hr["u"]))/2
        row.update(information_component_mw=info,state_component_mw=state,diagonal_difference_mw=hh["u"]-rr["u"],
             input_action_change_gt01=abs(hr["u"]-rr["u"])>.1,
             primary_sets_force_change_gt01=(r["range_lower_outer_mw"]>v["range_upper_outer_mw"]+.1 or
                                           v["range_lower_outer_mw"]>r["range_upper_outer_mw"]+.1),
             near_optimal_ranges_overlap=(r["range_lower_outer_mw"]<=v["range_upper_outer_mw"]+1e-7 and
                                         v["range_lower_outer_mw"]<=r["range_upper_outer_mw"]+1e-7))
        records.append(row)
    d=pd.DataFrame(records);d.insert(0,"region",region);d.to_parquet(dest,index=False)
    changed=d.input_action_change_gt01
    rep={"status":"completed","region":region,"history":history,"sample_stride":stride,"origins":len(d),
         "primary_near_optimal_epsilon_aud":epsilon_aud,"seconds":time.time()-start,
         "same_raw_SOC_change_share_gt01":float(changed.mean()),
         "same_hist_SOC_change_share_gt01":float((abs(d.u_history_at_history_soc-d.u_raw_at_history_soc)>.1).mean()),
         "actual_diagonal_change_share_gt01":float((abs(d.actual_history_u_mw-d.actual_raw_u_mw)>.1).mean()),
         "primary_sets_force_change_share_gt01":float(d.primary_sets_force_change_gt01.mean()),
         "forced_share_among_changed":float(d.loc[changed,"primary_sets_force_change_gt01"].mean()) if changed.any() else None,
         "overlap_share_among_changed":float(d.loc[changed,"near_optimal_ranges_overlap"].mean()) if changed.any() else None,
         "max_decomposition_identity_error_mw":float(abs(d.information_component_mw+d.state_component_mw-d.diagonal_difference_mw).max()),
         "max_cold_raw_vs_executed_diff_mw":float(abs(d.u_raw_at_raw_soc-d.actual_raw_u_mw).max()),
         "max_cold_history_vs_executed_diff_mw":float(abs(d.u_history_at_history_soc-d.actual_history_u_mw).max()),
         "numerical_bounds_note":"First action outer endpoints use LP/MIP dual bounds; primary incumbent+epsilon allows reported primary gap. Numerical certificates, not mathematical exact-arithmetic proof.",
         "scope":"fixed stride across complete exploration period; common SOC analysis explains model changes, not market causal effects"}
    repdest.write_text(json.dumps(rep,indent=2),encoding="utf-8");print(json.dumps(rep),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--regions",nargs="+",default=REGIONS)
    ap.add_argument("--mode",choices=["mixed","common","illustrative"],default="common");ap.add_argument("--stride",type=int,default=32)
    a=ap.parse_args()
    if a.mode=="illustrative":illustrative_sa_point();raise SystemExit
    for region in a.regions:
        if a.mode=="mixed":mixed_replays(region)
        else:common_state(region,a.stride)
