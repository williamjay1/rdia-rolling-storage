"""Matched-date economic comparisons of actually completed v5 policies."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from revision_v5_control_solver import ROOT,OUT
from revision_v5_control_replays import REGIONS

def paired_block_interval(interior,full_total,block=7,draws=10000,seed=20261001,nominal_days=974):
    x=np.asarray(interior,float);n=len(x);rng=np.random.default_rng(seed)
    offsets=np.arange(block);nblock=math.ceil(n/block);fixed=full_total-x.sum();values=[]
    for count in range(0,draws,250):
        size=min(250,draws-count);starts=rng.integers(0,n,(size,nblock))
        ix=((starts[:,:,None]+offsets)%n).reshape(size,-1)[:,:n]
        values.extend(((x[ix].sum(axis=1)+fixed)/nominal_days).tolist())
    return np.quantile(values,[.025,.975]).tolist()

def load_completed(region,policy,cutoff=60,scenario="main",phase="evaluation"):
    report=OUT/scenario/f"{region}_{phase}_c{cutoff}_{policy}.json"
    if not report.exists():return None
    r=json.loads(report.read_text());f=pd.read_parquet(r["trajectory_path"])
    f["target"]=f["target"].astype("datetime64[ns]")
    f["execution_day"]=f["execution_day"].astype("datetime64[ns]")
    return f,r

def compare_frame(a,b,total_a,total_b,region,contrast,assets=1):
    assert np.array_equal(a.target.to_numpy(dtype="datetime64[ns]"),b.target.to_numpy(dtype="datetime64[ns]"))
    assert np.array_equal(a.actual_price.to_numpy(),b.actual_price.to_numpy())
    diff=pd.DataFrame({"execution_day":a.execution_day,"delta_net_aud":a.net_aud-b.net_aud})
    daily=diff.groupby("execution_day").delta_net_aud
    delta=daily.sum();total=total_a-total_b;point=total/974
    ci=paired_block_interval(delta.iloc[1:-1],total)
    return {"region":region,"contrast":contrast,"asset_count":assets,
            "difference_full_value_aud":float(total),"difference_aud_per_nominal_day":float(point),
            "block7_ci_low_aud_per_nominal_day":ci[0],"block7_ci_high_aud_per_nominal_day":ci[1],
            "mechanical_aud_per_mw_year":point*365/assets,
            "mechanical_ci_low_aud_per_mw_year":ci[0]*365/assets,
            "mechanical_ci_high_aud_per_mw_year":ci[1]*365/assets,
            "conditional_daily_ci":"973 interior paired dates; fixed partial boundaries and common endpoint mark differences",
            "positive_daily_count":int((delta.iloc[1:-1]>0).sum()),"negative_daily_count":int((delta.iloc[1:-1]<0).sum()),
            "annualization_scope":"historical difference mechanically scaled to365 days per1MW; no forecast of future return"},diff

def comparisons():
    rows=[];dailyrows=[];complete=[]
    policies=["sparse_equal","full_equal","inverse_lead","inverse_archive_lead"]
    named_pairs=[]
    for p in policies:named_pairs.append((p,"raw",60,"main",p+" minus raw"))
    for cutoff in [30]:
        for p in ["sparse_equal","inverse_lead"]:named_pairs.append((p,"raw",cutoff,"main",f"fresh_c{cutoff} {p} minus raw"))
    named_pairs.append(("inverse_archive_lead","inverse_lead",60,"main","archive lead minus nominal issue lead"))
    for scenario in ["nominal1mwh","nominal4mwh","terminal0","terminal0.91"]:
        named_pairs.append(("sparse_equal","raw",60,scenario,scenario+" sparse minus raw"))
    for pa,pb,cutoff,scenario,label in named_pairs:
        aa=[];bb=[];totala=totalb=0.
        for region in REGIONS:
            a=load_completed(region,pa,cutoff,scenario);b=load_completed(region,pb,cutoff,scenario)
            if a is None or b is None:continue
            af,ar=a;bf,br=b;row,daily=compare_frame(af,bf,ar["net_value_aud"],br["net_value_aud"],region,label)
            rows.append(row);daily["region"]=region;daily["contrast"]=label;dailyrows.append(daily)
            aa.append(af);bb.append(bf);totala+=ar["net_value_aud"];totalb+=br["net_value_aud"]
        if len(aa)==5:
            agg_a=aa[0][["target","execution_day","actual_price"]].copy();agg_b=agg_a.copy()
            agg_a["net_aud"]=sum(f.net_aud for f in aa);agg_b["net_aud"]=sum(f.net_aud for f in bb)
            row,daily=compare_frame(agg_a,agg_b,totala,totalb,"five_independent_assets_sum",label,assets=5)
            rows.append(row);daily["region"]="five_independent_assets_sum";daily["contrast"]=label;dailyrows.append(daily);complete.append(label)
    # Validation winners are region-specific and are compared only after those
    # results and their completed evaluation trajectories exist.
    current_pairs={"sparse minus selected current-only":"sparse_equal",
                   "full_equal minus selected current-only":"full_equal",
                   "inverse_lead minus selected current-only":"inverse_lead"}
    for label in list(current_pairs)+["incremental history minus incremental level"]:
        aa=[];bb=[];ta=tb=0.
        for region in REGIONS:
            if label in current_pairs:
                path=OUT/"main"/f"{region}_current_only_selection.json"
                if not path.exists():continue
                selection=json.loads(path.read_text());pa=current_pairs[label];pb=selection["selected_policy"]
            else:
                path=OUT/"main"/f"{region}_incremental_selection.json"
                if not path.exists():continue
                selection=json.loads(path.read_text());pa=selection["selected_policies"]["history"];pb=selection["selected_policies"]["level"]
            a=load_completed(region,pa);b=load_completed(region,pb)
            if a is None or b is None:continue
            af,ar=a;bf,br=b;row,daily=compare_frame(af,bf,ar["net_value_aud"],br["net_value_aud"],region,label)
            row.update(policy_a=pa,policy_b=pb);rows.append(row);daily["region"]=region;daily["contrast"]=label;dailyrows.append(daily)
            aa.append(af);bb.append(bf);ta+=ar["net_value_aud"];tb+=br["net_value_aud"]
        if len(aa)==5:
            af=aa[0][["target","execution_day","actual_price"]].copy();bf=af.copy()
            af["net_aud"]=sum(f.net_aud for f in aa);bf["net_aud"]=sum(f.net_aud for f in bb)
            row,daily=compare_frame(af,bf,ta,tb,"five_independent_assets_sum",label,assets=5)
            rows.append(row);daily["region"]="five_independent_assets_sum";daily["contrast"]=label;dailyrows.append(daily);complete.append(label)
    # The interaction asks whether genuinely fresher current inputs change the
    # value of the same predeclared archive rule. All four policies continue
    # their own inventories, and every endpoint remains on the common scale.
    for policy in ("sparse_equal", "inverse_lead"):
        label=f"freshness archive-value interaction {policy}"
        aa=[];bb=[];ta=tb=0.
        for region in REGIONS:
            parts=[load_completed(region,policy,30),load_completed(region,"raw",30),
                   load_completed(region,policy,60),load_completed(region,"raw",60)]
            if any(part is None for part in parts):continue
            frames=[part[0] for part in parts];values=[part[1]["net_value_aud"] for part in parts]
            for f in frames[1:]:
                assert np.array_equal(frames[0].target.to_numpy(dtype="datetime64[ns]"),f.target.to_numpy(dtype="datetime64[ns]"))
                assert np.array_equal(frames[0].actual_price.to_numpy(),f.actual_price.to_numpy())
            af=frames[0][["target","execution_day","actual_price"]].copy();bf=af.copy()
            af["net_aud"]=frames[0].net_aud-frames[1].net_aud
            bf["net_aud"]=frames[2].net_aud-frames[3].net_aud
            va=values[0]-values[1];vb=values[2]-values[3]
            row,daily=compare_frame(af,bf,va,vb,region,label)
            row["contrast_scope"]="(archive minus raw at cutoff30) minus (archive minus raw at cutoff60); all four actual chronological replays"
            rows.append(row);daily["region"]=region;daily["contrast"]=label;dailyrows.append(daily)
            aa.append(af);bb.append(bf);ta+=va;tb+=vb
        if len(aa)==5:
            af=aa[0][["target","execution_day","actual_price"]].copy();bf=af.copy()
            af["net_aud"]=sum(f.net_aud for f in aa);bf["net_aud"]=sum(f.net_aud for f in bb)
            row,daily=compare_frame(af,bf,ta,tb,"five_independent_assets_sum",label,assets=5)
            row["contrast_scope"]="joint calendar blocks of the four-replay freshness interaction across five independent assets"
            rows.append(row);daily["region"]="five_independent_assets_sum";daily["contrast"]=label;dailyrows.append(daily);complete.append(label)
    if rows:pd.DataFrame(rows).to_csv(OUT/"paired_economic_comparisons.csv",index=False)
    if dailyrows:pd.concat(dailyrows).to_parquet(OUT/"paired_daily_contributions.parquet",index=False)
    return rows,complete

def decomposition():
    rows=[];grouped={}
    for region in REGIONS:
        rr=load_completed(region,"raw");hh=load_completed(region,"sparse_equal")
        rh=load_completed(region,"trade_raw__terminal_sparse_equal",scenario="input_2x2")
        hr=load_completed(region,"trade_sparse_equal__terminal_raw",scenario="input_2x2")
        if any(x is None for x in [rr,hh,rh,hr]):continue
        f=[x[0] for x in [rr,hh,rh,hr]];v=np.array([x[1]["net_value_aud"] for x in [rr,hh,rh,hr]])
        for g in f[1:]:assert np.array_equal(f[0].target.to_numpy(dtype="datetime64[ns]"),g.target.to_numpy(dtype="datetime64[ns]"))
        cases={"trading_component_symmetric":np.array([-.5,.5,-.5,.5]),
               "terminal_component_symmetric":np.array([-.5,.5,.5,-.5]),
               "interaction":np.array([1.,1.,-1.,-1.]),"total_history_difference":np.array([-1.,1.,0.,0.])}
        for name,w in cases.items():
            net=sum(a.net_aud*weight for a,weight in zip(f,w));df=pd.DataFrame({"execution_day":f[0].execution_day,"net":net})
            d=df.groupby("execution_day").net.sum();total=float(v@w);ci=paired_block_interval(d.iloc[1:-1],total)
            rows.append({"region":region,"component":name,"full_value_aud":total,"aud_per_nominal_day":total/974,
                         "block7_ci_low":ci[0],"block7_ci_high":ci[1],"definition":"four full chronological receding-policy replays; symmetric model-input decomposition, not frozen probes or market causal attribution"})
            grouped.setdefault(name,[]).append((d,total))
    for name,parts in grouped.items():
        if len(parts)!=5:continue
        d=sum(x[0] for x in parts);total=sum(x[1] for x in parts);ci=paired_block_interval(d.iloc[1:-1],total)
        rows.append({"region":"five_independent_assets_sum","component":name,"full_value_aud":total,"aud_per_nominal_day":total/974,"block7_ci_low":ci[0],"block7_ci_high":ci[1],"definition":"joint calendar-date blocks across all5 assets"})
    if rows:pd.DataFrame(rows).to_csv(OUT/"full_rolling_input_decomposition.csv",index=False)
    return rows

def main():
    rows,complete=comparisons();dec=decomposition()
    reports=[]
    for folder in ["main","input_2x2","nominal1mwh","nominal4mwh","terminal0","terminal0.91"]:
        for path in (OUT/folder).glob("*_evaluation_*.json"):
            r=json.loads(path.read_text());reports.append(r)
    metrics=[]
    for r in reports:
        row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        row["seconds_scope"]="last resumed segment" if r.get("resumed_prefix_origins",0) else "complete uninterrupted replay"
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUT/"all_completed_policy_metrics.csv",index=False)
    marker={"status":"interim" if not all(label in complete for label in ["sparse_equal minus raw","full_equal minus raw","inverse_lead minus raw","sparse minus selected current-only","full_equal minus selected current-only","inverse_lead minus selected current-only","incremental history minus incremental level"]) else "main_comparisons_completed",
            "completed_five_asset_contrasts":complete,"completed_policy_trajectories":len(reports),"paired_rows":len(rows),"decomposition_rows":len(dec),
            "uncertainty":"conditional circular day blocks; jointly aligned regions;973interior dates; fixed endpoint/partial boundary; no learner reselection"}
    (OUT/"aggregate_status.json").write_text(json.dumps(marker,indent=2),encoding="utf-8")
    print(json.dumps(marker),flush=True)

if __name__=="__main__":main()
