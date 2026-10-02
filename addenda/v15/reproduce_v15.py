"""Check payload hashes, then recompute the additive v15 scientific checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_manifest():
    manifest=json.loads((ROOT/"MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        p=(ROOT/item["path"]).resolve()
        assert p.is_relative_to(ROOT) and p.is_file()
        assert p.stat().st_size==item["bytes"] and sha(p)==item["sha256"],item["path"]
    return len(manifest["files"])

def call(arguments,cwd):
    p=subprocess.run([sys.executable,"-B",*map(str,arguments)],cwd=cwd,text=True,capture_output=True,encoding="utf-8",errors="replace")
    if p.returncode:raise RuntimeError(p.stdout+"\n"+p.stderr)
    print(p.stdout.strip(),flush=True)
    return p

def compare_csv(a,b):
    old,new=pd.read_csv(a),pd.read_csv(b)
    assert old.shape==new.shape and list(old.columns)==list(new.columns),(a.name,old.shape,new.shape)
    max_error=0.
    for c in old:
        if pd.api.types.is_numeric_dtype(old[c]):
            av,bv=old[c].to_numpy(float),new[c].to_numpy(float)
            assert np.array_equal(np.isnan(av),np.isnan(bv))
            error=float(np.nanmax(np.abs(av-bv)))
            max_error=max(max_error,error)
            assert np.allclose(av,bv,atol=1e-8,rtol=1e-12,equal_nan=True),(a.name,c,error)
        else:assert old[c].equals(new[c]),(a.name,c)
    return dict(file=a.name,rows=len(old),maximum_numeric_error=max_error,status="PASS")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--output",type=Path,help="New output directory outside the read-only package")
    args=parser.parse_args()
    count=verify_manifest()
    if args.check_only:
        print(json.dumps(dict(status="PASS_MANIFEST",files=count)),flush=True);return
    if args.output is None:parser.error("--output is required for a fresh scientific run")
    out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.exists():raise ValueError("Use a new output directory outside the package")
    out.mkdir(parents=True)
    call([ROOT/"scripts/revision_v15_statistics.py","--input-fixture",ROOT/"fixtures/statistics","--output",out/"statistics"],ROOT)
    stats=[]
    for name in ["family_inference.csv","mcs_membership.csv","break_even_incremental_budget.csv",
                 "pure_interior_cash_sensitivity.csv","current_selection_probabilities.csv","selection_sensitivity_distributions.csv"]:
        stats.append(compare_csv(ROOT/"frozen/statistics"/name,out/"statistics"/name))
    a=json.loads((ROOT/"frozen/statistics/formal_inference.json").read_text())
    b=json.loads((out/"statistics/formal_inference.json").read_text())
    assert a==b,"Exact SPA/RW/MCS receipt mismatch"
    theory=out/"theory";theory.mkdir()
    shutil.copyfile(ROOT/"scripts/verify_theory_v15.py",theory/"verify_theory_v15.py")
    call([theory/"verify_theory_v15.py"],theory)
    old=json.loads((ROOT/"frozen/theory/executed_validation_v15.json").read_text())
    new=json.loads((theory/"executed_validation_v15.json").read_text())
    assert old==new,"Theory receipt mismatch"
    setting="update60min_execute_two_planned_steps";policy="inverse_lead"
    call([ROOT/"scripts/revision_v15_parameter_pilot.py","--mode","single","--setting",setting,"--policy",policy,"--output",out/"parameters"],ROOT)
    name=setting+"__"+policy
    oldf=pd.read_parquet(ROOT/"frozen/parameters"/(name+".parquet"))
    newf=pd.read_parquet(out/"parameters"/(name+".parquet"))
    assert oldf.shape==newf.shape==(1488,13)
    fields=["soc_start_mwh","soc_end_mwh","charge_mw","discharge_mw","actual_price","net_aud"]
    err={c:float(np.max(np.abs(oldf[c].to_numpy()-newf[c].to_numpy()))) for c in fields}
    assert all(v<1e-8 for v in err.values()),err
    for c in ["target","execution_start","planning_origin_target","information_cutoff","replanned","setting","policy"]:
        assert oldf[c].equals(newf[c]),c
    oldp=json.loads((ROOT/"frozen/parameters"/(name+".json")).read_text())
    newp=json.loads((out/"parameters"/(name+".json")).read_text())
    for c in ["marked_value_aud","net_operating_aud","endpoint_mark_aud","initial_soc_mwh","final_soc_mwh"]:
        assert abs(oldp[c]-newp[c])<1e-8,c
    assert newp["solves"]==744 and newp["origins"]==1488
    receipt=dict(status="PASS_FRESH_EXTRACTED_V15_COMPUTATION",manifest_files=count,
         statistical_draws_per_block=10000,block_lengths_days=[7,28],statistics_csv_comparisons=stats,
         exact_formal_inference_json_match=True,theory_groups=len(new["checks"]),exact_theory_json_match=True,
         parameter_setting=setting,parameter_policy=policy,parameter_origins=1488,parameter_solves=744,
         parameter_numeric_field_errors=err,parameter_clock_identity_exact=True,
         parameter_marked_value_aud=newp["marked_value_aud"],
         source_script_sha256={p.name:sha(p) for p in sorted((ROOT/"scripts").glob("*.py"))},
         scope="Recomputed conditional statistics from exact derived daily fixture; finite synthetic algebra; one1488-action parameter replay. No original raw archive ingestion, full five-region optimisation or prospective confirmation.")
    (out/"execution_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(dict(status=receipt["status"],manifest_files=count,theory_groups=receipt["theory_groups"],parameter_origins=1488)),flush=True)

if __name__=="__main__":main()
