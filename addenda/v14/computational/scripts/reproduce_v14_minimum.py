"""Re-run two full Jan2024 trajectories and synthetic controls in a fresh OUT.

Does not alter frozen evidence or retrain/select on January outcomes.
"""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k]="1"
import argparse, datetime as dt, hashlib, importlib.metadata, json, platform
import shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
FIELDS=["soc_start_mwh","soc_end_mwh","charge_mw","discharge_mw","action_mw","actual_price","net_aud"]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding="utf-8")
def compare_frames(a,b):
    if len(a)!=1488 or len(b)!=1488:raise AssertionError("Expected complete1488-origin January replay")
    if not a.target.equals(b.target):raise AssertionError("Target alignment differs")
    result={}
    for name in FIELDS:
        x=a[name].to_numpy();y=b[name].to_numpy()
        same=bool(np.array_equal(x,y));error=float(np.max(np.abs(x-y)))
        result[name]={"bitwise_equal":same,"max_abs_difference":error}
        if not same:raise AssertionError(f"Trajectory field differs: {name}, max={error}")
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError("Fresh output must not exist: "+str(out))
    original=ROOT/"results/revision_v14/pilot"
    if out==original or original in out.parents:raise ValueError("Do not write within frozen pilot")
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();sys.path.insert(0,str(ROOT/"scripts"))
    import revision_v14_pilot as pilot
    pilot.OUT=out/"pilot";pilot.OUT.mkdir()
    for module in (pilot.core,pilot.safe,pilot.scaled,pilot.enum):module.OUT=out/"numerical_failures"
    for name in ("locked_design.json","trained_masks.json"):
        shutil.copyfile(original/name,pilot.OUT/name)
    design=json.loads((original/"locked_design.json").read_text())
    training=pilot.load("validation",[pilot.CURRENT,"inverse_lead"])
    ids=np.unique(np.linspace(0,len(training[pilot.CURRENT])-1,24,dtype=int))
    targets=training[pilot.CURRENT].target.iloc[ids].astype(str).tolist()
    if ids.tolist()!=design["training_indices"] or targets!=design["training_targets"]:
        raise AssertionError("Subset changed24 training indices/targets")
    checks={"complete_validation_origins_per_training_policy":len(training[pilot.CURRENT]),
            "training_indices_and_targets_exact":True,"training_masks_refit":False}
    replays={};fresh={}
    for policy in ("inverse_lead","static9"):
        pilot.replay(policy,.2)
        stem=policy+"_soc0.2"
        fresh[policy]=pd.read_parquet(pilot.OUT/(stem+".parquet"))
        frozen=pd.read_parquet(original/(stem+".parquet"))
        fresh_report=json.loads((pilot.OUT/(stem+".json")).read_text())
        frozen_report=json.loads((original/(stem+".json")).read_text())
        comparison=compare_frames(fresh[policy],frozen)
        if fresh_report["marked_value_aud"]!=frozen_report["marked_value_aud"]:
            raise AssertionError("Marked value differs")
        replays[policy]={"origins":len(frozen),"fields":comparison,
            "marked_value_aud":fresh_report["marked_value_aud"],"marked_value_exact":True,
            "fresh_seconds":fresh_report["seconds"]}
    static_compare=compare_frames(fresh["inverse_lead"],fresh["static9"])
    import revision_v14_toy_theory as toy
    toy.OUT=out/"theory"
    for module in (pilot.core,pilot.safe,pilot.scaled,pilot.enum):module.OUT=out/"numerical_failures"
    toy.main()
    actual=json.loads((toy.OUT/"executed_toy_controls.json").read_text())
    frozen=json.loads((ROOT/"results/revision_v14/theory/executed_toy_controls.json").read_text())
    def science(value):return {k:v for k,v in value.items() if k not in ("elapsed_seconds","sources")}
    if science(actual)!=science(frozen):raise AssertionError("Synthetic scientific outputs differ")
    receipt={"status":"PASS_FRESH_FULL_JANUARY_REPLAYS_AND_TOY", "run_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "package_root":str(ROOT),"output_root":str(out),"validation":checks,"replays":replays,
        "static9_fresh_exact_vs_inverse_lead":static_compare,
        "toy":{"status":actual["status"],"checks":actual["check_count"],"all_checks_pass":all(c["pass"] for c in actual["checks"]),
               "scientific_outputs_exact_to_frozen":True,"ignored_receipt_metadata_keys":["elapsed_seconds","sources"]},
        "elapsed_seconds":time.perf_counter()-started,
        "environment":{"python":sys.version,"platform":platform.platform(),
            "packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","pyarrow","scipy")}},
        "scope":"Exploratory month; floating-point window gate is not a rigorous or cumulative-profit certificate; no independent holdout, public release or raw archive redistribution."}
    write(out/"fresh_reproduction_receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"replays":2,"origins_each":1488,"toy_checks":actual["check_count"],
                      "seven_fields_and_marked_value_exact":True,"elapsed_seconds":receipt["elapsed_seconds"]}),flush=True)

if __name__=="__main__":main()
