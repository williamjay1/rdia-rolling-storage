"""Reproduce the matched experiment into a fresh, explicitly supplied folder."""
import argparse
from pathlib import Path
from revision_v8_spo import ROOT,verify,fit,replay,sha,write
from revision_v8_spo_audit import run,PLAN

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--oracle-only',action='store_true');a=ap.parse_args()
    out=a.output_dir.resolve()
    if (out/'design_freeze.json').exists() or (out/'selection_complete.json').exists():
        raise RuntimeError('Use a fresh output directory; existing executed evidence is retained')
    out.mkdir(parents=True,exist_ok=True)
    producer=ROOT/'scripts/revision_v8_spo.py';before=sha(producer)
    (out/'training_script_frozen.py').write_bytes(producer.read_bytes())
    verify(out)
    if not a.oracle_only:
        write(out/'evaluation_analysis_plan.json',PLAN)
        fit(out);replay(out);run(out)
    if sha(producer)!=before:raise RuntimeError('Producer source changed during execution')
