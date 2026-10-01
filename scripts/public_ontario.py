"""Reconstruct Ontario inputs from exact official IESO source versions.

This entry point never publishes or mirrors downloaded IESO source material.
The original two-arm design is retained; source downloads are hash checked.
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True,help='New raw source directory; use F:/AcademicData/... on the author host.')
    p.add_argument('--output-root',type=Path,required=True,help='New computation directory outside the repo and raw directory.')
    p.add_argument('--stage',choices=['download','prepare','all'],default='prepare')
    p.add_argument('--workers',type=int,choices=[1,2],default=2)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];raw=a.raw_root.resolve();out=a.output_root.resolve()
    for other in [raw,repo]:
        if out==other or out.is_relative_to(other) or other.is_relative_to(out):p.error('Raw, computation and repository directories must not overlap.')
    if raw.exists() or out.exists():p.error('Both directories must be new; existing sources and results are preserved.')
    if os.name=='nt' and out.drive.upper()!='D:':p.error('Use D: for computation on the author Windows host.')
    raw.parent.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(raw.parent).free<10_000_000 or shutil.disk_usage(out).free<500_000_000:p.error('Insufficient free space.')
    sys.dont_write_bytecode=True
    import revision_v8_transfer as m
    import revision_v5_control_solver as cs
    import revision_v5_control_safe_policy as safe
    import revision_v5_control_scaled_policy as scaled
    import revision_v5_control_enumerated_policy as enum
    for module in [cs,safe,scaled,enum]:
        module.ROOT=repo;module.OUT=out/'numerical_failures'
    m.ROOT=repo;m.RAW=raw;m.OUT=out
    m.freeze();m.amend();m.download()
    expected=json.loads((repo/'provenance/ieso_source_versions.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    keys={r['url']:r['sha256'] for r in expected['files']}
    for row in manifest['files']:
        actual=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if actual!=keys[row['url']] or actual!=row['sha256']:raise RuntimeError('Official version bytes differ; source retained for inspection; no replay executed.')
        Path(row['path']).chmod(0o444)
    if a.stage!='download':m.prepare()
    if a.stage=='all':
        m.smoke();m.run(a.workers);m.analyze()
    (out/'public_reconstruction_receipt.json').write_text(json.dumps({'stage':a.stage,'official_source_hashes_verified':True,'full_solver_grid_executed':a.stage=='all','source_versions':expected['files'],'raw_files_redistributed':False},indent=2),encoding='utf-8')
    print(json.dumps({'completed_stage':a.stage,'output':str(out),'official_source_hashes_verified':True}))

if __name__=='__main__':main()
