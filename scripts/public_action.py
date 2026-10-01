"""Explicit, portable entry for the additive v13 action diagnostics.

Historical scientific modules are preserved byte for byte. This adapter only
locates immutable inputs, redirects writers, verifies hashes, and selects tasks.
It does not alter an objective, physical constraint, epsilon or action threshold.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, datetime as dt, hashlib, importlib, json, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=str, allow_nan=False), encoding='utf-8')


def verify_addition(package, data_root, required):
    manifest = package / 'provenance/v13_action_files.json'
    rows = json.loads(manifest.read_text(encoding='utf-8'))['files']
    index = {r['path']: r for r in rows}
    verified = {}
    for rel in required:
        if rel not in index:
            raise RuntimeError(f'Undeclared v13 dependency: {rel}')
        base = data_root if rel.startswith('results/') else package
        path = (base / rel).resolve()
        if not path.is_relative_to(base):
            raise RuntimeError(f'Unsafe v13 manifest path: {rel}')
        if not path.is_file():
            raise FileNotFoundError(f'Missing v13 input: {rel}; install the v13 Release asset')
        row = index[rel]
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError(f'v13 input hash mismatch: {rel}')
        verified[rel] = row['sha256']
    return verified, sha(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path,
                        help='Release dataset root containing results/; defaults to package root')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New absent directory outside immutable code/data roots')
    parser.add_argument('--mode', required=True,
                        choices=['material', 'ranges', 'case', 'delay-build', 'delay-replay'])
    parser.add_argument('--raw-root', type=Path,
                        help='Read-only directory of official monthly AEMO ZIPs; required for delay-build')
    parser.add_argument('--region', choices=['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1'])
    parser.add_argument('--policy', choices=['raw', 'sparse_equal', 'inverse_lead', 'cstar_fixed'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    package = args.package_root.resolve()
    data_root = (args.data_root or package).resolve()
    if Path(__file__).resolve() != package / 'scripts/public_action.py':
        raise RuntimeError('Use the public_action.py inside the selected package root')
    sys.path.insert(0, str(package / 'scripts'))
    core = importlib.import_module('public_reproduce')
    action = importlib.import_module('revision_v13_action_diagnostics')
    if Path(action.__file__).resolve() != package / 'scripts/revision_v13_action_diagnostics.py':
        raise RuntimeError('Scientific module was imported outside the selected package')
    index, core_manifest_sha = core.manifest_index(package)
    regions = action.REGIONS
    required = []
    extra = ['scripts/public_action.py', 'scripts/revision_v13_action_diagnostics.py']
    if args.mode in ('material', 'ranges'):
        required += [f'results/revision_v5/control/diagnostics/{r}_raw_{h}_common_state.parquet'
                     for r in regions for h in action.HISTORIES]
    if args.mode == 'ranges':
        required += [f'results/revision_v5/history/{r}_{family}.parquet'
                     for r in regions for family in ('baseline_inputs', 'inputs_aligned')]
    if args.mode == 'case':
        extra += ['results/revision_v5/control/diagnostics/SA1_posthoc_20240923_1900_common_soc.json']
    if args.mode == 'delay-build':
        if args.raw_root is None or not args.raw_root.resolve().is_dir():
            parser.error('--delay-build requires an existing explicit --raw-root directory')
        required += [f'results/revision_v5/history/{r}_baseline_inputs.parquet' for r in regions]
        required += [f'results/revision_v5/control/main/{r}_current_only_selection.json' for r in regions]
        extra += ['scripts/revision_v5_history_inputs.py', 'scripts/build_vintage_training_panel_crossregime.py']
    if args.mode == 'delay-replay':
        if args.region is None or args.policy is None:
            parser.error('delay-replay requires --region and --policy')
        extra += [f'results/revision_v13/action/delay30/{args.region}_delayed_inputs.parquet']
    # Numerical solver and safety code are frozen scientific manifest entries.
    required += [f'scripts/{name}.py' for name in ('revision_v5_control_solver',
        'revision_v5_control_safe_policy', 'revision_v5_control_scaled_policy',
        'revision_v5_control_enumerated_policy', 'revision_v5_control_replays')]
    checks = core.verify_files(package, data_root, index, required=required)
    added_verified, add_manifest_sha = verify_addition(package, data_root, extra)
    if args.check_only:
        print(json.dumps({'status': 'PASS', 'mode': args.mode, 'executed': False,
                          'core_required_inputs': len(required),
                          'addition_inputs': len(added_verified)}, indent=2))
        return
    out = core.fresh_output(args.output_root, package, data_root)
    started = time.perf_counter()
    action.ROOT, action.OUT = data_root, out
    action.MANIFEST.clear()
    core.configure_nem(package, data_root, out)
    result = {}
    if args.mode in ('material', 'ranges'):
        frames, plan = action.full_and_plan(25)
        if args.mode == 'ranges':
            result = action.sensitivity(frames, plan)
    elif args.mode == 'case':
        result = action.sa_certificate()
    elif args.mode == 'delay-build':
        # The preserved historical reconstruction writes relative paths under
        # ROOT. Copy only its ten verified small/derived dependencies into the
        # fresh workspace, so every writer remains outside immutable roots.
        for rel in required:
            if not rel.startswith('results/'):
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_root / rel, target)
            if sha(target) != index[rel]['sha256']:
                raise RuntimeError(f'Copied input differs: {rel}')
        action.ROOT = out
        action.OUT = out / 'results/revision_v13/action'
        action.OUT.mkdir(parents=True)
        reader = importlib.import_module('revision_v5_history_inputs')
        raw = importlib.import_module('build_vintage_training_panel_crossregime')
        raw_root = args.raw_root.resolve()
        raw.RAW_DIRS = [raw_root] + sorted({p.parent for p in raw_root.rglob('*.zip')})
        raw.PROJECT, raw.CACHE = out, out / 'cache'
        reader.ROOT, reader.OUT = out, action.OUT
        # Require all 33 monthly archives before beginning expensive reads.
        import pandas as pd
        for month in pd.date_range('2023-12-01', '2026-08-01', freq='MS'):
            if raw.month_archive('PREDISPATCHPRICE', month.year, month.month) is None:
                raise FileNotFoundError(f'Missing official PREDISPATCHPRICE {month:%Y-%m}')
        action.delayed_build()
        result = json.loads((action.OUT / 'delay30/input_reconstruction_complete.json').read_text())
    elif args.mode == 'delay-replay':
        rel = f'results/revision_v13/action/delay30/{args.region}_delayed_inputs.parquet'
        target = out / f'delay30/{args.region}_delayed_inputs.parquet'
        target.parent.mkdir(parents=True)
        shutil.copy2(data_root / rel, target)
        action.delay_replay(args.region, args.policy)
        result = json.loads((out / f'delay30/control/delay30/{args.region}_evaluation_c60_{args.policy}.json').read_text())
    outputs = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(out.rglob('*')) if p.is_file()]
    import numpy, pandas, scipy, pyarrow
    receipt = {'status': 'PASS', 'mode': args.mode, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'seconds': time.perf_counter()-started, 'execution_platform': sys.platform,
        'runtime': {'python': sys.version, 'numpy': numpy.__version__, 'pandas': pandas.__version__,
                    'scipy': scipy.__version__, 'pyarrow': pyarrow.__version__,
                    'solver': 'SciPy bundled HiGHS via the preserved controller module'},
        'scientific_script_sha256': sha(package / 'scripts/revision_v13_action_diagnostics.py'),
        'adapter_sha256': sha(Path(__file__)), 'core_manifest_sha256': core_manifest_sha,
        'v13_files_manifest_sha256': add_manifest_sha,
        'core_input_verification': checks, 'v13_verified_inputs': added_verified,
        'fixed_sample_per_region_per_history': 25 if args.mode == 'ranges' else None,
        'primary_epsilons_aud': [.001, .01, .1], 'material_thresholds_mw': [.05, .1, .2],
        'mathematical_code_modified_by_adapter': False,
        'full_twenty_delayed_replays_executed_by_this_invocation': False,
        'result': result, 'outputs': outputs}
    write(out / 'public_action_execution_receipt.json', receipt)
    print(json.dumps({'status': 'PASS', 'mode': args.mode, 'seconds': receipt['seconds'],
                      'receipt': str(out / 'public_action_execution_receipt.json')}, indent=2))


if __name__ == '__main__':
    main()
