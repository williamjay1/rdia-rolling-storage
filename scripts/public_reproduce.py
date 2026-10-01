"""Portable entry point for the frozen NEM/RDIA study.

The historical scientific modules are preserved; this adapter only selects
immutable input locations, fresh output locations, and explicitly scoped tasks.
Smoke and oracle checks need no large Release asset. Full tasks need the frozen
derived dataset, not the original raw forecast archive.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import datetime as dt
import hashlib
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
CANDIDATES = ["raw", "current_shrink_a025", "current_shrink_a050",
              "current_smooth_a025", "current_smooth_a050"]
TASKS = ["il-mixed", "current-eval", "sensitivity", "walkforward",
         "statistics", "direct-tests", "theory", "synthetic", "branches",
         "common-state", "spo-fit"]
SMOKE_INPUT = "tests/fixtures/NSW1_raw_c60_inputs_128.parquet"
SMOKE_REFERENCE = "tests/fixtures/NSW1_raw_c60_reference_128.parquet"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def manifest_index(package):
    path = package / "package_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Scientific package_manifest.json is missing")
    m = json.loads(path.read_text(encoding="utf-8"))
    rows = m.get("source_files", []) + m.get("generated_files", [])
    indexed = {}
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or not (package / rel).resolve().is_relative_to(package):
            raise RuntimeError(f"Unsafe scientific manifest path: {rel}")
        key = rel.as_posix()
        if key in indexed:
            raise RuntimeError(f"Duplicate scientific manifest path: {key}")
        indexed[key] = row
    return indexed, sha(path)


def located(package, data_root, relative):
    # Release assets retain their original results/relative tree. Code and
    # authentic small fixtures stay in the Git repository.
    return (data_root if relative.startswith("results/") else package) / relative


def verify_files(package, data_root, index, required=None, verify_all=False):
    required = set(required or [])
    if verify_all:
        required.update(index)
    verified, absent = {}, []
    for rel, row in sorted(index.items()):
        path = located(package, data_root, rel)
        if not path.is_file():
            absent.append(rel)
            if rel in required:
                raise FileNotFoundError(
                    f"Required frozen input is missing: {rel}. Install the "
                    "checksummed NEM Release asset (see DATA_AVAILABILITY.md), "
                    "or use --smoke/--oracle-only without the large asset.")
            continue
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError(f"Scientific manifest mismatch: {rel}")
        verified[rel] = row["sha256"]
    undeclared = required - set(index)
    if undeclared:
        raise RuntimeError(f"Required files are undeclared: {sorted(undeclared)}")
    return {"verified_count": len(verified), "missing_count": len(absent),
            "missing_paths": absent, "verified_sha256": verified,
            "complete_dataset_verified": not absent}


def fresh_output(output, package, data_root):
    out = output.resolve()
    for source in {package, data_root}:
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Output must be separate from immutable code/data roots")
    if out.exists():
        raise FileExistsError(f"Use a fresh, absent output directory: {out}")
    # The local author's Windows computing-storage rule is D:. Linux/macOS
    # have no drive letter and may use any explicit separate writable path.
    if os.name == "nt" and out.drive.upper() != "D:":
        raise ValueError("On this Windows workflow, computational output must use D:")
    out.mkdir(parents=True, exist_ok=False)
    return out


def mod(package, name):
    result = importlib.import_module(name)
    if Path(result.__file__).resolve().parent != package / "scripts":
        raise RuntimeError(f"Imported scientific code outside package: {name}")
    return result


def configure_nem(package, data_root, output):
    # Four numerical-case writers are redirected before any optimization.
    for name in ("revision_v5_control_solver", "revision_v5_control_safe_policy",
                 "revision_v5_control_scaled_policy", "revision_v5_control_enumerated_policy"):
        m = mod(package, name)
        if hasattr(m, "ROOT"):
            m.ROOT = package
        m.OUT = output / "results/revision_v5/control"
    r = mod(package, "revision_v5_control_replays")
    r.ROOT, r.OUT = package, output / "results/revision_v5/control"
    r.HISTORY = data_root / "results/revision_v5/history"
    return r


def smoke(package, data_root, output):
    import numpy as np
    import pandas as pd
    r = configure_nem(package, data_root, output)
    inputs = pd.read_parquet(package / SMOKE_INPUT)
    if len(inputs) != 128 or not inputs.region.eq("NSW1").all():
        raise RuntimeError("Authentic NSW smoke fixture has changed")
    rep = r.replay("NSW1", inputs, "raw")
    fresh = pd.read_parquet(output / "results/revision_v5/control/main/NSW1_evaluation_c60_raw.parquet")
    # Reference actions/states are used only after the fresh cold solve.
    reference = pd.read_parquet(package / SMOKE_REFERENCE)
    if not fresh.target.equals(reference.target):
        raise RuntimeError("Smoke chronology differs")
    fields = ["soc_start", "charge_mw", "discharge_mw", "soc_end",
              "cashflow_aud", "degradation_aud", "net_aud"]
    errors = {key: float(np.max(abs(fresh[key].to_numpy() - reference[key].to_numpy())))
              for key in fields}
    if max(errors.values()) > 1e-6:
        raise RuntimeError(f"Smoke replay differs from frozen reference: {errors}")
    physics = rep["audit"]
    if physics["mode_violations_gt1e-6"] or physics["bounds_violations"]:
        raise RuntimeError("Smoke physics violation")
    return {"status": "PASS", "scope": "fresh 128-origin NSW latest/raw prefix only",
            "controller_reexecuted": True, "model_refitted": False,
            "reference_tolerance": 1e-6, "max_absolute_errors": errors,
            "physics": physics, "origins": 128,
            "full_task_grid_executed": False}


def oracle(package, data_root, output):
    configure_nem(package, data_root, output)
    s = mod(package, "revision_v8_spo")
    s.ROOT = package
    return s.verify(output)


def root_for_inputs_and_code(package, data_root):
    class RootRouter:
        def __truediv__(self, key):
            return (package if str(key) == "scripts" else data_root) / key
    return RootRouter()


def copy_reference(package, data_root, output, rel):
    target = output / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(located(package, data_root, rel), target)


def execute_task(args, package, data_root, output):
    import numpy as np
    if args.task in ("il-mixed", "current-eval", "sensitivity", "walkforward"):
        r = configure_nem(package, data_root, output)
        c = mod(package, "revision_v6_compute")
        c.ROOT, c.OUT = output, r.OUT
        c.V6 = output / "results/revision_v6"
        w = mod(package, "revision_v6_walkforward_complete")
        w.configure(package, output)
        w.OUT = data_root / "results/revision_v5/control"
        w.SOURCE_V6 = data_root / "results/revision_v6"
        w.reusable_prefix = lambda *_: (None, {"reusable": False,
            "reason": "Public execution always recomputes every origin; no old prefix reuse"})
        r.HISTORY = data_root / "results/revision_v5/history"
        reports = []
        for region in args.regions:
            reports.append(w.complete(region, require_prefix=False) if args.task == "walkforward"
                           else c.run(region, args.task, setting=args.setting, policy=args.policy))
        return {"status": "PASS", "task": args.task, "regions": args.regions,
                "fresh_controller_replay": True, "legacy_prefix_reuse": False,
                "reports": reports}
    if args.task == "statistics":
        s = mod(package, "revision_v7_statistics")
        s.ROOT, s.SOURCE, s.DEST = data_root, data_root / "results/revision_v5/control/main", output
        # Input recording accommodates code and data in separate roots.
        def record(path):
            p = Path(path)
            relative = p.relative_to(data_root if p.is_relative_to(data_root) else package).as_posix()
            s.MANIFEST[relative] = {"path": relative, "bytes": p.stat().st_size, "sha256": sha(p)}
        s.record = record
        s.run(args.draws)
        return {"status": "PASS", "task": args.task, "draws": args.draws,
                "scope": "statistics on saved paths; no solver or refit"}
    if args.task == "direct-tests":
        s = mod(package, "revision_v8_direct_tests")
        s.ROOT = data_root
        s.main(output, args.draws)
        return {"status": "PASS", "task": args.task, "draws": args.draws,
                "scope": "direct paired tests on saved paths; no solver or refit"}
    if args.task == "theory":
        s = mod(package, "revision_v8_theory_checks_v2")
        s.ROOT, s.OUT = package, output
        rng = np.random.default_rng(s.SEED)
        results = {"finite": s.finite_checks(rng), "reachable": s.reachable_checks(),
                   "storage": s.storage_checks(rng), "strong_concavity": s.quadratic_checks(rng),
                   "projection_controls": s.projection_controls(rng)}
        s.dump("pure_theory_results.json", results)
        return {"status": "PASS", "task": args.task, "LP_calls_executed": s.LP_CALLS,
                "scope": "pure synthetic/theoretical checks; observed surrogate omitted",
                "NEM_policy_replayed": False}
    if args.task == "synthetic":
        s = mod(package, "revision_v7_synthetic")
        saved = sys.argv
        try:
            sys.argv = [str(package / "scripts/revision_v7_synthetic.py"),
                        "--output", str(output), "--plan-only"]
            s.main()
            sys.argv = [str(package / "scripts/revision_v7_synthetic.py"),
                        "--output", str(output)]
            s.main()
        finally:
            sys.argv = saved
        return {"status": "PASS", "task": args.task, "scope": "explicitly synthetic, not market data"}
    if args.task in ("branches", "common-state"):
        r = configure_nem(package, data_root, output)
        d = mod(package, "revision_v5_control_diagnostics")
        x = mod(package, "revision_v6_extra")
        e = mod(package, "revision_v5_event_rolling")
        d.ROOT, d.OUT = package, r.OUT
        x.OUT = r.OUT
        e.ROOT, e.INPUTS = package, data_root / "results/revision_v5/history"
        e.OUT = output / "results/revision_v5/events"
        e.OUT.mkdir(parents=True, exist_ok=True)
        for region in args.regions:
            if args.task == "branches":
                for suffix in ("events.csv", "event_panel.parquet"):
                    copy_reference(package, data_root, output, f"results/revision_v5/events/{region}_{suffix}")
                raw = data_root / f"results/revision_v5/control/main/{region}_evaluation_c60_raw.parquet"
                e.probe_region(region, args.history, raw,
                               "revision_v5_control_scaled_policy", "ScaledSafeLexMPC")
            else:
                for policy in ("raw", args.history):
                    copy_reference(package, data_root, output,
                                   f"results/revision_v5/control/main/{region}_evaluation_c60_{policy}.parquet")
                function = d.common_state if args.history == "sparse_equal" else x.common_state_v6
                function(region, args.stride, .01, args.history)
        return {"status": "PASS", "task": args.task, "regions": args.regions,
                "history": args.history, "scope": "frozen selected events/states; fresh diagnostic solves"}
    if args.task == "spo-fit":
        configure_nem(package, data_root, output)
        s = mod(package, "revision_v8_spo")
        a = mod(package, "revision_v8_spo_audit")
        s.ROOT = package
        a.ROOT = root_for_inputs_and_code(package, data_root)
        # This environment variable changes only the frozen dataset location.
        os.environ["AOOR_DATA_ROOT"] = str(data_root)
        (output / "training_script_frozen.py").write_bytes((package / "scripts/revision_v8_spo.py").read_bytes())
        s.verify(output)
        s.write(output / "evaluation_analysis_plan.json", a.PLAN)
        s.fit(output)
        s.replay(output)
        a.run(output)
        return {"status": "PASS", "task": args.task,
                "scope": "full matched local fit, selection, two rolling replays, and audit"}
    raise ValueError(args.task)


def required_files(args, index):
    code = [p for p in index if p.startswith("scripts/") or p.startswith("configs/")]
    if args.smoke:
        return code + [SMOKE_INPUT, SMOKE_REFERENCE, "tests/fixtures/source_receipt.json"]
    if args.oracle_only or args.task in ("theory", "synthetic"):
        return code
    if not args.task:
        return code + [SMOKE_INPUT, SMOKE_REFERENCE, "tests/fixtures/source_receipt.json"]
    # The precise installed dataset is small enough to checksum on each full
    # task; this also prevents silent partial scientific reconstruction.
    return code + [p for p in index if p.startswith("results/")]


def runtime():
    import platform
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "threadpoolctl"):
        try:
            result[name] = importlib.import_module(name).__version__
        except ImportError:
            result[name] = "not installed"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--data-root", type=Path, help="root containing the Release results/ tree; default package root")
    p.add_argument("--output-root", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--oracle-only", action="store_true")
    mode.add_argument("--task", choices=TASKS)
    mode.add_argument("--check-only", action="store_true")
    p.add_argument("--verify-all", action="store_true", help="require every declared dataset file")
    p.add_argument("--regions", nargs="+", choices=REGIONS, default=REGIONS)
    p.add_argument("--setting", choices=["eta0.85", "eta0.95", "kappa2", "kappa10"])
    p.add_argument("--policy", choices=CANDIDATES + ["sparse_equal", "inverse_lead"])
    p.add_argument("--history", choices=["sparse_equal", "inverse_lead"], default="inverse_lead")
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--draws", type=int, default=10000)
    args = p.parse_args()
    if not (args.smoke or args.oracle_only or args.task or args.check_only or args.verify_all):
        p.error("Select --smoke, --oracle-only, --task, or --check-only")
    if args.setting and args.task != "sensitivity":
        p.error("--setting applies only to sensitivity")
    if args.policy and args.task not in ("current-eval", "sensitivity"):
        p.error("--policy applies only to current-eval or sensitivity")
    if args.task == "sensitivity" and args.policy and args.policy not in ("raw", "sparse_equal", "inverse_lead"):
        p.error("Sensitivity policy must be raw, sparse_equal or inverse_lead")
    if args.task == "current-eval" and args.policy and args.policy not in CANDIDATES:
        p.error("Current-eval requires a current-curve candidate")
    if args.stride < 1 or args.draws < 2000:
        p.error("Stride must be positive; bootstrap draws must be >=2000")
    package = args.package_root.resolve(strict=True)
    data_root = (args.data_root or package).resolve(strict=True)
    if not (package / "scripts/public_reproduce.py").is_file():
        p.error("Package root does not contain the public scientific entry point")
    sys.path.insert(0, str(package / "scripts"))
    index, manifest_sha = manifest_index(package)
    checks = verify_files(package, data_root, index, required_files(args, index), args.verify_all)
    if args.check_only or (args.verify_all and not (args.smoke or args.oracle_only or args.task)):
        print(json.dumps({"status": "PASS_COMPLETE" if checks["complete_dataset_verified"] else "PASS_AVAILABLE_DATA_ASSET_NOT_INSTALLED",
                          "manifest_sha256": manifest_sha, **{k:v for k,v in checks.items() if k != "verified_sha256"}}, indent=2))
        return
    if args.output_root is None:
        p.error("Execution requires an explicit fresh --output-root outside the immutable roots")
    output = fresh_output(args.output_root, package, data_root)
    start = time.perf_counter()
    try:
        result = smoke(package, data_root, output) if args.smoke else (
            oracle(package, data_root, output) if args.oracle_only else execute_task(args, package, data_root, output))
        result.update({"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "elapsed_seconds": time.perf_counter() - start,
                       "manifest_sha256": manifest_sha, "runtime": runtime(),
                       "input_checks": checks, "output_root": str(output),
                       "public_adapter_sha256": sha(Path(__file__))})
        write(output / "public_execution_receipt.json", result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("input_checks", "reports")}, indent=2))
    except Exception as exc:
        write(output / "public_failure_receipt.json", {"status":"FAILED", "error": repr(exc),
              "manifest_sha256": manifest_sha, "task":args.task,
              "smoke":args.smoke, "oracle_only":args.oracle_only})
        raise


if __name__ == "__main__":
    main()
