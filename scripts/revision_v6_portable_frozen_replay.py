"""Check or replay v6 frozen curves without modifying the supplied package.

Use the v6 manifest's actual hashes, not the earlier five-hash v5 declaration.
Every scientific execution writes to a fresh, separate D-drive directory.
Walk-forward candidate scores may read the package's historical trajectories;
these are selection evidence, never a newly executed full replay. Optional
--reuse-prefix explicitly continues an audited old prefix and labels that reuse.
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
from pathlib import Path
sys.dont_write_bytecode = True

CORE = [
    "revision_v5_control_solver.py", "revision_v5_control_safe_policy.py",
    "revision_v5_control_scaled_policy.py", "revision_v5_control_enumerated_policy.py",
    "revision_v5_control_replays.py", "revision_v5_control_aggregate.py",
    "revision_v6_compute.py", "revision_v6_attribution.py",
    "revision_v6_walkforward_complete.py", "revision_v6_portable_frozen_replay.py",
]
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
CANDIDATES = ["raw", "current_shrink_a025", "current_shrink_a050",
              "current_smooth_a025", "current_smooth_a050"]
SETTINGS = ["eta0.85", "eta0.95", "kappa2", "kappa10"]
LEDGER = "configs/revision_v6_runtime_source_hashes.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def index_expected(package):
    manifest = package / "package_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        rows = m.get("source_files", []) + m.get("generated_files", [])
        mapping = {}
        for row in rows:
            relative = Path(row["path"])
            resolved = (package / relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(package):
                raise RuntimeError(f"Unsafe manifest relative path: {relative}")
            key = relative.as_posix()
            if key in mapping:
                raise RuntimeError(f"Duplicate manifest path: {key}")
            mapping[key] = row
        return mapping, {"kind": "package_manifest", "path": str(manifest), "sha256": sha(manifest)}
    ledger = package / LEDGER
    if not ledger.exists():
        raise RuntimeError("Neither a v6 package manifest nor the explicit v6 runtime source ledger exists")
    m = json.loads(ledger.read_text(encoding="utf-8"))
    if m.get("status") != "declared v6 runtime sources":
        raise RuntimeError("Runtime ledger is not an explicit v6 source declaration")
    mapping = {p: {"path": p, "sha256": digest} for p, digest in m["scientific_code_sha256"].items()}
    return mapping, {"kind": "development runtime ledger; not a packaged-data freeze",
                     "path": str(ledger), "sha256": sha(ledger)}


def check_files(package, relative_files, expected, require_expected):
    got = {}
    for rel in sorted(set(relative_files)):
        path = (package / rel).resolve(strict=True)
        if not path.is_relative_to(package):
            raise RuntimeError(f"Input escaped package: {path}")
        digest = sha(path)
        declared = expected.get(rel)
        if require_expected and declared is None:
            raise RuntimeError(f"Required file has no manifest/hash-ledger declaration: {rel}")
        if declared:
            if digest != declared["sha256"]:
                raise RuntimeError(f"Declared hash differs: {rel}")
            if "bytes" in declared and path.stat().st_size != declared["bytes"]:
                raise RuntimeError(f"Declared size differs: {rel}")
        got[rel] = digest
    return got


def input_files(task, regions, smoke, reuse_prefix):
    files = []
    for region in regions:
        files.append(f"results/revision_v5/history/{region}_baseline_inputs.parquet")
        if task in ("il-mixed", "sensitivity"):
            files.append(f"results/revision_v5/history/{region}_inputs_aligned.parquet")
        if task == "walkforward":
            for phase in ("validation", "evaluation"):
                for policy in CANDIDATES:
                    stem = f"results/revision_v5/control/main/{region}_{phase}_c60_{policy}"
                    files.extend([stem + ".json", stem + ".parquet"])
            if reuse_prefix:
                stem = f"results/revision_v5/control/main/{region}_evaluation_c60_current_walkforward"
                files.extend([stem + ".json", stem + ".parquet",
                              f"results/revision_v6/compute/{region}_walkforward.json"])
        if smoke:
            # Read this reference only after the fresh 128-origin solve finishes.
            files.append(f"results/revision_v5/control/main/{region}_evaluation_c60_raw.parquet")
    return files


def configure(package, output):
    sys.path.insert(0, str(package / "scripts"))
    modules = {}
    for filename in CORE:
        if filename == Path(__file__).name:
            continue
        name = filename[:-3]
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != (package / "scripts").resolve():
            raise RuntimeError(f"Scientific module imported outside supplied scripts: {name}")
        modules[name] = module
    control = output / "results/revision_v5/control"
    for name in ("revision_v5_control_solver", "revision_v5_control_safe_policy",
                 "revision_v5_control_scaled_policy", "revision_v5_control_enumerated_policy"):
        module = modules[name]
        if hasattr(module, "ROOT"):
            module.ROOT = package
        module.OUT = control
    replay = modules["revision_v5_control_replays"]
    replay.ROOT, replay.OUT = package, control
    replay.HISTORY = package / "results/revision_v5/history"
    aggregate = modules["revision_v5_control_aggregate"]
    aggregate.ROOT, aggregate.OUT = package, control
    compute = modules["revision_v6_compute"]
    compute.ROOT, compute.OUT, compute.V6 = output, control, output / "results/revision_v6"
    attribution = modules["revision_v6_attribution"]
    attribution.ROOT, attribution.OUT = package, package / "results/revision_v5/control"
    attribution.V6, attribution.DEST = output / "results/revision_v6", output / "results/revision_v6/integrated"
    complete = modules["revision_v6_walkforward_complete"]
    complete.configure(package, output)
    # complete.configure also updates replay globals; all four numerical-failure
    # emitters must still target the fresh writable output, never the source.
    for name in ("revision_v5_control_solver", "revision_v5_control_safe_policy",
                 "revision_v5_control_scaled_policy", "revision_v5_control_enumerated_policy"):
        assert modules[name].OUT.resolve() == control.resolve()
    assert replay.ROOT == package and replay.OUT == control
    assert replay.HISTORY.resolve().is_relative_to(package)
    assert complete.OUT == package / "results/revision_v5/control"
    assert complete.COMPLETE_OUT == control
    paths = {name: {key: str(getattr(module, key)) for key in
                    ("ROOT", "OUT", "HISTORY", "V6", "DEST", "COMPLETE_OUT", "SOURCE_V6")
                    if hasattr(module, key)} for name, module in modules.items()}
    return replay, compute, complete, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--regions", nargs="+", choices=REGIONS, default=REGIONS)
    p.add_argument("--task", choices=["il-mixed", "current-eval", "sensitivity", "walkforward"])
    p.add_argument("--setting", choices=SETTINGS)
    p.add_argument("--policy", choices=CANDIDATES + ["sparse_equal", "inverse_lead"])
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="fresh 128-origin raw solve only")
    p.add_argument("--reuse-prefix", action="store_true", help="walkforward only: explicitly reuse its audited identical prefix")
    a = p.parse_args()
    package = a.package_root.resolve(strict=True)
    if not (package / "scripts").is_dir():
        p.error("Package root must contain scripts")
    if not (a.task or a.smoke or a.check_only):
        p.error("Select --task, --smoke, or --check-only")
    if a.smoke and a.task:
        p.error("The labelled smoke prefix cannot be combined with a full scientific task")
    if a.reuse_prefix and a.task != "walkforward":
        p.error("--reuse-prefix is explicit and applies only to --task walkforward")
    if a.setting and a.task != "sensitivity":
        p.error("--setting applies only to sensitivity")
    if a.policy and a.task not in ("current-eval", "sensitivity"):
        p.error("--policy applies only to current-eval or sensitivity")
    if a.task == "current-eval" and a.policy and a.policy not in CANDIDATES:
        p.error("Current-eval policy must be one of its five current-curve candidates")
    if a.task == "sensitivity" and a.policy and a.policy not in ("raw", "sparse_equal", "inverse_lead"):
        p.error("Sensitivity policy must be raw, sparse_equal, or inverse_lead")
    default = Path("D:/MLWork/AOOR_v6_portable_runs") / ("run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = (a.output_root or default).resolve()
    if output == package or output.is_relative_to(package) or package.is_relative_to(output):
        p.error("Fresh computation output must not overlap the read-only package in either direction")
    if output.exists():
        p.error("Fresh output already exists; cached package results will never execute a new full replay")
    if os.name == "nt" and output.drive.upper() != "D:":
        p.error("This Windows host requires D-drive computation output")
    expected, authority = index_expected(package)
    code = check_files(package, ["scripts/" + name for name in CORE], expected, True)
    if sha(__file__) != code["scripts/revision_v6_portable_frozen_replay.py"]:
        raise RuntimeError("Executing wrapper differs from the supplied package wrapper")
    needed = input_files(a.task, a.regions, a.smoke, a.reuse_prefix)
    inputs = check_files(package, needed, expected, authority["kind"] == "package_manifest")
    replay, compute, complete, paths = configure(package, output)
    receipt = {"status": "checked; no output created", "source_authority": authority,
               "package_root": str(package), "fresh_output_root": str(output),
               "regions": a.regions, "task": a.task, "smoke_only": a.smoke,
               "scientific_code_sha256": code, "input_sha256": inputs,
               "module_paths": paths, "source_bytecode_disabled": True,
               "source_candidate_trajectories_used_only_for_prior_scoring": a.task == "walkforward",
               "audited_execution_prefix_reuse_requested": a.reuse_prefix,
               "execution_kind": ("128-origin fresh smoke solve" if a.smoke else
                                  "continuous tail solve with explicit audited prefix reuse; not a fresh full solve"
                                  if a.reuse_prefix else "fresh complete task replay"),
               "model_refit": False, "raw_reconstruction": False,
               "source_result_cache_used_for_fresh_solver_execution": False,
               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    if a.check_only:
        print(json.dumps(receipt, indent=2), flush=True)
        return
    # Small smoke needs very little space. A full task should have room for its
    # trajectory and progress files; never redirect computation to another drive.
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < (64 * 1024**2 if a.smoke else 1024**3):
        raise RuntimeError("Insufficient D-drive space for the requested computation")
    output.mkdir(exist_ok=False)
    rp = output / "portable_v6_replay_receipt.json"
    rp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        if a.smoke:
            import numpy as np
            import pandas as pd
            checks = []
            for region in a.regions:
                source = replay.load_inputs(region, policies=["raw"]).sort_values("target").iloc[:128].copy()
                report = replay.replay(region, source, "raw", phase="portable_smoke", scenario="portable_smoke")
                new = pd.read_parquet(report["trajectory_path"])
                old = pd.read_parquet(package / "results/revision_v5/control/main" /
                                      f"{region}_evaluation_c60_raw.parquet").iloc[:128]
                fields = ["soc_start", "charge_mw", "discharge_mw", "soc_end",
                          "cashflow_aud", "degradation_aud", "net_aud"]
                assert len(new) == 128 and np.array_equal(old.target.to_numpy(), new.target.to_numpy())
                assert np.array_equal(old.actual_price.to_numpy(), new.actual_price.to_numpy())
                delta = float(np.max(np.abs(old[fields].to_numpy() - new[fields].to_numpy())))
                assert delta <= 1e-6, f"fresh raw prefix differs: {delta}"
                complete.validate_physics(report)
                checks.append({"region": region, "fresh_solved_origins": 128,
                               "recorded_reference_read_only_after_fresh_solve": True,
                               "maximum_state_action_cash_difference": delta,
                               "trajectory_path": report["trajectory_path"], "audit": report["audit"]})
            receipt["smoke_checks"] = checks
        elif a.task == "walkforward":
            if not a.reuse_prefix:
                complete.reusable_prefix = lambda *args: (None, {"reusable": False,
                    "reason": "portable full task requires every origin to be freshly solved"})
            for region in a.regions:
                complete.complete(region, require_prefix=a.reuse_prefix)
        else:
            for region in a.regions:
                if a.task == "sensitivity":
                    for setting in ([a.setting] if a.setting else SETTINGS):
                        compute.run(region, a.task, setting, a.policy)
                else:
                    compute.run(region, a.task, policy=a.policy)
        assert check_files(package, list(code), expected, True) == code
        assert check_files(package, list(inputs), expected, authority["kind"] == "package_manifest") == inputs
        receipt.update(status="completed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        rp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2), flush=True)
    except Exception as exc:
        receipt.update(status="failed; fresh output retained without skipping an origin", error=repr(exc))
        rp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
