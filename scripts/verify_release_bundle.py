"""Verify a complete release, then optionally execute its bounded reproductions.

The default is the complete 1.1.1 release, not a Git-only checkout. Checksum
verification, fresh solver execution, saved-path statistical reconstruction and
frozen-input redrawing are reported as separate operations. No raw download,
full policy grid, forecasting refit, or Zenodo deposit is performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True
VERSION = "1.1.1"
MANIFEST = "release_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def relative_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest paths must be nonempty strings")
    p = PurePosixPath(value)
    if (p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value) or p.as_posix() != value
            or not p.parts or p.parts[0] == ".git"):
        raise ValueError(f"Unsafe or noncanonical manifest path: {value!r}")
    return value


def inventory(root: Path) -> set[str]:
    """Exclude only the root .git entry; do not ignore caches or other files."""
    found = set()
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        here = Path(directory)
        if here == root:
            dirs[:] = [name for name in dirs if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in dirs + files:
            path = here / name
            if path.is_symlink():
                raise ValueError(f"Payload symlink is not permitted: {path}")
        found.update((here / name).relative_to(root).as_posix() for name in files)
    return found


def verify(root: Path, manifest_name: str, *, expected_version: str | None = None,
           checkout: bool = False) -> dict:
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_version is not None and manifest.get("version") != expected_version:
        raise ValueError(f"Expected release {expected_version}, found {manifest.get('version')!r}")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty manifest.files list is required")
    indexed = {}
    case_names = set()
    for row in rows:
        name = relative_name(row["path"])
        if name == manifest_name:
            raise ValueError("The manifest must exclude its own self-referential hash")
        if name.casefold() in case_names:
            raise ValueError(f"Duplicate/case-colliding manifest path: {name}")
        if (not isinstance(row.get("bytes"), int) or isinstance(row["bytes"], bool)
                or row["bytes"] < 0 or not isinstance(row.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError(f"Invalid byte count or SHA-256 for {name}")
        indexed[name] = row
        case_names.add(name.casefold())
    omitted = manifest.get("checkout_omitted_paths", [])
    if (not isinstance(omitted, list) or len(set(omitted)) != len(omitted)
            or any(relative_name(name) not in indexed for name in omitted)):
        raise ValueError("checkout_omitted_paths must enumerate unique declared payload paths")
    actual = inventory(root)
    expected = set(indexed) | {manifest_name}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = []
    verified = 0
    for name in sorted(set(indexed) & actual):
        row = indexed[name]
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append({"path": name, "error": "NOT_A_REGULAR_CONTAINED_FILE"})
        elif path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            errors.append({"path": name, "error": "SIZE_OR_SHA256_MISMATCH"})
        else:
            verified += 1
    disallowed_missing = sorted(set(missing) - (set(omitted) if checkout else set()))
    if errors or unexpected or disallowed_missing:
        status = "FAIL_RELEASE_INTEGRITY"
    elif missing:
        status = "CHECKOUT_INCOMPLETE_NOT_COMPLETE_RELEASE"
    else:
        status = "PASS_COMPLETE_RELEASE_INTEGRITY"
    return {"status": status, "verification_scope": "File inventory, byte counts and SHA-256 only; no computation",
            "manifest": manifest_name, "manifest_sha256": sha(manifest_path),
            "version": manifest.get("version"), "declared_payload_files": len(indexed),
            "declared_payload_bytes": sum(row["bytes"] for row in rows),
            "verified_payload_files": verified, "missing_paths": missing,
            "disallowed_missing_paths": disallowed_missing, "unexpected_paths": unexpected,
            "file_errors": errors, "checkout_mode": checkout,
            "complete_release_verified": status == "PASS_COMPLETE_RELEASE_INTEGRITY",
            "solver_execution": False, "statistical_recomputation": False}


def fresh_path(out: Path, roots: set[Path]) -> Path:
    out = out.resolve()
    if out.exists():
        raise ValueError(f"Use a new, absent output directory: {out}")
    for root in roots:
        if out == root or out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError("Output must not overlap an immutable payload/input root in either direction")
    return out


def public_worker(args) -> int:
    """Use the unchanged scientific adapter with an explicit output-path shim.

    This replaces only the adapter's author's-host D-drive restriction. Its
    manifest checks, solve code, input selection and comparison are unchanged.
    """
    root = args.package_root.resolve(strict=True)
    source = root / "scripts/public_reproduce.py"
    spec = importlib.util.spec_from_file_location("rdia_public_release_adapter", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def portable_fresh_output(output, package, data_root):
        out = fresh_path(Path(output), {package.resolve(), data_root.resolve()})
        out.mkdir(parents=True, exist_ok=False)
        return out
    module.fresh_output = portable_fresh_output
    command = [str(source), "--package-root", str(root)]
    if args._public_worker == "verify-all":
        command.append("--verify-all")
    else:
        if args.output_root is None:
            raise ValueError("Worker execution requires an explicit fresh --output-root")
        command += ["--smoke" if args._public_worker == "smoke" else "--oracle-only",
                    "--output-root", str(args.output_root)]
    sys.argv = command
    module.main()
    return 0


def run_step(name: str, command: list[str], scope: str, output: Path,
             root: Path, timeout: int, env: dict) -> dict:
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    stdout = logs / f"{name}.stdout.log"
    stderr = logs / f"{name}.stderr.log"
    start = time.perf_counter()
    result = {"name": name, "command": command, "scope": scope,
              "started_utc": utc(), "stdout_log": stdout.relative_to(output).as_posix(),
              "stderr_log": stderr.relative_to(output).as_posix()}
    print(f"START {name}: {scope}", flush=True)
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.run(command, cwd=root, env=env, stdout=out,
                                     stderr=err, timeout=timeout, check=False)
        result.update(exit_code=process.returncode,
                      status="PASS_PROCESS" if process.returncode == 0 else "FAIL_PROCESS")
    except subprocess.TimeoutExpired:
        result.update(exit_code=None, status="FAIL_TIMEOUT", timeout_seconds=timeout)
    except Exception as exc:
        result.update(exit_code=None, status="FAIL_PROCESS_START", error=str(exc))
    result.update(finished_utc=utc(), elapsed_seconds=time.perf_counter() - start)
    for label, path in (("stdout", stdout), ("stderr", stderr)):
        if path.is_file():
            result[label + "_sha256"] = sha(path)
    if result["status"] == "PASS_PROCESS":
        try:
            proof_path, proof = inspect_step_receipt(name, output, root, stdout)
            result["verified_execution_receipt"] = {
                "path": proof_path.relative_to(output).as_posix(),
                "sha256": sha(proof_path), "reported_status": proof["status"]}
        except Exception as exc:
            result.update(status="FAIL_MISSING_OR_INVALID_EXECUTION_RECEIPT", error=str(exc))
    print(f"{result['status']} {name} ({result['elapsed_seconds']:.2f} s)", flush=True)
    return result


def inspect_step_receipt(name: str, output: Path, cwd: Path, stdout: Path):
    """Require the real child receipt and the bounded execution counts."""
    paths = {"root_smoke": output / "smoke/public_execution_receipt.json",
             "root_oracle": output / "oracle/public_execution_receipt.json",
             "v14_computation": output / "v14/fresh_reproduction_receipt.json",
             "v15_computation": output / "v15/execution_receipt.json",
             "v16_redraw": cwd / "reproduction_receipt.json"}
    path = paths.get(name, stdout)
    proof = json.loads(path.read_text(encoding="utf-8"))
    valid = False
    if name == "root_manifest":
        valid = proof["status"] == "PASS_COMPLETE" and proof["missing_count"] == 0
    elif name == "v14_manifest":
        valid = proof["status"] == "PASS_COMPLETE_V14_MANIFEST"
    elif name == "root_smoke":
        valid = proof["status"] == "PASS" and proof["origins"] == 128 and proof["controller_reexecuted"] is True
    elif name == "root_oracle":
        valid = proof["status"] == "PASS" and proof["cases"] == 30
    elif name == "v14_computation":
        valid = (proof["status"] == "PASS_FRESH_FULL_JANUARY_REPLAYS_AND_TOY"
                 and set(proof["replays"]) == {"inverse_lead", "static9"}
                 and all(item["origins"] == 1488 and item["marked_value_exact"]
                         and all(field["bitwise_equal"] for field in item["fields"].values())
                         for item in proof["replays"].values())
                 and proof["toy"]["checks"] == 18 and proof["toy"]["all_checks_pass"])
    elif name == "v15_computation":
        valid = (proof["status"] == "PASS_FRESH_EXTRACTED_V15_COMPUTATION"
                 and proof["statistical_draws_per_block"] == 10000
                 and proof["block_lengths_days"] == [7, 28]
                 and proof["theory_groups"] == 11
                 and proof["parameter_origins"] == 1488 and proof["parameter_solves"] == 744
                 and proof["exact_formal_inference_json_match"]
                 and proof["exact_theory_json_match"] and proof["parameter_clock_identity_exact"])
    elif name == "v16_redraw":
        valid = (proof["status"] == "PASS_SIX_FROZEN_DATA_REDRAWS"
                 and len(proof["figures"]) == 6
                 and all(item["scientific_source_data"] == "EXACT_MATCH"
                         and item["collision_verdict"] == "PASS" for item in proof["figures"]))
    if not valid:
        raise ValueError(f"Child receipt does not establish the specified execution: {name}")
    return path, proof


def execute(args, root: Path, before: dict) -> int:
    output = fresh_path(args.output_root, {root})
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "execution_receipt.json"
    receipt = {"status": "RUNNING", "release_version": args.expected_version,
               "started_utc": utc(), "package_root": str(root), "output_root": str(output),
               "python": sys.version, "platform": platform.platform(), "steps": [],
               "initial_payload_verification": before,
               "path_adapter": "Only public_reproduce.fresh_output is replaced at runtime to accept explicit fresh paths on external Windows hosts; original source bytes and scientific functions remain unchanged",
               "scope": "Bounded fresh root smoke/oracle, v14 IL/static9 January replays and synthetic controls, v15 saved-path statistics/finite theory/one January hourly IL replay; optional six frozen-input figure redraws",
               "raw_source_download": False, "full_five_region_grid": False,
               "forecast_model_refit": False, "independent_confirmation": False,
               "zenodo_operation": False, "figures_requested": args.figures}
    write(receipt_path, receipt)
    if not before["complete_release_verified"]:
        receipt.update(status="FAIL_INITIAL_RELEASE_INTEGRITY", finished_utc=utc())
        write(receipt_path, receipt)
        return 1
    python = sys.executable
    worker = root / "scripts/verify_release_bundle.py"
    public = lambda mode, dest=None: [python, "-B", str(worker), "--package-root", str(root),
                                    "--_public-worker", mode] + (["--output-root", str(dest)] if dest else [])
    plans = [
        ("root_manifest", public("verify-all"), "330-file scientific integrity only; no solver"),
        ("root_smoke", public("smoke", output / "smoke"), "Fresh 128-origin NSW controller prefix and reference comparison"),
        ("root_oracle", public("oracle", output / "oracle"), "Thirty fresh SPO+ oracle/loss/subgradient cases; no model fit"),
        ("v14_manifest", [python, "-B", str(root / "addenda/v14/computational/scripts/verify_manifest.py"),
                          str(root / "addenda/v14/computational")], "Complete v14 payload byte verification; no replay"),
        ("v14_computation", [python, "-B", str(root / "addenda/v14/computational/scripts/reproduce_v14_minimum.py"),
                             "--output", str(output / "v14")], "Two fresh 1488-origin January IL/static9 replays and 18 finite synthetic controls"),
        ("v15_computation", [python, "-B", str(root / "addenda/v15/reproduce_v15.py"),
                             "--output", str(output / "v15")], "7/28-day saved-path statistics at 10000 draws, 11 finite theory groups, one fresh 1488-action/744-plan hourly IL replay")]
    receipt["planned_steps"] = [name for name, _, _ in plans] + (["v16_redraw"] if args.figures else [])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONOPTIMIZE="0",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.perf_counter()
    failure = False
    try:
        for name, command, scope in plans:
            result = run_step(name, command, scope, output, root, args.step_timeout_seconds, env)
            receipt["steps"].append(result)
            write(receipt_path, receipt)
            if result["status"] != "PASS_PROCESS":
                failure = True
                break
        if not failure and args.figures:
            source = root / "addenda/v16_figures"
            figure_check = verify(source, "manifest.json")
            receipt["v16_original_input_verification"] = figure_check
            if not figure_check["complete_release_verified"]:
                raise ValueError("v16 source payload failed its complete independent manifest")
            destination = output / "v16_redraw"
            shutil.copytree(source, destination)
            receipt["v16_fresh_copy_verification"] = verify(destination, "manifest.json")
            if not receipt["v16_fresh_copy_verification"]["complete_release_verified"]:
                raise ValueError("Copied v16 inputs differ before redraw")
            result = run_step("v16_redraw", [python, "-B", str(destination / "scripts/reproduce_v16_figures.py")],
                              "Six actual frozen-input native vector and 1200-dpi redraws with value/geometry checks; licensed Arial required",
                              output, destination, args.step_timeout_seconds, env)
            receipt["steps"].append(result)
            failure = result["status"] != "PASS_PROCESS"
    except Exception as exc:
        receipt["execution_error"] = str(exc)
        failure = True
    try:
        after = verify(root, MANIFEST, expected_version=args.expected_version)
        receipt["final_payload_verification"] = after
        receipt["immutable_payload_unchanged"] = (after["complete_release_verified"] and
                                                before["manifest_sha256"] == after["manifest_sha256"])
    except Exception as exc:
        receipt["final_payload_error"] = str(exc)
        receipt["immutable_payload_unchanged"] = False
    executed_names = {item["name"] for item in receipt["steps"]}
    receipt["not_executed_steps"] = [name for name in receipt["planned_steps"] if name not in executed_names]
    passed = (not failure and receipt["immutable_payload_unchanged"] and not receipt["not_executed_steps"]
              and all(item["status"] == "PASS_PROCESS" for item in receipt["steps"]))
    receipt.update(status="PASS_BOUNDED_FRESH_RELEASE_REPRODUCTION" if passed else "FAIL_BOUNDED_RELEASE_REPRODUCTION",
                   finished_utc=utc(), elapsed_seconds=time.perf_counter() - start)
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
                      "steps_executed": len(receipt["steps"]), "not_executed_steps": receipt["not_executed_steps"],
                      "immutable_payload_unchanged": receipt["immutable_payload_unchanged"]}, indent=2), flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Strict total inventory/size/SHA check; no computation")
    mode.add_argument("--execute", action="store_true", help="Verify, then actually run the bounded sequence")
    parser.add_argument("--output-root", type=Path, help="New directory outside the immutable package; required for execution")
    parser.add_argument("--expected-version", default=VERSION)
    parser.add_argument("--checkout", action="store_true", help="Check-only: enumerate precisely declared missing release-only paths; incomplete status exits 3")
    parser.add_argument("--figures", action="store_true", help="Execute optional redraws in a fresh copy; requires plotting dependencies and licensed Arial")
    parser.add_argument("--step-timeout-seconds", type=int, default=3600)
    parser.add_argument("--_public-worker", choices=["verify-all", "smoke", "oracle"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._public_worker:
        return public_worker(args)
    if not (args.check_only or args.execute):
        parser.error("Select --check-only or --execute")
    if args.checkout and not args.check_only:
        parser.error("--checkout is check-only; execution requires the complete release")
    if args.figures and not args.execute:
        parser.error("--figures requires --execute")
    if args.execute and args.output_root is None:
        parser.error("--execute requires an explicit fresh --output-root")
    if args.step_timeout_seconds < 1:
        parser.error("--step-timeout-seconds must be positive")
    root = args.package_root.resolve(strict=True)
    result = verify(root, MANIFEST, expected_version=args.expected_version, checkout=args.checkout)
    if args.check_only:
        print(json.dumps(result, indent=2), flush=True)
        return {"PASS_COMPLETE_RELEASE_INTEGRITY": 0, "CHECKOUT_INCOMPLETE_NOT_COMPLETE_RELEASE": 3}.get(result["status"], 1)
    return execute(args, root, result)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "FAIL_RELEASE_VERIFIER", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
