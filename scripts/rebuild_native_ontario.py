"""Rebuild a bounded native IESO source/schema probe, not an economic replay.

The default phase checks paths, dependencies and archived probe source identities
without writing directories or contacting IESO. Download phases preserve the
original probe's September 2026 selection and download limits. Retention can
expire; successful preflight does not guarantee that those reports still exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "addenda" / "v14" / "native_ontario"
ATTRIBUTION = (
    "Copyright © 2004-2022 Independent Electricity System Operator, all rights reserved. "
    "This information is subject to the Terms of Use set out in the IESO’s website (www.ieso.ca)"
)
NOTICE = ATTRIBUTION + (
    "\nSource terms: https://www.ieso.ca/terms-of-use\n"
    "No endorsement by IESO is implied.\n"
    "Local source downloads and price-bearing parsed outputs are not licensed by this "
    "project for public redistribution. Follow IESO terms and any supplemental terms.\n"
)
MIN_RAW_BYTES = 20 * 1024**2
MIN_WORK_BYTES = 100 * 1024**2


def nearest_existing(path: Path) -> Path:
    while not path.exists():
        parent = path.parent
        if parent == path:
            raise ValueError(f"No existing filesystem ancestor: {path}")
        path = parent
    if not path.is_dir():
        raise ValueError(f"A parent path is a file: {path}")
    return path


def overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_paths(raw: Path, out: Path) -> dict:
    for label, path in (("raw-root", raw), ("output-root", out)):
        if path.exists():
            raise ValueError(f"{label} must be a wholly new directory; existing path preserved: {path}")
        if overlaps(path, REPO):
            raise ValueError(f"{label} must not overlap this repository: {path}")
    if overlaps(raw, out):
        raise ValueError("raw-root and output-root must be separate, non-overlapping new directories")
    raw_parent, out_parent = nearest_existing(raw), nearest_existing(out)
    raw_free = shutil.disk_usage(raw_parent).free
    work_free = shutil.disk_usage(out_parent).free
    if raw_free < MIN_RAW_BYTES or work_free < MIN_WORK_BYTES:
        raise ValueError("Insufficient free space for the bounded source/schema probe")
    return {
        "raw_filesystem_ancestor": str(raw_parent),
        "work_filesystem_ancestor": str(out_parent),
        "raw_free_bytes": raw_free,
        "work_free_bytes": work_free,
        "same_filesystem_allowed": True,
    }


def check_sources() -> dict:
    adaptations = json.loads((SOURCE / "PUBLIC_CODE_ADAPTATIONS.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row["public_sha256"] for row in adaptations["transformations"]}
    identities = {}
    for name in ("probe_native_ontario.py", "validate_native_ontario.py"):
        digest = sha256(SOURCE / name)
        if digest != expected[name]:
            raise ValueError(f"Archived native probe source differs from its recorded identity: {name}")
        identities[name] = digest
    return identities


class StorageBridge:
    """Resolve only legacy drive-capacity checks against the specified filesystems."""

    def __init__(self, raw: Path, out: Path):
        self.raw, self.out = raw, out

    def disk_usage(self, path):
        legacy = str(path).replace("\\", "/").upper()
        if legacy in ("D:/", "E:/", "F:/"):
            return shutil.disk_usage(self.out if legacy == "D:/" else self.raw)
        return shutil.disk_usage(path)

    def __getattr__(self, name):
        return getattr(shutil, name)


def load_entry(name: str, bridge: StorageBridge):
    spec = importlib.util.spec_from_file_location("rdia_native_" + Path(name).stem, SOURCE / name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the archived entry point: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.shutil = bridge
    return module


def write_notice(directory: Path):
    path = directory / "IESO_NOTICE.txt"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(NOTICE)
    path.chmod(0o444)


def main(argv=None) -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    raw_default = Path("F:/AcademicData/RDIA_native_ontario/raw") / stamp if os.name == "nt" and Path("F:/").exists() else None
    out_default = Path("D:/MLWork/RDIA_native_ontario") / stamp if os.name == "nt" and Path("D:/").exists() else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=raw_default,
                        help="Wholly new raw parent directory outside this checkout; Windows default uses F: when present.")
    parser.add_argument("--output-root", type=Path, default=out_default,
                        help="Wholly new computation directory outside this checkout; Windows default uses D: when present.")
    parser.add_argument("--phase", choices=("check", "probe", "probe-and-validate"), default="check",
                        help="check: no writes/network; probe: bounded source/version probe; probe-and-validate: probe then XSD validation. No phase runs optimization.")
    args = parser.parse_args(argv)
    if args.raw_root is None or args.output_root is None:
        parser.error("Specify --raw-root and --output-root when the F:/D: defaults are unavailable")
    raw, out = args.raw_root.expanduser().resolve(), args.output_root.expanduser().resolve()
    try:
        space = check_paths(raw, out)
        identities = check_sources()
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("requests", "fitz", "lxml")}
    required = ("requests", "fitz", "lxml") if args.phase == "probe-and-validate" else ("requests", "fitz")
    missing = [name for name in required if not dependencies[name]]
    if missing:
        parser.error("Missing source-probe dependencies: " + ", ".join(missing) + "; install requests, PyMuPDF and, for validation, lxml")
    scope = {
        "phase": args.phase,
        "raw_root": str(raw),
        "output_root": str(out),
        "source_only": True,
        "optimization_executed": False,
        "external_economic_replay_executed": False,
        "public_upload_performed": False,
        "full_month_prices_reconstructed": False,
        "independent_holdout_claim": False,
        "archived_selection": "September 2026 listings; first/last retained numbered report on 2026-09-01, up to two XML versions per family",
        "maximum_source_download_bytes": 10 * 1024**2,
        "space_check": space,
        "dependencies_available": dependencies,
        "archived_entry_sha256": identities,
        "legacy_capacity_check_mapping": {"D:/": "output-root filesystem", "E:/": "raw-root filesystem", "F:/": "raw-root filesystem"},
    }
    if args.phase == "check":
        print(json.dumps({**scope, "status": "PASS_PREFLIGHT_ONLY", "directories_created": False, "network_access_performed": False}, indent=2))
        return 0

    # These paths were checked as new. Exclusive creation preserves any race-created
    # directory instead of silently merging with it or replacing source material.
    raw.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    write_notice(raw)
    write_notice(out)
    keys = ("RDIA_NATIVE_RAW_ROOT", "RDIA_NATIVE_OUTPUT_ROOT")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[keys[0]], os.environ[keys[1]] = str(raw), str(out)
    start = time.monotonic()
    receipt = {**scope, "status": "STARTED_SOURCE_PROBE", "directories_created": True,
               "network_phase_requested": True, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    exit_code = 0
    try:
        bridge = StorageBridge(raw, out)
        load_entry("probe_native_ontario.py", bridge).main()
        receipt["probe_completed"] = True
        if args.phase == "probe-and-validate":
            load_entry("validate_native_ontario.py", bridge).main()
            receipt["xsd_validation_completed"] = True
        receipt["status"] = "PASS_SOURCE_PROBE_ONLY"
    except Exception as error:
        receipt.update(status="FAILED_SOURCE_PROBE_PARTIAL_FILES_PRESERVED", error_type=type(error).__name__, error_message=str(error))
        exit_code = 1
    finally:
        # Every subdirectory here was created below this invocation's new raw root.
        # Add sidecar notices; never alter bytes in a downloaded source file.
        for directory in raw.iterdir():
            if directory.is_dir() and not (directory / "IESO_NOTICE.txt").exists():
                write_notice(directory)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        receipt["elapsed_seconds"] = time.monotonic() - start
        receipt["archived_sources_unchanged"] = identities == check_sources()
        with (out / "native_source_wrapper_receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        print(json.dumps(receipt, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
