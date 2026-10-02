"""Exercise the actual downloader against a local HTTP release fixture.

No GitHub request is sent. Only urllib's transport destination is remapped;
the production fetch, checksum, ZIP and complete-inventory code runs unchanged.
Fixtures are labelled synthetic and contain no research or provider data.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import importlib.util
import io
import json
import stat
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
PREFIX = "https://github.com/williamjay1/rdia-rolling-storage/releases/download/"
CURRENT = "1.1.1"
UA = "RDIA-reproducibility/1.1.1"


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fixture(version: str, variant="valid") -> bytes:
    payload = {
        "README.md": b"Synthetic downloader regression fixture; not market evidence.\n",
        "fixtures/synthetic_data.csv": b"fixture_id,value\nsynthetic_1,2\nsynthetic_2,-3\n",
        "scripts/fixture_model.py": b"# Synthetic fixture, not a scientific estimator.\nVALUE = 2\n",
    }
    records = [{"path": name, "bytes": len(body), "sha256": digest(body)}
               for name, body in sorted(payload.items())]
    manifest = {"version": version, "scope": "Synthetic test fixture only", "files": records}
    extras = {}
    if variant == "wrong_manifest_version":
        manifest["version"] = "0.0.0"
    elif variant == "duplicate_manifest":
        manifest["files"].append(copy.deepcopy(records[0]))
    elif variant == "invalid_manifest_size":
        manifest["files"][0]["bytes"] = -1
    elif variant == "corrupt_payload":
        payload["README.md"] += b"Unmanifested alteration.\n"
    elif variant == "missing_payload":
        del payload["scripts/fixture_model.py"]
    elif variant == "extra_payload":
        extras["undeclared.txt"] = b"Not declared.\n"
    elif variant == "traversal":
        extras["../escape.txt"] = b"Must never be extracted.\n"
    elif variant == "case_collision":
        extras["readme.md"] = b"Would overwrite README.md on Windows.\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, body in {**payload, **extras}.items():
            archive.writestr(name, body)
        if variant == "symlink":
            link = zipfile.ZipInfo("unsafe_link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "../escape.txt")
        archive.writestr("release_manifest.json", json.dumps(manifest, indent=2).encode())
    body = bytearray(buffer.getvalue())
    if variant == "bad_crc":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            header = archive.getinfo("fixtures/synthetic_data.csv").header_offset
        name_length, extra_length = struct.unpack_from("<HH", body, header + 26)
        body[header + 30 + name_length + extra_length] ^= 1
    elif variant == "oversize_header":
        central = body.index(b"PK\x01\x02")
        struct.pack_into("<L", body, central + 24, 512 * 1024**2 + 1)
    return bytes(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New absolute directory outside the immutable source/payload")
    args = parser.parse_args()
    root = args.package_root.resolve(strict=True)
    source = root / "scripts/download_release.py"
    require(source.is_file(), "Downloader source is missing")
    require(args.output_root.is_absolute(), "Use an explicit absolute output directory")
    out = args.output_root.resolve()
    require(not out.exists(), "Output directory must be new")
    require(not (out == root or out.is_relative_to(root) or root.is_relative_to(out)),
            "Output must not overlap the source/payload in either direction")
    source_before = digest(source.read_bytes())
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("download_release_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.DEFAULT_VERSION == CURRENT, "Unexpected current default release version")
    state = {"routes": {}, "requests": []}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append({"path": self.path, "user_agent": self.headers.get("User-Agent")})
            body = state["routes"].get(self.path)
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Length", str(len(body) if body is not None else 0))
            self.end_headers()
            if body is not None:
                self.wfile.write(body)
        def log_message(self, *arguments):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    original_urlopen = urllib.request.urlopen
    def local_urlopen(request, *arguments, **keywords):
        require(isinstance(request, urllib.request.Request), "Production fetch must use a Request")
        require(request.full_url.startswith(PREFIX), "Unexpected outgoing destination")
        require(request.get_method() == "GET", "Unexpected HTTP operation")
        parsed = urllib.parse.urlsplit(request.full_url)
        local = f"http://127.0.0.1:{server.server_port}" + parsed.path
        remapped = urllib.request.Request(local, headers=dict(request.header_items()), method="GET")
        return original_urlopen(remapped, *arguments, **keywords)
    urllib.request.urlopen = local_urlopen
    old_argv = sys.argv
    cases = []
    start = time.perf_counter()
    plans = [
        ("default_111_success", CURRENT, "valid", None, False),
        ("explicit_old_100_success", "1.0.0", "valid", None, True),
        ("wrong_archive_sha", CURRENT, "wrong_sha", "Archive checksum mismatch", False),
        ("duplicate_checksum", CURRENT, "duplicate_checksum", "Duplicate release checksum filename", False),
        ("malformed_checksum", CURRENT, "malformed_checksum", "Malformed release checksum line", False),
        ("zip_crc_rejection", CURRENT, "bad_crc", "ZIP CRC integrity failed", False),
        ("zip_expansion_rejection", CURRENT, "oversize_header", "Unexpected archive expansion", False),
        ("zip_path_rejection", CURRENT, "traversal", "Unsafe ZIP entry", False),
        ("zip_symlink_rejection", CURRENT, "symlink", "ZIP symlink is not permitted", False),
        ("zip_case_collision_rejection", CURRENT, "case_collision", "Duplicate ZIP entry", False),
        ("manifest_version_rejection", CURRENT, "wrong_manifest_version", "package version differs", False),
        ("manifest_duplicate_rejection", CURRENT, "duplicate_manifest", "Duplicate manifest path", False),
        ("manifest_size_rejection", CURRENT, "invalid_manifest_size", "Invalid manifest byte count", False),
        ("payload_sha_rejection", CURRENT, "corrupt_payload", "Extracted file differs", False),
        ("inventory_extra_rejection", CURRENT, "extra_payload", "Undeclared or missing file", False),
        ("inventory_missing_rejection", CURRENT, "missing_payload", "Undeclared or missing file", False),
    ]
    try:
        for name, version, variant, expected_error, explicit_version in plans:
            destination = out / name
            archive_name = f"rdia-rolling-storage-v{version}.zip"
            body = fixture(version, variant)
            checksum = digest(body) if variant != "wrong_sha" else "0" * 64
            checks = f"{checksum}  {archive_name}\n"
            if variant == "duplicate_checksum":
                checks += f"{checksum}  {archive_name}\n"
            elif variant == "malformed_checksum":
                checks = "NOT_A_CHECKSUM\n" + checks
            path = urllib.parse.urlsplit(PREFIX + f"v{version}/").path
            state["routes"] = {path + "SHA256SUMS": checks.encode(), path + archive_name: body}
            state["requests"] = []
            sys.argv = [str(source), "--output-root", str(destination)]
            if explicit_version:
                sys.argv += ["--version", version]
            stdout, stderr = io.StringIO(), io.StringIO()
            failure = None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    module.main()
                except Exception as error:
                    failure = error
            record = {"case": name, "fixture": "SYNTHETIC_LOCAL_HTTP_ONLY", "requests": list(state["requests"]),
                      "fixture_archive_sha256": digest(body), "archive_bytes": len(body)}
            require(all(r["user_agent"] == UA for r in state["requests"]), "Unexpected User-Agent")
            require(state["requests"][0]["path"] == path + "SHA256SUMS", "Wrong version/checksum request")
            if expected_error is None:
                require(failure is None, f"Successful fixture failed: {failure!r}")
                proof_path = destination / "download_verification.json"
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                require(proof["status"] == "PASS_COMPLETE_PUBLISHED_ARCHIVE_VERIFIED", "Missing success receipt")
                require(proof["version"] == version and proof["archive"] == archive_name, "Wrong requested version")
                require(proof["archive_sha256"] == digest(body), "Downloaded bytes differ")
                require(proof["extracted_files_verified"] == 3, "Wrong verified payload count")
                manifest = json.loads((destination / "package/release_manifest.json").read_text())
                actual = {p.relative_to(destination / "package").as_posix() for p in (destination / "package").rglob("*") if p.is_file()}
                require(actual == {r["path"] for r in manifest["files"]} | {"release_manifest.json"}, "Incomplete/excess extracted file set")
                for row in manifest["files"]:
                    supplied = (destination / "package" / row["path"]).read_bytes()
                    require(len(supplied) == row["bytes"] and digest(supplied) == row["sha256"], "Independent payload verification failed")
                require(len(state["requests"]) == 2, "Successful release did not download both assets")
                record.update(status="PASS_ACTUAL_DOWNLOAD_EXTRACT_COMPLETE_MANIFEST", downloaded_version=version,
                              verified_payload_files=3, receipt_sha256=digest(proof_path.read_bytes()))
            else:
                require(isinstance(failure, RuntimeError) and expected_error in str(failure),
                        f"Expected {expected_error!r}, observed {failure!r}")
                require(not (destination / "download_verification.json").exists(), "Failure emitted a PASS receipt")
                pre_extraction = variant in {"wrong_sha", "duplicate_checksum", "malformed_checksum", "bad_crc",
                                            "oversize_header", "traversal", "symlink", "case_collision"}
                if pre_extraction:
                    require(not (destination / "package").exists(), "Rejected archive was extracted")
                require(not (destination / "escape.txt").exists(), "Traversal escaped the package")
                record.update(status="PASS_EXPECTED_REJECTION", observed_error=str(failure),
                              extraction_attempted=not pre_extraction)
            (destination / "fixture_stdout.log").write_text(stdout.getvalue(), encoding="utf-8")
            (destination / "fixture_stderr.log").write_text(stderr.getvalue(), encoding="utf-8")
            cases.append(record)
            print(f"PASS {name}", flush=True)
        # A streaming size bound is an actual fetch check, not a parser-only check.
        path = urllib.parse.urlsplit(PREFIX + f"v{CURRENT}/oversize.bin").path
        state["routes"] = {path: b"0123456789abcdef"}
        state["requests"] = []
        destination = out / "stream_size_rejection"
        destination.mkdir()
        failure = None
        try:
            module.fetch(PREFIX + f"v{CURRENT}/oversize.bin", destination / "partial.bin", 8)
        except RuntimeError as error:
            failure = error
        require(failure is not None and "declared safety limit" in str(failure), "Streaming limit was not enforced")
        require((destination / "partial.bin").stat().st_size <= 8, "Oversize bytes were persisted")
        require(len(state["requests"]) == 1 and state["requests"][0]["user_agent"] == UA, "Streaming check bypassed HTTP")
        cases.append({"case": "stream_size_rejection", "status": "PASS_EXPECTED_REJECTION", "requests": list(state["requests"]), "observed_error": str(failure)})
        print("PASS stream_size_rejection", flush=True)
        # Existing output must survive without even requesting the checksum asset.
        destination = out / "existing_output_rejection"
        destination.mkdir()
        sentinel = destination / "preserve.txt"
        sentinel.write_bytes(b"PRESERVE_EXISTING_BYTES")
        state["requests"] = []
        sys.argv = [str(source), "--output-root", str(destination)]
        code = None
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                module.main()
            except SystemExit as error:
                code = error.code
        require(code == 2 and not state["requests"] and sentinel.read_bytes() == b"PRESERVE_EXISTING_BYTES", "Existing output was altered or fetched")
        cases.append({"case": "existing_output_rejection", "status": "PASS_EXPECTED_REJECTION", "exit_code": code, "http_requests": 0, "existing_bytes_preserved": True})
        print("PASS existing_output_rejection", flush=True)
        require(digest(source.read_bytes()) == source_before, "Downloader source changed during tests")
        receipt = {"status": "PASS_REAL_LOCAL_HTTP_DOWNLOADER_REGRESSION", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "cases": cases, "case_count": len(cases), "successful_complete_downloads": 2,
                   "default_version": CURRENT, "user_agent": UA, "downloader_sha256": source_before,
                   "test_script_sha256": digest(Path(__file__).read_bytes()), "downloader_source_unchanged": True,
                   "elapsed_seconds": time.perf_counter() - start, "github_network_requests": 0,
                   "scope": "Actual local HTTP transport, archive byte download, checksum/CRC/path/size validation, extraction and exact manifest inventory; labelled synthetic fixtures only. No scientific replay or actual published GitHub asset download."}
        (out / "download_fixture_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: receipt[k] for k in ["status", "case_count", "successful_complete_downloads", "downloader_sha256", "elapsed_seconds"]}, indent=2), flush=True)
        return 0
    finally:
        sys.argv = old_argv
        urllib.request.urlopen = original_urlopen
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
