"""Ontario HOEP transfer check: frozen public prices, two disclosed vintage arms.

This is a retrospective protocol check, not a trading-product validation.
Raw source files are immutable on F; all derived artifacts are on D. The
operator arm has nominal lead-derived issue times, never claimed receipt times.
The researcher arm has real hourly settlement labels and generated forecasts.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_v8/transfer"
RAW = None  # public_ontario.py requires an explicit raw source directory
THIS = Path(__file__).resolve()
POLICIES = ["raw", "current_shrink_a025", "current_shrink_a050",
            "current_smooth_a025", "current_smooth_a050", "sparse_equal", "inverse_lead"]
CURRENT = POLICIES[:5]
ARMS = ["operator_short", "researcher_12h"]
SOURCES = [
    {"year": 2022, "url": "https://reports-public.ieso.ca/public/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_2022_v396.csv", "version": "2022_v396", "listed_published_est": "2023-01-31 08:03:00"},
    {"year": 2023, "url": "https://reports-public.ieso.ca/public/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_2023_v395.csv", "version": "2023_v395", "listed_published_est": "2024-01-31 08:03:00"},
    {"year": 2024, "url": "https://reports-public.ieso.ca/public/PriceHOEPPredispOR/PUB_PriceHOEPPredispOR_2024_v395.csv", "version": "2024_v395", "listed_published_est": "2025-01-31 08:03:00"},
]
ATTRIBUTION = ("Copyright © 2004-2022 Independent Electricity System Operator, all rights reserved. "
               "This information is subject to the Terms of Use set out in the IESO’s website (www.ieso.ca)")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def controller_sources():
    return {name: sha(ROOT / "scripts" / name) for name in (
        "revision_v5_control_solver.py", "revision_v5_control_safe_policy.py",
        "revision_v5_control_scaled_policy.py", "revision_v5_control_enumerated_policy.py")}


def design():
    return {
        "status": "frozen before data download, forecast fitting, or any transfer replay",
        "frozen_utc": now(), "market": "Ontario IESO legacy HOEP", "currency": "CAD",
        "price_units": "CAD per MWh", "sampling": "native hourly; no duplication into half-hours",
        "report_clock": "fixed EST UTC-05:00 year-round; Date and Hour are hour-ending labels",
        "market_scope": "province-wide legacy hourly energy-price proxy; excludes uplift, global adjustment, ancillary services, network charges and market impact",
        "sources": SOURCES,
        "source_attribution": ATTRIBUTION,
        "licence": "IESO Terms of Use, last updated 2022-07-25: limited use/reproduction with attribution, not CC; raw bytes not relicensed or redistributed by this package",
        "primary_documentation": {
            "directory": "https://reports-public.ieso.ca/public/PriceHOEPPredispOR/",
            "unit_and_forecast_columns": "https://www.ieso.ca/power-data/data-directory",
            "licence": "https://www.ieso.ca/terms-of-use",
            "fixed_EST": "https://www.ieso.ca/-/media/Files/IESO/Document-Library/engage/ca/ca-Overview-2020-Training-and-QA.ashx",
            "legacy_hour_ahead_context": "https://www.ieso.ca/-/media/Files/IESO/Document-Library/market-renewal/MRP-Detailed-Design-Real-Time-Calculation-Engine.pdf",
        },
        "train": {"from_est": "2022-01-01 00:00:00", "to_exclusive_est": "2023-01-01 00:00:00", "purge": "all eighteen direct targets remain in 2022; common-origin training support"},
        "selection": {"from_est": "2023-10-01 00:00:00", "to_exclusive_est": "2024-01-01 00:00:00", "operator_end_purge_hours": 2,
            "candidates": CURRENT, "criterion": "maximum continuous-path net cash plus common observed endpoint inventory mark", "tie_rule": "raw first, then frozen list order", "no_evaluation_selection": True,
            "selection_source_published_before_evaluation": "2023_v395 listed 2024-01-31 08:03 EST; evaluation starts 2024-02-01"},
        "evaluation": {"from_est": "2024-02-01 00:00:00", "to_exclusive_est": "2024-08-01 00:00:00", "days": 182, "hours": 4368, "all_policies_reported": POLICIES},
        "controller": {"power_mw": 1.0, "nominal_energy_mwh": 2.0, "soc_bounds_mwh": [0.2, 1.8], "initial_soc_mwh": 1.0,
            "eta_c": 0.91, "eta_d": 0.91, "dt_hours": 1.0, "kappa_cad_per_grid_mwh": 5.0,
            "terminal_factor": 1.0, "primary_tolerance_cad": 1e-5, "throughput_tolerance_mwh": 1e-7,
            "kappa_scope": "local nominal cost assumption, not an FX-equivalent AUD cost estimate",
            "policy": "same three tolerance-ordered cold objectives and physical mode checks as NEM; equivalent original-unit fallback solvers", "source_sha256": controller_sources()},
        "arms": {
            "operator_short": {"forecast_source": "actual IESO Hour 1/2/3 Predispatch columns, never HOEP-as-forecast", "forecast_steps": 3, "trade_steps": 2, "terminal_steps": 1,
                "nominal_issue_rule": "target delivery interval start minus stated lead in hours; reconstructed from lead labels, not an observed run/receipt clock",
                "decision_cutoff": "nominal one-hour-ahead issue; execution at next hourly interval start; actual availability within that hour is assumed, not timestamp-verified",
                "raw": "for target steps j=0,1,2 use lead columns j+1; nominal same issue",
                "sparse_equal": "equal mean of all eligible lead columns for each delivery target; counts 3,2,1; cannot reproduce five-version NEM sparse support",
                "inverse_lead": "same eligible support weighted by 1/(nominal target-start lead hours)",
                "receipt_limitation": "annual file has no actual publication/receipt timestamp or revision audit for each forecast entry; no guarantee of globally coherent run identity"},
            "researcher_12h": {"forecast_source": "researcher-generated deterministic causal direct Ridge vintages; operator columns never used", "forecast_steps": 12, "trade_steps": 8, "terminal_steps": 4,
                "issued_every_hours": 1, "issue": "first execution interval start minus 1 hour",
                "actual_price_feature_delay_hours": 24, "feature_latest_price": "latest observed price interval ends at issue minus 24h (index origin-26)",
                "model": "18 separate direct Ridge(alpha=100, intercept=True) models; StandardScaler fit on 2022 common training origins only; frozen without retuning/refitting",
                "features": "origin price lags at latest, latest-24, latest-48, latest-168; preceding 24h and168h means/std; target-aligned prices at target-48 and target-168; known target hour(24), weekday(7), month(12) one-hot plus annual sine/cosine",
                "history_support": "seven issued hourly vintages ages 0..6h; forecast lead indexed target-origin+age; all target-aligned and issued no later than cutoff",
                "sparse_equal": "equal mean of ages 0,1,2,4,6h (five versions)", "inverse_lead": "all seven ages 0..6h weighted by 1/(j+1+age)",
                "data_asof_limitation": "final historical HOEP values are assumed available after24h; source has no historical actual-price revision/receipt archive; causal indexing does not prove participant real-time availability",
                "horizon_scope": "same step counts as NEM, but12/8 steps are12/8 hours, not6/4 hours"}},
        "current_family": {"shrink": "(1-alpha) current + alpha mean(current), alpha=.25 or.50", "smooth": "(1-alpha) current + alpha three-neighbor mean, endpoints two-neighbor mean, alpha=.25 or.50", "price_clip": "none"},
        "outcomes": {"net_value": "sum [actual*(d-c)*1h -5*(c+d)*1h] + last_actual*last_soc - first_actual*initial_soc",
            "contrasts": ["history-raw", "history-selected-current", "selected-current-raw"], "absorption": "(selected-current-raw)/(history-raw); informative positive-gain interpretation only if history-raw positive and denominator CI excludes0"},
        "uncertainty": {"method": "paired circular moving-block bootstrap of daily realized cash differences, endpoint contrast held fixed; conditional on continuous trajectories and frozen prior choice, not a replay bootstrap or selection-adjusted confidence guarantee",
            "blocks_days": [7, 14, 28], "primary_block_days": 14, "replicates": 10000, "seed": 20261001,
            "ratio": "do not provide a bounded ratio CI when denominator percentile CI includes0; retain point ratio with an explicit nonidentified flag"},
        "failure_reporting": "report all six calendar months of cash contrasts, count negative/zero days and all current candidate payoffs; never select arm/period by results",
        "resources": {"max_workers": 2, "estimated_raw_bytes": 1500000, "disk_free_bytes": {"raw_directory_parent": shutil.disk_usage(RAW.parent).free, "output_directory": shutil.disk_usage(OUT).free}},
        "driver_sha256": sha(THIS), "claims": "single Ontario pre-renewal period verification of protocol portability; neither universal market generalization nor new ML/solver superiority",
    }


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "design.json"
    if target.exists():
        doc = read_json(target)
        if doc["driver_sha256"] != sha(THIS) or doc["controller"]["source_sha256"] != controller_sources():
            raise RuntimeError("Frozen source mismatch: record an explicit amendment rather than overwrite design")
    else:
        write_json(target, design())
    print(json.dumps({"task": "freeze", "path": str(target), "sha256": sha(target)}), flush=True)


def require_frozen():
    doc = read_json(OUT / "design.json")
    allowed = doc["driver_sha256"] == sha(THIS)
    amendment_path = OUT / "design_amendment_01.json"
    if not allowed and amendment_path.exists():
        amendment = read_json(amendment_path)
        allowed = amendment["amended_driver_sha256"] == sha(THIS) and amendment["original_design_sha256"] == sha(OUT / "design.json")
    if not allowed or doc["controller"]["source_sha256"] != controller_sources():
        raise RuntimeError("Source changed after freeze")
    return doc


def amend():
    target = OUT / "design_amendment_01.json"
    if target.exists():
        if read_json(target)["amended_driver_sha256"] != sha(THIS):
            raise RuntimeError("Existing amendment source mismatch")
        return
    if any((OUT / name).exists() for name in ("prepare_receipt.json", "smoke_receipt.json", "execution_receipt.json")):
        raise RuntimeError("This amendment must precede model fitting/decision results")
    write_json(target, {"status": "frozen data-quality amendment before fitting or outcome computation", "amended_utc": now(),
        "original_design_sha256": sha(OUT / "design.json"), "amended_driver_sha256": sha(THIS),
        "reason": "Actual official CSV preflight found complete HOEP but missing predispatch entries; each lead column has16/31/17 missing cells in2022/23/24 respectively",
        "researcher": "require only finite HOEP for training/features; unused operator columns may be missing",
        "operator_common_guard": "all seven policies idle at every hour where the current three-step curve is incomplete; retain hour, zero flows, carry SOC continuously; no actual-price imputation, no missing-hour deletion",
        "operator_history": "eligible history means/weights renormalize over finite official entries; raw current curve must be complete for any policy to trade",
        "outputs": "record common guard and solver-executed flag; numerical loss fields absent (NaN in trajectory) at no-solve guard hours",
        "selection_rule_and_periods_unchanged": True, "all_arms_and_outcomes_retained": True})
    print(json.dumps({"task": "amend", "sha256": sha(target)}), flush=True)


def download():
    require_frozen()
    target = OUT / "source_manifest.json"
    if target.exists():
        manifest = read_json(target)
        for source in manifest["files"]:
            if sha(source["path"]) != source["sha256"]:
                raise RuntimeError("Immutable raw checksum mismatch")
        return manifest
    if shutil.disk_usage(RAW.parent).free < 10_000_000 or shutil.disk_usage(OUT).free < 500_000_000:
        raise RuntimeError("Insufficient F raw/D computation space")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    rawdir = RAW / stamp
    rawdir.mkdir(parents=True, exist_ok=False)
    files = []
    for source in SOURCES:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "Academic-HOEP-protocol-validation/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(2_000_001)
            headers = dict(response.headers.items())
            status = response.status
            final_url = response.url
        if len(data) > 2_000_000 or b"Date,Hour,HOEP,Hour 1 Predispatch" not in data[:2000]:
            raise RuntimeError("Unexpected source payload")
        path = rawdir / Path(source["url"]).name
        with path.open("xb") as handle:
            handle.write(data)
        os.chmod(path, 0o444)
        files.append({**source, "path": str(path), "bytes": len(data), "sha256": sha(path),
                      "fetched_utc": now(), "http_status": status, "final_url": final_url, "http_headers": headers})
    attribution_path = rawdir / "SOURCE_ATTRIBUTION.txt"
    with attribution_path.open("x", encoding="utf-8") as handle:
        handle.write(ATTRIBUTION + "\nSource terms: https://www.ieso.ca/terms-of-use\nNo endorsement by IESO is implied.\n")
    os.chmod(attribution_path, 0o444)
    manifest = {"status": "PASS", "downloaded_utc": now(), "design_sha256": sha(OUT / "design.json"),
                "files": files, "attribution": ATTRIBUTION, "attribution_path": str(attribution_path), "attribution_sha256": sha(attribution_path),
                "reuse_licence": "limited licence from IESO Terms; no CC licence inferred; reproduction bundle uses downloader and checksums, not redistributed raw bytes"}
    write_json(target, manifest)
    print(json.dumps({"task": "download", "files": len(files), "total_bytes": sum(x["bytes"] for x in files), "rawdir": str(rawdir)}), flush=True)
    return manifest


def load_prices():
    manifest = read_json(OUT / "source_manifest.json")
    pieces, audits = [], []
    for source in manifest["files"]:
        if sha(source["path"]) != source["sha256"]:
            raise RuntimeError("Raw data changed")
        data = Path(source["path"]).read_bytes()
        start = data.index(b"Date,Hour,HOEP,")
        frame = pd.read_csv(io.BytesIO(data[start:]))
        frame = frame.loc[frame["Date"].notna()].copy()
        frame["Hour"] = pd.to_numeric(frame["Hour"], errors="raise").astype(int)
        for column in ["HOEP", "Hour 1 Predispatch", "Hour 2 Predispatch", "Hour 3 Predispatch"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame["Hour"].between(1, 24).all():
            raise RuntimeError("Unexpected report hour")
        frame["interval_start_est"] = pd.to_datetime(frame["Date"]) + pd.to_timedelta(frame["Hour"]-1, unit="h")
        frame["interval_end_est"] = frame["interval_start_est"] + pd.Timedelta(hours=1)
        frame["interval_start_utc"] = frame["interval_start_est"].dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC")
        frame = frame.sort_values("interval_start_est").reset_index(drop=True)
        year = source["year"]
        expected = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left")
        if not np.array_equal(frame.interval_start_est.to_numpy(), expected.to_numpy()):
            raise RuntimeError(f"Missing/duplicated/reordered hourly support {year}")
        if not np.isfinite(frame.HOEP.to_numpy()).all():
            raise RuntimeError(f"Nonfinite settlement value {year}")
        audits.append({"year": year, "rows": len(frame), "days": len(expected)/24, "missing_hours": 0, "duplicate_hours": 0,
                       "min_hoep_cad_mwh": float(frame.HOEP.min()), "max_hoep_cad_mwh": float(frame.HOEP.max()), "source_sha256": source["sha256"],
                       "missing_predispatch_cells": {f"lead{lead}": int(frame[f"Hour {lead} Predispatch"].isna().sum()) for lead in (1,2,3)}})
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True)
    if not np.all(np.diff(frame.interval_start_est.to_numpy()) == np.timedelta64(1, "h")):
        raise RuntimeError("Non-contiguous annual join")
    return frame, audits


def features(prices, times, origins, horizon):
    origins = np.asarray(origins, dtype=int)
    latest = origins - 26
    target = origins + int(horizon)
    if latest.min()-168 < 0 or target.max() >= len(prices):
        raise ValueError("Feature support unavailable")
    columns = [prices[latest-k] for k in (0, 24, 48, 168)]
    for width in (24, 168):
        samples = prices[latest[:, None] - np.arange(width)[None, :]]
        columns.extend([samples.mean(axis=1), samples.std(axis=1)])
    for lag in (48, 168):
        index = target-lag
        if np.any(index > latest):
            raise RuntimeError("Target-aligned feature after asof cutoff")
        columns.append(prices[index])
    calendar = pd.DatetimeIndex(times[target])
    for variable, levels in ((calendar.hour, 24), (calendar.dayofweek, 7), (calendar.month-1, 12)):
        columns.extend((np.asarray(variable) == level).astype(float) for level in range(levels))
    fraction = (np.asarray(calendar.dayofyear)-1)/365.25
    columns.extend([np.sin(2*np.pi*fraction), np.cos(2*np.pi*fraction)])
    return np.column_stack(columns)


def current_transform(raw, policy):
    if policy == "raw":
        return raw.copy()
    alpha = .25 if policy.endswith("a025") else .5
    if "shrink" in policy:
        smooth = raw.mean(axis=1, keepdims=True)
    else:
        smooth = raw.copy()
        smooth[:, 1:-1] = (raw[:, :-2]+raw[:, 1:-1]+raw[:, 2:])/3
        smooth[:, 0] = (raw[:, 0]+raw[:, 1])/2
        smooth[:, -1] = (raw[:, -2]+raw[:, -1])/2
    return (1-alpha)*raw+alpha*smooth


def prepare():
    require_frozen()
    frame, audits = load_prices()
    prices = frame.HOEP.to_numpy(float)
    times = frame.interval_start_est.to_numpy()
    train_last = int(np.flatnonzero(times < np.datetime64("2023-01-01"))[-1])
    training = np.arange(194, train_last-17+1)
    selection = np.flatnonzero((times >= np.datetime64("2023-10-01")) & (times < np.datetime64("2024-01-01")))
    evaluation = np.flatnonzero((times >= np.datetime64("2024-02-01")) & (times < np.datetime64("2024-08-01")))
    all_origins = np.r_[selection, evaluation]
    vintage_origins = np.unique(np.concatenate([all_origins-age for age in range(7)]))
    coefficients, intercepts, means, scales, models = [], [], [], [], []
    predictions = np.empty((len(vintage_origins), 18), dtype=float)
    for horizon in range(18):
        x = features(prices, times, training, horizon)
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=100.0, fit_intercept=True, solver="cholesky").fit(scaler.transform(x), prices[training+horizon])
        predictions[:, horizon] = model.predict(scaler.transform(features(prices, times, vintage_origins, horizon)))
        coefficients.append(model.coef_); intercepts.append(model.intercept_); means.append(scaler.mean_); scales.append(scaler.scale_)
        models.append({"target_step": horizon, "target_start_lead_hours": horizon+1, "train_rows": len(training),
                       "last_train_target_end_est": str(pd.Timestamp(times[training[-1]+horizon])+pd.Timedelta(hours=1))})
    np.savez_compressed(OUT / "ridge_models.npz", coefficient=np.array(coefficients), intercept=np.array(intercepts),
                        scaler_mean=np.array(means), scaler_scale=np.array(scales), train_origins=training)
    np.savez_compressed(OUT / "generated_vintages.npz", origins=vintage_origins, predictions=predictions)
    lookup = {int(origin): row for row, origin in enumerate(vintage_origins)}
    cube = np.empty((len(all_origins), 7, 12), dtype=float)
    for age in range(7):
        rows = np.array([lookup[int(origin-age)] for origin in all_origins])
        cube[:, age, :] = predictions[rows[:, None], np.arange(12)[None, :]+age]
    raw = cube[:, 0, :]
    sparse = cube[:, [0, 1, 2, 4, 6], :].mean(axis=1)
    weights = 1/(np.arange(12)[None, :]+np.arange(7)[:, None]+1)
    inverse = (cube*weights[None, :, :]).sum(axis=1)/weights.sum(axis=0)[None, :]
    arms = {"researcher_12h": (all_origins, raw, sparse, inverse)}
    op_origins = np.r_[selection[:-2], evaluation]
    op = frame[["Hour 1 Predispatch", "Hour 2 Predispatch", "Hour 3 Predispatch"]].to_numpy(float)
    opraw, opequal, opinverse = [], [], []
    for step in range(3):
        available = op[op_origins+step, step:]
        leadweights = 1/np.arange(step+1, 4)
        finite = np.isfinite(available)
        counts = finite.sum(axis=1)
        mean = np.divide(np.where(finite, available, 0.).sum(axis=1), counts, out=np.full(len(counts), np.nan), where=counts>0)
        weight_sum = (finite*leadweights[None, :]).sum(axis=1)
        weighted = np.divide((np.where(finite, available, 0.)*leadweights[None, :]).sum(axis=1), weight_sum, out=np.full(len(counts), np.nan), where=weight_sum>0)
        opraw.append(available[:, 0]); opequal.append(mean); opinverse.append(weighted)
    arms["operator_short"] = (op_origins, np.column_stack(opraw), np.column_stack(opequal), np.column_stack(opinverse))
    prepared = []
    for arm, (origins, raw, sparse, inverse) in arms.items():
        output = frame.loc[origins, ["interval_start_est", "interval_end_est", "interval_start_utc", "HOEP"]].reset_index(drop=True)
        output.rename(columns={"HOEP": "actual_cad_mwh"}, inplace=True)
        output["origin_index"] = origins
        output["nominal_issue_est"] = output.interval_start_est-pd.Timedelta(hours=1)
        output["phase"] = np.where(output.interval_start_est < pd.Timestamp("2024-01-01"), "selection", "evaluation")
        output["common_current_complete"] = np.isfinite(raw).all(axis=1)
        if arm == "researcher_12h":
            output["latest_feature_price_end_est"] = pd.to_datetime(times[origins-26])+pd.Timedelta(hours=1)
            if not (output.latest_feature_price_end_est <= output.nominal_issue_est-pd.Timedelta(hours=24)).all():
                raise RuntimeError("Price feature availability failure")
        paths = {"raw": raw, "sparse_equal": sparse, "inverse_lead": inverse}
        paths.update({policy: current_transform(raw, policy) for policy in CURRENT[1:]})
        for policy in POLICIES:
            for step in range(raw.shape[1]):
                output[f"{policy}_{step:02d}"] = paths[policy][:, step]
        output.to_parquet(OUT / f"{arm}_inputs.parquet", index=False, compression="zstd")
        prepared.append({"arm": arm, "selection_origins": int(output.phase.eq("selection").sum()), "evaluation_origins": int(output.phase.eq("evaluation").sum()),
                         "forecast_steps": raw.shape[1], "input_sha256": sha(OUT / f"{arm}_inputs.parquet"), "terminal_history_vs_raw_max": float(np.nanmax(abs(sparse[:, -1]-raw[:, -1]))) if arm == "operator_short" else None,
                         "common_idle_guard_by_phase": {phase: int((output.phase.eq(phase)&~output.common_current_complete).sum()) for phase in ("selection", "evaluation")}})
    # Actual execution audit: changing not-yet-eligible price values leaves every
    # generated current-issue feature and prediction unchanged (no refitting).
    leak_checks = []
    for origin in evaluation[np.linspace(0, len(evaluation)-1, 24, dtype=int)]:
        modified = prices.copy(); modified[origin-25:] += 98765.4321
        for horizon in range(18):
            a = features(prices, times, [origin], horizon)
            b = features(modified, times, [origin], horizon)
            leak_checks.append(float(np.max(abs(a-b))))
    if max(leak_checks) != 0:
        raise RuntimeError("Future-value perturbation changed asof features")
    # Target alignment audit independently reconstructs all eligible prices.
    reconstruction = []
    for row in np.linspace(0, len(all_origins)-1, 50, dtype=int):
        origin = all_origins[row]
        for age in range(7):
            for step in range(12):
                reconstruction.append(abs(cube[row, age, step]-predictions[lookup[int(origin-age)], step+age]))
    receipt = {"status": "PASS", "prepared_utc": now(), "design_sha256": sha(OUT / "design.json"), "source_manifest_sha256": sha(OUT / "source_manifest.json"),
               "driver_sha256": sha(THIS), "data_audit": audits, "arms": prepared, "models": models,
               "training_origin_first_est": str(pd.Timestamp(times[training[0]])), "training_origin_last_est": str(pd.Timestamp(times[training[-1]])),
               "train_target_end_latest_est": str(pd.Timestamp(times[training[-1]+17])+pd.Timedelta(hours=1)),
               "future_perturbation_checks": len(leak_checks), "future_feature_max_abs_change": max(leak_checks),
               "target_alignment_cells_checked": len(reconstruction), "target_alignment_max_error": max(reconstruction),
               "generated_model_sha256": sha(OUT / "ridge_models.npz"), "generated_vintages_sha256": sha(OUT / "generated_vintages.npz"),
               "no_operator_forecast_columns_in_researcher_features": True,
               "operator_clock_status": "nominal lead-derived only; actual issue, receipt and revision clocks not verified",
               "amendment_sha256": sha(OUT / "design_amendment_01.json")}
    write_json(OUT / "prepare_receipt.json", receipt)
    print(json.dumps({"task": "prepare", "status": "PASS", "arms": prepared, "future_feature_change": max(leak_checks)}), flush=True)
    return receipt


def make_controller(arm, binary=False):
    for name in ("revision_v5_control_solver", "revision_v5_control_safe_policy", "revision_v5_control_scaled_policy", "revision_v5_control_enumerated_policy"):
        module = __import__(name)
        module.OUT = OUT / "numerical_fallback"
    from revision_v5_control_enumerated_policy import LexMPC
    return LexMPC(h=2 if arm == "operator_short" else 8, dt=1.0, eta=.91, kappa=5.0,
                  power=1.0, smin=.2, smax=1.8, primary_tolerance_aud=1e-5,
                  throughput_tolerance_mwh=1e-7, relax_if_mode_feasible=not binary)


def replay(arm, phase, policy, limit=None):
    require_frozen()
    tic = time.time()
    inputs = OUT / f"{arm}_inputs.parquet"
    frame = pd.read_parquet(inputs)
    frame = frame.loc[frame.phase.eq(phase)].sort_values("interval_start_est").reset_index(drop=True)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    folder = OUT / ("smoke" if limit is not None else "trajectories") / arm / phase
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / f"{policy}.json"
    trajectory_path = folder / f"{policy}.parquet"
    if report_path.exists():
        report = read_json(report_path)
        if report["driver_sha256"] != sha(THIS) or report["input_sha256"] != sha(inputs) or sha(trajectory_path) != report["trajectory_sha256"]:
            raise RuntimeError("Cannot reuse result with different sources")
        return report
    length = 3 if arm == "operator_short" else 12
    paths = frame[[f"{policy}_{step:02d}" for step in range(length)]].to_numpy(float)
    actual = frame.actual_cad_mwh.to_numpy(float)
    controller = make_controller(arm)
    records, soc = [], 1.0
    maxima = {"balance_error_mwh": 0., "state_continuity_error_mwh": 0., "soc_bound_violation_mwh": 0.,
              "power_bound_violation_mw": 0., "simultaneous_charge_discharge_mw": 0., "primary_loss_upper_cad": 0., "primary_algorithmic_gap_cad": 0.}
    for row, path in enumerate(paths):
        if not frame.common_current_complete.iloc[row]:
            if arm != "operator_short":
                raise RuntimeError("Unexpected missing generated curve")
            records.append((soc, 0., 0., soc, 0., 0., 0., np.nan, np.nan))
            continue
        solution = controller.solve(path, soc, cold=True)
        c, d, ns = solution["c"], solution["d"], solution["soc"]
        balance = abs(ns-(soc+.91*c-d/.91))
        maxima["balance_error_mwh"] = max(maxima["balance_error_mwh"], balance)
        maxima["soc_bound_violation_mwh"] = max(maxima["soc_bound_violation_mwh"], .2-ns, ns-1.8, 0.)
        maxima["power_bound_violation_mw"] = max(maxima["power_bound_violation_mw"], -c, -d, c-1, d-1, 0.)
        maxima["simultaneous_charge_discharge_mw"] = max(maxima["simultaneous_charge_discharge_mw"], min(c, d))
        maxima["primary_loss_upper_cad"] = max(maxima["primary_loss_upper_cad"], solution["policy_optimum_loss_upper_aud"])
        maxima["primary_algorithmic_gap_cad"] = max(maxima["primary_algorithmic_gap_cad"], solution["primary_objective_gap_aud"])
        cash = actual[row]*(d-c)
        wear = 5*(c+d)
        records.append((soc, c, d, ns, cash, wear, cash-wear, solution["policy_optimum_loss_upper_aud"], solution["primary_objective_gap_aud"]))
        soc = ns
        if (row+1) % 512 == 0:
            print(json.dumps({"arm": arm, "phase": phase, "policy": policy, "done": row+1, "total": len(paths)}), flush=True)
    trajectory = frame[["interval_start_est", "interval_end_est", "interval_start_utc", "nominal_issue_est", "actual_cad_mwh"]].copy()
    fields = ["soc_start_mwh", "charge_mw", "discharge_mw", "soc_end_mwh", "cashflow_cad", "degradation_cad", "net_cash_cad", "primary_loss_upper_cad", "primary_gap_cad"]
    trajectory[fields] = np.asarray(records)
    trajectory["solver_executed"] = frame.common_current_complete.to_numpy(bool)
    maxima["state_continuity_error_mwh"] = float(np.max(abs(trajectory.soc_start_mwh.to_numpy()[1:]-trajectory.soc_end_mwh.to_numpy()[:-1]))) if len(trajectory)>1 else 0.
    if maxima["balance_error_mwh"] > 1e-8 or maxima["state_continuity_error_mwh"] > 1e-10 or maxima["soc_bound_violation_mwh"] > 1e-8 or maxima["simultaneous_charge_discharge_mw"] > 1e-6 or maxima["primary_loss_upper_cad"] > 1.21e-5:
        raise RuntimeError(f"Trajectory audit failure: {maxima}")
    cash = float(trajectory.net_cash_cad.sum())
    mark = float(actual[-1]*soc-actual[0])
    trajectory.to_parquet(trajectory_path, index=False, compression="zstd")
    days = len(trajectory)/24
    report = {"status": "PASS", "arm": arm, "phase": phase, "policy": policy, "origins": len(trajectory), "days": days,
              "net_cash_cad": cash, "endpoint_mark_cad": mark, "net_value_cad": cash+mark, "net_value_cad_per_day": (cash+mark)/days,
              "initial_soc_mwh": 1., "final_soc_mwh": soc, "seconds": time.time()-tic, "completed_utc": now(),
              "controller": controller.definition(), "audit": maxima, "driver_sha256": sha(THIS), "design_sha256": sha(OUT / "design.json"),
              "input_sha256": sha(inputs), "trajectory_path": str(trajectory_path), "trajectory_sha256": sha(trajectory_path),
              "enumerated_fallback_count": controller.enumerated_fallback_count,
              "common_idle_guard_hours": int((~frame.common_current_complete).sum()), "solver_executed_hours": int(frame.common_current_complete.sum()),
              "scope": "continuous simulated dispatch settled against observed HOEP; CAD proxy economics excludes market-specific charges"}
    write_json(report_path, report)
    print(json.dumps({"task": "replay_complete", "arm": arm, "phase": phase, "policy": policy, "origins": len(frame), "net_value_cad": cash+mark, "seconds": report["seconds"], "audit": maxima}), flush=True)
    return report


def smoke():
    reports, checks = [], []
    for arm in ARMS:
        report = replay(arm, "evaluation", "raw", 128)
        frame = pd.read_parquet(OUT / f"{arm}_inputs.parquet").loc[lambda x: x.phase.eq("evaluation")].iloc[:128]
        trajectory = pd.read_parquet(report["trajectory_path"])
        length = 3 if arm == "operator_short" else 12
        controller = make_controller(arm)
        binary = make_controller(arm, binary=True)
        for row in range(127, -1, -1):
            if not frame.common_current_complete.iloc[row]:
                continue
            path = frame.iloc[row][[f"raw_{j:02d}" for j in range(length)]].to_numpy(float)
            soc = trajectory.iloc[row].soc_start_mwh
            solution = controller.solve(path, soc)
            chosen = trajectory.iloc[row]
            checks.append(max(abs(solution["c"]-chosen.charge_mw), abs(solution["d"]-chosen.discharge_mw), abs(solution["soc"]-chosen.soc_end_mwh)))
            if row % 8 == 0:
                exact = binary.solve(path, soc)
                checks.append(max(abs(exact["u"]-solution["u"]), abs(exact["value"]-solution["value"])))
        reports.append(report)
    maximum = max(checks)
    if maximum > 2e-6:
        raise RuntimeError(f"Cold reordering/binary smoke mismatch {maximum}")
    receipt = {"status": "PASS", "executed_utc": now(), "arms": reports, "origins_per_arm": 128,
               "reordered_cases": 256, "all_binary_comparison_cases": 32, "max_abs_output_difference": maximum,
               "driver_sha256": sha(THIS), "design_sha256": sha(OUT / "design.json"), "prepare_receipt_sha256": sha(OUT / "prepare_receipt.json"),
               "scope": "actual new solves from source; reversed common-state re-solves plus all-binary comparison; not full solver reproduction"}
    write_json(OUT / "smoke_receipt.json", receipt)
    print(json.dumps({"task": "smoke", "status": "PASS", "max_abs_difference": maximum}), flush=True)
    return receipt


def run(workers=2):
    require_frozen()
    if workers > 2:
        raise ValueError("Design caps concurrent workers at2")
    tic = time.time()
    jobs = [(arm, "selection", policy) for arm in ARMS for policy in CURRENT]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(replay, *job) for job in jobs]
        prior = [future.result() for future in futures]
    selections = []
    for arm in ARMS:
        candidates = [r for r in prior if r["arm"] == arm]
        selected = max(candidates, key=lambda r: (r["net_value_cad"], -CURRENT.index(r["policy"])))
        selections.append({"arm": arm, "selected_policy": selected["policy"], "selected_utc": now(), "criterion": "prior continuous net value including endpoint mark", "candidates": candidates,
                           "last_selected_delivery_est": "2023-12-31 21:00:00" if arm == "operator_short" else "2023-12-31 23:00:00", "first_evaluation_delivery_est": "2024-02-01 00:00:00"})
    write_json(OUT / "current_selection.json", {"status": "PASS", "selections": selections, "source": "strictly prior period only", "written_before_evaluation_jobs": True, "design_sha256": sha(OUT / "design.json")})
    jobs = [(arm, "evaluation", policy) for arm in ARMS for policy in POLICIES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(replay, *job) for job in jobs]
        evaluation = [future.result() for future in futures]
    receipt = {"status": "PASS", "completed_utc": now(), "seconds": time.time()-tic, "workers": workers,
               "prior_runs": len(prior), "evaluation_runs": len(evaluation), "total_origins": sum(r["origins"] for r in prior+evaluation),
               "selections": selections, "reports": evaluation, "design_sha256": sha(OUT / "design.json"), "driver_sha256": sha(THIS), "controller_sources_sha256": controller_sources(),
               "selection_receipt_sha256": sha(OUT / "current_selection.json"), "smoke_receipt_sha256": sha(OUT / "smoke_receipt.json")}
    write_json(OUT / "execution_receipt.json", receipt)
    return receipt


def bootstrap(daily, marks, seed):
    rng = np.random.default_rng(seed)
    n = len(daily)
    result = []
    for width in (7, 14, 28):
        starts = rng.integers(0, n, size=(10000, int(np.ceil(n/width))))
        indexes = (starts[:, :, None]+np.arange(width)[None, None, :]) % n
        indexes = indexes.reshape(10000, -1)[:, :n]
        sample = daily[indexes].mean(axis=1)+marks[None, :]/n
        result.append((width, sample))
    return result


def analyze():
    require_frozen()
    execution = read_json(OUT / "execution_receipt.json")
    selections = {r["arm"]: r["selected_policy"] for r in read_json(OUT / "current_selection.json")["selections"]}
    policy_rows, contrast_rows, months, boundaries, forecast_metrics = [], [], [], [], []
    for arm in ARMS:
        reports = {r["policy"]: r for r in execution["reports"] if r["arm"] == arm}
        trajectories = {}
        for policy, report in reports.items():
            if sha(report["trajectory_path"]) != report["trajectory_sha256"]:
                raise RuntimeError("Trajectory changed")
            trajectory = pd.read_parquet(report["trajectory_path"])
            expected = trajectory.actual_cad_mwh*(trajectory.discharge_mw-trajectory.charge_mw)-5*(trajectory.charge_mw+trajectory.discharge_mw)
            if np.max(abs(expected-trajectory.net_cash_cad)) > 1e-10:
                raise RuntimeError("Cash recomputation discrepancy")
            if abs(float(expected.sum())+float(trajectory.actual_cad_mwh.iloc[-1]*trajectory.soc_end_mwh.iloc[-1]-trajectory.actual_cad_mwh.iloc[0])-report["net_value_cad"]) > 1e-8:
                raise RuntimeError("Endpoint value discrepancy")
            trajectories[policy] = trajectory
            policy_rows.append({"arm": arm, "policy": policy, "selected": policy == selections[arm], "days": report["days"],
                                "net_value_cad": report["net_value_cad"], "net_value_cad_per_day": report["net_value_cad_per_day"],
                                "net_cash_cad": report["net_cash_cad"], "endpoint_mark_cad": report["endpoint_mark_cad"],
                                "active_hours": int(((trajectory.charge_mw+trajectory.discharge_mw)>1e-6).sum())})
        selected = selections[arm]
        matrix = np.column_stack([trajectories[p].net_cash_cad.to_numpy() for p in POLICIES])
        days = matrix.reshape(-1, 24, len(POLICIES)).sum(axis=1)
        marks = np.array([reports[p]["endpoint_mark_cad"] for p in POLICIES])
        sampled = bootstrap(days, marks, 20261001+ARMS.index(arm))
        raw_index, cur_index = 0, POLICIES.index(selected)
        points = days.mean(axis=0)+marks/len(days)
        for history in ("sparse_equal", "inverse_lead"):
            history_index = POLICIES.index(history)
            definitions = [("history_minus_raw", history_index, raw_index), ("history_minus_selected_current", history_index, cur_index), ("selected_current_minus_raw", cur_index, raw_index)]
            for label, first, second in definitions:
                for width, sample in sampled:
                    interval = np.quantile(sample[:, first]-sample[:, second], [.025, .975])
                    contrast_rows.append({"arm": arm, "history": history, "selected_policy": selected, "contrast": label, "block_days": width,
                                          "delta_cad_per_day": float(points[first]-points[second]), "ci_lower_cad_per_day": float(interval[0]), "ci_upper_cad_per_day": float(interval[1])})
            denominator = points[history_index]-points[raw_index]
            numerator = points[cur_index]-points[raw_index]
            ratios = []
            for width, sample in sampled:
                gain = sample[:, history_index]-sample[:, raw_index]
                ci_gain = np.quantile(gain, [.025, .975])
                valid = not (ci_gain[0] <= 0 <= ci_gain[1])
                ci = np.quantile((sample[:, cur_index]-sample[:, raw_index])/gain, [.025, .975]) if valid else [None, None]
                ratios.append({"block_days": width, "denominator_ci_cad_per_day": ci_gain.tolist(), "denominator_excludes_zero": bool(valid),
                               "absorption_ratio_ci": [float(x) if x is not None else None for x in ci],
                               "interpretation": "positive baseline history gain" if denominator > 0 and valid else "ratio not a defensible absorbed-positive-gain summary"})
            daily_gain = days[:, history_index]-days[:, raw_index]
            daily_residual = days[:, history_index]-days[:, cur_index]
            boundaries.append({"arm": arm, "history": history, "selected_policy": selected,
                               "absorption_point_ratio": float(numerator/denominator) if abs(denominator)>1e-12 else None,
                               "history_gain_cad_per_day": float(denominator), "selected_current_gain_cad_per_day": float(numerator),
                               "ci": ratios, "negative_history_raw_cash_days": int((daily_gain < -1e-6).sum()), "zero_history_raw_cash_days": int((abs(daily_gain) <= 1e-6).sum()),
                               "negative_history_selected_cash_days": int((daily_residual < -1e-6).sum()), "zero_history_selected_cash_days": int((abs(daily_residual) <= 1e-6).sum())})
            date = trajectories[history].interval_start_est
            cash = pd.DataFrame({"month": date.dt.strftime("%Y-%m"), "history_minus_raw": matrix[:, history_index]-matrix[:, raw_index],
                                 "history_minus_selected": matrix[:, history_index]-matrix[:, cur_index], "current_minus_raw": matrix[:, cur_index]-matrix[:, raw_index]})
            for month, group in cash.groupby("month", sort=True):
                duration = len(group)/24
                months.append({"arm": arm, "history": history, "month": month, "days": duration,
                               "history_minus_raw_cash_cad_per_day": float(group.history_minus_raw.sum()/duration),
                               "history_minus_selected_cash_cad_per_day": float(group.history_minus_selected.sum()/duration),
                               "current_minus_raw_cash_cad_per_day": float(group.current_minus_raw.sum()/duration),
                               "scope": "cash contributions on same continuous trajectory; endpoint inventory marks excluded; no monthly SOC reset"})
        inputs = pd.read_parquet(OUT / f"{arm}_inputs.parquet").loc[lambda x: x.phase.eq("evaluation")]
        first_target = inputs.origin_index.to_numpy(int)
        source, _ = load_prices()
        actual = source.HOEP.to_numpy(float)
        length = 3 if arm == "operator_short" else 12
        truth = actual[first_target[:, None]+np.arange(length)[None, :]]
        for policy in POLICIES:
            prediction = inputs[[f"{policy}_{j:02d}" for j in range(length)]].to_numpy(float)
            forecast_metrics.append({"arm": arm, "policy": policy, "forecast_targets": int(prediction.size),
                                     "available_forecast_targets": int(np.isfinite(prediction).sum()),
                                     "mae_cad_mwh": float(np.nanmean(abs(prediction-truth))), "rmse_cad_mwh": float(np.sqrt(np.nanmean((prediction-truth)**2))),
                                     "scope": "descriptive prediction accuracy only; source vintages differ between arms"})
    pd.DataFrame(policy_rows).to_csv(OUT / "policy_payoffs.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(months).to_csv(OUT / "monthly_cash_boundaries.csv", index=False)
    pd.DataFrame(forecast_metrics).to_csv(OUT / "forecast_metrics.csv", index=False)
    result = {"status": "PASS", "analyzed_utc": now(), "design_sha256": sha(OUT / "design.json"), "execution_receipt_sha256": sha(OUT / "execution_receipt.json"),
              "selected_current": selections, "policies": policy_rows, "contrasts": contrast_rows, "absorption_and_failures": boundaries,
              "all_months": months, "forecast_metrics": forecast_metrics,
              "inference_scope": "paired time-block stability of frozen realized trajectories; one market and six months; not a population, selection-adjusted, or causal market-effect interval",
              "cash_formula_recomputed": True, "endpoint_formula_recomputed": True, "source_sha256": controller_sources()}
    write_json(OUT / "summary.json", result)
    print(json.dumps({"task": "analyze", "selected": selections, "primary_contrasts": [r for r in contrast_rows if r["block_days"] == 14], "absorption": boundaries}), flush=True)
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["freeze", "amend", "download", "prepare", "smoke", "run", "analyze", "all"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root is not None:
        OUT = args.output_root.resolve()
    if os.name == "nt" and OUT.drive.upper() != "D:":
        raise ValueError("All computational outputs must remain on D")
    for step, function in [("freeze", freeze), ("amend", amend), ("download", download), ("prepare", prepare), ("smoke", smoke), ("run", lambda: run(args.workers)), ("analyze", analyze)]:
        if args.task in (step, "all"):
            function()


if __name__ == "__main__":
    main()
