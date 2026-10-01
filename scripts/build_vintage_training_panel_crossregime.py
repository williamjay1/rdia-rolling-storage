from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


RAW_DIRS = [
    Path(r"F:\AcademicData\AEMO_PredispatchRevision\raw\sample_2015_2020_v1_20260928"),
    Path(r"F:\AcademicData\AEMO_PredispatchRevision\raw\pre5ms_bridge_2021_01_09_v1_20260929"),
    Path(r"F:\AcademicData\AEMO_PredispatchRevision\raw\post5ms_2021_2026_v1_20260928"),
    Path(r"F:\AcademicData\AEMO_PredispatchRevision\raw\post5ms_2021_2026_retry_20260929"),
    Path(r"F:\AcademicData\AEMO_PredispatchRevision\raw\post5ms_2022_10_recovery_20260929"),
]
POST_ACTUAL_DIR = Path(r"D:\MLWork\AEMO_PredispatchRevision\cache\post5ms_actual_price_aoor_v3")
PROJECT = Path(r"D:\MLWork\AOOR_ContextualStorage_20260929")
CACHE = PROJECT / "cache" / "vintage_training_crossregime_v1"
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
MIN_LEAD, MAX_LEAD, HISTORY_MINUTES = 45.0, 450.0, 180


def month_shift(year: int, month: int, delta: int) -> tuple[int, int]:
    n = year * 12 + month - 1 + delta
    return n // 12, n % 12 + 1


def month_archive(table: str, year: int, month: int) -> Path | None:
    marker = f"{year}{month:02d}010000"
    matches = []
    for root in RAW_DIRS:
        if not root.is_dir():
            continue
        for path in root.glob("*.zip"):
            name = path.name.upper()
            if "_D_" in name or table.upper() not in name or marker not in name:
                continue
            matches.append(path)
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous raw sources for {table} {year}-{month:02d}: {matches}")
    return matches[0] if matches else None


def read_actual_month(year: int, month: int) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(year=year, month=month, day=1)
    next_year, next_month = month_shift(year, month, 1)
    end = pd.Timestamp(year=next_year, month=next_month, day=1)
    if (year, month) >= (2021, 10):
        path = POST_ACTUAL_DIR / f"actual_post5ms_{year}_{month:02d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing audited post-5MS actual prices: {path}")
        actual = pd.read_parquet(path, columns=["target", "REGIONID", "actual", "actual_source"])
        actual["target"] = pd.to_datetime(actual["target"])
        actual["actual"] = pd.to_numeric(actual["actual"], errors="coerce")
        actual = actual.loc[actual["REGIONID"].isin(REGIONS) & actual["actual"].notna() &
                            actual["target"].gt(start) & actual["target"].le(end)].copy()
        if actual.duplicated(["target", "REGIONID"], keep=False).any():
            raise RuntimeError(f"Duplicate actual target/region keys in {path.name}")
        return actual, source_hash(path)
    path = month_archive("TRADINGPRICE", year, month)
    if path is None:
        raise FileNotFoundError(f"Missing TRADINGPRICE archive for {year}-{month:02d}")
    actual = read_aemo(path, "TRADING", "PRICE",
                       ["SETTLEMENTDATE", "REGIONID", "RRP", "INVALIDFLAG", "PRICE_STATUS"])
    actual = actual[(actual["REGIONID"].isin(REGIONS)) & (actual["INVALIDFLAG"] == "0") &
                    (actual["PRICE_STATUS"] == "FIRM")].copy()
    actual["target"] = pd.to_datetime(actual["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    actual["actual"] = pd.to_numeric(actual["RRP"], errors="coerce")
    actual = actual.dropna(subset=["target", "actual"])
    actual = actual.loc[actual["target"].gt(start) & actual["target"].le(end),
                        ["target", "REGIONID", "actual"]].copy()
    actual["actual_source"] = "TRADINGPRICE"
    if actual.duplicated(["target", "REGIONID"], keep=False).any():
        raise RuntimeError(f"Duplicate actual target/region keys in {path.name}")
    return actual, source_hash(path)


def read_aemo(path: Path, table: str, report: str, columns: list[str]) -> pd.DataFrame:
    with ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise IOError(f"ZIP integrity error in {path.name}: {bad}")
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one CSV member, got {members}")
        with archive.open(members[0]) as stream:
            stream.readline()
            raw_header = next(csv.reader([stream.readline().decode("utf-8-sig").rstrip("\r\n")]))
        names = ["kind", "table", "report", "version"] + raw_header[4:]
        required = ["kind", "table", "report", "version", *columns]
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream, skiprows=2, header=None, names=names,
                                usecols=required, dtype=str, keep_default_na=False,
                                low_memory=False)
    return frame.loc[(frame["kind"] == "D") & (frame["table"] == table) &
                     (frame["report"] == report), columns].copy()


def source_hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def add_revision_features(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "REGIONID"]
    pred = pred.sort_values(keys + ["asof", "PREDISPATCHSEQNO", "runno_num"]).reset_index(drop=True)
    grouped = pred.groupby(keys, sort=False, observed=True)
    pred["revision_30m"] = pred["forecast"] - grouped["forecast"].shift(1)
    for periods, label in ((2, "60m"), (4, "120m"), (6, "180m")):
        pred[f"revision_{label}"] = pred["forecast"] - grouped["forecast"].shift(periods)
    delta = pred["revision_30m"]
    rolling_abs = delta.abs().groupby([pred["target"], pred["REGIONID"]], sort=False).rolling(
        6, min_periods=1).sum()
    rolling_sd = delta.groupby([pred["target"], pred["REGIONID"]], sort=False).rolling(
        6, min_periods=2).std()
    pred["abs_revision_180m"] = rolling_abs.reset_index(level=[0, 1], drop=True).reindex(pred.index)
    pred["revision_sd_180m"] = rolling_sd.reset_index(level=[0, 1], drop=True).reindex(pred.index)
    sign = np.sign(delta)
    previous_sign = sign.groupby([pred["target"], pred["REGIONID"]], sort=False).shift(1)
    reversal = ((sign != 0) & (previous_sign != 0) & (sign != previous_sign)).astype("int8")
    rolling_reversal = reversal.groupby([pred["target"], pred["REGIONID"]], sort=False).rolling(
        6, min_periods=1).sum()
    pred["reversals_180m"] = rolling_reversal.reset_index(level=[0, 1], drop=True).reindex(pred.index)
    return pred


def build_month(year: int, month: int) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(year=year, month=month, day=1)
    next_year, next_month = month_shift(year, month, 1)
    end = pd.Timestamp(year=next_year, month=next_month, day=1)
    previous_year, previous_month = month_shift(year, month, -1)
    forecast_paths = [month_archive("PREDISPATCHPRICE", y, m)
                      for y, m in ((previous_year, previous_month), (year, month))]
    forecast_paths = [path for path in forecast_paths if path is not None and path.exists()]
    if not forecast_paths:
        raise FileNotFoundError(f"No PREDISPATCHPRICE archive for {year}-{month:02d}")

    parts = []
    columns = ["PREDISPATCHSEQNO", "RUNNO", "REGIONID", "INTERVENTION", "RRP", "LASTCHANGED", "DATETIME"]
    for path in forecast_paths:
        data = read_aemo(path, "PREDISPATCH", "REGION_PRICES", columns)
        data = data[(data["REGIONID"].isin(REGIONS)) & (data["INTERVENTION"] == "0")].copy()
        data["target"] = pd.to_datetime(data["DATETIME"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        data["asof"] = pd.to_datetime(data["LASTCHANGED"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        data["forecast"] = pd.to_numeric(data["RRP"], errors="coerce")
        data["runno_num"] = pd.to_numeric(data["RUNNO"], errors="coerce").fillna(-1)
        parts.append(data[["PREDISPATCHSEQNO", "RUNNO", "REGIONID", "target", "asof", "forecast", "runno_num"]])
    pred = pd.concat(parts, ignore_index=True)
    pred = pred.dropna(subset=["target", "asof", "forecast"])
    pred = pred[(pred["target"] > start) & (pred["target"] <= end)].copy()
    pred["lead_min"] = (pred["target"] - pred["asof"]).dt.total_seconds() / 60.0
    pred = pred[pred["lead_min"].between(20, MAX_LEAD + HISTORY_MINUTES + 15)].copy()
    pred = pred.sort_values(["PREDISPATCHSEQNO", "RUNNO", "target", "REGIONID", "asof", "runno_num"])
    pred = pred.drop_duplicates(["PREDISPATCHSEQNO", "RUNNO", "target", "REGIONID", "asof"], keep="last")
    pred = pred.sort_values(["target", "REGIONID", "asof", "PREDISPATCHSEQNO", "runno_num"])
    pred = pred.drop_duplicates(["target", "REGIONID", "asof"], keep="last")

    actual, actual_file = read_actual_month(year, month)

    pred = add_revision_features(pred)
    pred = pred.merge(actual, on=["target", "REGIONID"], how="inner", validate="many_to_one")
    # Training rows cover the as-of lead range required by a 4-hour controller
    # plus a 2-hour continuation value. Earlier rows remain in the feature
    # history but are not used as supervised examples.
    pred = pred[pred["lead_min"].between(MIN_LEAD, MAX_LEAD)].copy()
    pred["year"] = pred["target"].dt.year.astype("int16")
    pred["month"] = pred["target"].dt.month.astype("int8")
    pred["weekday"] = pred["target"].dt.dayofweek.astype("int8")
    pred["target_hour"] = pred["target"].dt.hour.astype("int8")
    pred["target_minute"] = pred["target"].dt.minute.astype("int8")
    pred["weekend"] = (pred["weekday"] >= 5).astype("int8")
    pred = pred[["target", "REGIONID", "asof", "PREDISPATCHSEQNO", "RUNNO", "lead_min", "forecast", "actual",
                 "revision_30m", "revision_60m", "revision_120m", "revision_180m", "abs_revision_180m",
                 "revision_sd_180m", "reversals_180m", "actual_source", "year", "month", "weekday", "target_hour",
                 "target_minute", "weekend"]]
    info = {"year": year, "month": month, "actual_pairs": int(len(actual)), "training_vintage_rows": int(len(pred)),
            "forecast_files": [source_hash(path) for path in forecast_paths],
            "actual_file": actual_file, "actual_sources": sorted(actual["actual_source"].dropna().astype(str).unique().tolist()),
            "target_lead_range_minutes": [MIN_LEAD, MAX_LEAD],
            "revision_history_minutes": HISTORY_MINUTES}
    return pred, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--tag", default="post2021_aoor_v1", help="new output namespace; refuses overwrites")
    parser.add_argument("--skip-months", nargs="*", default=["202210"], help="YYYYMM target months to exclude")
    args = parser.parse_args()
    output_cache = PROJECT / "cache" / f"vintage_training_{args.tag}"
    output_cache.mkdir(parents=True, exist_ok=True)
    PROJECT.joinpath("results").mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT / "results" / f"vintage_training_panel_manifest_{args.tag}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    audit = []
    pieces = []
    t0 = time.time()
    y, m = args.start_year, args.start_month
    while (y, m) <= (args.end_year, args.end_month):
        month_key = f"{y}{m:02d}"
        if month_key in set(args.skip_months):
            print(f"excluded target month {month_key} by archive-completeness rule", flush=True)
            y, m = month_shift(y, m, 1)
            continue
        out = output_cache / f"vintage_training_{args.tag}_{y}_{m:02d}.parquet"
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite derived file: {out}")
        panel, info = build_month(y, m)
        panel.to_parquet(out, index=False, compression="zstd")
        info["output"] = str(out)
        audit.append(info)
        pieces.append(panel)
        print(f"{y}-{m:02d}: actual={info['actual_pairs']:,}; vintages={len(panel):,}; seconds={info.get('seconds', 'n/a')}", flush=True)
        y, m = month_shift(y, m, 1)
    full = pd.concat(pieces, ignore_index=True).sort_values(["target", "REGIONID", "asof"]).reset_index(drop=True)
    full_path = output_cache / f"vintage_training_{args.tag}_{args.start_year}_{args.start_month:02d}_{args.end_year}_{args.end_month:02d}.parquet"
    if full_path.exists():
        raise FileExistsError(f"Refusing to overwrite derived file: {full_path}")
    full.to_parquet(full_path, index=False, compression="zstd")
    manifest = {
        "project": "AOOR contextual storage control with forecast vintages",
        "purpose": "chronological training and evaluation panel of AEMO forecast vintages and realized prices",
        "generated_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "raw_source_directories": list(map(str, RAW_DIRS)), "post5ms_actual_directory": str(POST_ACTUAL_DIR),
        "excluded_target_months": args.skip_months,
        "window": [f"{args.start_year}-{args.start_month:02d}", f"{args.end_year}-{args.end_month:02d}"],
        "regions": REGIONS, "lead_window_minutes": [MIN_LEAD, MAX_LEAD], "revision_history_minutes": HISTORY_MINUTES,
        "rows": int(len(full)), "actual_target_region_pairs": int(full[["target", "REGIONID"]].drop_duplicates().shape[0]),
        "unique_delivery_targets": int(full["target"].nunique()), "output": str(full_path),
        "elapsed_seconds": round(time.time() - t0, 2), "monthly_source_audit": audit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "monthly_source_audit"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
