"""Locked small exploratory physical/clock grid; never edits previous evidence."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import revision_v5_control_solver as core
import revision_v5_control_safe_policy as safe
import revision_v5_control_scaled_policy as scaled
import revision_v5_control_enumerated_policy as enum

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_v15/parameters"
HISTORY = ROOT / "fixtures/parameters/NSW1_inputs_aligned.parquet"
BASELINE = ROOT / "fixtures/parameters/NSW1_baseline_inputs.parquet"
P = [f"p_{j:02d}" for j in range(1, 13)]
POLICIES = ["current_smooth_a050", "inverse_lead", "sparse_equal"]
SETTINGS = {
    "baseline": {},
    "power0.5": {"power": 0.5},
    "power2.0": {"power": 2.0},
    "terminal2.0": {"terminal_factor": 2.0},
    "native_prefix4h_trade2h": {"h": 4, "curve_columns": 8},
    "update60min_execute_two_planned_steps": {"hold_steps": 2},
}
SOURCE_FILES = {}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def load():
    frames = []
    for path in (HISTORY, BASELINE):
        SOURCE_FILES[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        f = pd.read_parquet(path, filters=[("cutoff_minutes", "==", 60), ("phase", "==", "evaluation"),
                                          ("policy", "in", POLICIES)], columns=["target", "policy", *P, "actual_00"])
        f = f.loc[(f.target >= pd.Timestamp("2024-01-01")) & (f.target < pd.Timestamp("2024-02-01"))]
        frames.append(f)
    frame = pd.concat(frames).drop_duplicates(["target", "policy"]).sort_values("target")
    result = {policy: frame.loc[frame.policy == policy].reset_index(drop=True) for policy in POLICIES}
    ref = result[POLICIES[0]]
    assert len(ref) == 1488 and ref.target.diff().dropna().eq(pd.Timedelta(minutes=30)).all()
    for policy, part in result.items():
        assert part.target.equals(ref.target) and part.actual_00.equals(ref.actual_00)
        assert np.isfinite(part[[*P, "actual_00"]].to_numpy(float)).all()
    return result


def model_for(setting):
    kwargs = {key: value for key, value in SETTINGS[setting].items() if key not in ("curve_columns", "hold_steps")}
    return enum.EnumeratedSafeLexMPC(**kwargs)


def lock(frames):
    path = OUT / "locked_design.json"
    fixed = {"stage": "EXPLORATORY_REANALYSIS_NOT_INDEPENDENT_CONFIRMATION", "region": "NSW1",
        "chronological_choice": "Earliest evaluation month, January 2024; not chosen by observed payoff",
        "period": ["2024-01-01", "2024-02-01"], "origins": 1488, "initial_soc_mwh": 1.0,
        "nominal_capacity_mwh": 2.0, "soc_bounds_mwh": [0.2, 1.8], "policies": POLICIES,
        "fixed_Cstar": "current_smooth_a050; original strict-2023 selector retained, no parameter-specific reselection",
        "settings": SETTINGS, "baseline_controller": model_for("baseline").definition(),
        "realized_accounting": "dt*actual*(discharge-charge)-dt*kappa*(charge+discharge); endpoint actual-price marking common with factor1, even when planning terminal_factor changes",
        "native_prefix_semantics": "First8 existing half-hour forecast coefficients (4h), h4 trading steps (2h), mean of4 remaining terminal steps (2h); no generated leads",
        "hourly_update_semantics": "Replan at every second existing half-hour origin using only that origin's T-60 eligible curve. Execute its first two planned actions; do not use the skipped origin's new curve. Audit every half-hour successor inventory and both mode/power bounds.",
        "clock": "Original cutoff target-60min, execution target-30min. Predictions retain their native half-hour targets.",
        "unchanged_efficiency_and_wear": {"eta_each_direction": 0.91, "wear_aud_per_grid_mwh": 5.0},
        "comparison": "S and IL minus same-setting fixed current processor; baseline-relative changes reported for all policies, no winning setting selected",
        "old_results": "Previously completed efficiency/wear, energy-size/power, terminal-zero, annual and 30-minute availability-delay evidence is referenced, not repeated",
        "primary_endpoint": "January marked cash contrasts; descriptive only, no new p-value or equivalence claim",
        "resource_ceiling": "18 runs x1488 origins; abort after unresolved numerical failure or estimated sequential month grid exceeding1800s; no automatic large matrix",
        "sources": SOURCE_FILES, "first_target": str(frames[POLICIES[0]].target.iloc[0]),
        "last_target": str(frames[POLICIES[0]].target.iloc[-1])}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")); existing.pop("locked_utc", None)
        assert existing == fixed, "Locked design changed"
    else:
        write("locked_design.json", {"locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **fixed})


def benchmark(frames):
    rows = []
    for setting in SETTINGS:
        for policy in POLICIES:
            model = model_for(setting); curves = frames[policy][P].iloc[:16].to_numpy(float)
            columns = SETTINGS[setting].get("curve_columns", 12)
            stock = 1.0; started = time.perf_counter()
            for curve in curves:
                sol = model.solve(curve[:columns], stock); stock = sol["soc"]
            rows.append({"setting": setting, "policy": policy, "solves": 16,
                         "seconds": time.perf_counter() - started})
    estimate = sum(row["seconds"] / 16 * 1488 / SETTINGS[row["setting"]].get("hold_steps", 1) for row in rows)
    report = {"status": "PASS_RUNTIME_ONLY_BENCHMARK", "benchmark_solutions": 288,
              "estimated_sequential_grid_seconds": estimate, "allowed_ceiling_seconds": 1800,
              "extra_safety_factor_two_seconds": estimate * 2, "memory_estimate": "Processed January arrays and one small h4/h8 model, well below1GiB",
              "economic_results_used_for_design": False, "rows": rows}
    write("runtime_benchmark.json", report)
    if estimate > 1800:
        raise RuntimeError("Prespecified pilot resource ceiling exceeded")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)


def replay(frames, setting, policy):
    path = OUT / f"{setting}__{policy}.parquet"
    if path.with_suffix(".json").exists():
        rep = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert rep["status"] == "PASS" and rep["origins"] == 1488
        return rep
    model = model_for(setting); cfg = SETTINGS[setting]
    columns = cfg.get("curve_columns", 12); hold = cfg.get("hold_steps", 1)
    data = frames[policy]; curves = data[P].to_numpy(float); prices = data.actual_00.to_numpy(float)
    stock = 1.0; rows = []; started = time.perf_counter(); solves = 0; maxerr = 0.0
    for i in range(0, 1488, hold):
        sol = model.solve(curves[i, :columns], stock); solves += 1
        plan = np.asarray(sol["x"], float); h = model.h
        for j in range(min(hold, 1488 - i)):
            index = i + j; c = float(plan[j]); d = float(plan[h + j]); initial = stock
            stock = float(initial + model.eta * c * model.dt - d * model.dt / model.eta)
            assert min(c, d) < 1e-6 and min(c, d) >= -1e-8 and max(c, d) <= model.power + 1e-8
            assert model.smin - 1e-8 <= stock <= model.smax + 1e-8
            expected = float(plan[3*h + j + 1]); maxerr = max(maxerr, abs(stock - expected))
            assert abs(stock - expected) < 1e-6
            cash = model.dt * (prices[index] * (d - c) - model.kappa * (c + d))
            rows.append({"target": data.target.iloc[index], "execution_start": data.target.iloc[index] - pd.Timedelta(minutes=30),
                "planning_origin_target": data.target.iloc[i], "information_cutoff": data.target.iloc[i] - pd.Timedelta(minutes=60),
                "replanned": j == 0, "setting": setting, "policy": policy, "soc_start_mwh": initial,
                "soc_end_mwh": stock, "charge_mw": c, "discharge_mw": d,
                "actual_price": float(prices[index]), "net_aud": float(cash)})
    f = pd.DataFrame(rows)
    assert len(f) == 1488 and f.target.equals(data.target)
    assert np.max(np.abs(f.soc_end_mwh.to_numpy()[:-1] - f.soc_start_mwh.to_numpy()[1:])) < 1e-12
    assert (f.information_cutoff < f.execution_start).all()
    if hold == 2:
        assert solves == 744 and f.replanned.tolist() == [index % 2 == 0 for index in range(1488)]
        assert f.planning_origin_target.iloc[::2].reset_index(drop=True).equals(f.planning_origin_target.iloc[1::2].reset_index(drop=True))
    mark = float(stock * prices[-1] - prices[0])
    rep = {"status": "PASS", "setting": setting, "policy": policy, "origins": len(f), "solves": solves,
        "net_operating_aud": float(f.net_aud.sum()), "endpoint_mark_aud": mark,
        "marked_value_aud": float(f.net_aud.sum() + mark), "initial_soc_mwh": 1.0, "final_soc_mwh": stock,
        "seconds": time.perf_counter() - started, "controller": model.definition(),
        "maximum_plan_successor_difference_mwh": maxerr,
        "fallback_counts": {"safe": model.fallback_count, "scaled": model.scaled_fallback_count,
                            "enumerated": model.enumerated_fallback_count},
        "scope": "Exploratory one-region one-month point sensitivity; common endpoint actual marking; no inference, new selector or actual participant replication"}
    f.to_parquet(path.with_suffix(".parquet"), index=False)
    write(path.with_suffix(".json").name, rep)
    print(json.dumps({key: rep[key] for key in ("status", "setting", "policy", "origins", "solves", "seconds")}), flush=True)
    return rep




def main():
    global OUT
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["plan", "run", "single"], default="plan")
    parser.add_argument("--setting", choices=list(SETTINGS), default="update60min_execute_two_planned_steps")
    parser.add_argument("--policy", choices=POLICIES, default="inverse_lead")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    if args.output:
        OUT=args.output.resolve()
        if OUT.exists():raise FileExistsError("Explicit output must be a new directory")
    OUT.mkdir(parents=True, exist_ok=True)
    assert OUT.drive.upper() == "D:" and shutil.disk_usage(ROOT).free > 2 * 1024**3
    for module in (core, safe, scaled, enum):
        module.OUT = OUT / "numerical_failures"
    frames = load(); lock(frames)
    if args.mode == "single":
        rep=replay(frames,args.setting,args.policy)
        write("fresh_single_execution_receipt.json", {"status": "PASS_FRESH_SINGLE_PARAMETER_REPLAY", "report": rep,
              "script_sha256": sha(__file__), "scope": "One requested locked setting and policy, not a new18-cell matrix"})
        print(json.dumps({"status": "PASS_FRESH_SINGLE_PARAMETER_REPLAY", "setting": args.setting,
                          "policy": args.policy, "origins": rep["origins"], "solves": rep["solves"],
                          "marked_value_aud": rep["marked_value_aud"]}),flush=True)
        return
    if not (OUT / "runtime_benchmark.json").exists():
        benchmark(frames)
    if args.mode == "plan":
        return
    began = time.perf_counter()
    reports = [replay(frames, setting, policy) for setting in SETTINGS for policy in POLICIES]
    rows = []
    for rep in reports:
        current = next(v for v in reports if v["setting"] == rep["setting"] and v["policy"] == POLICIES[0])
        base = next(v for v in reports if v["setting"] == "baseline" and v["policy"] == rep["policy"])
        rows.append({key: rep[key] for key in ("setting", "policy", "origins", "solves", "marked_value_aud", "seconds")} |
                    {"minus_same_setting_Cstar_aud": rep["marked_value_aud"] - current["marked_value_aud"],
                     "minus_same_setting_Cstar_aud_per_day": (rep["marked_value_aud"] - current["marked_value_aud"]) / 31,
                     "minus_own_baseline_aud": rep["marked_value_aud"] - base["marked_value_aud"]})
    pd.DataFrame(rows).to_csv(OUT / "parameter_grid.csv", index=False)
    summary = {"status": "PASS_ALL_LOCKED_PARAMETER_RUNS", "runs": len(reports), "origins_per_run": 1488,
               "rows": rows, "runtime_seconds_this_invocation": time.perf_counter() - began,
               "locked_design_sha256": sha(OUT / "locked_design.json"), "script_sha256": sha(__file__),
               "sources": SOURCE_FILES, "scope": "Exploratory stylized NSW January2024; no statistical confirmation, profitable setting selection, parameter-specific Cstar reselection, or long-horizon native extrapolation"}
    write("execution_receipt.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "runs", "origins_per_run", "runtime_seconds_this_invocation")}), flush=True)


if __name__ == "__main__":
    main()
