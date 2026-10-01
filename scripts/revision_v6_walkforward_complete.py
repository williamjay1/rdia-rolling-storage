"""Complete, availability-purged half-year processor replay, in new artefacts.

The old v6 task omitted 2026H2.  This task never alters that task's output.
When its already solved prefix has exactly the same forecast inputs and audited
policy, only the remaining origins are solved from the prefix's final inventory.
Otherwise a full replay is required.  Segment cash values are never spliced.

Selection scores use complete candidate validation trajectories and the single
contiguous evaluation prefix, marked separately because those two experiments
start independently.  Actual outcomes must end strictly before the first new
half-year origin's T-minus-60-minute information deadline.  Ex-post candidate
values in the output are diagnostics, not selection inputs.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

import revision_v5_control_replays as R
from revision_v6_attribution import CANDIDATES, half_mask, period_value

ROOT, OUT = R.ROOT, R.OUT
COMPLETE_OUT = OUT
SOURCE_V6 = ROOT / "results/revision_v6"
V6 = ROOT / "results/revision_v6"
P = [f"p_{j:02d}" for j in range(1, 13)]
POLICY = "current_walkforward_complete"
OLD_POLICY = "current_walkforward"
SOLVER_FILES = {
    "solver_sha256": "revision_v5_control_solver.py",
    "safety_sha256": "revision_v5_control_safe_policy.py",
    "scaled_safety_sha256": "revision_v5_control_scaled_policy.py",
    "enumerated_safety_sha256": "revision_v5_control_enumerated_policy.py",
}
FALLBACKS = [
    ("fallback_count", "fallback_records", "numerical_fallback_count", "numerical_fallbacks"),
    ("scaled_fallback_count", "scaled_fallback_records", "equivalent_unit_fallback_count", "equivalent_unit_fallbacks"),
    ("enumerated_fallback_count", "enumerated_fallback_records", "enumerated_fallback_count", "enumerated_fallbacks"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(package_root=None, output_root=None):
    """Source and output roots can differ; copied reports need no old D path."""
    global ROOT, OUT, COMPLETE_OUT, SOURCE_V6, V6
    ROOT = Path(package_root).resolve() if package_root else R.ROOT
    destination = Path(output_root).resolve() if output_root else ROOT
    OUT = ROOT / "results/revision_v5/control"
    COMPLETE_OUT = destination / "results/revision_v5/control"
    SOURCE_V6 = ROOT / "results/revision_v6"
    V6 = destination / "results/revision_v6"
    R.ROOT = ROOT
    R.HISTORY = ROOT / "results/revision_v5/history"
    R.OUT = COMPLETE_OUT


def trajectory_path(report_path, report):
    sibling = Path(report_path).with_suffix(".parquet")
    if sibling.exists():
        return sibling
    recorded = Path(report["trajectory_path"])
    assert recorded.exists(), f"missing trajectory beside {report_path}"
    return recorded


def load_traj(region, policy, phase="evaluation", scenario="main"):
    path = OUT / scenario / f"{region}_{phase}_c60_{policy}.json"
    if not path.exists():
        return None, None
    report = json.loads(path.read_text())
    frame = pd.read_parquet(trajectory_path(path, report))
    frame["target"] = frame.target.astype("datetime64[ns]")
    return frame, report


def validate_physics(report):
    a = report["audit"]
    for key in ("max_soc_balance_error_mwh", "max_continuity_error_mwh"):
        assert key in a and a[key] <= 1e-6, (key, a.get(key))
    for key in ("mode_violations_gt1e-6", "bounds_violations"):
        assert key in a and a[key] == 0, (key, a.get(key))
    for key in ("max_primary_loss_aud", "max_primary_loss_upper_aud"):
        assert key in a and a[key] <= 2e-5, (key, a.get(key))


def plan(region):
    evals, vals, inputs = {}, {}, {}
    for p in CANDIDATES:
        f, _ = load_traj(region, p)
        v, _ = load_traj(region, p, phase="validation")
        assert f is not None and v is not None, f"incomplete candidates {region}/{p}"
        evals[p] = f.sort_values("target").reset_index(drop=True)
        vals[p] = v.sort_values("target").reset_index(drop=True)
        inputs[p] = R.load_inputs(region, policies=[p]).sort_values("target").reset_index(drop=True)
    ref, base = evals["raw"], inputs["raw"]
    for p in CANDIDATES:
        assert ref.target.equals(evals[p].target), f"candidate chronology {p}"
        assert np.array_equal(ref.actual_price.to_numpy(), evals[p].actual_price.to_numpy())
        assert base.target.equals(inputs[p].target)
        assert np.array_equal(base.actual_00.to_numpy(), inputs[p].actual_00.to_numpy())
    assert ref.target.equals(base.target)
    halves = sorted({(int(t.year), 1 if t.month <= 6 else 2) for t in ref.target})
    stitched = base.copy()
    choices, observed = [], []
    for year, half in halves:
        mask = half_mask(base, year, half).to_numpy()
        first = pd.Timestamp(base.target[mask].iloc[0])
        deadline = first - pd.Timedelta(minutes=60)
        scores, last_targets, counts = {}, {}, {}
        for p in CANDIDATES:
            prior_v = vals[p][vals[p].target < deadline]
            prior_e = evals[p][evals[p].target < deadline]
            assert len(prior_v), "no prior validation data"
            scores[p] = period_value(prior_v, np.ones(len(prior_v), bool))["value_aud"]
            if len(prior_e):
                scores[p] += period_value(prior_e, np.ones(len(prior_e), bool))["value_aud"]
            last_targets[p] = str(max(prior_v.target.iloc[-1], prior_e.target.iloc[-1]
                                     if len(prior_e) else prior_v.target.iloc[-1]))
            counts[p] = len(prior_v) + len(prior_e)
            assert pd.Timestamp(last_targets[p]) < deadline
        best = max(scores, key=lambda p: (scores[p], p == "raw", p))
        stitched.loc[mask, P] = inputs[best].loc[mask, P].to_numpy()
        choices.append({"half": f"{year}H{half}", "selected": best,
                        "first_target": str(first), "selection_deadline": str(deadline),
                        "latest_outcome_target": last_targets,
                        "prior_outcome_counts": counts,
                        "prior_values_aud": scores})
        vv = {p: period_value(evals[p], mask)["value_aud"] for p in CANDIDATES}
        observed_best = max(vv, key=lambda p: (vv[p], p == "raw", p))
        observed.append({"half": f"{year}H{half}", "ex_post_best": observed_best,
                         "values_aud": vv, "walk_forward_choice": best,
                         "walk_forward_choice_value_aud": vv[best],
                         "regret_vs_ex_post_aud": vv[observed_best] - vv[best]})
    return stitched, inputs, choices, observed


def reusable_prefix(region, stitched, inputs, choices):
    old_report = OUT / "main" / f"{region}_evaluation_c60_{OLD_POLICY}.json"
    old_plan = SOURCE_V6 / "compute" / f"{region}_walkforward.json"
    if not old_report.exists() or not old_plan.exists():
        return None, {"reusable": False, "reason": "old complete replay/plan pending"}
    rep = json.loads(old_report.read_text())
    old = json.loads(old_plan.read_text())
    validate_physics(rep)
    for key, filename in SOLVER_FILES.items():
        assert rep[key] == sha(ROOT / "scripts" / filename), f"changed solver source {key}"
    solver = R.LexMPC()
    assert rep["policy_definition"] == solver.definition()
    assert rep["initial_soc_mwh"] == 1.0 and rep["intervals"] == len(stitched)
    # Rebuild the exact old full forecast array, including its implicit raw H2.
    old_inputs = inputs["raw"].copy()
    for h, p in old["stitched_policy"].items():
        year, half = int(h[:4]), int(h[-1])
        mask = half_mask(old_inputs, year, half).to_numpy()
        old_inputs.loc[mask, P] = inputs[p].loc[mask, P].to_numpy()
    old_x = old_inputs[P].to_numpy(float)
    assert rep["forecast_array_sha256"] == hashlib.sha256(old_x.tobytes()).hexdigest()
    new_x = stitched[P].to_numpy(float)
    different = np.flatnonzero(np.any(old_x != new_x, axis=1))
    n = int(different[0]) if len(different) else len(stitched)
    # Conservative reuse stops before the previously omitted final half-year,
    # even if some early H2 forecast rows happen to match.
    last_half = choices[-1]["half"]
    last_half_first = int(np.flatnonzero(half_mask(stitched, int(last_half[:4]), int(last_half[-1])).to_numpy())[0])
    n = min(n, last_half_first)
    if n == 0:
        return None, {"reusable": False, "reason": "strict selection changed the first half-year"}
    source_trajectory = trajectory_path(old_report, rep)
    frame = pd.read_parquet(source_trajectory).sort_values("target").reset_index(drop=True)
    assert frame.target.equals(stitched.target)
    assert np.array_equal(frame.actual_price.to_numpy(), stitched.actual_00.to_numpy())
    assert np.array_equal(old_x[:n], new_x[:n]), "forecast prefix differs"
    # Localize each legacy fallback by its exact scientific input and state.
    # If an input/state repeats on both sides of the continuation boundary,
    # provenance is ambiguous and the prefix is not reused.
    prefix_fallbacks = {}
    for _, _, count_key, records_key in FALLBACKS:
        records = rep.get(records_key, [])
        assert len(records) == rep.get(count_key, 0), f"fallback record count {records_key}"
        keep = []
        for record in records:
            matches = np.flatnonzero(np.all(old_x == np.asarray(record["path"], float), axis=1)
                                     & (frame.soc_start.to_numpy(float) == float(record["soc"])))
            if not len(matches) or (np.any(matches < n) and np.any(matches >= n)):
                return None, {"reusable": False, "reason": "legacy fallback cannot be localized unambiguously"}
            if np.all(matches < n):
                item = dict(record)
                item["source_matching_origins"] = matches.tolist()
                item["source_matching_targets"] = [str(frame.target.iloc[i]) for i in matches]
                keep.append(item)
        prefix_fallbacks[records_key] = keep
    prefix = frame.iloc[:n].copy()
    return prefix, {"reusable": True, "prefix_origins": n,
                    "first_recomputed_target": str(stitched.target.iloc[n]),
                    "prefix_final_soc_mwh": float(prefix.soc_end.iloc[-1]),
                    "source_report": str(old_report), "source_report_sha256": sha(old_report),
                    "source_trajectory": str(source_trajectory),
                    "source_trajectory_sha256": sha(source_trajectory),
                    "prefix_fallbacks": prefix_fallbacks,
                    "prefix_fields_sha256": hashlib.sha256(prefix[R.FIELDS].to_numpy(float).tobytes()).hexdigest(),
                    "prefix_forecasts_sha256": hashlib.sha256(new_x[:n].tobytes()).hexdigest()}


def complete(region, plan_only=False, require_prefix=False):
    start = time.time()
    stitched, inputs, choices, observed = plan(region)
    prefix, provenance = reusable_prefix(region, stitched, inputs, choices)
    if plan_only:
        print(json.dumps({"region": region, "halves": choices, "prefix": provenance}), flush=True)
        return
    assert not require_prefix or prefix is not None, provenance
    report_path = COMPLETE_OUT / "main" / f"{region}_evaluation_c60_{POLICY}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        assert rep["forecast_array_sha256"] == hashlib.sha256(stitched[P].to_numpy(float).tobytes()).hexdigest()
        cached = pd.read_parquet(trajectory_path(report_path, rep))
        assert cached.target.equals(stitched.target)
        assert np.array_equal(cached.actual_price.to_numpy(), stitched.actual_00.to_numpy())
        validate_physics(rep)
    elif prefix is None:
        rep = R.replay(region, stitched, POLICY)
    else:
        solver = R.LexMPC()
        n = len(prefix)
        soc = float(prefix.soc_end.iloc[-1])
        x = stitched[P].to_numpy(float)
        actual = stitched.actual_00.to_numpy(float)
        rows = prefix[R.FIELDS].to_numpy(float).tolist()
        for i in range(n, len(stitched)):
            sol = solver.solve(x[i], soc)
            c, d = sol["c"], sol["d"]
            cash = (d - c) * actual[i] * solver.dt
            wear = (c + d) * solver.kappa * solver.dt
            rows.append((soc, c, d, sol["soc"], cash, wear, cash - wear,
                         sol["primary_optimum_incumbent_aud"], sol["policy_primary_loss_aud"],
                         sol["policy_optimum_loss_upper_aud"], sol["primary_objective_gap_aud"]))
            soc = sol["soc"]
            if (i - n + 1) % 1000 == 0:
                print(json.dumps({"region": region, "new_tail_origins": i-n+1,
                                  "seconds": time.time()-start}), flush=True)
        frame = pd.DataFrame({"target": stitched.target, "actual_price": actual})
        frame["execution_start"] = frame.target - pd.Timedelta(hours=solver.dt)
        frame["execution_day"] = frame.execution_start.dt.floor("D")
        frame[R.FIELDS] = np.asarray(rows, float)
        frame.insert(0, "region", region)
        assert np.array_equal(frame[R.FIELDS].iloc[:n].to_numpy(float), prefix[R.FIELDS].to_numpy(float))
        assert frame.soc_start.iloc[n] == prefix.soc_end.iloc[-1], "SOC reset at continuation"
        dest = report_path.with_suffix(".parquet")
        rep = R.metric(frame, solver, region, POLICY, "evaluation", hashlib.sha256(x.tobytes()).hexdigest())
        rep.update(cutoff_minutes=60, scenario="main", nominal_energy_mwh=2.0,
                   eta_c=.91, eta_d=.91, kappa_aud_per_grid_mwh=5.0,
                   seconds=time.time()-start, trajectory_path=str(dest), endpoint_mark_factor=1.0,
                   reused_prefix_origins=n, recomputed_tail_origins=len(stitched)-n,
                   replay_driver_sha256=sha(__file__), prefix_provenance=provenance)
        for key, filename in SOLVER_FILES.items():
            rep[key] = sha(ROOT / "scripts" / filename)
        for count, records, count_key, records_key in FALLBACKS:
            inherited = provenance["prefix_fallbacks"][records_key]
            rep[count_key] = len(inherited) + getattr(solver, count)
            rep[records_key] = inherited + getattr(solver, records)
        validate_physics(rep)
        # Write only new artefacts after the entire trajectory audit has passed.
        frame.to_parquet(dest, index=False)
        report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    validate_physics(rep)
    out = {"region": region, "halves": choices, "observed": observed,
           "stitched_policy": {h["half"]: h["selected"] for h in choices},
           "replay_policy": POLICY, "replay_net_value_aud": rep["net_value_aud"],
           "prefix_provenance": provenance,
           "selection_score_definition": "marked validation episode + single marked evaluation prefix; observed targets strictly before first T-minus-60 deadline",
           "availability_assumption": "an actual interval price is usable only after its target timestamp; finalized-price publication latency is not separately observed",
           "stitching_note": "selected forecasts replayed as one continuous inventory trajectory, including every observed half-year; no SOC restart or segment-cash splice"}
    p = V6 / "compute" / f"{region}_walkforward_complete.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"region": region, "policy": POLICY, "value_aud": rep["net_value_aud"],
                      "halves": out["stitched_policy"], "prefix": provenance,
                      "audit": rep["audit"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=R.REGIONS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--require-prefix", action="store_true")
    ap.add_argument("--package-root", help="source package root containing scripts/results")
    ap.add_argument("--output-root", help="distinct writable root; source reports remain read-only")
    a = ap.parse_args()
    configure(a.package_root, a.output_root)
    for region in a.regions:
        complete(region, a.plan_only, a.require_prefix)
