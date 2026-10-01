"""v6 side tasks: inverse-lead updating branches and common-state ranges."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse, json, time
import numpy as np
import pandas as pd
from revision_v5_control_solver import OUT
from revision_v5_control_replays import REGIONS, load_inputs
from revision_v5_control_diagnostics import common_state, near_optimal_range
from revision_v5_control_enumerated_policy import EnumeratedSafeLexMPC as LexMPC
from revision_v5_event_rolling import probe_region

BASELINE = ("D:/MLWork/AOOR_ContextualStorage_20260929/results/revision_v5/"
            "control/main/{region}_evaluation_c60_raw.parquet")


def common_state_v6(region, stride=32, epsilon_aud=.01, history="inverse_lead"):
    """Aligned common-state ranges when the history rule lives in another file.

    The frozen v5 routine reads both policies from the baseline panel, which
    contains only the latest and sparse rules. Inverse weighting is stored in the
    full aligned panel, so the two arrays are loaded separately and checked on a
    common chronology before the common-state solves.
    """
    folder = OUT / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{region}_raw_{history}_common_state.parquet"
    repdest = folder / f"{region}_raw_{history}_common_state.json"
    if repdest.exists():
        print(f"skip {repdest}", flush=True)
        return
    a = load_inputs(region, kind="baseline", cutoff=60, policies=["raw"]).sort_values("target").reset_index(drop=True)
    b = load_inputs(region, kind="full", cutoff=60, policies=[history]).sort_values("target").reset_index(drop=True)
    ar = pd.read_parquet(OUT / "main" / f"{region}_evaluation_c60_raw.parquet")
    br = pd.read_parquet(OUT / "main" / f"{region}_evaluation_c60_{history}.parquet")
    for frame in (ar, br):
        frame["target"] = frame["target"].astype("datetime64[ns]")
    ar = ar.sort_values("target").reset_index(drop=True)
    br = br.sort_values("target").reset_index(drop=True)
    assert a.target.equals(b.target) and a.target.equals(ar.target) and a.target.equals(br.target)
    ac = a[[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float)
    bc = b[[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float)
    model = LexMPC(); records = []; start = time.time()
    for i in range(0, len(a), stride):
        sr = float(ar.soc_start.iloc[i]); sh = float(br.soc_start.iloc[i])
        rr = model.solve(ac[i], sr); hr = model.solve(bc[i], sr)
        rh = model.solve(ac[i], sh); hh = model.solve(bc[i], sh)
        r = near_optimal_range(model, ac[i], sr, epsilon_aud)
        v = near_optimal_range(model, bc[i], sr, epsilon_aud)
        row = {"target": a.target.iloc[i], "sample_index": i, "raw_soc_mwh": sr,
               "history_soc_mwh": sh, "u_raw_at_raw_soc": rr["u"], "u_history_at_raw_soc": hr["u"],
               "u_raw_at_history_soc": rh["u"], "u_history_at_history_soc": hh["u"],
               "actual_raw_u_mw": float(ar.discharge_mw.iloc[i] - ar.charge_mw.iloc[i]),
               "actual_history_u_mw": float(br.discharge_mw.iloc[i] - br.charge_mw.iloc[i])}
        row.update({f"raw_{k}": value for k, value in r.items()})
        row.update({f"history_{k}": value for k, value in v.items()})
        info = ((hr["u"] - rr["u"]) + (hh["u"] - rh["u"])) / 2
        state = ((rh["u"] - rr["u"]) + (hh["u"] - hr["u"])) / 2
        row.update(information_component_mw=info, state_component_mw=state,
                   diagonal_difference_mw=hh["u"] - rr["u"],
                   input_action_change_gt01=abs(hr["u"] - rr["u"]) > .1,
                   primary_sets_force_change_gt01=(r["range_lower_outer_mw"] > v["range_upper_outer_mw"] + .1 or
                                                   v["range_lower_outer_mw"] > r["range_upper_outer_mw"] + .1),
                   near_optimal_ranges_overlap=(r["range_lower_outer_mw"] <= v["range_upper_outer_mw"] + 1e-7 and
                                               v["range_lower_outer_mw"] <= r["range_upper_outer_mw"] + 1e-7))
        records.append(row)
    d = pd.DataFrame(records); d.insert(0, "region", region); d.to_parquet(dest, index=False)
    changed = d.input_action_change_gt01
    rep = {"status": "completed", "region": region, "history": history, "sample_stride": stride,
           "origins": len(d), "primary_near_optimal_epsilon_aud": epsilon_aud,
           "seconds": time.time() - start,
           "same_raw_SOC_change_share_gt01": float(changed.mean()),
           "same_hist_SOC_change_share_gt01": float((abs(d.u_history_at_history_soc - d.u_raw_at_history_soc) > .1).mean()),
           "actual_diagonal_change_share_gt01": float((abs(d.actual_history_u_mw - d.actual_raw_u_mw) > .1).mean()),
           "primary_sets_force_change_share_gt01": float(d.primary_sets_force_change_gt01.mean()),
           "forced_share_among_changed": float(d.loc[changed, "primary_sets_force_change_gt01"].mean()) if changed.any() else None,
           "overlap_share_among_changed": float(d.loc[changed, "near_optimal_ranges_overlap"].mean()) if changed.any() else None,
           "max_decomposition_identity_error_mw": float(abs(d.information_component_mw + d.state_component_mw - d.diagonal_difference_mw).max()),
           "max_cold_raw_vs_executed_diff_mw": float(abs(d.u_raw_at_raw_soc - d.actual_raw_u_mw).max()),
           "max_cold_history_vs_executed_diff_mw": float(abs(d.u_history_at_history_soc - d.actual_history_u_mw).max()),
           "numerical_bounds_note": "First action outer endpoints use LP/MIP dual bounds; primary incumbent+epsilon allows the reported primary gap. Numerical certificates, not exact-arithmetic proof.",
           "input_sources": {"latest": "results/revision_v5/history/%s_baseline_inputs.parquet" % region,
                             "history": "results/revision_v5/history/%s_inputs_aligned.parquet" % region},
           "scope": "fixed stride across the complete exploration period; common-SOC analysis explains model changes, not market causal effects"}
    repdest.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["branches", "common", "both"], required=True)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--history", default="inverse_lead")
    a = ap.parse_args()
    for region in a.regions:
        start = time.time()
        if a.task in ("branches", "both"):
            probe_region(region, a.history, BASELINE.format(region=region),
                         "revision_v5_control_scaled_policy", "ScaledSafeLexMPC")
        if a.task in ("common", "both"):
            if a.history == "sparse_equal":
                common_state(region, a.stride, .01, a.history)
            else:
                common_state_v6(region, a.stride, .01, a.history)
        print(json.dumps({"region": region, "task": a.task, "seconds": time.time() - start}), flush=True)


if __name__ == "__main__":
    main()
