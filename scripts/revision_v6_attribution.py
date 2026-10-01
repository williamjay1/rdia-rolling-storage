"""v6 attribution, selection-validity and sensitivity analysis.

Everything here reads already completed trajectories. No solver call is made
except in the walk-forward task, which replays the stitched policy once per
region. Sub-period values use the same objective as the frozen selection:
rolling net energy cash plus a matching opening/closing inventory mark.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from revision_v5_control_solver import ROOT, OUT
from revision_v5_control_replays import REGIONS, load_inputs, replay
from revision_v5_control_aggregate import paired_block_interval

V6 = ROOT / "results" / "revision_v6"
DEST = V6 / "integrated"
CANDIDATES = ["raw", "current_shrink_a025", "current_shrink_a050",
              "current_smooth_a025", "current_smooth_a050"]
P = [f"p_{j:02d}" for j in range(1, 13)]
NOMINAL_DAYS = 974.0
NUM_COMMON = {}


# ---------------------------------------------------------------- trajectories
def load_traj(region, policy, phase="evaluation", cutoff=60, scenario="main"):
    report = OUT / scenario / f"{region}_{phase}_c{cutoff}_{policy}.json"
    if not report.exists():
        return None, None
    rep = json.loads(report.read_text())
    f = pd.read_parquet(rep["trajectory_path"])
    f["target"] = f["target"].astype("datetime64[ns]")
    f["execution_day"] = f["execution_day"].astype("datetime64[ns]")
    return f, rep


def period_value(frame, mask):
    """Selection-consistent value over a contiguous sub-period."""
    sub = frame[mask]
    if sub.empty:
        return None
    mark = float(sub.soc_end.iloc[-1] * sub.actual_price.iloc[-1]
                 - sub.soc_start.iloc[0] * sub.actual_price.iloc[0])
    return {"origins": int(len(sub)), "operating_aud": float(sub.net_aud.sum()),
            "endpoint_mark_aud": mark, "value_aud": float(sub.net_aud.sum() + mark),
            "first_target": str(sub.target.iloc[0]), "last_target": str(sub.target.iloc[-1])}


def half_mask(frame, year, half):
    t = frame.target
    lo = pd.Timestamp(f"{year}-01-01") if half == 1 else pd.Timestamp(f"{year}-07-01")
    hi = pd.Timestamp(f"{year}-07-01") if half == 1 else pd.Timestamp(f"{year + 1}-01-01")
    return (t >= lo) & (t < hi)


def select(cands, frame_by_policy, mask):
    vals = {p: period_value(frame_by_policy[p], mask)["value_aud"] for p in cands}
    best = max(vals, key=lambda p: (vals[p], p == "raw", p))
    return best, vals


# ------------------------------------------------------- 2023 selection stability
def selection_stability():
    rows = []
    detail = []
    for region in REGIONS:
        frames = {}
        for p in CANDIDATES:
            f, _ = load_traj(region, p, phase="validation")
            if f is None:
                raise RuntimeError(f"missing validation trajectory {region}/{p}")
            frames[p] = f
        frozen = json.loads((OUT / "main" / f"{region}_current_only_selection.json").read_text())
        rec = {"region": region, "frozen_selection": frozen["selected_policy"]}
        periods = {
            "2023_full": lambda f: pd.Series(True, index=f.index),
            "2023_H1": lambda f: half_mask(f, 2023, 1),
            "2023_H2": lambda f: half_mask(f, 2023, 2),
            "2023_Q1": lambda f: (f.target >= "2023-01-01") & (f.target < "2023-04-01"),
            "2023_Q2": lambda f: (f.target >= "2023-04-01") & (f.target < "2023-07-01"),
            "2023_Q3": lambda f: (f.target >= "2023-07-01") & (f.target < "2023-10-01"),
            "2023_Q4": lambda f: (f.target >= "2023-10-01") & (f.target < "2024-01-01"),
        }
        ref = frames["raw"]
        for name, fn in periods.items():
            mask = fn(ref).to_numpy()
            best, vals = select(CANDIDATES, frames, mask)
            rec[name] = {"winner": best,
                         "values_aud": {k: round(v, 2) for k, v in vals.items()},
                         "margin_over_runner_up_aud": round(
                             vals[best] - sorted(vals.values())[-2], 2)}
            rows.append({"region": region, "period": name, "winner": best,
                         "value_raw_aud": vals["raw"], "value_winner_aud": vals[best],
                         "margin_over_runner_up_aud": vals[best] - sorted(vals.values())[-2],
                         "frozen_2023_choice": frozen["selected_policy"],
                         "frozen_choice_is_winner": best == frozen["selected_policy"]})
        # leave-one-half-out: select on one half, apply to the other.
        for train, test in (((2023, 1), (2023, 2)), ((2023, 2), (2023, 1))):
            tr = ref[half_mask(ref, *train).to_numpy()]
            te = half_mask(ref, *test).to_numpy()
            best_tr, _ = select(CANDIDATES, frames, half_mask(ref, *train).to_numpy())
            best_te, vals_te = select(CANDIDATES, frames, te)
            rec[f"train_{train[0]}H{train[1]}_test_{test[0]}H{test[1]}"] = {
                "train_winner": best_tr, "test_winner": best_te,
                "test_value_of_train_winner_aud": vals_te[best_tr],
                "test_value_of_test_winner_aud": vals_te[best_te],
                "regret_aud": vals_te[best_te] - vals_te[best_tr],
                "test_origins": int(te.sum())}
        rows.append({"region": region, "period": "summary",
                     "winner": rec["2023_H1"]["winner"] + "->" + rec["2023_H2"]["winner"],
                     "frozen_choice_is_winner": rec["2023_H1"]["winner"] == rec["2023_H2"]["winner"]})
        detail.append(rec)
    DEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DEST / "v6_selection_stability.csv", index=False)
    (DEST / "v6_selection_stability_detail.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    return rows


# --------------------------------------------------- out-of-sample evaluation
def out_of_sample():
    rows = []
    for region in REGIONS:
        frozen = json.loads((OUT / "main" / f"{region}_current_only_selection.json").read_text())
        picked = frozen["selected_policy"]
        frames, values = {}, {}
        for p in CANDIDATES:
            f, r = load_traj(region, p)
            if f is None:
                continue
            frames[p] = f
            values[p] = r["net_value_aud"]
        if len(frames) != 5:
            raise RuntimeError(f"incomplete evaluation grid {region}: {sorted(frames)}")
        ex_post = max(values, key=lambda p: (values[p], p == "raw", p))
        for p in CANDIDATES:
            rows.append({"region": region, "policy": p, "evaluation_value_aud": values[p],
                         "evaluation_aud_per_day": values[p] / NOMINAL_DAYS,
                         "selected_in_2023": p == picked, "ex_post_best": p == ex_post})
        rows.append({"region": region, "policy": "REGRET_2023_CHOICE",
                     "evaluation_value_aud": values[ex_post] - values[picked],
                     "evaluation_aud_per_day": (values[ex_post] - values[picked]) / NOMINAL_DAYS,
                     "selected_in_2023": True, "ex_post_best": False,
                     "note": "value forgone by keeping the 2023 choice instead of the ex-post best candidate"})
        rows.append({"region": region, "policy": "REGRET_VS_RAW",
                     "evaluation_value_aud": values[picked] - values["raw"],
                     "evaluation_aud_per_day": (values[picked] - values["raw"]) / NOMINAL_DAYS,
                     "selected_in_2023": True, "ex_post_best": False,
                     "note": "value added by the 2023-selected processor over the untreated latest curve"})
    pd.DataFrame(rows).to_csv(DEST / "v6_current_out_of_sample.csv", index=False)
    return rows


# ------------------------------------------------------------- walk-forward
def walkforward_score(region, upto_half, frames_eval):
    """Prior-data score for each candidate strictly before the given half."""
    val = {}
    for p in CANDIDATES:
        fv, _ = load_traj(region, p, phase="validation")
        s = period_value(fv, np.ones(len(fv), bool))["value_aud"]
        for (year, half), fr in frames_eval.items():
            if (year, half) >= upto_half:
                continue
            s += period_value(fr[p], np.ones(len(fr[p]), bool))["value_aud"]
        val[p] = s
    best = max(val, key=lambda p: (val[p], p == "raw", p))
    return best, val


def walkforward(region, dry_run=False):
    frames = {}
    for p in CANDIDATES:
        f, _ = load_traj(region, p)
        if f is None:
            raise RuntimeError(f"missing evaluation trajectory {region}/{p}")
        frames[p] = f.sort_values("target").reset_index(drop=True)
    halves = [(2024, 1), (2024, 2), (2025, 1), (2025, 2), (2026, 1)]
    ref = frames["raw"]
    halves = [h for h in halves if half_mask(ref, *h).any()]
    by_half = {}
    for h in halves:
        mask = half_mask(ref, *h).to_numpy()
        by_half[h] = {p: frames[p][mask].reset_index(drop=True) for p in CANDIDATES}
    chosen, detail = {}, []
    for h in halves:
        best, val = walkforward_score(region, h, by_half)
        chosen[h] = best
        detail.append({"half": f"{h[0]}H{h[1]}", "selected": best,
                       "prior_values_aud": {k: round(v, 2) for k, v in val.items()}})
    # observed value of each candidate inside each half, for reference
    observed = []
    for h in halves:
        mask = half_mask(ref, *h).to_numpy()
        v = {p: period_value(frames[p], mask)["value_aud"] for p in CANDIDATES}
        b = max(v, key=lambda p: (v[p], p == "raw", p))
        observed.append({"half": f"{h[0]}H{h[1]}", "ex_post_best": b,
                         "values_aud": {k: round(x, 2) for k, x in v.items()},
                         "walk_forward_choice": chosen[h],
                         "walk_forward_choice_value_aud": round(v[chosen[h]], 2),
                         "regret_vs_ex_post_aud": round(v[b] - v[chosen[h]], 2)})
    # build the stitched input policy
    a = load_inputs(region, kind="baseline", cutoff=60, policies=["raw"]).sort_values("target").reset_index(drop=True)
    stitched = a.copy()
    for h in halves:
        mask = half_mask(a, *h).to_numpy()
        src = load_inputs(region, kind="baseline", cutoff=60, policies=[chosen[h]]).sort_values("target").reset_index(drop=True)
        stitched.loc[mask, P] = src.loc[mask, P].to_numpy()
    rep = None
    if not dry_run:
        rep = replay(region, stitched, "current_walkforward")
    out = {"region": region, "halves": detail, "observed": observed,
           "stitched_policy": {f"{k[0]}H{k[1]}": v for k, v in chosen.items()},
           "replay_net_value_aud": None if rep is None else rep["net_value_aud"],
           "stitching_note": ("each half supplies the candidate selected from strictly prior data; "
                              "the stitched path is replayed once as a single continuous policy")}
    (V6 / "compute").mkdir(parents=True, exist_ok=True)
    (V6 / "compute" / f"{region}_walkforward.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if rep is not None:
        print(json.dumps({"region": region, "walkforward_value": rep["net_value_aud"],
                          "choices": out["stitched_policy"]}), flush=True)
    return out


# --------------------------------------------------------- decomposition (2x2)
WEIGHTS = {"trading_component_symmetric": np.array([-.5, .5, -.5, .5]),
           "terminal_component_symmetric": np.array([-.5, .5, .5, -.5]),
           "interaction": np.array([1., 1., -1., -1.]),
           "total_history_difference": np.array([-1., 1., 0., 0.])}


def _four_replays(region, history):
    rr = load_traj(region, "raw")
    hh = load_traj(region, history)
    rh = load_traj(region, f"trade_raw__terminal_{history}", scenario="input_2x2")
    hr = load_traj(region, f"trade_{history}__terminal_raw", scenario="input_2x2")
    if any(x[0] is None for x in (rr, hh, rh, hr)):
        return None
    frames = [x[0] for x in (rr, hh, rh, hr)]
    values = np.array([x[1]["net_value_aud"] for x in (rr, hh, rh, hr)])
    for g in frames[1:]:
        assert np.array_equal(frames[0].target.to_numpy(dtype="datetime64[ns]"),
                              g.target.to_numpy(dtype="datetime64[ns]")), f"chronology {region}/{history}"
    return frames, values


def decompose(history):
    rows, parts = [], {}
    for region in REGIONS:
        got = _four_replays(region, history)
        if got is None:
            return None
        frames, values = got
        for name, w in WEIGHTS.items():
            net = sum(a.net_aud * weight for a, weight in zip(frames, w))
            daily = pd.DataFrame({"execution_day": frames[0].execution_day,
                                  "net": net}).groupby("execution_day").net.sum()
            total = float(values @ w)
            ci = paired_block_interval(daily.iloc[1:-1], total)
            rows.append({"history_rule": history, "region": region, "component": name,
                         "full_value_aud": total, "aud_per_nominal_day": total / NOMINAL_DAYS,
                         "block7_ci_low": ci[0], "block7_ci_high": ci[1],
                         "definition": ("four full chronological replays with receding policies; "
                                        "symmetric model-input allocation, not market causal attribution")})
            parts.setdefault(name, []).append((daily, total))
    for name, plist in parts.items():
        if len(plist) != 5:
            continue
        daily = sum(x[0] for x in plist)
        total = sum(x[1] for x in plist)
        ci = paired_block_interval(daily.iloc[1:-1], total)
        rows.append({"history_rule": history, "region": "five_independent_assets_sum",
                     "component": name, "full_value_aud": total,
                     "aud_per_nominal_day": total / NOMINAL_DAYS,
                     "block7_ci_low": ci[0], "block7_ci_high": ci[1],
                     "definition": "joint calendar-date blocks across five independent 1 MW assets"})
    pd.DataFrame(rows).to_csv(DEST / f"v6_decomposition_{history}.csv", index=False)
    return rows


# ----------------------------------------------------------------- sensitivity
def sensitivity_table():
    rows = []
    settings = [("baseline", .91, 5.0, "main"), ("eta", .85, 5.0, "eta0.85"),
                ("eta", .95, 5.0, "eta0.95"), ("kappa", .91, 2.0, "kappa2"),
                ("kappa", .91, 10.0, "kappa10")]
    for region in REGIONS:
        for label, eta, kappa, scenario in settings:
            vals = {}
            for p in ("raw", "sparse_equal", "inverse_lead"):
                f, r = load_traj(region, p, scenario=scenario)
                if f is None:
                    continue
                vals[p] = r["net_value_aud"]
            if len(vals) != 3:
                continue
            rows.append({"region": region, "setting": label, "eta": eta, "kappa_aud_per_grid_mwh": kappa,
                         "scenario": scenario,
                         "value_raw_aud": vals["raw"], "value_sparse_aud": vals["sparse_equal"],
                         "value_inverse_aud": vals["inverse_lead"],
                         "sparse_minus_raw_aud": vals["sparse_equal"] - vals["raw"],
                         "sparse_minus_raw_aud_per_day": (vals["sparse_equal"] - vals["raw"]) / NOMINAL_DAYS,
                         "sparse_minus_raw_aud_per_mw_year": (vals["sparse_equal"] - vals["raw"]) / NOMINAL_DAYS * 365,
                         "inverse_minus_raw_aud_per_day": (vals["inverse_lead"] - vals["raw"]) / NOMINAL_DAYS,
                         "inverse_minus_raw_aud_per_mw_year": (vals["inverse_lead"] - vals["raw"]) / NOMINAL_DAYS * 365})
    d = pd.DataFrame(rows)
    d.to_csv(DEST / "v6_sensitivity_eta_kappa.csv", index=False)
    return rows


# ---------------------------------------------------------------------- probes
def probe_summary():
    ev = ROOT / "results" / "revision_v5" / "events"
    rows = []
    for alternate in ("sparse_equal", "inverse_lead"):
        for region in REGIONS:
            path = ev / f"{region}_{alternate}_rolling_probes.csv"
            if not path.exists():
                return None
            d = pd.read_csv(path)
            for branch in ("history_rolling", "one_history_then_raw"):
                g = d[d.branch == branch]
                rows.append({"alternate": alternate, "region": region, "branch": branch,
                             "probes": len(g),
                             "mean_marked_aud": float(g.marked_delta_aud.mean()),
                             "median_marked_aud": float(g.marked_delta_aud.median()),
                             "min_marked_aud": float(g.marked_delta_aud.min()),
                             "max_marked_aud": float(g.marked_delta_aud.max()),
                             "positive": int((g.marked_delta_aud > .01).sum()),
                             "negative": int((g.marked_delta_aud < -.01).sum()),
                             "material_first_changes": int((g.first_action_delta_mw.abs() > .1).sum())})
            d2 = d[d.branch == "history_rolling"]
            rows.append({"alternate": alternate, "region": region, "branch": "history_rolling_pooled",
                         "probes": len(d2), "mean_marked_aud": float(d2.marked_delta_aud.mean()),
                         "median_marked_aud": float(d2.marked_delta_aud.median()),
                         "min_marked_aud": float(d2.marked_delta_aud.min()),
                         "max_marked_aud": float(d2.marked_delta_aud.max()),
                         "positive": int((d2.marked_delta_aud > .01).sum()),
                         "negative": int((d2.marked_delta_aud < -.01).sum()),
                         "material_first_changes": int((d2.first_action_delta_mw.abs() > .1).sum())})
    d = pd.DataFrame(rows)
    d.to_csv(DEST / "v6_branch_probes.csv", index=False)
    return rows


# --------------------------------------------------------------- common state
def common_state_summary():
    folder = OUT / "diagnostics"
    rows = []
    for region in REGIONS:
        path = folder / f"{region}_raw_inverse_lead_common_state.json"
        parquet = folder / f"{region}_raw_inverse_lead_common_state.parquet"
        if not path.exists():
            return None
        r = json.loads(path.read_text())
        changed = separated = overlap = small = None
        if parquet.exists():
            d = pd.read_parquet(parquet)
            changed = int(d.input_action_change_gt01.sum())
            separated = int(d.loc[d.input_action_change_gt01, "primary_sets_force_change_gt01"].sum())
            overlap = int(d.loc[d.input_action_change_gt01, "near_optimal_ranges_overlap"].sum())
            small = changed - separated - overlap
        rows.append({"region": region, "history": r["history"], "origins": r["origins"],
                     "changed_origins": changed if changed is not None else
                     int(round(r["same_raw_SOC_change_share_gt01"] * r["origins"])),
                     "changed_share": r["same_raw_SOC_change_share_gt01"],
                     "separated_ranges": separated, "overlapping_ranges": overlap,
                     "smaller_separation": small,
                     "forced_share_among_changed": r["forced_share_among_changed"],
                     "overlap_share_among_changed": r["overlap_share_among_changed"]})
    d = pd.DataFrame(rows)
    d.to_csv(DEST / "v6_common_state_inverse.csv", index=False)
    totals = d[["changed_origins", "separated_ranges", "overlapping_ranges",
                "smaller_separation"]].sum()
    NUM_COMMON.clear()
    NUM_COMMON.update({"ranges.il.changed": int(totals.changed_origins),
                       "ranges.il.sep": int(totals.separated_ranges),
                       "ranges.il.overlap": int(totals.overlapping_ranges),
                       "ranges.il.small": int(totals.smaller_separation)})
    (DEST / "v6_common_state_inverse_totals.json").write_text(
        json.dumps(NUM_COMMON, indent=2), encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["selection", "outofsample", "walkforward", "decomp-sparse",
                             "decomp-inverse", "sensitivity", "probes", "commonstate", "all"])
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    t = a.task
    if t in ("selection", "all"):
        print(json.dumps({"selection_stability": "written"} if selection_stability() else {}), flush=True)
    if t in ("outofsample", "all"):
        try:
            out_of_sample()
            print(json.dumps({"out_of_sample": "written"}), flush=True)
        except RuntimeError as exc:
            print(json.dumps({"out_of_sample": "pending", "reason": str(exc)}), flush=True)
    if t in ("decomp-sparse", "all"):
        print(json.dumps({"decomp_sparse": "written" if decompose("sparse_equal") else "pending"}), flush=True)
    if t in ("decomp-inverse", "all"):
        print(json.dumps({"decomp_inverse": "written" if decompose("inverse_lead") else "pending"}), flush=True)
    if t in ("sensitivity", "all"):
        print(json.dumps({"sensitivity": "written" if sensitivity_table() else "pending"}), flush=True)
    if t in ("probes", "all"):
        print(json.dumps({"probes": "written" if probe_summary() else "pending"}), flush=True)
    if t in ("commonstate", "all"):
        print(json.dumps({"common_state": "written" if common_state_summary() else "pending"}), flush=True)
    if t == "walkforward":
        for region in a.regions:
            walkforward(region)


if __name__ == "__main__":
    main()
