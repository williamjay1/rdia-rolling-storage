"""Append-only exploratory economic/precision/branch diagnostics from frozen data.

No controller is changed or rerun and no predictor is fitted. Cost assumptions,
equivalence bands, detection calibration and delayed-sale classification are
post-hoc diagnostics, not preregistered or independent evidence.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
CONTRASTS = ["S_minus_R", "S_minus_Cstar", "IL_minus_Cstar"]
COSTS = [0., 1000., 2000., 5000.]
BANDS = [1000., 2000., 5000.]
BLOCKS = [7, 14, 28]
DRAWS = 10000
SEED = 20261001
SOURCES = {}
CHECKS = []
OUT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def record(rel):
    p = ROOT / rel
    if rel not in SOURCES:
        SOURCES[rel] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha(p)}
    return p


def load_json(rel):
    return json.loads(record(rel).read_text(encoding="utf-8"))


def load_csv(rel):
    return pd.read_csv(record(rel))


def load_parquet(rel, **kwargs):
    return pd.read_parquet(record(rel), **kwargs)


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


def save_csv(name, rows):
    f = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT / name, index=False, float_format="%.12g")
    return f


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]))
    return result


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.minimum.accumulate(q[::-1])[::-1])
    return result


def tail_p(observed, centered):
    hits = int(np.count_nonzero(centered >= observed))
    return (1 + hits) / (len(centered) + 1)


def check(name, value, tolerance):
    CHECKS.append({"name": name, "value": float(value), "tolerance": tolerance,
                   "PASS": bool(value <= tolerance)})
    if value > tolerance:
        raise AssertionError((name, value, tolerance))


def inference(bootstrap_means):
    panel = load_parquet("results/revision_v7/statistics/daily_contrast_panel.parquet")
    days = pd.DatetimeIndex(sorted(panel.execution_day.unique()))
    assert len(days) == 973 and len(panel) == 973 * 5 * 3
    x = np.empty((973, 5, 3))
    totals = np.empty((5, 3))
    selections = {}
    for r, region in enumerate(REGIONS):
        for c, contrast in enumerate(CONTRASTS):
            f = panel[panel.region.eq(region) & panel.contrast.eq(contrast)].set_index("execution_day")
            assert len(f) == 973 and not f.index.duplicated().any()
            x[:, r, c] = f.reindex(days).net_cash_difference_aud.to_numpy(float)
        policy = load_json(f"results/revision_v5/control/main/{region}_current_only_selection.json")["selected_policy"]
        selections[region] = policy
        values = {p: load_json(f"results/revision_v5/control/main/{region}_evaluation_c60_{p}.json")["net_value_aud"]
                  for p in ["raw", "sparse_equal", "inverse_lead", policy]}
        totals[r] = [values["sparse_equal"] - values["raw"],
                     values["sparse_equal"] - values[policy], values["inverse_lead"] - values[policy]]
    assert np.isfinite(x).all()
    fixed = totals - x.sum(axis=0)
    scopes = REGIONS + ["five_assets_sum", "five_assets_equal_capacity_mean"]
    results, precisions, equivalents, costs, breakeven = [], [], [], [], []
    for length in BLOCKS:
        boot, error = bootstrap_means(x, DRAWS, length, SEED)
        check(f"literal_block_index_mean_error_{length}", error, 1e-8)
        sampled = (973 * boot + fixed[None, :, :]) / 974.
        observed = totals / 974.
        scope_boot = np.concatenate([sampled, sampled.sum(axis=1)[:, None, :],
                                     sampled.mean(axis=1)[:, None, :]], axis=1)
        scope_obs = np.vstack([observed, observed.sum(axis=0), observed.mean(axis=0)])
        for ri, scope in enumerate(scopes):
            capacity = 5. if scope == "five_assets_sum" else 1.
            for ci in (1, 2):
                name = CONTRASTS[ci]
                obs = float(scope_obs[ri, ci])
                bs = scope_boot[:, ri, ci]
                centered = bs - obs
                lo, hi = np.quantile(bs, [.025, .975])
                p1 = tail_p(obs, centered)
                p2 = tail_p(abs(obs), abs(centered))
                row = {"scope": scope, "contrast": name, "block_days": length,
                       "interior_dates": 973, "draws": DRAWS, "seed": SEED,
                       "effect_aud_per_nominal_day": obs, "ci95_lo": lo,
                       "ci95_hi": hi, "one_sided_p": p1, "two_sided_p": p2,
                       "capacity_mw_for_cost": capacity,
                       "gross_mechanical_aud_per_mw_year": obs * 365. / capacity}
                results.append(row)
                per_mw_centered = centered / capacity
                per_mw_obs = obs / capacity
                basic90 = [per_mw_obs - np.quantile(per_mw_centered, .95),
                           per_mw_obs - np.quantile(per_mw_centered, .05)]
                precision = {"scope": scope, "contrast": name, "block_days": length,
                             "conditional_bootstrap_se_aud_per_mw_day": float(np.std(per_mw_centered, ddof=1)),
                             "observed_aud_per_mw_day": per_mw_obs,
                             "percentile_ci95_width_aud_per_mw_day": (hi - lo) / capacity,
                             "basic_ci90_lo_aud_per_mw_day": basic90[0],
                             "basic_ci90_hi_aud_per_mw_day": basic90[1]}
                for alpha in (.05, .025):
                    critical = float(np.quantile(per_mw_centered, 1 - alpha))
                    for target, q in [(80, .20), (90, .10)]:
                        mde = critical - float(np.quantile(per_mw_centered, q))
                        precision[f"conditional_shift_mde{target}_alpha{alpha:g}_aud_per_mw_year"] = mde * 365.
                for annual in BANDS:
                    delta = annual / 365.
                    p_lower = tail_p(per_mw_obs + delta, per_mw_centered)
                    p_upper = tail_p(delta - per_mw_obs, -per_mw_centered)
                    p_tost = max(p_lower, p_upper)
                    equivalents.append({"scope": scope, "contrast": name,
                        "block_days": length, "illustrative_positive_margin_aud_per_mw_year": annual,
                        "gross_effect_aud_per_mw_year": per_mw_obs * 365.,
                        "basic90_lo_aud_per_mw_year": basic90[0] * 365.,
                        "basic90_hi_aud_per_mw_year": basic90[1] * 365.,
                        "lower_one_sided_p": p_lower, "upper_one_sided_p": p_upper,
                        "tost_intersection_p": p_tost,
                        "equivalence_at_alpha05": bool(p_tost < .05),
                        "basic90_strictly_inside_positive_width_band": bool(basic90[0] > -delta and basic90[1] < delta),
                        "scope_note": "Exploratory null-centered circular-block TOST; illustrative symmetric gross-value margin, not an elicited indifference band"})
                precisions.append(precision)
                breakeven.append({"scope": scope, "contrast": name, "block_days": length,
                    "point_break_even_aud_per_mw_year": obs * 365. / capacity,
                    "ci95_lo_break_even": lo * 365. / capacity,
                    "ci95_hi_break_even": hi * 365. / capacity,
                    "positive_point_covers_nonnegative_cost": bool(obs > 0)})
                for cost in COSTS:
                    shift = capacity * cost / 365.
                    costs.append({"scope": scope, "contrast": name, "block_days": length,
                        "incremental_archive_cost_aud_per_mw_year": cost,
                        "gross_aud_per_mw_year": obs * 365. / capacity,
                        "net_aud_per_mw_year": obs * 365. / capacity - cost,
                        "net_ci95_lo_aud_per_mw_year": lo * 365. / capacity - cost,
                        "net_ci95_hi_aud_per_mw_year": hi * 365. / capacity - cost,
                        "net_aud_per_scope_day": obs - shift,
                        "net_ci95_lo_aud_per_scope_day": lo - shift,
                        "net_ci95_hi_aud_per_scope_day": hi - shift,
                        "net_point_positive": bool(obs > shift),
                        "conditional_95_interval_positive": bool(lo > shift),
                        "cost_is_measured": False})
    inference_df = pd.DataFrame(results)
    inference_df["holm_primary_two_p"] = np.nan
    for length in BLOCKS:
        take = inference_df.scope.eq("five_assets_sum") & inference_df.block_days.eq(length)
        inference_df.loc[take, "holm_primary_two_p"] = holm(inference_df.loc[take, "one_sided_p"])
    original = load_csv("results/revision_v8/direct_tests/direct_mean_tests.csv")
    for row in inference_df[inference_df.scope.eq("five_assets_sum")].itertuples():
        ref = original[original.accounting.eq("marked_nominal") & original.contrast.eq(row.contrast) & original.block_days.eq(row.block_days)].iloc[0]
        for field, own in [("effect", row.effect_aud_per_nominal_day), ("ci_lo", row.ci95_lo), ("ci_hi", row.ci95_hi), ("one_sided_p", row.one_sided_p), ("holm_two_primary_p", row.holm_primary_two_p)]:
            check(f"frozen_direct_{row.contrast}_{row.block_days}_{field}", abs(own - ref[field]), 2e-8)
    region_df = inference_df[inference_df.scope.isin(REGIONS)].copy()
    for length in BLOCKS:
        idx = region_df.block_days.eq(length)
        region_df.loc[idx, "holm10_one_sided"] = holm(region_df.loc[idx, "one_sided_p"])
        region_df.loc[idx, "holm10_two_sided"] = holm(region_df.loc[idx, "two_sided_p"])
        region_df.loc[idx, "bh10_one_sided_exploratory"] = bh(region_df.loc[idx, "one_sided_p"])
    region_df["holm30_one_sided_all_blocks"] = holm(region_df.one_sided_p)
    region_df["holm30_two_sided_all_blocks"] = holm(region_df.two_sided_p)
    save_csv("block_length_inference.csv", inference_df)
    save_csv("regional_exploratory_multiplicity.csv", region_df)
    save_csv("conditional_precision_and_mde.csv", precisions)
    save_csv("illustrative_cost_width_equivalence.csv", equivalents)
    save_csv("cost_sensitivity.csv", costs)
    save_csv("break_even_costs.csv", breakeven)
    # Separate descriptive annual cash subperiods: no arbitrary mark allocation.
    annual_rows = []
    for year in sorted(set(days.year)):
        take = days.year == year
        yy = x[take]
        for length in BLOCKS:
            boot, error = bootstrap_means(yy, DRAWS, length, SEED)
            check(f"annual_{year}_literal_block_error_{length}", error, 1e-8)
            means = np.concatenate([boot, boot.mean(axis=1)[:, None, :]], axis=1)
            obs = np.vstack([yy.mean(axis=0), yy.mean(axis=(0, 1))])
            for ri, scope in enumerate(REGIONS + ["five_assets_equal_capacity_mean"]):
                for ci in (1, 2):
                    observed = obs[ri, ci]
                    centered = means[:, ri, ci] - observed
                    lo, hi = np.quantile(means[:, ri, ci], [.025, .975])
                    annual_rows.append({"year": int(year), "scope": scope,
                        "contrast": CONTRASTS[ci], "block_days": length,
                        "interior_dates": int(take.sum()), "mean_cash_difference_aud_per_date": observed,
                        "ci95_lo": lo, "ci95_hi": hi,
                        "one_sided_p": tail_p(observed, centered),
                        "two_sided_p": tail_p(abs(observed), abs(centered)),
                        "scope_note": "Observed annual interior-date cash; 2026 Jan-Aug; no year-specific endpoint mark; exploratory"})
    annual_df = pd.DataFrame(annual_rows)
    for length in BLOCKS:
        idx = annual_df.block_days.eq(length)
        annual_df.loc[idx, "holm36_one_sided_same_block"] = holm(annual_df.loc[idx, "one_sided_p"])
        annual_df.loc[idx, "holm36_two_sided_same_block"] = holm(annual_df.loc[idx, "two_sided_p"])
    annual_df["holm108_one_sided_all_blocks"] = holm(annual_df.one_sided_p)
    annual_df["holm108_two_sided_all_blocks"] = holm(annual_df.two_sided_p)
    save_csv("annual_exploratory_multiplicity.csv", annual_df)
    for ci in (1, 2):
        pd.DataFrame(x[:, :, ci], columns=REGIONS).corr().to_csv(OUT / f"regional_daily_{CONTRASTS[ci]}_pearson.csv", float_format="%.12g")
    save_csv("regional_daily_cash_descriptives.csv", [{"region": r, "contrast": CONTRASTS[c],
        "dates": len(x), "mean": float(x[:, j, c].mean()), "sd": float(x[:, j, c].std(ddof=1)),
        "lag1_sample_correlation": float(np.corrcoef(x[1:, j, c], x[:-1, j, c])[0, 1]),
        "p01": float(np.quantile(x[:, j, c], .01)), "p99": float(np.quantile(x[:, j, c], .99)),
        "min": float(x[:, j, c].min()), "max": float(x[:, j, c].max())}
        for j, r in enumerate(REGIONS) for c in (1, 2)])
    return selections, inference_df, pd.DataFrame(costs), pd.DataFrame(precisions)


def branch_diagnostics(selections):
    start_rows, opportunity_rows, case_evidence = [], [], None
    prices, regional_rows, feature_raw_nsw = {}, [], None
    age = load_csv("results/revision_v5/history/verified_history_age_counts.csv")
    freshness = load_csv("results/revision_v5/history/verified_input_freshness.csv")
    metrics = load_csv("results/revision_v5/integrated/aligned_point_forecast_metrics.csv")
    original_probes = load_csv("results/revision_v5/integrated/all_completed_rolling_probes.csv")
    for region in REGIONS:
        base_rel = f"results/revision_v5/history/{region}_baseline_inputs.parquet"
        base = load_parquet(base_rel, filters=[("phase", "==", "evaluation"), ("cutoff_minutes", "==", 60), ("policy", "in", ["raw", "sparse_equal"])])
        frames = {p: base[base.policy.eq(p)].sort_values("target").reset_index(drop=True) for p in ["raw", "sparse_equal"]}
        raw, sparse = frames["raw"], frames["sparse_equal"]
        assert len(raw) == len(sparse) == 46741 and raw.target.equals(sparse.target)
        prices[region] = pd.Series(raw.actual_00.to_numpy(float), index=pd.DatetimeIndex(raw.target), name=region)
        if region == "NSW1":
            feature_raw_nsw = raw.copy()
        actual = raw.actual_00.to_numpy(float)
        rr = {"region": region, "origins": len(raw), "selected_current_policy": selections[region],
            "actual_price_mean": float(actual.mean()), "actual_price_sd": float(actual.std(ddof=0)),
            "actual_price_p99": float(np.quantile(actual, .99)), "actual_price_max": float(actual.max()),
            "negative_price_share": float(np.mean(actual < 0)), "at_least_1000_price_share": float(np.mean(actual >= 1000)),
            "lag1_price_correlation": float(np.corrcoef(actual[1:], actual[:-1])[0, 1]),
            "mean_abs_S_R_curve_difference": float(np.abs(raw[[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float)-sparse[[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float)).mean())}
        for name, policy in [("R", "raw"), ("S", "sparse_equal"), ("IL", "inverse_lead"), ("Cstar", selections[region])]:
            mm = metrics[metrics.region.eq(region) & metrics.policy.eq(policy)]
            assert len(mm) == 1
            rr[f"{name}_curve_rmse"] = float(mm.iloc[0].rmse_all12_aud_per_mwh)
        for row in age[age.region.eq(region) & age.cutoff_minutes.eq(60)].itertuples():
            rr[f"archive_{row.field}_median"] = row.median
        fresh = freshness[freshness.region.eq(region)].iloc[0]
        rr["adjacent_raw_curve_changed_share"] = fresh.raw_curve_changed_share
        rr["adjacent_raw_curve_mean_abs_change"] = fresh.mean_abs_curve_change
        regional_rows.append(rr)
        events = load_csv(f"results/revision_v5/events/{region}_events.csv")
        panel = load_parquet(f"results/revision_v5/events/{region}_event_panel.parquet")
        assert panel.target.equals(raw.target)
        selected = events[events.rolling_probe].sort_values("first_pos")
        assert len(selected) == 20
        eligible = np.flatnonzero(events.first_pos.to_numpy() + 48 <= len(panel))
        selection_indices = eligible[np.unique(np.linspace(0, len(eligible)-1, 20).round().astype(int))]
        check(f"{region}_selected_indices_match_frozen_chronological_event_grid", len(set(events.loc[selection_indices, "event_id"]) ^ set(selected.event_id)), 0)
        paths = load_parquet(f"results/revision_v5/events/{region}_sparse_equal_rolling_probe_paths.parquet")
        for ev in selected.itertuples():
            histories = {p: paths[paths.event_id.eq(ev.event_id) & paths.branch.eq(p)].sort_values("step").reset_index(drop=True)
                         for p in ["raw_rolling", "history_rolling"]}
            r, h = histories["raw_rolling"], histories["history_rolling"]
            assert len(r) == len(h) == 48 and r.target.equals(h.target)
            first = int(ev.first_pos)
            assert raw.target.iloc[first] == r.target.iloc[0]
            check(f"branch_{region}_{ev.event_id}_prices_shared", abs(r.actual_price.to_numpy()-h.actual_price.to_numpy()).max(), 0.)
            cash_delta = float(h.net_cash_aud.sum()-r.net_cash_aud.sum())
            stock_delta = float(h.soc_end.iloc[-1]-r.soc_end.iloc[-1])
            marked_delta = cash_delta + float(r.actual_price.iloc[-1]) * stock_delta
            ref = original_probes[original_probes.region.eq(region) & original_probes.event_id.eq(ev.event_id) & original_probes.branch.eq("history_rolling")].iloc[0]
            check(f"branch_{region}_{ev.event_id}_cash_reconciliation", abs(cash_delta-ref.cash_delta_aud), 2e-7)
            check(f"branch_{region}_{ev.event_id}_marked_reconciliation", abs(marked_delta-ref.marked_delta_aud), 2e-7)
            opportunities = []
            for step in range(47):
                # Diagnostic scope: same entering inventory, raw sells more,
                # sparse expects a later trading-horizon price peak, and sparse
                # subsequently sells at a lower realized price. The total
                # window loss is a separate flag, not a causal trade attribution.
                if abs(float(r.soc_start.iloc[step]-h.soc_start.iloc[step])) > .01:
                    continue
                discharge_gap = float(r.discharge_mw.iloc[step]-h.discharge_mw.iloc[step])
                retained = float(h.soc_end.iloc[step]-r.soc_end.iloc[step])
                source_row = sparse.iloc[first+step]
                future_peak = float(source_row[[f"p_{j:02d}" for j in range(2, 9)]].max())
                current_forecast = float(source_row.p_01)
                sale_step = None
                for later in range(step+1, min(step+8, 48)):
                    if h.discharge_mw.iloc[later] > .1:
                        sale_step = later
                        break
                actual_current = float(r.actual_price.iloc[step])
                if sale_step is None:
                    continue
                actual_sale = float(h.actual_price.iloc[sale_step])
                if (discharge_gap > .05 and retained > .025 and actual_current > 0
                        and future_peak > current_forecast + 5 and actual_sale < actual_current - 5):
                    opportunity = {"region": region, "event_id": int(ev.event_id), "step": step,
                        "target": str(r.target.iloc[step]), "next_sparse_sale_step": sale_step,
                        "entering_soc_raw": float(r.soc_start.iloc[step]), "entering_soc_sparse": float(h.soc_start.iloc[step]),
                        "same_state_gap_mwh": float(abs(r.soc_start.iloc[step]-h.soc_start.iloc[step])),
                        "raw_minus_sparse_discharge_mw": discharge_gap, "retained_inventory_mwh": retained,
                        "sparse_current_forecast": current_forecast, "sparse_future_peak_trading_block": future_peak,
                        "current_actual_price": actual_current, "later_sale_actual_price": actual_sale,
                        "cash_delta_aud_window": cash_delta, "marked_delta_aud_window": marked_delta,
                        "window_loss_gt1aud": bool(marked_delta < -1.)}
                    opportunities.append(opportunity)
                    opportunity_rows.append(opportunity)
            sf = panel.iloc[first]
            curve_delta = sparse.iloc[first][[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float) - raw.iloc[first][[f"p_{j:02d}" for j in range(1, 13)]].to_numpy(float)
            sr = {"region": region, "event_id": int(ev.event_id), "event_start_execution": str(ev.start),
                  "first_decision_target": str(r.target.iloc[0]), "last_decision_target": str(r.target.iloc[-1]),
                  "first_pos": first, "initial_soc_mwh": float(r.soc_start.iloc[0]),
                  "actual_current_price": float(r.actual_price.iloc[0]),
                  "raw_current_forecast": float(raw.p_01.iloc[first]), "sparse_current_forecast": float(sparse.p_01.iloc[first]),
                  "raw_forecast_peak_first8": float(sf.forecast_peak), "raw_forecast_range_first8": float(sf.forecast_range),
                  "revision_absolute_mean": float(sf.revision_absolute_mean),
                  "S_R_mean_abs_curve_gap_first12": float(abs(curve_delta).mean()),
                  "first_trigger_peak": bool(ev.first_trigger_peak), "first_trigger_range": bool(ev.first_trigger_range),
                  "first_trigger_revision": bool(ev.first_trigger_revision), "cash_delta_aud": cash_delta,
                  "marked_delta_aud": marked_delta, "final_inventory_difference_mwh": stock_delta,
                  "opportunities_above005mw": len(opportunities),
                  "window_loss_gt1aud": bool(marked_delta < -1.)}
            for threshold in [.05, .1, .2]:
                qualifying = [q for q in opportunities if q["raw_minus_sparse_discharge_mw"] > threshold
                              and q["retained_inventory_mwh"] > threshold * .5]
                sr[f"delay_opportunity_gt{threshold:g}mw"] = bool(qualifying)
                sr[f"delayed_sale_failure_gt{threshold:g}mw"] = bool(qualifying and marked_delta < -1.)
                if threshold == .1:
                    sr["first_delay_step"] = qualifying[0]["step"] if qualifying else np.nan
                    sr["first_delay_target"] = qualifying[0]["target"] if qualifying else ""
            start_rows.append(sr)
            if region == "SA1" and ev.event_id == 178:
                step = int(np.flatnonzero(r.target.eq(pd.Timestamp("2024-09-23 19:00:00")))[0])
                source_r, source_h = raw.iloc[first+step], sparse.iloc[first+step]
                fig = load_json("figures/revision_v11/figure_5_decision_failure_source.json")
                case_evidence = {"source_row_identity": {"region":"SA1", "phase":"evaluation", "cutoff_minutes":60,
                    "target":"2024-09-23 19:00:00", "branch_event_id":178, "branch_step":step},
                    "R_p01": float(source_r.p_01), "S_p01": float(source_h.p_01), "actual00": float(source_r.actual_00),
                    "R_discharge_mw": float(r.discharge_mw.iloc[step]), "S_discharge_mw": float(h.discharge_mw.iloc[step]),
                    "R_soc_start": float(r.soc_start.iloc[step]), "S_soc_start": float(h.soc_start.iloc[step]),
                    "cash_delta_aud":cash_delta,"marked_delta_aud":marked_delta,"end_inventory_delta_mwh":stock_delta,
                    "figure_source_values":{"R_p01":fig["source_data"]["R"][0],"S_p01":fig["source_data"]["S"][0],"actual00":fig["source_data"]["current_realized"]},
                    "source_manifest_entries":[SOURCES[base_rel],SOURCES[f"results/revision_v5/events/{region}_sparse_equal_rolling_probe_paths.parquet"],SOURCES["figures/revision_v11/figure_5_decision_failure_source.json"]]}
                for name in ["R_p01","S_p01","actual00"]:
                    check(f"SA_illustration_{name}_figure_vs_parquet",abs(case_evidence[name]-case_evidence["figure_source_values"][name]),0.)
    starts = save_csv("branch_start_features_100.csv", start_rows)
    assert len(starts) == 100
    save_csv("delayed_sale_opportunities_all.csv", opportunity_rows)
    frequency = []
    for scope in REGIONS + ["all_100_selected"]:
        d = starts if scope == "all_100_selected" else starts[starts.region.eq(scope)]
        for threshold in [.05,.1,.2]:
            fails = d[d[f"delayed_sale_failure_gt{threshold:g}mw"]]
            opportunities = d[f"delay_opportunity_gt{threshold:g}mw"]
            frequency.append({"scope":scope,"discharge_gap_threshold_mw":threshold,"selected_windows":len(d),
                "delay_opportunity_windows":int(opportunities.sum()),"failure_windows":len(fails),
                "failure_share_of_selected_windows":len(fails)/len(d),
                "sum_marked_delta_in_failed_windows_aud":float(fails.marked_delta_aud.sum()),
                "mean_marked_delta_in_failed_windows_aud":float(fails.marked_delta_aud.mean()) if len(fails) else np.nan,
                "median_marked_delta_in_failed_windows_aud":float(fails.marked_delta_aud.median()) if len(fails) else np.nan,
                "worst_marked_delta_in_failed_windows_aud":float(fails.marked_delta_aud.min()) if len(fails) else np.nan,
                "all_negative_window_count_gt1aud":int(d.window_loss_gt1aud.sum()),
                "cash_sum_in_failed_windows_aud":float(fails.cash_delta_aud.sum()),
                "scope_note":"Post-hoc same-SOC diagnostic on fixed nonrandom starts; window losses are not losses causally attributable only to the flagged sale"})
    frequencies = save_csv("delayed_sale_frequency_by_region.csv", frequency)
    save_json("delayed_sale_operational_definition.json", {
        "status":"EXPLORATORY_POST_HOC_DIAGNOSTIC", "primary_discharge_gap_mw":.1,
        "same_entering_soc_absolute_difference_max_mwh":.01,
        "retained_inventory_min_mwh":.05,"current_actual_price_min_exclusive":0.,
        "sparse_expected_future_p02_p08_above_p01_min_aud_mwh":5.,
        "future_sale":"first sparse discharge >0.1 MW among next seven observed origins within the 48-origin window",
        "future_realized_sale_price_below_current_min_aud_mwh":5.,
        "failure_requires_whole_window_marked_delta_less_than_aud":-1.,
        "sensitivity_discharge_gaps_mw":[.05,.1,.2],
        "no_population_prevalence_or_causal_trade_loss_claim":True,
        "selection_rule":"Pre-2023 componentwise forecast peak/range/absolute-revision q95; triggers separated by >4h make new events; eligible events have 48 remaining origins; 20 rounded linspace indices across ordered eligible events per region. No realized-profit selection.",
        "opportunities_do_not_cover":"Different-SOC decisions, postponements with no later material sale in next seven origins, nonpositive-price dispatch, or other reasons for negative window value."})
    save_json("SA_case_source_row_verification.json", case_evidence)
    price_panel = pd.concat(prices.values(), axis=1)
    assert len(price_panel) == 46741 and not price_panel.isna().any().any()
    price_panel.corr().to_csv(OUT/"regional_price_pearson.csv",float_format="%.12g")
    price_panel.corr(method="spearman").to_csv(OUT/"regional_price_spearman.csv",float_format="%.12g")
    region_df = save_csv("regional_market_and_forecast_descriptives.csv", regional_rows)
    save_csv("regional_branch_start_ranges.csv",starts.groupby("region").agg(
        starts=("event_id","size"),first_target=("first_decision_target","min"),last_target=("first_decision_target","max"),
        min_initial_soc=("initial_soc_mwh","min"),max_initial_soc=("initial_soc_mwh","max"),
        min_current_price=("actual_current_price","min"),max_current_price=("actual_current_price","max"),
        median_forecast_revision=("revision_absolute_mean","median")).reset_index())
    return starts, frequencies, region_df, feature_raw_nsw


def spo_price_bands(raw, selections):
    targets = raw.target.to_numpy()
    truth = raw[[f"actual_{j:02d}" for j in range(12)]].to_numpy(float)
    predictions = {"R":raw[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)}
    cstar = load_parquet("results/revision_v5/history/NSW1_baseline_inputs.parquet",filters=[("phase","==","evaluation"),("cutoff_minutes","==",60),("policy","==",selections["NSW1"])])
    cstar = cstar.sort_values("target").reset_index(drop=True)
    assert np.array_equal(cstar.target.to_numpy(),targets)
    predictions["Cstar"] = cstar[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)
    reports = {}
    for name, family in [("Ridge","ridge"),("SPO","spo")]:
        f=load_parquet(f"results/revision_v8/spo/{family}_evaluation_forecasts.parquet").sort_values("target").reset_index(drop=True)
        assert np.array_equal(f.target.to_numpy(),targets)
        predictions[name]=f[[f"p_{j:02d}" for j in range(1,13)]].to_numpy(float)
        reports[name]=load_json(f"results/revision_v8/spo/{family}_evaluation_report.json")
    labels=["negative (<0)","ordinary [0,100)","elevated [100,1000)","extreme >=1000"]
    bins = np.digitize(truth,[0.,100.,1000.])
    rows=[]
    for name,pred in predictions.items():
        error=pred-truth
        if name in reports:
            check(f"{name}_overall_rmse_vs_frozen_report",abs(np.sqrt(np.mean(error**2))-reports[name]["forecast_price_rmse"]),2e-6)
            check(f"{name}_overall_mae_vs_frozen_report",abs(np.abs(error).mean()-reports[name]["forecast_price_mae"]),2e-6)
        total_squared_error=float(np.sum(error**2))
        for band,label in enumerate(labels):
            take=bins==band;d=error[take]
            rows.append({"policy":name,"realized_horizon_price_band":label,"forecast_target_cells":int(take.sum()),
                "origins_with_any_target_in_band":int(take.any(axis=1).sum()),"horizons":12,
                "cell_share":float(take.mean()),"mae_aud_mwh":float(abs(d).mean()),
                "rmse_aud_mwh":float(np.sqrt(np.mean(d*d))),"bias_aud_mwh":float(d.mean()),
                "share_of_policy_total_squared_error":float(np.sum(d*d)/total_squared_error),
                "scope":"Retrospective descriptive binning by each horizon actual price; 12 overlapping forecast targets per origin are not independent observations"})
    band_df=save_csv("spo_forecast_error_by_realized_price_band.csv",rows)
    local=load_csv("results/revision_v8/spo/evaluation_local_regret_by_origin.csv")
    assert np.array_equal(pd.to_datetime(local.target).to_numpy(),targets)
    current_bins=np.digitize(truth[:,0],[0.,100.,1000.]);origin_rows=[]
    for name,pred in predictions.items():
        if name in ["R","Cstar"]:
            pol="raw" if name=="R" else selections["NSW1"]
            traj=load_parquet(f"results/revision_v5/control/main/NSW1_evaluation_c60_{pol}.parquet")
        else:traj=load_parquet(f"results/revision_v8/spo/{name.lower()}_evaluation_trajectory.parquet")
        traj=traj.sort_values("target").reset_index(drop=True)
        assert np.array_equal(traj.target.to_numpy(),targets)
        for band,label in enumerate(labels):
            take=current_bins==band
            e=pred[take]-truth[take]
            origin_rows.append({"policy":name,"current_actual_price_band":label,"origins":int(take.sum()),
                "all12_curve_rmse_for_origins_aud_mwh":float(np.sqrt(np.mean(e*e))),
                "fixed_soc_local_reference_regret_aud_per_origin":float(local.loc[take,name].mean()),
                "rolling_net_cash_aud_per_origin":float(traj.loc[take,"net_aud"].mean()),
                "rolling_discharge_mw_mean":float(traj.loc[take,"discharge_mw"].mean()),
                "rolling_charge_mw_mean":float(traj.loc[take,"charge_mw"].mean()),
                "scope":"Current-realized-price retrospective stratum; local regret at fixed SOC differs from sequential rolling cash; no conditioning used to train/select models"})
    origin_df=save_csv("spo_local_regret_and_cash_by_current_price_band.csv",origin_rows)
    save_json("spo_price_band_definition.json",{"status":"EXPLORATORY_DESCRIPTIVE_ONLY","market":"NSW","origins":46741,
        "cells":int(truth.size),"bands_actual_aud_mwh":["<0","[0,100)","[100,1000)",">=1000"],
        "no_new_training_or_selection":True,"forecast_error_table_bins":"Each horizon actual value",
        "regret_cash_table_bins":"Contemporaneous actual_00 value","no_error_mechanism_or_causal_improvement_inferred":True})
    return band_df,origin_df


def tex_number(v, decimals=0):
    return f"{v:,.{decimals}f}".replace(",",r"\,")


def tex_fragments(costs, precision, frequencies, regions, bands):
    parts=[r"% Append-only v13 exploratory tables, generated from frozen sources.",
        r"\subsection{Exploratory costs, precision, and update-window diagnostics}",
        r"All diagnostics in this subsection were added after the original outcomes were known. Costs are illustrative incremental archive costs above the current-only alternative, not measured expenses. Mechanical annualization uses 365 days per 1 MW; it does not predict future profit. The five-asset mean gives equal weight to five independently controlled 1 MW assets, while simultaneous block resampling preserves their observed cross-region dependence.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Illustrative net archive value after incremental annual costs (A\$/MW-year). Values are marked historical point contrasts mechanically annualized; intervals translate by the same fixed cost.}",
        r"\begin{tabular}{llrrrr}\toprule Scope & Rule versus $C^*$ & Cost 0 & 1,000 & 2,000 & 5,000 \\\midrule"]
    for scope in REGIONS+["five_assets_equal_capacity_mean"]:
        for contrast in ["S_minus_Cstar","IL_minus_Cstar"]:
            d=costs[costs.scope.eq(scope)&costs.contrast.eq(contrast)&costs.block_days.eq(7)].sort_values("incremental_archive_cost_aud_per_mw_year")
            label=scope.replace("1","") if scope in REGIONS else "Five-asset mean"
            rule="$S$" if contrast.startswith("S_") else "$IL$"
            parts.append(label+" & "+rule+" & "+" & ".join(tex_number(v) for v in d.net_aud_per_mw_year)+r" \\")
    parts +=[r"\bottomrule\end{tabular}",r"\label{tab:v13-cost-sensitivity}\end{table}",
        r"The gross point break-even cost is the corresponding cost-zero entry. A negative break-even point covers no nonnegative incremental archive cost. The complete CSVs retain unrounded values and 7-, 14-, and 28-day conditional intervals; neither a positive point nor mechanical annualization establishes a commercial return.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Conditional precision and detectability on the observed block-resampling law. MDEs are shift calibrations at one-sided $\alpha=0.05$ and 80\% detection, not observed power or independent evidence.}",
        r"\begin{tabular}{lrrrr}\toprule Rule versus $C^*$ & Block & SE (A\$/MW-day) & MDE80 (A\$/MW-year) & 90\% basic interval (A\$/MW-year) \\\midrule"]
    for row in precision[precision.scope.eq("five_assets_equal_capacity_mean")].itertuples():
        rule="$S$" if row.contrast.startswith("S_") else "$IL$"
        # Access dotted-alpha column names through the original DataFrame row.
        q=precision[(precision.scope==row.scope)&(precision.contrast==row.contrast)&(precision.block_days==row.block_days)].iloc[0]
        parts.append(f"{rule} & {row.block_days} & {tex_number(row.conditional_bootstrap_se_aud_per_mw_day,2)} & {tex_number(q['conditional_shift_mde80_alpha0.05_aud_per_mw_year'])} & [{tex_number(row.basic_ci90_lo_aud_per_mw_day*365)}, {tex_number(row.basic_ci90_hi_aud_per_mw_day*365)}]"+r" \\")
    parts +=[r"\bottomrule\end{tabular}\label{tab:v13-precision}\end{table}",
        r"For centered paired bootstrap error $e_b$, the conditional shift required for target detection $1-\beta$ is $q_{1-\alpha}(e_b)-q_{\beta}(e_b)$. This keeps the observed dependence and tails fixed; it is not a prospective sample-size guarantee. Exploratory equivalence uses two one-sided null-centered bootstrap tests at $\alpha=0.05$ with illustrative positive-width bands of $\pm$A\$1,000, $\pm$A\$2,000, and $\pm$A\$5,000/MW-year. These bands were not elicited or preregistered. Nonrejection of superiority is never treated as equivalence.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Post-hoc delayed-sale diagnostic in 100 fixed, nonrandom sparse update windows. A flag requires the operational conditions defined below and a negative marked whole-window difference exceeding A\$1. Losses are whole-window differences, not causal losses attributed only to one postponed sale.}",
        r"\begin{tabular}{lrrrr}\toprule Region & Windows & Delay opportunities & Failed windows & Mean marked difference in failures (A\$) \\\midrule"]
    for row in frequencies[frequencies.discharge_gap_threshold_mw.eq(.1)].itertuples():
        label=row.scope.replace("1","") if row.scope in REGIONS else "All selected"
        mean="--" if not np.isfinite(row.mean_marked_delta_in_failed_windows_aud) else tex_number(row.mean_marked_delta_in_failed_windows_aud,2)
        parts.append(f"{label} & {row.selected_windows} & {row.delay_opportunity_windows} & {row.failure_windows} & {mean}"+r" \\")
    parts +=[r"\bottomrule\end{tabular}\label{tab:v13-delay-frequency}\end{table}",
        r"At a flagged within-window origin, entering inventories differ by at most 0.01 MWh, R discharges over 0.1 MW more, and S retains over 0.05 MWh more. S's maximum issued price over horizons 2--8 exceeds its horizon-1 price by over A\$5/MWh. Its first subsequent discharge above 0.1 MW within the next seven observed origins occurs at a realized price at least A\$5/MWh lower than the positive current realized price. Failure additionally requires a negative marked 48-origin window contrast below A\$-1. Thresholds 0.05 and 0.2 MW are retained as descriptive sensitivities. This restrictive post-hoc rule omits different-state cases and cannot estimate market-wide prevalence.",
        r"Selected event starts come from pre-2023 forecast-only 95th-percentile peak, range, or revision triggers. Consecutive triggers no more than four hours apart form events; eligibility requires 48 remaining origins. Twenty rounded equally spaced indices across the chronologically ordered eligible events are chosen in each region. Selection uses no realized profit. The 100-row start-feature file exposes dates, entering stock, forecast discrepancies, trigger flags, and outcomes; no randomized or representative sampling claim is made.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{NSW forecast errors by realized target-price band. Each origin contributes twelve overlapping horizon targets; counts are cells, not independent observations. No refitting or evaluation-based selection is performed.}",
        r"\begin{tabular}{llrrrr}\toprule Policy & Realized price band (A\$/MWh) & Cells & MAE & RMSE & Squared-error share \\\midrule"]
    for row in bands[bands.policy.isin(["Ridge","SPO"])].itertuples():
        label=row.realized_horizon_price_band.replace(">=",r"$\geq$").replace("<",r"$<$")
        parts.append(f"{row.policy} & {label} & {tex_number(row.forecast_target_cells)} & {tex_number(row.mae_aud_mwh,2)} & {tex_number(row.rmse_aud_mwh,2)} & {100*row.share_of_policy_total_squared_error:.1f}\\%"+r" \\")
    parts +=[r"\bottomrule\end{tabular}\label{tab:v13-spo-price-errors}\end{table}",
        r"Regional feature and correlation matrices are descriptive. Price and daily-contrast correlations use common chronological rows, not five between-region observations. They preserve exposure dependence but neither explain TAS/SA causally nor identify omitted market structure. Market concentration and realized archive implementation costs are unobserved."]
    (OUT/"supplement_economic_diagnostics.tex").write_text("\n".join(parts)+"\n",encoding="utf-8")


def main():
    global ROOT,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args();ROOT=args.root.resolve(strict=True)
    OUT=(args.output_dir or ROOT/"results/revision_v13/economic").resolve()
    if os.name=="nt" and OUT.drive.upper()!="D:":raise ValueError("Computation outputs must use D:")
    if (OUT/"execution_receipt.json").exists():raise FileExistsError("Preserve completed diagnostics; choose another fresh output folder")
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    sys.path.insert(0,str(ROOT/"scripts"))
    statistics=importlib.import_module("revision_v7_statistics")
    record("scripts/revision_v7_statistics.py")
    selections,infer,costs,precision=inference(statistics.bootstrap_means)
    starts,frequency,regional,raw_nsw=branch_diagnostics(selections)
    bands,origin_bands=spo_price_bands(raw_nsw,selections)
    tex_fragments(costs,precision,frequency,regional,bands)
    save_json("methods_and_scope.json",{
        "status":"APPENDED_EXPLORATORY_FROZEN_DATA_DIAGNOSTICS", "draws":DRAWS,"seed":SEED,"block_days":BLOCKS,
        "primary_nominal_exposure_days":974,"paired_interior_dates":973,
        "marked_draws_formula":"(973 * paired bootstrap interior-date mean + fixed historical boundaries/endpoint differences)/974",
        "cost_assumptions_aud_per_mw_year":COSTS,"cost_components_unmeasured":["incremental storage","data processing","monitoring","implementation annual equivalent"],
        "cost_is_incremental_over_current_only":True,"mechanical_annualization_days":365,
        "all_assets_power_mw":1.,"equal_capacity_average_divisor":5.,
        "precision":"Conditional bootstrap SE and shift MDE; no observed-power argument, no prospective sample-size or adoption guarantee",
        "mde_formula":"quantile(centered_error,1-alpha)-quantile(centered_error,1-target_detection)",
        "equivalence":"Exploratory gross-contrast TOST at illustrative positive-width +/-1000/2000/5000 AUD/MW-year. Centered bootstrap basic 90% interval is the corresponding approximate dual; percentile 95% primary intervals remain unchanged.",
        "regional_multiplicity":"10 regional one-sided/two-sided hypotheses per block: Holm; BH descriptive secondary; 30 all-block Holm also retained. These do not correct original outcome-informed research development.",
        "annual_multiplicity":"36 region/equal-mean x 2 contrast x 3 year hypotheses per block, and 108 across blocks; exploratory Holm. Annual cash excludes endpoint marks and 2026 is Jan-Aug.",
        "branch_scope":"100 forecast-defined but nonrandom, chronological-event-grid starts; post-hoc operational classifier; negative whole-window differences are not single-trade causal loss estimates",
        "no_controller_run_or_model_training":True,"no_original_files_modified":True,
        "supply_raw_material":"All newly analyzed inputs are existing frozen outputs; no newly downloaded data."})
    output_rows=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="execution_receipt.json"]
    receipt={"status":"PASS","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"elapsed_seconds":time.perf_counter()-start,
        "script_path":"scripts/revision_v13_economic_diagnostics.py","script_sha256":sha(Path(__file__)),"input_sources":list(SOURCES.values()),
        "numeric_checks":CHECKS,"all_checks_pass":all(x["PASS"] for x in CHECKS),"output_manifest":output_rows,
        "counts":{"cost_rows":len(costs),"inference_rows":len(infer),"precision_rows":len(precision),"selected_branch_starts":len(starts),"error_band_rows":len(bands),"regret_cash_band_rows":len(origin_bands)},
        "five_asset_equal_mean_cost0":costs[costs.scope.eq("five_assets_equal_capacity_mean")&costs.block_days.eq(7)&costs.incremental_archive_cost_aud_per_mw_year.eq(0)].to_dict(orient="records"),
        "delay_frequency_primary":frequency[frequency.discharge_gap_threshold_mw.eq(.1)].astype(object).where(pd.notna(frequency),None).to_dict(orient="records"),
        "research_scope":"All appended analyses are exploratory; frozen controller/forecast outputs only; no raw clock shift or new random branch replication."}
    save_json("execution_receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ["status","elapsed_seconds","counts","five_asset_equal_mean_cost0","delay_frequency_primary"]},indent=2,allow_nan=False))


if __name__=="__main__":main()
