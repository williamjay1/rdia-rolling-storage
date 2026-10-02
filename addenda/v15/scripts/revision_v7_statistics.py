"""Read-only v7 statistics on frozen chronological storage trajectories.

No optimiser, model fitting or pre-v7 result writer is imported or called.
Inference conditions on the five named assets, saved policies and realised SOC
paths. The finite-family Reality Check does not correct earlier research choices.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/revision_v5/control/main"
DEST = ROOT / "results/revision_v7/statistics"
REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
CANDIDATES = ["raw", "current_shrink_a025", "current_shrink_a050",
              "current_smooth_a025", "current_smooth_a050"]
POLICIES = CANDIDATES + ["sparse_equal", "inverse_lead"]
PERIODS = {"2023_Q1": "2023-04-01", "2023_H1": "2023-07-01",
           "2023_full": "2024-01-01"}
CONTRASTS = ["S_minus_R", "S_minus_Cstar", "IL_minus_Cstar"]
COLS = ["region", "target", "actual_price", "execution_start", "execution_day",
        "soc_start", "charge_mw", "discharge_mw", "soc_end", "cashflow_aud",
        "degradation_aud", "net_aud"]
MANIFEST: dict[str, dict] = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    p = Path(path)
    rel = str(p.relative_to(ROOT)).replace("\\", "/")
    if rel not in MANIFEST:
        MANIFEST[rel] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha(p)}


def write_json(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(DEST / name, index=False, float_format="%.12g")


def load(region, policy, phase):
    path = SOURCE / f"{region}_{phase}_c60_{policy}.parquet"
    jp = path.with_suffix(".json")
    record(path); record(jp)
    rep = json.loads(jp.read_text(encoding="utf-8"))
    assert rep['policy_definition']['eta_c']==rep['policy_definition']['eta_d']==.91
    assert rep['policy_definition']['kappa_aud_per_grid_mwh']==5.0
    f = pd.read_parquet(path, columns=COLS).sort_values("target").reset_index(drop=True)
    for c in ("target", "execution_start", "execution_day"):
        f[c] = f[c].astype("datetime64[ns]")
    n = 17507 if phase == "validation" else 46741
    assert len(f) == rep["intervals"] == n, (region, phase, policy, len(f))
    assert not f.target.duplicated().any() and f.region.eq(region).all()
    assert np.isfinite(f.select_dtypes(include=["number"]).to_numpy()).all()
    assert (f.target - f.execution_start).eq(pd.Timedelta(minutes=30)).all()
    assert f.execution_start.dt.normalize().equals(f.execution_day)
    cash = .5 * f.actual_price.to_numpy() * (f.discharge_mw - f.charge_mw).to_numpy()
    wear = 2.5 * (f.charge_mw + f.discharge_mw).to_numpy()
    assert np.max(np.abs(cash - f.cashflow_aud.to_numpy())) < 1e-8
    assert np.max(np.abs(wear - f.degradation_aud.to_numpy())) < 1e-8
    assert np.max(np.abs(cash - wear - f.net_aud.to_numpy())) < 1e-8
    balance = f.soc_end - f.soc_start - .455 * f.charge_mw + .5 / .91 * f.discharge_mw
    assert abs(balance).max() < 1e-6
    assert np.max(np.abs(f.soc_start.to_numpy()[1:] - f.soc_end.to_numpy()[:-1])) < 1e-6
    mark = float(f.soc_end.iloc[-1] * f.actual_price.iloc[-1]
                 - f.soc_start.iloc[0] * f.actual_price.iloc[0])
    value = float(f.net_aud.sum() + mark)
    assert abs(value - float(rep["net_value_aud"])) < 2e-6
    assert abs(mark - float(rep["endpoint_mark_aud"])) < 2e-6
    return f, value


def bootstrap_means(x, draws, length, seed):
    """Synchronous circular moving blocks, exact n including final short block.

    Each sampled time index applies to every region/policy column. Prefix block
    sums save memory; independent literal-index checks audit their arithmetic.
    """
    n = len(x)
    flat = np.asarray(x, dtype=np.float64).reshape(n, -1)
    extension = np.concatenate([flat, flat[:length]], axis=0)
    prefix = np.concatenate([np.zeros((1, flat.shape[1])), np.cumsum(extension, axis=0)])
    q, rem = divmod(n, length)
    rng = np.random.default_rng(seed)
    means = np.empty((draws, flat.shape[1]))
    max_literal_error = 0.0
    for lo in range(0, draws, 64):
        hi = min(lo + 64, draws)
        starts = rng.integers(0, n, size=(hi - lo, q + bool(rem)))
        sums = (prefix[starts[:, :q] + length] - prefix[starts[:, :q]]).sum(axis=1)
        if rem:
            sums += prefix[starts[:, -1] + rem] - prefix[starts[:, -1]]
        means[lo:hi] = sums / n
        if lo == 0:
            for j in range(min(3, hi)):
                index = np.concatenate([(s + np.arange(length)) % n
                                        for s in starts[j, :q]])
                if rem:
                    index = np.r_[index, (starts[j, -1] + np.arange(rem)) % n]
                direct = flat[index].mean(axis=0)
                max_literal_error = max(max_literal_error,
                                        float(np.max(np.abs(direct - means[j]))))
    assert max_literal_error < 1e-8, max_literal_error
    return means.reshape((draws,) + x.shape[1:]), max_literal_error


def pvalue(observed, simulations):
    hits = int(np.count_nonzero(np.asarray(simulations) >= observed))
    p = (1 + hits) / (len(simulations) + 1)
    return p, math.sqrt(p * (1 - p) / (len(simulations) + 1)), hits


def reality_checks(x, boot, phase, length, draws, seed):
    n = len(x); factor = math.sqrt(n)
    gains = x[:, :, 1:5] - x[:, :, :1]
    mu = gains.mean(axis=0)
    bg = boot[:, :, 1:5] - boot[:, :, :1]
    centered = bg - mu[None, :, :]
    rows = []
    for ri, region in enumerate(REGIONS):
        observed = factor * max(0.0, float(mu[ri].max()))
        bs = factor * np.maximum(0, centered[:, ri, :].max(axis=1))
        pv, mcse, hits = pvalue(observed, bs)
        best = int(mu[ri].argmax())
        rows.append(dict(phase=phase, scope=region, family="four_processed_vs_raw",
                         candidate_count=4, raw_zero_in_max=True, interior_days=n,
                         block_days=length, bootstrap_draws=draws, seed=seed,
                         best_rule=CANDIDATES[best + 1], best_mean_daily_gain_aud=float(mu[ri, best]),
                         statistic=observed, p_value=pv, monte_carlo_se=mcse,
                         exceedances=hits))
    pmu = mu.sum(axis=0)
    pb = centered.sum(axis=1)
    observed = factor * max(0.0, float(pmu.max()))
    bs = factor * np.maximum(0, pb.max(axis=1))
    pv, mcse, hits = pvalue(observed, bs)
    rows.append(dict(phase=phase, scope="five_assets_sum", family="four_uniform_rules_vs_raw",
                     candidate_count=4, raw_zero_in_max=True, interior_days=n,
                     block_days=length, bootstrap_draws=draws, seed=seed,
                     best_rule=CANDIDATES[int(pmu.argmax()) + 1],
                     best_mean_daily_gain_aud=float(pmu.max()), statistic=observed,
                     p_value=pv, monte_carlo_se=mcse, exceedances=hits))
    # Full regional rule family has 5^5 portfolios including all-raw. Its maximum
    # separates algebraically by region. This includes the original mixed C*.
    observed = factor * np.maximum(0, mu.max(axis=1)).sum()
    bs = factor * np.maximum(0, centered.max(axis=2)).sum(axis=1)
    # Independent finite enumeration verifies the separable 3,125-family maximum.
    combinations = np.asarray(list(itertools.product(range(5), repeat=5)))
    region_index = np.arange(5)[None, :]
    aug = np.column_stack([np.zeros(5), mu])
    enumerated = aug[region_index, combinations].sum(axis=1).max()
    assert abs(enumerated * factor - observed) < 1e-8
    for j in range(3):
        augb = np.column_stack([np.zeros(5), centered[j]])
        literal = augb[region_index, combinations].sum(axis=1).max()
        assert abs(literal * factor - bs[j]) < 1e-8
    pv, mcse, hits = pvalue(float(observed), bs)
    rows.append(dict(phase=phase, scope="five_assets_sum", family="all_3125_regional_rule_combinations",
                     candidate_count=3124, raw_zero_in_max=True, interior_days=n,
                     block_days=length, bootstrap_draws=draws, seed=seed,
                     best_rule="region_specific_ex_post_max_including_raw",
                     best_mean_daily_gain_aud=float(observed / factor), statistic=float(observed),
                     p_value=pv, monte_carlo_se=mcse, exceedances=hits))
    # A separately named six-test correction protects the reporting of all five
    # regional plus uniform-portfolio RC tests. Combination-family row separate.
    vals = np.asarray([v["p_value"] for v in rows[:6]])
    order = np.argsort(vals)
    holm = np.maximum.accumulate(np.minimum(1.0, vals[order] * (6 - np.arange(6))))
    for rank, j in enumerate(order):
        rows[int(j)]["holm_p_across_6_reported_scope_tests"] = float(holm[rank])
    rows[-1]["holm_p_across_6_reported_scope_tests"] = None
    return rows


def marked_value(f):
    return float(f.net_aud.sum() + f.soc_end.iloc[-1] * f.actual_price.iloc[-1]
                 - f.soc_start.iloc[0] * f.actual_price.iloc[0])


def select_prefix(validation, frozen):
    detail, choices = [], {}
    for name, end in PERIODS.items():
        deadline = pd.Timestamp(end) - pd.Timedelta(minutes=60)
        choices[name] = []
        for region in REGIONS:
            ref = validation[region]["raw"]
            # Conservative complete-path purge agrees with the saved full-year
            # eligibility: all twelve label endpoints strictly before deadline.
            mask = (ref.target >= "2023-01-01") & (ref.target + pd.Timedelta(minutes=330) < deadline)
            vals = {p: marked_value(validation[region][p].loc[mask]) for p in CANDIDATES}
            best = max(CANDIDATES, key=lambda p: (vals[p], p == "raw", p))
            choices[name].append(best)
            if name == "2023_full":
                assert best == frozen[region]
            for p in CANDIDATES:
                sub = validation[region][p].loc[mask]
                assert sub.soc_start.iloc[0] == 1.0
                assert sub.target.iloc[-1] + pd.Timedelta(minutes=330) < deadline
                independent = float(np.sum(sub.net_aud.to_numpy(), dtype=np.float64)
                                    + sub.soc_end.to_numpy()[-1] * sub.actual_price.to_numpy()[-1]
                                    - sub.soc_start.to_numpy()[0] * sub.actual_price.to_numpy()[0])
                assert abs(vals[p] - independent) < 1e-8
                detail.append(dict(selection_period=name, region=region, candidate=p,
                                   selected=best, winner=p == best, origins=int(mask.sum()),
                                   score_aud=vals[p], cash_aud=float(sub.net_aud.sum()),
                                   inventory_mark_aud=vals[p] - float(sub.net_aud.sum()),
                                   first_target=str(sub.target.iloc[0]), last_target=str(sub.target.iloc[-1]),
                                   latest_complete_label=str(sub.target.iloc[-1] + pd.Timedelta(minutes=330)),
                                   selection_deadline=str(deadline), initial_soc_mwh=1.0,
                                   first_evaluation_target="2024-01-01 00:00:00",
                                   chronological_only=True, full_path_labels_purged=True))
    return choices, detail


def selected_daily(x, chosen):
    return np.stack([x[:, j, POLICIES.index(p)] for j, p in enumerate(chosen)], axis=1)


def selected_boot(boot, chosen):
    return np.stack([boot[:, j, POLICIES.index(p)] for j, p in enumerate(chosen)], axis=1)


def conditional_full_draws(day, bootmean, full_values):
    """Keep endpoint plus partial-boundary contribution fixed, divisor 974."""
    fixed = np.asarray(full_values) - day.sum(axis=0)
    return (len(day) * bootmean + fixed) / 974.0


def sensitivity_rows(evaluation, totals, choices, x, boot, length):
    rows = []
    for period, chosen in choices.items():
        cv = np.array([totals[r][p] for r, p in zip(REGIONS, chosen)])
        rv = np.array([totals[r]["raw"] for r in REGIONS])
        sv = np.array([totals[r]["sparse_equal"] for r in REGIONS])
        iv = np.array([totals[r]["inverse_lead"] for r in REGIONS])
        oracle = np.array([max(totals[r][p] for p in CANDIDATES) for r in REGIONS])
        cd = selected_daily(x, chosen); cb = selected_boot(boot, chosen)
        rawdraw = conditional_full_draws(x[:, :, 0], boot[:, :, 0], rv)
        cdraw = conditional_full_draws(cd, cb, cv)
        sdraw = conditional_full_draws(x[:, :, 5], boot[:, :, 5], sv)
        idraw = conditional_full_draws(x[:, :, 6], boot[:, :, 6], iv)
        for j, region in enumerate(REGIONS + ["five_assets_sum"]):
            index = slice(None) if j == 5 else j
            value = lambda v: float(v[index].sum())
            reduce = lambda b: b[:, index].sum(axis=1) if j == 5 else b[:, index]
            r, c, s, il, o = [value(v) for v in (rv, cv, sv, iv, oracle)]
            sg, ig = s - r, il - r
            ci = lambda b: [float(v) for v in np.quantile(reduce(b), [.025, .975])]
            rows.append(dict(selection_period=period, region=region, block_days=length,
                             selected_policy=";".join(chosen) if j == 5 else chosen[j],
                             evaluation_nominal_days=974, selection_origins={"2023_Q1":4307,"2023_H1":8675,"2023_full":17507}[period],
                             raw_value_aud=r, selected_value_aud=c, sparse_value_aud=s, inverse_value_aud=il,
                             selected_minus_raw_aud_per_day=(c-r)/974,
                             sparse_minus_raw_aud_per_day=sg/974,
                             sparse_minus_selected_aud_per_day=(s-c)/974,
                             inverse_minus_raw_aud_per_day=ig/974,
                             inverse_minus_selected_aud_per_day=(il-c)/974,
                             sparse_absorption_fraction=(c-r)/sg if abs(sg)>1e-12 else None,
                             inverse_absorption_fraction=(c-r)/ig if abs(ig)>1e-12 else None,
                             sparse_ratio_has_positive_raw_gain=sg>0,
                             inverse_ratio_has_positive_raw_gain=ig>0,
                             ex_post_candidate_regret_aud=o-c,
                             ex_post_candidate_regret_aud_per_day=(o-c)/974,
                             sparse_minus_selected_ci_lo=ci(sdraw-cdraw)[0],
                             sparse_minus_selected_ci_hi=ci(sdraw-cdraw)[1],
                             inverse_minus_selected_ci_lo=ci(idraw-cdraw)[0],
                             inverse_minus_selected_ci_hi=ci(idraw-cdraw)[1],
                             selected_minus_raw_ci_lo=ci(cdraw-rawdraw)[0],
                             selected_minus_raw_ci_hi=ci(cdraw-rawdraw)[1],
                             interval_scope="conditional saved choices/trajectories; fixed boundaries+inventory; no selector refit"))
    return rows


def heterogeneity(x, boot, totals, cstar, length):
    cd, cb = selected_daily(x, cstar), selected_boot(boot, cstar)
    contrasts = [x[:,:,5]-x[:,:,0], x[:,:,5]-cd, x[:,:,6]-cd]
    bcontrasts = [boot[:,:,5]-boot[:,:,0], boot[:,:,5]-cb, boot[:,:,6]-cb]
    vals = np.array([[totals[r]["sparse_equal"] - totals[r]["raw"] for r in REGIONS],
                     [totals[r]["sparse_equal"] - totals[r][p] for r,p in zip(REGIONS,cstar)],
                     [totals[r]["inverse_lead"] - totals[r][p] for r,p in zip(REGIONS,cstar)]])
    # Four independent restrictions, rather than an n=5 cluster test.
    a = np.column_stack([np.eye(4), -np.ones(4)])
    results, means, covariance, portfolio = [], [], [], []
    for name, d, bm, full in zip(CONTRASTS, contrasts, bcontrasts, vals):
        n, r = d.shape; mu=d.mean(axis=0); grand=float(mu.mean())
        vt=float(np.mean((d-grand)**2))
        vb=float(np.mean((mu-grand)**2)); vw=float(np.mean((d-mu)**2))
        # Independent sum-of-squares reconstruction checks population denominator.
        sst=float(np.dot((d-grand).ravel(),(d-grand).ravel()))
        ssb=float(n*np.dot(mu-grand,mu-grand)); ssw=float(np.sum(np.var(d,axis=0,ddof=0))*n)
        assert np.isclose(vt,vb+vw,rtol=1e-12,atol=1e-7)
        assert np.isclose(sst,ssb+ssw,rtol=1e-12,atol=1e-4)
        centered=bm-mu[None,:]
        v=np.sqrt(n)*(centered@a.T)
        cov=np.cov(v,rowvar=False,ddof=1)
        rank=int(np.linalg.matrix_rank(cov)); assert rank==4
        inverse=np.linalg.inv(cov)
        obs=np.sqrt(n)*(a@mu)
        q=float(obs@inverse@obs)
        qb=np.einsum('bi,ij,bj->b',v,inverse,v)
        pv,mcse,hits=pvalue(q,qb)
        results.append(dict(contrast=name,block_days=length,interior_days=n,fixed_regions=r,
                            daily_grand_mean_per_asset_aud=grand,daily_five_asset_sum_mean_aud=float(mu.sum()),
                            total_panel_population_variance=vt,between_region_population_variance=vb,
                            within_region_population_variance=vw,between_fraction=vb/vt,
                            within_fraction=vw/vt,identity_residual=vt-vb-vw,
                            joint_mean_equality_statistic=q,restriction_rank=rank,
                            covariance_condition_number=float(np.linalg.cond(cov)),
                            centered_block_bootstrap_p=pv,monte_carlo_se=mcse,exceedances=hits,
                            hypothesis="five fixed-asset daily mean contrasts are equal",
                            inference_scope="conditional named assets/time series; no random-regions population inference"))
        mcov=np.cov(bm,rowvar=False,ddof=1)
        for j,region in enumerate(REGIONS):
            for k,other in enumerate(REGIONS):
                covariance.append(dict(contrast=name,block_days=length,region=region,other_region=other,
                                       covariance_bootstrap_mean_aud2=float(mcov[j,k])))
        fdraw=conditional_full_draws(d,bm,full)
        for j,region in enumerate(REGIONS+["five_assets_sum"]):
            b=bm.sum(axis=1) if j==5 else bm[:,j]
            fd=fdraw.sum(axis=1) if j==5 else fdraw[:,j]
            point=float(full.sum()/974) if j==5 else float(full[j]/974)
            ci=np.quantile(b,[.025,.975]); fci=np.quantile(fd,[.025,.975])
            means.append(dict(contrast=name,block_days=length,region=region,
                              interior_daily_cash_mean_aud=float(mu.sum()) if j==5 else float(mu[j]),
                              daily_cash_ci_lo=float(ci[0]),daily_cash_ci_hi=float(ci[1]),
                              full_value_per_nominal_day_aud=point,
                              full_value_ci_lo=float(fci[0]),full_value_ci_hi=float(fci[1])))
        dcov=np.cov(d,rowvar=False,ddof=0)
        diag=float(np.trace(dcov)); cross=float(dcov.sum()-diag)
        direct=float(np.var(d.sum(axis=1),ddof=0))
        assert np.isclose(diag+cross,direct,rtol=1e-12,atol=1e-7)
        portfolio.append(dict(contrast=name,block_days=length,
                              daily_sum_variance_aud2=direct,sum_region_variances_aud2=diag,
                              twice_sum_between_region_time_covariances_aud2=cross,
                              identity_residual=diag+cross-direct,
                              bootstrap_five_asset_mean_variance_aud2=float(mcov.sum()),
                              bootstrap_sum_diagonal_mean_variances_aud2=float(np.trace(mcov)),
                              bootstrap_cross_mean_covariances_aud2=float(mcov.sum()-np.trace(mcov))))
    return results,means,covariance,portfolio,contrasts


def market_context(evaluation):
    hp=ROOT/'results/revision_v5/history/verified_history_age_counts.csv'
    fp=ROOT/'results/revision_v5/history/verified_input_freshness.csv'
    record(hp);record(fp)
    age=pd.read_csv(hp); fresh=pd.read_csv(fp).set_index('region')
    rows=[]
    for region in REGIONS:
        raw=evaluation[region]['raw']; price=raw.actual_price.to_numpy()
        a=age.loc[(age.region==region)&(age.cutoff_minutes==60)].set_index('field')
        f=fresh.loc[region]
        rows.append(dict(region=region,settlement_intervals=len(raw),
                         actual_price_mean_aud_per_mwh=float(price.mean()),
                         actual_price_std_aud_per_mwh=float(price.std(ddof=0)),
                         actual_price_median_aud_per_mwh=float(np.median(price)),
                         actual_price_p05_aud_per_mwh=float(np.quantile(price,.05)),
                         actual_price_p95_aud_per_mwh=float(np.quantile(price,.95)),
                         actual_price_p99_aud_per_mwh=float(np.quantile(price,.99)),
                         actual_price_max_aud_per_mwh=float(price.max()),
                         negative_price_share=float(np.mean(price<0)),
                         price_at_least_1000_share=float(np.mean(price>=1000)),
                         oldest_record_age_median_min=float(a.loc['oldest_age_min','median']),
                         latest_record_age_at_cutoff_median_min=float(a.loc['latest_age_min','median']),
                         prior6_age_vs_coherent_median_min=float(a.loc['lag6_age_since_coherent_min','median']),
                         versions_median=float(a.loc['count','median']),
                         fresh_freeze_raw_curve_changed_share=float(f.raw_curve_changed_share),
                         old_archive_lead_execution_median_min=float(f.old_median_lead_to_execution_min),
                         exploratory_scope='fixed five-market description; no five-point regression or significance test'))
    return rows


def regime_diagnostics(cubes, days, cstar):
    """Year-specific descriptions, without post-hoc year-wise significance tests."""
    rows=[]
    for phase,x in cubes.items():
        cd=selected_daily(x,cstar)
        metric={f'{p}_minus_R':x[:,:,j]-x[:,:,0] for j,p in enumerate(CANDIDATES[1:],1)}
        if phase=='evaluation':
            metric.update(S_minus_R=x[:,:,5]-x[:,:,0],S_minus_Cstar=x[:,:,5]-cd,
                          IL_minus_Cstar=x[:,:,6]-cd)
        for year in sorted(set(days[phase].year)):
            take=days[phase].year==year
            for name,d in metric.items():
                for j,region in enumerate(REGIONS+['five_assets_sum']):
                    values=d[take].sum(axis=1) if j==5 else d[take,j]
                    rows.append(dict(phase=phase,execution_year=int(year),contrast=name,
                                     region=region,interior_days=int(take.sum()),
                                     daily_cash_mean_aud=float(values.mean()),
                                     daily_cash_std_aud=float(values.std(ddof=0)),
                                     operating_cash_sum_aud=float(values.sum()),
                                     description_only=True,
                                     scope='observed calendar-year interior dates; 2026 is Jan-Aug; excludes endpoint marks; no per-year p-value'))
    return rows


def run(draws):
    assert draws>=2000
    start=time.time();DEST.mkdir(parents=True,exist_ok=True)
    record(Path(__file__))
    validation,evaluation,totals,frozen={},{},{},{}
    for region in REGIONS:
        validation[region]={};evaluation[region]={};totals[region]={}
        for p in CANDIDATES:
            validation[region][p],_=load(region,p,'validation')
        for p in POLICIES:
            evaluation[region][p],totals[region][p]=load(region,p,'evaluation')
        sp=SOURCE/f'{region}_current_only_selection.json';record(sp)
        frozen[region]=json.loads(sp.read_text())['selected_policy']
    phases={'validation':validation,'evaluation':evaluation}
    cubes={}; days={}; long=[]
    maximum_daily_aggregation_error=0.0
    for phase,frames in phases.items():
        pnames=CANDIDATES if phase=='validation' else POLICIES
        ref=frames[REGIONS[0]]['raw']
        day_index=pd.DatetimeIndex(ref.execution_day.unique()).sort_values()
        interior=day_index[1:-1]; days[phase]=interior
        assert len(interior)==(364 if phase=='validation' else 973)
        cube=np.empty((len(interior),5,len(pnames)))
        codes=day_index.get_indexer(ref.execution_day)
        for ri,region in enumerate(REGIONS):
            for pi,p in enumerate(pnames):
                f=frames[region][p]
                assert f.target.equals(ref.target)
                assert f.execution_day.equals(ref.execution_day)
                assert np.array_equal(f.actual_price.to_numpy(),frames[region]['raw'].actual_price.to_numpy())
                daily=f.groupby('execution_day').net_aud.sum().reindex(day_index)
                direct=np.bincount(codes,weights=f.net_aud.to_numpy(),minlength=len(day_index))
                err=float(np.max(np.abs(direct-daily.to_numpy())))
                maximum_daily_aggregation_error=max(maximum_daily_aggregation_error,err)
                assert err<1e-8
                cube[:,ri,pi]=daily.iloc[1:-1].to_numpy()
                for date,cash,count in zip(day_index,daily, f.groupby('execution_day').size().reindex(day_index)):
                    long.append(dict(phase=phase,region=region,policy=p,execution_day=date,
                                     net_cash_aud=float(cash),observed_intervals=int(count),
                                     is_interior=date in interior))
        cubes[phase]=cube
    pd.DataFrame(long).to_parquet(DEST/'daily_operating_cash_panel.parquet',index=False)
    choices,detail=select_prefix(validation,frozen)
    write_csv('selection_period_candidates.csv',detail)
    write_json('selection_period_choices.json',{'choices':choices,'regions':REGIONS,
              'eligibility':'target+330min < following boundary-60min; shared saved 2023 prefix; no fresh midyear initialisation',
              'evaluation':'each candidate saved 2024-2026 replay starts at common 1MWh, fixed choice throughout; no selection refitting in intervals'})
    rc=[];sens=[];hets=[];effects=[];covs=[];ports=[];literal_errors={}
    cstar=[frozen[r] for r in REGIONS]
    for block in [7,14,28]:
        for phase in ['validation','evaluation']:
            # Preserve the already disclosed evaluation resampling seed. All
            # columns share time indices, without independent regional draws.
            seed=20261003 if phase=='validation' else 20261001
            boot,err=bootstrap_means(cubes[phase],draws,block,seed)
            literal_errors[f'{phase}_L{block}']=err
            rc.extend(reality_checks(cubes[phase],boot,phase,block,draws,seed))
            if phase=='evaluation':
                sens.extend(sensitivity_rows(evaluation,totals,choices,cubes[phase],boot,block))
                h,m,c,p,contrast_mats=heterogeneity(cubes[phase],boot,totals,cstar,block)
                hets+=h;effects+=m;covs+=c;ports+=p
        print(json.dumps({'block_days':block,'completed':True,'seconds':round(time.time()-start,2)}),flush=True)
    panel=[]
    for name,d in zip(CONTRASTS,contrast_mats):
        for j,region in enumerate(REGIONS):
            for day,value in zip(days['evaluation'],d[:,j]):
                panel.append(dict(execution_day=day,region=region,contrast=name,net_cash_difference_aud=float(value)))
    pd.DataFrame(panel).to_parquet(DEST/'daily_contrast_panel.parquet',index=False)
    context=market_context(evaluation)
    regimes=regime_diagnostics(cubes,days,cstar)
    for block in [7,14,28]:
        group=[r for r in hets if r['block_days']==block]
        ps=np.asarray([r['centered_block_bootstrap_p'] for r in group])
        order=np.argsort(ps)
        hp=np.maximum.accumulate(np.minimum(1.0,ps[order]*(3-np.arange(3))))
        for rank,j in enumerate(order):
            group[int(j)]['holm_p_across_3_contrast_equality_tests']=float(hp[rank])
    compact_rc=[]
    for phase in ['validation','evaluation']:
        for region in REGIONS+['five_assets_sum']:
            fam='all_3125_regional_rule_combinations' if region=='five_assets_sum' else 'four_processed_vs_raw'
            subset=[r for r in rc if r['phase']==phase and r['scope']==region and r['family']==fam]
            row=dict(phase=phase,region=region,family=fam)
            row.update({f'p_block_{r["block_days"]}':r['p_value'] for r in subset})
            compact_rc.append(row)
    compact_selection=[]
    for r in sens:
        if r['region']=='five_assets_sum' and r['block_days']==7:
            compact_selection.append({k:r[k] for k in [
                'selection_period','selection_origins','sparse_minus_selected_aud_per_day',
                'inverse_minus_selected_aud_per_day','sparse_absorption_fraction',
                'inverse_absorption_fraction','ex_post_candidate_regret_aud',
                'sparse_minus_selected_ci_lo','sparse_minus_selected_ci_hi',
                'inverse_minus_selected_ci_lo','inverse_minus_selected_ci_hi']})
    compact_het=[]
    for name in CONTRASTS:
        group=[r for r in hets if r['contrast']==name]
        row=dict(contrast=name,between_percent=100*group[0]['between_fraction'],
                 within_percent=100*group[0]['within_fraction'])
        row.update({f'joint_equality_p_block_{r["block_days"]}':r['centered_block_bootstrap_p'] for r in group})
        row.update({f'holm_3_p_block_{r["block_days"]}':r['holm_p_across_3_contrast_equality_tests'] for r in group})
        compact_het.append(row)
    for filename,rows in [('reality_check.csv',rc),('selection_length_sensitivity.csv',sens),
                          ('heterogeneity.csv',hets),('region_effects.csv',effects),
                          ('synchronised_mean_covariance.csv',covs),
                          ('portfolio_covariance_decomposition.csv',ports),('market_context.csv',context),
                          ('regime_daily_means.csv',regimes),('table_reality_check_compact.csv',compact_rc),
                          ('table_selection_length_compact.csv',compact_selection),
                          ('table_heterogeneity_compact.csv',compact_het),
                          ('table_reality_check_evaluation.csv',[r for r in compact_rc if r['phase']=='evaluation'])]:
        write_csv(filename,rows)
    validation_record=dict(status='PASS',source_trajectories=60,
                           actual_trajectories_read_and_audited=True,
                           validation_days=364,evaluation_days=973,
                           cash_wear_soc_continuity_endpoint_checks='PASS',
                           daily_bincount_vs_groupby_max_abs_error=maximum_daily_aggregation_error,
                           literal_block_index_checks=literal_errors,
                           mixed_family_maximum_vs_explicit_3125_enumeration='PASS: observation and first three centered draws per phase/block',
                           between_plus_within_and_portfolio_covariance_identities='PASS',
                           full_year_selection_matches_all_five_frozen_choices=True,
                           no_solver_invoked=True)
    write_json('arithmetic_validation.json',validation_record)
    summary=dict(status='computed_and_arithmetically_verified',bootstrap_draws=draws,
                 block_lengths=[7,14,28],regions=REGIONS,candidates=CANDIDATES,
                 reality_check=rc,selection_length_sensitivity=sens,
                 heterogeneity=hets,region_effects=effects,market_context=context,
                 calendar_year_descriptive_means=regimes,
                 software={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},
                 selection_choices=choices,validation=validation_record,
                 reference={'author':'White (2000)','doi':'10.1111/1468-0262.00152'},
                 statistical_scope='fixed candidate families and five named assets conditional on saved policies and observed chronology; exploratory post-hoc research remains uncorrected',
                 assumptions='short-range weak dependence/stability required for circular block approximation; 7/14/28-day sensitivity does not validate stationarity or structural-break robustness',
                 endpoint_scope='Reality Check uses interior daily operating cash only; marked full-value intervals keep partial boundaries and endpoint inventory fixed',
                 descriptive_variance_scope='between/within panel identity describes spread of region-day observations, not variance of a random sample of regions or the five-asset sum',
                 runtime_seconds=time.time()-start)
    write_json('summary.json',summary)
    write_json('source_manifest.json',{'status':'read_only_sources','files':list(MANIFEST.values()),
               'analysis_script_sha256':sha(Path(__file__)),'results_created_under':str(DEST)})
    print(json.dumps({'status':summary['status'],'output':str(DEST),'seconds':summary['runtime_seconds'],
                      'selection_choices':choices},ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=10000)
    output=ap.add_mutually_exclusive_group()
    output.add_argument('--output-dir',type=Path,help='Separate results directory; use when running frozen package.')
    output.add_argument('--output-root',type=Path,help='Separate D-drive root; append results/revision_v7/statistics.')
    args=ap.parse_args()
    if args.output_dir:
        DEST=args.output_dir.resolve()
    elif args.output_root:
        DEST=args.output_root.resolve()/'results/revision_v7/statistics'
    if os.name=='nt':
        assert DEST.drive.upper()=='D:', f'Research outputs must be on D: {DEST}'
    run(args.draws)
