"""Direct archive-adoption tests and descriptive annual market context.

Reads frozen v7/v5 evidence; never fits/selects a program or invokes a solver.
The two primary tests are declared before execution in the v8 revision plan.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd
from revision_v7_statistics import bootstrap_means, pvalue

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/revision_v8/direct_tests'
REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
NAMES = ['S_minus_R', 'S_minus_Cstar', 'IL_minus_Cstar']
FILES = {}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def record(p):
    FILES[str(p.relative_to(ROOT)).replace('\\', '/')] = {
        'bytes': p.stat().st_size, 'sha256': sha(p)}

def read_json(p):
    record(p)
    return json.loads(p.read_text(encoding='utf-8'))

def main(out, draws):
    assert draws == 10000
    out.mkdir(parents=True, exist_ok=True)
    if out.drive:
        assert out.drive.upper() == 'D:'
    source = ROOT / 'results/revision_v7/statistics/daily_contrast_panel.parquet'
    record(source); panel = pd.read_parquet(source)
    days = pd.DatetimeIndex(sorted(panel.execution_day.unique()))
    assert len(days) == 973
    x = np.empty((973, 5, 3))
    for r, region in enumerate(REGIONS):
        for c, contrast in enumerate(NAMES):
            f = panel.loc[(panel.region == region) & (panel.contrast == contrast)].set_index('execution_day')
            assert len(f) == 973 and not f.index.duplicated().any()
            x[:, r, c] = f.reindex(days).net_cash_difference_aud.to_numpy()
    assert np.isfinite(x).all()
    # Full marked values preserve exact unrounded original program reports.
    root = ROOT / 'results/revision_v5/control/main'
    total = np.zeros(3); annual = []; annual_cash = []
    for r, region in enumerate(REGIONS):
        select = read_json(root / f'{region}_current_only_selection.json')['selected_policy']
        values = {p: read_json(root / f'{region}_evaluation_c60_{p}.json')['net_value_aud']
                  for p in ['raw', 'sparse_equal', 'inverse_lead', select]}
        total += [values['sparse_equal'] - values['raw'],
                  values['sparse_equal'] - values[select],
                  values['inverse_lead'] - values[select]]
        pricefile = root / f'{region}_evaluation_c60_raw.parquet'
        record(pricefile)
        prices = pd.read_parquet(pricefile, columns=['execution_day', 'actual_price'])
        prices['execution_day'] = pd.to_datetime(prices.execution_day)
        prices = prices.loc[prices.execution_day.isin(days)]
        for year in sorted(prices.execution_day.dt.year.unique()):
            p = prices.loc[prices.execution_day.dt.year == year, 'actual_price'].to_numpy()
            take = days.year == year
            annual.append(dict(region=region, year=int(year), interior_dates=int(take.sum()),
                settlement_intervals=len(p), price_mean=float(p.mean()), price_sd=float(p.std(ddof=0)),
                negative_share=float(np.mean(p < 0)), at_least_1000_share=float(np.mean(p >= 1000)),
                price_p99=float(np.quantile(p, .99)), price_max=float(p.max()),
                unit='AUD/MWh', description_only=True))
            cash = x[take, r].mean(axis=0)
            annual_cash.append(dict(region=region, year=int(year), S_minus_R=float(cash[0]),
                S_minus_Cstar=float(cash[1]), IL_minus_Cstar=float(cash[2]),
                Cstar_minus_R=float(cash[0] - cash[1]),
                sparse_absorption=float((cash[0] - cash[1]) / cash[0]),
                unit='AUD/interior execution date', description_only=True))
    nominal = 974
    marked = total / nominal
    fixed = total - x.sum(axis=(0, 1))
    mu = x.sum(axis=1).mean(axis=0)
    results = []; ratios = []; checks = []
    for length in [7, 14, 28]:
        boot, error = bootstrap_means(x, draws, length, 20261001)
        bsum = boot.sum(axis=1)
        checks.append(dict(block_days=length, literal_index_mean_error=error))
        for accounting, observed, sampled in [
            ('interior_cash', mu, bsum),
            ('marked_nominal', marked, (len(days) * bsum + fixed) / nominal),
        ]:
            obs = np.r_[observed, observed[0] - observed[1]]
            bs = np.column_stack([sampled, sampled[:, 0] - sampled[:, 1]])
            rows = []
            for j, name in enumerate(NAMES + ['Cstar_minus_R']):
                centered = bs[:, j] - obs[j]
                one, mc, hits = pvalue(obs[j], centered)
                two, _, twohits = pvalue(abs(obs[j]), np.abs(centered))
                lo, hi = np.quantile(bs[:, j], [.025, .975])
                rows.append(dict(contrast=name, accounting=accounting, block_days=length,
                    observations=973, bootstrap_draws=draws, seed=20261001,
                    effect=float(obs[j]), ci_lo=float(lo), ci_hi=float(hi),
                    null='mean <= 0', one_sided_p=one, one_sided_exceedances=hits,
                    monte_carlo_se=mc, two_sided_p=two, two_sided_exceedances=twohits,
                    primary_test=name in ['S_minus_Cstar', 'IL_minus_Cstar'],
                    primary_block=length == 7, holm_two_primary_p=None))
            primary = [rows[1], rows[2]]
            order = np.argsort([r['one_sided_p'] for r in primary])
            adjusted = np.maximum.accumulate(np.minimum(1., np.asarray(
                [primary[int(j)]['one_sided_p'] for j in order]) * [2, 1]))
            for rank, j in enumerate(order):
                primary[int(j)]['holm_two_primary_p'] = float(adjusted[rank])
            results.extend(rows)
            # Ratio is signed and can exceed one. Do not censor denominator
            # signs, trim draws to [0,1], or interpret non-rejection as equivalence.
            denom = bs[:, 0]
            assert not np.any(denom == 0)
            ratio = (denom - bs[:, 1]) / denom
            lo, hi = np.quantile(ratio, [.025, .975])
            ratios.append(dict(accounting=accounting, block_days=length,
                sparse_absorption=float((obs[0] - obs[1]) / obs[0]),
                percentile_ratio_lo=float(lo), percentile_ratio_hi=float(hi),
                nonpositive_denominator_share=float(np.mean(denom <= 0)),
                interpretation='joint conditional descriptive ratio, not a probability; non-rejection is not equivalence'))
    a = pd.DataFrame(annual); c = pd.DataFrame(annual_cash)
    compact = []
    for year in sorted(a.year.unique()):
        ay = a.loc[a.year == year]; take = days.year == year
        cash = x[take].sum(axis=1).mean(axis=0)
        compact.append(dict(year=int(year), interior_dates=int(take.sum()),
            mean_of_regional_price_sd=float(ay.price_sd.mean()),
            negative_share=float(np.average(ay.negative_share, weights=ay.settlement_intervals)),
            at_least_1000_share=float(np.average(ay.at_least_1000_share, weights=ay.settlement_intervals)),
            S_minus_R=float(cash[0]), S_minus_Cstar=float(cash[1]), IL_minus_Cstar=float(cash[2]),
            Cstar_minus_R=float(cash[0] - cash[1]), sparse_absorption=float((cash[0] - cash[1]) / cash[0])))
    datasets = {'direct_mean_tests.csv': results, 'absorption_ratio_uncertainty.csv': ratios,
        'annual_price_context.csv': annual, 'annual_regional_cash.csv': annual_cash,
        'annual_context_compact.csv': compact}
    for name, data in datasets.items():
        pd.DataFrame(data).to_csv(out / name, index=False, float_format='%.12g')
    validation = dict(status='PASS', days=973, assets=5, source_panel_rows=len(panel),
        no_solver_or_selection=True, full_marked_values_from_unrounded_reports=total.tolist(),
        fixed_boundary_and_mark_differences=fixed.tolist(), literal_block_checks=checks,
        scientific_scope='Conditional on saved fitted/selected programs and observed historical chronology; the observed evaluation years informed earlier research development. Primary seven-day one-sided tests S−C* and IL−C*, with within-pair Holm adjustment. Other block lengths/accounting are sensitivity analyses, not independent confirmations. A large p does not establish zero, equivalence or an absence of meaningful archive value.',
        annual_scope='Three descriptive annual periods, 2026 January–August, not a feature-effect regression or causal attribution.')
    receipt = dict(validation=validation, status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), sources=FILES,
        outputs={name: {'sha256': sha(out/name), 'bytes': (out/name).stat().st_size} for name in datasets},
        results=results, ratios=ratios, annual_context=compact)
    (out/'execution_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'output': str(out),
        'primary_marked_seven_day': [r for r in results if r['accounting']=='marked_nominal' and r['block_days']==7 and r['primary_test']],
        'annual_context': compact}, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output-dir', type=Path, default=DEFAULT)
    ap.add_argument('--draws', type=int, default=10000); args = ap.parse_args()
    main(args.output_dir.resolve(), args.draws)
