"""Information-defined event probes with genuinely updated forecast continuation.

All actions use the corresponding historical eligible forecast. Realized prices
enter reward accounting and end-of-probe marks only. A probe is a finite local
comparison, not a decomposition of full-period policy value.
"""
from __future__ import annotations
import argparse
import importlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('D:/MLWork/AOOR_ContextualStorage_20260929')
OUT = ROOT / 'results/revision_v5/events'
REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
INPUTS = ROOT / 'results/revision_v5/history'
STEPS = 48

def paths(region, policy, cutoff=60):
    source = INPUTS / f'{region}_inputs_aligned.parquet'
    if (cutoff == 60 and policy in ('raw', 'sparse_equal')) or not source.exists():
        source = INPUTS / f'{region}_baseline_inputs.parquet'
    columns = ['target'] + [f'p_{h:02d}' for h in range(1, 13)] + [f'actual_{h:02d}' for h in range(12)]
    f = pd.read_parquet(source, columns=columns,
                        filters=[('policy', '==', policy), ('cutoff_minutes', '==', cutoff),
                                 ('target', '>=', pd.Timestamp('2024-01-01'))])
    f = f.sort_values('target').reset_index(drop=True)
    if len(f) != 46741 or f.target.duplicated().any():
        raise ValueError((region, policy, cutoff, len(f)))
    f.attrs['source'] = str(source)
    return f, f[[f'p_{h:02d}' for h in range(1, 13)]].to_numpy(float)

def make_solver(module, class_name):
    return getattr(importlib.import_module(module), class_name)()

def simulate(solver, curves, prices, initial_soc, first_curve=None):
    soc = float(initial_soc)
    rows = []
    for k, (curve, actual) in enumerate(zip(curves, prices)):
        chosen = first_curve if k == 0 and first_curve is not None else curve
        action = solver.solve(chosen, soc)
        c, d, nxt = float(action['c']), float(action['d']), float(action['soc'])
        cash = (d - c) * float(actual) * solver.dt
        wear = (c + d) * solver.kappa * solver.dt
        rows.append({'step': k, 'soc_start': soc, 'charge_mw': c,
                     'discharge_mw': d, 'soc_end': nxt, 'actual_price': float(actual),
                     'cashflow_aud': cash, 'wear_aud': wear, 'net_cash_aud': cash - wear})
        soc = nxt
    return pd.DataFrame(rows)

def probe_region(region, alternate, baseline_file, module, class_name):
    events = pd.read_csv(OUT / f'{region}_events.csv')
    events = events[events.rolling_probe]
    panel = pd.read_parquet(OUT / f'{region}_event_panel.parquet')
    base, raw = paths(region, 'raw')
    alt, historical = paths(region, alternate)
    trajectory = pd.read_parquet(baseline_file).sort_values('target').reset_index(drop=True)
    targets = base.target.astype('datetime64[ns]')
    assert targets.equals(alt.target.astype('datetime64[ns]'))
    assert targets.equals(panel.target.astype('datetime64[ns]'))
    assert targets.equals(trajectory.target.astype('datetime64[ns]'))
    prices = trajectory.actual_price.to_numpy(float)
    solver_source_hash = hashlib.sha256(Path(importlib.import_module(module).__file__).read_bytes()).hexdigest()
    solver = make_solver(module, class_name)
    summaries, details = [], []
    start_time = time.perf_counter()
    replay_discrepancy = 0.
    for event in events.itertuples(index=False):
        p = int(event.first_pos)
        stop = p + STEPS
        soc = float(trajectory.soc_start.iloc[p])
        branches = {
            'raw_rolling': simulate(solver, raw[p:stop], prices[p:stop], soc),
            'one_history_then_raw': simulate(solver, raw[p:stop], prices[p:stop], soc,
                                             first_curve=historical[p]),
            'history_rolling': simulate(solver, historical[p:stop], prices[p:stop], soc),
        }
        reference = branches['raw_rolling']
        old = trajectory.iloc[p:stop]
        for field, original in [('soc_start', 'soc_start'), ('charge_mw', 'charge_mw'),
                                ('discharge_mw', 'discharge_mw'), ('soc_end', 'soc_end')]:
            discrepancy = float(np.max(np.abs(reference[field].to_numpy() - old[original].to_numpy())))
            replay_discrepancy = max(replay_discrepancy, discrepancy)
        # No realization is read by the optimizer. These alternative accounting
        # marks expose the finite-probe endpoint dependence explicitly.
        final_price = float(prices[stop - 1])
        for branch_name, result in branches.items():
            result['region'] = region
            result['event_id'] = int(event.event_id)
            result['branch'] = branch_name
            result['target'] = base.target.iloc[p:stop].to_numpy()
            result['initial_soc'] = soc
            details.append(result)
            net = float(result.net_cash_aud.sum())
            ending = float(result.soc_end.iloc[-1])
            base_net = float(reference.net_cash_aud.sum())
            base_ending = float(reference.soc_end.iloc[-1])
            summaries.append({'region': region, 'event_id': int(event.event_id),
                              'start': base.target.iloc[p] - pd.Timedelta(minutes=30),
                              'end': base.target.iloc[stop - 1], 'branch': branch_name,
                              'elapsed_hours': float((base.target.iloc[stop - 1] - base.target.iloc[p] + pd.Timedelta(minutes=30)).total_seconds() / 3600),
                              'internal_idle_intervals': int((base.target.iloc[p:stop].diff().dropna() / pd.Timedelta(minutes=30) - 1).sum()),
                              'alternate_policy': alternate, 'steps': STEPS, 'initial_soc': soc,
                              'first_net_mw': float(result.discharge_mw.iloc[0] - result.charge_mw.iloc[0]),
                              'first_action_delta_mw': float(result.discharge_mw.iloc[0] - result.charge_mw.iloc[0] -
                                                           reference.discharge_mw.iloc[0] + reference.charge_mw.iloc[0]),
                              'net_cash_aud': net, 'cash_delta_aud': net - base_net,
                              'final_soc': ending, 'final_soc_delta': ending - base_ending,
                              'last_price_mark': final_price,
                              'marked_delta_aud': net - base_net + final_price * (ending - base_ending),
                              'efficiency_marked_delta_aud': net - base_net + solver.eta * final_price * (ending - base_ending),
                              'throughput_mwh': float((result.charge_mw + result.discharge_mw).sum() * solver.dt),
                              'first_trigger_peak': bool(event.first_trigger_peak),
                              'first_trigger_range': bool(event.first_trigger_range),
                              'first_trigger_revision': bool(event.first_trigger_revision),
                              'selection_uses_profit': False})
    if replay_discrepancy > 1e-5:
        raise RuntimeError(f'{region}: cold raw rolling does not reproduce canonical trajectory: {replay_discrepancy}')
    pd.concat(details, ignore_index=True).to_parquet(OUT / f'{region}_{alternate}_rolling_probe_paths.parquet', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / f'{region}_{alternate}_rolling_probes.csv', index=False)
    complete = {'status': 'complete', 'region': region, 'alternate': alternate,
                'selected_events': len(events), 'steps_per_probe': STEPS,
                'solver_calls': int(len(events) * STEPS * 3),
                'runtime_seconds': time.perf_counter() - start_time,
                'raw_replay_max_discrepancy': replay_discrepancy,
                'future_predictions_updated': True, 'future_prices_used_by_actions': False,
                'finite_window_not_full_value_decomposition': True,
                'baseline_forecast_array_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                'alternate_forecast_array_sha256': hashlib.sha256(historical.tobytes()).hexdigest(),
                'solver_source_sha256': solver_source_hash,
                'frozen_core_sha256': hashlib.sha256(Path(importlib.import_module('revision_v5_control_solver').__file__).read_bytes()).hexdigest(),
                'safe_wrapper_sha256': hashlib.sha256(Path(importlib.import_module('revision_v5_control_safe_policy').__file__).read_bytes()).hexdigest(),
                'numerical_fallback_count': getattr(solver, 'fallback_count', 0),
                'numerical_fallback_records': getattr(solver, 'fallback_records', []),
                'policy_definition': solver.definition(),
                'selection_uses_profit': False}
    (OUT / f'{region}_{alternate}_rolling_probe_complete.json').write_text(json.dumps(complete, indent=2), encoding='utf-8')
    print(json.dumps(complete), flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', nargs='+', default=REGIONS)
    parser.add_argument('--alternate', default='sparse_equal')
    parser.add_argument('--solver-module', default='revision_v5_control_scaled_policy')
    parser.add_argument('--solver-class', default='ScaledSafeLexMPC')
    parser.add_argument('--baseline-template', required=True,
                        help='Absolute path template with {region} for canonical raw trajectory.')
    args = parser.parse_args()
    summaries = [probe_region(r, args.alternate, args.baseline_template.format(region=r),
                             args.solver_module, args.solver_class) for r in args.regions]
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(OUT / f'{args.alternate}_all_rolling_probes.csv', index=False)
    print(all_summary.groupby('branch')[['cash_delta_aud', 'marked_delta_aud']].agg(['count', 'mean', 'median', 'min', 'max']).to_string())

if __name__ == '__main__':
    main()
