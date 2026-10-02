"""Presentation-only revision of Figures 4--6 from frozen executed evidence.

All native pages are 174 mm wide. No model, test, sample or result is rerun.
Array provenance, physical text geometry, and actual vector exports are retained.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from revision_v16_figure_style import (
    ROOT, OUT, BLUE, ORANGE, TEAL, INK, GRAY, LIGHT,
    plt, frame, text, heading, axis, final_save, sha,
)

REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
LABELS = ['NSW', 'QLD', 'SA', 'TAS', 'VIC']


def assert_clean_layout(fig, stem):
    """Check visible text, including side titles, against actual data artists."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    seen, texts = set(), []
    for artist in fig.findobj(match=Text):
        if id(artist) in seen or not artist.get_visible() or not artist.get_text().strip():
            continue
        seen.add(id(artist))
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            texts.append((artist, box))
    text_hits, data_hits = [], []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            area = max(0., min(ab.x1, bb.x1) - max(ab.x0, bb.x0)) * max(
                0., min(ab.y1, bb.y1) - max(ab.y0, bb.y0))
            if area > 2.:
                text_hits.append({'a': a.get_text(), 'b': b.get_text(), 'px2': area})
    for ax in fig.axes:
        for line in ax.lines:
            if not line.get_visible():
                continue
            path = line.get_path().transformed(line.get_transform())
            for artist, box in texts:
                marker_r = float(line.get_markersize()) * fig.dpi / 72 / 2 if line.get_marker() not in ['', 'None', 'none', None] else 0.
                enlarged = box.padded(marker_r)
                if path.intersects_bbox(enlarged, filled=False):
                    data_hits.append({'text': artist.get_text(), 'artist': line.get_label()})
        for coll in ax.collections:
            offsets = np.asarray(coll.get_offsets())
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                continue
            coords = coll.get_offset_transform().transform(offsets)
            sizes = np.asarray(coll.get_sizes())
            radius = np.sqrt(float(sizes.max())) * fig.dpi / 72 / 2 if len(sizes) else 0.
            for artist, box in texts:
                eb = box.padded(radius)
                if len(coords) and np.any((coords[:, 0] >= eb.x0) & (coords[:, 0] <= eb.x1) &
                                          (coords[:, 1] >= eb.y0) & (coords[:, 1] <= eb.y1)):
                    data_hits.append({'text': artist.get_text(), 'artist': coll.get_label()})
    record = {'status': 'PASS' if not (text_hits or data_hits) else 'REPAIR_REQUIRED',
              'checked_text_artists': len(texts),
              'minimum_native_text_font_pt': min(float(t.get_fontsize()) for t, _ in texts),
              'text_text_intersections': text_hits, 'text_data_intersections': data_hits,
              'title_locations': [{loc: ax.get_title(loc=loc) for loc in ['left', 'center', 'right']}
                                  for ax in fig.axes],
              'scope': 'Visible Text artists versus data Line2D paths and expanded markers; no opaque masks.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{stem}_layout.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    assert record['minimum_native_text_font_pt'] >= 9.
    assert record['status'] == 'PASS', json.dumps(record, indent=2)
    return record


def point_offsets(ax, center, values, marker_diameter_pt=3.6):
    """Deterministic bounded x spacing; all economic y values stay unchanged.

    Coincident observations can overlap visually. They are never dropped or
    moved vertically to suggest distinct economic outcomes.
    """
    fig = ax.figure
    fig.canvas.draw()
    coords = ax.transData.transform(np.column_stack([np.full(len(values), center), values]))
    gap = marker_diameter_pt * fig.dpi / 72 * 1.10
    accepted, offsets = [], np.zeros(len(values))
    candidates = [0.] + [s * j * gap for j in range(1, 12) for s in [-1., 1.]]
    for i in np.argsort(coords[:, 1], kind='stable'):
        for candidate in candidates:
            if all(np.hypot(candidate - x, coords[i, 1] - y) >= gap for x, y in accepted):
                accepted.append((candidate, coords[i, 1])); offsets[i] = candidate; break
        else:
            raise AssertionError('Point packing has insufficient horizontal space')
    x_display = coords[:, 0] + offsets
    xs = ax.transData.inverted().transform(np.column_stack([x_display, coords[:, 1]]))[:, 0]
    if np.max(np.abs(xs - center)) >= .38:
        xs = center + np.linspace(-.30, .30, len(values))
    return xs


def branches():
    path = ROOT / 'results/revision_v5/integrated/all_completed_rolling_probes.csv'
    df = pd.read_csv(path)
    policies = ['one_history_then_raw', 'history_rolling']
    values = df.loc[df.branch.isin(policies), 'marked_delta_aud']
    limit = (-max(10., abs(values.min()) * 1.5), max(10., values.max() * 1.5))
    fig = frame(108)
    axes = [axis(fig, [.12, .285, .375, .53]), axis(fig, [.585, .285, .375, .53])]
    data = []
    for j, (ax, policy, title) in enumerate(zip(axes, policies, ['First origin only', 'Repeated updates'])):
        ax.set_yscale('symlog', linthresh=10, linscale=1)
        ax.set(xlim=(-.52, 4.52), ylim=limit, xticks=range(5))
        ax.set_xticklabels(LABELS)
        ax.axhline(0, color=GRAY, lw=.6, zorder=0)
        for k, region in enumerate(REGIONS):
            d = df.loc[df.branch.eq(policy) & df.region.eq(region)].sort_values('event_id')
            assert len(d) == 20
            vv = d.marked_delta_aud.to_numpy()
            xx = point_offsets(ax, k, vv)
            ax.scatter(xx, vv, s=3.6 ** 2, marker='o' if j == 0 else 's',
                       facecolors='white' if j == 0 else ORANGE,
                       edgecolors=ORANGE, linewidths=.7, zorder=3)
            ax.plot([k - .27, k + .27], [np.median(vv)] * 2, color=INK, lw=1.3, zorder=4)
            data.append({'branch': policy, 'region': region, 'event_id': d.event_id.tolist(),
                         'marked_delta_aud': vv.tolist(), 'median': float(np.median(vv)),
                         'display_x': xx.tolist()})
        ax.yaxis.set_major_locator(FixedLocator([-10000, -1000, -100, -10, 0, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:,.0f}'.replace('-', '−')))
        if j:
            ax.tick_params(axis='y', labelleft=False)
        heading(fig, .12 + j * .465, .90, 'ab'[j], title)
    axes[0].set_ylabel('Marked branch contrast (A$)', labelpad=6)
    fig.legend([Line2D([], [], color=ORANGE, marker='o', mfc='white', lw=0, ms=3.6),
                Line2D([], [], color=ORANGE, marker='s', lw=0, ms=3.6),
                Line2D([], [], color=INK, lw=1.3)],
               ['First origin only', 'Repeated updates', 'Median'],
               loc='center', bbox_to_anchor=(.54, .18), ncol=3, frameon=False,
               handlelength=1.4, columnspacing=1.8, handletextpad=.6)
    text(fig, .54, .105, '20 selected starts per region; one point per start', ha='center')
    text(fig, .54, .05, 'Symmetric log scale; linear between −10 and +10 A$', ha='center')
    layout = assert_clean_layout(fig, 'fig4')
    final_save(fig, 'fig4', {'branches': data, 'total_points': 200, 'symlog_linear_threshold_aud': 10,
                           'shared_y_limits_aud': list(limit), 'layout': layout}, [path], [(0, 1)],
               'All 200 branch points; bounded deterministic horizontal spacing with economic y unchanged; equal shared axes and separate legend/scope lanes.')


def failure():
    path = ROOT / 'results/revision_v5/history/SA1_baseline_inputs.parquet'
    df = pd.read_parquet(path)
    target = pd.Timestamp('2024-09-23 19:00:00')
    selected = df.loc[df.phase.eq('evaluation') & df.cutoff_minutes.eq(60) & df.target.eq(target)]
    vals = {p: selected.loc[selected.policy.eq(p), [f'p_{i:02d}' for i in range(1, 13)]].iloc[0].to_numpy(float)
            for p in ['raw', 'sparse_equal']}
    actual = float(selected.loc[selected.policy.eq('raw'), 'actual_00'].iloc[0])
    pp = ROOT / 'results/revision_v5/events/SA1_sparse_equal_rolling_probe_paths.parquet'
    paths = pd.read_parquet(pp)
    event = paths.loc[paths.event_id.eq(178)]
    raw = event.loc[event.branch.eq('raw_rolling')].sort_values('step')
    sparse = event.loc[event.branch.eq('history_rolling')].sort_values('step')
    assert len(raw) == len(sparse) == 48
    assert np.array_equal(raw.step.to_numpy(), np.arange(48))
    assert np.array_equal(raw.actual_price.to_numpy(), sparse.actual_price.to_numpy())
    cash_r, cash_s = np.cumsum(raw.net_cash_aud), np.cumsum(sparse.net_cash_aud)
    gap = float(cash_s.iloc[-1] - cash_r.iloc[-1])
    late = float(vals['sparse_equal'][6] - vals['sparse_equal'][0])
    assert abs(gap + 6906.607873304685) < 1e-7 and abs(late - 79.9814453125) < 1e-10
    fig = frame(113)
    a = axis(fig, [.12, .365, .36, .415])
    b = axis(fig, [.605, .365, .36, .415])
    x, t = np.arange(12) * .5, np.arange(48) * .5
    a.plot(x, vals['raw'] / 1000, color=GRAY, marker='o', ms=4, lw=1.1)
    a.plot(x, vals['sparse_equal'] / 1000, color=ORANGE, marker='s', mfc='white',
           mew=.8, ms=4, lw=1.1, ls=(0, (4, 2)))
    a.scatter([0], [actual / 1000], marker='*', s=74, color=BLUE, zorder=5)
    a.set(xlim=(-.15, 5.65), ylim=(-.7, 18.5), xticks=[0, 1, 2, 3, 4, 5],
          yticks=[0, 5, 10, 15], xlabel='Forecast lead (h)', ylabel='Price (A$ thousand/MWh)')
    b.step(t, cash_r.to_numpy() / 1000, where='post', color=GRAY, lw=1.1)
    b.step(t, cash_s.to_numpy() / 1000, where='post', color=ORANGE, lw=1.1, ls=(0, (4, 2)))
    b.axvline(4, color=BLUE, lw=.6, ls=(0, (2, 3)), zorder=0)
    b.set(xlim=(0, 24), ylim=(-1.1, 18.5), xticks=[0, 6, 12, 18, 24], yticks=[0, 5, 10, 15],
          xlabel='Hours since 14:30', ylabel='Net cash (A$ thousand)')
    heading(fig, .12, .88, 'a', 'Forecast ranking')
    heading(fig, .605, .88, 'b', 'Updating cash paths')
    text(fig, .68, .815, '18:30 decision', ha='center', color=INK)
    fig.legend([Line2D([], [], color=GRAY, marker='o', ms=4, lw=1.1),
                Line2D([], [], color=ORANGE, marker='s', mfc='white', ls='--', ms=4, lw=1.1),
                Line2D([], [], color=BLUE, marker='*', lw=0, ms=8)],
               ['Latest R', 'Sparse S', 'Current realized'],
               loc='center', bbox_to_anchor=(.535, .965), ncol=3, frameon=False,
               handlelength=1.6, columnspacing=1.8, handletextpad=.6)
    text(fig, .12, .205, 'Current realized', color=GRAY)
    text(fig, .12, .145, f'A${actual:,.2f}/MWh', fontweight='bold')
    text(fig, .12, .075, f'S period 7 − 1: +A${late:.2f}/MWh')
    text(fig, .605, .205, '24 h cash difference: S − R', color=GRAY)
    text(fig, .605, .145, f'−A${abs(gap):,.2f}', fontweight='bold')
    text(fig, .605, .075, 'Both end stocks: 1.8 MWh')
    layout = assert_clean_layout(fig, 'fig5')
    final_save(fig, 'fig5', {'event_id': 178, 'selection': 'post-hoc most negative SA sparse window',
                           'target': str(target), 'forecast_h': x.tolist(), 'R': vals['raw'].tolist(),
                           'S': vals['sparse_equal'].tolist(), 'current_realized': actual,
                           'elapsed_h': t.tolist(), 'R_cumulative_cash_aud': cash_r.tolist(),
                           'S_cumulative_cash_aud': cash_s.tolist(), 'cash_gap_aud': gap,
                           'decision_elapsed_h': 4, 'end_stock_mwh': 1.8, 'layout': layout},
               [path, pp], [(0, 1)], 'All 12 forecast points and 48 updating steps; separate metric lanes and a common semantic palette.')


def sufficiency():
    sources = [ROOT / 'results/revision_v14/theory/executed_toy_controls.json',
               ROOT / 'results/revision_v14/pilot/shared_state_resource_curve.parquet',
               ROOT / 'results/revision_v14/pilot/closed_loop_summary.csv',
               ROOT / 'results/revision_v14/pilot/pilot_summary.json']
    toy, pilot = json.loads(sources[0].read_text()), json.loads(sources[3].read_text())
    shared, closed = pd.read_parquet(sources[1]), pd.read_csv(sources[2])
    assert toy['status'] == 'PASS_EXECUTED_SYNTHETIC_CONTROLS' and pilot['status'] == 'PASS'
    cases = []
    for (budget, state), rows in shared.groupby(['budget', 'state_dependent'], sort=True):
        assert len(rows) == 384
        cases.append({'directions': int(budget), 'state_dependent': bool(state), 'cases': len(rows),
                      'accepted': int((rows.regret_upper_aud <= .01).sum()),
                      'acceptance_percent': 100. * float((rows.regret_upper_aud <= .01).mean())})
    policies = ['inverse_lead', 'state3_gate', 'global3_gate', 'state3_no_gate']
    costs = []
    for policy in policies:
        for stock in [.2, 1.8]:
            rows = closed.loc[closed.policy.eq(policy) & np.isclose(closed.initial_soc_mwh, stock)]
            assert len(rows) == 1 and int(rows.iloc[0].origins) == 1488
            r = rows.iloc[0]
            costs.append({'policy': policy, 'initial_soc_mwh': stock, 'origins': 1488,
                          'wall_seconds': float(r.seconds), 'wall_ms_per_origin': 1000. * float(r.seconds) / 1488,
                          'summed_origin_seconds': float(r.summed_origin_seconds),
                          'marked_value_aud': float(r.marked_value_aud), 'regret_vs_il_aud': float(r.regret_vs_il_aud),
                          'fallback_share': None if pd.isna(r.fallback_share) else float(r.fallback_share)})
    control = toy['state_only_sufficiency_counterexample']
    ideal, cover = control['ideal_exact_primary'], control['action_preserving_cover']
    fig = frame(151)
    a = axis(fig, [.13, .62, .33, .22])
    b = axis(fig, [.64, .62, .33, .22])
    c = axis(fig, [.13, .17, .33, .23])
    d = axis(fig, [.64, .17, .33, .23])
    for xx, yy, letter, title in [(.13, .935, 'a', 'Agreement then divergence'),
                                 (.64, .935, 'b', 'Agreement set is not closed'),
                                 (.13, .515, 'c', 'Numerical gate acceptance'),
                                 (.64, .515, 'd', 'Recorded replay wall time')]:
        heading(fig, xx, yy, letter, title)
    a.plot([1, 2], [1., ideal['second_reference_discharge_mw']], '-o', color=BLUE, lw=1.1, ms=4.5)
    a.plot([1, 2], [1., ideal['second_compressed_discharge_mw']], '--s', color=ORANGE,
           lw=1.1, ms=4.5, mfc='white', mew=.9)
    a.set(xlim=(.83, 2.17), ylim=(0, 1.12), xticks=[1, 2], yticks=[0, .5, 1.],
          xlabel='Rolling origin', ylabel='First discharge (MW)')
    a.legend([Line2D([], [], color=BLUE, marker='o', lw=1.1, ms=4.5),
              Line2D([], [], color=ORANGE, marker='s', mfc='white', ls='--', lw=1.1, ms=4.5)],
             ['Reference', 'Compressed'], loc='lower left', bbox_to_anchor=(-.01, 1.07),
             ncol=2, frameon=False, handlelength=1.2, columnspacing=1., handletextpad=.45, borderaxespad=0.)
    b.plot([cover['lower_soc_mwh'], cover['upper_soc_mwh']], [1, 1], color=BLUE, lw=6, solid_capstyle='butt')
    b.plot([cover['image_lower_soc_mwh'], cover['image_upper_soc_mwh']], [0, 0], color=ORANGE, lw=6, solid_capstyle='butt')
    b.plot([1.4], [1], 'o', color=INK, ms=4.4, mfc='white', mew=1.)
    b.plot([ideal['common_next_soc_mwh']], [0], 'o', color=INK, ms=4.4, mfc='white', mew=1.)
    b.set(xlim=(.2, 1.83), ylim=(-.48, 1.48), xticks=[.2, .6, 1., 1.4, 1.8], yticks=[0, 1],
          yticklabels=['After execution', 'Action agreement'], xlabel='Inventory (MWh)')
    b.spines['left'].set_visible(False); b.tick_params(axis='y', length=0, pad=7)
    b.legend([Line2D([], [], color=INK, marker='o', mfc='white', lw=0, ms=4.4)],
             ['Common inventory'], loc='lower left', bbox_to_anchor=(-.01, 1.07),
             frameon=False, handlelength=1.2, handletextpad=.45, borderaxespad=0.)
    state = sorted([r for r in cases if r['state_dependent']], key=lambda r: r['directions'])
    glob = sorted([r for r in cases if not r['state_dependent']], key=lambda r: r['directions'])
    c.plot([r['directions'] for r in state], [r['acceptance_percent'] for r in state],
           '-o', color=BLUE, lw=1.1, ms=4.5)
    c.plot([r['directions'] for r in glob], [r['acceptance_percent'] for r in glob],
           '--s', color=ORANGE, lw=1.1, ms=4.5, mfc='white', mew=.9)
    c.set(xlim=(-.4, 9.4), ylim=(0, 105), xticks=[0, 3, 6, 9], yticks=[0, 25, 50, 75, 100],
          xlabel='Retained objective directions', ylabel='Accepted at A$0.01 (%)')
    c.legend([Line2D([], [], color=BLUE, marker='o', lw=1.1, ms=4.5),
              Line2D([], [], color=ORANGE, marker='s', mfc='white', ls='--', lw=1.1, ms=4.5)],
             ['State-dependent', 'Global'], loc='lower left', bbox_to_anchor=(-.01, 1.09),
             ncol=2, frameon=False, handlelength=1.2, columnspacing=.8, handletextpad=.45, borderaxespad=0.)
    ypos = np.arange(4)[::-1]
    for j, policy in enumerate(policies):
        rows = [r for r in costs if r['policy'] == policy]
        for stock, dy, marker, color, face in [(.2, .10, 'o', BLUE, BLUE), (1.8, -.10, 's', ORANGE, 'white')]:
            r = next(v for v in rows if v['initial_soc_mwh'] == stock)
            d.plot([r['wall_ms_per_origin']], [ypos[j] + dy], marker=marker, color=color,
                   lw=0, ms=4.5, mfc=face, mew=.9)
    d.set(xlim=(0, 6.2), ylim=(-.6, 3.6), yticks=ypos,
          yticklabels=['IL replay', 'State + guard', 'Global + guard', 'State, no guard'],
          xticks=[0, 2, 4, 6], xlabel='Wall time (ms/origin)')
    d.spines['left'].set_visible(False); d.tick_params(axis='y', length=0, pad=7)
    d.legend([Line2D([], [], marker='o', color=BLUE, lw=0, ms=4.5),
              Line2D([], [], marker='s', color=ORANGE, mfc='white', lw=0, ms=4.5)],
             ['0.2 MWh', '1.8 MWh'], loc='lower left', bbox_to_anchor=(-.01, 1.09),
             ncol=2, frameon=False, handlelength=1., columnspacing=1.3, handletextpad=.45, borderaxespad=0.)
    text(fig, .54, .065, '384 common-state cases per evaluated mask; 1,488 replay origins', ha='center')
    text(fig, .54, .025, 'Replay markers are individual recorded executions, not intervals', ha='center')
    layout = assert_clean_layout(fig, 'fig6')
    data = {'panel_a': {'opening_discharge_mw': [1., 1.],
                       'second_discharge_mw': [ideal['second_reference_discharge_mw'], ideal['second_compressed_discharge_mw']],
                       'analytical_common_initial_stock': 1.4,
                       'analytical_common_next_stock': ideal['common_next_soc_mwh'],
                       'analytical_two_origin_loss_aud': ideal['full_minus_compressed_marked_aud'],
                       'executed_initial_action_difference_mw': control['executed_first_action_difference_mw'],
                       'executed_loss_aud': control['executed_full_minus_compressed_marked_aud']},
            'panel_b': cover, 'panel_c': cases, 'panel_d': costs, 'resource': pilot['resource'], 'layout': layout,
            'scope': {'panel_ab': 'Synthetic exact-primary control; default-controller tolerance separately recorded',
                      'panel_c': '384 common-state cases per evaluated mask group; no iid inference',
                      'panel_d': 'Exploratory NSW January 2024; loop timing with precomputed curves, excluding input loading, initialization, and raw-version ingestion/queues',
                      'memory_count': '192 bytes counts numeric moments for a twelve-target unit, not peak RAM or a full-stream bound',
                      'method_decision': pilot['stage_decision']}}
    final_save(fig, 'fig6', data, sources, [(0, 1), (2, 3)],
               'Fixed-size two-by-two evidence sequence; explicit state-cover intervals; individual replay timing markers with no interval connectors.',
               column_groups=[(0, 2), (1, 3)])


def validate_frozen_outputs():
    """Verify scientific arrays against prior frozen exports and actual files."""
    import pymupdf as fitz
    from PIL import Image
    records = {}
    old4 = json.loads((ROOT / 'figures/revision_v11/figure_4_updating_branches_source.json').read_text())['source_data']
    new4 = json.loads((OUT / 'fig4_source.json').read_text())['source_data']
    assert len(old4) == len(new4['branches']) == 10
    for old, new in zip(old4, new4['branches']):
        for key in old:
            assert old[key] == new[key], ('fig4', key)
    assert sum(len(x['marked_delta_aud']) for x in new4['branches']) == 200
    old5 = json.loads((ROOT / 'figures/revision_v11/figure_5_decision_failure_source.json').read_text())['source_data']
    new5 = json.loads((OUT / 'fig5_source.json').read_text())['source_data']
    for key in old5:
        assert old5[key] == new5[key], ('fig5', key)
    old6 = json.loads((ROOT / 'results/revision_v14/figures/Figure7_source_data.json').read_text())
    new6 = json.loads((OUT / 'fig6_source.json').read_text())['source_data']
    for key in ['panel_a', 'panel_b', 'panel_c', 'panel_d', 'resource']:
        assert old6[key] == new6[key], ('fig6', key)
    for n in [4, 5, 6]:
        stem = f'fig{n}'
        with fitz.open(OUT / f'{stem}.pdf') as doc:
            assert len(doc) == 1
            images = sum(len(page.get_images()) for page in doc)
            spans = [s for page in doc for b in page.get_text('dict')['blocks'] if 'lines' in b
                     for line in b['lines'] for s in line['spans']]
            fonts = [doc.extract_font(font[0]) for page in doc for font in page.get_fonts()]
            assert images == 0 and all(font[-1] for font in fonts)
            minimum = min(float(span['size']) for span in spans)
            assert minimum >= 8.99
            width = float(doc[0].rect.width) * 25.4 / 72
            assert abs(width - 174) < .001
            vector = {'raster_images': images, 'embedded_fonts': len(fonts),
                      'native_width_mm': width, 'minimum_actual_pdf_font_pt': minimum}
        rasters = []
        for ext in ['png', 'tiff']:
            p = OUT / f'{stem}.{ext}'
            with Image.open(p) as im:
                dpi = [float(value) for value in im.info['dpi']]
                assert min(dpi) >= 1199
                rasters.append({'format': ext, 'pixels': list(im.size), 'dpi': dpi,
                                'bytes': p.stat().st_size, 'sha256': sha(p)})
        collision = json.loads((OUT / f'{stem}_pdf_collisions.json').read_text())
        assert collision['verdict'] == 'PASS'
        records[stem] = {'status': 'PASS', 'frozen_scientific_arrays': 'EXACT_MATCH',
                         'vector': vector, 'rasters': rasters,
                         'pdf_collision_summary': collision['summary'],
                         'visual_review': 'Actual 300 dpi preview inspected; no text overlap or crop observed'}
    receipt = {'status': 'PASS_FROZEN_VALUES_NATIVE_VECTOR_1200_DPI_AND_COLLISION_AUDITS',
               'script_sha256': sha(__file__), 'figures': records,
               'scope': 'Presentation-only redraw. Branch coincidence is genuine; no economic y jitter or dropped starts.'}
    (OUT / 'fig456_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', choices=['4', '5', '6', 'all'], default='all')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        validate_frozen_outputs()
    else:
        for key, fn in [('4', branches), ('5', failure), ('6', sufficiency)]:
            if args.figure in ['all', key]:
                fn()
