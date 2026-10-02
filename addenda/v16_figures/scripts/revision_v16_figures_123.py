"""Fixed-size v16 redraw of frozen figures 1--3; no scientific recomputation.

Uses the locally reviewed, pinned nature-skills alignment auditor. The physical
canvas, text sizes and annotation lanes are set before vector/raster export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'mpl_v16_123'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from revision_v16_figure_style import (RC as SHARED_RC, frame as shared_frame,
    axis as shared_axis, text as shared_text, heading as shared_heading,
    final_save as shared_save)

OUT = ROOT / 'figures' / 'revision_v16'
REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
LABELS = ['NSW', 'QLD', 'SA', 'TAS', 'VIC']
BLUE, ORANGE, TEAL = '#0072B2', '#D55E00', '#009E73'
INK, GRAY, LIGHT = '#171717', '#60666B', '#D6DADF'
RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.titlesize': 9,
    'axes.linewidth': .6, 'lines.linewidth': 1.1, 'lines.markersize': 3.6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.minor.visible': False, 'ytick.minor.visible': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'text.usetex': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.bbox': None,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.transparent': False,
}


RC = SHARED_RC
frame = shared_frame
axis = shared_axis
annotation = shared_text
heading = shared_heading


def final_save(fig, name, source_data, paths=(), groups=(), design=None,
               geometric_intent=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        points = [float(v * 72 / fig.dpi) for v in bbox.extents]
        text_artists.append({'text': artist.get_text(),
                             'font_pt': float(artist.get_fontsize()),
                             'bbox_pt': points})
    # The bounding-box audit includes figure-level annotations and both left/
    # right title objects; it does not rely on the center-only get_title().
    assert min(t['font_pt'] for t in text_artists) >= 9
    page_pt = np.asarray([fig.get_figwidth(), fig.get_figheight()]) * 72
    for t in text_artists:
        b = t['bbox_pt']
        assert b[0] >= -.25 and b[1] >= -.25, (name, t)
        assert b[2] <= page_pt[0] + .25 and b[3] <= page_pt[1] + .25, (name, t)
    overlap = []
    for i, a in enumerate(text_artists):
        for b in text_artists[i+1:]:
            ba, bb = a['bbox_pt'], b['bbox_pt']
            ox = min(ba[2], bb[2]) - max(ba[0], bb[0])
            oy = min(ba[3], bb[3]) - max(ba[1], bb[1])
            if ox > .15 and oy > .15:
                overlap.append({'a': a['text'], 'b': b['text'],
                                'overlap_pt': [ox, oy]})
    assert not overlap, (name, overlap)
    shared_save(fig, name, source_data, paths=paths, groups=groups, design=design)
    manifest_path = OUT / f'{name}_source.json'
    record = json.loads(manifest_path.read_text(encoding='utf-8'))
    record['text_pair_intersections'] = overlap
    record['geometric_intent'] = geometric_intent
    record['caption_change_required'] = False
    manifest_path.write_text(json.dumps(record, indent=2), encoding='utf-8')


def timing():
    fig = frame(82)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    heading(fig, .035, .947, 'a', 'Availability and execution')
    columns = [.235, .585, .920]
    line_y = .715
    # Connector geometry stays below all column headings and timestamp labels.
    canvas.plot([columns[0], columns[2]], [line_y, line_y],
                color=GRAY, lw=.8, transform=canvas.transAxes)
    for x, label, time in zip(columns,
                             ['Archive cutoff', 'Freshness cutoff', 'Delivery end'],
                             ['T − 60 min', 'T − 30 min', 'T']):
        canvas.plot([x, x], [line_y-.016, line_y+.016],
                    color=GRAY, lw=.8, transform=canvas.transAxes)
        annotation(fig, x, .849, label, ha='center')
        annotation(fig, x, .790, time, ha='center', fontweight='bold')
    canvas.add_patch(Rectangle((columns[1], .585), columns[2]-columns[1], .073,
                               facecolor='#E5EFF6', edgecolor='none',
                               transform=canvas.transAxes))
    annotation(fig, (columns[1]+columns[2])/2, .621,
               'Execute first action', ha='center')
    annotation(fig, .095, .502, 'Common inventory at T − 30 min')
    heading(fig, .035, .377, 'b', 'Twelve half-hour objective inputs')
    start, width, gap = .095, .0635, .006
    y, h = .239, .073
    for i in range(12):
        x = start + i * (width+gap)
        color = BLUE if i < 8 else TEAL
        canvas.add_patch(Rectangle((x, y), width, h,
                                   facecolor='#E5EFF6' if i < 8 else '#E7F4EE',
                                   edgecolor=color, lw=.6,
                                   transform=canvas.transAxes))
        annotation(fig, x + width/2, y+h/2, str(i+1), ha='center')
    annotation(fig, start + 3.5*(width+gap) + width/2, .171,
               'Trading: periods 1–8', ha='center')
    annotation(fig, start + 9.5*(width+gap) + width/2, .171,
               'Continuation: 9–12', ha='center')
    annotation(fig, .5, .063,
               'NEM time (UTC+10) · Modification time proxies availability', ha='center')
    final_save(fig, 'fig1', {'clock': {
        'archive_cutoff_min': -60, 'freshness_cutoff_min': -30,
        'execution_interval_min': [-30, 0], 'T': 'half-hour delivery end',
        'trading_periods': list(range(1, 9)),
        'continuation_periods': list(range(9, 13)), 'receipt_observed': False}},
        design='Schematic with separated labels, release timeline, execution strip and objective cells.',
        geometric_intent={
            'text_inside_colored_cells': 'Intentional cell labels; no cell boundary intersects text.',
            'timeline_ticks': 'Ticks are below timestamp labels, with a distinct execution lane.',
            'information_scope': 'Modification time proxy only; no observed participant receipt.'})


def economic():
    path = ROOT / 'results/revision_v5/control/paired_economic_comparisons.csv'
    df = pd.read_csv(path)
    fig = frame(83)
    contrasts = ['sparse_equal minus raw', 'sparse minus selected current-only',
                 'inverse_lead minus selected current-only']
    titles = ['S − R', 'S − C*', 'IL − C*']
    colors = [ORANGE, ORANGE, TEAL]
    source_data = []
    starts = [.106, .410, .714]
    for j, (contrast, title, color, x0) in enumerate(zip(contrasts, titles, colors, starts)):
        d = df[df.contrast.eq(contrast)].set_index('region').loc[REGIONS]
        estimate = d.mechanical_aud_per_mw_year.to_numpy()/1000
        low = d.mechanical_ci_low_aud_per_mw_year.to_numpy()/1000
        high = d.mechanical_ci_high_aud_per_mw_year.to_numpy()/1000
        assert np.all(low <= estimate) and np.all(estimate <= high)
        ax = axis(fig, [x0, .245, .233, .556])
        y = np.arange(len(REGIONS))
        ax.axvline(0, color=GRAY, lw=.6, ls=(0, (3, 3)), zorder=0)
        ax.errorbar(estimate, y, xerr=[estimate-low, high-estimate],
                    fmt='o' if j < 2 else 's', color=color, ecolor=color,
                    elinewidth=1.1, capsize=2.3, capthick=.8, markersize=4,
                    zorder=3)
        ax.set(xlim=(-20, 40), ylim=(4.6, -.6), xticks=[-20, 0, 20, 40], yticks=y)
        ax.set_yticklabels(LABELS if j == 0 else ['']*len(REGIONS))
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        heading(fig, x0, .908, chr(97+j), title)
        source_data.append({'contrast': contrast, 'region': REGIONS,
                            'estimate': estimate.tolist(), 'low': low.tolist(),
                            'high': high.tolist()})
    annotation(fig, .527, .131,
               'Model-value contrast (A$ thousand/MW-year)', ha='center')
    annotation(fig, .527, .062, 'Conditional seven-day paired-block intervals', ha='center')
    final_save(fig, 'fig2', source_data, [path], [(0, 1, 2)],
               design='Three aligned forest panels; identical scales and complete five-region intervals.',
               geometric_intent={
                   'statistic': 'Original conditional seven-day paired-block intervals only.',
                   'scale': 'Frozen mechanical annualization / 1000; no new inference.',
                   'labels': 'Panel and row names occupy dedicated outer lanes.'})


def allocation():
    fig = frame(102)
    source_data = []
    paths = [ROOT / f'results/revision_v6/integrated/v6_decomposition_{p}.csv'
             for p in ['sparse_equal', 'inverse_lead']]
    plot_axes = []
    y = np.array([0, 1, 2, 3, 4, 5.75])
    for j, (path, title) in enumerate(zip(paths, ['Sparse S − R', 'Inverse lead IL − R'])):
        df = pd.read_csv(path)
        extra = [r for r in df.region.unique() if r not in REGIONS]
        assert len(extra) == 1
        regions = REGIONS + extra
        data = df.pivot(index='region', columns='component', values='aud_per_nominal_day').loc[regions]
        T = data.trading_component_symmetric.to_numpy()
        C = data.terminal_component_symmetric.to_numpy()
        D = data.total_history_difference.to_numpy()
        assert np.max(np.abs(T+C-D)) < 1e-8
        bounds = [.139+j*.458, .231, .279, .572]
        ax = axis(fig, bounds)
        plot_axes.append(len(fig.axes)-1)
        ax.axvline(0, color=GRAY, lw=.6, zorder=0)
        positive, negative = np.zeros(6), np.zeros(6)
        for values, color in [(T, BLUE), (C, TEAL)]:
            left = np.where(values >= 0, positive, negative)
            ax.barh(y, values, left=left, height=.49, color=color,
                    edgecolor='none', zorder=2)
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        ax.scatter(D, y, marker='D', s=17, color=INK, zorder=4)
        ax.set(xlim=(-20, 195), ylim=(6.35, -.70), xticks=[0, 50, 100, 150], yticks=y)
        ax.set_yticklabels(LABELS+['Five-asset sum'] if j == 0 else ['']*6)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.axhline(4.90, color=LIGHT, lw=.6, zorder=0)
        heading(fig, bounds[0], .866, chr(97+j), title)
        # A second axes is an annotation lane, never a white cover over bars.
        lane = fig.add_axes([bounds[0]+bounds[2]+.011, bounds[1], .066, bounds[3]])
        lane.set_axis_off()
        lane.set_xlim(0, 1)
        lane.set_ylim(ax.get_ylim())
        lane.text(.98, -.70, 'D', ha='right', va='bottom', fontsize=9, fontweight='bold')
        for yy, total in zip(y, D):
            lane.text(.98, yy, f'{total:.2f}'.replace('-', '−'), ha='right', va='center',
                      fontsize=9, fontweight='bold' if yy == y[-1] else 'normal')
        source_data.append({'policy': path.stem, 'region': regions,
                            'T': T.tolist(), 'C': C.tolist(), 'D': D.tolist()})
    fig.legend([Rectangle((0, 0), 1, 1, facecolor=BLUE),
                Rectangle((0, 0), 1, 1, facecolor=TEAL),
                Line2D([], [], color=INK, marker='D', ls='none', ms=4)],
               ['Trading T', 'Continuation C', 'Total D'], loc='center',
               bbox_to_anchor=(.535, .955), frameon=False, ncol=3,
               handlelength=1.5, columnspacing=1.8, handletextpad=.6, borderaxespad=0)
    annotation(fig, .535, .127, 'Model-value allocation (A$/nominal day)', ha='center')
    annotation(fig, .535, .061, 'T + C = D · Five-asset row is a sum, not an average', ha='center')
    final_save(fig, 'fig3', source_data, paths, [tuple(plot_axes)],
               design='Signed component panels; dedicated total columns; aggregate sum separated from regional rows.',
               geometric_intent={
                   'total_values': 'Aligned in independent blank-space lanes, not over data bars.',
                   'interaction': 'Already divided between T and C; never stacked again.',
                   'aggregate': 'Five independent asset accounting values summed; not statistical independence.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', nargs='+', choices=['1', '2', '3'], default=['1', '2', '3'])
    args = parser.parse_args()
    with plt.rc_context(RC):
        for number in args.figures:
            {'1': timing, '2': economic, '3': allocation}[number]()
