"""Nature-style, fixed-size figure redraws from frozen storage-study inputs.

No fitting, selection, inference, interpolation or additional experiment is run.
Style references and AOR overrides are recorded separately in the release bundle.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.text import Text
from nature_qa_v11.audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/revision_v11'
REGIONS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
LABELS = ['NSW', 'QLD', 'SA', 'TAS', 'VIC']
BLUE, ORANGE, TEAL = '#0072B2', '#D55E00', '#009E73'
INK, GRAY, LIGHT = '#171717', '#60666B', '#D6DADF'
RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'axes.titlesize': 9,
    'axes.linewidth': .55, 'lines.linewidth': 1.1, 'lines.markersize': 3.2,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .55, 'ytick.major.width': .55,
    'xtick.minor.visible': False, 'ytick.minor.visible': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'text.usetex': False, 'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.bbox': None, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.transparent': False,
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def src(relative):
    return ROOT / relative

def frame(height_mm):
    width_inches = 160 / 25.4
    return plt.figure(figsize=(width_inches, height_mm / 25.4), dpi=300)

def text(fig, x, y, value, **kw):
    kw.setdefault('fontsize', 9)
    return fig.text(x, y, value, va='center', **kw)

def heading(fig, x, y, letter, value):
    text(fig, x, y, letter, fontweight='bold', fontsize=10)
    text(fig, x + .035, y, value, fontweight='bold')

def legend(fig, handles, labels, y=.97, ncol=None):
    return fig.legend(handles, labels, loc='center', bbox_to_anchor=(.52, y),
                      frameon=False, ncol=ncol or len(labels), handlelength=1.7,
                      columnspacing=1.5, handletextpad=.55, borderaxespad=0)

def axis(fig, bounds):
    ax = fig.add_axes(bounds)
    ax.tick_params(direction='out', pad=4)
    return ax

def input_record(paths):
    return [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]

def final_save(fig, name, data, paths=(), groups=(), design=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Final physical plot-area rectangles are measured, not inferred from ticks.
    measured = [[float(v * scale) for v, scale in zip(ax.get_position().bounds,
                (fig.get_figwidth()*72, fig.get_figheight()*72,
                 fig.get_figwidth()*72, fig.get_figheight()*72))] for ax in fig.axes]
    align = []
    for group in groups:
        rects = [measured[i] for i in group]
        deviation = max(np.ptp([r[k] for r in rects]) for k in (1, 2, 3))
        assert deviation <= 1.5, (name, group, deviation)
        align.append({'axes': list(group), 'same_row_deviation_pt': float(deviation),
                      'tolerance_pt': 1.5, 'status': 'pass'})
    # Capture all visible text artists, including figure-level annotation lanes.
    renderer = fig.canvas.get_renderer()
    artists = []
    for t in fig.findobj(match=Text):
        if t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(renderer)
            artists.append({'text': t.get_text(), 'font_pt': t.get_fontsize(),
                            'bbox_pt': [float(v*72/fig.dpi) for v in b.extents]})
    selected = sorted({i for g in groups for i in g}) if groups else list(range(len(fig.axes)))
    panel_ids = [f'p{i}' for i in selected]
    require_matplotlib_panel_alignment(
        fig, axes=[fig.axes[i] for i in selected], panel_ids=panel_ids,
        row_groups=[[f'p{i}' for i in g] for g in groups] if groups else None,
        json_out=OUT / f'{name}.alignment.json', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True, require_panel_labels=False,
    )
    # Figure-level letters are checked separately: trajectory columns have four axes.
    fig.savefig(OUT / f'{name}.pdf', bbox_inches=None)
    fig.savefig(OUT / f'{name}.svg', bbox_inches=None)
    fig.savefig(OUT / f'{name}.png', dpi=1200, bbox_inches=None)
    fig.savefig(OUT / f'{name}.tiff', dpi=1200, bbox_inches=None,
                pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUT / f'{name}_preview.png', dpi=300, bbox_inches=None)
    record = {'name': name, 'width_mm': 160, 'height_mm': fig.get_figheight()*25.4,
              'raster_dpi': 1200, 'vector_formats': ['pdf', 'svg'],
              'inputs': input_record(paths), 'source_data': data,
              'axes_rectangles_pt': measured, 'alignment_groups': align,
              'rendered_text_artists': artists, 'design': design,
              'status': 'exported; fresh PDF geometry and visual QA required'}
    record['exports'] = [{'path': (OUT/f'{name}.{s}').name,
                          'sha256': sha(OUT/f'{name}.{s}'),
                          'bytes': (OUT/f'{name}.{s}').stat().st_size} for s in ('pdf','svg','png','tiff')]
    (OUT / f'{name}_source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    plt.close(fig)
    print(name, 'exported', flush=True)

def timing():
    fig = frame(75)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    heading(fig, .035, .95, 'a', 'Availability and execution clock')
    x0,x1,x2 = .15,.56,.88
    y=.67
    ax.plot([x0,x2],[y,y],color=GRAY,lw=.8,transform=ax.transAxes)
    for x in [x0,x1,x2]:
        ax.plot([x,x],[y-.018,y+.018],color=GRAY,lw=.8,transform=ax.transAxes)
    for x,label in [(x0,'T − 60 min'),(x1,'T − 30 min'),(x2,'T')]:
        text(fig,x,.745,label,ha='center',fontweight='bold')
    text(fig,x0,.825,'Archive cutoff',ha='center')
    text(fig,x1,.825,'Freshness cutoff',ha='center')
    text(fig,x2,.825,'Delivery end',ha='center')
    ax.add_patch(Rectangle((x1,.555),x2-x1,.064,facecolor='#E5EFF6',
                           edgecolor='none',transform=ax.transAxes))
    text(fig,(x1+x2)/2,.587,'Execute first action',ha='center')
    text(fig,.14,.48,'Both clocks start from the inventory at T − 30 min')
    heading(fig,.035,.355,'b','Twelve half-hour forecast inputs')
    start=.10; width=.064; gap=.005; yy=.225; hh=.065
    for k in range(12):
        col=BLUE if k<8 else TEAL
        ax.add_patch(Rectangle((start+k*(width+gap),yy),width,hh,
                              edgecolor=col,facecolor='white',lw=.7,transform=ax.transAxes))
        text(fig,start+k*(width+gap)+width/2,yy+hh/2,str(k+1),ha='center')
    text(fig,start+3.5*(width+gap)+width/2,.16,'Trading inputs: periods 1–8',ha='center')
    text(fig,start+9.5*(width+gap)+width/2,.16,'Continuation: 9–12',ha='center')
    text(fig,.50,.065,'NEM time (UTC+10) · Modification time proxies availability',ha='center')
    final_save(fig,'figure_1_information_timing',{'clock':{'archive_cutoff_min':-60,
        'freshness_cutoff_min':-30,'execution_interval_min':[-30,0],
        'T':'half-hour delivery end','trading_periods':list(range(1,9)),
        'continuation_periods':list(range(9,13)), 'receipt_observed':False}},
        design='Schematic-led sequence; clock and forecast blocks occupy separate lanes.')

def economic():
    path=src('results/revision_v5/control/paired_economic_comparisons.csv')
    df=pd.read_csv(path); fig=frame(80)
    contrasts=['sparse_equal minus raw','sparse minus selected current-only',
               'inverse_lead minus selected current-only']
    headings=['S − R','S − C*','IL − C*']; cols=[ORANGE,ORANGE,TEAL]
    data=[]
    for j,(contrast,title,color) in enumerate(zip(contrasts,headings,cols)):
        dd=df[df.contrast.eq(contrast)].set_index('region').loc[REGIONS]
        m=dd.mechanical_aud_per_mw_year.to_numpy()/1000
        lo=dd.mechanical_ci_low_aud_per_mw_year.to_numpy()/1000
        hi=dd.mechanical_ci_high_aud_per_mw_year.to_numpy()/1000
        assert np.all(lo<=m) and np.all(m<=hi)
        ax=axis(fig,[.115+j*.294,.24,.245,.57])
        y=np.arange(5)
        ax.axvline(0,color=GRAY,lw=.65,ls=(0,(3,3)),zorder=0)
        ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt='o' if j<2 else 's',
                    color=color,ecolor=color,elinewidth=1.1,capsize=2.3,
                    capthick=.8,markersize=4,zorder=3)
        ax.set(xlim=(-20,40),ylim=(4.6,-.6),xticks=[-20,0,20,40],yticks=y)
        ax.set_yticklabels(LABELS if j==0 else ['']*5)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
        heading(fig,.115+j*.294,.90,chr(97+j),title)
        data.append({'contrast':contrast,'region':REGIONS,'estimate':m.tolist(),
                     'low':lo.tolist(),'high':hi.tolist()})
    text(fig,.54,.115,'Model-value contrast (A$ thousand/MW-year)',ha='center')
    final_save(fig,'figure_2_regional_economic_contrasts',data,[path],[(0,1,2)],
        'Quantitative comparison grid; identical scales; original conditional intervals.')

def allocation():
    fig=frame(94); data=[]; axes_data=[]
    paths=[src(f'results/revision_v6/integrated/v6_decomposition_{p}.csv')
           for p in ['sparse_equal','inverse_lead']]
    for j,(path,title) in enumerate(zip(paths,['Sparse − latest','Inverse lead − latest'])):
        df=pd.read_csv(path)
        extra=[r for r in df.region.unique() if r not in REGIONS]
        assert len(extra)==1
        regions=REGIONS+extra
        d=df.pivot(index='region',columns='component',values='aud_per_nominal_day').loc[regions]
        T=d.trading_component_symmetric.to_numpy(); C=d.terminal_component_symmetric.to_numpy()
        D=d.total_history_difference.to_numpy()
        assert np.max(np.abs(T+C-D))<1e-8
        bounds=[.135+j*.47,.23,.30,.56]
        ax=axis(fig,bounds); axes_data.append(len(fig.axes)-1); y=np.arange(6)
        ax.axvline(0,color=GRAY,lw=.65,zorder=0)
        pos=np.zeros(6); neg=np.zeros(6)
        for vals,col in [(T,BLUE),(C,TEAL)]:
            left=np.where(vals>=0,pos,neg)
            ax.barh(y,vals,left=left,height=.52,color=col,edgecolor='none',zorder=2)
            pos+=np.maximum(vals,0); neg+=np.minimum(vals,0)
        ax.scatter(D,y,marker='D',s=17,color=INK,zorder=4)
        ax.set(xlim=(-20,195),ylim=(5.75,-.65),xticks=[0,50,100,150],yticks=y)
        ax.set_yticklabels(LABELS+['Five assets'] if j==0 else ['']*6)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.axhline(4.55,color=LIGHT,lw=.55,zorder=0)
        heading(fig,.135+j*.47,.86,chr(97+j),title)
        # Totals are placed in an explicit lane outside the data rectangles.
        lane=fig.add_axes([bounds[0]+bounds[2]+.013,bounds[1],.06,bounds[3]])
        lane.set_axis_off();lane.set_ylim(ax.get_ylim());lane.set_xlim(0,1)
        lane.text(.98,-.91,'D',ha='right',va='center',fontsize=9,fontweight='bold')
        for yy,dd in zip(y,D):
            lane.text(.98,yy,f'{dd:.2f}',ha='right',va='center',fontsize=9,
                      fontweight='bold' if yy==5 else 'normal')
        data.append({'policy':path.stem,'region':regions,'T':T.tolist(),'C':C.tolist(),'D':D.tolist()})
    legend(fig,[Rectangle((0,0),1,1,color=BLUE),Rectangle((0,0),1,1,color=TEAL),
                Line2D([],[],color=INK,marker='D',ls='none',ms=4)],
           ['Trading T','Continuation C','Total D'],y=.96)
    text(fig,.54,.115,'Model-value allocation (A$/nominal day)',ha='center')
    final_save(fig,'figure_3_channel_allocation',data,paths,[tuple(axes_data)],
        'Signed component grid with total values in independent annotation lanes; I is not restacked.')

def branches():
    path=src('results/revision_v5/integrated/all_completed_rolling_probes.csv')
    df=pd.read_csv(path); fig=frame(95);data=[]
    policies=['one_history_then_raw','history_rolling']
    combined=df[df.branch.isin(policies)].marked_delta_aud
    lim=(-max(10,abs(combined.min())*1.5),max(10,combined.max()*1.5))
    for j,(policy,title) in enumerate(zip(policies,['First-origin-only sparse','Repeated sparse'])):
        ax=axis(fig,[.15+j*.44,.23,.35,.58])
        ax.set_yscale('symlog',linthresh=10,linscale=1)
        ax.set_ylim(lim);ax.axhline(0,color=GRAY,lw=.65,zorder=0)
        for k,region in enumerate(REGIONS):
            d=df[df.branch.eq(policy)&df.region.eq(region)].sort_values('event_id')
            assert len(d)==20
            v=d.marked_delta_aud.to_numpy(); x=k+np.linspace(-.15,.15,20)
            ax.scatter(x,v,s=15,facecolors='white' if j==0 else ORANGE,
                       edgecolors=ORANGE,lw=.65,zorder=3)
            ax.plot([k-.22,k+.22],[np.median(v)]*2,color=INK,lw=1.15,zorder=4)
            data.append({'branch':policy,'region':region,'event_id':d.event_id.tolist(),
                         'marked_delta_aud':v.tolist(),'median':float(np.median(v))})
        ax.set(xlim=(-.55,4.55),xticks=range(5));ax.set_xticklabels(LABELS)
        ax.yaxis.set_major_locator(FixedLocator([-10000,-1000,-100,-10,0,10,100,1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'.replace('-','−')))
        if j==1:ax.tick_params(axis='y',labelleft=False)
        heading(fig,.15+j*.44,.90,chr(97+j),title)
    fig.axes[0].set_ylabel('Marked branch contrast (A$)',labelpad=6)
    text(fig,.54,.12,'20 selected starts per region · Black bars: medians',ha='center')
    final_save(fig,'figure_4_updating_branches',data,[path],[(0,1)],
        'Paired branch grid; complete nonrandom sparse sample; shared symmetric-log scale ±A$10 linear.')

def case_paths(region,event):
    path=src(f'results/revision_v5/events/{region}_sparse_equal_rolling_probe_paths.parquet')
    df=pd.read_parquet(path); d=df[df.event_id.eq(event)]
    raw=d[d.branch.eq('raw_rolling')].sort_values('step')
    hist=d[d.branch.eq('history_rolling')].sort_values('step')
    assert len(raw)==len(hist)==48
    assert np.array_equal(raw.step.to_numpy(),np.arange(48))
    assert np.array_equal(raw.actual_price.to_numpy(),hist.actual_price.to_numpy())
    return path,raw,hist

def failure():
    path=src('results/revision_v5/history/SA1_baseline_inputs.parquet')
    df=pd.read_parquet(path)
    target=pd.Timestamp('2024-09-23 19:00:00')
    d=df[df.phase.eq('evaluation')&df.cutoff_minutes.eq(60)&df.target.eq(target)]
    vals={p:d[d.policy.eq(p)][[f'p_{i:02d}' for i in range(1,13)]].iloc[0].to_numpy(dtype=float)
          for p in ['raw','sparse_equal']}
    actual=float(d[d.policy.eq('raw')].actual_00.iloc[0])
    pp,raw,hist=case_paths('SA1',178)
    cas=np.cumsum(raw.net_cash_aud.to_numpy());cah=np.cumsum(hist.net_cash_aud.to_numpy())
    gap=cah[-1]-cas[-1]
    assert abs(gap+6906.607873304685)<1e-7
    assert abs(vals['sparse_equal'][6]-vals['sparse_equal'][0]-79.9814453125)<1e-10
    fig=frame(100)
    axa=axis(fig,[.115,.37,.345,.40]);axb=axis(fig,[.615,.37,.345,.40])
    x=np.arange(12)*.5
    for p,col,marker,ls in [('raw',GRAY,'o','-'),('sparse_equal',ORANGE,'s',(0,(4,2)))]:
        axa.plot(x,vals[p]/1000,color=col,marker=marker,ls=ls,ms=3.5,lw=1.1)
    axa.scatter([0],[actual/1000],marker='*',s=58,c=BLUE,zorder=5)
    axa.set(xlim=(-.15,5.65),ylim=(-.7,18.5),xticks=[0,1,2,3,4,5],yticks=[0,5,10,15])
    axa.set_xlabel('Forecast lead (h)',labelpad=6)
    axa.set_ylabel('Price (A$ thousand/MWh)',labelpad=6)
    t=np.arange(48)*.5
    axb.step(t,cas/1000,where='post',color=GRAY,lw=1.2)
    axb.step(t,cah/1000,where='post',color=ORANGE,lw=1.2,ls=(0,(4,2)))
    axb.axvline(4,color=BLUE,lw=.65,ls=(0,(2,3)),zorder=0)
    axb.set(xlim=(0,24),ylim=(-1.1,18.5),xticks=[0,6,12,18,24],yticks=[0,5,10,15])
    axb.set_xlabel('Hours since 14:30',labelpad=6)
    axb.set_ylabel('Net cash (A$ thousand)',labelpad=6)
    heading(fig,.115,.90,'a','Forecast opportunity ranking')
    heading(fig,.615,.90,'b','Updating cash paths')
    text(fig,.673,.815,'18:30 decision',ha='center')
    legend(fig,[Line2D([],[],color=GRAY,marker='o',ms=3.5),
                Line2D([],[],color=ORANGE,marker='s',ls='--',ms=3.5),
                Line2D([],[],color=BLUE,marker='*',ls='none',ms=7)],
           ['Latest R','Sparse S','Current realized'],y=.975)
    text(fig,.105,.205,f'Current realized: A${actual:,.2f}/MWh')
    text(fig,.105,.14,f'S period 7 − 1: +A${vals["sparse_equal"][6]-vals["sparse_equal"][0]:.2f}/MWh')
    text(fig,.605,.205,f'24 h S − R: −A${abs(gap):,.2f}')
    text(fig,.605,.14,'Both end stocks: 1.8 MWh')
    final_save(fig,'figure_5_decision_failure',{'event_id':178,'selection':'post-hoc most negative SA sparse window',
        'target':str(target),'forecast_h':x.tolist(),'R':vals['raw'].tolist(),'S':vals['sparse_equal'].tolist(),
        'current_realized':actual,'elapsed_h':t.tolist(),'R_cumulative_cash_aud':cas.tolist(),
        'S_cumulative_cash_aud':cah.tolist(),'cash_gap_aud':float(gap),
        'decision_elapsed_h':4,'end_stock_mwh':1.8},[path,pp],[(0,1)],
        'Forecast mechanism and actual updating consequence; all explanatory text outside plot envelopes.')

def protocol():
    fig=frame(90);ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    titles=['Define the comparison','Choose a comparator','Layer A: input allocation',
            'Layer B: action response','Layer C: updating branches','Report value and costs']
    bodies=[['Replaceable input blocks','Common feasible set and clock'],
            ['Attainable prior-selected rule','Fix controller and accounting'],
            ['Four chronological replays','T + C = D; each carries stock'],
            ['Common-state outer ranges','Tolerance; overlap inconclusive'],
            ['First-only versus sustained use','Failures; no summing windows'],
            ['Comparator-specific value','Adoption needs measured costs']]
    positions=[(.04,.72),(.55,.72),(.55,.44),(.04,.44),(.04,.16),(.55,.16)]
    w,h=.41,.215
    for i,((xx,yy),title,body) in enumerate(zip(positions,titles,bodies)):
        ax.add_patch(Rectangle((xx,yy),w,h,facecolor='white',edgecolor=LIGHT,lw=.65,transform=ax.transAxes))
        text(fig,xx+.025,yy+.16,str(i+1),fontsize=10,fontweight='bold')
        text(fig,xx+.065,yy+.16,title,fontweight='bold')
        text(fig,xx+.025,yy+.098,body[0])
        text(fig,xx+.025,yy+.050,body[1])
    for i in range(5):
        x1,y1=positions[i];x2,y2=positions[i+1]
        if y1==y2:
            if x2>x1:a=(x1+w+.010,y1+h/2);b=(x2-.010,y2+h/2)
            else:a=(x1-.010,y1+h/2);b=(x2+w+.010,y2+h/2)
        else:a=(x1+w/2,y1-.010);b=(x2+w/2,y2+h+.010)
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,
                                    lw=.7,color=GRAY,transform=ax.transAxes))
    text(fig,.50,.073,'Static certificates anchor layer B.',ha='center')
    text(fig,.50,.028,'Layer C measures consequences outside those bounds.',ha='center')
    final_save(fig,'figure_6_rdia_protocol',{'steps':[{'title':t,'body':b} for t,b in zip(titles,bodies)]},
        design='Six-step schematic in a two-column serpentine sequence; connectors restricted to empty gutters.')

def illustrations():
    fig=frame(171);data=[];paths=[];rects=[]
    cases=[('NSW1',518,'NSW · 19 August 2025'),('SA1',178,'SA · 23 September 2024')]
    for col,(region,event,label) in enumerate(cases):
        path,raw,hist=case_paths(region,event);paths.append(path); t=np.arange(48)*.5
        heading(fig,.13+col*.445,.965,chr(97+col),label)
        axes=[]
        for row,yy in enumerate([.695,.505,.315,.125]):
            ax=axis(fig,[.13+col*.445,yy,.36,.135]);axes.append(ax);rects.append(len(fig.axes)-1)
            ax.set(xlim=(0,24),xticks=[0,6,12,18,24]); ax.tick_params(axis='x',labelbottom=row==3)
            if row==0:
                ax.step(t,raw.actual_price.to_numpy()/1000,where='post',color=GRAY,lw=1.1)
            elif row==1:
                for d,c,l in [(raw,GRAY,'-'),(hist,ORANGE,(0,(4,2)))]:
                    ax.step(t,(d.discharge_mw-d.charge_mw).to_numpy(),where='post',color=c,ls=l,lw=1.1)
                ax.set(ylim=(-1.15,1.15),yticks=[-1,0,1]);ax.axhline(0,color=LIGHT,lw=.5,zorder=0)
            elif row==2:
                for d,c,l in [(raw,GRAY,'-'),(hist,ORANGE,(0,(4,2)))]:
                    ax.step(t,d.soc_start.to_numpy(),where='post',color=c,ls=l,lw=1.1)
                ax.set(ylim=(.1,1.87),yticks=[.2,1,1.8])
            else:
                cash=np.cumsum(hist.net_cash_aud.to_numpy()-raw.net_cash_aud.to_numpy())
                ax.step(t,cash/1000,where='post',color=BLUE,lw=1.1)
                ax.axhline(0,color=LIGHT,lw=.5,zorder=0)
                ax.set_xlabel('Hours since branch start',labelpad=6)
            if col==0:
                ax.set_ylabel(['Price\n(A$ thousand/MWh)','Net discharge\n(MW)',
                               'Inventory\n(MWh)','Cash S − R\n(A$ thousand)'][row],labelpad=6)
        data.append({'region':region,'event_id':event,'elapsed_h':t.tolist(),
          'actual_price':raw.actual_price.tolist(),
          'R_net_discharge_mw':(raw.discharge_mw-raw.charge_mw).tolist(),
          'S_net_discharge_mw':(hist.discharge_mw-hist.charge_mw).tolist(),
          'R_soc_start_mwh':raw.soc_start.tolist(),'S_soc_start_mwh':hist.soc_start.tolist(),
          'cash_difference_aud':np.cumsum(hist.net_cash_aud.to_numpy()-raw.net_cash_aud.to_numpy()).tolist(),
          'selection':'post-hoc illustrative window, not random'})
    legend(fig,[Line2D([],[],color=GRAY),Line2D([],[],color=ORANGE,ls='--'),
                Line2D([],[],color=BLUE)],['Latest R','Sparse S','Cash contrast S − R'],y=.91)
    text(fig,.52,.025,'Illustrative post-hoc windows · Cash trajectories exclude terminal marking',ha='center')
    groups=[(0,4),(1,5),(2,6),(3,7)]
    final_save(fig,'figure_S1_updated_rolling_illustrations',data,paths,groups,
        'Two illustrative trajectory columns; four evidence rows; cash and marked value are distinguished.')

FUNCTIONS={'1':timing,'2':economic,'3':allocation,'4':branches,'5':failure,'6':protocol,'S1':illustrations}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='*',default=list(FUNCTIONS))
    args=parser.parse_args()
    with plt.rc_context(RC):
        for key in args.figures:FUNCTIONS[key]()
