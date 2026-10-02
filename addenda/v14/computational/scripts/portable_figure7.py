"""Figure 7: executed state-coverage and resource-boundary diagnostics.

Native vector PDF/SVG and independently rendered 1200 dpi PNG/TIFF. Numerical
data come from executed controls/pilot files, not from the figure's labels.
"""
from __future__ import annotations
import os
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]


def _fresh_figure_output():
    parser=argparse.ArgumentParser(description='Render Figure 7 from supplied frozen source data, without rerunning optimization.')
    parser.add_argument('--output',type=Path,required=True,help='New output directory outside the read-only package; must not already exist.')
    args=parser.parse_args()
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:
        parser.error('Output must overlap neither the read-only package nor its ancestors.')
    if out.exists():
        parser.error('Output must be a new directory that does not already exist.')
    out.mkdir(parents=True,exist_ok=False)
    return out


OUT=_fresh_figure_output()
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_config')
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image
import fitz


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bbox_list(b):return [float(x) for x in b.extents]


def overlap(a,b):
    return max(0.,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0.,min(a.y1,b.y1)-max(a.y0,b.y0))


def main():
    began=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    theory=ROOT/'results/revision_v14/theory/executed_toy_controls.json'
    shared=ROOT/'results/revision_v14/pilot/shared_state_resource_curve.parquet'
    replay=ROOT/'results/revision_v14/pilot/closed_loop_summary.csv'
    summary=ROOT/'results/revision_v14/pilot/pilot_summary.json'
    toy=json.loads(theory.read_text());pilot=json.loads(summary.read_text())
    df=pd.read_parquet(shared);closed=pd.read_csv(replay)
    assert toy['status']=='PASS_EXECUTED_SYNTHETIC_CONTROLS' and pilot['status']=='PASS'
    cases=[]
    for (budget,state),f in df.groupby(['budget','state_dependent'],sort=True):
        assert len(f)==384
        cases.append({'directions':int(budget),'state_dependent':bool(state),'cases':len(f),
                      'accepted':int((f.regret_upper_aud<=.01).sum()),
                      'acceptance_percent':100.*float((f.regret_upper_aud<=.01).mean())})
    policies=['inverse_lead','state3_gate','global3_gate','state3_no_gate']
    costs=[]
    for p in policies:
        for stock in [.2,1.8]:
            f=closed.loc[(closed.policy==p)&np.isclose(closed.initial_soc_mwh,stock)]
            assert len(f)==1 and int(f.iloc[0].origins)==1488
            r=f.iloc[0]
            costs.append({'policy':p,'initial_soc_mwh':stock,'origins':1488,
                'wall_seconds':float(r.seconds),'wall_ms_per_origin':1000.*float(r.seconds)/1488,
                'summed_origin_seconds':float(r.summed_origin_seconds),
                'marked_value_aud':float(r.marked_value_aud),'regret_vs_il_aud':float(r.regret_vs_il_aud),
                'fallback_share':None if pd.isna(r.fallback_share)else float(r.fallback_share)})
    control=toy['state_only_sufficiency_counterexample']
    ideal=control['ideal_exact_primary'];cover=control['action_preserving_cover']
    sources=[{'file':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size}
             for p in [theory,shared,replay,summary]]
    data={'sources':sources,'panel_a':{'opening_discharge_mw':[1.,1.],
            'second_discharge_mw':[ideal['second_reference_discharge_mw'],ideal['second_compressed_discharge_mw']],
            'analytical_common_initial_stock':1.4,'analytical_common_next_stock':ideal['common_next_soc_mwh'],
            'analytical_two_origin_loss_aud':ideal['full_minus_compressed_marked_aud'],
            'executed_initial_action_difference_mw':control['executed_first_action_difference_mw'],
            'executed_loss_aud':control['executed_full_minus_compressed_marked_aud']},
        'panel_b':cover,'panel_c':cases,'panel_d':costs,'resource':pilot['resource'],
        'scope':{'panel_ab':'synthetic exact-primary control, default controller tolerance separately recorded',
            'panel_c':'384 shared-state cases per actually evaluated direction/mask group; no iid inference',
            'panel_d':'exploratory NSW January 2024 replays; origin-loop wall time from precomputed input curves on one host, excluding input loading, initialization and raw-version ingestion/queue; not CPU or production cost',
            'memory_count':'192 bytes is an analytical numeric-moment count for a twelve-target accounting unit, not measured peak RAM or an upper bound for all active targets; a complete stream may have more active targets',
            'method_decision':pilot['stage_decision']}}
    (OUT/'Figure7_source_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

    plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,
        'xtick.labelsize':9.2,'ytick.labelsize':9.2,'axes.titlesize':10.,
        'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':3.,'ytick.major.size':3.,'legend.fontsize':9.2,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white','figure.facecolor':'white','axes.unicode_minus':False})
    blue='#0072B2';orange='#D55E00';neutral='#444444';muted='#B6C1C8'
    fig,axes=plt.subplots(2,2,figsize=(178/25.4,143/25.4),dpi=150)
    fig.subplots_adjust(left=.115,right=.975,bottom=.10,top=.895,wspace=.52,hspace=.65)
    plotted=[]
    for ax,label,title in zip(axes.flat,'abcd',
            ['Opening agreement, later divergence','Agreement set is not closed',
             'Numerical gate acceptance','Observed replay wall time']):
        ax.set_title(title,loc='left',pad=26)
        ax.text(-.19,1.22,label,transform=ax.transAxes,fontsize=12,fontweight='bold',
                va='bottom',ha='left',clip_on=False,gid='panel_label')
        for side in ['top','right']:ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',pad=3)

    a=axes[0,0]
    # Exact primary quantities are shown; the executed default controller matches
    # within its declared tolerance and is retained in the source receipt.
    plotted+=a.plot([1,2],[1.,1.],'-o',color=blue,lw=1.4,ms=5.5,label='Reference',zorder=3)
    plotted+=a.plot([1,2],[1.,ideal['second_compressed_discharge_mw']],'-s',color=orange,
                   lw=1.4,ms=5.5,mfc='white',mew=1.2,label='Compressed',zorder=4)
    a.set(xlim=(.83,2.17),ylim=(0,1.12),xticks=[1,2],yticks=[0,.5,1.],
          xlabel='Rolling origin',ylabel='First discharge (MW)')
    a.legend(loc='lower left',bbox_to_anchor=(-.01,1.015),ncol=2,frameon=False,
             handlelength=1.4,columnspacing=1.,handletextpad=.45,borderaxespad=0.)

    b=axes[0,1]
    b.add_patch(Rectangle((cover['lower_soc_mwh'],.80),cover['upper_soc_mwh']-cover['lower_soc_mwh'],.4,
                         facecolor=blue,edgecolor='none',alpha=.28))
    b.add_patch(Rectangle((cover['image_lower_soc_mwh'],-.20),cover['image_upper_soc_mwh']-cover['image_lower_soc_mwh'],.4,
                         facecolor=orange,edgecolor='none',alpha=.28))
    plotted+=b.plot([cover['lower_soc_mwh'],cover['upper_soc_mwh']],[1,1],color=blue,lw=2.2)
    plotted+=b.plot([cover['image_lower_soc_mwh'],cover['image_upper_soc_mwh']],[0,0],color=orange,lw=2.2)
    plotted+=b.plot([1.4],[1],'o',color=neutral,ms=4.4)
    plotted+=b.plot([ideal['common_next_soc_mwh']],[0],'o',color=neutral,ms=4.4)
    b.set(xlim=(.2,1.83),ylim=(-.48,1.48),xticks=[.2,.6,1.,1.4,1.8],yticks=[0,1],
          yticklabels=['After execution','Action agreement'],xlabel='Inventory (MWh)')
    b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0,pad=7)

    c=axes[1,0]
    state=sorted([r for r in cases if r['state_dependent']],key=lambda r:r['directions'])
    glob=sorted([r for r in cases if not r['state_dependent']],key=lambda r:r['directions'])
    plotted+=c.plot([r['directions'] for r in state],[r['acceptance_percent'] for r in state],
                   '-o',color=blue,lw=1.4,ms=5.3,label='State-dependent')
    plotted+=c.plot([r['directions'] for r in glob],[r['acceptance_percent'] for r in glob],
                   '--s',color=orange,lw=1.4,ms=5.3,mfc='white',mew=1.2,label='Global')
    c.set(xlim=(-.4,9.4),ylim=(0,105),xticks=[0,3,6,9],yticks=[0,25,50,75,100],
          xlabel='Retained objective directions',ylabel='Accepted at A$0.01 (%)')
    c.legend(loc='lower left',bbox_to_anchor=(-.01,1.015),ncol=2,frameon=False,
             handlelength=1.4,columnspacing=.8,handletextpad=.45,borderaxespad=0.)

    d=axes[1,1]
    ticks=np.arange(4)[::-1];labels=['IL replay','State + guard','Global + guard','State, no guard']
    for j,p in enumerate(policies):
        rr=[r for r in costs if r['policy']==p];ys=ticks[j]
        vals=[r['wall_ms_per_origin'] for r in rr]
        plotted+=d.plot([min(vals),max(vals)],[ys,ys],color=neutral,lw=.8,zorder=1)
        low=next(r for r in rr if r['initial_soc_mwh']==.2)
        high=next(r for r in rr if r['initial_soc_mwh']==1.8)
        plotted+=d.plot([low['wall_ms_per_origin']],[ys+.075],'o',color=blue,ms=5,zorder=3)
        plotted+=d.plot([high['wall_ms_per_origin']],[ys-.075],'s',color=orange,ms=5,mfc='white',mew=1.1,zorder=3)
    d.set(xlim=(0,5.8),ylim=(-.55,3.55),yticks=ticks,yticklabels=labels,
          xticks=[0,2,4],xlabel='Wall time (ms/origin)')
    d.spines['left'].set_visible(False);d.tick_params(axis='y',length=0,pad=7)
    d.legend(handles=[Line2D([],[],marker='o',color=blue,lw=0,ms=5,label='0.2 MWh'),
                      Line2D([],[],marker='s',color=orange,lw=0,ms=5,mfc='white',label='1.8 MWh')],
             loc='lower left',bbox_to_anchor=(-.01,1.015),ncol=2,frameon=False,
             handlelength=.8,columnspacing=1.,handletextpad=.45,borderaxespad=0.)

    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    texts=[]
    for txt in fig.findobj(match=Text):
        if txt.get_visible() and txt.get_text().strip():
            bb=txt.get_window_extent(renderer)
            if bb.width>0 and bb.height>0:
                texts.append((txt,bb))
    outside=[]
    for txt,bb in texts:
        if bb.x0<canvas.x0-.5 or bb.y0<canvas.y0-.5 or bb.x1>canvas.x1+.5 or bb.y1>canvas.y1+.5:
            outside.append({'text':txt.get_text(),'bbox':bbox_list(bb)})
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        for b,bb in texts[i+1:]:
            area=overlap(ab,bb)
            # Identical duplicated axis labels in the artist tree are the same
            # object, not separate text. Ignore only exact shared object IDs.
            if a is not b and area>2.:
                collisions.append({'text_1':a.get_text(),'text_2':b.get_text(),'overlap_px2':area})
    crosses=[]
    for line in plotted:
        path=line.get_path().transformed(line.get_transform())
        for txt,bb in texts:
            # Axis/title/legend text lies outside data paths. The path test also
            # checks one-point markers using a point-in-expanded-box test.
            if path.intersects_bbox(bb,filled=False):
                crosses.append({'text':txt.get_text(),'line_label':line.get_label()})
    layout={'text_count':len(texts),'text_outside_canvas':outside,
            'text_bbox_intersections':collisions,'text_data_path_intersections':crosses,
            'checked_titles':[{loc:ax.get_title(loc=loc)for loc in ['left','center','right']}for ax in axes.flat],
            'minimum_tick_font_pt_at_156mm':9.2*156/178,
            'status':'PASS' if not(outside or collisions or crosses)else'REPAIR_REQUIRED'}
    (OUT/'Figure7_layout_check.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
    assert layout['status']=='PASS',json.dumps(layout,indent=2)
    stem=OUT/'Figure7_rolling_sufficiency'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'),dpi=1200)
    fig.savefig(stem.with_suffix('.tiff'),dpi=1200,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUT/'Figure7_preview_300dpi.png',dpi=300)
    plt.close(fig)
    outputs=[]
    for p in [stem.with_suffix(ext)for ext in ['.pdf','.svg','.png','.tiff']]+[OUT/'Figure7_preview_300dpi.png']:
        meta={'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
        if p.suffix in ['.png','.tiff']:
            with Image.open(p)as im:meta.update({'pixels':list(im.size),
                'reported_dpi':None if im.info.get('dpi')is None else [float(x)for x in im.info['dpi']]})
        outputs.append(meta)
    with fitz.open(stem.with_suffix('.pdf'))as doc:
        vector={'pages':len(doc),'embedded_raster_images':sum(len(pg.get_images())for pg in doc),
                'page_mm':[x/72*25.4 for x in [doc[0].rect.width,doc[0].rect.height]],
                'minimum_actual_pdf_text_font_pt':min(s['size'] for b in doc[0].get_text('dict')['blocks']if 'lines'in b for l in b['lines']for s in l['spans'])}
    assert vector['embedded_raster_images']==0
    caption=("Figure 7. Rolling action sufficiency and its resource boundary. "
        "a, In a synthetic control using the baseline physical model, both exact primary policies initially discharge 1 MW; "
        "after the common inventory transition, the same two forecast templates give 1 versus 0.184 MW. "
        "Both realized prices are A$50/MWh and zero terminal marking gives a two-origin value loss of A$18.36. "
        "The default controller reproduces the analytical quantities within its monetary allowance. "
        "b, The initial action-agreement interval [1.2989011,1.8] MWh maps to [0.7494505,1.2505495] MWh and is not forward closed; "
        "black dots identify the initial 1.4 MWh stock and its common successor. "
        "c, Numerical first-action certificate acceptance at A$0.01 on 384 common-state cases per evaluated mask group. "
        "Only evaluated global masks, with three and six directions, are shown. The nine-direction state mask recovers the full IL objective. "
        "d, Recorded origin-loop wall time per origin using precomputed input curves for NSW January 2024, 1,488 origins, at two initial stocks. "
        "Markers are individual executions, not confidence intervals or CPU/production cost estimates. "
        "Timing excludes input loading, initialization and raw-version ingestion or queues. "
        "The analytical moment count is 192 numeric bytes for a twelve-target accounting unit, with the same IL moments retained by guarded variants; "
        "this is not measured peak RAM or a complete-stream upper bound, and a full stream may have more active targets. "
        "Masks and solves are additional, while common keys and allocation overhead are excluded symmetrically. These exploratory outputs support a resource failure boundary, "
        "rather than a cumulative cash non-inferiority or superior-compression claim.")
    (OUT/'Figure7_caption.txt').write_text(caption+'\n',encoding='utf-8')
    receipt={'status':'PASS_PROGRAMMATIC_VECTOR_DPI_AND_LAYOUT_CHECKS','elapsed_seconds':time.perf_counter()-began,
        'source_data_sha256':sha(OUT/'Figure7_source_data.json'),'script_sha256':sha(__file__),
        'native_vector':vector,'outputs':outputs,'layout':layout,
        'visual_review':'PENDING_ACTUAL_IMAGE_VIEW','scope':data['scope']}
    (OUT/'Figure7_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'vector':vector,'outputs':outputs,
                      'elapsed_seconds':receipt['elapsed_seconds']}))


if __name__=='__main__':main()
