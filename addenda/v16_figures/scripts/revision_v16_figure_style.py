"""Shared physical-size typography and vector export for the v16 figure revision."""
from pathlib import Path
import hashlib
import json
import os
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/revision_v16'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib_v16'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from nature_qa_v11.audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, ORANGE, TEAL = '#0072B2', '#D55E00', '#009E73'
INK, GRAY, LIGHT = '#171717', '#60666B', '#D6DADF'
RC = {
    'font.family':'sans-serif', 'font.sans-serif':['Arial'],
    'font.size':9, 'axes.labelsize':9, 'axes.titlesize':9,
    'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9,
    'axes.linewidth':.6, 'lines.linewidth':1.1, 'lines.markersize':3.5,
    'xtick.major.size':3, 'ytick.major.size':3,
    'xtick.major.width':.6, 'ytick.major.width':.6,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.grid':False, 'xtick.minor.visible':False, 'ytick.minor.visible':False,
    'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':INK,
    'xtick.color':INK, 'ytick.color':INK,
    'figure.facecolor':'white', 'axes.facecolor':'white',
    'savefig.facecolor':'white', 'savefig.transparent':False,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'text.usetex':False, 'mathtext.fontset':'dejavusans', 'savefig.bbox':None,
}
plt.rcParams.update(RC)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frame(height_mm, width_mm=174):
    return plt.figure(figsize=(width_mm/25.4, height_mm/25.4), dpi=300)

def text(fig,x,y,value,**kw):
    kw.setdefault('fontsize',9)
    kw.setdefault('va','center')
    return fig.text(x,y,value,**kw)

def heading(fig,x,y,letter,value):
    text(fig,x,y,letter,fontweight='bold',fontsize=10)
    text(fig,x+.032,y,value,fontweight='bold')

def axis(fig,bounds):
    ax=fig.add_axes(bounds)
    ax.tick_params(direction='out',pad=4)
    return ax

def final_save(fig,name,data,paths=(),groups=(),design=None,column_groups=()):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[]
    canvas=(fig.get_figwidth()*72,fig.get_figheight()*72)
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text().strip():
            b=artist.get_window_extent(renderer)
            box=[float(v*72/fig.dpi) for v in b.extents]
            texts.append({'text':artist.get_text(),'font_pt':float(artist.get_fontsize()),'bbox_pt':box})
            if box[0]<-.25 or box[1]<-.25 or box[2]>canvas[0]+.25 or box[3]>canvas[1]+.25:
                raise ValueError(f'Text outside native page: {name}: {artist.get_text()}: {box}')
    measured=[]
    for ax in fig.axes:
        x,y,w,h=ax.get_position().bounds
        measured.append([x*canvas[0],y*canvas[1],w*canvas[0],h*canvas[1]])
    selected=sorted({i for g in tuple(groups)+tuple(column_groups) for i in g}) if groups or column_groups else list(range(len(fig.axes)))
    if selected:
        require_matplotlib_panel_alignment(
            fig,axes=[fig.axes[i] for i in selected],panel_ids=[f'p{i}' for i in selected],
            row_groups=[[f'p{i}' for i in g] for g in groups] or None,
            column_groups=[[f'p{i}' for i in g] for g in column_groups] or None,
            json_out=OUT/f'{name}.alignment.json',tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,strict=True,require_panel_labels=False)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches=None)
    fig.savefig(OUT/f'{name}.svg',bbox_inches=None)
    fig.savefig(OUT/f'{name}.png',dpi=1200,bbox_inches=None)
    fig.savefig(OUT/f'{name}.tiff',dpi=1200,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches=None)
    inputs=[]
    for p in paths:
        p=Path(p)
        try: rel=p.relative_to(ROOT).as_posix()
        except ValueError: rel=p.name
        inputs.append({'path':rel,'sha256':sha(p)})
    record={'name':name,'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,
        'raster_dpi':1200,'vector_formats':['pdf','svg'],'inputs':inputs,'source_data':data,
        'axes_rectangles_pt':measured,'rendered_text_artists':texts,'design':design,
        'alignment_groups':[list(g) for g in groups],'column_groups':[list(g) for g in column_groups],
        'status':'EXPORTED_PENDING_FRESH_PDF_AND_VISUAL_AUDIT',
        'exports':[{'path':f'{name}.{s}','sha256':sha(OUT/f'{name}.{s}'),'bytes':(OUT/f'{name}.{s}').stat().st_size} for s in ('pdf','svg','png','tiff')]}
    (OUT/f'{name}_source.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    plt.close(fig)
    print(name,'exported',flush=True)

save_figure=final_save
