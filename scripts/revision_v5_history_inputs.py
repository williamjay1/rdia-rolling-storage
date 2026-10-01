"""Reconstruct complete target-vintage histories without using future records.

Existing raw ZIPs remain read-only. All products live in revision_v5/history.
The original coherent origin chronology and realized outcomes are preserved.
"""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_n]='2'
import argparse, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(2);pa.set_io_thread_count(2)
import build_vintage_training_panel_crossregime as raw_builder
from revision_v4_forecast_models import FC,Y,bound_paths

ROOT=Path(r'D:\MLWork\AOOR_ContextualStorage_20260929')
OUT=ROOT/'results/revision_v5/history'
V4=ROOT/'results/revision_v4/forecast'
REGIONS=['NSW1','QLD1','SA1','TAS1','VIC1']
P=[f'p_{j+1:02d}' for j in range(12)]
KEY=['REGIONID','target','PREDISPATCHSEQNO','RUNNO']

def json_write(path,obj): path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def load_region(region):
    f=pd.read_parquet(V4/f'{region}_enriched_episodes.parquet').sort_values('target').reset_index(drop=True)
    f['target']=pd.to_datetime(f.target).astype('datetime64[ns]');f['curve_asof']=pd.to_datetime(f.curve_asof).astype('datetime64[ns]')
    return f
def phase_frame(q):
    q=q.loc[((q.target.dt.year==2023)&(q.target+pd.Timedelta(minutes=330)<pd.Timestamp('2024-01-01')-pd.Timedelta(minutes=60)))|((q.target>=pd.Timestamp('2024-01-01'))&(q.target<pd.Timestamp('2026-09-01')))].copy()
    q['phase']=np.where(q.target.dt.year==2023,'validation','evaluation')
    return q.reset_index(drop=True)
def export(frame,x,region,policy,cutoff,query=None):
    x=bound_paths(np.asarray(x,dtype=np.float32),frame)
    out=frame[['target','phase',*Y]].copy()
    out.insert(0,'region',region);out['cutoff_minutes']=cutoff;out['policy']=policy
    out['query_time']=frame.curve_asof.to_numpy() if query is None else np.asarray(query)
    out[P]=x
    return out
def initial():
    OUT.mkdir(parents=True,exist_ok=True)
    design={'status':'recorded before new outputs','source':'v4 corrected coherent episode chronology; all comparisons exploratory',
      'aggregation':'all target-specific archived versions available by common cutoff; nominal inverse-lead weights=1/(target-nominal_run_start), not inverse age since decision; archive-asof denominator separately retained',
      'cutoffs_minutes_before_interval_end':[60,30], 'execution_start':'target-30min',
      'current_shrink':'(1-a)*current curve + a*current curve grand mean, a[0,.25,.5]',
      'current_smooth':'(1-a)*current curve + a*three-point mean; first/last use two available neighbors; a[0,.25,.5]',
      'zero_parameter':'raw; not independently duplicated',
      'selection':'control agent selects per-region a only by 2023 strictly purged model-condition operational net value; fixed before test',
      'incremental':'base fixed Ridge1000 fit2015-2020; g fixed Ridge1000 fit2021-2022 actual minus frozen base prediction; alpha[0,.25,.5,1] solely2023 operational validation',
      'vintage_source_limitation':'existing lead-filtered D panels are not used as full histories; all positive-lead intervention0 F archives reconstructed',
      'storage_free_bytes':{'D':shutil.disk_usage('D:/').free,'E':shutil.disk_usage('E:/').free,'F':shutil.disk_usage('F:/').free}}
    json_write(OUT/'design.json',design)
    audit=[]
    for region in REGIONS:
        f=phase_frame(load_region(region));raw=f[FC].to_numpy(np.float32)
        versions=[raw]+[raw-f[[f'{pre}_{j:02d}' for j in range(12)]].to_numpy(np.float32) for pre in ('rev_30m','rev_60m','rev_120m','rev_net')]
        sparse=bound_paths(np.nanmean(np.stack(versions,axis=1),axis=1),f)
        v4=pd.read_parquet(V4/f'{region}_mean_paths.parquet')
        v4=v4.loc[v4.model.eq('history_average')].sort_values('timestamp')
        check=sparse[f.phase.eq('evaluation')]
        old=v4[[f'price_j{j+1}' for j in range(12)]].to_numpy(np.float32)
        if not np.array_equal(check,old): raise AssertionError((region,'sparse v4 export mismatch',float(np.max(abs(check-old)))))
        pieces=[export(f,raw,region,'raw',60),export(f,sparse,region,'sparse_equal',60)]
        mean=raw.mean(axis=1,keepdims=True)
        sm=raw.copy();sm[:,1:-1]=(raw[:,:-2]+raw[:,1:-1]+raw[:,2:])/3
        sm[:,0]=(raw[:,0]+raw[:,1])/2;sm[:,-1]=(raw[:,-2]+raw[:,-1])/2
        for a,tag in ((.25,'a025'),(.5,'a050')):
            pieces.extend([export(f,(1-a)*raw+a*mean,region,'current_shrink_'+tag,60),export(f,(1-a)*raw+a*sm,region,'current_smooth_'+tag,60)])
        path=OUT/f'{region}_baseline_inputs.parquet';pd.concat(pieces,ignore_index=True).to_parquet(path,index=False,compression='zstd')
        row={'region':region,'path':str(path),'validation':int(f.phase.eq('validation').sum()),'evaluation':int(f.phase.eq('evaluation').sum()),'policies':6,'sparse_v4_exact':True,'sha256':digest(path)}
        audit.append(row);print(json.dumps(row),flush=True)
    json_write(OUT/'baseline_ready.json',{'status':'complete','schema':P+Y,'regions':audit})

def read_raw(year,month):
    path=raw_builder.month_archive('PREDISPATCHPRICE',year,month)
    if path is None: raise FileNotFoundError((year,month,'raw archive'))
    cols=['PREDISPATCHSEQNO','RUNNO','REGIONID','INTERVENTION','RRP','LASTCHANGED','DATETIME']
    r=raw_builder.read_aemo(path,'PREDISPATCH','REGION_PRICES',cols)
    r=r.loc[r.REGIONID.isin(REGIONS)&r.INTERVENTION.eq('0')].copy()
    r['target']=pd.to_datetime(r.DATETIME,format='%Y/%m/%d %H:%M:%S',errors='coerce')
    r['asof']=pd.to_datetime(r.LASTCHANGED,format='%Y/%m/%d %H:%M:%S',errors='coerce')
    r['target']=r.target.astype('datetime64[ns]');r['asof']=r['asof'].astype('datetime64[ns]')
    r['forecast']=pd.to_numeric(r.RRP,errors='coerce');r['runno_num']=pd.to_numeric(r.RUNNO,errors='coerce').fillna(-1)
    r=r.dropna(subset=['target','asof','forecast'])
    seq=r.PREDISPATCHSEQNO.astype(str)
    run_date=pd.to_datetime(seq.str[:8],format='%Y%m%d',errors='coerce')
    run_period=pd.to_numeric(seq.str[8:],errors='coerce')
    if run_date.isna().any() or not run_period.between(1,48).all():raise AssertionError('invalid YYYYMMDDPP nominal sequence code')
    r['nominal_issue']=run_date+pd.to_timedelta(240+(run_period-1)*30,unit='m')
    r['nominal_issue']=r.nominal_issue.astype('datetime64[ns]')
    r=r.loc[r.target.gt(r['asof']),KEY+['asof','forecast','runno_num','nominal_issue']].copy()
    delay=(r['asof']-r.nominal_issue).dt.total_seconds()/60
    info={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'positive_lead_rows':len(r),
      'nominal_issue_rule':'SEQ date+04:00+(PP-1)*30min; inferred from official PP01 ending04:30, half-hour run operation and 04:30 file generated04:02 example; not actual participant receipt',
      'asof_minus_nominal_minutes':{'min':float(delay.min()),'p01':float(delay.quantile(.01)),'median':float(delay.median()),'p99':float(delay.quantile(.99)),'max':float(delay.max())},
      'nominal_positive_target_lead':bool(r.target.gt(r.nominal_issue).all()),'pp_first_rows':int(run_period.eq(1).sum()),'pp_last_rows':int(run_period.eq(48).sum())}
    if not info['nominal_positive_target_lead']:raise AssertionError(info)
    return r,info

def full_history(month_start='2021-01-01', aligned=False):
    """Partition by target month; carry preceding-run-month forecasts forward."""
    OUT.mkdir(parents=True,exist_ok=True)
    monthly=OUT/'monthly';monthly.mkdir(exist_ok=True)
    origins=pd.concat([load_region(r).loc[lambda q:q.target.ge('2021-01-01'),['target','REGIONID','PREDISPATCHSEQNO','RUNNO','curve_asof']] for r in REGIONS],ignore_index=True)
    origins=origins.sort_values(['REGIONID','target']).reset_index(drop=True);origins['origin_id']=np.arange(len(origins))
    n=len(origins); targets=origins.target.to_numpy('datetime64[ns]')
    audits=[];carry=None
    sources=[]
    if month_start!='2021-01-01':
        previous_ready=json.loads((OUT/'full_history_raw_ready.json').read_text())
        audits=previous_ready['months'];sources=previous_ready['sources']
    for start in pd.date_range(month_start,'2026-09-01',freq='MS'):
        tic=time.time();end=start+pd.offsets.MonthBegin(1);dest=monthly/f'paths_{start:%Y%m}.parquet'
        # The final origin has an observed terminal horizon at Sep1 00:00.
        # Reconstruct that target only from available August archive records;
        # no September-origin decisions or post-cutoff records are introduced.
        current_archive=None if start==pd.Timestamp('2026-09-01') else raw_builder.month_archive('PREDISPATCHPRICE',start.year,start.month)
        if current_archive is None:
            # October2022 is absent from the original supervised archive
            # chronology. It cannot be manufactured as a source month.
            needed=sum(int(((targets+np.timedelta64(30*j,'m')>=np.datetime64(start))&(targets+np.timedelta64(30*j,'m')<np.datetime64(end))).sum()) for j in range(12))
            if not needed:
                audits.append({'month':str(start.date()),'status':'no raw archive and no common coherent queries; explicitly omitted','query_rows':0})
                carry=None;continue
            # The September30 18:30 origin has exactly five terminal labels
            # at October1 00:00, already present in September's archive tail.
            # Preserve this observed boundary rather than inventing October.
        # Resume safely: cached monthly derivatives have their own provenance.
        if dest.exists() and (monthly/f'audit_{start:%Y%m}.json').exists():
            a=json.loads((monthly/f'audit_{start:%Y%m}.json').read_text());audits.append(a)
            path=raw_builder.month_archive('PREDISPATCHPRICE',start.year,start.month)
            sources.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'cached_derived_month':True} if path is not None else {'path':None,'cached_boundary_month':str(start.date()),'source':'previous archive carry'})
            carry=None
            continue
        if carry is None:
            previous=start-pd.offsets.MonthBegin(1)
            previous_archive=raw_builder.month_archive('PREDISPATCHPRICE',previous.year,previous.month)
            if previous_archive is not None:
                prev,prev_info=read_raw(previous.year,previous.month)
                carry=prev.loc[prev.target.ge(start)].copy();sources.append(prev_info);del prev
            else:
                carry=pd.DataFrame(columns=KEY+['asof','forecast','runno_num','nominal_issue'])
        if current_archive is not None:
            raw,info=read_raw(start.year,start.month);sources.append(info)
        else:
            raw=carry.iloc[:0].copy();info={'path':None,'missing_current_archive':str(start.date()),'boundary_source':'preceding archive carry only; no October-origin decisions'}
            if not len(carry):raise RuntimeError((str(start),'missing archive and no preceding tail'))
        pred=pd.concat([carry,raw],ignore_index=True) if len(carry) else raw.copy()
        carry=raw.loc[raw.target.ge(end)].copy();del raw
        pred=pred.loc[pred.target.ge(start)&pred.target.lt(end)].copy()
        pred=pred.sort_values(['PREDISPATCHSEQNO','RUNNO','target','REGIONID','asof','runno_num']).drop_duplicates(KEY+['asof'],keep='last')
        pred=pred.sort_values(['REGIONID','target','asof','PREDISPATCHSEQNO','runno_num']).drop_duplicates(['REGIONID','target','asof'],keep='last').reset_index(drop=True)
        # Same target histories are ordered by actual record timestamp. Cumsums
        # permit exact all-vintage averages in O(V+Q), not O(V*Q).
        pred['lead_min']=(pred.target-pred['asof']).dt.total_seconds()/60
        g=pred.groupby(['REGIONID','target'],sort=False,observed=True)
        pred['count']=g.cumcount()+1
        pred['sum']=g.forecast.cumsum();pred['p2']=pred.forecast**2;pred['sum2']=g.p2.cumsum()
        pred['archive_w']=1/pred.lead_min;pred['archive_wp']=pred.archive_w*pred.forecast
        pred['archive_wsum']=g.archive_w.cumsum();pred['archive_wpsum']=g.archive_wp.cumsum()
        pred['inverse_archive_lead']=pred.archive_wpsum/pred.archive_wsum
        pred['nominal_lead_min']=(pred.target-pred.nominal_issue).dt.total_seconds()/60
        pred['w']=1/pred.nominal_lead_min;pred['wp']=pred.w*pred.forecast
        pred['wsum']=g.w.cumsum();pred['wpsum']=g.wp.cumsum()
        pred['full_equal']=pred['sum']/pred['count'];pred['inverse_lead']=pred.wpsum/pred.wsum
        pred['sd']=np.sqrt(np.maximum(pred.sum2/pred['count']-pred.full_equal**2,0))
        pred['first_asof']=g['asof'].transform('first')
        # Sparse matching follows old prefilter <=645min so the v4 rule is
        # reproduced rather than silently enlarged with early raw vintages.
        sparse_pred=pred.loc[pred.lead_min.between(20,645)].copy()
        sg=sparse_pred.groupby(['REGIONID','target'],sort=False,observed=True)
        sparse_pred['sparse_equal']=pd.concat([sparse_pred.forecast]+[sg.forecast.shift(j) for j in (1,2,4,6)],axis=1).mean(axis=1)
        for lag in (1,2,4,6):
            sparse_pred[f'lag{lag}_asof']=sg['asof'].shift(lag)
            sparse_pred[f'lag{lag}_p']=sg.forecast.shift(lag)
        lookup_sparse=sparse_pred.set_index(KEY)
        queries=[]
        for j in range(12):
            t=targets+np.timedelta64(30*j,'m');idx=np.flatnonzero((t>=np.datetime64(start))&(t<np.datetime64(end)))
            if not len(idx):continue
            q=origins.iloc[idx].copy();q['origin_target']=q.target.to_numpy();q['target']=t[idx];q['step']=j
            queries.append(q)
        if not queries: continue
        q=pd.concat(queries,ignore_index=True)
        # Fresh coherent issue is selected per first target, then matched for
        # every horizon. Do not stitch future targets from different issues.
        first=q.loc[q.step.eq(0)].copy();first['cutoff']=first.origin_target-pd.Timedelta(minutes=30)
        fresh_first=pd.merge_asof(first.sort_values('cutoff'),pred.sort_values('asof')[['target','REGIONID','asof','PREDISPATCHSEQNO','RUNNO']],left_on='cutoff',right_on='asof',by=['target','REGIONID'],direction='backward',suffixes=('','_fresh'))
        # Current month only supplies first-step origins. Origin lookahead
        # extending into the next month is handled via selected issue metadata
        # saved for those origins rather than attempting a new first query.
        freshmeta=OUT/'fresh_origin_issue.parquet'
        freshpart=fresh_first[['origin_id','asof','PREDISPATCHSEQNO_fresh','RUNNO_fresh']].rename(columns={'asof':'fresh_asof','PREDISPATCHSEQNO_fresh':'fresh_seq','RUNNO_fresh':'fresh_run'})
        prior=pd.read_parquet(freshmeta) if freshmeta.exists() else freshpart.iloc[:0]
        freshall=pd.concat([prior,freshpart],ignore_index=True).drop_duplicates('origin_id',keep='last')
        freshall.to_parquet(freshmeta,index=False,compression='zstd')
        q=q.merge(freshall,on='origin_id',how='left',validate='many_to_one')
        pieces=[]
        for cutoff in (60,30):
            qq=q.copy();qq['cutoff']=qq.origin_target-pd.Timedelta(minutes=cutoff)
            if cutoff==30:
                qq['PREDISPATCHSEQNO']=qq.fresh_seq;qq['RUNNO']=qq.fresh_run
                qq['curve_asof']=qq.fresh_asof
            coherent=qq.merge(pred[KEY+['asof','forecast']],on=KEY,how='left',validate='many_to_one')
            coherent['coherent_valid']=coherent['asof'].eq(coherent.curve_asof)&coherent['asof'].le(coherent.cutoff)&coherent.forecast.notna()
            hist=pd.merge_asof(qq.sort_values('cutoff'),pred.sort_values('asof'),left_on='cutoff',right_on='asof',by=['target','REGIONID'],direction='backward',suffixes=('','_history'))
            # Real sparse constituent prices and ages, anchored to coherent
            # issue, are stored separately from all-available history.
            sparse=lookup_sparse.reindex(pd.MultiIndex.from_frame(qq[KEY]))
            z=qq[['origin_id','REGIONID','origin_target','target','step','curve_asof','cutoff']].copy().rename(columns={'origin_target':'origin'})
            z['cutoff_minutes']=cutoff
            z['raw']=coherent.forecast.to_numpy();z['coherent_valid']=coherent.coherent_valid.to_numpy()
            z['sparse_equal']=sparse.sparse_equal.to_numpy()
            for col in ('full_equal','inverse_lead','inverse_archive_lead','count','sd','lead_min','nominal_lead_min'):
                # merge_asof sorts rows; restore qq index via the origin/step
                # identifiers retained in its returned rows.
                ordered=hist.set_index(['origin_id','step']).reindex(pd.MultiIndex.from_frame(qq[['origin_id','step']]))
                z[col]=ordered[col].to_numpy()
            z['latest_asof']=ordered['asof'].to_numpy();z['oldest_asof']=ordered.first_asof.to_numpy()
            z['latest_nominal_issue']=ordered.nominal_issue.to_numpy()
            z['oldest_age_min']=(z.cutoff-z.oldest_asof).dt.total_seconds()/60
            z['latest_age_min']=(z.cutoff-z.latest_asof).dt.total_seconds()/60
            for lag in (1,2,4,6):
                z[f'lag{lag}_lead_min']=(z.target-pd.to_datetime(sparse[f'lag{lag}_asof'].to_numpy())).dt.total_seconds()/60
                z[f'lag{lag}_p']=sparse[f'lag{lag}_p'].to_numpy()
            pieces.append(z)
        z=pd.concat(pieces,ignore_index=True)
        z.to_parquet(dest,index=False,compression='zstd')
        a={'month':str(start.date()),'raw_target_rows':len(pred),'query_rows':len(z),'invalid_coherent_fields':int((~z.coherent_valid).sum()),'future_history_records':int((z.latest_asof>z.cutoff).sum()),'nominal_issue_after_cutoff_rows':int((z.latest_nominal_issue>z.cutoff).sum()),'seconds':time.time()-tic,'raw_source_audit':info}
        if a['future_history_records']:raise AssertionError(a)
        json_write(monthly/f'audit_{start:%Y%m}.json',a);audits.append(a);print(json.dumps(a),flush=True)
    json_write(OUT/('full_history_raw_aligned_ready.json' if aligned else 'full_history_raw_ready.json'),{'status':'complete','sources':sources,'months':audits,'origins':n,'first':'2021-01-01','last':'2026-08-31','source_semantics':'all intervention0 positive lead existing archived versions; all asof<=cutoff; LASTCHANGED proxy not measured receipt','terminal_target':'2026-09-01 00:00:00 from August archive only'})
    assemble(aligned=aligned)

def assemble(aligned=False):
    suffix='_aligned' if aligned else ''
    reports=[]
    for region in REGIONS:
        ep=load_region(region);allpieces=[];aggframes=[]
        # Keep memory bounded by a single region; do not materialize the
        # five-region, all-month long history simultaneously.
        reg=pd.concat([pd.read_parquet(p,filters=[('REGIONID','==',region)]) for p in sorted((OUT/'monthly').glob('paths_*.parquet'))],ignore_index=True)
        for cutoff in (60,30):
            rr=reg.loc[reg.cutoff_minutes.eq(cutoff)]
            valid=rr.groupby('origin',sort=False).agg(rows=('step','size'),valid=('coherent_valid','sum'))
            good=valid.index[(valid.rows==12)&(valid.valid==12)]
            rr=rr.loc[rr.origin.isin(good)].copy()
            meta=ep.loc[ep.target.isin(good)].copy().reset_index(drop=True)
            agg=meta[['target','REGIONID','curve_asof',*FC,*Y,'hour_sin','hour_cos','doy_sin','doy_cos','weekday','decision_lead_min']].copy()
            agg['cutoff_minutes']=cutoff
            for field in ('raw','sparse_equal','full_equal','inverse_lead','inverse_archive_lead','count','sd','lead_min','nominal_lead_min','oldest_age_min','latest_age_min','lag1_p','lag2_p','lag4_p','lag6_p','lag1_lead_min','lag2_lead_min','lag4_lead_min','lag6_lead_min'):
                wide=rr.pivot(index='origin',columns='step',values=field).reindex(meta.target)
                cols=[f'{field}_{j:02d}' for j in range(12)];agg[cols]=wide.to_numpy()
            qt=rr.loc[rr.step.eq(0)].set_index('origin').curve_asof.reindex(meta.target).to_numpy()
            agg['curve_asof']=qt;agg['decision_lead_min']=(agg.target-agg.curve_asof).dt.total_seconds()/60
            if cutoff==60:
                old=meta[FC].to_numpy(np.float32);new=agg[[f'raw_{j:02d}' for j in range(12)]].to_numpy(np.float32)
                if not np.array_equal(old,new):raise AssertionError((region,'raw60 not exactly reproduced',float(np.nanmax(abs(old-new)))))
                orig=[old]+[old-meta[[f'{pre}_{j:02d}' for j in range(12)]].to_numpy(np.float32) for pre in ('rev_30m','rev_60m','rev_120m','rev_net')]
                expected=bound_paths(np.nanmean(np.stack(orig,axis=1),axis=1),meta)
                actual=bound_paths(agg[[f'sparse_equal_{j:02d}' for j in range(12)]].to_numpy(np.float32),meta)
                # Raw prices are double precision. The original sparse rule
                # subtracts stored float32 differences: record its tiny
                # cancellation drift and export exact original rule at60.
                drift=float(np.max(abs(actual-expected)))
                agg[[f'sparse_equal_{j:02d}' for j in range(12)]]=expected
            else:drift=None
            aggframes.append(agg)
            f=phase_frame(agg);raw=f[[f'raw_{j:02d}' for j in range(12)]].to_numpy(np.float32)
            for field in ('raw','sparse_equal','full_equal','inverse_lead','inverse_archive_lead'):
                allpieces.append(export(f,f[[f'{field}_{j:02d}' for j in range(12)]].to_numpy(),region,field,cutoff,f.curve_asof))
            mean=raw.mean(axis=1,keepdims=True);sm=raw.copy();sm[:,1:-1]=(raw[:,:-2]+raw[:,1:-1]+raw[:,2:])/3
            sm[:,0]=(raw[:,0]+raw[:,1])/2;sm[:,-1]=(raw[:,-2]+raw[:,-1])/2
            for a,tag in ((.25,'a025'),(.5,'a050')):
                allpieces.extend([export(f,(1-a)*raw+a*mean,region,'current_shrink_'+tag,cutoff,f.curve_asof),export(f,(1-a)*raw+a*sm,region,'current_smooth_'+tag,cutoff,f.curve_asof)])
            reports.append({'region':region,'cutoff_minutes':cutoff,'origins_all_2021on':len(meta),'validation':int(f.phase.eq('validation').sum()),'evaluation':int(f.phase.eq('evaluation').sum()),'raw60_equal':cutoff==60,'sparse_direct_raw_cancellation_drift_max_aud_mwh':drift,'missing_common_origins':int(ep.target.ge('2021-01-01').sum()-len(meta))})
        aggpath=OUT/f'{region}_history_features{suffix}.parquet';pd.concat(aggframes,ignore_index=True).to_parquet(aggpath,index=False,compression='zstd')
        path=OUT/f'{region}_inputs{suffix}.parquet';pd.concat(allpieces,ignore_index=True).to_parquet(path,index=False,compression='zstd')
        print(json.dumps({'region':region,'assembled':str(path),'sha256':digest(path)}),flush=True)
    json_write(OUT/f'full_inputs{suffix}_ready.json',{'status':'complete','reports':reports,'schema':P+Y,'input_files':[str(OUT/f'{r}_inputs{suffix}.parquet') for r in REGIONS],'boundary_repair':'original Aug31 18:30 coherent origin terminal target Sep1 00:00 reconstructed from actual August archive; preceding incomplete derivatives preserved' if aligned else None})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',action='store_true');ap.add_argument('--full',action='store_true');ap.add_argument('--assemble',action='store_true');ap.add_argument('--boundary',action='store_true');ap.add_argument('--aligned',action='store_true');a=ap.parse_args()
    if a.initial:initial()
    if a.full:full_history(aligned=a.aligned)
    if a.boundary:full_history(month_start='2026-09-01',aligned=True)
    if a.assemble:assemble(aligned=a.aligned)
