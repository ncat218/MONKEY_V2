"""Paired descriptive contrasts of frozen quality scoring, never method selection."""
import numpy as np

from lowres.common import identity,read_json,sha256
from rev4 import require
from rev4.durable_io import save_json,write_text
from rev4.quality_combo_plan import ARMS,BRANCHES,CELLS,CONTRASTS,DETECTORS,RECIPE
from rev4.sr_recheck import metric_vector,METRICS


def effects(matrices,repetitions=2000,seed=20261107):
    arrays={key:np.asarray(value,dtype=np.int64) for key,value in matrices.items()}
    require(set(arrays)==set(CELLS),'Incomplete scorer factorial')
    n=len(next(iter(arrays.values())))
    require(n>0 and all(v.shape==(n,11) and (v>=0).all() for v in arrays.values()),'Unpaired scorer matrices')
    points={key:metric_vector(v.sum(0)) for key,v in arrays.items()}
    samples={key:[] for key in arrays}
    rng=np.random.default_rng(seed)
    for start in range(0,repetitions,50):
        take=rng.integers(0,n,size=(min(50,repetitions-start),n))
        for key,v in arrays.items():samples[key].append(metric_vector(v[take].sum(1)))
    samples={key:np.concatenate(v) for key,v in samples.items()}
    def pack(point,draw):
        ci=np.quantile(draw,[.025,.975],axis=0)
        return {k:dict(delta=float(point[j]),ci95=[float(ci[0,j]),float(ci[1,j])]) for j,k in enumerate(METRICS)}
    contrasts={new+' minus '+old:pack(points[new]-points[old],samples[new]-samples[old]) for new,old in CONTRASTS}
    # Does adding the scorer interact with ASF or restoration? Do not add separate gains.
    for label,left,right in (
        ('quality_x_asf_native',('ASF/native/Q','ASF/native/F'),('B1/native/Q','B1/native/F')),
        ('quality_x_edsr_asf',('ASF/edsr_x2/Q','ASF/edsr_x2/F'),('ASF/native/Q','ASF/native/F'))):
        point=(points[left[0]]-points[left[1]])-(points[right[0]]-points[right[1]])
        draw=(samples[left[0]]-samples[left[1]])-(samples[right[0]]-samples[right[1]])
        contrasts[label]=pack(point,draw)
    return contrasts


def mean_h_effects(clean,lr2,repetitions=2000,seed=20261107):
    """One source draw shared by Clean/LR2, then mean the two micro-Hmeans."""
    n=len(clean[CELLS[0]])
    require(n==len(lr2[CELLS[0]]) and n>0,'Clean/LR2 pairing differs')
    arrays={key:np.stack([clean[key],lr2[key]],axis=1).astype(np.int64) for key in CELLS}
    rng=np.random.default_rng(seed);draws={key:[] for key in CELLS}
    points={key:float(metric_vector(v.sum(0))[:,0].mean()) for key,v in arrays.items()}
    for start in range(0,repetitions,50):
        indices=rng.integers(0,n,size=(min(50,repetitions-start),n))
        for key,v in arrays.items():draws[key].append(metric_vector(v[indices].sum(1))[...,0].mean(1))
    draws={key:np.concatenate(v) for key,v in draws.items()}
    return {new+' minus '+old:dict(delta=points[new]-points[old],
        ci95=np.quantile(draws[new]-draws[old],[.025,.975]).tolist()) for new,old in CONTRASTS}


def report(a,tasks):
    from rev4.quality_combo import fold,parent_record,validate
    from rev4.fusion_evaluation import metrics
    digest=identity(a.trial_identity);records={name:{} for name in DETECTORS};record_hashes={}
    for name in DETECTORS:
        for task in tasks:
            path=a.output/'records'/name/task['condition']/(task['source_image_id']+'.json')
            r=read_json(path);parent,parent_sha=parent_record(a,name,task)
            validate(r,parent,parent_sha,digest);records[name][task['sample_id']]=r
            record_hashes[str(path.relative_to(a.output))]=sha256(path)
    matrices={};scores={};contrasts={};transitions={};subgroups={}
    for condition in RECIPE['conditions']:
        group=sorted([t for t in tasks if t['condition']==condition],key=lambda t:t['source_image_id'])
        require(len(group)==(2262 if condition in ('clean','lr2') else 500),'Reporting source count changed')
        matrices[condition]={};scores[condition]={};transitions[condition]={}
        for name in DETECTORS:
            for branch in BRANCHES:
                for arm in ARMS:
                    cell=f'{name}/{branch}/{arm}'
                    values=[records[name][t['sample_id']]['branches'][branch]['arms'][arm] for t in group]
                    matrix=np.asarray([r['counts']+sum(r['bins'],[]) for r in values],dtype=np.int64)
                    matrices[condition][cell]=matrix
                    total=matrix.sum(0);summary=metrics(total[:3]);summary['bins_lr2_reference']=total[3:].reshape(4,2).tolist()
                    scores[condition][cell]=summary
                    gained=lost=0
                    for t,r in zip(group,values):
                        before=set(records['B1'][t['sample_id']]['branches']['native']['arms']['F']['matched_ann_ids'])
                        after=set(r['matched_ann_ids']);gained+=len(after-before);lost+=len(before-after)
                    transitions[condition][cell]=dict(rescued_vs_B1_F=gained,lost_vs_B1_F=lost)
        contrasts[condition]=effects(matrices[condition])
        if condition in ('clean','lr2'):
            subgroups[condition]={}
            for label,expected in (('A',1130),('B',1132)):
                take=[i for i,t in enumerate(group) if fold(t['source_image_id'])==label]
                require(len(take)==expected,'Tri A/B descriptive subgroup changed')
                subset={key:v[take] for key,v in matrices[condition].items()}
                subgroups[condition][label]=dict(sources=expected,contrasts=effects(subset),
                    scores={key:metrics(v.sum(0)[:3]) for key,v in subset.items()})
    mean=mean_h_effects(matrices['clean'],matrices['lr2'])
    latency={name:read_json(a.output/'latency'/f'{name}.json') for name in DETECTORS}
    require(all(v['all_outputs_equal'] and v['identity_sha256']==digest for v in latency.values()),'Latency checks missing')
    output=a.output/'results'
    save_json(output/'records_sha256.json',record_hashes)
    save_json(output/'quality_combo_summary.json',dict(identity=a.trial_identity,recipe=RECIPE,scores=scores,
        contrasts=contrasts,mean_clean_lr2_hmean_pp=mean,transitions=transitions,A_B_descriptive=subgroups,
        latency={name:v['summary'] for name,v in latency.items()},no_training=True,no_test=True,
        caveats=['repeatedly viewed development; A and B are not fresh holdouts',
                 'one detector seed, source bootstrap excludes training-seed and method-selection uncertainty',
                 'frozen B1 scorer transferred to ASF/EDSR without adaptation; distribution shift may hurt',
                 'no primary GT edits; annotation incompleteness and score selection remain limitations',
                 'multiple unadjusted endpoints; no automatic winner or TEST evaluation']))
    lines=['# Frozen Tri box-quality x ASF x EDSR','',
        'Inference only. Original scorer, normalizer, candidate floor and cutoffs are frozen; no fitting or selection.',
        'F=.40/.60 raw baseline; RAW=.5976406401908236; Q=frozen 13-feature score>=.3315908613227198.',
        '2262 previously used development sources; fixed500 stress sources. A/B only descriptive. TEST not read.',
        'F controls reproduce parent B1/ASF per-source counts, matching IDs and restored bytes.','',
        '| Condition | Pipeline | P | R | Hmean | TP | FP | FN | Recall <4px |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in RECIPE['conditions']:
        for cell in CELLS:
            s=scores[condition][cell];gt,tp=s['bins_lr2_reference'][0]
            lines.append(f'| {condition} | {cell} | {s["precision"]:.2%} | {s["recall"]:.2%} | {s["hmean"]:.2%} | '
                f'{s["tp"]} | {s["fp"]} | {s["fn"]} | {tp/max(gt,1):.2%} |')
    lines+=['','| Condition | Contrast | Delta H pp (95% CI) | Delta tiny recall pp (95% CI) | Delta FP (95% CI) |',
             '|---|---|---:|---:|---:|']
    for condition in ('clean','lr2'):
        for label,result in contrasts[condition].items():
            h,t,fp=(result[k] for k in ('hmean_pp','recall_lt4_pp','fp_count'))
            lines.append(f'| {condition} | {label} | {h["delta"]:+.3f} {h["ci95"]} | '
                         f'{t["delta"]:+.3f} {t["ci95"]} | {fp["delta"]:+.0f} {fp["ci95"]} |')
    lines+=['','## Mean Clean/LR2 Hmean','', '| Contrast | Delta pp (paired 95% CI) |','|---|---:|']
    for label,v in mean.items():lines.append(f'| {label} | {v["delta"]:+.3f} {v["ci95"]} |')
    lines+=['','## Dedicated latency audit','',
        '| Detector | Condition | Input | F median ms | Q median ms | Q/F |','|---|---|---|---:|---:|---:|']
    for name,v in latency.items():
        for condition,branches in v['summary'].items():
            for branch,s in branches.items():
                lines.append(f'| {name} | {condition} | {branch} | {s["F"]["median_ms"]:.1f} | '
                    f'{s["Q"]["median_ms"]:.1f} | {s["Q_over_F_median"]:.3f} |')
    lines+=['','Latency:20 fixed sources, Clean/LR2, alternating two passes with warmup. '
        'Restoration measured once per input and added equally; PNG encoding, hashes and metrics excluded. Shared GPU.',
        'JSON includes P/R, all four size bins, matched-ID transitions, source-paired CIs, interactions and A/B breakdown.',
        'Descriptive development evidence only; source CIs exclude training-seed variation and selection bias.','']
    write_text(output/'quality_combo_report.md','\n'.join(lines))
    save_json(output/'complete.json',dict(identity=a.trial_identity,no_training=True,no_test=True,
        report_sha256=sha256(output/'quality_combo_report.md'),summary_sha256=sha256(output/'quality_combo_summary.json'),
        records_sha256=sha256(output/'records_sha256.json')))
