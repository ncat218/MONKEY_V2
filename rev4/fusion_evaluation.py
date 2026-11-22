"""Fixed-threshold transfer across corruptions; no target threshold selection or TEST."""
from contextlib import contextmanager
import copy
import hashlib
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from lowres.common import read_json,sha256
from rev4 import require,calibration
from rev4.durable_io import save_json,write_text
from rev4.fusion_plan import ARMS,PAIRS,CONDITIONS,PRIMARY,RECIPE
from rev4.local_scratch import local_directory


def stress_ids(rows):
    return sorted(sorted(rows,key=lambda sid:hashlib.sha256(f'42:fusion-stress:{sid}'.encode()).digest())[:RECIPE['stress_sources']])


def variant(image,condition):
    from PIL import Image,ImageFilter
    clean=image.convert('RGB')
    if condition in ('lr4','lr8'):
        scale=int(condition[2:]);return clean.resize(tuple(max(1,v//scale) for v in clean.size),Image.Resampling.BICUBIC),'PNG'
    if condition in ('blur1','blur2'):
        return clean.filter(ImageFilter.GaussianBlur(radius=1.88 if condition=='blur1' else 3.75)),'PNG'
    require(condition in ('jpeg30','jpeg10'),'Unknown stress variant')
    return clean,'JPEG'


@contextmanager
def materialize(a,task,row):
    from PIL import Image
    from rev4.stage_a import open_native_lr,make_task
    if task['condition'] in ('clean','lr2'):
        yield task;return
    source=a.clean_root/row['clean_relpath']
    require(sha256(source)==row['clean_sha256'],'Stress source SHA changed')
    with open_native_lr(source)[0] as image:
        require(list(image.size)==row['clean_size'] and int(image.getexif().get(274,1))==1,'Stress EXIF/size changed')
        image,format=variant(image,task['condition'])
        # Canonical metadata-free encoding; decoded pixels are audited against frozen scripts.
        image=Image.fromarray(np.asarray(image))
    try:
        with local_directory(prefix='monkey-fusion-stress-') as directory:
            path=directory/('variant.jpg' if format=='JPEG' else 'variant.png')
            options=dict(quality=int(task['condition'][4:])) if format=='JPEG' else {}
            image.save(path,format=format,**options)
            with Image.open(path) as check:check.load();require(check.size==image.size,'Stress write/decode failed')
            yield make_task(task['source_image_id'],task['condition'],path,image.size,sha256(path))
    finally:
        image.close()


def specification(a):
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    selected=stress_ids(rows)
    tasks=list(tasks)+[dict(sample_id=f'{c}::{s}',source_image_id=s,condition=c) for c in CONDITIONS[2:] for s in selected]
    return rows,tasks,selected


def corruption_probe(a):
    """Compare decoded outputs with the frozen scripts on two TRAIN sources."""
    import csv,subprocess,sys
    from lowres.common import jsonl
    from lowres.generate_lr_only import generate_variants
    from rev4.stage_a import open_native_lr
    from rev4.training_data import TRAIN_NAME
    rows=[]
    for row in jsonl(a.calibration_prepared/TRAIN_NAME):
        rows.append(row)
        if len(rows)==2:break
    with local_directory(prefix='monkey-fusion-corruption-probe-') as directory:
        directory=Path(directory);selection=directory/'sources.txt'
        write_text(selection,'\n'.join(v['clean_relpath'] for v in rows)+'\n')
        reference=directory/'reference'
        subprocess.run([sys.executable,str(a.blur_generator),'--input-dir',str(a.clean_root),'--source-list',str(selection),
                        '--output-dir',str(reference),'--blur-sigmas','1.88','3.75','--jpeg-qualities','30','10',
                        '--expect-count','2','--strict-exif'],check=True)
        lookup={v['source']:{} for v in csv.DictReader((reference/'manifest.csv').open())}
        for entry in csv.DictReader((reference/'manifest.csv').open()):
            condition=('blur1' if entry['parameter']=='1.88' else 'blur2') if entry['condition']=='blur' else 'jpeg'+entry['parameter']
            lookup[entry['source']][condition]=reference/entry['out_path']
        for row in rows:
            with open_native_lr(a.clean_root/row['clean_relpath'])[0] as image:
                for manifest,_ in generate_variants(image.convert('RGB'),Path(row['clean_relpath']),directory/'lr_reference'):
                    scale=int(manifest[1])
                    if scale in (4,8):lookup[row['clean_relpath']]['lr'+str(scale)]=directory/'lr_reference'/manifest[6]
        for row in rows:
            for condition in CONDITIONS[2:]:
                task=dict(source_image_id=row['source_image_id'],condition=condition)
                with materialize(a,task,row) as current,open_native_lr(lookup[row['clean_relpath']][condition])[0] as ref:
                    with open_native_lr(Path(current['path']))[0] as actual:
                        require(actual.size==ref.size and np.array_equal(np.asarray(actual.convert('RGB')),np.asarray(ref.convert('RGB'))),
                                'Stress corruption pixels differ from frozen generator')
    save_json(a.output/'corruption_probe.json',dict(status='PASS',train_sources=[v['source_image_id'] for v in rows],
        decoded_pixel_matches=12,blur_jpeg_generator_sha256=sha256(a.blur_generator),
        metadata='temporary PNG metadata omitted; decoded RGB identical; JPEG uses same quality/default encoder'))


def configuration(a,arm):
    from mmengine.config import Config
    cfg=Config.fromfile(str(a.model_config))
    if arm in ARMS:
        cfg.custom_imports=dict(imports=list(cfg.get('custom_imports',{}).get('imports',[]))+['rev4.fusion_plugins'],allow_failed_imports=False)
        cfg.model.type='FusionStudyDBNet';cfg.model.backbone['freeze']=True
        cfg.model.neck.type='StudyFPNC';cfg.model.neck.mode=arm
        cfg.model.neck.asf_cfg=dict(attention_type='ScaleChannelSpatial') if arm=='asf' else None
    return cfg


def evaluate(a,arm):
    from rev4.stage_a import detector_for,forward
    from rev4.decision_core import capture_candidates,CandidateMatcher
    if arm in ARMS:
        trained=read_json(a.output/'arms'/arm/'trained.json')
        require(trained['identity']==a.trial_identity and sha256(trained['checkpoint'])==trained['checkpoint_sha256'],'Training checkpoint mismatch')
        weight=Path(trained['checkpoint'])
    else:weight=a.b1_checkpoint if arm=='B1' else a.bclean_checkpoint
    config_path=a.output/'evaluation'/arm/'config.py'
    config_path.parent.mkdir(parents=True,exist_ok=True)
    write_text(config_path,configuration(a,arm).pretty_text)
    options=copy.copy(a);options.checkpoint=weight;options.model_config=config_path
    detector,native,_,_=detector_for(options)
    rows,tasks,selected=specification(a);policy=read_json(a.evaluation_protocol)
    target=a.output/'evaluation'/arm/'counts.json'
    record=read_json(target) if target.exists() else dict(identity=a.trial_identity,arm=arm,checkpoint_sha256=sha256(weight),tasks=[])
    require(record['identity']==a.trial_identity and record['checkpoint_sha256']==sha256(weight),'Evaluation identity changed')
    require([v['sample_id'] for v in record['tasks']]==[v['sample_id'] for v in tasks[:len(record['tasks'])]],'Evaluation prefix differs')
    (a.output/'scratch').mkdir(exist_ok=True)
    checked={}
    for number,task in enumerate(tasks[len(record['tasks']):],len(record['tasks'])+1):
        sid=task['source_image_id'];row=rows[sid]
        with materialize(a,task,row) as current:
            _,pmap,sample,_,restore,_,_=forward(detector,native,current,row)
            matchers={}
            for mask in (.3,.4):
                geometry=('fixed',mask,1.5,5,.01)
                prediction,scores,_=capture_candidates(native,pmap,sample,restore,geometry)
                matchers[mask]=CandidateMatcher(prediction,scores,row,policy)
            results=[]
            for mask,score in PAIRS:
                matcher=matchers[mask]
                result=matcher.score(score)
                if checked.get(task['condition'],0)<2:matcher.assert_official(score)
                results.append(result)
            checked[task['condition']]=checked.get(task['condition'],0)+1
            if arm in ('B1','B-Clean') and task['condition'] in ('clean','lr2') and checked[task['condition']]<=2:
                prior=read_json(a.fixed_run/'models'/arm/task['condition']/f'{sid}.json')
                require(results[PRIMARY]['counts']==prior['results']['0.40/0.60']['counts'][:3],'Frozen baseline sentinel differs')
            record['tasks'].append(dict(sample_id=task['sample_id'],source_image_id=sid,condition=task['condition'],
                                        image_sha256=current['image_sha256'],size=current['size'],
                                        counts=[v['counts'] for v in results],bins=[v['bins'] for v in results]))
        if number%100==0 or number==len(tasks):
            save_json(target,record);print('Evaluation',arm,number,'/',len(tasks),flush=True)
    require(len(record['tasks'])==7524,'Full/stress coverage differs')
    timing(a,arm,detector,native,rows,tasks)


def timing(a,arm,detector,native,rows,tasks):
    import torch
    from rev4.stage_a import forward
    from lowres.inference import unpack
    target=a.output/'evaluation'/arm/'timing.json'
    if target.exists():require(read_json(target)['identity']==a.trial_identity,'Timing identity changed');return
    sources=set(sorted(rows)[:4]);results=[]
    processor=copy.copy(native);processor.mask_thr=.4;processor.min_text_score=.6
    torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        if task['condition'] not in ('clean','lr2') or task['source_image_id'] not in sources:continue
        processed,_,sample,_,restore,_,_=forward(detector,native,task,rows[task['source_image_id']])
        values=[]
        with torch.inference_mode():
            for _ in range(2):detector.model(**processed,mode='tensor')
            torch.cuda.synchronize()
            for _ in range(5):
                start=time.perf_counter();pmap=detector.model(**processed,mode='tensor')
                unpack(processor(pmap,[copy.deepcopy(sample)])[0],restore);torch.cuda.synchronize()
                values.append(1000*(time.perf_counter()-start))
        results.append(dict(sample_id=task['sample_id'],condition=task['condition'],median_ms=float(np.median(values))))
    require(len(results)==8,'Incomplete timing cohort')
    save_json(target,dict(identity=a.trial_identity,measurements=results,
        median_ms={c:float(np.median([v['median_ms'] for v in results if v['condition']==c])) for c in ('clean','lr2')},
        scope='model plus CPU quad decode, 8 fixed images, 2 warmups/5 repeats; no IO/preprocess; short non-isolated screen'))


def metrics(counts):
    tp,gt,pred=map(int,counts)
    return dict(tp=tp,fp=pred-tp,fn=gt-tp,precision=tp/pred if pred else 0.,recall=tp/gt if gt else 0.,hmean=2*tp/(gt+pred) if gt+pred else 0.)


def paired_ci(before,after):
    require(before.shape==after.shape,'Unpaired source counts')
    rng=np.random.default_rng(20261107);values=[]
    for _ in range(2000):
        selected=rng.integers(0,len(before),len(before))
        values.append(100*(metrics(after[selected].sum(0))['hmean']-metrics(before[selected].sum(0))['hmean']))
    return [float(v) for v in np.quantile(values,[.025,.975])]


def report(a):
    rows,tasks,selected=specification(a);cohort=set(selected)
    names=('B1','B-Clean')+ARMS
    matrices={};bins={};stability={}
    for name in names:
        r=read_json(a.output/'evaluation'/name/'counts.json')
        require(r['identity']==a.trial_identity and [v['sample_id'] for v in r['tasks']]==[v['sample_id'] for v in tasks],'Incomplete evaluation')
        matrices[name]={};bins[name]={};stability[name]={}
        for condition in CONDITIONS:
            subset=[v for v in r['tasks'] if v['condition']==condition]
            matrices[name][condition]=np.array([v['counts'][PRIMARY] for v in subset])
            bins[name][condition]=np.asarray([v['bins'][PRIMARY] for v in subset]).sum(0).tolist()
            stability[name][condition]=[metrics(np.asarray([v['counts'][k] for v in subset]).sum(0)) for k in range(len(PAIRS))]
            if condition in ('clean','lr2'):
                matrices[name]['anchor_'+condition]=np.array([v['counts'][PRIMARY] for v in subset if v['source_image_id'] in cohort])
    totals={name:{condition:metrics(v.sum(0)) for condition,v in group.items()} for name,group in matrices.items()}
    comparisons={}
    for newer,older in [('asf','nearest'),('bilinear','nearest'),('dysample','bilinear'),('dysample','nearest')]+[(v,'B1') for v in ARMS]:
        comparisons[newer+'_minus_'+older]={condition:dict(delta_pp=100*(totals[newer][condition]['hmean']-totals[older][condition]['hmean']),
            ci95_pp=paired_ci(matrices[older][condition],matrices[newer][condition])) for condition in CONDITIONS}
    training={v:read_json(a.output/'arms'/v/'trained.json') for v in ARMS}
    require(all(v['stream']==training['nearest']['stream'] for v in training.values()),'Unequal full TRAIN streams')
    root=a.output/'results';root.mkdir(exist_ok=True)
    save_json(root/'fusion_summary.json',dict(identity=a.trial_identity,scores=totals,comparisons=comparisons,bins_lr2_reference=bins,
        threshold_pairs=PAIRS,threshold_sensitivity=stability,training=training,stress_sources=selected,no_test=True,
        timing={name:read_json(a.output/'evaluation'/name/'timing.json') for name in names},
        caveats=['development repeatedly used','one seed','corruption transfer is not cross-dataset generalization',
                 'ICDAR2015 TEST reserved for final locked benchmark','no target-specific threshold selection']))
    lines=['# Fusion and upsampling study','',
        'One allocation: technical probe and continuous 1000-update pilot gate, then 10000 total updates per arm.',
        'Backbone frozen; FPNC and DBHead trainable in every arm. Fixed mask .40 / score .60 primary. No TEST.','',
        '| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP | Tiny LR2 recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        c,d=totals[name]['clean'],totals[name]['lr2'];gt,tp=bins[name]['lr2'][0]
        lines.append(f'| {name} | {c["hmean"]:.2%} | {d["hmean"]:.2%} | {d["precision"]:.2%} | {d["recall"]:.2%} | {d["fp"]} | {tp/gt:.2%} |')
    lines+=['','## Fixed-threshold corruption transfer (same 500-source subset)','',
            '| Model | Clean | LR2 | LR4 | LR8 | Blur1 | Blur2 | JPEG30 | JPEG10 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        conditions=('anchor_clean','anchor_lr2')+CONDITIONS[2:]
        lines.append('| '+name+' | '+' | '.join(f'{totals[name][c]["hmean"]:.2%}' for c in conditions)+' |')
    lines+=['','## Paired source Hmean differences','']
    for label,values in comparisons.items():
        for condition in ('clean','lr2'):
            v=values[condition];lines.append(f'- {label} / {condition}: {v["delta_pp"]:+.3f} pp, 95% CI {v["ci95_pp"]}.')
    lines+=['','Threshold sensitivity uses the same declared settings on every condition; no target-specific best threshold is selected.',
            'Bins use GT height mapped to native LR2 across conditions. See JSON for full counts, bins, threshold curves and costs.',
            'This is one-seed exploratory development; source CIs do not capture seed or selection uncertainty. '
            'Robustness on corruptions does not prove transfer to a different dataset. Final TextVQA/ICDAR2015 TEST is not run.','']
    if getattr(a,'recover_from',None):
        lines+=['Recovery of job106090: Nearest/ASF final checkpoints and verified prediction prefixes were reused.',
                'Only temporary-file handling changed; recipe, split, evaluator and thresholds are unchanged. '
                'See recovery_import.json for exact parent hashes. Timing was remeasured in the recovery allocation.','']
    write_text(root/'fusion_report.md','\n'.join(lines))
    save_json(root/'complete.json',dict(identity=a.trial_identity,no_test=True,report_sha256=sha256(root/'fusion_report.md')))
