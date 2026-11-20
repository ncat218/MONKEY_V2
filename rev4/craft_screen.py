"""Standalone CRAFT complementarity and TRAIN teacher-map QC; no training/fusion."""
from collections import Counter,defaultdict
import csv
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from lowres.common import identity as fingerprint,jsonl,read_json,save_json,sha256
from lowres.evaluate_shared import evaluate_image,rates,bootstrap_paired
from rev4 import require,b1_training,calibration
from rev4.craft_core import SETTINGS,preprocess,decode,complementarity
from rev4.stage_a import open_native_lr,make_task
from rev4.training_data import TRAIN_NAME,TRAIN_SOURCES


def load_model(a):
    import torch
    from rev4.craft_vendor.craft import CRAFT
    model=CRAFT(pretrained=False,freeze=False)
    state=torch.load(a.craft_weight,map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model=model.float().cuda().eval().requires_grad_(False)
    require(all(p.dtype==torch.float32 for p in model.parameters()),'Expected FP32 CRAFT')
    return model


def infer(model,task,row):
    import torch
    require(sha256(task['path'])==task['image_sha256'],'CRAFT image SHA mismatch')
    with open_native_lr(Path(task['path']))[0] as source:
        require(list(source.size)==task['size'] and int(source.getexif().get(274,1))==1,'CRAFT image dimensions/EXIF mismatch')
        rgb=np.asarray(source.convert('RGB'))
    image,metadata=preprocess(rgb)
    tensor=torch.from_numpy(image)[None].cuda()
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():maps,_=model(tensor)
    require(maps.dtype==torch.float32 and maps.shape[0]==1 and maps.shape[-1]==2 and
            tuple(maps.shape[1:3])==tuple(v//2 for v in metadata['padded_hw']) and torch.isfinite(maps).all(),
            'CRAFT output map shape/dtype/finite gate failed')
    array=maps[0].cpu().numpy()
    prediction=decode(array[:,:,0],array[:,:,1],metadata,row['clean_size'])
    elapsed=(time.perf_counter()-start)*1000
    return prediction,array[:,:,0],array[:,:,1],metadata,elapsed


def qc_train(a,model,policy):
    from shapely.geometry import shape
    from PIL import Image,ImageDraw

    rows={}
    for row in jsonl(a.calibration_prepared/TRAIN_NAME):
        source=row['source_image_id']
        require(source not in rows,'Duplicate frozen TRAIN source')
        rows[source]=row
    require(len(rows)==TRAIN_SOURCES,'Frozen TRAIN source count mismatch')
    validation=set(a.original_remaining_ids)|set(a.s1_ids)|set(a.s2_ids)
    require(not set(rows)&validation,'TRAIN and validation overlap')
    selected=sorted(rows,key=lambda s:hashlib.sha256(f'42:craft-qc:{s}'.encode()).hexdigest())[:200]
    root=a.output/'train_qc';root.mkdir(parents=True,exist_ok=True)
    candidates=defaultdict(list);metrics=[]
    for number,source in enumerate(selected,1):
        row=rows[source]
        task=make_task(source,'train_clean',a.clean_root/row['clean_relpath'],row['clean_size'],row['clean_sha256'])
        target=root/'sources'/f'{source}.json';maps_file=target.with_suffix('.npz')
        if target.is_file() and maps_file.is_file():
            record=read_json(target)
            require(record['identity']==a.trial_identity,'QC cache changed')
        else:
            prediction,region,affinity,metadata,ms=infer(model,task,row)
            scored=evaluate_image(row['gt_clean'],row['roi_clean'],prediction,policy)
            record=dict(identity=a.trial_identity,source_id=source,task=task,metadata=metadata,
                        prediction=prediction,counts=scored['counts'],matched=scored['matched_ann_ids'],
                        maps_storage='FP16 for visual QC only; not training targets or metric decoding')
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(maps_file,region=region.astype(np.float16),affinity=affinity.astype(np.float16))
            save_json(target,record)
        metrics.append(record['counts'])
        for gt in row['gt_clean']['valid']:
            geometry=shape(gt['geometry'])
            h=(geometry.bounds[3]-geometry.bounds[1])*(max(1,row['clean_size'][1]//2)/row['clean_size'][1])
            bucket=sum(h>=cut for cut in (4,8,16))
            case=dict(source_id=source,ann_id=str(gt['ann_id']),bounds=list(geometry.bounds),bin=bucket,
                      word_matched=str(gt['ann_id']) in set(map(str,record['matched'])))
            candidates[bucket].append(case)
        if number%25==0:print(f'CRAFT TRAIN QC {number}/200 sources',flush=True)
    cases=[]
    for bucket,items in sorted(candidates.items()):
        chosen=sorted(items,key=lambda c:hashlib.sha256(f'42:{c["source_id"]}:{c["ann_id"]}'.encode()).hexdigest())[:8]
        for case in chosen:
            case.update(review_label='',reviewer='',notes='',bin_population=len(items),samples_in_bin=len(chosen))
            cases.append(case)
    (root/'examples').mkdir(exist_ok=True)
    for index,case in enumerate(cases):
        row=rows[case['source_id']];record=read_json(root/'sources'/f'{case["source_id"]}.json')
        with np.load(root/'sources'/f'{case["source_id"]}.npz',allow_pickle=False) as maps:
            region,affinity=maps['region'].astype(np.float32),maps['affinity'].astype(np.float32)
        x0,y0,x1,y1=case['bounds'];margin=max(6,.3*max(x1-x0,y1-y0))
        bounds=np.array([max(0,x0-margin),max(0,y0-margin),min(row['clean_size'][0],x1+margin),min(row['clean_size'][1],y1+margin)])
        with open_native_lr(Path(record['task']['path']))[0] as source:
            crop=source.convert('RGB').crop(tuple(np.r_[np.floor(bounds[:2]),np.ceil(bounds[2:])].astype(int)))
        ratio=np.asarray(record['metadata']['actual_scale_xy'])/2
        lo=np.maximum(np.floor(bounds[:2]*ratio).astype(int),(0,0))
        hi=np.minimum(np.ceil(bounds[2:]*ratio).astype(int),(region.shape[1],region.shape[0]))
        panel=Image.new('RGB',(1200,300),'white');draw=ImageDraw.Draw(panel)
        draw.text((4,3),f'Clean TRAIN {case["source_id"]} ann={case["ann_id"]}; word match={case["word_matched"]}; character quality PENDING',fill='black')
        views=[crop]
        for array in (region,affinity):
            heat=array[lo[1]:hi[1],lo[0]:hi[0]]
            require(heat.size>0,'Empty pseudo-label QC crop')
            color=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat,0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
            views.append(Image.fromarray(color))
        for column,(view,label) in enumerate(zip(views,('Clean word context','Character region (soft)','Affinity (soft)'))):
            draw.text((column*400+4,25),label,fill='black')
            view.thumbnail((395,240));panel.paste(view,(column*400,50))
        case['image']=f'examples/{index:04d}.png';panel.save(root/case['image'])
    save_json(root/'qc_summary.json',dict(identity=a.trial_identity,source_ids=selected,seed=42,
        sources=200,word_counts=np.asarray(metrics).sum(axis=0).tolist(),review_cases=cases,
        status='PENDING_CHARACTER_REVIEW',no_character_gt=True,transcription_check=False,
        interpretation='word-level match and soft map visualization do not validate character pseudo-labels; no head training'))
    with (root/'review.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['source_id','ann_id','bin','word_matched','image','review_label','reviewer','notes']
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    return dict(sources=200,cards=len(cases),status='PENDING_CHARACTER_REVIEW')


def run(a):
    from shapely.geometry import shape
    import torch
    from rev4.paired_eda_core import fp_transitions,trace_matching

    require(b1_training.environment_identity(a,a.calibration_prepared)==a.b1_identity,'Historical runtime mismatch')
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    model=load_model(a)
    policy=read_json(a.evaluation_protocol)
    source_ids=sorted(rows);positions={s:i for i,s in enumerate(source_ids)}
    matrices={name:np.zeros((len(rows),2,4),dtype=np.int64) for name in ('CRAFT','B1')}
    paired=defaultdict(Counter);bins=defaultdict(lambda:np.zeros((4,2),dtype=np.int64));times=defaultdict(list)
    complementary_bins=defaultdict(Counter);fp_correspondence=defaultdict(Counter)
    baseline_identity=read_json(a.fixed_run/'models/B1/identity.json')
    for number,task in enumerate(tasks,1):
        source,condition=task['source_image_id'],task['condition'];row=rows[source]
        target=a.output/'samples'/condition/f'{source}.json'
        if target.is_file():
            record=read_json(target)
            require(record['identity']==a.trial_identity and record['image_sha256']==task['image_sha256'],'CRAFT cache changed')
        else:
            prediction,region,affinity,metadata,ms=infer(model,task,row)
            result,_,_,fp_indices=trace_matching(prediction,row,policy)
            record=dict(identity=a.trial_identity,sample_id=task['sample_id'],image_sha256=task['image_sha256'],
                        prediction=prediction,metadata=metadata,counts=result['counts'],matched=result['matched_ann_ids'],
                        fp_indices=fp_indices,latency_model_and_decode_ms=ms)
            save_json(target,record)
        baseline=read_json(a.fixed_run/'models/B1'/condition/f'{source}.json')
        require(baseline['identity_sha256']==fingerprint(baseline_identity) and baseline['image_sha256']==task['image_sha256'],
                'Fixed B1 baseline record changed')
        fixed=baseline['results']['0.40/0.60']
        b1_matched=[ann for ann,info in fixed['instances'].items() if info['tp']]
        current=set(map(str,record['matched']));b1_set=set(b1_matched)
        valid=[str(gt['ann_id']) for gt in row['gt_clean']['valid']]
        paired[condition].update(complementarity(b1_matched,current,valid))
        fp_correspondence[condition].update(item['kind'] for item in fp_transitions(fixed,record,row,policy))
        for gt in row['gt_clean']['valid']:
            geometry=shape(gt['geometry']);h=(geometry.bounds[3]-geometry.bounds[1])*(max(1,row['clean_size'][1]//2)/row['clean_size'][1])
            k=sum(h>=cut for cut in (4,8,16));ann=str(gt['ann_id'])
            label='both' if ann in current and ann in b1_set else ('craft_only' if ann in current else ('b1_only' if ann in b1_set else 'neither'))
            complementary_bins[condition,k].update([label])
            for name,matched in (('CRAFT',current),('B1',b1_set)):
                bins[name,condition][k]+=[1,int(ann in matched)]
        j=int(condition=='lr2');i=positions[source]
        matrices['CRAFT'][i,j]=record['counts'];matrices['B1'][i,j]=fixed['counts']
        times[condition].append(record['latency_model_and_decode_ms'])
        if number%100==0:print(f'CRAFT inference {number}/{len(tasks)} tasks',flush=True)
    qc=qc_train(a,model,policy)
    scores={}
    for name,matrix in matrices.items():
        scores[name]={}
        for j,c in enumerate(('clean','lr2')):
            vector=matrix[:,j].sum(axis=0).tolist();tp,gt,pred,_=vector
            scores[name][c]=dict(**rates(vector),tp=tp,fp=pred-tp,fn=gt-tp)
    comparisons={c:bootstrap_paired(np.concatenate([matrices['B1'][:,j],matrices['CRAFT'][:,j]],axis=1),2000,20261107)
                 for j,c in enumerate(('clean','lr2'))}
    summary=dict(identity=a.trial_identity,settings=SETTINGS,scores=scores,comparison_CRAFT_minus_B1=comparisons,
        complementarity={c:dict(v) for c,v in paired.items()},
        complementarity_by_size=[dict(condition=c,bin=k,counts=dict(v)) for (c,k),v in sorted(complementary_bins.items())],
        fp_correspondence_B1_to_CRAFT={c:dict(v) for c,v in fp_correspondence.items()},
        bins=[dict(model=n,condition=c,gt_tp=v.tolist()) for (n,c),v in bins.items()],
        craft_median_model_decode_ms={c:float(np.median(v)) for c,v in times.items()},
        craft_peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,train_teacher_qc=qc,
        no_test=True,training=False,fusion=False,
        caveats=['CRAFT defaults versus development-calibrated B1','different input resize policies recorded per image',
                 'complementary GT sets are not an achievable fused detector result','no character pseudo-label quality claim'])
    output=a.output/'results';output.mkdir(parents=True,exist_ok=True)
    save_json(output/'craft_summary.json',summary)
    lines=['# CRAFT independent complementarity screen','',
        'Pinned General CRAFT, official default thresholds .7/.4/.4, canvas 1280, mag ratio 1.5, quad. No training or TEST.',
        'B1 uses fixed .40/.60. Both use the unchanged TextOCR evaluator and actual per-axis coordinates.','',
        '| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP |','|---|---:|---:|---:|---:|---:|']
    for name,v in scores.items():
        c,d=v['clean'],v['lr2']
        lines.append(f'| {name} | {c["hmean"]:.2%} | {d["hmean"]:.2%} | {d["precision"]:.2%} | {d["recall"]:.2%} | {d["fp"]} |')
    lines+=['','## GT complementarity (separate one-to-one matching, not fusion)','']
    for condition,values in paired.items():lines.append(f'- {condition}: {dict(values)}')
    lines+=['','FP geometric correspondence (older=B1, newer=CRAFT; IoU >=.5, not a fused detector):',
            str({c:dict(v) for c,v in fp_correspondence.items()})]
    lines+=['',f'TRAIN teacher-map QC: {qc["sources"]} sources, {qc["cards"]} cards, character review PENDING.',
        'Word-level GT cannot establish correct character/affinity pseudo-labels. Teacher head training is not run.',
        'Official CRAFT resizing and Monkey resizing differ; this screen does not isolate architecture effects. '
        'Low LR2 complementarity does not by itself reject a Clean TRAIN teacher. '
        'Source CIs are descriptive on repeatedly used development, not training-seed uncertainty.',
        'CRAFT timing includes model and CPU map decode, excludes file IO/preprocessing; no direct claim versus historical B1 timings.','']
    (output/'craft_report.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(output/'complete.json',dict(no_test=True,training=False,identity=a.trial_identity,
              report_sha256=sha256(output/'craft_report.md'),teacher_qc='PENDING_CHARACTER_REVIEW'))
    print('CRAFT screen complete:',output/'craft_report.md',flush=True)
