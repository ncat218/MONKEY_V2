"""Use the original frozen scorer on B1/ASF P-maps, including restored inputs."""
import copy
import hashlib
import time

import numpy as np

from lowres.common import read_json, sha256, identity
from rev4 import require
from rev4.durable_io import save_json, write_text
from rev4.quality_combo_plan import (ARMS, ASSETS, BRANCHES, CELLS, CONTRASTS,
                                    DETECTORS, RECIPE)


def frozen_model(repo):
    from mixab.boxquality_devA_janek import load_model
    from mixab.boxquality_janek import TRAIN_IDS_SHA256
    root = repo/ASSETS
    provenance = read_json(root/'provenance.json')
    require(provenance['commit'] == RECIPE['scorer_commit'], 'Wrong scorer commit')
    for name, value in provenance['files'].items():
        require(sha256(repo/name) == value['sha256'], 'Vendor bytes changed: '+name)
    require(sha256(repo/'docs/handoff/data/train_ids.csv') == TRAIN_IDS_SHA256,
            'Scorer TRAIN source list differs from the detector TRAIN list')
    model = load_model(root/'model_janek.npz', root/'boxquality_fit_janek.json')
    require(model['cut_full'] == RECIPE['thresholds']['Q'] and
            model['cut_raw'] == RECIPE['thresholds']['RAW'], 'Frozen cutoff changed')
    require(model['mean'].shape == model['std'].shape == (13,) and
            model['beta'].shape == (14,) and np.all(model['std'] > 0) and
            all(np.isfinite(model[k]).all() for k in ('mean','std','beta')), 'Invalid scorer tensors')
    return model


def fold(source):
    return 'A' if int(hashlib.sha256(source.encode()).hexdigest(), 16) % 2 == 0 else 'B'


def same_candidates(left, right):
    lp, lr = left
    rp, rr = right
    return lr == rr and lp == rp


def evaluate_pool(prediction, raw, features, model, row, policy, official=False):
    """Keep original packed scores/order for matching, just like Tri's evaluator."""
    from mixab.boxquality_devA_janek import full_scores
    from rev4.decision_core import CandidateMatcher
    require(len(raw) == len(features) == len(prediction['polygons']) == len(prediction['scores']),
            'Candidate feature alignment changed')
    require(all(len(f) == 13 and np.isfinite(f).all() for f in features), 'Invalid feature vector')
    keep = [i for i, r in enumerate(raw) if r >= .20]
    pred = dict(polygons=[prediction['polygons'][i] for i in keep],
                scores=[prediction['scores'][i] for i in keep])
    raw = [raw[i] for i in keep]
    probabilities = full_scores(model, [features[i] for i in keep])
    matcher = CandidateMatcher(pred, raw, row, policy)
    output = {}
    for name, scores, cut in (('F', raw, .6), ('RAW', raw, model['cut_raw']),
                              ('Q', probabilities, model['cut_full'])):
        matcher.raw_scores = scores
        result = matcher.assert_official(cut) if official else matcher.score(cut)
        indices = [i for i, score in enumerate(scores) if score >= cut]
        selected = dict(polygons=[pred['polygons'][i] for i in indices], scores=[pred['scores'][i] for i in indices])
        result['output_sha256'] = identity(selected)
        output[name] = result
    return output


def assert_reference(result, reference):
    require(result['counts'] == reference['counts'][1] and result['bins'] == reference['bins'][1]
            and set(result['matched_ann_ids']) == set(reference['matched'][1]),
            'Frozen .40/.60 control differs from parent per-source counts/matching')


def probability_forward(detector, task, row):
    """Forward without postprocessing, metrics or hashes; timing includes preprocessing."""
    import torch
    _, inputs = next(iter(detector.preprocess([task['path']], batch_size=1)))
    with torch.inference_mode():
        processed = detector.model.data_preprocessor(inputs, training=False)
        probability = detector.model(**processed, mode='tensor')
    require(probability.dtype == torch.float32 and probability.ndim == 3 and
            probability.shape[0] == 1 and probability.shape[-2:] == processed['inputs'].shape[-2:]
            and torch.isfinite(probability).all(), 'Invalid FP32 padded P-map')
    sample = processed['data_samples'][0]
    meta = sample.metainfo
    require(tuple(meta['ori_shape'][:2]) == tuple(reversed(task['size'])) and
            all(a <= b for a,b in zip(meta['img_shape'][:2], probability.shape[-2:])),
            'Invalid input resize metadata')
    return probability, sample, np.asarray(row['clean_size'])/np.asarray(task['size'])


def infer(detector, native, task, row, policy, model, official=False):
    import torch
    from mixab.boxquality_janek import capture_with_features
    from rev4.decision_core import capture_candidates
    torch.cuda.synchronize()
    start = time.perf_counter()
    probability, sample, restore = probability_forward(detector, task, row)
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    plain, scores, _ = capture_candidates(native, probability, sample, restore, ('fixed', .4, 1.5, 5, .01))
    plain_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    pred, raw, features = capture_with_features(native, probability, sample, restore)
    features_ms = (time.perf_counter()-start)*1000
    require(same_candidates((plain, scores), (pred, raw)), 'Feature capture changed native polygons/raw scores')
    start = time.perf_counter()
    arms = evaluate_pool(pred, raw, features, model, row, policy, official)
    return dict(size=task['size'], image_sha256=task['image_sha256'], arms=arms,
                diagnostic_ms=dict(forward=forward_ms, plain_capture=plain_ms, feature_capture=features_ms,
                                   evaluation=(time.perf_counter()-start)*1000),
                candidates=len(raw), feature_capture_exact=True, official_parity=official)


def parent_record(a, detector_name, task):
    root = a.sr_reference if detector_name == 'B1' else a.combo_reference
    path = root/'records'/task['condition']/(task['source_image_id']+'.json')
    record = read_json(path)
    expected = a.sr_identity_sha if detector_name == 'B1' else a.combo_identity_sha
    require(record['identity_sha256'] == expected and record['sample_id'] == task['sample_id'],
            'Parent record identity changed')
    return record, sha256(path)


def validate(record, parent, parent_sha, digest):
    require(record['identity_sha256'] == digest and record['parent_sha256'] == parent_sha
            and record['sample_id'] == parent['sample_id']
            and record['input_sha256'] == parent['input_sha256']
            and set(record['branches']) == set(BRANCHES), 'Quality cached identity mismatch')
    for branch in BRANCHES:
        r, p = record['branches'][branch], parent['branches'][branch]
        require(r['size'] == p['size'] and r['image_sha256'] == p['image_sha256']
                and r['feature_capture_exact'] is True and set(r['arms']) == set(ARMS),
                'Restored bytes/candidates/arms changed')
        assert_reference(r['arms']['F'], p)
        for result in r['arms'].values():
            tp, gt, pred = result['counts']
            require(all(type(x) is int and x >= 0 for x in result['counts'])
                    and tp <= min(gt, pred) and gt == p['counts'][1][1]
                    and len(set(result['matched_ann_ids'])) == tp
                    and all(0 <= v[1] <= v[0] for v in result['bins'])
                    and sum(v[0] for v in result['bins']) == gt
                    and sum(v[1] for v in result['bins']) == tp, 'Invalid result counts/bins')


def collect_record(a, name, task, row, detector, native, sr, device, policy, model, official=False):
    from rev4.fusion_evaluation import materialize
    from rev4.sr_recheck import resized
    parent, parent_sha = parent_record(a, name, task)
    result = dict(sample_id=task['sample_id'], source_image_id=task['source_image_id'], condition=task['condition'],
                  identity_sha256=identity(a.trial_identity), parent_sha256=parent_sha,
                  input_sha256=parent['input_sha256'], detector=name, branches={})
    with materialize(a, task, row) as original:
        require(original['image_sha256'] == parent['input_sha256'], 'Original input differs from parent')
        for branch in BRANCHES:
            with resized(a, original, branch, sr, device) as current:
                image, restoration_ms = current
                ref = parent['branches'][branch]
                require(image['image_sha256'] == ref['image_sha256'] and image['size'] == ref['size'],
                        'Quality restored pixels differ from parent')
                value = infer(detector, native, image, row, policy, model, official)
                value['restoration_ms'] = restoration_ms
                result['branches'][branch] = value
    validate(result, parent, parent_sha, identity(a.trial_identity))
    return result


def deployed(detector, native, task, row, model, arm):
    from lowres.inference import unpack
    from mixab.boxquality_janek import capture_with_features
    from mixab.boxquality_devA_janek import full_scores
    probability, sample, restore = probability_forward(detector, task, row)
    if arm == 'F':
        processor = copy.copy(native)
        processor.mask_thr, processor.min_text_score = .4, .6
        processor.unclip_ratio, processor.min_text_width, processor.epsilon_ratio = 1.5, 5, .01
        return unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
    pred, raw, features = capture_with_features(native, probability, sample, restore)
    keep = [i for i, r in enumerate(raw) if r >= .20]
    scores = full_scores(model, [features[i] for i in keep])
    keep = [i for i, score in zip(keep, scores) if score >= model['cut_full']]
    return dict(polygons=[pred['polygons'][i] for i in keep], scores=[pred['scores'][i] for i in keep])


def latency(a, name, detector, native, sr, device, rows, tasks, model):
    import torch
    from rev4.fusion_evaluation import materialize
    from rev4.sr_recheck import resized
    order = sorted(rows, key=lambda s: hashlib.sha256(('20261126:quality-latency:'+s).encode()).digest())
    chosen, warm = set(order[5:5+RECIPE['latency_sources']]), set(order[:5])
    selected = sorted((t for t in tasks if t['source_image_id'] in chosen | warm and t['condition'] in ('clean','lr2')),
                      key=lambda t: (t['source_image_id'] not in warm,t['sample_id']))
    samples=[]
    for number, task in enumerate(selected):
        sid=task['source_image_id'];row=rows[sid]
        reference=read_json(a.output/'records'/name/task['condition']/(sid+'.json'))
        with materialize(a, task, row) as original:
            for branch in BRANCHES:
                with resized(a, original, branch, sr, device) as (image, sr_ms):
                    for repeat in range(RECIPE['latency_passes'] if sid in chosen else 1):
                        for arm in (('F','Q') if (number+repeat)%2==0 else ('Q','F')):
                            torch.cuda.synchronize();start=time.perf_counter()
                            prediction=deployed(detector,native,image,row,model,arm)
                            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                            require(identity(prediction)==reference['branches'][branch]['arms'][arm]['output_sha256'],
                                    'Dedicated latency output differs from evaluated prediction')
                            if sid in chosen:
                                samples.append(dict(source_image_id=sid,condition=task['condition'],branch=branch,
                                    arm=arm,repeat=repeat,detector_post_ms=elapsed,restoration_ms=sr_ms,total_ms=elapsed+sr_ms))
    summary={}
    for condition in ('clean','lr2'):
        summary[condition]={}
        for branch in BRANCHES:
            arms={}
            for arm in ('F','Q'):
                group=[v for v in samples if v['condition']==condition and v['branch']==branch and v['arm']==arm]
                values=np.array([v['total_ms'] for v in group])
                arms[arm]=dict(mean_ms=float(values.mean()),median_ms=float(np.median(values)),
                    p95_ms=float(np.quantile(values,.95)),images=len(group)//RECIPE['latency_passes'])
            arms['Q_over_F_median']=arms['Q']['median_ms']/arms['F']['median_ms']
            summary[condition][branch]=arms
    save_json(a.output/'latency'/f'{name}.json',dict(identity_sha256=identity(a.trial_identity),summary=summary,
        samples=samples,scope='20 fixed sources x Clean/LR2, batch1, five warm-up sources, two alternating passes; '
        'restoration measured once and added equally; encoding/hashes/evaluator excluded; shared GPU',
        all_outputs_equal=True))


def run(a):
    import gc
    import torch
    from rev4.edsr import EDSR
    from rev4.fusion_evaluation import specification
    from rev4.sr_recheck import completed_b1_detector
    from rev4.sr_asf import completed_asf_detector
    model=frozen_model(a.repo)
    rows,tasks,selected=specification(a)
    require(len(rows)==2262 and len(tasks)==7524 and len(selected)==500,'Quality cohort changed')
    policy=read_json(a.evaluation_protocol);digest=identity(a.trial_identity)
    sr=EDSR().float().eval().requires_grad_(False)
    sr.load_state_dict(torch.load(a.edsr_weight,map_location='cpu',weights_only=True),strict=True)
    device=torch.device('cuda:0');sr.to(device)
    probe_file=a.output/'preflight.json'
    probes=read_json(probe_file) if probe_file.exists() else dict(identity=a.trial_identity,detectors={})
    require(probes['identity']==a.trial_identity,'Quality probe identity changed')
    # Complete BOTH detector probes before any full pass. Releasing one detector avoids VRAM accumulation.
    for name,loader in (('B1',completed_b1_detector),('ASF',completed_asf_detector)):
        if name in probes['detectors']:continue
        detector,native=loader(a);seen=set();checks=[]
        for task in tasks:
            if task['condition'] in seen:continue
            result=collect_record(a,name,task,rows[task['source_image_id']],detector,native,sr,device,policy,model,True)
            checks.append(dict(sample_id=task['sample_id'],features_and_frozen_counts_equal=True,official_parity=True))
            target=a.output/'records'/name/task['condition']/(task['source_image_id']+'.json')
            save_json(target,result);seen.add(task['condition'])
        require(len(seen)==8,'Incomplete per-condition GPU probes')
        probes['detectors'][name]=checks;save_json(probe_file,probes)
        del detector,native;gc.collect();torch.cuda.empty_cache()
    probes.update(status='PASS',no_training=True,no_test=True);save_json(probe_file,probes)
    print('PASS: both detectors x8 conditions x3 input branches; all fixed controls and official matching',flush=True)
    for name,loader in (('B1',completed_b1_detector),('ASF',completed_asf_detector)):
        detector,native=loader(a)
        for number,task in enumerate(tasks,1):
            sid=task['source_image_id'];row=rows[sid]
            require(sha256(a.clean_root/row['clean_relpath'])==row['clean_sha256'],'Clean source changed')
            target=a.output/'records'/name/task['condition']/(sid+'.json')
            if target.exists():
                record=read_json(target);parent,parent_sha=parent_record(a,name,task)
                validate(record,parent,parent_sha,digest)
            else:
                record=collect_record(a,name,task,row,detector,native,sr,device,policy,model)
                save_json(target,record)
            if number%50==0 or number==len(tasks):
                save_json(a.output/'progress.json',dict(detector=name,completed_tasks=number,total_tasks=len(tasks),
                    identity_sha256=digest,completed_cells=number*len(BRANCHES)*len(ARMS)))
                print('Quality transfer',name,number,'/',len(tasks),flush=True)
        latency_file=a.output/'latency'/f'{name}.json'
        if latency_file.exists():
            require(read_json(latency_file)['identity_sha256']==digest,'Latency identity changed')
        else:latency(a,name,detector,native,sr,device,rows,tasks,model)
        del detector,native;gc.collect();torch.cuda.empty_cache()
    from rev4.quality_combo_report import report
    report(a,tasks)
