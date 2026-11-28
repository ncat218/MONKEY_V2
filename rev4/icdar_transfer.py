"""Frozen original-ICDAR2015 transfer; official MMOCR vanilla matching only."""
import copy
import hashlib
import json
from pathlib import Path
import time
import zipfile

from lowres.common import identity, read_json, sha256
from rev4 import require
from rev4.durable_io import save_json, write_text

GT_NAME = 'textdet_test__recovered_byte_identical_to_Phase1.json'
DATA_HASHES = {
    GT_NAME: 'fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718',
    'dataset_audit.json': '8aa000d1585b442eb6a7d329a3e566bc6c3cc09890e101e5ea01a3bab6a4bc07',
    'ch4_test_images.zip': 'ecfb2488333372b7381b14ccb5ac2127de88fb935f6759fe89e5c041b0e87358',
}
MODEL_HASHES = {
    'B0': '4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82',
    'B-Clean': '861bbde2aaf0b83cc98fab148d374276aa84012a7599a43dc1e7a227628aacfc',
    'B1': '6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f',
    'Nearest': '6fee5adfbc30df8c029f52c4c92edf5d52ac6a61c0a94b82502f2469abb2938e',
    'ASF': '9cae7ad418cc906edb56667c03c0d5f9e25f02b7ad929e4024ff03ebaec38059',
}
EDSR_SHA = '2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1'
BASE_CONFIG_SHA = '94e4edd820b7cb8faca4a9414928af76f5a4f9cb225812fabad13eec74269899'
ASF_CONFIG_SHA = '271db41039259ff43b3c261ebde68c603a57cf7e61fea66bf97b4b70023127fd'
GROUPS = {
    'B0': {'native': ['B0/reproduce', 'B0/frozen', 'B0/common']},
    'B-Clean': {'native': ['B-Clean/native/F']},
    'B1': {v: [f'B1/{v}/F'] for v in ('native', 'bicubic_x2', 'edsr_x2')},
    'Nearest': {'native': ['Nearest/native/F']},
    'ASF': {'native': ['ASF/native/F'], 'bicubic_x2': ['ASF/bicubic_x2/F'],
            'edsr_x2': ['ASF/edsr_x2/F', 'ASF/edsr_x2/Q']},
}
CELLS = [c for branches in GROUPS.values() for cells in branches.values() for c in cells]
RECIPE = dict(
    dataset='original 500 ICDAR2015 images and byte-identical Phase1 GT; no LR generation',
    models=MODEL_HASHES, cells=CELLS, no_training=True, no_threshold_selection=True,
    evaluator='installed MMOCR HmeanIOUMetric; vanilla; IoU >.5; ignore overlap >.5',
    bins_original_height_px=['<8', '8-<16', '16-<32', '>=32'],
    height='axis-aligned GT polygon y extent in ORIGINAL image, fixed before upscaling',
    bootstrap_repetitions=2000, bootstrap_seed=20261107,
    B0_reproduce=dict(mask=.3, epsilon=.01, native_min_score=.3,
                      packed_score_cutoff=.6000000000000001, expected=[1800, 2077, 1978]),
    B0_frozen=dict(mask=.3, score=.45, epsilon=.005),
    common=dict(mask=.4, score=.6, epsilon=.01, unclip=1.5, min_width=5, max_candidates=3000),
    Q=dict(raw_floor=.2, cutoff=.3315908613227198, refit=False),
    restoration='same frozen EDSR64 x2 always-on, including original images; tile256/halo40',
    precision='FP32; matmul TF32 off; cuDNN TF32 on',
    latency='20 source IDs chosen by hash, 3 warmups, 2 repetitions; separate deployed paths; shared GPU',
    disclosure='ICDAR test already viewed during reproduction; B0 trained on ICDAR TRAIN. Transfer back, not fresh zero-shot.',
)


def audit_data(folder, image_dir=None, decode=False):
    """Never trust filenames alone; validate original bytes, annotations and IDs."""
    folder = Path(folder)
    for name, digest in DATA_HASHES.items():
        require(sha256(folder/name) == digest, 'ICDAR source file SHA mismatch: '+name)
    rows = read_json(folder/GT_NAME)['data_list']
    audit = read_json(folder/'dataset_audit.json')
    expected = {Path(r['img_path']).name: r for r in audit['records']}
    names = [Path(r['img_path']).name for r in rows]
    require(len(rows) == len(names) == len(set(names)) == len(expected) == 500
            and set(names) == set(expected), 'ICDAR 500 source IDs differ')
    require([sum(g['ignore'] == flag for r in rows for g in r['instances'])
             for flag in (False, True)] == [2077, 3153], 'Original valid/ignored GT changed')
    if image_dir is not None:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
    by_name = {Path(r['img_path']).name: r for r in rows}
    seen, bins, orientations = set(), [0]*4, {}
    with zipfile.ZipFile(folder/'ch4_test_images.zip') as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if member.is_dir():
                continue
            require(name in expected and name not in seen, 'Unknown/duplicate ICDAR image: '+name)
            pixels = archive.read(member)
            require(hashlib.sha256(pixels).hexdigest() == expected[name]['sha256'], 'ICDAR image SHA: '+name)
            row = by_name[name]
            require([row['width'], row['height']] == [expected[name]['width'], expected[name]['height']],
                    'GT/audit dimensions differ: '+name)
            if decode:
                from io import BytesIO
                from PIL import Image
                with Image.open(BytesIO(pixels)) as image:
                    image.load()
                    require(image.size == (row['width'], row['height']), 'Decoded dimensions: '+name)
                    orientation = int(image.getexif().get(274, 1))
                    orientations[str(orientation)] = orientations.get(str(orientation), 0)+1
                    require(orientation == 1, 'Nonidentity ICDAR EXIF: '+name)
            for gt in row['instances']:
                require(type(gt['ignore']) is bool and len(gt['polygon']) >= 6, 'Invalid original GT')
                if not gt['ignore']:
                    bins[height_bin(gt['polygon'])] += 1
            if image_dir is not None:
                dest = image_dir/name
                if dest.exists():
                    require(sha256(dest) == expected[name]['sha256'], 'Existing staged image changed')
                else:
                    dest.write_bytes(pixels)
            seen.add(name)
    require(seen == set(expected) and sum(bins) == 2077, 'Incomplete archive coverage')
    return rows, dict(status='PASS', images=500, valid_gt=2077, ignored_gt=3153,
                      file_sha256=DATA_HASHES, bin_gt=bins, exif_orientations=orientations,
                      all_image_sha256_equal=True)


def height_bin(polygon):
    ys = polygon[1::2]
    h = max(ys)-min(ys)
    require(h >= 0, 'Negative GT height')
    return 0 if h < 8 else 1 if h < 16 else 2 if h < 32 else 3


def match_matrix(matrix):
    """Exactly installed vanilla matching: row-major, strict IoU > .5."""
    import numpy as np
    gt, pred = set(), set()
    for g, p in zip(*np.nonzero(matrix > .5)):
        if int(g) not in gt and int(p) not in pred:
            gt.add(int(g)); pred.add(int(p))
    return sorted(gt)


def evaluate(prediction, row):
    import numpy as np
    from mmocr.evaluation.metrics import HmeanIOUMetric
    metric = HmeanIOUMetric(strategy='vanilla', match_iou_thr=.5, ignore_precision_thr=.5)
    # All method-specific filtering is already done; do not reapply raw .6 to Q outputs.
    metric.pred_score_thrs = np.array([0.])
    metric.process([], [dict(pred_instances=dict(
        polygons=[np.asarray(p, dtype=np.float32) for p in prediction['polygons']],
        scores=np.asarray(prediction['scores'], dtype=np.float32)),
        gt_instances=dict(polygons=[np.asarray(g['polygon'], dtype=np.float32) for g in row['instances']],
                          ignored=np.asarray([g['ignore'] for g in row['instances']], dtype=bool)))])
    result = metric.results[0]
    ious = result['iou_metric']
    matched = match_matrix(ious)
    valid = [g for g in row['instances'] if not g['ignore']]
    require(len(valid) == ious.shape[0], 'Official GT order/count differs')
    bins = [[0, 0] for _ in range(4)]
    for i, g in enumerate(valid):
        k = height_bin(g['polygon']); bins[k][0] += 1; bins[k][1] += i in matched
    return dict(counts=[len(matched), len(valid), int(ious.shape[1])], bins=bins,
                matched_gt_indices=matched)


def vector(record):
    return record['counts']+[v for pair in record['bins'] for v in pair]


def rates(values):
    tp, gt, pred = map(int, values[:3])
    return dict(tp=tp, gt=gt, pred=pred, fp=pred-tp, fn=gt-tp,
                precision=tp/pred if pred else 0., recall=tp/gt if gt else 0.,
                hmean=2*tp/(gt+pred) if gt+pred else 0.,
                bins=[dict(gt=int(values[i]), tp=int(values[i+1]),
                           recall=float(values[i+1]/values[i]) if values[i] else None)
                      for i in range(3, 11, 2)])


def deployed(detector, native, task, original_row, method, scorer):
    """One deployed inference pass; never passes GT to model or scorer."""
    import numpy as np
    from rev4.quality_combo import probability_forward
    from lowres.inference import unpack
    probability, sample, restore = probability_forward(detector, task,
        dict(clean_size=[original_row['width'], original_row['height']]))
    return postprocess(native, probability, sample, restore, method, scorer, parity=False)


def postprocess(native, probability, sample, restore, method, scorer, parity=True):
    import numpy as np
    from lowres.inference import unpack
    processor = copy.copy(native)
    processor.mask_thr, processor.min_text_score = .4, .6
    processor.epsilon_ratio = .01
    if method == 'B0/frozen':
        processor.mask_thr, processor.min_text_score, processor.epsilon_ratio = .3, .45, .005
    if method == 'B0/reproduce':
        processor.mask_thr, processor.min_text_score = .3, .3
    if method.endswith('/Q'):
        from mixab.boxquality_janek import capture_with_features
        from mixab.boxquality_devA_janek import full_scores
        pred, raw, features = capture_with_features(native, probability, sample, restore)
        pool = [i for i, v in enumerate(raw) if v >= .2]
        qs = full_scores(scorer, [features[i] for i in pool])
        keep = [i for i, q in zip(pool, qs) if q >= scorer['cut_full']]
        # Every Q pass checks that the same captured pool reproduces direct F polygons/scores.
        if parity:
            captured_f = {k: [v[i] for i, r in enumerate(raw) if r >= .6] for k, v in pred.items()}
            direct_f = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
            require(captured_f == direct_f, 'Q instrumentation changed F candidates')
        return {k: [v[i] for i in keep] for k, v in pred.items()}
    pred = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
    if method == 'B0/reproduce':
        keep = np.asarray(pred['scores'], dtype=np.float32) >= .6000000000000001
        pred = {k: [v for v, selected in zip(values, keep) if selected] for k, values in pred.items()}
    return pred


def load_detector(a, name):
    from types import SimpleNamespace
    from rev4.stage_a import detector_for
    opts = SimpleNamespace(model_config=a.configs['ASF' if name == 'ASF' else 'base'],
                           checkpoint=a.weights[name])
    detector, pp, _, _ = detector_for(opts)
    detector.model.eval().requires_grad_(False)
    return detector, pp


def task_for(row, image_root):
    name = Path(row['img_path']).name
    return dict(path=str(image_root/name), size=[row['width'], row['height']],
                source_image_id=Path(name).stem, condition='original', sample_id='original::'+Path(name).stem,
                image_sha256=sha256(image_root/name))


def evaluate_group(a, name, rows, image_root, sr, scorer):
    import gc
    import torch
    from rev4.sr_recheck import resized
    from rev4.quality_combo import probability_forward
    detector, native = load_detector(a, name)
    for branch, cells in GROUPS[name].items():
        for index, row in enumerate(rows, 1):
            source = Path(row['img_path']).stem
            targets = {cell: a.output/'records'/cell/(source+'.json') for cell in cells}
            if all(p.exists() for p in targets.values()):
                for p in targets.values():
                    require(read_json(p)['identity_sha256'] == a.digest, 'Cached ICDAR identity mismatch')
                continue
            original = task_for(row, image_root)
            with resized(a, original, branch, sr, torch.device('cuda:0')) as (task, sr_ms):
                torch.cuda.synchronize(); start = time.perf_counter()
                probability, sample, restore = probability_forward(detector, task,
                    dict(clean_size=[row['width'], row['height']]))
                torch.cuda.synchronize(); forward_ms = (time.perf_counter()-start)*1000
                for cell in cells:
                    prediction = postprocess(native, probability, sample, restore, cell, scorer)
                    result = evaluate(prediction, row)
                    save_json(targets[cell], dict(identity_sha256=a.digest, source=source,
                        source_image_sha256=original['image_sha256'], branch=branch,
                        cell=cell, result=result, prediction=prediction,
                        diagnostic_timing=dict(forward_ms=forward_ms, restoration_ms=sr_ms)))
            if index % 25 == 0 or index == len(rows):
                save_json(a.output/'heartbeat.json', dict(model=name, branch=branch, image=index, total=len(rows)))
                print('ICDAR', name, branch, index, '/', len(rows), flush=True)
        if name == 'B0':
            total = [sum(read_json(p)['result']['counts'][i] for p in
                         (a.output/'records/B0/reproduce').glob('*.json')) for i in range(3)]
            require(total == RECIPE['B0_reproduce']['expected'],
                    'B0 REPRODUCTION FAIL; stop before adapted models: '+str(total))
            save_json(a.output/'b0_reproduction_gate.json', dict(status='PASS', counts=total,
                expected=RECIPE['B0_reproduce']['expected'], no_retuning=True))
    measure_latency(a, name, rows, image_root, detector, native, sr, scorer)
    del detector
    gc.collect(); torch.cuda.empty_cache()


def measure_latency(a, name, rows, image_root, detector, native, sr, scorer):
    import numpy as np
    import torch
    from rev4.sr_recheck import resized
    ordered = sorted(rows, key=lambda r: hashlib.sha256(('20261128:icdar-latency:'+r['img_path']).encode()).digest())[:23]
    target = a.output/'latency'/(name+'.json')
    if target.exists():
        require(read_json(target)['identity_sha256'] == a.digest, 'Latency cache changed'); return
    values = {}
    for branch, cells in GROUPS[name].items():
        for n, row in enumerate(ordered):
            original = task_for(row, image_root)
            with resized(a, original, branch, sr, torch.device('cuda:0')) as (task, sr_ms):
                for repeat in range(1 if n < 3 else 2):
                    for cell in (cells if repeat == 0 else list(reversed(cells))):
                        torch.cuda.synchronize(); start = time.perf_counter()
                        pred = deployed(detector, native, task, row, cell, scorer)
                        torch.cuda.synchronize(); elapsed = 1000*(time.perf_counter()-start)+sr_ms
                        ref = read_json(a.output/'records'/cell/(original['source_image_id']+'.json'))
                        require(pred == ref['prediction'], 'Dedicated latency output mismatch')
                        if n >= 3: values.setdefault(cell, []).append(elapsed)
    save_json(target, dict(identity_sha256=a.digest, values=values,
        summary={c: dict(median_ms=float(np.median(v)), p95_ms=float(np.quantile(v,.95)), images=20)
                 for c, v in values.items()}))


def report(a, rows):
    import csv
    import io
    import numpy as np
    matrices, summary, timings = {}, {}, {}
    for name in GROUPS:
        timings.update(read_json(a.output/'latency'/(name+'.json'))['summary'])
    for cell in CELLS:
        values = [read_json(a.output/'records'/cell/(Path(r['img_path']).stem+'.json')) for r in rows]
        require(all(v['identity_sha256'] == a.digest and v['cell'] == cell for v in values), 'Final record identity')
        matrices[cell] = np.asarray([vector(v['result']) for v in values], dtype=np.int64)
        require(matrices[cell].shape == (500, 11), 'Incomplete ICDAR source coverage')
        summary[cell] = rates(matrices[cell].sum(axis=0))
        require(summary[cell]['gt'] == 2077, 'Final ICDAR GT count')
    # One fixed threshold per method. No best-test-cutoff sweep or checkpoint selection.
    draws = np.random.default_rng(RECIPE['bootstrap_seed']).integers(500, size=(2000, 500))
    def boot(m):
        v = m[draws].sum(axis=1).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.column_stack([2*v[:,0]/(v[:,1]+v[:,2]), v[:,0]/v[:,2], v[:,0]/v[:,1],
                                    v[:,4]/v[:,3], (v[:,4]+v[:,6])/(v[:,3]+v[:,5]),
                                    v[:,2]-v[:,0], v[:,1]-v[:,0]])
    comparisons = {}
    for baseline in ('B0/reproduce', 'B0/frozen', 'B0/common', 'B-Clean/native/F', 'B1/native/F', 'Nearest/native/F'):
        comparisons[baseline] = {}
        old = boot(matrices[baseline])
        for cell in CELLS:
            delta = boot(matrices[cell])-old
            item = {}
            for i, key in enumerate(('hmean_pp','precision_pp','recall_pp','tiny_recall_pp','small_lt16_recall_pp','fp_count','fn_count')):
                finite = delta[:,i][np.isfinite(delta[:,i])]*(100 if i < 5 else 1)
                item[key] = dict(ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                                 valid_bootstrap_replicates=len(finite))
            comparisons[baseline][cell] = item
    result = dict(identity=read_json(a.output/'identity.json'), recipe=RECIPE, scores=summary,
                  comparisons=comparisons, latency=timings, no_training=True, no_test_tuning=True)
    save_json(a.output/'results/icdar2015_summary.json', result)
    np.savez_compressed(a.output/'results/source_counts.npz', **{c.replace('/','__'):v for c,v in matrices.items()})
    lines = ['# Original ICDAR2015: frozen pipeline transfer', '',
        '500 original images; 2,077 valid GT and 3,153 ignored GT. Byte-identical Phase1 manifest/archive verified.',
        'B0 trained on ICDAR TRAIN; test previously viewed during reproduction. This is transfer back, not fresh zero-shot.',
        'Official installed MMOCR vanilla matching: IoU >0.5; prediction ignore overlap >0.5. No TextOCR matcher.',
        'No training, checkpoint choice, threshold search or method selection on this test. EDSR always on where specified.',
        'Tiny = original GT y-extent <8px; small additionally reported as <16px. Bins fixed before SR. N/A if no eligible GT.', '',
        '| Pipeline | P % | R % | H % | Tiny <8 R % | Small <16 R % | TP | FP | FN | Median ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cell,v in summary.items():
        t = v['bins'][0]['recall']; small_gt = sum(b['gt'] for b in v['bins'][:2]); small_tp = sum(b['tp'] for b in v['bins'][:2])
        tr = 'N/A' if t is None else f'{100*t:.2f}'
        sr = 'N/A' if not small_gt else f'{100*small_tp/small_gt:.2f}'
        lines.append(f'| {cell} | {100*v["precision"]:.2f} | {100*v["recall"]:.2f} | {100*v["hmean"]:.2f} | {tr} | {sr} | {v["tp"]} | {v["fp"]} | {v["fn"]} | {timings[cell]["median_ms"]:.1f} |')
    lines += ['', 'B0/reproduce: mask .30, epsilon .01, native score .30 then packed fixed .60; required TP/GT/pred=1800/2077/1978.',
        'B0/frozen: TextOCR-selected .30/.45, epsilon .005. B0/common and other F cells: .40/.60, epsilon .01.',
        'Q: frozen raw pool >=.20 and quality cutoff .3315908613227198; no re-fit or test cutoff selection.',
        '', '## Paired Hmean changes versus B0 and B1', '',
        '| Pipeline | vs B0 reproduce pp [95% CI] | vs B0 frozen pp [95% CI] | vs B1 pp [95% CI] |',
        '|---|---:|---:|---:|']
    for cell in CELLS:
        values = []
        for base in ('B0/reproduce','B0/frozen','B1/native/F'):
            d = 100*(summary[cell]['hmean']-summary[base]['hmean']); lo,hi = comparisons[base][cell]['hmean_pp']['ci95']
            values.append(f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        lines.append('| '+cell+' | '+' | '.join(values)+' |')
    lines += ['', '2,000 paired-image bootstrap replicates, seed20261107. CIs exclude training-seed uncertainty and historical test exposure.',
        'Latency is a separate 20-source audit, three warmups, two repetitions, includes SR and deployed detector/postprocessing; shared GPU.',
        'Q checks direct-F candidate parity during evaluation; the separate deployed timing excludes this diagnostic duplicate pass.',
        'All per-image predictions/counts and all four bins retained in records/ and JSON. Negative transfer must be reported, not retuned.']
    write_text(a.output/'results/icdar2015_report.md', '\n'.join(lines)+'\n')
    save_json(a.output/'results/complete.json', dict(status='PASS', identity_sha256=a.digest, images=500,
        cells=len(CELLS), report_sha256=sha256(a.output/'results/icdar2015_report.md'),
        summary_sha256=sha256(a.output/'results/icdar2015_summary.json')))


def run(a):
    import torch
    import PIL
    from rev4.local_scratch import local_directory
    from rev4.edsr import EDSR
    from rev4.quality_combo import frozen_model
    require(PIL.__version__ == '11.3.0' and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Require audited Pillow and one allocated GPU')
    scorer = frozen_model(a.repo)
    sr = EDSR().float().eval().requires_grad_(False)
    sr.load_state_dict(torch.load(a.edsr_weight,map_location='cpu',weights_only=True), strict=True)
    sr.to('cuda:0')
    with local_directory('monkey-icdar-original-') as image_root:
        rows, audit = audit_data(a.data, image_root, decode=True)
        save_json(a.output/'data_audit.json', audit)
        print('PASS ICDAR archive, annotation, all image hashes, dimensions and EXIF', flush=True)
        for name in GROUPS:
            if name != 'B0':
                require(read_json(a.output/'b0_reproduction_gate.json')['status'] == 'PASS', 'B0 gate required')
            evaluate_group(a, name, rows, image_root, sr, scorer)
    (a.output/'results').mkdir(exist_ok=True)
    report(a, rows)
