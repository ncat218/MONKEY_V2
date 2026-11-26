"""Query-free zoom at fixed operating points (development only, no TEST, no threshold search).

Zoom variants borrow the locate -> crop -> magnify -> detect-again mechanism of query-guided cropping
(Vision-RL2 / SD-RPN / ViCrop) and AutoFocus; the "query" is replaced by the detector's own weak or small
P-map evidence. Whole-image EDSR / bicubic on fine-tuned B1 is job 163012 and is not repeated here.
"""
import copy
import time
from pathlib import Path

import numpy as np

from lowres.common import read_json, sha256
from rev4 import require
from rev4 import zoom_core as Z
from rev4.bundle_plan import (FUSION_RUN, MAIN_CONDITIONS, REFERENCE_TOTALS, RESTORATION_RECORDS, SETTINGS,
                              STRESS_VARIANTS, STRESS_ZOOM_MODELS, ZOOM, ZOOM_RULES, ZOOM_VARIANTS, setting_key,
                              variants_for)
from rev4.durable_io import save_json

MASKS = tuple(sorted({mask for mask, _ in SETTINGS.values()}))
PAIRS = tuple(SETTINGS[name] for name in ('primary', 'common'))
PRIMARY = setting_key(*SETTINGS['primary'])
COMMON = setting_key(*SETTINGS['common'])


def geometry(mask):
    return ('fixed', mask, 1.5, 5, .01)


def load_task_image(path):
    """Decode exactly like the detector test pipeline (LoadImageFromFile, color_ignore_orientation)."""
    import mmcv

    return mmcv.imfrombytes(Path(path).read_bytes(), flag='color_ignore_orientation', backend='cv2')


def forward_array(detector, bgr):
    import torch

    _, inputs = next(iter(detector.preprocess([np.ascontiguousarray(bgr)], batch_size=1)))
    with torch.inference_mode():
        processed = detector.model.data_preprocessor(inputs, training=False)
        probability = detector.model(**processed, mode='tensor')
    require(probability.dtype == torch.float32 and probability.ndim == 3 and probability.shape[0] == 1 and
            probability.shape[-2:] == processed['inputs'].shape[-2:] and bool(torch.isfinite(probability).all()),
            'Expected a finite FP32 P-map at the padded input grid')
    sample = processed['data_samples'][0]
    require(tuple(sample.metainfo['ori_shape'][:2]) == tuple(bgr.shape[:2]), 'Unexpected crop resize metadata')
    return processed, probability, sample


def capture(native, probability, sample, restore, mask):
    from rev4.decision_core import capture_candidates

    prediction, raw, _ = capture_candidates(native, probability, sample, np.asarray(restore, dtype=np.float64),
                                            geometry(mask))
    return prediction, raw


def score_preds(preds, cutoff, row, policy):
    from rev4.decision_core import CandidateMatcher

    prediction = dict(polygons=[p['points'].reshape(-1).tolist() for p in preds], scores=[p['packed'] for p in preds])
    matcher = CandidateMatcher(prediction, [p['score'] for p in preds], row, policy)
    return Z.score_detail(matcher, cutoff, [p['h_ref'] for p in preds]), matcher


class Study:
    def __init__(self, ctx, name):
        import torch
        from rev4.stage_a import detector_for

        self.ctx, self.name = ctx, name
        options = copy.copy(ctx)
        options.checkpoint = ctx.b1_checkpoint if name == 'B1' else ctx.bclean_checkpoint
        require(sha256(options.checkpoint) == (ctx.b1_digest if name == 'B1' else ctx.bclean_digest),
                f'{name} checkpoint changed')
        self.detector, self.native, _, _ = detector_for(options)
        self.policy = read_json(ctx.evaluation_protocol)
        self.device = torch.device('cuda:0')
        self.edsr = None
        if any('edsr' in v for v in ZOOM_VARIANTS[name]):
            from rev4 import edsr
            model = edsr.EDSR()
            model.load_state_dict(torch.load(ctx.edsr_weights, map_location='cpu', weights_only=True), strict=True)
            self.edsr = model.to(self.device).eval().requires_grad_(False)

    def restore_image(self, bgr):
        import torch
        from rev4 import edsr

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        with torch.inference_mode():
            restored = edsr.restore(self.edsr, rgb, self.device)
        require(restored.shape == (2*rgb.shape[0], 2*rgb.shape[1], 3), 'Unexpected EDSR crop size')
        return np.ascontiguousarray(restored[:, :, ::-1])

    def window_pass(self, image, rects, restoration, to_clean, ratio):
        out = dict(rects=[Z.rect_to_clean(r, to_clean) for r in rects], preds={m: [] for m in MASKS}, ms=0.0)
        for index, (x0, y0, x1, y1) in enumerate(rects):
            start = time.perf_counter()
            crop = image[y0:y1, x0:x1]
            up = 1
            if restoration == 'edsr':
                crop, up = self.restore_image(crop), 2
            _, probability, sample = forward_array(self.detector, crop)
            for mask in MASKS:
                prediction, raw = capture(self.native, probability, sample, (1/up, 1/up), mask)
                shifted = Z.shift_scale(prediction, (x0, y0), to_clean)
                out['preds'][mask].append(Z.make_preds(shifted, raw, ratio, 'zoom', index))
            out['ms'] += (time.perf_counter()-start)*1000
        return out

    def record(self, task, row):
        from rev4.stage_a import forward

        variants = variants_for(self.name, task['condition'])
        params = dict(ZOOM)
        start = time.perf_counter()
        _, probability, sample, factor, restore, _, _ = forward(self.detector, self.native, task, row)
        ms = dict(global_pass=(time.perf_counter()-start)*1000)
        clean_w, clean_h = row['clean_size']
        width, height = task['size']
        to_clean = np.asarray(restore, dtype=np.float64)
        ratio = Z.ref_ratio(row['clean_size'])
        task_to_ref = max(1, clean_h//2)/height
        params['margin_clean'] = params['margin_task_px']*float(max(to_clean))
        image_rect = (0.0, 0.0, float(clean_w), float(clean_h))
        global_preds = {}
        for mask in MASKS:
            prediction, raw = capture(self.native, probability, sample, restore, mask)
            global_preds[mask] = Z.make_preds(prediction, raw, ratio, 'global')
        results = {}
        for mask, cutoff in PAIRS:
            key = setting_key(mask, cutoff)
            values, matcher = score_preds(global_preds[mask], cutoff, row, self.policy)
            fast = matcher.score(cutoff)
            require(values[:3] == fast['counts'] and values[3:11] == [x for pair in fast['bins'] for x in pair],
                    'Detailed scoring differs from CandidateMatcher')
            results.setdefault('global', {})[key] = values
        if 'poly' in variants:
            from lowres.inference import unpack

            for mask, cutoff in PAIRS:
                processor = copy.copy(self.native)
                processor.mask_thr, processor.min_text_score, processor.text_repr_type = mask, cutoff, 'poly'
                prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
                preds = Z.make_preds(prediction, prediction['scores'], ratio, 'global')
                values, _ = score_preds(preds, 0.0, row, self.policy)
                results.setdefault('poly', {})[setting_key(mask, cutoff)] = values
        needed = {ZOOM_RULES[v][0] for v in variants if v in ZOOM_RULES}
        image = None
        if needed:
            image = load_task_image(task['path'])
            require(tuple(image.shape[:2]) == (height, width), 'Decoded task image size differs')
        windows = {}
        if 'guided' in needed:
            start = time.perf_counter()
            boxes = Z.pmap_evidence(probability[0].detach().cpu().numpy(), sample.metainfo['img_shape'][:2],
                                    factor, task_to_ref, params)
            windows['guided'] = Z.select_windows(boxes, width, height, params['k_max'], params['positions'],
                                                 params['min_window_px'])
            ms['guided_select'] = (time.perf_counter()-start)*1000
        if 'oracle' in needed:
            windows['oracle'] = Z.select_windows(Z.gt_evidence(row, task['size'], params['tau_ref']), width, height,
                                                 params['k_max'], params['positions'], params['min_window_px'])
        if 'tile' in needed:
            windows['tile'] = Z.tile_windows(width, height)
        passes = {}
        for variant in variants:
            if variant not in ZOOM_RULES:
                continue
            source, mode, restoration = ZOOM_RULES[variant]
            key = (source, restoration)
            if key not in passes:
                passes[key] = self.window_pass(image, windows[source], restoration, to_clean, ratio)
                ms[f'{source}_{restoration}'] = passes[key]['ms']
            current = passes[key]
            for mask, cutoff in PAIRS:
                fused = Z.fuse(global_preds[mask], current['preds'][mask], current['rects'], image_rect, mode, cutoff, params)
                values, _ = score_preds(fused, cutoff, row, self.policy)
                results.setdefault(variant, {})[setting_key(mask, cutoff)] = values
        require(set(results) == set(variants), f'Variant coverage differs at {task["sample_id"]}')
        return dict(sample_id=task['sample_id'], source_image_id=task['source_image_id'], condition=task['condition'],
                    image_sha256=task['image_sha256'], size=list(task['size']), results=results, ms=ms,
                    windows={k: [list(map(int, r)) for r in v] for k, v in windows.items()})


def task_list(ctx, name):
    from rev4.calibration import validation_cohort
    from rev4.fusion_evaluation import specification

    if name in STRESS_ZOOM_MODELS:
        rows, tasks, _ = specification(ctx)
    else:
        rows, tasks = validation_cohort(ctx, ctx.calibration_prepared)
    return rows, list(tasks)


class Sentinels:
    """Cross-checks against frozen historical jobs on the first tasks of every condition."""
    def __init__(self, ctx, name):
        self.ctx, self.name, self.seen, self.log = ctx, name, {}, []
        self.fusion = {}
        if name in STRESS_ZOOM_MODELS:
            path = ctx.output.parent.parent/FUSION_RUN/'evaluation'/name/'counts.json'
            from rev4.fusion_plan import PRIMARY as FUSION_PRIMARY
            self.fusion = {t['sample_id']: t['counts'][FUSION_PRIMARY] for t in read_json(path)['tasks']}
        self.restoration = {}
        path = ctx.output.parent.parent/RESTORATION_RECORDS
        if name == 'B1' and path.is_file():
            self.restoration = {r['source_image_id']: r['branches'] for r in read_json(path)['records']}

    def check(self, item):
        condition = item['condition']
        if self.seen.get(condition, 0) >= 2:
            return
        self.seen[condition] = self.seen.get(condition, 0)+1
        source = item['source_image_id']
        primary = item['results']['global'][PRIMARY][:3]
        if condition in MAIN_CONDITIONS:
            prior = read_json(self.ctx.fixed_run/'models'/self.name/condition/f'{source}.json')
            require(primary == prior['results']['0.40/0.60']['counts'][:3],
                    f'{self.name} global pass differs from fixed diagnostics at {item["sample_id"]}')
            self.log.append(dict(sample_id=item['sample_id'], check='fixed diagnostics .40/.60', match=True))
        else:
            require(primary == self.fusion[item['sample_id']],
                    f'{self.name} stress pass differs from fusion counts at {item["sample_id"]}')
            self.log.append(dict(sample_id=item['sample_id'], check='fusion stress .40/.60', match=True))
        if condition == 'lr2' and source in self.restoration:
            # Log only: the required checks above already pin the global pass to frozen counts.
            match = item['results']['global'][COMMON][:3] == self.restoration[source]['lr2']['counts'][:3]
            self.log.append(dict(sample_id=item['sample_id'], check='job 13127 lr2 .30/.40', match=bool(match)))


def run(ctx, name):
    rows, tasks = task_list(ctx, name)
    root = ctx.output/'zoom'/name
    target = root/'records.json'
    identity = dict(trial=ctx.trial_identity, model=name, main_variants=list(ZOOM_VARIANTS[name]),
                    stress_variants=list(STRESS_VARIANTS) if name in STRESS_ZOOM_MODELS else [],
                    zoom=ZOOM, rules={k: list(v) for k, v in ZOOM_RULES.items()}, settings=[list(p) for p in PAIRS],
                    tasks=len(tasks))
    record = read_json(target) if target.exists() else dict(identity=identity, tasks=[])
    require(record['identity'] == identity, 'Zoom record identity changed')
    require([t['sample_id'] for t in record['tasks']] == [t['sample_id'] for t in tasks[:len(record['tasks'])]],
            'Zoom record prefix differs')
    from rev4.fusion_evaluation import materialize

    study = Study(ctx, name)
    sentinels = Sentinels(ctx, name)
    for item in record['tasks']:
        sentinels.seen[item['condition']] = 2
    for number, task in enumerate(tasks[len(record['tasks']):], len(record['tasks'])+1):
        row = rows[task['source_image_id']]
        with materialize(ctx, task, row) as current:
            item = study.record(current, row)
        sentinels.check(item)
        record['tasks'].append(item)
        if number % 50 == 0 or number == len(tasks):
            save_json(target, record)
            print(f'Zoom {name} {number}/{len(tasks)}', flush=True)
    require(len(record['tasks']) == len(tasks), 'Incomplete zoom coverage')
    totals = {c: np.asarray([t['results']['global'][PRIMARY][:3] for t in record['tasks'] if t['condition'] == c]).sum(0).tolist()
              for c in MAIN_CONDITIONS}
    require(totals == REFERENCE_TOTALS[name], f'{name} global totals differ from fixed diagnostics: {totals}')
    save_json(root/'complete.json', dict(identity=identity, tasks=len(tasks), records_sha256=sha256(target),
                                         global_primary_totals=totals, sentinels=sentinels.log))
    print('Zoom study complete:', name, flush=True)
