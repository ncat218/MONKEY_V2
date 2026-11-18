"""Inference-only B0/B1/B-Clean EDA; no TEST, no threshold/model fitting."""
import copy
from pathlib import Path

from lowres.common import identity as fingerprint, read_json, save_json, sha256
from rev4 import calibration, require
from rev4.adaptation_jobs import REPO, lock_record
from rev4.adaptation_plan import COMMON, EDA_STAGE, NATIVE
from rev4.paired_eda_core import gt_attributes, instance_evidence
from rev4.stage_a import detector_for, forward

MODES = ('native', 'common', 'calibrated')


def mode_result(record, mode):
    return record['results'][record['modes'][mode]]


def pair_key(pair):
    return f'{pair[0]:.2f}/{pair[1]:.2f}'


def model_contexts(a):
    models = []
    for name, checkpoint in (('B0', a.checkpoint), ('B1', a.b1_checkpoint)):
        selected = a.calibration_summary['models'][name]['calibrated']
        reference = a.calibration_run/f'counts/{name}.json'
        record = read_json(reference)
        require(record['identity'] == a.calibration_summary['identity'], f'{name} reference identity changed')
        grid = calibration.summarize_grid(record['counts'], calibration.validation_cohort(a, a.calibration_prepared)[1])
        require(grid[calibration.GRID.index(calibration.NATIVE)]['scores'] ==
                a.calibration_summary['models'][name]['native']['scores'], f'{name} reference totals changed')
        models.append(dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                           calibrated_pair=[selected['mask_thr'], selected['min_text_score']],
                           reference_path=str(reference), reference_sha256=sha256(reference)))
    if a.include_bclean:
        root = REPO/'.runtime/bclean/v1'
        summary = read_json(root/'results/bclean_summary.json')
        require(summary['stage'] == 'exploratory B-Clean v1' and summary['no_test'] and summary['validation_sources'] == 2262 and
                summary['identity']['b0_sha256'] == a.lock['checkpoint_sha256'] and
                summary['identity']['code'] == a.code_identity, 'B-Clean summary/source identity changed')
        selected = summary['selected_checkpoint']
        checkpoint = Path(selected['checkpoint'])
        require(checkpoint.resolve().is_relative_to(root.resolve()) and
                sha256(checkpoint) == selected['identity']['checkpoint_sha256'], 'B-Clean checkpoint identity changed')
        pair = summary['calibrated']
        reference = root/'validation/calibration_counts.json'
        models.insert(1, dict(name='B-Clean', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        calibrated_pair=[pair['mask_thr'], pair['min_text_score']], reference_path=str(reference),
                        reference_sha256=sha256(reference), summary_sha256=sha256(root/'results/bclean_summary.json')))
    return models


def output_shapes(value):
    import torch

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: output_shapes(item) for key, item in value.items()}
    return [output_shapes(item) for item in value]


def probability_statistics(probability, row, factor, valid, img_shape):
    import numpy as np
    from shapely.affinity import scale, translate
    from shapely.geometry import shape
    from lowres.inference import paint

    result = {}
    for entry in row['gt_clean']['valid']:
        geometry = scale(shape(entry['geometry']), xfact=factor[0], yfact=factor[1], origin=(0, 0))
        lo = np.maximum(np.floor(geometry.bounds[:2]).astype(int), (0, 0))
        hi = np.minimum(np.ceil(geometry.bounds[2:]).astype(int)+1, (img_shape[1], img_shape[0]))
        item = dict(mean=None, q10=None, median=None, q90=None)
        if np.all(hi > lo):
            mask = np.zeros((hi[1]-lo[1], hi[0]-lo[0]), dtype=np.uint8)
            paint(mask, translate(geometry, xoff=-lo[0], yoff=-lo[1]), (1., 1.), 1)
            allowed = mask.astype(bool) & valid[lo[1]:hi[1], lo[0]:hi[0]].astype(bool)
            values = probability[lo[1]:hi[1], lo[0]:hi[0]][allowed]
            if values.size:
                q10, med, q90 = np.quantile(values, [.1, .5, .9]).tolist()
                item = dict(mean=float(values.mean()), q10=q10, median=med, q90=q90)
        result[str(entry['ann_id'])] = item
    return result


def infer_model(a, model, rows, tasks, cohort_identity):
    import numpy as np
    from mmocr.utils import offset_polygon
    from lowres.inference import kernel_evidence, unpack

    reference = read_json(model['reference_path'])
    require(len(reference['counts']) == len(tasks), 'Reference tasks/counts length mismatch')
    identity = dict(cohort=cohort_identity, model=model)
    lock_record(a.output/'models'/model['name']/'identity.json', identity)
    digest = fingerprint(identity)
    pairs = dict(native=NATIVE, common=COMMON, calibrated=tuple(model['calibrated_pair']))
    options = copy.copy(a)
    options.checkpoint = Path(model['checkpoint'])
    detector, native, _, shrink = detector_for(options)
    shapes = {}
    handles = [detector.model.backbone.register_forward_hook(lambda m, i, o: shapes.update(backbone=output_shapes(o))),
               detector.model.neck.register_forward_hook(lambda m, i, o: shapes.update(neck=output_shapes(o)))]
    processors, candidates = {}, {}
    policy = read_json(a.evaluation_protocol)
    for pair in set(pairs.values()):
        processor = copy.copy(native)
        processor.mask_thr, processor.min_text_score = pair
        processors[pair] = processor
        if pair[0] not in candidates:
            candidate = copy.copy(processor)
            candidate.min_text_score = 0.
            candidates[pair[0]] = candidate
    try:
        for number, (task, expected) in enumerate(zip(tasks, reference['counts']), 1):
            target = a.output/'models'/model['name']/task['condition']/f'{task["source_image_id"]}.json'
            if target.is_file():
                previous = read_json(target)
                require(previous['identity_sha256'] == digest and previous['sample_id'] == task['sample_id'] and
                        previous['image_sha256'] == task['image_sha256'], 'EDA per-image cache changed')
                continue
            row = rows[task['source_image_id']]
            processed, probability, sample, scale_factor, restore, pred_native, ms = forward(detector, native, task, row)
            prob = probability[0].detach().cpu().numpy()
            img_shape = tuple(sample.metainfo['img_shape'][:2])
            kernel, valid, _ = kernel_evidence(prob, img_shape, scale_factor, task, row, task['size'], shrink, offset_polygon)
            network = np.asarray(task['size'])/np.asarray(row['clean_size'])*scale_factor
            pstats = probability_statistics(prob, row, network, valid, img_shape)
            precutoff = {str(mask): unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
                         for mask, processor in candidates.items()}
            results = {}
            for pair, processor in processors.items():
                prediction = pred_native if pair == NATIVE else unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
                item = instance_evidence(prediction, precutoff[str(pair[0])], row, policy, kernel, pair[1])
                require(item['counts'] == expected[calibration.GRID.index(pair)],
                        f'EDA counts differ from historical calibration: {model["name"]} {task["sample_id"]} {pair}')
                results[pair_key(pair)] = item
            attributes = {str(gt['ann_id']): gt_attributes(gt, row['clean_size']) for gt in row['gt_clean']['valid']}
            record = dict(identity_sha256=digest, sample_id=task['sample_id'], source_image_id=task['source_image_id'],
                          condition=task['condition'], image_sha256=task['image_sha256'],
                          modes={mode: pair_key(pair) for mode, pair in pairs.items()}, results=results,
                          gt_attributes=attributes, kernel_evidence=kernel, polygon_probability_statistics=pstats,
                          precutoff=precutoff, native_latency_ms=ms,
                          dimensions=dict(native_size=task['size'], resized_hw=list(img_shape),
                                          padded_input=list(processed['inputs'].shape), pmap=list(probability.shape), **shapes))
            save_json(target, record)
            if number == 1 or number % 100 == 0 or number == len(tasks):
                print(f'EDA {model["name"]}: {number}/{len(tasks)} tasks, counts reproduced', flush=True)
    finally:
        for handle in handles:
            handle.remove()


def visualize(a, models, rows, tasks, selected, output):
    """Inference-only selected GT crops; full P-maps are not dumped for all images."""
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from lowres.inference import unpack
    from rev4.stage_a import open_native_lr

    task_index = {(t['source_image_id'], t['condition']): t for t in tasks}
    panel_dir = output/'panels'
    panel_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for index, case in enumerate(selected):
        groups.setdefault((case['source_image_id'], case['condition']), []).append((index, case))
    # Persist only small probability crops and overlays, not full-resolution feature tensors.
    for model in models:
        relevant = [(key, cases) for key, cases in groups.items() if any(model['name'] in (c['older'], c['newer']) for _, c in cases)]
        if not relevant:
            continue
        options = copy.copy(a)
        options.checkpoint = Path(model['checkpoint'])
        detector, native, _, _ = detector_for(options)
        processor = copy.copy(native)
        processor.mask_thr, processor.min_text_score = COMMON
        for (source, condition), cases in relevant:
            row, task = rows[source], task_index[source, condition]
            _, probability, sample, scale_factor, restore, _, _ = forward(detector, native, task, row)
            prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
            pmap = probability[0].detach().cpu().numpy()
            network = np.asarray(task['size'])/np.asarray(row['clean_size'])*scale_factor
            cached = read_json(a.output/'models'/model['name']/condition/f'{source}.json')
            require(cached['results'][pair_key(COMMON)]['prediction'] == prediction, 'Visualization predictions changed')
            for index, case in cases:
                if model['name'] not in (case['older'], case['newer']):
                    continue
                gt = next(g for g in row['gt_clean']['valid'] if str(g['ann_id']) == case['ann_id'])
                from shapely.geometry import shape
                geometry = shape(gt['geometry'])
                x0, y0, x1, y1 = geometry.bounds
                margin = max(8., .3*max(x1-x0, y1-y0))
                bounds = np.array([max(0, x0-margin), max(0, y0-margin),
                                   min(row['clean_size'][0], x1+margin), min(row['clean_size'][1], y1+margin)])
                native_factor = np.asarray(task['size'])/np.asarray(row['clean_size'])
                native_bounds = np.r_[np.floor(bounds[:2]*native_factor), np.ceil(bounds[2:]*native_factor)].astype(int)
                with open_native_lr(Path(task['path']))[0] as image:
                    crop = image.convert('RGB').crop(tuple(native_bounds))
                scale = min(640/max(crop.width, 1), 240/max(crop.height, 1))
                canvas = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))))
                draw = ImageDraw.Draw(canvas)
                def vertices(coords):
                    return [tuple(point) for point in (np.asarray(coords).reshape(-1, 2)*native_factor-native_bounds[:2])*scale]
                draw.line(vertices(geometry.exterior.coords)+[vertices(geometry.exterior.coords)[0]], fill='lime', width=2)
                for polygon in prediction['polygons']:
                    points = np.asarray(polygon).reshape(-1, 2)
                    if points[:, 0].max() < bounds[0] or points[:, 0].min() > bounds[2] or points[:, 1].max() < bounds[1] or points[:, 1].min() > bounds[3]:
                        continue
                    points = vertices(points)
                    draw.line(points+[points[0]], fill='orange', width=2)
                p0 = np.maximum(np.floor(bounds[:2]*network).astype(int), (0, 0))
                p1 = np.minimum(np.ceil(bounds[2:]*network).astype(int), (sample.metainfo['img_shape'][1], sample.metainfo['img_shape'][0]))
                heat = pmap[p0[1]:p1[1], p0[0]:p1[0]]
                if heat.size:
                    rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1)*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                    heat_image = Image.fromarray(rgb).resize(canvas.size)
                else:
                    heat_image = Image.new('RGB', canvas.size)
                item = cached['results'][pair_key(COMMON)]['instances'][case['ann_id']]
                match = item['assigned']
                title = f'{model["name"]} {condition} {case["group"]} TP={item["tp"]}'
                detail = f'IoU={match["iou"]:.3f}, score={match["score"]:.3f}' if match else f'candidate IoU={item["candidate_iou"]:.3f}, score={item["candidate_score"]}'
                panel = Image.new('RGB', (640, 540), 'white')
                pd = ImageDraw.Draw(panel)
                pd.text((8, 4), title, fill='black')
                pd.text((8, 24), detail, fill='black')
                panel.paste(canvas, (0, 50))
                panel.paste(heat_image, (0, 300))
                panel.save(panel_dir/f'{index:04d}_{model["name"]}.png')
    for index, case in enumerate(selected):
        files = [panel_dir/f'{index:04d}_{name}.png' for name in (case['older'], case['newer'])]
        require(all(p.is_file() for p in files), 'Incomplete paired visualization')
        images = [Image.open(p).convert('RGB') for p in files]
        combined = Image.new('RGB', (1280, 540), 'white')
        combined.paste(images[0], (0, 0)); combined.paste(images[1], (640, 0))
        folder = output/'examples'/case['group']
        folder.mkdir(parents=True, exist_ok=True)
        combined.save(folder/f'{index:04d}_{case["source_image_id"]}_{case["ann_id"]}.png')
        for im in images:
            im.close()


def run(a):
    import torch
    from rev4.paired_eda_report import summarize

    rows, tasks = calibration.validation_cohort(a, a.calibration_prepared)
    cohort = lock_record(a.output/'cohort_identity.json', dict(stage=EDA_STAGE, code=a.code_identity,
                validation_sha256=sha256(a.calibration_prepared/'validation_remaining.jsonl'),
                tasks_sha256=sha256(a.calibration_prepared/'validation_tasks.jsonl'),
                policy_sha256=sha256(a.evaluation_protocol), sources=len(rows),
                diagnostic_rule='fixed historical c0=.30 and P-map threshold=.30; descriptive not causal'))
    models = model_contexts(a)
    for model in models:
        infer_model(a, model, rows, tasks, cohort)
        torch.cuda.empty_cache()
    selected, output = summarize(a, models, rows, tasks)
    visualize(a, models, rows, tasks, selected, output)
    save_json(output/'complete.json', dict(cohort=cohort, models=models, examples=len(selected), no_test=True))
    print('Paired EDA complete:', output/'eda_report.md', flush=True)
