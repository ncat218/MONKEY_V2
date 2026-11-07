"""FP32 DBNet inference and P-map evidence with a clean identity gate."""
import copy
import importlib.metadata
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, shape

from . import evaluate_shared as evaluator
from .common import jsonl, read_json, save_json, sha256
from .prepare import E0_CHECKPOINT_SHA, E0_CONFIG_SHA, EVALUATOR_SHA, PROTOCOL_SHA, VALIDATION_SHA
from .report import markdown, summarize

CONDITIONS = ('clean', 'lr2', 'lr4', 'lr8', 'b_up_lr4')
MASK_THRESHOLD = 0.30
SCORE_THRESHOLD = 0.30


def unpack(sample, restore):
    instances = sample.pred_instances
    polygons = [(np.asarray(p, dtype=np.float64).reshape(-1, 2)*restore).reshape(-1).tolist()
                for p in instances.polygons]
    return dict(polygons=polygons, scores=instances.scores.detach().cpu().tolist())


def paint(mask, geometry, factors, value):
    if geometry.geom_type == 'Polygon':
        rings = [geometry.exterior, *geometry.interiors]
        points = [np.rint(np.asarray(ring.coords)*factors).astype(np.int32) for ring in rings]
        cv2.fillPoly(mask, points, value)
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            paint(mask, part, factors, value)


def kernel_evidence(probability, img_shape, scale_factor, task, gt, native_size, shrink_ratio, offset):
    clean = np.asarray(gt['clean_size'])
    network = np.asarray(task['size'])/clean * scale_factor
    native_y = native_size[1]/clean[1]
    height, width = img_shape
    valid = np.zeros(probability.shape, dtype=np.uint8)
    paint(valid, shape(gt['roi_clean']), network, 1)
    valid[height:, :] = 0
    valid[:, width:] = 0
    for entry in gt['gt_clean']['ignored']:
        paint(valid, shape(entry['geometry']), network, 0)
    above = probability > MASK_THRESHOLD
    evidence = {}
    for entry in gt['gt_clean']['valid']:
        geometry = shape(entry['geometry'])
        y0, y1 = geometry.bounds[1::2]
        item = dict(native_height=(y1-y0)*native_y, network_input_height=(y1-y0)*network[1],
                    kernel_pixels=0, coverage=None)
        evidence[str(entry['ann_id'])] = item
        # MMOCR's shrink distance and integer rasterization on the P-map grid.
        if geometry.geom_type != 'Polygon' or geometry.interiors:
            continue
        points = (np.asarray(geometry.exterior.coords[:-1])*network).astype(np.int32)
        polygon = Polygon(points)
        distance = int(polygon.area*(1-shrink_ratio**2)/(polygon.length+.001)+.5)
        shrunk = np.asarray(offset(points, -distance))
        if shrunk.size < 6 or shrunk.size % 2:
            continue
        kernel = shrunk.reshape(-1, 2).astype(np.int32)
        x0, y0 = np.maximum(kernel.min(axis=0), (0, 0))
        x1, y1 = np.minimum(kernel.max(axis=0)+1, (width, height))
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        cv2.fillPoly(local, [kernel-np.array([x0, y0], dtype=np.int32)], 1)
        selected = local.astype(bool) & valid[y0:y1, x0:x1].astype(bool)
        size = int(selected.sum())
        item['kernel_pixels'] = size
        if size >= 4:
            item['coverage'] = float(np.count_nonzero(above[y0:y1, x0:x1] & selected)/size)
    return evidence, valid, above


def check_identity(source, result, historical):
    if (result['counts'] != historical['counts'] or
            result['matched_ann_ids'] != historical['matched_ann_ids']):
        raise RuntimeError(f'Clean identity mismatch at {source}: '
                           f'new counts={result["counts"]}; E0 counts={historical["counts"]}')


def pilot(args):
    import torch
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.apis import TextDetInferencer
    from mmocr.utils import offset_polygon, register_all_modules

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The pilot requires exactly one visible CUDA device')
    for path, expected in ((args.model_config, E0_CONFIG_SHA), (args.checkpoint, E0_CHECKPOINT_SHA),
                           (args.evaluation_protocol, PROTOCOL_SHA), (args.validation_manifest, VALIDATION_SHA),
                           (evaluator.__file__, EVALUATOR_SHA)):
        if sha256(path) != expected:
            raise ValueError(f'Frozen identity mismatch: {path}')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    cfg = Config.fromfile(str(args.model_config))
    if cfg.get('textdet_parameter_precision') != 'fp32' or cfg.model.backbone.use_bf16:
        raise RuntimeError('Expected the audited FP32 E0 configuration')
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    register_all_modules(init_default_scope=True)
    detector = TextDetInferencer(model=str(args.model_config), weights=str(args.checkpoint), device='cuda:0')
    detector.model.float().eval()
    load_checkpoint(detector.model, str(args.checkpoint), map_location='cpu', strict=True)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in detector.model.parameters()):
        raise RuntimeError('FP32 parameter verification failed')
    pp = detector.model.det_head.postprocessor
    if (pp.mask_thr, pp.min_text_score, pp.min_text_width, pp.unclip_ratio, pp.text_repr_type,
            pp.epsilon_ratio, pp.max_candidates) != (.3, .3, 5, 1.5, 'quad', .01, 3000):
        raise ValueError('Native DB postprocessor differs from job 10326')
    candidate_pp = copy.copy(pp)
    candidate_pp.min_text_score = 0.0
    shrink = detector.model.det_head.module_loss.shrink_ratio
    if shrink != .4:
        raise ValueError('Unexpected DBNet kernel shrink ratio')
    prepared = args.output/'frozen_v2_0'
    result_dir = args.output/'results'
    fingerprint = dict(config=E0_CONFIG_SHA, checkpoint=E0_CHECKPOINT_SHA, validation=VALIDATION_SHA,
                       source={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                       tasks=sha256(prepared/'tasks.jsonl'), protocol=PROTOCOL_SHA,
                       runtime={name: importlib.metadata.version(name) for name in
                                ('torch', 'mmocr', 'mmcv', 'mmengine', 'opencv-python', 'shapely')},
                       gpu=torch.cuda.get_device_name(0), precision='fp32', cuda_matmul_tf32=False, cudnn_tf32=True)
    signature = result_dir/'inference_identity.json'
    if signature.exists() and read_json(signature) != fingerprint:
        raise RuntimeError('Existing results use different code, input or runtime identity; choose a new runtime root')
    save_json(signature, fingerprint)
    tasks = {condition: {} for condition in CONDITIONS}
    for task in jsonl(prepared/'tasks.jsonl'):
        tasks[task['condition']][task['source_image_id']] = task
    selected = set(read_json(prepared/'split_lock.json')['diagnostic'])
    if len(selected) != 500 or any(set(condition) != selected for condition in tasks.values()):
        raise ValueError('Expected five conditions for all 500 diagnostic sources')
    historical = {r['source_image_id']: r for r in jsonl(prepared/'e0_clean_matched_counts.jsonl')}
    gt_rows = {r['source_image_id']: r for r in jsonl(args.validation_manifest) if r['source_image_id'] in selected}
    protocol = read_json(args.evaluation_protocol)
    all_results = {condition: {} for condition in CONDITIONS}
    counts = Counter()
    probe = result_dir/'pmap_equivalence.json'
    torch.cuda.reset_peak_memory_stats()
    for condition in CONDITIONS:
        folder = result_dir/'samples'/condition
        folder.mkdir(parents=True, exist_ok=True)
        for source, task in sorted(tasks[condition].items()):
            receipt = folder/(source+'.json')
            map_path = folder/(source+'.npz')
            gt = gt_rows[source]
            if receipt.is_file() and map_path.is_file() and probe.is_file():
                record = read_json(receipt)
                if record['sample_id'] != task['sample_id']:
                    raise ValueError(f'Cached sample ID mismatch: {receipt}')
            else:
                if sha256(task['path']) != task['image_sha256']:
                    raise ValueError(f'Image bytes changed: {task["sample_id"]}')
                started = time.perf_counter()
                _, inputs = next(iter(detector.preprocess([task['path']], batch_size=1)))
                with torch.inference_mode():
                    processed = detector.model.data_preprocessor(inputs, training=False)
                    p = detector.model(**processed, mode='tensor')
                if (p.dtype != torch.float32 or p.ndim != 3 or p.shape[0] != 1 or
                        p.shape[-2:] != processed['inputs'].shape[-2:] or not torch.isfinite(p).all()):
                    raise RuntimeError('Expected finite FP32 P-map at the padded input grid')
                sample = processed['data_samples'][0]
                meta = sample.metainfo
                img_shape = tuple(meta['img_shape'][:2])
                factor = np.asarray(meta['scale_factor'], dtype=np.float64)
                if tuple(meta['ori_shape'][:2]) != tuple(reversed(task['size'])) or factor.shape != (2,):
                    raise ValueError('Unexpected resize metadata')
                if any(actual > padded for actual, padded in zip(img_shape, p.shape[-2:])):
                    raise ValueError('Unpadded image shape exceeds P-map')
                restore = np.asarray(gt['clean_size'])/np.asarray(task['size'])
                prediction = unpack(pp(p, [copy.deepcopy(sample)])[0], restore)
                latency = (time.perf_counter()-started)*1000
                if not probe.is_file():
                    with torch.inference_mode():
                        native = detector.model.predict(processed['inputs'], [copy.deepcopy(sample)])
                    if unpack(native[0], restore) != prediction:
                        raise RuntimeError('P-map decode differs from native model.predict')
                    save_json(probe, dict(sample_id=task['sample_id'], exact_polygons_and_scores_equal=True,
                                          shape=list(p.shape), img_shape=list(img_shape)))
                candidates = unpack(candidate_pp(p, [copy.deepcopy(sample)])[0], restore)
                result = evaluator.evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol)
                if condition == 'clean':
                    check_identity(source, result, historical[source])
                native_size = tasks['lr4'][source]['size'] if condition == 'b_up_lr4' else task['size']
                probability = p[0].detach().cpu().numpy()
                evidence, valid, above = kernel_evidence(probability, img_shape, factor, task, gt,
                                                         native_size, shrink, offset_polygon)
                temp = map_path.with_suffix('.npz.tmp')
                with temp.open('wb') as sink:
                    np.savez_compressed(sink, p=probability.astype(np.float16),
                                        valid=np.packbits(valid), above_mask_thr=np.packbits(above))
                temp.replace(map_path)
                record = dict(sample_id=task['sample_id'], metrics=result, prediction=prediction,
                              candidates_before_score=candidates, gt=evidence, latency_ms=latency,
                              resize=dict(ori_shape=meta['ori_shape'][:2], img_shape=img_shape,
                                          pmap_shape=list(probability.shape), scale_factor=factor.tolist(),
                                          lr_to_clean=restore.tolist()))
                save_json(receipt, record)
            if condition == 'clean':
                check_identity(source, record['metrics'], historical[source])
            all_results[condition][source] = record
            counts[condition] += 1
            if counts[condition] % 50 == 0:
                print(condition, counts[condition], '/ 500', flush=True)
        if condition == 'clean':
            save_json(result_dir/'clean_identity.json', dict(sources=500, counts_and_matching_equal=True,
                      historical=read_json(prepared/'preflight.json')['e0_clean_reference']))
    summary = summarize(all_results, gt_rows, protocol)
    summary['peak_allocated_gpu_mib_this_launch'] = torch.cuda.max_memory_allocated()/1024**2
    (result_dir/'pilot_report.md').write_text(markdown(summary), encoding='utf-8')
    save_json(result_dir/'pilot_summary.json', summary)
    print('Pilot complete:', result_dir/'pilot_summary.json')
