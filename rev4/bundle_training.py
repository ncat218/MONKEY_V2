"""Bundle probes, equal-recipe trainings (seed replicates, PRC2) and checkpoint evaluation."""
import copy
from fractions import Fraction
import time
import traceback
from pathlib import Path

import numpy as np

from lowres.common import read_json, sha256
from rev4 import b1_training, calibration, require
from rev4.bundle_plan import (EVAL_MASKS, GRID, MAIN_CONDITIONS, NATIVE, PRC2, PRC_SEED, SEED_REPLICATE, SETTINGS,
                              STRESS_MASKS, TRAIN_ARMS, setting_key)
from rev4.durable_io import save_json
from rev4.training_data import RECIPE, TRAIN_SOURCES


def arm_root(ctx, arm):
    require(arm in TRAIN_ARMS, f'Unknown bundle arm {arm}')
    return ctx.output/'train'/arm


def configuration(ctx, arm, resume=None):
    work = arm_root(ctx, arm)/'weights'
    cfg = b1_training.configuration(ctx, ctx.calibration_prepared, work, resume)
    if arm == 'B1_s43':
        cfg.randomness = dict(seed=SEED_REPLICATE, deterministic=False)
    elif arm == 'BClean_s43':
        cfg.randomness = dict(seed=SEED_REPLICATE, deterministic=False)
        cfg.train_dataloader.dataset.ann_file = str(ctx.bclean_annotations)
        cfg.train_dataloader.sampler.type = 'CleanControlSampler'
        pipeline = cfg.train_dataloader.dataset.pipeline
        pipeline[-1]['meta_keys'] = tuple(pipeline[-1]['meta_keys'])+('clean_draw_slot',)
        cfg.custom_imports['imports'].append('rev4.bclean_plugins')
        cfg.custom_hooks = [dict(type='CleanControlAuditHook')]
    else:
        cfg.randomness = dict(seed=PRC_SEED, deterministic=False)
        cfg.custom_imports['imports'].append('rev4.prc2_plugins')
        pipeline = cfg.train_dataloader.dataset.pipeline
        require(len(pipeline) == 8 and pipeline[6]['type'] == 'Pad' and tuple(pipeline[6]['size']) == (640, 640),
                'Require the locked B1 padding and pipeline')
        for index, (before, after) in {3: ('ImgAugWrapper', 'Rev4PRCImgAug'), 4: ('RandomCrop', 'Rev4PRCRandomCrop'),
                                       5: ('Resize', 'Rev4PRCResize'), 7: ('PackTextDetInputs', 'Rev4PRCPack')}.items():
            require(pipeline[index]['type'] == before, f'Unexpected B1 transform at {index}')
            pipeline[index]['type'] = after
        require(cfg.model.det_head.type == 'DBHead', 'Require the B1 DBNet head')
        cfg.model.det_head.type = 'Rev4PRC2DBHead'
        cfg.custom_hooks = [dict(type='Rev4PRC2AuditHook', check_after=PRC2['train_check_microbatches'],
                                 min_active=PRC2['train_min_active_fraction'])]
    # MMEngine passes the runner seed to the sampler as a default argument, so the seed also sets the
    # source draw order of Rev4BalancedLR2Sampler / CleanControlSampler.
    require('seed' not in cfg.train_dataloader.sampler, 'Sampler seed must come from cfg.randomness')
    require(cfg.load_from == str(resume if resume else ctx.checkpoint), 'Bundle arms initialize from B0')
    return cfg


def runtime_flags():
    import torch

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def fit(ctx, arm):
    import torch
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules

    root = arm_root(ctx, arm)
    done = root/'trained.json'
    if done.exists():
        record = read_json(done)
        require(record['identity'] == ctx.trial_identity, f'{arm} trained identity changed')
        for path, digest in record['checkpoints'].items():
            require(sha256(path) == digest, f'{arm} completed checkpoint changed')
        return record
    require(b1_training.environment_identity(ctx, ctx.calibration_prepared) == ctx.b1_identity,
            'Historical B1 runtime changed')
    if arm == 'BClean_s43':
        require(sha256(ctx.bclean_annotations) == ctx.trial_identity['bclean_annotation_sha256'],
                'B-Clean annotation bytes changed')
    work = root/'weights'
    work.mkdir(parents=True, exist_ok=True)
    last = work/'last_checkpoint'
    resume = Path(last.read_text(encoding='utf-8').strip()).resolve(strict=True) if last.is_file() else None
    if resume is not None:
        require(resume.is_relative_to(work.resolve()), f'{arm} resume checkpoint outside the run')
    cfg = configuration(ctx, arm, resume)
    teacher = None
    if arm == 'PRC2':
        from rev4.stage_a import detector_for
        teacher, _, _, _ = detector_for(ctx)
    runtime_flags()
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.model.float()
    load_checkpoint(runner.model, str(ctx.checkpoint), map_location='cpu', strict=True)
    if teacher is not None:
        runner.model.det_head.bind_teacher(teacher.model, PRC2)
    # runner.train() builds the loop first, so building it here keeps the B1 / PRC v1 RNG call order.
    sampler_seed = getattr(runner.train_loop.dataloader.sampler, 'seed', None)
    require(sampler_seed == cfg.randomness['seed'], f'{arm} sampler seed {sampler_seed} differs from cfg.randomness')
    runner.train()
    total = RECIPE['optimizer_updates']*RECIPE['accumulation']
    require(runner.iter == total, f'{arm} training ended early')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in runner.model.parameters()),
            f'{arm} parameters must remain FP32')
    audit = {}
    if arm == 'BClean_s43' and resume is None:
        hook = next(h for h in runner.hooks if type(h).__name__ == 'CleanControlAuditHook')
        require(hook.batches == total and hook.images == 2*total, 'B-Clean s43 view budget differs from B1')
        audit = dict(microbatches=hook.batches, images=hook.images, peak_mib=hook.peak_mib)
    if arm == 'PRC2':
        hook = next(h for h in runner.hooks if type(h).__name__ == 'Rev4PRC2AuditHook')
        audit = dict(microbatches=hook.batches, active=hook.active, active_fraction=hook.active/max(1, hook.batches),
                     selected_kernels_mean=hook.selected/max(1, hook.batches),
                     eligible_kernels_mean=hook.eligible/max(1, hook.batches), peak_mib=hook.peak_mib)
    paths = [work/f'iter_{u*RECIPE["accumulation"]}.pth' for u in RECIPE['validation_candidates']]
    result = dict(identity=ctx.trial_identity, arm=arm, seed=int(cfg.randomness['seed']),
                  checkpoints={str(p): sha256(p) for p in paths}, optimizer_updates=RECIPE['optimizer_updates'],
                  micro_iterations=runner.iter, resumed_from=str(resume) if resume else None,
                  seconds_this_launch=time.perf_counter()-start,
                  peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2, audit=audit)
    save_json(done, result)
    print('Training complete:', arm, flush=True)
    return result


def pass_counts(ctx, arm, checkpoint, updates, masks, rows, tasks, policy, tag):
    from lowres.evaluate_shared import evaluate_image
    from rev4.decision_core import CandidateMatcher
    from rev4.fusion_evaluation import materialize
    from rev4.stage_a import detector_for, forward
    from rev4.zoom_study import capture

    target = arm_root(ctx, arm)/'eval'/f'{tag}.json'
    identity = dict(trial=ctx.trial_identity, arm=arm, updates=updates, checkpoint_sha256=sha256(checkpoint),
                    masks=list(masks), tasks=len(tasks))
    record = read_json(target) if target.exists() else dict(identity=identity, tasks=[])
    require(record['identity'] == identity, f'{arm} evaluation identity changed')
    require([t['sample_id'] for t in record['tasks']] == [t['sample_id'] for t in tasks[:len(record['tasks'])]],
            f'{arm} evaluation prefix differs')
    if len(record['tasks']) == len(tasks):
        return record
    options = copy.copy(ctx)
    options.checkpoint = Path(checkpoint)
    detector, native, _, _ = detector_for(options)
    main_settings = [(m, s) for m, s in GRID if m in masks]
    if SETTINGS['primary'][0] in masks:
        main_settings.append(SETTINGS['primary'])
    stress_settings = [SETTINGS['primary'], SETTINGS['common']]
    binned = {SETTINGS['primary'], SETTINGS['common']}
    checked = 0
    for number, task in enumerate(tasks[len(record['tasks']):], len(record['tasks'])+1):
        row = rows[task['source_image_id']]
        main = task['condition'] in MAIN_CONDITIONS
        settings, here = (main_settings, masks) if main else (stress_settings, STRESS_MASKS)
        with materialize(ctx, task, row) as current:
            _, probability, sample, _, restore, native_prediction, _ = forward(detector, native, current, row)
            counts, bins = {}, {}
            for mask in here:
                prediction, raw = capture(native, probability, sample, restore, mask)
                matcher = CandidateMatcher(prediction, raw, row, policy)
                for m, s in settings:
                    if m != mask:
                        continue
                    result = matcher.score(s)
                    counts[setting_key(m, s)] = result['counts']
                    if (m, s) in binned:
                        bins[setting_key(m, s)] = [x for pair in result['bins'] for x in pair]
        if main and NATIVE[0] in here and checked < 2:
            official = evaluate_image(row['gt_clean'], row['roi_clean'], native_prediction, policy)['counts'][:3]
            require(counts[setting_key(*NATIVE)] == official, 'Indexed native counts differ from the evaluator')
            checked += 1
        record['tasks'].append(dict(sample_id=task['sample_id'], source_image_id=task['source_image_id'],
                                    condition=task['condition'], counts=counts, bins=bins))
        if number % 200 == 0 or number == len(tasks):
            save_json(target, record)
            print(f'Evaluate {arm} {tag} {number}/{len(tasks)}', flush=True)
    del detector
    return record


def totals(record, key):
    from rev4.fusion_evaluation import metrics

    by = {c: np.zeros(3, dtype=np.int64) for c in MAIN_CONDITIONS}
    for task in record['tasks']:
        if task['condition'] in by:
            by[task['condition']] += np.asarray(task['counts'][key], dtype=np.int64)
    return {c: metrics(v) for c, v in by.items()}


def choose_setting(record):
    def rank(pair):
        scores = totals(record, setting_key(*pair))
        value = sum(Fraction(2*s['tp'], (2*s['tp']+s['fp']+s['fn']) or 1) for s in scores.values())
        return value, pair == NATIVE
    best = max(GRID, key=rank)
    return dict(mask_thr=best[0], min_text_score=best[1], scores=totals(record, setting_key(*best)))


def evaluate(ctx, arm):
    import torch
    from rev4.fusion_evaluation import specification

    trained = read_json(arm_root(ctx, arm)/'trained.json')
    require(trained['identity'] == ctx.trial_identity, f'{arm} trained identity changed')
    rows, main_tasks = calibration.validation_cohort(ctx, ctx.calibration_prepared)
    _, all_tasks, _ = specification(ctx)
    require([t['sample_id'] for t in all_tasks[:len(main_tasks)]] == [t['sample_id'] for t in main_tasks],
            'Stress specification must extend the main task order')
    policy = read_json(ctx.evaluation_protocol)
    work = arm_root(ctx, arm)/'weights'
    late, early = work/'iter_20000.pth', work/'iter_10000.pth'
    for path in (late, early):
        require(sha256(path) == trained['checkpoints'][str(path)], f'{arm} checkpoint changed before evaluation')
    full = {10000: pass_counts(ctx, arm, late, 10000, EVAL_MASKS, rows, list(all_tasks), policy, 'u10000_full_stress')}
    torch.cuda.empty_cache()
    native5 = pass_counts(ctx, arm, early, 5000, (NATIVE[0],), rows, main_tasks, policy, 'u5000_native')
    key = setting_key(*NATIVE)
    native = {10000: totals(full[10000], key), 5000: totals(native5, key)}
    balanced = {u: (v['clean']['hmean']+v['lr2']['hmean'])/2 for u, v in native.items()}
    selected = max((5000, 10000), key=lambda u: (balanced[u], -u))
    if selected not in full:
        torch.cuda.empty_cache()
        full[selected] = pass_counts(ctx, arm, early, 5000, EVAL_MASKS, rows, main_tasks, policy, 'u5000_full')
    chosen = full[selected]
    names = {10000: 'u10000_full_stress', 5000: 'u5000_full'}
    summary = dict(identity=ctx.trial_identity, arm=arm, training=trained, native_selection=dict(
        native={str(u): v for u, v in native.items()}, balanced={str(u): v for u, v in balanced.items()},
        selected_updates=selected, rule='max mean native Clean/LR2 Hmean, earlier tie'),
        fixed={name: totals(chosen, setting_key(*pair)) for name, pair in SETTINGS.items()},
        calibrated=choose_setting(chosen), final_10k_fixed={name: totals(full[10000], setting_key(*pair))
                                                             for name, pair in SETTINGS.items()},
        selected_record=str(arm_root(ctx, arm)/'eval'/f'{names[selected]}.json'),
        stress_record=str(arm_root(ctx, arm)/'eval'/'u10000_full_stress.json'))
    save_json(arm_root(ctx, arm)/'evaluation_summary.json', summary)
    print('Evaluation complete:', arm, flush=True)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 16 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def prc2_probe(ctx):
    """Replay the PRC TRAIN stream on LR2 samples: alignment, the v1 matching path and v2 kernel coverage."""
    import random

    import cv2
    import imgaug
    import torch
    from mmengine.utils import import_modules_from_strings
    from mmocr.registry import DATASETS, MODELS
    from mmocr.utils import register_all_modules
    from shapely.geometry import Polygon, box, mapping, shape
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import unpack
    from rev4.prc2_plugins import kernel_selection, teacher_kernels
    from rev4.stage_a import detector_for

    teacher, _, _, _ = detector_for(ctx)
    model = teacher.model
    cfg = configuration(ctx, 'PRC2', None)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    imgaug.seed(0)
    dataset = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    require(len(dataset) == 2*TRAIN_SOURCES, 'Unexpected PRC2 TRAIN dataset size')
    loss_module = MODELS.build(copy.deepcopy(cfg.model.det_head.module_loss))
    policy = read_json(ctx.evaluation_protocol)
    positions = np.linspace(0, TRAIN_SOURCES-1, PRC2['probe_samples']).astype(int)
    records = []
    for position in positions:
        item = dataset[TRAIN_SOURCES+int(position)]
        sample = item['data_samples']
        if sample.metainfo.get('condition') != 'lr2' or 'prc_clean' not in sample:
            continue
        h, w = sample.metainfo['prc_content_shape']
        valid = sample.prc_valid
        student = item['inputs'].float().mean(0).numpy()[:h, :w]
        canvas = sample.prc_clean.float().mean(0).numpy()[:h, :w]
        region = valid.numpy()[:h, :w]
        blurred = cv2.GaussianBlur(canvas, (0, 0), 1.0)
        corr = pearson(student[region], blurred[region])
        corr_flip = pearson(student[region], np.ascontiguousarray(blurred[:, ::-1])[region])
        with torch.no_grad():
            inputs = model.data_preprocessor(dict(inputs=[sample.prc_clean.cuda()], data_samples=None), training=False)
            probability = model(**inputs, mode='tensor')[0]
        valid_dev = valid.cuda()
        masked = probability.masked_fill(~valid_dev, 0.)
        prediction = unpack(model.det_head.postprocessor.get_text_instances(masked, sample.new()), (1., 1.))
        polygons = sample.gt_instances.polygons
        ignored = sample.gt_instances.ignored.cpu().tolist()
        gt = dict(valid=[], ignored=[])
        text = np.zeros(tuple(valid.shape), dtype=np.uint8)
        for index, (poly, flag) in enumerate(zip(polygons, ignored)):
            points = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            cv2.fillPoly(text, [np.rint(points).astype(np.int32)], 1)
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                continue
            gt['ignored' if flag else 'valid'].append(dict(ann_id=index, geometry=mapping(polygon),
                                                           original_height=polygon.bounds[3]-polygon.bounds[1]))
        result = evaluate_image(gt, mapping(box(0, 0, w, h)), prediction, policy)
        shapes = []
        for coords in prediction['polygons']:
            polygon = Polygon(np.asarray(coords, dtype=float).reshape(-1, 2))
            shapes.append(polygon if polygon.is_valid else polygon.buffer(0))
        best = []
        for entry in gt['valid']:
            g = shape(entry['geometry'])
            values = []
            for p in shapes:
                if p.is_empty:
                    continue
                inter = g.intersection(p).area
                values.append(inter/max(1e-9, g.area+p.area-inter))
            best.append(max(values) if values else 0.)
        background = valid.numpy() & ~cv2.dilate(text, np.ones((5, 5), np.uint8)).astype(bool)
        prob = probability.detach().cpu().numpy()
        size = tuple(probability.shape)
        kernels, keep = teacher_kernels(loss_module, sample, size)
        labels = torch.as_tensor(np.asarray(kernels), dtype=torch.long, device=probability.device)
        allowed = torch.as_tensor(np.asarray(keep) > 0, device=probability.device) & valid_dev
        mask, selected, eligible = kernel_selection(probability, labels, allowed, PRC2['teacher_mask_thr'],
                                                    PRC2['coverage'], PRC2['min_kernel_pixels'])
        content = float(max(1, h*w))
        records.append(dict(sample_id=sample.metainfo.get('sample_id'), content=[int(h), int(w)],
                            valid_fraction=float(region.mean()), alignment_corr=corr, flipped_corr=corr_flip,
                            gt_valid=len(gt['valid']), v1_predictions=len(prediction['polygons']),
                            v1_pred_area_frac_max=float(max([p.area for p in shapes] or [0.])/content),
                            v1_ignored_roi=result['ignored_outside_roi'], v1_ignored_annotations=result['ignored_on_annotations'],
                            v1_invalid=result['invalid_prediction_count'], v1_matched=len(result['matched_ann_ids']),
                            v1_gt_best_iou_ge_05=int(sum(v >= .5 for v in best)),
                            v1_gt_best_iou_median=float(np.median(best)) if best else None,
                            teacher_max=float(prob.max()), teacher_bg_mean=float(prob[background].mean()) if background.any() else None,
                            teacher_text_mean=float(prob[text.astype(bool) & valid.numpy()].mean()) if text.any() else None,
                            v2_eligible=eligible, v2_selected=selected, v2_pixels=int(mask.sum().item())))
    require(len(records) >= PRC2['probe_samples']//2, 'Too few LR2 samples in the PRC2 probe')
    def med(key):
        values = [r[key] for r in records if r[key] is not None and r[key] == r[key]]
        return float(np.median(values)) if values else None
    summary = dict(samples=len(records), alignment_corr_median=med('alignment_corr'), flipped_corr_median=med('flipped_corr'),
                   v1_predictions_mean=float(np.mean([r['v1_predictions'] for r in records])),
                   v1_pred_area_frac_max_median=med('v1_pred_area_frac_max'),
                   v1_matched_mean=float(np.mean([r['v1_matched'] for r in records])),
                   v1_best_iou_ge_05_mean=float(np.mean([r['v1_gt_best_iou_ge_05'] for r in records])),
                   v1_ignored_roi_mean=float(np.mean([r['v1_ignored_roi'] for r in records])),
                   v1_ignored_annotations_mean=float(np.mean([r['v1_ignored_annotations'] for r in records])),
                   gt_valid_mean=float(np.mean([r['gt_valid'] for r in records])),
                   teacher_bg_mean_median=med('teacher_bg_mean'), teacher_text_mean_median=med('teacher_text_mean'),
                   v2_selected_mean=float(np.mean([r['v2_selected'] for r in records])),
                   v2_eligible_mean=float(np.mean([r['v2_eligible'] for r in records])),
                   v2_active_fraction=float(np.mean([r['v2_selected'] > 0 for r in records])))
    passed = (summary['alignment_corr_median'] is not None and
              summary['alignment_corr_median'] >= PRC2['min_alignment_corr'] and
              summary['teacher_bg_mean_median'] is not None and
              summary['teacher_bg_mean_median'] <= PRC2['max_teacher_bg_mean'] and
              summary['v2_active_fraction'] >= PRC2['min_active_fraction'] and
              summary['v2_selected_mean'] >= PRC2['min_mean_selected'])
    del model, teacher
    torch.cuda.empty_cache()
    return dict(status='PASS' if passed else 'FAIL', summary=summary, records=records,
                criteria={k: PRC2[k] for k in ('min_alignment_corr', 'max_teacher_bg_mean', 'min_active_fraction',
                                                'min_mean_selected')})


def zoom_identity(ctx):
    """Exercise the full zoom record path (incl. EDSR on crops) on four real tasks before the long phases.

    Checks: path vs in-memory decode give the same P-map; a full-image window reproduces the global pass;
    the global pass reproduces frozen fixed-diagnostics (Clean/LR2) and fusion (stress) counts.
    """
    import torch
    from rev4 import zoom_core as Z
    from rev4 import zoom_study as S
    from rev4.fusion_evaluation import materialize
    from rev4.stage_a import forward

    study = S.Study(ctx, 'B1')
    sentinels = S.Sentinels(ctx, 'B1')
    rows, tasks = S.task_list(ctx, 'B1')
    picks = [next(t for t in tasks if t['condition'] == c) for c in ('clean', 'lr2', 'lr4', 'jpeg10')]
    mask, cutoff = SETTINGS['primary']
    checks = []
    for task in picks:
        row = rows[task['source_image_id']]
        with materialize(ctx, task, row) as current:
            _, p_path, sample, _, restore, _, _ = forward(study.detector, study.native, current, row)
            image = S.load_task_image(current['path'])
            _, p_array, sample_array = S.forward_array(study.detector, image)
            difference = float((p_path-p_array).abs().max())
            require(difference <= 1e-5, f'In-memory decode changes the P-map by {difference}')
            ratio = Z.ref_ratio(row['clean_size'])
            prediction, raw = S.capture(study.native, p_path, sample, restore, mask)
            reference, _ = S.score_preds(Z.make_preds(prediction, raw, ratio, 'global'), cutoff, row, study.policy)
            crop_prediction, crop_raw = S.capture(study.native, p_array, sample_array, (1., 1.), mask)
            shifted = Z.shift_scale(crop_prediction, (0, 0), np.asarray(restore, dtype=np.float64))
            window, _ = S.score_preds(Z.make_preds(shifted, crop_raw, ratio, 'zoom', 0), cutoff, row, study.policy)
            require(window[:3] == reference[:3], 'Full-image window does not reproduce the global pass')
            started = time.perf_counter()
            item = study.record(current, row)
            seconds = time.perf_counter()-started
        sentinels.check(item)
        checks.append(dict(sample_id=task['sample_id'], pmap_max_abs_difference=difference, counts=reference[:3],
                           variants=sorted(item['results']), record_seconds=seconds))
    del study
    torch.cuda.empty_cache()
    return dict(status='PASS', checks=checks, sentinels=sentinels.log)


def config_check(ctx, arm):
    """Build the arm's TRAIN dataset and sampler as the runner will, and run the pipeline on its first draw."""
    import itertools

    from mmengine.utils import import_modules_from_strings
    from mmocr.registry import DATA_SAMPLERS, DATASETS, HOOKS, MODELS
    from mmocr.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = configuration(ctx, arm, None)
    import_modules_from_strings(**cfg.custom_imports)
    for hook in cfg.custom_hooks:
        require(HOOKS.get(hook['type']) is not None, f'{arm} hook {hook["type"]} is not registered')
    require(MODELS.get(cfg.model.det_head.type) is not None, f'{arm} head {cfg.model.det_head.type} is not registered')
    dataset = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    require(len(dataset) == 2*TRAIN_SOURCES, f'{arm} TRAIN dataset size differs')

    def draws(seed):
        sampler = DATA_SAMPLERS.build(copy.deepcopy(cfg.train_dataloader.sampler),
                                      default_args=dict(dataset=dataset, seed=seed))
        require(getattr(sampler, 'seed', None) == seed, f'{arm} sampler ignores the runner seed')
        return type(sampler).__name__, [int(i) for i in itertools.islice(iter(sampler), 8)]

    seed = int(cfg.randomness['seed'])
    sampler, first = draws(seed)
    reference = draws(PRC_SEED)[1]
    require((first == reference) == (seed == PRC_SEED), f'{arm} seed {seed} does not change the draw order as expected')
    sample = dataset[first[0]]['data_samples']
    return dict(status='PASS', seed=seed, sampler=sampler, first_draws=first, first_draws_seed42=reference,
                first_condition=sample.metainfo.get('condition'), ann_file=str(cfg.train_dataloader.dataset.ann_file),
                head=cfg.model.det_head.type, hooks=[h['type'] for h in cfg.custom_hooks])


def probe(ctx):
    require(b1_training.environment_identity(ctx, ctx.calibration_prepared) == ctx.b1_identity,
            'Historical B1 runtime changed')
    runtime_flags()
    configs = {}
    for arm in TRAIN_ARMS:
        try:
            configs[arm] = config_check(ctx, arm)
        except Exception:
            configs[arm] = dict(status='FAIL', error=traceback.format_exc())
    if all(configs[a]['status'] == 'PASS' for a in ('B1_s43', 'BClean_s43')) and \
            configs['B1_s43']['first_draws'] != configs['BClean_s43']['first_draws']:
        configs['BClean_s43'] = dict(status='FAIL', error='B-Clean s43 draw order differs from B1 s43')
    try:
        zoom = zoom_identity(ctx)
    except Exception:
        zoom = dict(status='FAIL', error=traceback.format_exc())
    try:
        prc2 = prc2_probe(ctx)
    except Exception:
        prc2 = dict(status='FAIL', error=traceback.format_exc())
    save_json(ctx.output/'preflight.json', dict(status='PASS', identity=ctx.trial_identity, configs=configs,
                                                zoom=zoom, prc2=prc2))
    print('Probe: configs', {a: c['status'] for a, c in configs.items()}, '| zoom', zoom['status'],
          '| PRC2', prc2['status'], flush=True)
