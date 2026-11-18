"""Post-hoc pilot: equal-budget continued B1 versus paired feature distillation.

This module never opens TEST and does not modify the historical Rev.4 recipe.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
import random
from statistics import median
import time

from lowres.common import read_json, save_json, sha256
from rev4 import b1_training, calibration, require
from rev4.feature_plan import ARMS, PAIRS, PILOT
from rev4.stage_a import detector_for, forward


def configuration(a, arm):
    require(arm in ARMS, 'Unknown feature pilot arm')
    cfg = b1_training.configuration(a, a.calibration_prepared, a.output/arm/'weights', None)
    cfg.load_from = str(a.b1_checkpoint)
    cfg.resume = False
    cfg.train_cfg.max_iters = PILOT['optimizer_updates']*PILOT['accumulation']
    cfg.default_hooks.checkpoint.update(interval=cfg.train_cfg.max_iters, max_keep_ckpts=1)
    cfg.optim_wrapper.optimizer.lr = PILOT['learning_rate']
    warmup = PILOT['warmup_updates']*PILOT['accumulation']
    cfg.param_scheduler = [dict(type='LinearLR', start_factor=.1, begin=0, end=warmup, by_epoch=False),
                           dict(type='PolyLR', power=.9, eta_min=PILOT['learning_rate']*.01,
                                begin=warmup, end=cfg.train_cfg.max_iters, by_epoch=False)]
    cfg.custom_imports['imports'] += ['rev4.prc_plugins', 'rev4.feature_distillation_plugins']
    cfg.custom_hooks = [dict(type='FeaturePilotAuditHook')]
    # Identical augmentation tracking and paired data reads in both arms.
    # The continued arm ignores the auxiliary tensors and has ordinary DBHead.
    pipeline = cfg.train_dataloader.dataset.pipeline
    replacements = {3: ('ImgAugWrapper', 'Rev4PRCImgAug'), 4: ('RandomCrop', 'Rev4PRCRandomCrop'),
                    5: ('Resize', 'Rev4PRCResize'), 7: ('PackTextDetInputs', 'FeaturePilotPack')}
    require(len(pipeline) == 8 and pipeline[6]['type'] == 'Pad' and tuple(pipeline[6]['size']) == (640, 640),
            'Unexpected locked training transforms')
    for index, (before, after) in replacements.items():
        require(pipeline[index]['type'] == before, f'Unexpected transform at {index}')
        pipeline[index]['type'] = after
    if arm == 'feature_kd':
        require(cfg.model.det_head.type == 'DBHead', 'Expected B1 DBHead')
        cfg.model.det_head.type = 'FeaturePilotHead'
    return cfg


@contextmanager
def preserve_rng():
    import numpy as np
    import torch

    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def runner_for(a, arm):
    import torch
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    register_all_modules(init_default_scope=True)
    cfg = configuration(a, arm)
    import_modules_from_strings(**cfg.custom_imports)
    runner = Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.model.float()
    load_checkpoint(runner.model, str(a.b1_checkpoint), map_location='cpu', strict=True)
    if arm == 'feature_kd':
        options = copy.copy(a)
        options.checkpoint = a.b1_checkpoint
        with preserve_rng():
            detector, _, _, _ = detector_for(options)
        runner.model.det_head.bind_teacher(detector.model, PILOT['weight'])
        require(set(runner.model.state_dict()) == set(detector.model.state_dict()),
                'Teacher or extra inference parameters leaked into the student state_dict')
    require(all(p.dtype == torch.float32 for p in runner.model.parameters() if p.is_floating_point()),
            'Student weights must be FP32')
    return runner


def probe(a):
    import torch
    from lowres.evaluate_shared import evaluate_image

    runtime = b1_training.environment_identity(a, a.calibration_prepared)
    require(runtime == a.b1_identity, 'Feature pilot environment differs from B1')
    rows, tasks = calibration.validation_cohort(a, a.calibration_prepared)
    reference = read_json(a.calibration_counts)
    require(reference['identity'] == a.calibration_summary['identity'], 'B1 reference identity differs')
    selected = set(sorted(rows)[:2])
    options = copy.copy(a)
    options.checkpoint = a.b1_checkpoint
    detector, pp, _, _ = detector_for(options)
    pp.mask_thr, pp.min_text_score = PAIRS[1]
    grid_index = reference['identity']['grid'].index(list(PAIRS[1]))
    checked = []
    for task, counts in zip(tasks, reference['counts']):
        if task['source_image_id'] not in selected:
            continue
        row = rows[task['source_image_id']]
        prediction = forward(detector, pp, task, row)[5]
        actual = evaluate_image(row['gt_clean'], row['roi_clean'], prediction,
                                read_json(a.evaluation_protocol))['counts']
        require(actual == counts[grid_index], 'B1 baseline probe did not reproduce stored counts')
        checked.append(task['sample_id'])
    require(len(checked) == 4, 'Expected four baseline identity tasks')
    del detector
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner = runner_for(a, 'feature_kd')
    model = runner.model
    model.train()
    model.det_head.fd_probe = True
    teacher = model.det_head._fd_teacher
    require(not teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'Teacher is not frozen')
    loader = iter(runner.train_dataloader)
    checked_batches = []
    for _ in range(8):
        batch = next(loader)
        conditions = [s.get('condition') for s in batch['data_samples']]
        require(conditions == ['clean', 'lr2'], 'Probe microbatch is not balanced')
        processed = model.data_preprocessor(batch, training=True)
        losses = model(**processed, mode='loss')
        kd = losses['loss_feature_kd']
        if float(losses['fd_active']) == 0:
            continue
        grad = torch.autograd.grad(kd, model.det_head._fd_probe_feature, retain_graph=True)[0]
        require(torch.isfinite(grad).all() and grad.abs().sum() > 0, 'KD has no finite nonzero student gradient')
        total = sum(value.mean() if isinstance(value, torch.Tensor) else sum(x.mean() for x in value)
                    for name, value in losses.items() if 'loss' in name)
        require(torch.isfinite(total), 'Nonfinite probe loss')
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        require(grads and all(torch.isfinite(g).all() for g in grads), 'Nonfinite parameter gradient')
        require(all(p.grad is None for p in teacher.parameters()), 'Teacher received gradients')
        checked_batches.append(dict(sample_ids=[s.get('sample_id') for s in batch['data_samples']],
                                    weight_sum=float(losses['fd_weight_sum']), kd=float(kd),
                                    gradient_l1=float(grad.abs().sum()), total_loss=float(total.detach())))
        break
    require(checked_batches, 'No valid text mask in the first eight real TRAIN batches')
    save_json(a.output/'preflight.json', dict(status='PASS', identity=a.pilot_identity, runtime=runtime,
                                            baseline_tasks=checked, train_probe=checked_batches,
                                            peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2))
    print('PASS: B1 identity, real TRAIN geometry/mask, KD gradient, frozen teacher and state_dict', flush=True)


def train(a, arm):
    import torch
    from rev4.feature_distillation_plugins import FeaturePilotAuditHook

    receipt = a.output/arm/'trained.json'
    if receipt.is_file():
        result = read_json(receipt)
        require(result['identity'] == a.pilot_identity and sha256(result['checkpoint']) == result['checkpoint_sha256'],
                'Completed pilot checkpoint identity changed')
        return
    work = a.output/arm/'weights'
    require(not (work/'last_checkpoint').exists(), 'Partial pilot exists; stop for review before restarting')
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    runner = runner_for(a, arm)
    runner.train()
    expected = PILOT['optimizer_updates']*PILOT['accumulation']
    require(runner.iter == expected, 'Pilot training ended early')
    audit = next(h for h in runner.hooks if isinstance(h, FeaturePilotAuditHook))
    require(audit.batches == expected and (arm != 'feature_kd' or audit.active > 0), 'Missing pilot audit steps')
    checkpoint = work/f'iter_{expected}.pth'
    save_json(receipt, dict(identity=a.pilot_identity, arm=arm, checkpoint=str(checkpoint),
                            checkpoint_sha256=sha256(checkpoint), optimizer_updates=PILOT['optimizer_updates'],
                            micro_iterations=runner.iter, active_microbatches=audit.active,
                            mean_kd_mse=audit.kd_sum/audit.batches, peak_gpu_mib=audit.peak_mib,
                            seconds=time.perf_counter()-start))


def evaluate(a, arm):
    from shapely.geometry import shape
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import unpack

    trained = read_json(a.output/arm/'trained.json')
    require(trained['identity'] == a.pilot_identity and sha256(trained['checkpoint']) == trained['checkpoint_sha256'],
            'Pilot evaluation checkpoint changed')
    rows, tasks = calibration.validation_cohort(a, a.calibration_prepared)
    output = a.output/arm/'validation.json'
    record = read_json(output) if output.is_file() else dict(identity=a.pilot_identity, arm=arm,
                    checkpoint_sha256=trained['checkpoint_sha256'], tasks=[])
    require(record['identity'] == a.pilot_identity and record['checkpoint_sha256'] == trained['checkpoint_sha256'],
            'Pilot validation cache identity changed')
    require([r['sample_id'] for r in record['tasks']] == [t['sample_id'] for t in tasks[:len(record['tasks'])]],
            'Pilot validation task order changed')
    options = copy.copy(a)
    options.checkpoint = Path(trained['checkpoint'])
    detector, native, _, _ = detector_for(options)
    pp = copy.copy(native)
    pp.mask_thr, pp.min_text_score = PAIRS[1]
    policy = read_json(a.evaluation_protocol)
    for index, task in enumerate(tasks[len(record['tasks']):], len(record['tasks'])+1):
        row = rows[task['source_image_id']]
        _, probability, sample, _, restore, prediction, ms = forward(detector, native, task, row)
        predictions = (prediction, unpack(pp(probability, [copy.deepcopy(sample)])[0], restore))
        scored = [evaluate_image(row['gt_clean'], row['roi_clean'], p, policy) for p in predictions]
        scale_y = task['size'][1]/row['clean_size'][1]
        heights = {str(gt['ann_id']): (shape(gt['geometry']).bounds[3]-shape(gt['geometry']).bounds[1])*scale_y
                   for gt in row['gt_clean']['valid']}
        bins = []
        for result in scored:
            matched = set(map(str, result['matched_ann_ids']))
            group = [[0, 0] for _ in range(4)]
            for ann_id, height in heights.items():
                k = sum(height >= cutoff for cutoff in (4, 8, 16))
                group[k][0] += 1
                group[k][1] += int(ann_id in matched)
            bins.append(group)
        record['tasks'].append(dict(sample_id=task['sample_id'], source_image_id=task['source_image_id'],
                                    condition=task['condition'], counts=[r['counts'] for r in scored],
                                    matched=[list(map(str, r['matched_ann_ids'])) for r in scored],
                                    bins=bins, native_latency_ms=ms))
        if index == 1 or index % 100 == 0 or index == len(tasks):
            save_json(output, record)
            print(f'{arm}: {index}/{len(tasks)} paired validation tasks', flush=True)


def summarize(a):
    import numpy as np
    from lowres.evaluate_shared import rates
    from lowres.report import interval

    rows, tasks = calibration.validation_cohort(a, a.calibration_prepared)
    sources = sorted(rows)
    reference = read_json(a.calibration_counts)
    baseline_indices = [reference['identity']['grid'].index(list(p)) for p in PAIRS]
    table = {'B1': {c: {} for c in calibration.CONDITIONS}}
    for task, counts in zip(tasks, reference['counts']):
        table['B1'][task['condition']][task['source_image_id']] = [counts[i] for i in baseline_indices]
    records = {}
    training = {}
    for arm in ARMS:
        record = read_json(a.output/arm/'validation.json')
        require(record['identity'] == a.pilot_identity and len(record['tasks']) == len(tasks), 'Incomplete validation')
        records[arm] = record['tasks']
        training[arm] = read_json(a.output/arm/'trained.json')
        table[arm] = {c: {} for c in calibration.CONDITIONS}
        for task in record['tasks']:
            table[arm][task['condition']][task['source_image_id']] = task['counts']
    rng = np.random.default_rng(20261107)
    indices = rng.integers(0, len(sources), size=(2000, len(sources)))
    scores, boots = {}, {}
    for arm, by_condition in table.items():
        scores[arm], boots[arm] = {}, {}
        for condition, per_source in by_condition.items():
            vectors = np.asarray([per_source[s] for s in sources], dtype=np.int64)
            values, boot_values = [], []
            for index, pair in enumerate(PAIRS):
                counts = vectors[:, index].sum(axis=0).tolist()
                tp, gt, pred, _ = counts
                values.append(dict(mask_thr=pair[0], score_thr=pair[1], **rates(counts), tp=tp, fp=pred-tp, fn=gt-tp))
                bt, bg, bp = vectors[:, index, :3][indices].sum(axis=1).T
                boot_values.append(2*bt/np.maximum(bg+bp, 1))
            scores[arm][condition], boots[arm][condition] = values, boot_values
    differences = {}
    for newer, older in (('continued', 'B1'), ('feature_kd', 'B1'), ('feature_kd', 'continued')):
        differences[f'{newer}_minus_{older}'] = {
            condition: [dict(hmean_pp=100*(scores[newer][condition][i]['hmean']-scores[older][condition][i]['hmean']),
                             ci95_pp=interval(100*(boots[newer][condition][i]-boots[older][condition][i])))
                        for i in range(len(PAIRS))] for condition in calibration.CONDITIONS}
    bins, timing = {}, {}
    for arm, arm_records in records.items():
        bins[arm], timing[arm] = {}, {}
        for condition in calibration.CONDITIONS:
            items = [r for r in arm_records if r['condition'] == condition]
            bins[arm][condition] = np.asarray([r['bins'][1] for r in items]).sum(axis=0).tolist()
            timing[arm][condition] = median(r['native_latency_ms'] for r in items)
    summary = dict(stage='exploratory feature pilot v1', identity=a.pilot_identity, sources=len(sources),
                   training=training, scores=scores, differences=differences, calibrated_height_bins=bins,
                   median_native_latency_ms=timing, bootstrap_seed=20261107, bootstrap_repetitions=2000,
                   interpretation='descriptive validation; one training seed; no TEST; no automatic go/no-go')
    lines = ['# Exploratory LR2 feature-distillation pilot', '',
             'Post-hoc extension, not a replacement for frozen Rev.4. Same B1 initialization and 1,000 extra optimizer updates per arm.',
             '2,262 remaining-validation sources. No TEST, no new threshold search. One training seed.', '',
             '| Model | Mask/score | Clean Hmean | LR2 Hmean | LR2 FP |', '|---|---|---:|---:|---:|']
    for arm in table:
        for i, pair in enumerate(PAIRS):
            clean, lr = scores[arm]['clean'][i], scores[arm]['lr2'][i]
            lines.append(f'| {arm} | {pair[0]:.2f}/{pair[1]:.2f} | {clean["hmean"]:.2%} | {lr["hmean"]:.2%} | {lr["fp"]:,} |')
    lines += ['', '| Comparison at fixed .30/.40 | Clean delta pp (95% CI) | LR2 delta pp (95% CI) |', '|---|---:|---:|']
    for label, values in differences.items():
        cells = []
        for c in calibration.CONDITIONS:
            item = values[c][1]
            lo, hi = item['ci95_pp']
            cells.append(f'{item["hmean_pp"]:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        lines.append(f'| {label} | {cells[0]} | {cells[1]} |')
    active = training['feature_kd']['active_microbatches']
    lines += ['', f'KD active in {active}/2000 microbatches. See pilot_summary.json for bins, training cost and native timing.',
              'CIs are descriptive paired-source intervals on development validation and exclude training-seed uncertainty.',
              'Native timing is a rough end-to-end detector measure, not a controlled deployment latency benchmark.', '']
    output = a.output/'results'
    output.mkdir(parents=True, exist_ok=True)
    (output/'pilot_report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(output/'pilot_summary.json', summary)
    print('Pilot complete:', output/'pilot_report.md', flush=True)
