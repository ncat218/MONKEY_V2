"""FP32 Clean/LR2 baseline training and locked native checkpoint selection."""
import copy
import importlib.metadata
from pathlib import Path
import time

from lowres.common import jsonl, read_json, save_json, sha256
from rev4 import require
from rev4.stage_a import detector_for, forward
from rev4.training_data import RECIPE, TRAIN_NAME, TRAIN_SHA256, TRAIN_SOURCES, VAL_SOURCES


def validate_prepared(a):
    folder = a.output/'prepared'
    receipt = read_json(folder/'prepared.json')
    require(receipt['train_sources'] == TRAIN_SOURCES and receipt['remaining_validation_sources'] == VAL_SOURCES and
            receipt['generator_sha256'] == a.lock['lr_generator_sha256'] and
            receipt['file_sha256'][TRAIN_NAME] == TRAIN_SHA256, 'B1 prepared identity mismatch')
    for name, digest in receipt['file_sha256'].items():
        require(sha256(folder/name) == digest, f'Prepared B1 artifact changed: {name}')
    return folder


def configuration(a, prepared, work, resume):
    from mmengine.config import Config

    cfg = Config.fromfile(str(a.model_config))
    require(cfg.get('textdet_parameter_precision') == 'fp32' and not cfg.model.backbone.use_bf16,
            'Expected the locked B0 FP32 architecture')
    cfg.model.backbone['freeze'] = False
    pipeline = copy.deepcopy(cfg.train_pipeline)
    require(pipeline[0]['type'] == 'LoadImageFromFile' and pipeline[-1]['type'] == 'PackTextDetInputs',
            'Unexpected historical training pipeline')
    pipeline[0]['type'] = 'Rev4LoadCleanOrLR2'
    pipeline[-1]['meta_keys'] = tuple(pipeline[-1]['meta_keys'])+('condition', 'sample_id')
    imports = list(cfg.get('custom_imports', {}).get('imports', []))
    cfg.custom_imports = dict(imports=imports+['rev4.b1_plugins'], allow_failed_imports=False)
    cfg.train_dataloader = dict(batch_size=RECIPE['micro_batch_size'], num_workers=4,
                                persistent_workers=True, pin_memory=True, drop_last=True,
                                sampler=dict(type='Rev4BalancedLR2Sampler'),
                                dataset=dict(type='OCRDataset', data_root='', data_prefix=dict(img_path=''),
                                             ann_file=str(prepared/'train_annotations.json'), filter_cfg=None,
                                             pipeline=pipeline, test_mode=False))
    cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
    cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
    total = RECIPE['optimizer_updates']*RECIPE['accumulation']
    warmup = RECIPE['warmup_optimizer_updates']*RECIPE['accumulation']
    checkpoint = RECIPE['checkpoint_every_updates']*RECIPE['accumulation']
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=total, val_interval=total)
    cfg.optim_wrapper = dict(type='OptimWrapper', accumulative_counts=RECIPE['accumulation'],
                             optimizer=dict(type='AdamW', lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay']),
                             clip_grad=dict(max_norm=RECIPE['clip_grad'], norm_type=2, error_if_nonfinite=True))
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=.1, begin=0, end=warmup, by_epoch=False),
        dict(type='PolyLR', power=.9, eta_min=RECIPE['learning_rate']*.01,
             begin=warmup, end=total, by_epoch=False),
    ]
    cfg.default_hooks = dict(timer=dict(type='IterTimerHook'),
                             logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
                             param_scheduler=dict(type='ParamSchedulerHook'),
                             sampler_seed=dict(type='DistSamplerSeedHook'),
                             checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=checkpoint,
                                             max_keep_ckpts=3, save_last=True, save_optimizer=True,
                                             save_param_scheduler=True, filename_tmpl='iter_{}.pth'))
    cfg.custom_hooks = [dict(type='Rev4B1AuditHook')]
    cfg.visualizer = dict(type='TextDetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
    cfg.log_processor = dict(type='LogProcessor', by_epoch=False, window_size=50)
    cfg.work_dir = str(work)
    cfg.load_from = str(resume if resume else a.checkpoint)
    cfg.resume = resume is not None
    cfg.launcher = 'none'
    cfg.auto_scale_lr = None
    cfg.randomness = dict(seed=RECIPE['seed'], deterministic=False)
    return cfg


def environment_identity(a, prepared):
    import PIL
    import torch

    runtime = {name: importlib.metadata.version(name) for name in a.s1_identity['runtime']}
    require(runtime == a.s1_identity['runtime'] and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == a.s1_identity['gpu'], 'Require the audited H200 MIG environment')
    require(PIL.__version__ == RECIPE['pillow'], 'B1 requires the audited LR2 Pillow version')
    return dict(recipe=RECIPE, prepared_sha256=sha256(prepared/'prepared.json'),
                checkpoint_sha256=a.lock['checkpoint_sha256'], config_sha256=a.lock['config_sha256'],
                stage_a_summary_sha256=a.s2a_summary_sha256, stage_b_summary_sha256=a.s2b_summary_sha256,
                runtime=runtime, gpu=torch.cuda.get_device_name(0),
                code={name: sha256(Path(__file__).with_name(name)) for name in
                      ('training_data.py', 'b1_plugins.py', 'b1_training.py')})


def fit(a, prepared):
    import torch
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules

    work = a.output/'weights'
    work.mkdir(parents=True, exist_ok=True)
    completed = a.output/'trained.json'
    if completed.is_file():
        existing = read_json(completed)
        for path, digest in existing['checkpoints'].items():
            require(sha256(path) == digest, f'Completed B1 checkpoint changed: {path}')
        return existing
    last = work/'last_checkpoint'
    resume = None
    if last.is_file():
        resume = Path(last.read_text(encoding='utf-8').strip()).resolve(strict=True)
        require(resume.is_relative_to(work.resolve()), 'B1 resume checkpoint outside the locked work directory')
    cfg = configuration(a, prepared, work, resume)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.model.float()
    load_checkpoint(runner.model, str(a.checkpoint), map_location='cpu', strict=True)
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in runner.model.parameters()),
            'B1 initial parameters must be FP32')
    runner.train()
    require(runner.iter == RECIPE['optimizer_updates']*RECIPE['accumulation'], 'B1 training ended early')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in runner.model.parameters()),
            'B1 final parameters must be FP32')
    paths = [work/f'iter_{updates*RECIPE["accumulation"]}.pth' for updates in RECIPE['validation_candidates']]
    result = dict(checkpoints={str(path): sha256(path) for path in paths},
                  optimizer_updates=RECIPE['optimizer_updates'], micro_iterations=runner.iter,
                  resumed_from=str(resume) if resume else None,
                  train_seconds_this_launch=time.perf_counter()-start,
                  peak_allocated_gpu_mib_this_launch=torch.cuda.max_memory_allocated()/1024**2)
    save_json(completed, result)
    return result


def checkpoint_score(a, prepared, checkpoint, digest, updates):
    from lowres.evaluate_shared import evaluate_image, rates

    target = a.output/'validation'/f'updates_{updates}.json'
    identity = dict(checkpoint_sha256=digest, updates=updates,
                    validation_sha256=sha256(prepared/'validation_remaining.jsonl'),
                    tasks_sha256=sha256(prepared/'validation_tasks.jsonl'))
    if target.is_file():
        previous = read_json(target)
        require(previous['identity'] == identity, 'Cached validation checkpoint identity mismatch')
        return previous
    rows = {row['source_image_id']: row for row in jsonl(prepared/'validation_remaining.jsonl')}
    require(len(rows) == VAL_SOURCES and not set(rows) & (set(a.s1_ids) | set(a.s2_ids)),
            'Checkpoint selection requires only the 2,262 remaining-validation sources')
    tasks = list(jsonl(prepared/'validation_tasks.jsonl'))
    coverage = {(t['condition'], t['source_image_id']) for t in tasks}
    require(len(tasks) == len(coverage) == 2*VAL_SOURCES and
            coverage == {(condition, source) for condition in ('clean', 'lr2') for source in rows},
            'Invalid remaining-validation task coverage')
    protocol = read_json(a.evaluation_protocol)
    evaluation = copy.copy(a)
    evaluation.checkpoint = Path(checkpoint)
    detector, pp, _, _ = detector_for(evaluation)
    totals = {'clean': [0, 0, 0, 0], 'lr2': [0, 0, 0, 0]}
    for i, task in enumerate(tasks, 1):
        condition = task['condition']
        gt = rows[task['source_image_id']]
        _, _, _, _, _, prediction, _ = forward(detector, pp, task, gt)
        counts = evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol)['counts']
        totals[condition] = [current+value for current, value in zip(totals[condition], counts)]
        if i % 500 == 0:
            print(f'B1 remaining validation {updates} updates: {i}/{len(tasks)}', flush=True)
    scores = {}
    for condition, counts in totals.items():
        tp, gt_count, pred_count, _ = counts
        scores[condition] = dict(**rates(counts), tp=tp, fp=pred_count-tp, fn=gt_count-tp)
    result = dict(identity=identity, checkpoint=str(checkpoint), sources=len(rows), scores=scores,
                  balanced_hmean=(scores['clean']['hmean']+scores['lr2']['hmean'])/2)
    save_json(target, result)
    return result


def choose_checkpoint(candidates):
    return max(candidates, key=lambda item: (item['balanced_hmean'], -item['identity']['updates']))


def run(a):
    prepared = validate_prepared(a)
    identity = environment_identity(a, prepared)
    path = a.output/'b1_identity.json'
    if path.is_file():
        require(read_json(path) == identity, 'B1 existing run uses a different identity')
    else:
        save_json(path, identity)
    trained = fit(a, prepared)
    candidates = []
    for updates in RECIPE['validation_candidates']:
        checkpoint = a.output/'weights'/f'iter_{updates*RECIPE["accumulation"]}.pth'
        candidates.append(checkpoint_score(a, prepared, checkpoint, trained['checkpoints'][str(checkpoint)], updates))
    best = choose_checkpoint(candidates)
    result = dict(stage='B1', selected_checkpoint=best, validation_candidates=candidates,
                  training=trained, train_sources=TRAIN_SOURCES, validation_sources=VAL_SOURCES,
                  calibration='pending same-grid B0/B1/M calibration on remaining validation')
    lines = ['# Rev.4 B1 · Native validation', '',
             'Train sources 15,222. Remaining validation sources 2,262. Clean replay 50%.', '',
             '| Optimizer updates | Clean Hmean | LR2 Hmean | Mean Hmean |',
             '|---|---:|---:|---:|']
    for candidate in candidates:
        score = candidate['scores']
        lines.append(f'| {candidate["identity"]["updates"]:,} | {score["clean"]["hmean"]:.2%} | '
                     f'{score["lr2"]["hmean"]:.2%} | {candidate["balanced_hmean"]:.2%} |')
    lines += ['', f'Selected checkpoint `{Path(best["checkpoint"]).name}`.', '',
              'Calibration is pending on remaining validation.', '']
    (a.output/'b1_report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(a.output/'b1_summary.json', result)
    print('B1 complete:', a.output/'b1_report.md', flush=True)
