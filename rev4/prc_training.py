"""Train M with the locked B1 recipe and one PRC loss."""
from pathlib import Path
import time

from lowres.common import read_json, save_json, sha256
from rev4 import b1_training, calibration, require
from rev4.stage_a import detector_for
from rev4.training_data import RECIPE


PRC = dict(weight=1.0, loss='masked probability MSE', teacher='frozen B0',
           teacher_match='native DBNet matching augmented TRAIN GT',
           alignment='paired clean warped to LR2 augmentation grid')


def configuration(a, prepared, work, resume):
    cfg = b1_training.configuration(a, prepared, work, resume)
    cfg.custom_imports['imports'].append('rev4.prc_plugins')
    pipeline = cfg.train_dataloader.dataset.pipeline
    require(len(pipeline) == 8 and pipeline[6]['type'] == 'Pad' and tuple(pipeline[6]['size']) == (640, 640),
            'Require the locked B1 padding and pipeline')
    replacements = {3: ('ImgAugWrapper', 'Rev4PRCImgAug'), 4: ('RandomCrop', 'Rev4PRCRandomCrop'),
                    5: ('Resize', 'Rev4PRCResize'), 7: ('PackTextDetInputs', 'Rev4PRCPack')}
    for index, (before, after) in replacements.items():
        require(pipeline[index]['type'] == before, f'Unexpected B1 transform at {index}')
        pipeline[index]['type'] = after
    require(cfg.model.det_head.type == 'DBHead', 'Require the B1 DBNet head')
    cfg.model.det_head.type = 'Rev4PRCDBHead'
    cfg.custom_hooks = [dict(type='Rev4PRCAuditHook')]
    return cfg


def fit(a, prepared):
    import torch
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from rev4.prc_plugins import Rev4PRCAuditHook

    completed = a.output/'trained.json'
    if completed.is_file():
        result = read_json(completed)
        for path, digest in result['checkpoints'].items():
            require(sha256(path) == digest, f'Completed M checkpoint changed: {path}')
        return result
    work = a.output/'weights'
    work.mkdir(parents=True, exist_ok=True)
    last = work/'last_checkpoint'
    resume = Path(last.read_text(encoding='utf-8').strip()).resolve(strict=True) if last.is_file() else None
    if resume is not None:
        require(resume.is_relative_to(work.resolve()), 'M resume checkpoint outside the run')
    cfg = configuration(a, prepared, work, resume)
    teacher, _, _, _ = detector_for(a)
    import_modules_from_strings(**cfg.custom_imports)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.model.float()
    load_checkpoint(runner.model, str(a.checkpoint), map_location='cpu', strict=True)
    runner.model.det_head.bind_teacher(teacher.model, read_json(a.evaluation_protocol), PRC['weight'])
    runner.train()
    audit = next(hook for hook in runner.hooks if isinstance(hook, Rev4PRCAuditHook))
    require(runner.iter == RECIPE['optimizer_updates']*RECIPE['accumulation'] and audit.active_microbatches > 0,
            'M training incomplete or PRC received no matched TRAIN kernels')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in runner.model.parameters()),
            'M parameters must be FP32')
    paths = [work/f'iter_{updates*RECIPE["accumulation"]}.pth' for updates in RECIPE['validation_candidates']]
    result = dict(checkpoints={str(path): sha256(path) for path in paths}, optimizer_updates=RECIPE['optimizer_updates'],
                  micro_iterations=runner.iter, active_microbatches=audit.active_microbatches,
                  peak_allocated_gpu_mib=audit.peak_mib, train_seconds_this_launch=time.perf_counter()-start,
                  resumed_from=str(resume) if resume else None)
    save_json(completed, result)
    return result


def run(a):
    import torch

    prepared = a.calibration_prepared
    rows, tasks = calibration.validation_cohort(a, prepared)
    runtime = b1_training.environment_identity(a, prepared)
    require(runtime == a.b1_identity, 'Require the audited B1 identity')
    identity = dict(b1_identity_sha256=a.b1_identity_sha256, calibration_sha256=a.calibration_sha256,
                    diagnostics_sha256=a.diagnostics_sha256, g4_sha256=a.g4_sha256, recipe=PRC,
                    code={name: sha256(Path(__file__).with_name(name)) for name in
                          ('prc_training.py', 'prc_geometry.py', 'prc_plugins.py')})
    identity_file = a.output/'identity.json'
    if identity_file.is_file():
        require(read_json(identity_file) == identity, 'M run uses a different identity')
    else:
        save_json(identity_file, identity)
    trained = fit(a, prepared)
    torch.cuda.empty_cache()
    candidates = []
    for updates in RECIPE['validation_candidates']:
        checkpoint = a.output/'weights'/f'iter_{updates*RECIPE["accumulation"]}.pth'
        candidates.append(b1_training.checkpoint_score(a, prepared, checkpoint, trained['checkpoints'][str(checkpoint)],
                                                       updates))
    best = b1_training.choose_checkpoint(candidates)
    cache = a.output/'validation/calibration_counts.json'
    grid_identity = dict(run=identity, checkpoint=best['identity'], grid=[list(pair) for pair in calibration.GRID])
    record = read_json(cache) if cache.is_file() else dict(identity=grid_identity, counts=[])
    require(record['identity'] == grid_identity, 'M calibration cache identity mismatch')
    if len(record['counts']) < len(tasks):
        calibration.evaluate_model(a, Path(best['checkpoint']), rows, tasks, read_json(a.evaluation_protocol),
                                   cache, record)
    grid = calibration.summarize_grid(record['counts'], tasks)
    native = grid[calibration.GRID.index(calibration.NATIVE)]
    require(native['scores'] == best['scores'], 'M grid did not reproduce selected native checkpoint')
    save_json(cache, record)
    selected = calibration.choose_threshold(grid)
    baseline = a.calibration_summary['models']['B1']['calibrated']
    delta = {c: 100*(selected['scores'][c]['hmean']-baseline['scores'][c]['hmean']) for c in calibration.CONDITIONS}
    summary = dict(stage='M PRC', identity=identity, training=trained, validation_sources=len(rows),
                   selected_checkpoint=best, validation_candidates=candidates, native=native, calibrated=selected,
                   grid=grid, b1_calibrated=baseline, m_minus_b1_calibrated_pp=delta)
    lines = ['# Rev.4 M PRC validation', '',
             f'{len(rows):,} remaining-validation sources. Selected `{Path(best["checkpoint"]).name}` '
             f'after {best["identity"]["updates"]:,} optimizer updates.', '',
             '| Model | Mask | Score | Clean Hmean | LR2 Hmean |', '|---|---:|---:|---:|---:|']
    for label, item in (('B1-cal', baseline), ('M native', native), ('M-cal', selected)):
        lines.append(f'| {label} | {item["mask_thr"]:.2f} | {item["min_text_score"]:.2f} | '
                     f'{item["scores"]["clean"]["hmean"]:.2%} | {item["scores"]["lr2"]["hmean"]:.2%} |')
    lines += ['', f'On validation, M-cal differs from B1-cal by {delta["clean"]:+.2f} pp on Clean '
              f'and {delta["lr2"]:+.2f} pp on LR2.',
              f'PRC active in {trained["active_microbatches"]:,}/{trained["micro_iterations"]:,} microbatches.', '']
    (a.output/'prc_report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(a.output/'prc_summary.json', summary)
    print('M PRC complete:', a.output/'prc_report.md', flush=True)
