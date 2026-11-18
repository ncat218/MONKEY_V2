"""A versioned 100% Clean control; historical B1 recipe/files are unchanged."""
from pathlib import Path
import time

from lowres.common import jsonl, read_json, save_json, sha256
from rev4 import b1_training, calibration, require
from rev4.adaptation_jobs import lock_record
from rev4.adaptation_plan import BCLEAN_STAGE
from rev4.training_data import RECIPE, TRAIN_NAME, TRAIN_SOURCES, verify_clean


def clean_groups(annotation):
    """Replace each LR2 draw by the Clean view of that exact source."""
    data = annotation['data_list']
    require(len(data) == 2*TRAIN_SOURCES, 'Unexpected frozen B1 annotation count')
    clean, lr2 = data[:TRAIN_SOURCES], data[TRAIN_SOURCES:]
    for c, low in zip(clean, lr2):
        require(c['condition'] == 'clean' and low['condition'] == 'lr2' and
                c['sample_id'].split('::', 1)[1] == low['sample_id'].split('::', 1)[1] and
                c['img_path'] == low['img_path'], 'B1 source draw groups do not correspond')
    result = []
    for slot in (0, 1):
        result.extend(dict(item, clean_draw_slot=slot, sample_id=f'{item["sample_id"]}::draw{slot}') for item in clean)
    return dict(annotation, data_list=result)


def prepare(a):
    source = a.calibration_prepared/'train_annotations.json'
    target = a.output/'prepared/train_annotations_clean.json'
    receipt = target.parent/'prepared.json'
    if receipt.is_file():
        previous = read_json(receipt)
        require(previous['source_sha256'] == sha256(source) and previous['annotation_sha256'] == sha256(target),
                'B-Clean prepared data changed')
        return target
    expected = {}
    for index, row in enumerate(jsonl(a.calibration_prepared/TRAIN_NAME), 1):
        path = verify_clean(a.clean_root.resolve(strict=True), row)
        require(row['source_image_id'] not in expected, 'Duplicate frozen TRAIN source')
        expected[row['source_image_id']] = str(path)
        if index % 2000 == 0:
            print(f'B-Clean TRAIN SHA256/dimensions/EXIF: {index}/{TRAIN_SOURCES}', flush=True)
    require(len(expected) == TRAIN_SOURCES and not set(expected) & (set(a.s1_ids) | set(a.s2_ids) |
            {r['source_image_id'] for r in jsonl(a.calibration_prepared/'validation_remaining.jsonl')}),
            'B-Clean TRAIN source overlap/count failure')
    original = read_json(source)
    require({item['sample_id'].split('::', 1)[1]: item['img_path']
             for item in original['data_list'][:TRAIN_SOURCES]} == expected,
            'B-Clean annotation paths/IDs differ from verified TRAIN pixels')
    converted = clean_groups(original)
    save_json(target, converted)
    valid_instances = sum(not obj['ignore'] for row in converted['data_list'] for obj in row['instances'])
    save_json(receipt, dict(source_sha256=sha256(source), annotation_sha256=sha256(target),
                            unique_sources=TRAIN_SOURCES, records=len(converted['data_list']),
                            clean_records=2*TRAIN_SOURCES, clean_sha256_dimensions_exif_verified=len(expected),
                            valid_instances_across_draw_groups=valid_instances,
                            source_draws='same balanced_indices stream as B1; LR2 draws replaced by Clean'))
    return target


def configuration(a, annotations):
    cfg = b1_training.configuration(a, a.calibration_prepared, a.output/'weights', None)
    cfg.train_dataloader.dataset.ann_file = str(annotations)
    cfg.train_dataloader.sampler.type = 'CleanControlSampler'
    pipeline = cfg.train_dataloader.dataset.pipeline
    pipeline[-1]['meta_keys'] = tuple(pipeline[-1]['meta_keys'])+('clean_draw_slot',)
    cfg.custom_imports['imports'].append('rev4.bclean_plugins')
    cfg.custom_hooks = [dict(type='CleanControlAuditHook')]
    # B0 initialization, full fine-tuning, original 10k updates and scheduler.
    require(cfg.load_from == str(a.checkpoint), 'B-Clean must initialize from B0, not B1')
    return cfg


def fit(a, annotations, identity):
    import torch
    from mmengine.runner import Runner, load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules
    from rev4.bclean_plugins import CleanControlAuditHook

    receipt = a.output/'trained.json'
    if receipt.is_file():
        record = read_json(receipt)
        require(record['identity'] == identity, 'B-Clean trained identity changed')
        for path, digest in record['checkpoints'].items():
            require(sha256(path) == digest, 'B-Clean completed checkpoint changed')
        return record
    require(not (a.output/'weights/last_checkpoint').exists(), 'Partial B-Clean training exists; stop for budget review')
    cfg = configuration(a, annotations)
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
    runner.train()
    audit = next(h for h in runner.hooks if isinstance(h, CleanControlAuditHook))
    require(runner.iter == 20000 and audit.batches == 20000 and audit.images == 40000,
            'B-Clean did not complete the same updates/view budget as B1')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in runner.model.parameters()),
            'B-Clean parameters must remain FP32')
    candidates = [a.output/'weights'/f'iter_{u*RECIPE["accumulation"]}.pth' for u in RECIPE['validation_candidates']]
    result = dict(identity=identity, optimizer_updates=10000, micro_iterations=20000, clean_image_views=40000,
                  checkpoints={str(p): sha256(p) for p in candidates}, seconds=time.perf_counter()-start,
                  peak_gpu_mib=audit.peak_mib)
    save_json(receipt, result)
    return result


def run(a):
    import torch

    identity = lock_record(a.output/'identity.json', dict(stage=BCLEAN_STAGE, recipe=dict(RECIPE, clean_replay=1.0),
                training_input='100% Clean; same source draw sequence, augmentation and effective batch as B1',
                b0_sha256=a.lock['checkpoint_sha256'], b1_summary_sha256=a.b1_summary_sha256,
                b1_identity_sha256=a.b1_identity_sha256, code=a.code_identity))
    annotations = prepare(a)
    trained = fit(a, annotations, identity)
    torch.cuda.empty_cache()
    candidates = []
    for updates in RECIPE['validation_candidates']:
        checkpoint = a.output/'weights'/f'iter_{updates*RECIPE["accumulation"]}.pth'
        candidates.append(b1_training.checkpoint_score(a, a.calibration_prepared, checkpoint,
                                                       trained['checkpoints'][str(checkpoint)], updates))
    best = b1_training.choose_checkpoint(candidates)
    rows, tasks = calibration.validation_cohort(a, a.calibration_prepared)
    cache = a.output/'validation/calibration_counts.json'
    grid_identity = dict(identity=identity, checkpoint=best['identity'], grid=[list(p) for p in calibration.GRID])
    record = read_json(cache) if cache.is_file() else dict(identity=grid_identity, counts=[])
    require(record['identity'] == grid_identity, 'B-Clean calibration cache changed')
    if len(record['counts']) < len(tasks):
        calibration.evaluate_model(a, Path(best['checkpoint']), rows, tasks, read_json(a.evaluation_protocol), cache, record)
        save_json(cache, record)
    grid = calibration.summarize_grid(record['counts'], tasks)
    native = grid[calibration.GRID.index(calibration.NATIVE)]
    require(native['scores'] == best['scores'], 'B-Clean native validation did not reproduce checkpoint selection')
    selected = calibration.choose_threshold(grid)
    summary = dict(stage=BCLEAN_STAGE, identity=identity, training=trained, selected_checkpoint=best,
                   validation_candidates=candidates, native=native, calibrated=selected, grid=grid,
                   b1_calibrated=a.calibration_summary['models']['B1']['calibrated'],
                   validation_sources=len(rows), no_test=True, training_augmentation='same as B1, including random scale/crop')
    output = a.output/'results'
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# B-Clean control: full fine-tuning on 100% Clean', '',
             'Same B0 initialization and 10,000 optimizer updates as B1; same TRAIN IDs and source draw order.',
             'Remaining-validation only. Clean still receives the original random scale/crop augmentation.', '',
             '| Model | Mask/score | Clean Hmean | LR2 Hmean |', '|---|---|---:|---:|']
    for name, value in [('B0-cal', a.calibration_summary['models']['B0']['calibrated']),
                        ('B-Clean native', native), ('B-Clean calibrated', selected), ('B1-cal', summary['b1_calibrated'])]:
        lines.append(f'| {name} | {value["mask_thr"]:.2f}/{value["min_text_score"]:.2f} | '
                     f'{value["scores"]["clean"]["hmean"]:.2%} | {value["scores"]["lr2"]["hmean"]:.2%} |')
    lines += ['', 'Paired instance transitions and source-bootstrap differences are produced by submit_paired_eda.py --include-bclean.',
              'One training seed. These validation values support model/threshold selection; they are not final test results.', '']
    (output/'bclean_report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(output/'bclean_summary.json', summary)
    print('B-Clean complete:', output/'bclean_report.md', flush=True)
