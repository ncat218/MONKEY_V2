"""Rev.4 S2-A worker. This module is called only by the gated Slurm launcher."""
from collections import Counter
import copy
import csv
import importlib.metadata
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zlib import crc32

from lowres.common import jsonl, read_json, save_json, sha256, write_jsonl
from rev4 import CONDITIONS, require

BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261107
LR_COLUMNS = ('source', 'scale', 'clean_width', 'clean_height',
              'lr_width', 'lr_height', 'lr_path', 'pillow_version')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
PILLOW_CHUNK_LIMIT_ERROR = 'Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK'


def safe_path(root, name):
    rel = Path(name)
    result = (root/rel).resolve()
    require(not rel.is_absolute() and '..' not in rel.parts and result.is_relative_to(root),
            f'Invalid relative image path {name}')
    return result


def rows_for(a, ids):
    requested = set(ids)
    rows = {}
    for row in jsonl(a.validation_manifest):
        source = row['source_image_id']
        if source in requested:
            require(source not in rows, f'Duplicate validation source {source}')
            rows[source] = row
    require(rows.keys() == requested, 'Missing frozen validation source')
    return rows


def make_task(source, condition, path, size, digest):
    return dict(sample_id=f'{condition}::{source}', source_image_id=source,
                condition=condition, path=str(path), size=list(size), image_sha256=digest)


def png_without_icc(raw, path):
    """Build a temporary view that omits only the compressed ICC metadata chunk."""
    require(raw.startswith(PNG_SIGNATURE), f'Invalid PNG signature {path}')
    offset = len(PNG_SIGNATURE)
    while offset + 12 <= len(raw):
        length = int.from_bytes(raw[offset:offset+4], 'big')
        kind = raw[offset+4:offset+8]
        end = offset+length+12
        require(end <= len(raw), f'Truncated PNG chunk {path}')
        if kind == b'iCCP':
            require(crc32(raw[offset+4:end-4]) == int.from_bytes(raw[end-4:end], 'big'),
                    f'Invalid ICC chunk CRC {path}')
            return raw[:offset]+raw[end:]
        if kind in (b'IDAT', b'IEND'):
            break
        offset = end
    raise ValueError(f'Pillow rejected PNG metadata without a preceding ICC chunk: {path}')


def open_native_lr(path):
    from PIL import Image

    try:
        return Image.open(path), None
    except ValueError as exc:
        if str(exc) != PILLOW_CHUNK_LIMIT_ERROR:
            raise
    raw = path.read_bytes()
    return Image.open(BytesIO(png_without_icc(raw, path))), raw


def verify_detector_png_decode(raw, image, path):
    """Check that the original PNG is still valid for the frozen detector loader."""
    import cv2
    import numpy as np

    detector_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    require(detector_bgr is not None and np.array_equal(detector_bgr[:, :, ::-1], np.asarray(image.convert('RGB'))),
            f'Frozen OpenCV decode does not match the temporary Pillow RGB view: {path}')


def prepare(a):
    import PIL
    from PIL import Image

    require(PIL.__version__ == '11.3.0', 'Rev.4 image generation requires Pillow 11.3.0')
    clean_root = a.clean_root.resolve(strict=True)
    rows = rows_for(a, a.s2_ids)
    mapping = dict(zip(a.s2_paths, a.s2_ids, strict=True))
    require(len(mapping) == 500, 'Expected unique paths for 500 sources')
    require(len({row['clean_sha256'] for row in rows.values()}) == 500, 'Expected unique clean hashes')
    final = a.output/'prepared'
    require(not final.exists(), f'Unexpected incomplete preparation directory {final}')
    with TemporaryDirectory(prefix='.preparing-', dir=a.output) as temp:
        stage = Path(temp)/'prepared'
        stage.mkdir()
        clean_stage = Path(temp)/'clean'
        clean_stage.mkdir()
        tasks = []
        for relative, source in mapping.items():
            row = rows[source]
            require(relative == row['clean_relpath'], f'Frozen path mapping mismatch {source}')
            clean = safe_path(clean_root, relative)
            require(sha256(clean) == row['clean_sha256'], f'Clean image SHA mismatch {source}')
            with Image.open(clean) as image:
                require(list(image.size) == row['clean_size'] and int(image.getexif().get(274, 1)) == 1,
                        f'Clean geometry or EXIF mismatch {source}')
            staged = safe_path(clean_stage, relative)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(clean, staged)
            require(sha256(staged) == row['clean_sha256'], f'Staged clean SHA mismatch {source}')
            tasks.append(make_task(source, 'clean', clean, row['clean_size'], row['clean_sha256']))
        generated = stage/'native_lr'
        subprocess.run([sys.executable, str(a.lr_generator), '--input-dir', str(clean_stage),
                        '--output-dir', str(generated)], check=True)
        manifest_path = generated/'manifest.csv'
        seen, files = set(), set()
        icc_rechecks = []
        with manifest_path.open(encoding='utf-8', newline='') as handle:
            manifest = csv.DictReader(handle)
            require(tuple(manifest.fieldnames or ()) == LR_COLUMNS, 'LR manifest columns changed')
            for entry in manifest:
                name, factor = entry['source'], int(entry['scale'])
                source = mapping[name]
                key = source, factor
                require(factor in (2, 4, 8) and key not in seen, f'Unexpected LR variant {key}')
                seen.add(key)
                clean_w, clean_h = rows[source]['clean_size']
                size = clean_w//factor, clean_h//factor
                require(entry['pillow_version'] == '11.3.0' and
                        (int(entry['clean_width']), int(entry['clean_height'])) == (clean_w, clean_h) and
                        (int(entry['lr_width']), int(entry['lr_height'])) == size, f'LR manifest size mismatch {key}')
                image_path = safe_path(generated, entry['lr_path'])
                require(image_path not in files, f'Duplicate LR image path {key}')
                files.add(image_path)
                opened, original_with_icc = open_native_lr(image_path)
                png_digest = sha256(image_path)
                with opened as image:
                    require(image.size == size and image.format == 'PNG' and
                            int(image.getexif().get(274, 1)) == 1, f'LR image geometry mismatch {key}')
                    if original_with_icc is not None:
                        verify_detector_png_decode(original_with_icc, image, image_path)
                        icc_rechecks.append(dict(source_id=source, scale=factor, png_sha256=png_digest,
                                                 detector_rgb_matches=True))
                        print(f'ICC metadata recheck LR{factor} {source}: original PNG RGB matches', flush=True)
                    if factor == 2:
                        up = stage/'b_up_lr2'/Path(name).parent/f'{Path(name).name}.png'
                        up.parent.mkdir(parents=True, exist_ok=True)
                        image.convert('RGB').resize((clean_w, clean_h), Image.Resampling.BICUBIC).save(up, format='PNG')
                        tasks.append(make_task(source, 'b_up_lr2', final/up.relative_to(stage),
                                               (clean_w, clean_h), sha256(up)))
                tasks.append(make_task(source, f'lr{factor}', final/image_path.relative_to(stage),
                                       size, png_digest))
        require(seen == {(source, scale) for source in a.s2_ids for scale in (2, 4, 8)},
                'Expected exactly 1,500 LR variants')
        require(len(tasks) == len({t['sample_id'] for t in tasks}) == 2500, 'Expected 2,500 tasks')
        tasks.sort(key=lambda row: (CONDITIONS.index(row['condition']), row['source_image_id']))
        write_jsonl(stage/'tasks.jsonl', tasks)
        save_json(stage/'prepared.json', dict(tasks_sha256=sha256(stage/'tasks.jsonl'),
                  lr_manifest_sha256=sha256(manifest_path), s2_ids_sha256=a.lock['s2_ids_sha256'],
                  generator_sha256=sha256(a.lr_generator), counts=dict(Counter(t['condition'] for t in tasks)),
                  icc_metadata_rechecks=icc_rechecks))
        stage.rename(final)
    print(f'Prepared S2-A inputs: {final}', flush=True)


def detector_for(a):
    import torch
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.apis import TextDetInferencer
    from mmocr.utils import register_all_modules

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one CUDA device')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    cfg = Config.fromfile(str(a.model_config))
    require(cfg.get('textdet_parameter_precision') == 'fp32' and not cfg.model.backbone.use_bf16,
            'Require the historical FP32 config')
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    register_all_modules(init_default_scope=True)
    detector = TextDetInferencer(model=str(a.model_config), weights=str(a.checkpoint), device='cuda:0')
    detector.model.float().eval()
    load_checkpoint(detector.model, str(a.checkpoint), map_location='cpu', strict=True)
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in detector.model.parameters()),
            'Model parameters must be FP32')
    pp = detector.model.det_head.postprocessor
    require((pp.mask_thr, pp.min_text_score, pp.min_text_width, pp.unclip_ratio, pp.text_repr_type,
             pp.epsilon_ratio, pp.max_candidates) == (.3, .3, 5, 1.5, 'quad', .01, 3000),
            'Unexpected native DB postprocessor')
    shrink = detector.model.det_head.module_loss.shrink_ratio
    require(shrink == .4, 'Unexpected DBNet shrink ratio')
    candidate = copy.copy(pp)
    candidate.min_text_score = 0.0
    return detector, pp, candidate, shrink


def forward(detector, pp, task, gt):
    import numpy as np
    import torch
    from lowres.inference import unpack

    require(sha256(task['path']) == task['image_sha256'], f'Image SHA mismatch {task["sample_id"]}')
    start = time.perf_counter()
    _, inputs = next(iter(detector.preprocess([task['path']], batch_size=1)))
    with torch.inference_mode():
        processed = detector.model.data_preprocessor(inputs, training=False)
        probability = detector.model(**processed, mode='tensor')
    require(probability.dtype == torch.float32 and probability.ndim == 3 and probability.shape[0] == 1 and
            probability.shape[-2:] == processed['inputs'].shape[-2:] and torch.isfinite(probability).all(),
            'Expected finite FP32 P-map at the padded input grid')
    sample = processed['data_samples'][0]
    meta = sample.metainfo
    factor = np.asarray(meta['scale_factor'], dtype=np.float64)
    img_shape = tuple(meta['img_shape'][:2])
    require(tuple(meta['ori_shape'][:2]) == tuple(reversed(task['size'])) and factor.shape == (2,) and
            all(actual <= padded for actual, padded in zip(img_shape, probability.shape[-2:])),
            'Unexpected resize metadata')
    restore = np.asarray(gt['clean_size'])/np.asarray(task['size'])
    prediction = unpack(pp(probability, [copy.deepcopy(sample)])[0], restore)
    return processed, probability, sample, factor, restore, prediction, (time.perf_counter()-start)*1000


def verify_s1(a, detector, pp, rows, protocol, history, result_dir):
    import torch
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import check_identity, unpack

    prior = result_dir/'s1_clean_identity.json'
    probe = result_dir/'pmap_equivalence.json'
    if prior.is_file() and probe.is_file():
        require(read_json(prior) == dict(sources=500, counts_and_matching_equal=True) and
                read_json(probe)['exact_polygons_and_scores_equal'], 'Invalid cached S1 identity')
        return
    clean_root = a.clean_root.resolve(strict=True)
    for index, source in enumerate(a.s1_ids, 1):
        gt = rows[source]
        task = make_task(source, 'clean', safe_path(clean_root, gt['clean_relpath']),
                         gt['clean_size'], gt['clean_sha256'])
        processed, probability, sample, _, restore, prediction, _ = forward(detector, pp, task, gt)
        check_identity(source, evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol), history[source])
        if index == 1:
            with torch.inference_mode():
                native = detector.model.predict(processed['inputs'], [copy.deepcopy(sample)])
            require(unpack(native[0], restore) == prediction, 'P-map decode differs from native prediction')
            save_json(probe, dict(sample_id=task['sample_id'], cohort='S1',
                                  exact_polygons_and_scores_equal=True, shape=list(probability.shape)))
        if index % 100 == 0:
            print(f'S1 clean identity {index}/500', flush=True)
    save_json(prior, dict(sources=500, counts_and_matching_equal=True))


def infer(a):
    import numpy as np
    import torch
    from mmocr.utils import offset_polygon
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import kernel_evidence, unpack

    prepared = a.output/'prepared'
    receipt = read_json(prepared/'prepared.json')
    require(receipt['tasks_sha256'] == sha256(prepared/'tasks.jsonl') and
            receipt['lr_manifest_sha256'] == sha256(prepared/'native_lr/manifest.csv') and
            receipt['s2_ids_sha256'] == a.lock['s2_ids_sha256'] and
            receipt['generator_sha256'] == a.lock['lr_generator_sha256'], 'Prepared identity mismatch')
    tasks = {condition: {} for condition in CONDITIONS}
    for task in jsonl(prepared/'tasks.jsonl'):
        condition, source = task['condition'], task['source_image_id']
        require(source not in tasks[condition] and task['sample_id'] == f'{condition}::{source}',
                f'Invalid task {task["sample_id"]}')
        tasks[condition][source] = task
    require(all(set(items) == set(a.s2_ids) for items in tasks.values()), 'Tasks must cover only locked S2')
    history_path = a.s1/'frozen_v2_0/e0_clean_matched_counts.jsonl'
    history = {row['source_image_id']: row for row in jsonl(history_path)}
    require(set(history) == set(a.s1_ids) and len(history) == 500, 'Require all 500 historical S1 references')
    runtime = {name: importlib.metadata.version(name) for name in a.s1_identity['runtime']}
    require(runtime == a.s1_identity['runtime'] and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == a.s1_identity['gpu'], 'Require the audited H200 MIG runtime')
    result_dir = a.output/'results'
    identity = dict(s1_history_sha256=sha256(history_path), prepared_sha256=sha256(prepared/'prepared.json'),
                    lock_sha256=sha256(a.lock_path), worker_sha256=sha256(Path(__file__)),
                    launcher_sha256=sha256(a.launcher_path), runtime=runtime, gpu=torch.cuda.get_device_name(0))
    identity_path = result_dir/'inference_identity.json'
    if identity_path.exists():
        require(read_json(identity_path) == identity, 'Existing inference output uses another identity')
    save_json(identity_path, identity)
    rows = rows_for(a, a.s1_ids+a.s2_ids)
    protocol = read_json(a.evaluation_protocol)
    detector, pp, candidate, shrink = detector_for(a)
    verify_s1(a, detector, pp, rows, protocol, history, result_dir)
    results = {condition: {} for condition in CONDITIONS}
    torch.cuda.reset_peak_memory_stats()
    for condition in CONDITIONS:
        folder = result_dir/'samples'/condition
        folder.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(a.s2_ids, 1):
            task = tasks[condition][source]
            gt = rows[source]
            receipt_file, map_file = folder/f'{source}.json', folder/f'{source}.npz'
            if receipt_file.exists() and map_file.exists():
                record = read_json(receipt_file)
                require(record['sample_id'] == task['sample_id'], f'Invalid cached S2 sample {source}')
            else:
                _, p, sample, factor, restore, prediction, latency = forward(detector, pp, task, gt)
                candidates = unpack(candidate(p, [copy.deepcopy(sample)])[0], restore)
                scored = evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol)
                probability = p[0].detach().cpu().numpy()
                meta = sample.metainfo
                img_shape = tuple(meta['img_shape'][:2])
                native_size = tasks['lr2'][source]['size'] if condition == 'b_up_lr2' else task['size']
                gt_evidence, valid, above = kernel_evidence(probability, img_shape, factor, task, gt,
                                                            native_size, shrink, offset_polygon)
                temp = map_file.with_suffix('.npz.tmp')
                with temp.open('wb') as sink:
                    np.savez_compressed(sink, p=probability.astype(np.float16),
                                        valid=np.packbits(valid), above_mask_thr=np.packbits(above))
                temp.replace(map_file)
                record = dict(sample_id=task['sample_id'], metrics=scored, prediction=prediction,
                              candidates_before_score=candidates, gt=gt_evidence, latency_ms=latency,
                              resize=dict(ori_shape=meta['ori_shape'][:2], img_shape=img_shape,
                                          pmap_shape=list(probability.shape), scale_factor=factor.tolist(),
                                          lr_to_clean=restore.tolist()))
                save_json(receipt_file, record)
            results[condition][source] = record
            if index % 50 == 0:
                print(f'S2-A {condition} {index}/500', flush=True)
    summary = summarize(results, {source: rows[source] for source in a.s2_ids}, protocol)
    summary['s2_ids_sha256'] = a.lock['s2_ids_sha256']
    summary['peak_allocated_gpu_mib_this_launch'] = torch.cuda.max_memory_allocated()/1024**2
    (result_dir/'s2a_report.md').write_text(markdown(summary), encoding='utf-8')
    save_json(result_dir/'s2a_summary.json', summary)
    print('TASK 2A complete:', result_dir/'s2a_report.md', flush=True)


def summarize(results, rows, protocol):
    import numpy as np
    from lowres.report import bootstrap, diagnostic, interval, metrics

    sources = sorted(rows)
    boot = bootstrap(results, sources, repetitions=BOOTSTRAP_REPETITIONS)
    clean = results['clean']
    baseline = metrics(clean.values())
    conditions = {}
    for name, by_source in results.items():
        item = metrics(by_source.values())
        item.update(hmean_drop_pp=100*(baseline['hmean']-item['hmean']),
                    hmean_drop_95ci_pp=interval(100*(boot['clean']-boot[name])),
                    recall_drop_pp=100*(baseline['recall']-item['recall']),
                    latency_median_ms=float(np.median([row['latency_ms'] for row in by_source.values()])))
        if name != 'clean':
            item.update(diagnostic(by_source, clean, rows, sources, protocol))
            item['primary_failure_counts'] = dict(Counter(x['primary'] for x in item['failures']))
            item['sensitivity_failure_counts'] = dict(Counter(x['sensitivity'] for x in item['failures']))
        conditions[name] = item
    lr2, up = conditions['lr2'], conditions['b_up_lr2']
    lost = lr2['failures']
    lost_fraction = len(lost)/lr2['clean_tp'] if lr2['clean_tp'] else None
    heights = {str(h): sum(x['native_height'] >= h for x in lost)/len(lost) if lost else None for h in (4, 6, 8)}
    low, high = lr2['hmean_drop_95ci_pp']
    recovery = None
    if lr2['hmean_drop_pp'] and not low <= 0 <= high:
        recovery = (up['hmean']-lr2['hmean'])/(baseline['hmean']-lr2['hmean'])
    gates = dict(G1=bool(lr2['hmean_drop_pp'] >= 5 and low > 0),
                 G2=bool(lost_fraction is not None and lost_fraction >= .15),
                 G3=bool(heights['6'] is not None and heights['6'] >= .30))
    return dict(stage='S2-A', source_count=len(sources), bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
                bootstrap_seed=BOOTSTRAP_SEED, conditions=conditions, gates=gates,
                lr2_clean_tp_to_fn_fraction=lost_fraction, lr2_lost_fraction_at_min_native_height=heights,
                b_up_lr2_gain_pp=100*(up['hmean']-lr2['hmean']),
                b_up_lr2_gain_95ci_pp=interval(100*(boot['b_up_lr2']-boot['lr2'])),
                b_up_lr2_recovery_ratio=recovery, g4='qualitative review of B_up, maps and LR2 transitions',
                next_branch='LR2 selected. Blur/JPEG remains descriptive' if all(gates.values()) else
                            'LR2 failed at least one gate. Blind calibration on TRAIN precedes Blur/JPEG S2 results')


def markdown(summary):
    lines = ['# CS431 Rev.4 · S2-A', '',
             f'Validation sources {summary["source_count"]}. B0 FP32. Paired bootstrap '
             f'{summary["bootstrap_repetitions"]} replicates, seed {summary["bootstrap_seed"]}.', '',
             '| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['conditions'].items():
        survival = row.get('clean_tp_survival')
        label = 'N/A' if survival is None else f'{survival:.2%}'
        lines.append(f'| {name} | {row["precision"]:.2%} | {row["recall"]:.2%} | {row["hmean"]:.2%} | '
                     f'{row["tp"]} | {row["fp"]} | {row["fn"]} | {label} |')
    lr2 = summary['conditions']['lr2']
    low, high = lr2['hmean_drop_95ci_pp']
    up_low, up_high = summary['b_up_lr2_gain_95ci_pp']
    lost_fraction = summary['lr2_clean_tp_to_fn_fraction']
    height_fraction = summary['lr2_lost_fraction_at_min_native_height']['6']
    lost_label = 'N/A' if lost_fraction is None else f'{lost_fraction:.2%}'
    height_label = 'N/A' if height_fraction is None else f'{height_fraction:.2%}'
    lines += ['', f'LR2 Hmean drop {lr2["hmean_drop_pp"]:.2f} pp, 95% CI [{low:.2f}, {high:.2f}].',
              f'LR2 clean TP → FN {lr2["clean_tp_to_fn"]}/{lr2["clean_tp"]} ({lost_label}).',
              f'LR2 lost GT with native height at least 6 px {height_label}.',
              f'B_up LR2 gain {summary["b_up_lr2_gain_pp"]:.2f} pp, '
              f'95% CI [{up_low:.2f}, {up_high:.2f}].', '']
    for name, passed in summary['gates'].items():
        lines.append(f'- {name} {"PASS" if passed else "FAIL"}')
    lines += ['', f'G4 {summary["g4"]}.', '', summary['next_branch']+'.', '']
    return '\n'.join(lines)
