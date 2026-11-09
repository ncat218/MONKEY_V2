"""Descriptive Blur/JPEG on the locked S2 cohort after LR2 passed."""
from collections import Counter
import copy
import csv
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from lowres.common import jsonl, read_json, save_json, sha256, write_jsonl
from rev4 import require
from rev4.stage_a import (
    BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, detector_for, forward, make_task,
    open_native_lr, rows_for, safe_path, verify_detector_png_decode,
)

CONDITIONS = {
    'blur1': ('blur', '1.88', 'PNG'),
    'blur2': ('blur', '3.75', 'PNG'),
    'jpeg30': ('jpeg', '30', 'JPEG'),
    'jpeg10': ('jpeg', '10', 'JPEG'),
}
BLUR_SIGMAS = ('1.88', '3.75')
JPEG_QUALITIES = ('30', '10')
MANIFEST_COLUMNS = ('source', 'condition', 'parameter', 'width', 'height', 'out_path',
                    'pillow_version', 'source_sha256', 'output_sha256', 'source_exif_orientation')


def condition_for(entry):
    pair = entry['condition'], entry['parameter']
    for condition, (kind, value, image_format) in CONDITIONS.items():
        if pair == (kind, value):
            return condition, image_format
    raise ValueError(f'Unexpected Blur/JPEG severity {pair}')


def validate_generated(rows, source_ids, generated, final, manifest_path):
    from PIL import Image

    files, seen = set(), set()
    tasks, icc_rechecks = [], []
    with manifest_path.open(encoding='utf-8', newline='') as handle:
        manifest = csv.DictReader(handle)
        require(tuple(manifest.fieldnames or ()) == MANIFEST_COLUMNS, 'Blur/JPEG manifest columns changed')
        for entry in manifest:
            source = source_ids[entry['source']]
            condition, image_format = condition_for(entry)
            key = source, condition
            require(key not in seen, f'Duplicate Blur/JPEG variant {key}')
            seen.add(key)
            row = rows[source]
            size = list(map(int, (entry['width'], entry['height'])))
            require(entry['pillow_version'] == '11.3.0' and entry['source_exif_orientation'] == '1' and
                    entry['source_sha256'] == row['clean_sha256'] and size == row['clean_size'],
                    f'Blur/JPEG source identity mismatch {key}')
            path = safe_path(generated, entry['out_path'])
            require(path not in files, f'Duplicate Blur/JPEG output path {key}')
            files.add(path)
            digest = sha256(path)
            require(digest == entry['output_sha256'], f'Blur/JPEG output SHA mismatch {key}')
            image, raw_with_icc = open_native_lr(path) if image_format == 'PNG' else (Image.open(path), None)
            with image:
                require(list(image.size) == size and image.format == image_format and
                        int(image.getexif().get(274, 1)) == 1, f'Blur/JPEG output geometry mismatch {key}')
                if raw_with_icc is not None:
                    verify_detector_png_decode(raw_with_icc, image, path)
                    icc_rechecks.append(dict(source_id=source, condition=condition,
                                             png_sha256=digest, detector_rgb_matches=True))
                    print(f'ICC metadata recheck {condition} {source}: original PNG RGB matches', flush=True)
            tasks.append(make_task(source, condition, final/'blur_jpeg'/path.relative_to(generated), size, digest))
    require(seen == {(source, condition) for source in rows for condition in CONDITIONS},
            'Blur/JPEG manifest must cover every locked S2 source at every severity')
    tasks.sort(key=lambda task: (task['condition'], task['source_image_id']))
    return tasks, icc_rechecks


def prepare(a):
    import PIL
    from PIL import Image

    require(PIL.__version__ == '11.3.0', 'Rev.4 generation requires Pillow 11.3.0')
    require(sha256(a.blur_generator) == a.lock['blur_jpeg_generator_sha256'] and
            sha256(a.source_list) == a.lock['s2_paths_sha256'], 'Frozen generator or source list changed')
    rows = rows_for(a, a.s2_ids)
    source_ids = dict(zip(a.s2_paths, a.s2_ids, strict=True))
    require(len(rows) == len(source_ids) == 500, 'Expected 500 unique S2 paths and sources')
    root = a.clean_root.resolve(strict=True)
    for relative, source in source_ids.items():
        row = rows[source]
        path = safe_path(root, relative)
        require(relative == row['clean_relpath'] and sha256(path) == row['clean_sha256'],
                f'Frozen clean image identity mismatch {source}')
        with Image.open(path) as image:
            require(list(image.size) == row['clean_size'] and int(image.getexif().get(274, 1)) == 1,
                    f'Clean canvas or EXIF mismatch {source}')
    final = a.output/'prepared'
    require(not final.exists(), f'Unexpected incomplete preparation directory {final}')
    with TemporaryDirectory(prefix='.preparing-', dir=a.output) as temp:
        stage = Path(temp)/'prepared'
        stage.mkdir()
        generated = stage/'blur_jpeg'
        subprocess.run([sys.executable, str(a.blur_generator), '--input-dir', str(root),
                        '--source-list', str(a.source_list), '--output-dir', str(generated),
                        '--blur-sigmas', *BLUR_SIGMAS, '--jpeg-qualities', *JPEG_QUALITIES,
                        '--expect-count', '500', '--strict-exif'], check=True)
        manifest_path = generated/'manifest.csv'
        tasks, icc_rechecks = validate_generated(rows, source_ids, generated, final, manifest_path)
        write_jsonl(stage/'tasks.jsonl', tasks)
        save_json(stage/'prepared.json', dict(tasks_sha256=sha256(stage/'tasks.jsonl'),
                  manifest_sha256=sha256(manifest_path), s2_ids_sha256=a.lock['s2_ids_sha256'],
                  generator_sha256=sha256(a.blur_generator), counts=dict(Counter(t['condition'] for t in tasks)),
                  icc_metadata_rechecks=icc_rechecks))
        stage.rename(final)
    print(f'Prepared 2,000 S2-B variants: {final}', flush=True)


def clean_baseline(a):
    from lowres.report import metrics

    directory = a.s2a_run/'results/samples/clean'
    records, hashes = {}, {}
    for source in a.s2_ids:
        path = directory/f'{source}.json'
        record = read_json(path)
        require(record['sample_id'] == f'clean::{source}' and (directory/f'{source}.npz').is_file(),
                f'Missing or inconsistent S2-A clean sample {source}')
        records[source] = record
        hashes[source] = sha256(path)
    aggregate = metrics(records.values())
    expected = a.s2a_summary['conditions']['clean']
    require(all(expected[key] == value for key, value in aggregate.items()),
            'S2-A clean samples do not reconstruct the locked clean summary')
    return records, hashes


def verify_clean_sentinel(a, detector, pp, gt_rows, protocol, clean, result_dir):
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import check_identity

    receipt = result_dir/'s2a_clean_identity.json'
    source = a.s2_ids[0]
    expected = dict(sample_id=f'clean::{source}', baseline_job=a.s2a_job_id, counts_and_matching_equal=True)
    if receipt.is_file():
        require(read_json(receipt) == expected, 'Invalid cached clean identity')
        return
    gt = gt_rows[source]
    path = safe_path(a.clean_root.resolve(strict=True), gt['clean_relpath'])
    task = make_task(source, 'clean', path, gt['clean_size'], gt['clean_sha256'])
    _, _, _, _, _, prediction, _ = forward(detector, pp, task, gt)
    result = evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol)
    check_identity(source, result, clean[source]['metrics'])
    save_json(receipt, expected)
    print(f'Reused 500 S2-A clean samples. Clean identity probe passed on {source}', flush=True)


def infer(a):
    import numpy as np
    import torch
    from mmocr.utils import offset_polygon
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import kernel_evidence, unpack

    prepared = a.output/'prepared'
    prepared_receipt = read_json(prepared/'prepared.json')
    require(prepared_receipt['tasks_sha256'] == sha256(prepared/'tasks.jsonl') and
            prepared_receipt['manifest_sha256'] == sha256(prepared/'blur_jpeg/manifest.csv') and
            prepared_receipt['s2_ids_sha256'] == a.lock['s2_ids_sha256'] and
            prepared_receipt['generator_sha256'] == a.lock['blur_jpeg_generator_sha256'],
            'Prepared S2-B identity mismatch')
    tasks = {condition: {} for condition in CONDITIONS}
    for task in jsonl(prepared/'tasks.jsonl'):
        condition, source = task['condition'], task['source_image_id']
        require(source not in tasks[condition] and task['sample_id'] == f'{condition}::{source}',
                f'Invalid S2-B task {task["sample_id"]}')
        tasks[condition][source] = task
    require(all(set(items) == set(a.s2_ids) for items in tasks.values()), 'S2-B tasks differ from locked S2')
    runtime = {name: importlib.metadata.version(name) for name in a.s1_identity['runtime']}
    require(runtime == a.s1_identity['runtime'] and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == a.s1_identity['gpu'], 'Require the audited H200 MIG runtime')
    clean, hashes = clean_baseline(a)
    result_dir = a.output/'results'
    identity = dict(stage_a_summary_sha256=a.s2a_summary_sha256, stage_a_identity_sha256=a.s2a_identity_sha256,
                    baseline_clean_sha256=hashes, prepared_sha256=sha256(prepared/'prepared.json'),
                    lock_sha256=sha256(a.lock_path), worker_sha256=sha256(Path(__file__)),
                    launcher_sha256=sha256(a.s2b_launcher), runtime=runtime, gpu=torch.cuda.get_device_name(0))
    identity_path = result_dir/'inference_identity.json'
    if identity_path.exists():
        require(read_json(identity_path) == identity, 'Existing S2-B results use another identity')
    save_json(identity_path, identity)
    rows = rows_for(a, a.s2_ids)
    protocol = read_json(a.evaluation_protocol)
    detector, pp, candidate, shrink = detector_for(a)
    verify_clean_sentinel(a, detector, pp, rows, protocol, clean, result_dir)
    results = {'clean': clean}
    torch.cuda.reset_peak_memory_stats()
    for condition in CONDITIONS:
        results[condition] = {}
        folder = result_dir/'samples'/condition
        folder.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(a.s2_ids, 1):
            task, gt = tasks[condition][source], rows[source]
            require(task['size'] == gt['clean_size'], f'Expected native clean canvas for {task["sample_id"]}')
            receipt, pmap = folder/f'{source}.json', folder/f'{source}.npz'
            if receipt.is_file() and pmap.is_file():
                record = read_json(receipt)
                require(record['sample_id'] == task['sample_id'], f'Unexpected cached S2-B sample {source}')
            else:
                _, p, sample, factor, restore, prediction, latency = forward(detector, pp, task, gt)
                candidates = unpack(candidate(p, [copy.deepcopy(sample)])[0], restore)
                scored = evaluate_image(gt['gt_clean'], gt['roi_clean'], prediction, protocol)
                probability = p[0].detach().cpu().numpy()
                meta = sample.metainfo
                img_shape = tuple(meta['img_shape'][:2])
                evidence, valid, above = kernel_evidence(probability, img_shape, factor, task, gt,
                                                         task['size'], shrink, offset_polygon)
                temp = pmap.with_suffix('.npz.tmp')
                with temp.open('wb') as sink:
                    np.savez_compressed(sink, p=probability.astype(np.float16),
                                        valid=np.packbits(valid), above_mask_thr=np.packbits(above))
                temp.replace(pmap)
                record = dict(sample_id=task['sample_id'], metrics=scored, prediction=prediction,
                              candidates_before_score=candidates, gt=evidence, latency_ms=latency,
                              resize=dict(ori_shape=meta['ori_shape'][:2], img_shape=img_shape,
                                          pmap_shape=list(probability.shape), scale_factor=factor.tolist(),
                                          lr_to_clean=restore.tolist()))
                save_json(receipt, record)
            results[condition][source] = record
            if index % 50 == 0:
                print(f'S2-B {condition} {index}/500', flush=True)
    summary = summarize(results, rows, protocol)
    summary.update(s2_ids_sha256=a.lock['s2_ids_sha256'], baseline_job=a.s2a_job_id,
                   baseline_summary_sha256=a.s2a_summary_sha256,
                   peak_allocated_gpu_mib_this_launch=torch.cuda.max_memory_allocated()/1024**2)
    save_json(result_dir/'review_examples.json', review_examples(summary, tasks, rows, a))
    (result_dir/'s2b_report.md').write_text(markdown(summary), encoding='utf-8')
    save_json(result_dir/'s2b_summary.json', summary)
    print('TASK 2B complete:', result_dir/'s2b_report.md', flush=True)


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
        drop = 100*(baseline['hmean']-item['hmean'])
        ci = interval(100*(boot['clean']-boot[name]))
        item.update(hmean_drop_pp=drop, hmean_drop_95ci_pp=ci,
                    recall_drop_pp=100*(baseline['recall']-item['recall']),
                    latency_median_ms=float(np.median([row['latency_ms'] for row in by_source.values()])))
        if name != 'clean':
            item.update(diagnostic(by_source, clean, rows, sources, protocol))
            failures = item['failures']
            item['primary_failure_counts'] = dict(Counter(x['primary'] for x in failures))
            item['sensitivity_failure_counts'] = dict(Counter(x['sensitivity'] for x in failures))
            lost = len(failures)/item['clean_tp'] if item['clean_tp'] else None
            item['clean_tp_to_fn_fraction'] = lost
            item['gates'] = dict(G1=bool(drop >= 5 and ci[0] > 0),
                                 G2=bool(lost is not None and lost >= .15), G3='NOT ASSESSED')
            item['tau'] = ('N/A (stress-only)' if name in ('blur2', 'jpeg10') else
                           'N/A (LR2 selected by registered priority)')
            item['cheap_recovery_control'] = 'N/A'
        conditions[name] = item
    return dict(stage='S2-B', mode='descriptive', selected_condition='lr2', source_count=len(sources),
                bootstrap_repetitions=BOOTSTRAP_REPETITIONS, bootstrap_seed=BOOTSTRAP_SEED,
                conditions=conditions, g3='NOT ASSESSED', tau='N/A', clean_baseline='reused from S2-A')


def review_examples(summary, tasks, rows, a):
    examples = []
    clean_root = a.clean_root.resolve(strict=True)
    for condition in CONDITIONS:
        seen = set()
        for item in summary['conditions'][condition]['failures']:
            category = item['primary']
            if category in seen:
                continue
            seen.add(category)
            source = item['source_image_id']
            gt = next(ann['geometry'] for ann in rows[source]['gt_clean']['valid']
                      if str(ann['ann_id']) == item['ann_id'])
            clean = safe_path(clean_root, rows[source]['clean_relpath'])
            examples.append(dict(condition=condition, **item, clean_gt=gt, clean_image=str(clean),
                                 condition_image=tasks[condition][source]['path'],
                                 clean_pmap=str(a.s2a_run/'results/samples/clean'/f'{source}.npz'),
                                 condition_pmap=str(a.output/'results/samples'/condition/f'{source}.npz')))
    return dict(selection='first source-ordered lost word per condition and primary category', examples=examples)


def markdown(summary):
    lines = ['# CS431 Rev.4 · S2-B descriptive', '',
             f'Sources {summary["source_count"]}. B0 FP32. Clean baseline reused from S2-A job '
             f'{summary["baseline_job"]}. Paired bootstrap {summary["bootstrap_repetitions"]} '
             f'replicates, seed {summary["bootstrap_seed"]}.', '',
             '| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['conditions'].items():
        survival = row.get('clean_tp_survival')
        label = 'N/A' if survival is None else f'{survival:.2%}'
        lines.append(f'| {name} | {row["precision"]:.2%} | {row["recall"]:.2%} | {row["hmean"]:.2%} | '
                     f'{row["tp"]} | {row["fp"]} | {row["fn"]} | {label} |')
    lines += ['', '| Condition | Hmean drop pp (95% CI) | Clean TP → FN | G1 | G2 | G3 |',
              '|---|---:|---:|---|---|---|']
    for name in CONDITIONS:
        row = summary['conditions'][name]
        lo, hi = row['hmean_drop_95ci_pp']
        fraction = row['clean_tp_to_fn_fraction']
        lost = 'N/A' if fraction is None else f'{fraction:.2%}'
        g1, g2 = ('PASS' if row['gates'][gate] else 'FAIL' for gate in ('G1', 'G2'))
        lines.append(f'| {name} | {row["hmean_drop_pp"]:.2f} [{lo:.2f}, {hi:.2f}] | '
                     f'{row["clean_tp_to_fn"]}/{row["clean_tp"]} ({lost}) | {g1} | {g2} | NOT ASSESSED |')
    lines += ['', 'LR2 remains selected by the registered S2-A priority. τ = N/A.', '',
              'B_unsharp and JPEG cheap recovery are N/A. Size bins, P-map diagnostics and '
              'failure taxonomy are in s2b_summary.json. See review_examples.json for image and P-map paths.', '']
    return '\n'.join(lines)
