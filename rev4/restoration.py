"""Frozen B1-cal comparison of LR2, bicubic x2 and EDSR x2."""
from collections import Counter
import copy
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import time

from lowres.common import read_json, save_json, sha256
from rev4 import require
from rev4.b1_diagnostics import THRESHOLDS
from rev4.b1_training import environment_identity
from rev4.calibration import validation_cohort
from rev4.stage_a import detector_for, forward, make_task


BRANCHES = ('lr2', 'bicubic_x2', 'edsr_x2')
WEIGHT_URL = 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr64_x2.pt'
WEIGHT_SHA256 = '2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1'
HEIGHT_BINS = ('<4', '4–<8', '8–<16', '≥16')


def resized_task(source, branch, path, image, expected_size):
    from PIL import Image

    require(image.size == tuple(expected_size) and image.mode == 'RGB', f'Invalid x2 canvas for {source}')
    # Rebuild the image to keep both generated PNG paths free of source metadata.
    Image.frombytes('RGB', image.size, image.tobytes()).save(path)
    return make_task(source, branch, path, list(image.size), sha256(path))


def run(a):
    import numpy as np
    import torch
    from PIL import Image
    from shapely.geometry import shape
    from lowres.evaluate_shared import evaluate_image
    from rev4 import edsr

    prepared = a.calibration_prepared
    rows, tasks = validation_cohort(a, prepared)
    require(environment_identity(a, prepared) == a.b1_identity, 'Require the audited B1 runtime')
    reference = read_json(a.calibration_counts)
    require(reference['identity'] == a.calibration_summary['identity'] and len(reference['counts']) == len(tasks),
            'Unexpected calibration count grid')
    index = reference['identity']['grid'].index(list(THRESHOLDS))
    native_tasks = {t['source_image_id']: (t, v[index]) for t, v in zip(tasks, reference['counts'])
                    if t['condition'] == 'lr2'}
    sources = sorted(rows)
    identity = dict(calibration_sha256=a.calibration_sha256, b1_checkpoint_sha256=a.b1_digest,
                    edsr_sha256=WEIGHT_SHA256, edsr_url=WEIGHT_URL, thresholds=list(THRESHOLDS),
                    diagnostics_sha256=sha256(a.diagnostics_records),
                    calibration_counts_sha256=a.calibration_counts_sha256, calibration_identity=reference['identity'],
                    code={name: sha256(Path(__file__).with_name(name)) for name in ('edsr.py', 'restoration.py')})
    progress_path = a.restoration_run/'paired_records.json'
    progress = read_json(progress_path) if progress_path.is_file() else dict(identity=identity, records=[])
    records = progress['records']
    require(progress['identity'] == identity and [r['source_image_id'] for r in records] == sources[:len(records)],
            'Restoration cache identity or source order mismatch')

    options = copy.copy(a)
    options.checkpoint = a.b1_checkpoint
    detector, processor, _, _ = detector_for(options)
    processor.mask_thr, processor.min_text_score = THRESHOLDS
    device = torch.device('cuda:0')
    model = edsr.EDSR()
    model.load_state_dict(torch.load(a.restoration_weights, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval().requires_grad_(False)
    protocol = read_json(a.evaluation_protocol)

    with TemporaryDirectory(prefix='lr2-restoration-', dir=a.restoration_run) as temp:
        temporary = Path(temp)/'restored.png'
        for number, source in enumerate(sources[len(records):], len(records)+1):
            row = rows[source]
            original, counts = native_tasks[source]
            task_size = original['size']
            native_y = task_size[1]/row['clean_size'][1]
            heights = {}
            for entry in row['gt_clean']['valid']:
                _, low, _, high = shape(entry['geometry']).bounds
                heights[str(entry['ann_id'])] = sum((high-low)*native_y >= cut for cut in (4, 8, 16))
            expected_size = (2*task_size[0], 2*task_size[1])
            record = dict(source_image_id=source, branches={})
            for branch in BRANCHES:
                prep_ms = 0.
                task = original
                if branch != 'lr2':
                    start = time.perf_counter()
                    with Image.open(original['path']) as image:
                        rgb = image.convert('RGB')
                    if branch == 'bicubic_x2':
                        image = rgb.resize(expected_size, Image.Resampling.BICUBIC)
                    else:
                        image = Image.fromarray(edsr.restore(model, np.asarray(rgb), device))
                    prep_ms = (time.perf_counter()-start)*1000
                    task = resized_task(source, branch, temporary, image, expected_size)
                _, _, _, _, _, prediction, detect_ms = forward(detector, processor, task, row)
                result = evaluate_image(row['gt_clean'], row['roi_clean'], prediction, protocol)
                if branch == 'lr2':
                    require(result['counts'] == counts, f'LR2 baseline differs from job 12396 for {source}')
                matched = list(map(str, result['matched_ann_ids']))
                matched_bins = Counter(heights[ann] for ann in matched)
                gt_bins = Counter(heights.values())
                record['branches'][branch] = dict(counts=result['counts'], matched=matched,
                                                  bins=[[gt_bins[i], matched_bins[i]] for i in range(len(HEIGHT_BINS))],
                                                  latency_ms=prep_ms+detect_ms)
            records.append(record)
            if number == 1 or number % 50 == 0 or number == len(sources):
                save_json(progress_path, progress)
                print(f'LR2 restoration {number:,}/{len(sources):,} paired sources', flush=True)
    summarize(a.restoration_run, records, identity, a.diagnostics_records)


def summarize(output, records, identity, diagnostics):
    import numpy as np
    from lowres.evaluate_shared import rates
    from lowres.report import bootstrap, interval

    sources = [record['source_image_id'] for record in records]
    per_branch = {branch: {r['source_image_id']: r['branches'][branch] for r in records} for branch in BRANCHES}
    boot = bootstrap({b: {s: dict(metrics=x) for s, x in rows.items()} for b, rows in per_branch.items()},
                     sources, repetitions=2000)
    scores = {}
    for branch, rows in per_branch.items():
        values = list(rows.values())
        counts = np.asarray([x['counts'] for x in values]).sum(axis=0).tolist()
        tp, gt, pred, _ = counts
        bins = np.asarray([x['bins'] for x in values]).sum(axis=0).tolist()
        scores[branch] = dict(**rates(counts), tp=tp, fp=pred-tp, fn=gt-tp,
                              latency_median_ms=median(x['latency_ms'] for x in values),
                              bins={key: dict(gt=n, matched=m, recall=m/n if n else None)
                                    for key, (n, m) in zip(HEIGHT_BINS, bins)})
    differences = {}
    for newer, older in (('edsr_x2', 'bicubic_x2'), ('edsr_x2', 'lr2'), ('bicubic_x2', 'lr2')):
        differences[f'{newer} − {older}'] = dict(hmean_pp=100*(scores[newer]['hmean']-scores[older]['hmean']),
                                               ci95_pp=interval(100*(boot[newer]-boot[older])))
    failures = {r['source_image_id']: {x['ann_id']: x['primary'] for x in r['misses']}
                for r in read_json(diagnostics)['records']}
    transitions, examples = {}, {}
    for branch in BRANCHES[1:]:
        recovered = Counter()
        lost = 0
        examples[branch] = {'recovered': [], 'lost': [], 'more_fp': []}
        for source in sources:
            original = per_branch['lr2'][source]
            current = per_branch[branch][source]
            gained = sorted(set(current['matched'])-set(original['matched']))
            missing = sorted(set(original['matched'])-set(current['matched']))
            recovered.update(failures[source].get(word, 'outside_clean_tp_lr2_fn_diagnostics') for word in gained)
            lost += len(missing)
            for key, ids in (('recovered', gained), ('lost', missing)):
                examples[branch][key].extend(dict(source_image_id=source, ann_id=word) for word in ids[:max(0, 5-len(examples[branch][key]))])
            extra_fp = (current['counts'][2]-current['counts'][0])-(original['counts'][2]-original['counts'][0])
            if extra_fp > 0 and len(examples[branch]['more_fp']) < 5:
                examples[branch]['more_fp'].append(dict(source_image_id=source, delta_fp=extra_fp))
        transitions[branch] = dict(recovered=sum(recovered.values()), lost=lost, recovered_by_prior_diagnosis=dict(recovered))

    lines = ['# LR2 restoration before B1-cal', '', f'{len(sources):,} paired remaining-validation sources.', '',
             '| Input | Precision | Recall | Hmean | TP | FP | FN | Median ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for branch, s in scores.items():
        lines.append(f'| {branch} | {s["precision"]:.2%} | {s["recall"]:.2%} | {s["hmean"]:.2%} | '
                     f'{s["tp"]:,} | {s["fp"]:,} | {s["fn"]:,} | {s["latency_median_ms"]:.1f} |')
    lines += ['', '| Hmean difference | pp | Paired 95% CI (pp) |', '|---|---:|---:|']
    for name, delta in differences.items():
        lo, hi = delta['ci95_pp']
        lines.append(f'| {name} | {delta["hmean_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    lines += ['', '| Native LR2 height | GT | LR2 recall | Bicubic recall | EDSR recall |', '|---|---:|---:|---:|---:|']
    for key in HEIGHT_BINS:
        n = scores['lr2']['bins'][key]['gt']
        values = ' | '.join(f'{scores[b]["bins"][key]["recall"]:.2%}' if n else 'N/A' for b in BRANCHES)
        lines.append(f'| {key} px | {n:,} | {values} |')
    lines += ['', 'Latency includes restoration and detector decode, forward and postprocessing. '
                  'Temporary PNG encoding, file integrity checks and evaluation are excluded.', '']
    (output/'restoration_report.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(output/'restoration_summary.json', dict(identity=identity, sources=len(sources), scores=scores,
              differences=differences, bootstrap_repetitions=2000, bootstrap_seed=20261107,
              transitions_from_lr2=transitions, examples=examples))
    print('Restoration complete:', output/'restoration_report.md', flush=True)
