"""Native and same-grid DBNet calibration on the frozen remaining validation."""
import copy
from fractions import Fraction
from pathlib import Path

from lowres.common import jsonl, read_json, save_json, sha256
from rev4 import require
from rev4.b1_training import environment_identity
from rev4.stage_a import detector_for, forward
from rev4.training_data import RECIPE, VAL_SOURCES


CONDITIONS = ('clean', 'lr2')
NATIVE = (.30, .30)
GRID = tuple((mask, score) for mask in RECIPE['calibration']['mask_thr']
             for score in RECIPE['calibration']['min_text_score'])


def validation_cohort(a, prepared):
    rows = {row['source_image_id']: row for row in jsonl(prepared/'validation_remaining.jsonl')}
    expected = set(a.original_remaining_ids) - set(a.s2_ids)
    require(len(rows) == VAL_SOURCES and set(rows) == expected and not set(rows) & set(a.s1_ids),
            'Require exactly the frozen 2,262 remaining validation sources')
    tasks = list(jsonl(prepared/'validation_tasks.jsonl'))
    coverage = {(task['condition'], task['source_image_id']) for task in tasks}
    require(len(tasks) == len(coverage) and coverage == {(c, s) for c in CONDITIONS for s in rows},
            'Expected paired clean and LR2 tasks for every remaining validation source')
    return rows, tasks


def summarize_grid(counts, tasks):
    import numpy as np
    from lowres.evaluate_shared import rates

    vectors = np.asarray(counts, dtype=np.int64)
    require(vectors.shape == (len(tasks), len(GRID), 4), 'Invalid calibration count vectors')
    totals = {condition: vectors[[task['condition'] == condition for task in tasks]].sum(axis=0).tolist()
              for condition in CONDITIONS}
    grid = []
    for index, (mask, cutoff) in enumerate(GRID):
        scores = {}
        for condition in CONDITIONS:
            vector = totals[condition][index]
            tp, gt, predictions, _ = vector
            scores[condition] = dict(**rates(vector), tp=tp, fp=predictions-tp, fn=gt-tp)
        grid.append(dict(mask_thr=mask, min_text_score=cutoff, scores=scores,
                         balanced_hmean=sum(item['hmean'] for item in scores.values())/len(CONDITIONS)))
    return grid


def choose_threshold(grid):
    def ranking(candidate):
        # Compare exact micro-Hmean ratios so rounding does not change a tie.
        score = sum(Fraction(2*item['tp'], (2*item['tp']+item['fp']+item['fn']) or 1)
                    for item in candidate['scores'].values())
        native = (candidate['mask_thr'], candidate['min_text_score']) == NATIVE
        return score, native

    return max(grid, key=ranking)


def evaluate_model(a, checkpoint, rows, tasks, protocol, cache, record):
    from lowres.evaluate_shared import evaluate_image
    from lowres.inference import unpack

    options = copy.copy(a)
    options.checkpoint = checkpoint
    detector, native_pp, _, _ = detector_for(options)
    processors = {}
    for mask, cutoff in GRID:
        if (mask, cutoff) == NATIVE:
            continue
        processor = copy.copy(native_pp)
        processor.mask_thr, processor.min_text_score = mask, cutoff
        processors[mask, cutoff] = processor

    counts = record['counts']
    for index, task in enumerate(tasks[len(counts):], len(counts)+1):
        row = rows[task['source_image_id']]
        _, probability, sample, _, restore, native, _ = forward(detector, native_pp, task, row)
        vector = []
        for pair in GRID:
            prediction = native
            if pair != NATIVE:
                processor = processors[pair]
                prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
            vector.append(evaluate_image(row['gt_clean'], row['roi_clean'], prediction, protocol)['counts'])
        counts.append(vector)
        if index % 100 == 0 and index < len(tasks):
            save_json(cache, record)
        if index == 1 or index % 100 == 0 or index == len(tasks):
            print(f'{cache.stem} calibration {index:,}/{len(tasks):,} tasks', flush=True)


def markdown(summary):
    lines = ['# Rev.4 B0/B1 calibration', '',
             f'{summary["sources"]:,} remaining validation sources paired across Clean and LR2. '
             'Threshold selection uses mean micro-Hmean over the two conditions.', '',
             '| Model | Mode | Mask | Score | Condition | Precision | Recall | Hmean | TP | FP | FN |',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for model, data in summary['models'].items():
        for mode in ('native', 'calibrated'):
            item = data[mode]
            for condition, score in item['scores'].items():
                lines.append(f'| {model} | {mode} | {item["mask_thr"]:.2f} | {item["min_text_score"]:.2f} | '
                             f'{condition} | {score["precision"]:.2%} | {score["recall"]:.2%} | '
                             f'{score["hmean"]:.2%} | {score["tp"]:,} | {score["fp"]:,} | {score["fn"]:,} |')
    lines += ['', '| B1 − B0 Hmean | Clean (pp) | LR2 (pp) | Mean (pp) |', '|---|---:|---:|---:|']
    for mode, delta in summary['b1_minus_b0_hmean_pp'].items():
        lines.append(f'| {mode} | {delta["clean"]:+.2f} | {delta["lr2"]:+.2f} | {delta["mean"]:+.2f} |')
    lines += ['', 'B1 native counts reproduce job 12187. These validation scores also support threshold selection.', '']
    return '\n'.join(lines)


def run(a):
    prepared = a.calibration_prepared
    rows, tasks = validation_cohort(a, prepared)
    runtime = environment_identity(a, prepared)
    require(runtime == a.b1_identity, 'Calibration environment differs from the locked B1 identity')
    identity = dict(b1_summary_sha256=a.b1_summary_sha256, b1_identity_sha256=a.b1_identity_sha256,
                    b0_checkpoint_sha256=a.lock['checkpoint_sha256'], b1_checkpoint_sha256=a.b1_digest,
                    validation_sha256=sha256(prepared/'validation_remaining.jsonl'),
                    tasks_sha256=sha256(prepared/'validation_tasks.jsonl'),
                    evaluator_policy_sha256=a.lock['evaluator_policy_sha256'],
                    grid=[list(pair) for pair in GRID], calibration_code_sha256=sha256(Path(__file__)))
    protocol = read_json(a.evaluation_protocol)
    models = {}
    for name, checkpoint in (('B0', a.checkpoint), ('B1', a.b1_checkpoint)):
        cache = a.calibration_run/'counts'/f'{name}.json'
        record = read_json(cache) if cache.is_file() else dict(identity=identity, counts=[])
        require(record['identity'] == identity, f'{name} calibration cache identity mismatch')
        needs_evaluation = len(record['counts']) < len(tasks)
        if needs_evaluation:
            evaluate_model(a, checkpoint, rows, tasks, protocol, cache, record)
        grid = summarize_grid(record['counts'], tasks)
        native = grid[GRID.index(NATIVE)]
        if name == 'B1':
            require(native['scores'] == a.b1_summary['selected_checkpoint']['scores'],
                    'B1 native calibration counts differ from job 12187')
        if needs_evaluation:
            save_json(cache, record)
        models[name] = dict(native=native, calibrated=choose_threshold(grid), grid=grid)

    deltas = {}
    for mode in ('native', 'calibrated'):
        b0, b1 = (models[name][mode] for name in ('B0', 'B1'))
        deltas[mode] = {c: 100*(b1['scores'][c]['hmean']-b0['scores'][c]['hmean']) for c in CONDITIONS}
        deltas[mode]['mean'] = 100*(b1['balanced_hmean']-b0['balanced_hmean'])
    summary = dict(stage='B0/B1 calibration', identity=identity, sources=len(rows), models=models,
                   b1_minus_b0_hmean_pp=deltas, b1_native_reproduced=True)
    report = a.calibration_run/'results/calibration_report.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown(summary), encoding='utf-8')
    save_json(report.with_name('calibration_summary.json'), summary)
    print('Calibration complete:', report, flush=True)
