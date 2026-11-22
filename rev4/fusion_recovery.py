"""Audited import of immutable job106090 artifacts into a new recovery identity."""
import copy
from pathlib import Path

from lowres.common import read_json, sha256
from rev4 import require
from rev4.durable_io import save_json
from rev4.fusion_plan import PAIRS, RECIPE

PARENT_IDENTITY_SHA = '7f64ebf61528d584b813c4d4f1874cc9faf27fe9484bdfe02671cffd269583d7'
WEIGHT_SHA = {
    'nearest': '6fee5adfbc30df8c029f52c4c92edf5d52ac6a61c0a94b82502f2469abb2938e',
    'asf': '9cae7ad418cc906edb56667c03c0d5f9e25f02b7ad929e4024ff03ebaec38059',
}
PARENT_FILES = {
    'evaluation/B1/counts.json': 'e88da4950fa9e18866727cdf3069457dbbd20d2e60be36a4f139c2a5fa694ba4',
    'evaluation/B-Clean/counts.json': 'bac6e566d092040dadaa6a9274a1ac9d2c49bd5b749d44c6ec9d494105eb9c20',
    'evaluation/nearest/counts.json': '74e01dadb419a0ebb9d43f13be3b9055c48e09b2b45f76a1ee82320620b516bf',
    'evaluation/asf/counts.json': '19cfbf772f6d6e1141240880d87195aa2208a0fa6c4670513f89c5f741a56f6c',
    'arms/nearest/trained.json': 'ca3152cf49a4596e4465de5769a0947f33c6d19a263b736f21418b4773ea6bb6',
    'arms/asf/trained.json': '0898917bd7e57fe0598106186972a529ed5ebdbcf2e834169af74091dccd1fcf',
}
CHANGED_SOURCE = {'submit_fusion_study.py', 'rev4/fusion_evaluation.py'}
NEW_SOURCE = {'rev4/local_scratch.py', 'rev4/fusion_recovery.py', 'tests/test_fusion_recovery.py'}


def validate_identity(parent, current):
    require({k: v for k, v in parent.items() if k != 'launch'} ==
            {k: v for k, v in current.items() if k != 'launch'}, 'Recovery data/runtime identity changed')
    require(parent['launch']['recipe'] == current['launch']['recipe'] == RECIPE, 'Recovery recipe changed')
    old, new = parent['launch']['source'], current['launch']['source']
    require(set(new) == set(old) | NEW_SOURCE, 'Unexpected recovery source additions/removals')
    require(all(new[name] == digest for name, digest in old.items() if name not in CHANGED_SOURCE),
            'Training, architecture, evaluator or other frozen source changed')


def validate_counts(record, parent, tasks, expected_size, checkpoint_sha):
    require(record['identity'] == parent and record['checkpoint_sha256'] == checkpoint_sha,
            'Parent evaluation identity/checkpoint mismatch')
    rows = record['tasks']
    require(len(rows) == expected_size, 'Unexpected parent evaluation coverage')
    require([v['sample_id'] for v in rows] == [v['sample_id'] for v in tasks[:expected_size]],
            'Parent evaluation is not the exact expected task prefix')
    for row, task in zip(rows, tasks):
        require(all(row[k] == task[k] for k in ('source_image_id', 'condition')), 'Parent task metadata differs')
        require(len(row['counts']) == len(row['bins']) == len(PAIRS), 'Parent threshold grid differs')
        for counts, bins in zip(row['counts'], row['bins']):
            require(len(counts) == 3 and all(type(v) is int and v >= 0 for v in counts), 'Invalid parent counts')
            tp, gt, pred = counts
            require(tp <= min(gt, pred), 'Impossible parent matching counts')
            require(len(bins) == 4 and all(len(b) == 2 and all(type(v) is int and v >= 0 for v in b)
                                          and b[1] <= b[0] for b in bins), 'Invalid parent height bins')
            require(sum(b[0] for b in bins) == gt and sum(b[1] for b in bins) == tp, 'Parent bins/counts mismatch')


def restore(a):
    from rev4.fusion_evaluation import specification

    parent_root, output = a.recover_from.resolve(), a.output.resolve()
    require(parent_root != output and not output.is_relative_to(parent_root), 'Recovery must preserve parent run')
    require(sha256(parent_root/'identity.json') == PARENT_IDENTITY_SHA, 'Unexpected parent job identity')
    parent = read_json(parent_root/'identity.json')
    require(read_json(parent_root/'submission/latest_job.json')['job_id'] == '106090', 'Wrong parent job')
    validate_identity(parent, a.trial_identity)
    for name, digest in PARENT_FILES.items():
        require(sha256(parent_root/name) == digest, 'Parent artifact changed: '+name)
    preflight = read_json(parent_root/'preflight.json')
    require(preflight['identity'] == parent and preflight['status'] == 'PASS', 'Parent probe did not pass')
    _, tasks, _ = specification(a)
    require(len(tasks) == 7524, 'Recovery task coverage changed')
    imported = {}
    trained = {}
    for arm in ('nearest', 'asf'):
        name = f'arms/{arm}/trained.json'
        value = read_json(parent_root/name)
        path = parent_root/f'arms/{arm}/weights/iter_20000.pth'
        require(value['identity'] == parent and value['arm'] == arm and value['optimizer_updates'] == 10000
                and value['micro_iterations'] == 20000, 'Parent training incomplete')
        require(path.resolve() == Path(value['checkpoint']).resolve()
                and sha256(path) == value['checkpoint_sha256'] == WEIGHT_SHA[arm], 'Parent weights changed')
        trained[arm] = value
        imported[name] = value
    require(trained['nearest']['stream'] == trained['asf']['stream'], 'Parent training streams differ')
    checkpoint_shas = {'B1': a.trial_identity['b1_checkpoint_sha256'],
                       'B-Clean': a.trial_identity['bclean_checkpoint_sha256'], **WEIGHT_SHA}
    for arm in ('B1', 'B-Clean', 'nearest', 'asf'):
        name = f'evaluation/{arm}/counts.json'
        value = read_json(parent_root/name)
        require(value['arm'] == arm, 'Parent evaluation arm differs')
        validate_counts(value, parent, tasks, 5700 if arm == 'asf' else 7524, checkpoint_shas[arm])
        imported[name] = value
    # Validate every source before writing. Never modify parent files or weights.
    for name, old_value in imported.items():
        value = copy.deepcopy(old_value)
        value['identity'] = a.trial_identity
        target = output/name
        if target.exists():
            current = read_json(target)
            if name.startswith('evaluation/'):
                require(current['identity'] == value['identity'] and current['checkpoint_sha256'] == value['checkpoint_sha256']
                        and current['arm'] == value['arm'] and current['tasks'][:len(value['tasks'])] == value['tasks'],
                        'Existing recovery prediction prefix changed')
            else:
                require(current == value, 'Existing recovery training receipt differs')
        else:
            save_json(target, value)
    save_json(output/'recovery_import.json', dict(status='PASS', identity=a.trial_identity, parent_job='106090',
        parent_root=str(parent_root), parent_identity_sha256=PARENT_IDENTITY_SHA, parent_files=PARENT_FILES,
        reused_checkpoints=WEIGHT_SHA, reused_task_counts={'B1': 7524, 'B-Clean': 7524, 'nearest': 7524, 'asf': 5700},
        policy='No retraining nearest/asf; preserve exact prediction prefix; remeasure short timing in this allocation',
        source_changes=sorted(CHANGED_SOURCE), no_test=True))
    print('PASS recovery import: nearest/ASF final weights preserved; resume ASF after 5700 tasks', flush=True)
