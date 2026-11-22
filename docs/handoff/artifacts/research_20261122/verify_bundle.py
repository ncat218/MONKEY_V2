"""Verify downloaded evidence; never read dataset images or launch inference/jobs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_metrics(counts, expected):
    tp, gt, pred, _ = map(int, counts)
    assert (tp, pred-tp, gt-tp) == (expected['tp'], expected['fp'], expected['fn'])
    for key, value in dict(precision=tp/max(pred, 1), recall=tp/max(gt, 1),
                           hmean=2*tp/max(gt+pred, 1)).items():
        assert abs(value-expected[key]) < 1e-12, (key, value, expected[key])


def verify_counts(root, repo):
    import numpy as np

    with (root/'data/validation_ids.csv').open(encoding='utf-8', newline='') as stream:
        validation = list(csv.DictReader(stream))
    with (repo/'docs/handoff/data/train_ids.csv').open(encoding='utf-8', newline='') as stream:
        train = list(csv.DictReader(stream))
    train_ids = {row['source_image_id'] for row in train}
    val_ids = {row['source_image_id'] for row in validation}
    assert len(train) == len(train_ids) == 15222
    assert len(validation) == len(val_ids) == 3262
    assert not train_ids & val_ids
    remaining = {row['source_image_id'] for row in validation if row['role']=='remaining-validation'}
    assert len(remaining) == 2262
    for role in ('S1', 'S2'):
        ids = set((root/f'data/{role}_source_ids.txt').read_text().splitlines())
        assert len(ids) == 500
        assert ids == {row['source_image_id'] for row in validation if row['role']==role}

    for stage in ('latent_pilot', 'latent_10k'):
        summary = read(root/f'{stage}/results/latent_summary.json')
        for arm in ('head', 'plain', 'gated'):
            with np.load(root/f'{stage}/{arm}/source_counts.npz', allow_pickle=False) as data:
                assert set(data['source_ids'].tolist()) == remaining
                assert data['counts'].shape == (2262, 2, 2, 4)
                assert data['bins'].shape == (2262, 2, 2, 4, 2)
                assert data['conditions'].tolist() == ['clean', 'lr2']
                assert data['threshold_pairs'].tolist() == [[.3, .4], [.4, .6]]
                assert (data['counts'] >= 0).all()
                for j, cond in enumerate(('clean', 'lr2')):
                    check_metrics(data['counts'][:, j, 1].sum(axis=0), summary['scores'][arm][cond])
                    assert data['bins'][:, j, 1].sum(axis=0).tolist() == summary['height_bins'][arm][cond]
    summary = read(root/'fixed_diagnostics/results/fixed_diagnostics_summary.json')
    with np.load(root/'fixed_diagnostics/results/paired_source_counts.npz', allow_pickle=False) as data:
        assert set(data['source_ids'].tolist()) == remaining
        for model in ('B-Clean', 'B1'):
            for cond in ('clean', 'lr2'):
                key = model.replace('-', '_')+'_'+cond
                check_metrics(data[key].sum(axis=0), summary['scores'][model][cond])
    summary = read(root/'craft/results/craft_summary.json')
    with gzip.open(root/'craft/per_source_counts.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for cond in ('clean', 'lr2'):
        group = [row for row in rows if row['condition']==cond]
        assert len(group) == 2262
        assert {row['source_image_id'] for row in group} == remaining
        counts = [sum(int(row[key]) for row in group) for key in ('tp', 'gt', 'pred', 'covered')]
        check_metrics(counts, summary['scores']['CRAFT'][cond])
        assert all(len(json.loads(row['matched_ann_ids'])) == int(row['tp']) for row in group)
    print('PASS: TRAIN/VAL IDs disjoint and published per-source counts reproduce metrics/bins')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', action='store_true', help='Check IDs and count aggregates (NumPy)')
    parser.add_argument('--weights', action='store_true', help='Verify all seven public checkpoint files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parents[3]
    manifest = read(root/'bundle_manifest.json')
    for name, item in manifest['files'].items():
        path = root/name
        assert path.resolve().is_relative_to(root), name
        assert path.is_file(), f'Missing {name}'
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'Mismatch: {name}; run git lfs pull'
    for stage in ('latent_pilot', 'latent_10k'):
        complete = read(root/f'{stage}/results/complete.json')
        assert sha(root/f'{stage}/results/latent_report.md') == complete['report_sha256']
    if args.weights:
        for name, item in manifest['weights'].items():
            path = repo/'docs/handoff/weights'/name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'Weight mismatch: {name}'
        print('PASS: seven public checkpoint SHA256/size checks')
    if args.counts:
        verify_counts(root, repo)
    print(f"PASS: {len(manifest['files'])} artifact hashes; no inference, training or TEST reads")


if __name__ == '__main__':
    main()
