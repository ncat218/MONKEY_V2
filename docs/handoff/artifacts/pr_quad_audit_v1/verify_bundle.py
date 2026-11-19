"""Verify the downloaded bundle; optionally check NumPy counts against references."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda:source.read(1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts',action='store_true',help='Also validate shapes, historical counts and selected metrics (NumPy)')
    parser.add_argument('--weights',action='store_true',help='Also validate the LFS weight downloads')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
    for name,item in manifest['files'].items():
        path=root/name
        assert path.is_file(),f'Missing {name}'
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],f'Hash/size mismatch: {name}'
    if args.weights:
        for name,item in manifest['weights'].items():
            path=root/item['relative_path']
            assert sha(path)==item['sha256'],f'{name} weight hash mismatch; run git lfs pull'
    complete=json.loads((root/'results/complete.json').read_text(encoding='utf-8'))
    assert sha(root/'results/decision_audit_report.md')==complete['report_sha256']
    if args.counts:
        import numpy as np
        digest=hashlib.sha256()
        with gzip.open(root/'data/validation_remaining.jsonl.gz','rb') as source:
            for data in iter(lambda:source.read(1024*1024),b''):
                digest.update(data)
        assert digest.hexdigest()==manifest['decompressed_validation_sha256'],'Frozen GT manifest changed'
        with gzip.open(root/'provenance/eda_common_counts.csv.gz','rt',encoding='utf-8',newline='') as source:
            reference={(r['model'],r['source_id'],r['condition']):[int(r[k]) for k in ('tp','gt','pred')]
                       for r in csv.DictReader(source)}
        summary=json.loads((root/'results/decision_audit_summary.json').read_text(encoding='utf-8'))
        all_sources=None
        for model in ('B0','B-Clean','B1'):
            with np.load(root/'counts'/f'{model}_source_counts.npz',allow_pickle=False) as data:
                ids=data['source_ids'].tolist()
                settings=[json.loads(s) for s in data['settings']]
                counts=data['counts']
                assert len(ids)==len(set(ids))==2262 and counts.shape==(2262,2,72,5)
                assert all_sources is None or ids==all_sources
                all_sources=ids
                common=next(i for i,s in enumerate(settings) if s['geometry']=='base' and s['min_text_score']==.4)
                for i,source in enumerate(ids):
                    for j,condition in enumerate(('clean','lr2')):
                        assert counts[i,j,common,:3].tolist()==reference[model,source,condition]
                selected=summary['selected'][model]
                setting_i=settings.index(selected['setting'])
                for j,condition in enumerate(('clean','lr2')):
                    totals=counts[:,j,setting_i].sum(axis=0).tolist()
                    assert totals==[selected[condition][k] for k in ('tp','gt','pred','tiny_gt','tiny_tp')]
                assert (counts>=0).all()
        assert len(reference)==3*2262*2
    print('PASS: completed-job report, bundle hashes'+(', source counts and metrics' if args.counts else '')+
          (', LFS weights' if args.weights else '')+'. Human review remains PENDING.')


if __name__=='__main__':
    main()
