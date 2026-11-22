#!/usr/bin/env python3
"""Export completed research evidence without reading TEST or changing active runs."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile
from zoneinfo import ZoneInfo


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();repo=a.repo.resolve();out=a.output_dir.resolve()
    assert out.is_relative_to(repo/'.runtime') and not out.exists(),'Use a new output inside .runtime'
    out.mkdir(parents=True)
    def copy(source,target):
        source=repo/source;target=out/target
        assert source.is_file() and source.stat().st_size>0,source
        if source.suffix=='.json':load(source)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def compress(source,target):
        source=repo/source;target=out/target
        assert source.is_file() and source.stat().st_size>0,source
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('wb') as sink:
            with gzip.GzipFile(filename='',mode='wb',fileobj=sink,mtime=0) as zipped:shutil.copyfileobj(src,zipped)
    def zip_files(source,target,include):
        source=repo/source;target=out/target;target.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(source.rglob('*')):
                if path.is_file() and include(path.relative_to(source)):
                    assert path.stat().st_size>0,path
                    z.write(path,path.relative_to(source))
    completed=[('craft','.runtime/craft_screen/run_v1','craft_report.md'),
               ('latent_pilot','.runtime/latent_refinement/pilot_v3','latent_report.md'),
               ('latent_10k','.runtime/latent_refinement/long_10k_v1','latent_report.md'),
               ('fixed_diagnostics','.runtime/rev4_fixed_diagnostics/v1','fixed_diagnostics_report.md')]
    for name,root,report in completed:
        root=Path(root);receipt=load(repo/root/'results/complete.json')
        assert receipt['report_sha256']==digest(repo/root/'results'/report),(name,'report hash')
        for path in sorted((repo/root/'results').iterdir()):
            if not path.is_file():continue
            target=Path(name)/'results'/path.name
            if path.suffix=='.csv' and path.stat().st_size>1_000_000:compress(path.relative_to(repo),str(target)+'.gz')
            else:copy(path.relative_to(repo),target)
        for item in ['identity.json','run_identity.json','preflight.json','corruption_probe.json',
                     'recovery_import.json','submission/latest_job.json']:
            if (repo/root/item).is_file():copy(root/item,Path(name)/'provenance'/item)
        if name.startswith('latent'):
            import numpy as np
            for arm in ('head','plain','gated'):
                base=root/arm
                for item in ('trained.json','latency.json','inference_config.py'):
                    copy(base/item,Path(name)/arm/item)
                source=repo/base/'validation.json';r=load(source)
                assert len(r['tasks'])==4524 and r['identity']==load(repo/root/'identity.json')
                train=load(repo/base/'trained.json');assert digest(train['checkpoint'])==train['checkpoint_sha256']
                ids=sorted({v['source_image_id'] for v in r['tasks']});assert len(ids)==2262
                index={v:i for i,v in enumerate(ids)};counts=np.zeros((2262,2,2,4),dtype=np.int64)
                bins=np.zeros((2262,2,2,4,2),dtype=np.int64);seen=set()
                for task in r['tasks']:
                    key=task['source_image_id'],task['condition'];assert key not in seen;seen.add(key)
                    i=index[key[0]];j=int(key[1]=='lr2');counts[i,j]=task['counts'];bins[i,j]=task['bins']
                summary=load(repo/root/'results/latent_summary.json')
                for j,c in enumerate(('clean','lr2')):
                    tp,gt,pred,_=map(int,counts[:,j,1].sum(0));score=summary['scores'][arm][c]
                    assert (tp,gt-tp,pred-tp)==(score['tp'],score['fn'],score['fp'])
                target=out/name/arm/'source_counts.npz'
                np.savez_compressed(target,source_ids=np.asarray(ids),conditions=np.asarray(['clean','lr2']),
                    threshold_pairs=np.asarray([[.3,.4],[.4,.6]]),counts=counts,bins=bins)
                compress(base/'validation.json',Path(name)/arm/'validation.json.gz')
    feature=Path('.runtime/lr2_feature_distillation/pilot_v1')
    for name in ('pilot_report.md','pilot_summary.json'):copy(feature/'results'/name,Path('feature_kd/results')/name)
    for arm in ('continued','feature_kd'):
        for name in ('trained.json','validation.json'):
            if name=='validation.json':compress(feature/arm/name,Path('feature_kd')/arm/(name+'.gz'))
            else:copy(feature/arm/name,Path('feature_kd')/arm/name)
    fixed=Path('.runtime/rev4_fixed_diagnostics/v1/review')
    for name in ('selected_review.json','diagnosis_summary.json','human_review.csv'):copy(fixed/name,Path('fixed_diagnostics/review')/name)
    for name in ('gt_diagnosis.csv','fp_taxonomy_proxies.csv'):compress(fixed/name,Path('fixed_diagnostics/review')/(name+'.gz'))
    zip_files(fixed,'fixed_diagnostics/review/paired_cards.zip',lambda rel:rel.parts[0]=='examples')
    craft=Path('.runtime/craft_screen/run_v1')
    zip_files(craft/'train_qc','craft/train_teacher_qc.zip',lambda rel:True)
    target=out/'craft/per_source_counts.csv.gz'
    with gzip.open(target,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['source_image_id','condition','tp','gt','pred','covered','matched_ann_ids'])
        count=0
        for path in sorted((repo/craft/'samples').glob('*/*.json')):
            r=load(path);writer.writerow([path.stem,path.parent.name,*r['counts'],json.dumps(r['matched'])]);count+=1
        assert count==4524
    # Source membership only. This does not open a TEST manifest or any TEST image.
    from submit_b1_diagnostics import verified_inputs
    context=verified_inputs()
    assert digest(context.validation_manifest)=='5cbf2230b0f83ab30816aca8afc5cb466e0b3301030ccd25d7d6d62d3136dfff'
    memberships={'S1':set(context.s1_ids),'S2':set(context.s2_ids)}
    target=out/'data/validation_ids.csv';target.parent.mkdir(parents=True,exist_ok=True)
    ids=set();n=0
    with context.validation_manifest.open() as source,target.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['source_image_id','clean_relpath','clean_sha256','role'])
        for line in source:
            r=json.loads(line);sid=r['source_image_id'];assert sid not in ids;ids.add(sid);n+=1
            role='S1' if sid in memberships['S1'] else ('S2' if sid in memberships['S2'] else 'remaining-validation')
            writer.writerow([sid,r['clean_relpath'],r['clean_sha256'],role])
    assert n==3262 and not memberships['S1']&memberships['S2']
    for name,values in memberships.items():(out/'data'/f'{name}_source_ids.txt').write_text('\n'.join(sorted(values))+'\n')
    for name in ('pilot_report.md','pilot_summary.json'):
        if (context.s1/'results'/name).is_file():copy((context.s1/'results'/name).relative_to(repo),Path('s1/results')/name)
    for name in ('s2_audit.json','pre_s2_lock_receipt.md'):
        source=Path('.runtime/rev4_task1/recheck-20261108-120822')/name
        if (repo/source).is_file():copy(source,Path('technical_preflight')/name)
    train=list(csv.DictReader((repo/'docs/handoff/data/train_ids.csv').open()))
    assert len(train)==15222 and not ids&{r['source_image_id'] for r in train}
    write(out/'data/split_receipt.json',dict(train_sources=15222,validation_sources=3262,test_sources_historical=3263,
        validation_roles=dict(S1=500,S2=500,remaining_validation=2262),train_validation_disjoint=True,
        train_manifest_sha256='ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d',
        validation_manifest_sha256=digest(context.validation_manifest),test_manifest_read=False,
        test_manifest_historical_target_sha256='054f44643c7d3fa0006c8eb3e502c602953957c34cf957cc5d954dd04f4e079f'))
    # Active run: immutable preflight artifacts and clearly dated progress, never partial scores.
    fusion=Path('.runtime/fusion_study/v1')
    for name in ('identity.json','preflight.json','corruption_probe.json','stress_sources.json','submission/latest_job.json'):
        if (repo/fusion/name).is_file():copy(fusion/name,Path('fusion_snapshot')/name)
    log=repo/fusion/'logs/job_106090.stdout.log'
    if log.exists():
        (out/'fusion_snapshot/stdout_tail.txt').write_text('\n'.join(log.read_text(errors='replace').splitlines()[-35:])+'\n')
    now=datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()
    jobs=subprocess.run(['sacct','-X','-j','74219,74220,77412,81918,88935,106090','--parsable2',
                         '--format=JobID,JobName,State,Elapsed,ExitCode'],capture_output=True,text=True,check=True)
    (out/'job_status.psv').write_text(jobs.stdout,encoding='utf-8')
    write(out/'snapshot.json',dict(snapshot_at=now,timezone='Asia/Bangkok',no_test=True,
        completed=['74220','81918','88935','73827','48499'],active_run_snapshot='106090',
        checkpoint_weights='Published separately with tensor-equality metadata; no intermediate optimizer checkpoints',
        dataset_images='Not included; use TextVQA/TextOCR official distribution',
        fixed_review_cards=127,craft_train_qc_sources=200,craft_train_review_cards=32))
    inventory={str(path.relative_to(out)).replace('\\','/'):{'bytes':path.stat().st_size,'sha256':digest(path)}
               for path in sorted(out.rglob('*')) if path.is_file()}
    write(out/'export_inventory.json',dict(files=inventory,snapshot_at=now,no_test=True))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(out))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(inventory)),indent=2))


if __name__=='__main__':main()
