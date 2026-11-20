"""Verified reuse of finished v2 arms; never reuse damaged caches or restart them."""
import copy
from pathlib import Path

from lowres.common import identity,read_json,sha256
from rev4 import require
from rev4.durable_io import save_json
from rev4.latent_plan import RECIPE

PARENT_JOB='77412'
PARENT_TRIAL_SHA='e141bfe7be822a1eef9ab71e627d5bf560aff88eebd6751c64b295e9a5bd832b'
PARENT_CHECKPOINTS=dict(head='298b0f37ac859cd41a65d66c828b7477b2fa004614e404dd7d698a9aa7da2afb',
                        plain='683786dfac51356af01499ee09f7c95327f1e9e47bbb6b3d4800231c6078c153')


def parent_inputs(ctx,root):
    root=Path(root)
    parent=read_json(root/'identity.json')
    require(read_json(root/'submission/latest_job.json')['job_id']==PARENT_JOB,'Unexpected recovery parent job')
    require(parent['launch']['recipe']==RECIPE,'Training recipe changed')
    require(parent['launch']['source']['rev4/latent_trial.py']==PARENT_TRIAL_SHA,'Unexpected parent trial implementation')
    for name in ('rev4/latent_plan.py','rev4/latent_modules.py','rev4/latent_plugins.py'):
        require(parent['launch']['source'][name]==sha256(Path(__file__).resolve().parents[1]/name),
                'Training architecture, sampler or hook changed: '+name)
    for key in parent:
        if key!='launch':require(parent[key]==ctx.trial_identity[key],'Recovery input identity changed: '+key)
    probe=read_json(root/'preflight.json')
    require(probe['status']=='PASS' and probe['identity']==parent,'Original GPU probe did not PASS')
    records={}
    for arm,expected in PARENT_CHECKPOINTS.items():
        path=root/arm/'trained.json';record=read_json(path)
        checkpoint=(root/arm/'weights/iter_2000.pth').resolve(strict=True)
        require(record['identity']==parent and Path(record['checkpoint']).resolve()==checkpoint,'Parent training identity mismatch')
        require(record['checkpoint_sha256']==expected==sha256(checkpoint),'Parent checkpoint bytes changed')
        require(record['optimizer_updates']==1000 and record['micro_iterations']==2000,'Parent training did not finish')
        records[arm]=dict(receipt=str(path),receipt_sha256=sha256(path),checkpoint=str(checkpoint),checkpoint_sha256=expected)
    require(read_json(records['head']['receipt'])['draws']==read_json(records['plain']['receipt'])['draws'],
            'Finished arm TRAIN/augmentation streams differ')
    require(not list((root/'gated').rglob('*.pth')),'Gated checkpoint exists; inspect before retry')
    return dict(parent_job=PARENT_JOB,parent_identity_sha256=identity(parent),
                parent_preflight_sha256=sha256(root/'preflight.json'),finished=records,
                skipped_corrupt_caches=['plain/validation.json','plain/latency.json'],
                gated_prior_attempt='No recoverable checkpoint; initialization logs damaged; prior update count unknown')


def verified_validation(record,parent,tasks):
    require(record['identity']==parent and record['arm']=='head','Head validation identity mismatch')
    require([v['sample_id'] for v in record['tasks']]==[v['sample_id'] for v in tasks],'Incomplete inherited validation')
    for value,task in zip(record['tasks'],tasks):
        require(value['source_image_id']==task['source_image_id'] and value['condition']==task['condition'],'Inherited task mismatch')
        require(len(value['counts'])==len(value['bins'])==2,'Invalid operating point coverage')
        for counts,bins in zip(value['counts'],value['bins']):
            require(len(counts)==4 and len(bins)==4,'Invalid inherited count dimensions')
            require(all(type(n) is int and n>=0 for n in counts),'Invalid inherited count values')
            require(all(len(pair)==2 and all(type(n) is int and n>=0 for n in pair) and pair[1]<=pair[0] for pair in bins),'Invalid size bins')
            require(sum(pair[0] for pair in bins)==counts[1] and sum(pair[1] for pair in bins)==counts[0],
                    'Inherited size bins do not reproduce GT/TP')


def import_finished(ctx,root,tasks):
    root=Path(root);parent=read_json(root/'identity.json')
    reuse=ctx.trial_identity['recovery']
    imported=[]
    for arm,source in reuse['finished'].items():
        target=ctx.output/arm/'trained.json'
        if target.exists():
            current=read_json(target)
            require(current['identity']==ctx.trial_identity and current['checkpoint_sha256']==source['checkpoint_sha256'],'Imported checkpoint changed')
            continue
        require(sha256(source['receipt'])==source['receipt_sha256'],'Parent receipt changed during import')
        record=read_json(source['receipt'])
        record['identity']=ctx.trial_identity
        record['recovered_from']=dict(**source,parent_job=PARENT_JOB)
        record['training_rerun']=False
        record['peak_gpu_mib_valid']=False
        record['peak_gpu_note']='Inherited counter may have been reset by runner; do not use as valid training peak'
        save_json(target,record);imported.append(arm+'/trained.json')
    for name in ('validation.json','latency.json'):
        target=ctx.output/'head'/name
        if target.exists():
            require(read_json(target)['identity']==ctx.trial_identity,'Imported cache changed')
            continue
        source=root/'head'/name
        record=read_json(source)
        if name=='validation.json':verified_validation(record,parent,tasks)
        else:
            require(record['identity']==parent and set(record['results'])=={'B1','head'},'Invalid inherited timing')
            for value in record['results'].values():require(len(value['measurements'])==8,'Incomplete inherited timing')
        record['identity']=ctx.trial_identity
        record['recovered_from']=dict(path=str(source),sha256=sha256(source),parent_job=PARENT_JOB)
        save_json(target,record);imported.append('head/'+name)
    save_json(ctx.output/'recovery_import.json',dict(identity=ctx.trial_identity,imported=imported,
                                                   training_to_run=['gated'],head_training_reused=True,plain_training_reused=True))
    print('Recovery verified: reuse A0/A1 checkpoints and A0 evaluation; recompute damaged A1 caches; restart interrupted A2.',flush=True)
