#!/usr/bin/env python3
"""One Slurm allocation: probes, in-run pilot/full training, fixed-threshold stress transfer."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from lowres.common import read_json,run_lock,sha256,identity
from rev4 import require
from rev4.durable_io import save_json,storage_probe,write_text
from rev4.fusion_plan import ARMS,RECIPE
import submit_module_trials as original

REPO=Path(__file__).resolve().parent
BASE_CLEAN_SHA='861bbde2aaf0b83cc98fab148d374276aa84012a7599a43dc1e7a227628aacfc'
FPNC_SHA='493d4369e4ca42eafba42c8d671782aba3cdfc8fd5aa04a921b599ca5459c3fe'
MODELS=('B1','B-Clean')+ARMS


def light(recover=False):
    a=original.light('latent');a.output=REPO/'.runtime/fusion_study/v1'
    extra=['submit_fusion_study.py','rev4/durable_io.py','rev4/fusion_plan.py','rev4/fusion_modules.py',
           'rev4/fusion_plugins.py','rev4/fusion_training.py','rev4/fusion_evaluation.py','tests/test_fusion_study.py',
           'rev4/local_scratch.py','rev4/fusion_recovery.py','tests/test_fusion_recovery.py']
    extra+=[str(p.relative_to(REPO)).replace('\\','/') for p in sorted((REPO/'rev4/dysample_vendor').iterdir()) if p.is_file()]
    a.launch_identity=dict(stage='fusion study v1; single allocation',recipe=RECIPE,no_test=True,
        source={**a.launch_identity['source'],**{name:sha256(REPO/name) for name in extra}})
    if recover:
        from rev4.fusion_recovery import PARENT_IDENTITY_SHA
        a.recover_from=REPO/'.runtime/fusion_study/v1'
        a.output=REPO/'.runtime/fusion_study/recovery_v2'
        a.launch_identity.update(stage='fusion study recovery v2; single allocation',
            recovery=dict(parent_job='106090',parent_identity_sha256=PARENT_IDENTITY_SHA,
                          changes='node-local temporary images; bounded cleanup retry; verified checkpoint/count reuse'))
    return a


def verified(a):
    import inspect
    from mmocr.models.textdet.necks import FPNC
    from rev4.fusion_evaluation import stress_ids
    from rev4.calibration import validation_cohort
    ctx=original.verified(a)
    ctx.recover_from=getattr(a,'recover_from',None)
    parent=REPO/'.runtime/latent_refinement/long_10k_v1'
    completed=read_json(parent/'results/complete.json')
    require(read_json(parent/'submission/latest_job.json')['job_id']=='88935' and
            completed['report_sha256']==sha256(parent/'results/latent_report.md'),'Require complete long-run reference')
    ctx.bclean_checkpoint=REPO/'.runtime/bclean/v1/weights/iter_20000.pth'
    require(sha256(ctx.bclean_checkpoint)==BASE_CLEAN_SHA,'B-Clean checkpoint changed')
    require(sha256(inspect.getfile(FPNC))==FPNC_SHA,'Installed FPNC changed')
    require(sha256(ctx.blur_generator)==ctx.lock['blur_jpeg_generator_sha256'],'Frozen corruption generator changed')
    rows,_=validation_cohort(ctx,ctx.calibration_prepared)
    selected=stress_ids(rows);require(len(selected)==500,'Invalid stress cohort')
    ctx.trial_identity.update(bclean_checkpoint_sha256=BASE_CLEAN_SHA,installed_fpnc_sha256=FPNC_SHA,
                             stress_ids_sha256=identity(selected),stress_sources=500,
                             reference_long_completion_sha256=sha256(parent/'results/complete.json'))
    return ctx


def command(a):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    args=[str(a.python),'-u',str(REPO/'submit_fusion_study.py'),'--worker']
    if getattr(a,'recover_from',None):args.append('--recover')
    return ['sbatch','--parsable','--job-name=monkey_v2_fusion_study','--ntasks=1','--account=uit','--partition=normal',
        '--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G','--time=12:00:00','--chdir='+str(REPO),
        '--output='+str(a.output/'logs/job_%j.stdout.log'),'--error='+str(a.output/'logs/job_%j.stderr.log'),
        '--wrap',env+shlex.join(args)]


def child(a,phase,arm=None):
    args=[str(a.python),'-u',str(REPO/'submit_fusion_study.py'),'--phase',phase]
    if getattr(a,'recover_from',None):args.append('--recover')
    if arm:args+=['--arm',arm]
    print('START',phase,arm or '',flush=True)
    result=subprocess.run(args,cwd=REPO)
    print('END',phase,arm or '',result.returncode,flush=True)
    require(result.returncode==0,f'Fusion phase failed: {phase} {arm}; stopped before next phase')


def allocated(a,args):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Require Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Study source changed')
    ctx=verified(a);p=ctx.output/'identity.json'
    if p.exists():require(read_json(p)==ctx.trial_identity,'Study identity changed')
    else:save_json(p,ctx.trial_identity)
    from rev4 import fusion_training as training,fusion_evaluation as evaluation
    if args.phase:
        if args.phase!='probe':require(read_json(ctx.output/'preflight.json')['status']=='PASS' and
            read_json(ctx.output/'preflight.json')['identity']==ctx.trial_identity,'Technical probe must PASS')
        if args.phase=='probe':
            (ctx.output/'scratch').mkdir(exist_ok=True)
            evaluation.corruption_probe(ctx);training.probe(ctx)
        elif args.phase=='train':training.train(ctx,args.arm)
        elif args.phase=='evaluate':evaluation.evaluate(ctx,args.arm)
        else:evaluation.report(ctx)
        return
    with (ctx.output/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage_probe(ctx.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_fusion_study','tests.test_fusion_recovery',
                        'tests.test_latent_recovery','-v'],cwd=REPO,check=True)
        rows,_,selected=evaluation.specification(ctx)
        save_json(ctx.output/'stress_sources.json',dict(source_ids=selected,sha256=identity(selected),
                  selection='lowest SHA256(42:fusion-stress:source_id), independent of predictions',development=True))
        child(a,'probe')
        if ctx.recover_from:
            from rev4.fusion_recovery import restore
            restore(ctx)
        # Baselines verify transfer and the evaluator before spending full training budget.
        for name in ('B1','B-Clean'):child(a,'evaluate',name)
        for arm in ARMS:
            child(a,'train',arm)  # 1k pilot gate and 10k full run share this Runner/optimizer.
            child(a,'evaluate',arm)
        child(a,'report')
        report=ctx.output/'results/fusion_report.md'
        require(report.stat().st_size>100 and read_json(ctx.output/'results/complete.json')['report_sha256']==sha256(report),'Incomplete final report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    actions.add_argument('--phase',choices=('probe','train','evaluate','report'),help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=MODELS,help=argparse.SUPPRESS)
    parser.add_argument('--recover',action='store_true',help='Recover job106090 into a new audited run; reuse finished training')
    args=parser.parse_args()
    require((args.phase in ('train','evaluate'))==bool(args.arm),'Unexpected/missing arm')
    require(args.phase!='train' or args.arm in ARMS,'Baseline checkpoints must not be trained')
    a=light(args.recover)
    if args.worker or args.phase:
        faulthandler.enable(all_threads=True)
        try:allocated(a,args)
        except BaseException:
            failure=traceback.format_exc();print(failure,file=sys.stderr,flush=True);print('FUSION FAILURE:\n'+failure,flush=True)
            try:write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}_{args.phase or "worker"}.txt',failure)
            except BaseException:print(traceback.format_exc(),file=sys.stderr,flush=True)
            raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from login node')
    if args.plan_only:
        print(json.dumps(dict(recipe=RECIPE,arms=ARMS,models=MODELS,output=str(a.output),
             full_input_checks='inside allocation; no TEST or target threshold selection',sbatch=command(a)),indent=2));return
    with run_lock(a.output/'submission'):
        launch=a.output/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'Study code changed; preserve old version')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket=a.output/'submission/latest_job.json'
        if ticket.exists():
            old=read_json(ticket)['job_id']
            if old in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',old);return
        storage_probe(a.output);save_json(launch,a.launch_identity);(a.output/'logs').mkdir(exist_ok=True)
        job=subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response');print('Submitted ONE fusion pilot/full job',job,flush=True)
        save_json(ticket,dict(job_id=job,identity=a.launch_identity))
        print('Logs:',a.output/'logs','\nReport:',a.output/'results/fusion_report.md',flush=True)


if __name__=='__main__':main()
