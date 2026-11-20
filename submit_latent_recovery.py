#!/usr/bin/env python3
"""Recover v2 finished arms without retraining them; complete the interrupted pilot."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from lowres.common import read_json,run_lock,sha256
from rev4 import require
from rev4.durable_io import save_json,storage_probe,write_text
from rev4.latent_recovery import PARENT_JOB
import submit_module_trials as original

REPO=Path(__file__).resolve().parent


def light():
    a=original.light('latent')
    a.parent_output=a.output
    a.output=REPO/'.runtime/latent_refinement/pilot_v3'
    extra=('submit_latent_recovery.py','rev4/latent_recovery.py','rev4/durable_io.py','tests/test_latent_recovery.py')
    a.launch_identity=dict(stage='latent recovery v3',parent_job=PARENT_JOB,
        recipe=original.RECIPE,no_test=True,source={**a.launch_identity['source'],**{name:sha256(REPO/name) for name in extra}})
    return a


def verified(a):
    from rev4.latent_recovery import parent_inputs
    ctx=original.verified(a)
    ctx.parent_output=a.parent_output
    ctx.trial_identity['recovery']=parent_inputs(ctx,a.parent_output)
    return ctx


def command(a):
    argv=[str(a.python),'-u',str(REPO/'submit_latent_recovery.py'),'--worker']
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    return ['sbatch','--parsable','--job-name=monkey_v2_latent_recovery','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=03:00:00','--chdir='+str(REPO),
        '--output='+str(a.output/'logs/job_%j.stdout.log'),'--error='+str(a.output/'logs/job_%j.stderr.log'),
        '--wrap',env+shlex.join(argv)]


def parent_stopped():
    rows=subprocess.check_output(['sacct','-X','-j',PARENT_JOB,'--noheader','--parsable2','--format=State,ExitCode'],text=True).splitlines()
    rows=[v.strip() for v in rows if v.strip()]
    require(len(rows)==1 and rows[0].split('|')[:2]==['FAILED','1:0'],'Parent must be the stopped failed v2 job')


def child(a,phase,arm=None):
    argv=[str(a.python),'-u',str(REPO/'submit_latent_recovery.py'),'--phase',phase]
    if arm:argv+=['--arm',arm]
    print('START phase',phase,arm or '',flush=True)
    result=subprocess.run(argv,cwd=REPO)
    print('END phase',phase,arm or '','returncode',result.returncode,flush=True)
    require(result.returncode==0,f'Latent recovery phase failed: {phase} {arm}, returncode={result.returncode}')


def allocated(a,args):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Require Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Recovery source changed after submission')
    ctx=verified(a)
    identity_path=ctx.output/'identity.json'
    if identity_path.exists():require(read_json(identity_path)==ctx.trial_identity,'Recovery identity changed')
    else:save_json(identity_path,ctx.trial_identity)
    from rev4 import latent_trial
    if args.phase:
        if args.phase!='probe':
            p=read_json(ctx.output/'preflight.json')
            require(p['status']=='PASS' and p['identity']==ctx.trial_identity,'Recovery GPU probe must PASS')
        if args.phase=='probe':latent_trial.probe(ctx)
        elif args.phase=='train':latent_trial.train(ctx,args.arm)
        elif args.phase=='evaluate':latent_trial.evaluate(ctx,args.arm)
        else:latent_trial.report(ctx)
        return
    with (ctx.output/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage_probe(ctx.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_latent_modules','tests.test_latent_config',
                        'tests.test_latent_recovery','-v'],cwd=REPO,check=True)
        from rev4.calibration import validation_cohort
        from rev4.latent_recovery import import_finished
        _,tasks=validation_cohort(ctx,ctx.calibration_prepared)
        import_finished(ctx,ctx.parent_output,tasks)
        child(a,'probe')
        for arm in original.ARMS:
            child(a,'train',arm)
            child(a,'evaluate',arm)
        child(a,'report')
        report=ctx.output/'results/latent_report.md'
        complete=read_json(ctx.output/'results/complete.json')
        require(report.stat().st_size>100 and complete['report_sha256']==sha256(report),'Incomplete recovery report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    mode.add_argument('--phase',choices=('probe','train','evaluate','report'),help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=original.ARMS,help=argparse.SUPPRESS)
    args=parser.parse_args()
    require((args.phase in ('train','evaluate'))==bool(args.arm),'Unexpected/missing arm')
    a=light()
    if args.worker or args.phase:
        faulthandler.enable(all_threads=True)
        try:allocated(a,args)
        except BaseException:
            failure=traceback.format_exc()
            print(failure,file=sys.stderr,flush=True)
            print('RECOVERY FAILURE (also stderr):\n'+failure,flush=True)
            try:write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}_{args.phase or "worker"}.txt',failure)
            except BaseException:print('FAILURE FILE WRITE ALSO FAILED:\n'+traceback.format_exc(),file=sys.stderr,flush=True)
            raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from login node')
    parent_stopped()
    if args.plan_only:
        print(json.dumps(dict(output=str(a.output),parent_job=PARENT_JOB,reuse_training=['head','plain'],
            remaining_training=['gated'],recompute=['plain validation','plain latency'],source=a.launch_identity,
            input_audit='full checkpoint and environment verification inside allocation',sbatch=command(a)),indent=2))
        return
    with run_lock(a.output/'submission'):
        launch=a.output/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'Source changed; preserve recovery output')
        if (a.output/'results/complete.json').exists():
            print('Already completed',a.output/'results');return
        previous=a.output/'submission/latest_job.json'
        if previous.exists():
            job=read_json(previous)['job_id']
            queued=subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split()
            if job in queued:print('Already queued/running',job);return
        storage_probe(a.output)
        save_json(launch,a.launch_identity)
        (a.output/'logs').mkdir(parents=True,exist_ok=True)
        job=subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        print('Slurm accepted latent recovery job',job,flush=True)
        save_json(previous,dict(job_id=job,parent_job=PARENT_JOB,identity=a.launch_identity))
        print('Logs:',a.output/'logs','\nReport:',a.output/'results/latent_report.md',flush=True)


if __name__=='__main__':main()
