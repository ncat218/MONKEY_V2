#!/usr/bin/env python3
"""Light Slurm launcher: frozen completed B1 plus pretrained EDSR, inference only."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from lowres.common import read_json, run_lock, sha256
from rev4 import require
from rev4.durable_io import save_json, storage_probe, write_text
from rev4.sr_plan import RECIPE
import submit_fusion_study as fusion

REPO = Path(__file__).resolve().parent
WEIGHT_SHA = '2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1'
FILES = ('submit_edsr_recheck.py','rev4/sr_recheck.py','rev4/sr_plan.py','rev4/edsr.py',
         'rev4/restoration.py','tests/test_sr_recheck.py')


def light():
    a = fusion.light()
    a.output = REPO/'.runtime/edsr_recheck/v1'
    a.launch_identity = dict(stage='EDSR recheck v1; inference only',recipe=RECIPE,no_test=True,no_training=True,
        source={**a.launch_identity['source'],**{name:sha256(REPO/name) for name in FILES}})
    return a


def verified(a):
    ctx = fusion.verified(a)
    ctx.fusion_reference = REPO/'.runtime/fusion_study/recovery_v2'
    completion = read_json(ctx.fusion_reference/'results/complete.json')
    require(read_json(ctx.fusion_reference/'submission/latest_job.json')['job_id']=='111683'
            and completion['report_sha256']==sha256(ctx.fusion_reference/'results/fusion_report.md'),
            'Require completed fusion111683')
    reference = read_json(ctx.fusion_reference/'results/fusion_summary.json')
    require(completion['identity']==reference['identity'] and reference['no_test'] is True
            and reference['identity']['b1_checkpoint_sha256']==ctx.b1_digest, 'Fusion B1 reference differs')
    ctx.old_restoration = REPO/'.runtime/lr2_restoration/run'
    old = read_json(ctx.old_restoration/'restoration_summary.json')
    ctx.edsr_weight = ctx.old_restoration.parent/'weights/edsr64_x2-torchsr-v1.0.3.pt'
    require(sha256(ctx.edsr_weight)==old['identity']['edsr_sha256']==WEIGHT_SHA, 'Pinned EDSR weights differ')
    require(old['identity']['b1_checkpoint_sha256']==ctx.b1_digest and old['identity']['thresholds']==[.3,.4],
            'Historical SR did not use the same completed B1')
    require(sha256(REPO/'rev4/edsr.py')==old['identity']['code']['edsr.py'], 'Frozen EDSR implementation changed')
    ctx.trial_identity.update(edsr_sha256=WEIGHT_SHA, fusion_completion_sha256=sha256(ctx.fusion_reference/'results/complete.json'),
        fusion_counts_sha256=sha256(ctx.fusion_reference/'evaluation/B1/counts.json'),
        old_restoration_sha256=sha256(ctx.old_restoration/'restoration_summary.json'),
        old_records_sha256=sha256(ctx.old_restoration/'paired_records.json'))
    return ctx


def command(a):
    env = 'export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env += ' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    return ['sbatch','--parsable','--job-name=monkey_v2_edsr_recheck','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=12:00:00','--chdir='+str(REPO),'--output='+str(a.output/'logs/job_%j.stdout.log'),
        '--error='+str(a.output/'logs/job_%j.stderr.log'),
        '--wrap',env+shlex.join([str(a.python),'-u',str(REPO/'submit_edsr_recheck.py'),'--worker'])]


def worker(a):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'SR worker requires Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'SR source changed after submission')
    with (a.output/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx = verified(a)
        p = a.output/'identity.json'
        if p.exists():require(read_json(p)==ctx.trial_identity,'SR run identity changed')
        else:save_json(p,ctx.trial_identity)
        storage_probe(a.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_sr_recheck','-v'],cwd=REPO,check=True)
        from rev4.sr_recheck import run
        print('Frozen detector B1 SHA256:',ctx.b1_digest,'; EDSR SHA256:',WEIGHT_SHA,flush=True)
        run(ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args();a = light()
    if args.worker:
        faulthandler.enable(all_threads=True)
        try:worker(a)
        except BaseException:
            error = traceback.format_exc()
            write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}.txt',error)
            print(error,file=sys.stderr,flush=True);raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login')
    if args.plan_only:
        print(json.dumps(dict(recipe=RECIPE,output=str(a.output),checks='inside allocation; no training/TEST',sbatch=command(a)),indent=2))
        return
    with run_lock(a.output/'submission'):
        p = a.output/'submission/launch.json'
        if p.exists():require(read_json(p)==a.launch_identity,'Existing SR run source changed; preserve this version')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket = a.output/'submission/latest_job.json'
        if ticket.exists():
            previous = read_json(ticket)['job_id']
            if previous in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',previous);return
        (a.output/'logs').mkdir(parents=True,exist_ok=True)
        storage_probe(a.output);save_json(p,a.launch_identity)
        job = subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        save_json(ticket,dict(job_id=job,identity=a.launch_identity))
        print('Submitted EDSR inference-only job',job,'\nReport:',a.output/'results/edsr_recheck_report.md',flush=True)


if __name__=='__main__':main()
