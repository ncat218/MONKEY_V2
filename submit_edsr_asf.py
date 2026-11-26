#!/usr/bin/env python3
"""Queue frozen EDSR+ASF inference after the existing B1+EDSR controls succeed."""
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
from rev4.sr_asf_plan import RECIPE,ASF_SHA,BASELINE_JOB
import submit_edsr_recheck as baseline

REPO=Path(__file__).resolve().parent
FILES=('submit_edsr_asf.py','rev4/sr_asf.py','rev4/sr_asf_plan.py','tests/test_sr_asf.py')


def light():
    a=baseline.light();a.output=REPO/'.runtime/edsr_asf/v1'
    a.launch_identity=dict(stage='EDSR x ASF v1; frozen inference only',recipe=RECIPE,no_test=True,no_training=True,
        source={**a.launch_identity['source'],**{name:sha256(REPO/name) for name in FILES}})
    return a


def dependency():
    lines=subprocess.check_output(['sacct','-X','-j',BASELINE_JOB,'--noheader','--parsable2','--format=State,ExitCode'],text=True).splitlines()
    lines=[v.strip() for v in lines if v.strip()]
    require(len(lines)==1,'Cannot establish B1+EDSR parent job state')
    state,code=lines[0].split('|')[:2]
    if state=='COMPLETED':
        require(code=='0:0','B1+EDSR parent exit is not zero');return None
    require(state in ('RUNNING','PENDING','COMPLETING'),'B1+EDSR parent failed; preserve inputs and stop: '+state)
    return BASELINE_JOB


def verified(a):
    ctx=baseline.verified(a)
    ctx.sr_reference=REPO/'.runtime/edsr_recheck/v1'
    require(read_json(ctx.sr_reference/'submission/latest_job.json')['job_id']==BASELINE_JOB,'Wrong EDSR control job')
    completed=read_json(ctx.sr_reference/'results/complete.json')
    parent=read_json(ctx.sr_reference/'identity.json')
    require(completed['identity']==parent and completed['no_test'] and completed['no_training']
        and completed['report_sha256']==sha256(ctx.sr_reference/'results/edsr_recheck_report.md')
        and completed['summary_sha256']==sha256(ctx.sr_reference/'results/edsr_recheck_summary.json'),
        'Require complete unchanged B1+EDSR controls')
    require({k:v for k,v in parent.items() if k!='launch'}=={k:v for k,v in ctx.trial_identity.items() if k!='launch'},
            'B1+EDSR data/runtime/weights differ from ASF experiment')
    require(all(a.launch_identity['source'].get(k)==v for k,v in parent['launch']['source'].items()),
            'Shared SR/data/evaluation source changed')
    trained=read_json(ctx.fusion_reference/'arms/asf/trained.json')
    ctx.asf_checkpoint=Path(trained['checkpoint']);ctx.asf_digest=ASF_SHA
    require(trained['identity']==read_json(ctx.fusion_reference/'identity.json')
        and trained['optimizer_updates']==10000 and trained['micro_iterations']==20000
        and trained['checkpoint_sha256']==ASF_SHA==sha256(ctx.asf_checkpoint),'Completed ASF weights changed')
    ctx.sr_identity_sha=identity(parent)
    ctx.trial_identity.update(asf_checkpoint_sha256=ASF_SHA,sr_reference_identity_sha256=ctx.sr_identity_sha,
        sr_completion_sha256=sha256(ctx.sr_reference/'results/complete.json'),
        native_asf_counts_sha256=sha256(ctx.fusion_reference/'evaluation/asf/counts.json'))
    return ctx


def command(a,dep=None):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    args=['sbatch','--parsable','--job-name=monkey_v2_edsr_asf','--ntasks=1','--account=uit','--partition=normal',
        '--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G','--time=12:00:00',
        '--chdir='+str(REPO),'--output='+str(a.output/'logs/job_%j.stdout.log'),
        '--error='+str(a.output/'logs/job_%j.stderr.log')]
    if dep:args.append('--dependency=afterok:'+dep)
    return args+['--wrap',env+shlex.join([str(a.python),'-u',str(REPO/'submit_edsr_asf.py'),'--worker'])]


def worker(a):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'ASF+SR requires Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'ASF+SR code changed after submission')
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=verified(a);p=a.output/'identity.json'
        if p.exists():require(read_json(p)==ctx.trial_identity,'ASF+SR identity changed')
        else:save_json(p,ctx.trial_identity)
        storage_probe(a.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_sr_asf','tests.test_sr_recheck','-v'],cwd=REPO,check=True)
        from rev4.sr_asf import run
        print('Frozen ASF checkpoint:',ASF_SHA,'; completed B1+EDSR controls:',BASELINE_JOB,flush=True)
        run(ctx)


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group()
    group.add_argument('--plan-only',action='store_true');group.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();a=light()
    if args.worker:
        faulthandler.enable(all_threads=True)
        try:worker(a)
        except BaseException:
            error=traceback.format_exc();write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}.txt',error)
            print(error,file=sys.stderr,flush=True);raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login')
    dep=dependency()
    if args.plan_only:
        print(json.dumps(dict(recipe=RECIPE,dependency=dep,output=str(a.output),sbatch=command(a,dep)),indent=2));return
    with run_lock(a.output/'submission'):
        launch=a.output/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'ASF+SR source changed; preserve old version')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket=a.output/'submission/latest_job.json'
        if ticket.exists():
            prior=read_json(ticket)['job_id']
            if prior in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',prior);return
        (a.output/'logs').mkdir(parents=True,exist_ok=True);storage_probe(a.output);save_json(launch,a.launch_identity)
        job=subprocess.check_output(command(a,dep),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        save_json(ticket,dict(job_id=job,baseline_job=BASELINE_JOB,dependency=dep,identity=a.launch_identity))
        print('Submitted EDSR+ASF inference-only job',job,'; dependency:',dep or 'parent already complete',
              '\nReport:',a.output/'results/edsr_asf_report.md',flush=True)


if __name__=='__main__':main()
