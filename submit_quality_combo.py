#!/usr/bin/env python3
"""Queue frozen Tri scorer transfer after the existing EDSR x ASF job; never cancel a running parent."""
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
from rev4.quality_combo_plan import ASSETS,PARENT_JOB,RECIPE
import submit_edsr_asf as parent

REPO=Path(__file__).resolve().parent
FILES=('submit_quality_combo.py','rev4/quality_combo.py','rev4/quality_combo_plan.py',
       'rev4/quality_combo_report.py','tests/test_quality_combo.py')


def light():
    a=parent.light();a.repo=REPO;a.output=REPO/'.runtime/tri_quality_combo/v1'
    provenance=read_json(REPO/ASSETS/'provenance.json')
    sources=list(FILES)+list(provenance['files'])+[ASSETS+'/provenance.json']
    a.launch_identity=dict(stage='frozen Tri scorer x ASF x EDSR v1',recipe=RECIPE,no_test=True,no_training=True,
        source={**a.launch_identity['source'],**{p:sha256(REPO/p) for p in sources}})
    return a


def dependency():
    lines=subprocess.check_output(['sacct','-X','-j',PARENT_JOB,'--noheader','--parsable2',
                                   '--format=State,ExitCode'],text=True).splitlines()
    lines=[v.strip() for v in lines if v.strip()]
    require(len(lines)==1,'Cannot establish EDSR+ASF parent status')
    state,code=lines[0].split('|')[:2]
    if state=='COMPLETED':
        require(code=='0:0','Parent exit nonzero');return None
    require(state in ('RUNNING','PENDING','COMPLETING'),'Parent failed; do not run extension: '+state)
    return PARENT_JOB


def verified(a):
    from rev4.quality_combo import frozen_model
    frozen_model(REPO)
    ctx=parent.verified(a);ctx.repo=REPO
    ctx.combo_reference=REPO/'.runtime/edsr_asf/v1'
    require(read_json(ctx.combo_reference/'submission/latest_job.json')['job_id']==PARENT_JOB,'Wrong ASF parent')
    complete=read_json(ctx.combo_reference/'results/complete.json')
    original=read_json(ctx.combo_reference/'identity.json')
    require(complete['identity']==original and complete['no_test'] and complete['no_training']
        and complete['report_sha256']==sha256(ctx.combo_reference/'results/edsr_asf_report.md')
        and complete['summary_sha256']==sha256(ctx.combo_reference/'results/edsr_asf_summary.json'),
        'Require complete unchanged EDSR+ASF controls')
    require({k:v for k,v in ctx.trial_identity.items() if k!='launch'}==
            {k:v for k,v in original.items() if k!='launch'},'Parent data/runtime/weights differ')
    require(all(a.launch_identity['source'].get(k)==v for k,v in original['launch']['source'].items()),
            'Shared parent code changed')
    ctx.combo_identity_sha=identity(original)
    ctx.trial_identity.update(combo_completion_sha256=sha256(ctx.combo_reference/'results/complete.json'),
        scorer_sha256=sha256(REPO/ASSETS/'model_janek.npz'),scorer_provenance_sha256=sha256(REPO/ASSETS/'provenance.json'))
    return ctx


def command(a,dep=None):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    args=['sbatch','--parsable','--job-name=monkey_v2_tri_quality','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=12:00:00','--chdir='+str(REPO),'--output='+str(a.output/'logs/job_%j.stdout.log'),
        '--error='+str(a.output/'logs/job_%j.stderr.log')]
    if dep:args.append('--dependency=afterok:'+dep)
    return args+['--wrap',env+shlex.join([str(a.python),'-u',str(REPO/'submit_quality_combo.py'),'--worker'])]


def worker(a):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Require Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Quality source changed since submission')
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=verified(a);signature=a.output/'identity.json'
        if signature.exists():require(read_json(signature)==ctx.trial_identity,'Quality run identity changed')
        else:save_json(signature,ctx.trial_identity)
        storage_probe(a.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_quality_combo','-v'],cwd=REPO,check=True)
        from rev4.quality_combo import run
        run(ctx)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--plan-only',action='store_true');group.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();a=light()
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
        if launch.exists():require(read_json(launch)==a.launch_identity,'Existing quality run source changed')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket=a.output/'submission/latest_job.json'
        if ticket.exists():
            prior=read_json(ticket)['job_id']
            if prior in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',prior);return
        storage_probe(a.output);(a.output/'logs').mkdir(parents=True,exist_ok=True);save_json(launch,a.launch_identity)
        job=subprocess.check_output(command(a,dep),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        save_json(ticket,dict(job_id=job,parent_job=PARENT_JOB,dependency=dep,identity=a.launch_identity))
        print('Submitted frozen quality combination job',job,'; dependency:',dep or 'parent complete',
              '\nReport:',a.output/'results/quality_combo_report.md',flush=True)


if __name__=='__main__':main()
