#!/usr/bin/env python3
"""Independent latent-refinement training pilot and CRAFT inference screen."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import traceback

from lowres.common import read_json,run_lock,save_json,sha256
from rev4 import require
from rev4.adaptation_jobs import REPO
from rev4.latent_plan import ARMS,RECIPE,PAIRS

CRAFT_SHA='4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17'


def light(kind):
    stack=REPO.parent/'monkeyocr_workspace/environment_and_models__phase1_verified_stack'
    files=['submit_module_trials.py']
    if kind=='latent':
        files+=['rev4/latent_plan.py','rev4/latent_modules.py','rev4/latent_plugins.py','rev4/latent_trial.py','tests/test_latent_modules.py','tests/test_latent_config.py']
    else:
        files+=['rev4/craft_screen.py','rev4/craft_core.py','tests/test_craft_core.py']
        files += [str(p.relative_to(REPO)).replace('\\','/') for p in sorted((REPO/'rev4/craft_vendor').glob('*')) if p.is_file()]
    return SimpleNamespace(kind=kind,output=REPO/'.runtime'/('latent_refinement/pilot_v2' if kind=='latent' else 'craft_screen/run_v1'),
        python=stack/'python311_venv/bin/python',cuda=stack/'cuda_toolkit_12_8_1__private_nvidia_conda_environment',
        s1=REPO/'.runtime/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0',
        craft_weight=REPO/'.runtime/craft_screen/assets/craft_mlt_25k.pth',
        launch_identity=dict(kind=kind,recipe=RECIPE if kind=='latent' else 'CRAFT fixed official-default screen; no training',
                             source={f:sha256(REPO/f) for f in files},parent_job='73827',no_test=True))


def verified(a):
    import submit_b1_diagnostics
    from rev4.training_data import TRAIN_NAME,TRAIN_SHA256
    ctx=submit_b1_diagnostics.verified_inputs()
    ctx.kind,ctx.output,ctx.launch_identity=a.kind,a.output,a.launch_identity
    ctx.craft_weight=a.craft_weight
    ctx.fixed_run=REPO/'.runtime/rev4_fixed_diagnostics/v1'
    complete=read_json(ctx.fixed_run/'results/complete.json')
    require(complete['report_sha256']==sha256(ctx.fixed_run/'results/fixed_diagnostics_report.md'),
            'Require complete unchanged fixed diagnostic job 73827')
    require(read_json(ctx.fixed_run/'submission/latest_job.json')['job_id']=='73827','Unexpected fixed diagnostic job')
    require(sha256(ctx.calibration_prepared/TRAIN_NAME)==TRAIN_SHA256,'Frozen TRAIN manifest changed')
    if a.kind=='craft':
        require(sha256(a.craft_weight)==CRAFT_SHA,'CRAFT checkpoint hash mismatch')
    ctx.trial_identity=dict(launch=a.launch_identity,b1_checkpoint_sha256=ctx.b1_digest,
        train_manifest_sha256=TRAIN_SHA256,prepared_sha256=sha256(ctx.calibration_prepared/'prepared.json'),
        validation_sha256=sha256(ctx.calibration_prepared/'validation_remaining.jsonl'),
        tasks_sha256=sha256(ctx.calibration_prepared/'validation_tasks.jsonl'),
        evaluator_sha256=sha256(ctx.evaluation_protocol),fixed_completion_sha256=sha256(ctx.fixed_run/'results/complete.json'),
        craft_checkpoint_sha256=CRAFT_SHA if a.kind=='craft' else None)
    return ctx


def command(a,dependency=None):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    argv=[str(a.python),'-u',str(REPO/'submit_module_trials.py'),'--kind',a.kind,'--worker']
    cmd=['sbatch','--parsable','--job-name=monkey_v2_'+a.kind,'--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time='+('06:00:00' if a.kind=='latent' else '03:00:00'),'--chdir='+str(REPO),
        '--output='+str(a.output/'logs/job_%j.stdout.log'),'--error='+str(a.output/'logs/job_%j.stderr.log')]
    if dependency:
        cmd.append('--dependency=afterok:'+dependency)
    return cmd+['--wrap',env+shlex.join(argv)]


def dependency():
    rows=subprocess.check_output(['sacct','-X','-j','73827','--noheader','--parsable2','--format=State,ExitCode'],text=True).splitlines()
    rows=[r.strip() for r in rows if r.strip()]
    require(len(rows)==1,'Cannot establish parent job state')
    state,code=rows[0].split('|')[:2]
    if state=='COMPLETED':
        require(code=='0:0','Parent diagnostic exit code is not zero')
        return None
    require(state in ('PENDING','RUNNING','COMPLETING'),'Parent diagnostic failed; stop before new trials: '+state)
    return '73827'


def child(a,phase,arm=None):
    args=[str(a.python),'-u',str(REPO/'submit_module_trials.py'),'--kind',a.kind,'--phase',phase]
    if arm: args+=['--arm',arm]
    subprocess.run(args,cwd=REPO,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',required=True,choices=('latent','craft'))
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    actions.add_argument('--phase',choices=('probe','train','evaluate','report'),help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    args=parser.parse_args()
    require(not args.phase or args.kind=='latent','Only latent trial has training phases')
    require(not args.arm or args.phase in ('train','evaluate'),'Unexpected arm argument')
    require(args.phase not in ('train','evaluate') or args.arm,'Missing training/evaluation arm')
    a=light(args.kind)
    if args.plan_only:
        if a.kind=='craft': require(a.craft_weight.is_file() and sha256(a.craft_weight)==CRAFT_SHA,'CRAFT asset not ready')
        print(json.dumps(dict(identity=a.launch_identity,output=str(a.output),
             arms=ARMS if a.kind=='latent' else ['CRAFT standalone','complementarity vs fixed B1','200 TRAIN pseudo-label QC'],
             full_checks='inside allocation; stop on FAIL before training',sbatch=command(a,'73827')),indent=2))
    elif args.worker or args.phase:
        import fcntl
        faulthandler.enable(all_threads=True)
        require(os.environ.get('SLURM_JOB_ID'),'Worker requires Slurm allocation')
        require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Source changed after submission')
        ctx=verified(a)
        from rev4.adaptation_jobs import lock_record
        lock_record(ctx.output/'identity.json',ctx.trial_identity)
        if args.phase:
            from rev4 import latent_trial
            if args.phase!='probe':
                require(read_json(ctx.output/'preflight.json')['identity']==ctx.trial_identity and
                        read_json(ctx.output/'preflight.json')['status']=='PASS','Probe must PASS first')
            if args.phase=='probe':latent_trial.probe(ctx)
            elif args.phase=='train':latent_trial.train(ctx,args.arm)
            elif args.phase=='evaluate':latent_trial.evaluate(ctx,args.arm)
            else:latent_trial.report(ctx)
            return
        with (ctx.output/'worker.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            faulthandler.enable(all_threads=True)
            try:
                test='tests.test_latent_modules' if a.kind=='latent' else 'tests.test_craft_core'
                tests=[test]+(['tests.test_latent_config'] if a.kind=='latent' else [])
                subprocess.run([str(a.python),'-m','unittest',*tests,'-v'],cwd=REPO,check=True)
                if a.kind=='latent':
                    child(ctx,'probe')
                    for arm in ARMS:
                        child(ctx,'train',arm)
                        child(ctx,'evaluate',arm)
                    child(ctx,'report')
                else:
                    from rev4.craft_screen import run
                    run(ctx)
            except BaseException:
                failure=traceback.format_exc()
                (ctx.output/f'failure_{os.environ["SLURM_JOB_ID"]}.txt').write_text(failure,encoding='utf-8')
                raise
    else:
        require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login node')
        if a.kind=='craft':require(sha256(a.craft_weight)==CRAFT_SHA,'CRAFT checkpoint missing or changed')
        dep=dependency()
        with run_lock(a.output/'submission'):
            launch=a.output/'submission/launch.json'
            if launch.is_file():require(read_json(launch)==a.launch_identity,'Existing trial source changed; use a new version')
            if (a.output/'results/complete.json').is_file():
                print('Already complete:',a.output/'results');return
            receipt=a.output/'submission/latest_job.json'
            if receipt.is_file():
                previous=read_json(receipt)['job_id']
                queue=subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split()
                if previous in queue:print('Already queued/running:',previous);return
            (a.output/'logs').mkdir(parents=True,exist_ok=True)
            save_json(launch,a.launch_identity)
            job=subprocess.check_output(command(a,dep),text=True).strip().split(';')[0]
            require(job.isdecimal(),'Unexpected sbatch response: '+job)
            save_json(receipt,dict(job_id=job,parent_job='73827',dependency=dep,identity=a.launch_identity))
            print(f'Submitted {a.kind} job {job}\nLogs: {a.output/"logs"}\nResults: {a.output/"results"}',flush=True)


if __name__=='__main__':main()
