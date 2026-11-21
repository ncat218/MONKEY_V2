#!/usr/bin/env python3
"""Equal-budget 10k-update head/plain/gated runs, initialized afresh from B1."""
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
from rev4.latent_long_plan import ARMS,RECIPE
import submit_module_trials as original

REPO=Path(__file__).resolve().parent
PILOT_JOB='81918'
PILOT_REPORT_SHA='9c80e00ed90507f5a898a48f91e1278b58664da32fb5c63cf12736e0e093ae9e'


def light():
    a=original.light('latent')
    a.output=REPO/'.runtime/latent_refinement/long_10k_v1'
    extra=('submit_latent_long.py','rev4/durable_io.py','rev4/latent_long_plan.py',
           'rev4/latent_long_plugins.py','tests/test_latent_long.py','tests/test_latent_recovery.py')
    a.launch_identity=dict(stage='latent 10000-update extension',pilot_job=PILOT_JOB,
        recipe=RECIPE,no_test=True,source={**a.launch_identity['source'],**{name:sha256(REPO/name) for name in extra}})
    return a


def verified(a):
    ctx=original.verified(a)
    root=REPO/'.runtime/latent_refinement/pilot_v3'
    receipt=read_json(root/'results/complete.json')
    require(receipt['report_sha256']==PILOT_REPORT_SHA==sha256(root/'results/latent_report.md'),'Pilot report changed')
    require(read_json(root/'submission/latest_job.json')['job_id']==PILOT_JOB,'Unexpected pilot job')
    for name in ('rev4/latent_modules.py','rev4/latent_plugins.py'):
        require(receipt['identity']['launch']['source'][name]==sha256(REPO/name),'Pilot architecture changed')
    ctx.trial_identity['pilot_reference']=dict(job_id=PILOT_JOB,report_sha256=PILOT_REPORT_SHA,
                                              complete_sha256=sha256(root/'results/complete.json'))
    ctx.latent_recipe=RECIPE
    ctx.latent_audit_hook='LatentLongAuditHook'
    ctx.latent_extra_imports=['rev4.latent_long_plugins']
    return ctx


def parent_complete():
    rows=subprocess.check_output(['sacct','-X','-j',PILOT_JOB,'--noheader','--parsable2','--format=State,ExitCode'],text=True).splitlines()
    rows=[v.strip() for v in rows if v.strip()]
    require(len(rows)==1 and rows[0].split('|')[:2]==['COMPLETED','0:0'],'Pilot accounting must be COMPLETED exit zero')


def command(a):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    argv=[str(a.python),'-u',str(REPO/'submit_latent_long.py'),'--worker']
    return ['sbatch','--parsable','--job-name=monkey_v2_latent_10k','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=06:00:00','--chdir='+str(REPO),
        '--output='+str(a.output/'logs/job_%j.stdout.log'),'--error='+str(a.output/'logs/job_%j.stderr.log'),
        '--wrap',env+shlex.join(argv)]


def child(a,phase,arm=None):
    argv=[str(a.python),'-u',str(REPO/'submit_latent_long.py'),'--phase',phase]
    if arm:argv+=['--arm',arm]
    print('START',phase,arm or '',flush=True)
    result=subprocess.run(argv,cwd=REPO)
    print('END',phase,arm or '',result.returncode,flush=True)
    require(result.returncode==0,f'10k phase failed: {phase} {arm}, exit={result.returncode}')


def check_training(ctx,arm):
    import torch
    r=read_json(ctx.output/arm/'trained.json')
    require(r['identity']==ctx.trial_identity and r['optimizer_updates']==10000 and r['micro_iterations']==20000,'Incomplete long training')
    require(sha256(r['checkpoint'])==r['checkpoint_sha256'],'Long checkpoint changed')
    c=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
    require(c['meta']['iter']==20000 and c['optimizer']['state'],'Incomplete runner checkpoint')
    steps=[int(v['step']) for v in c['optimizer']['state'].values()]
    require(min(steps)==max(steps)==10000,'Optimizer did not complete 10000 updates')
    require(r['full_stream']['micro_iterations']==20000 and r['module_updates']['det_head']>0,'Missing whole-stream/module audit')
    if arm!='head':require(r['module_updates']['feature_adapter']>0,'Adapter was not trained')
    head=ctx.output/'head/trained.json'
    if head.exists():require(read_json(head)['full_stream']==r['full_stream'],'Whole TRAIN or augmentation stream differs across arms')
    print('PASS long training audit',arm,'optimizer_steps',min(steps),'module_updates',r['module_updates'],flush=True)


def allocated(a,args):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Require Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Long-run source changed')
    ctx=verified(a)
    p=ctx.output/'identity.json'
    if p.exists():require(read_json(p)==ctx.trial_identity,'Long-run identity changed')
    else:save_json(p,ctx.trial_identity)
    from rev4 import latent_trial
    if args.phase:
        if args.phase!='probe':
            gate=read_json(ctx.output/'preflight.json')
            require(gate['status']=='PASS' and gate['identity']==ctx.trial_identity,'GPU probe must PASS')
        if args.phase=='probe':latent_trial.probe(ctx)
        elif args.phase=='train':
            latent_trial.train(ctx,args.arm);check_training(ctx,args.arm)
        elif args.phase=='evaluate':latent_trial.evaluate(ctx,args.arm)
        else:
            for arm in ARMS:check_training(ctx,arm)
            latent_trial.report(ctx)
        return
    with (ctx.output/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage_probe(ctx.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_latent_modules','tests.test_latent_config',
                        'tests.test_latent_recovery','tests.test_latent_long','-v'],cwd=REPO,check=True)
        child(a,'probe')
        for arm in ARMS:
            child(a,'train',arm)
            child(a,'evaluate',arm)
        child(a,'report')
        report=ctx.output/'results/latent_report.md'
        require(report.stat().st_size>100 and read_json(ctx.output/'results/complete.json')['report_sha256']==sha256(report),
                'Incomplete 10k report')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    actions.add_argument('--phase',choices=('probe','train','evaluate','report'),help=argparse.SUPPRESS)
    parser.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    args=parser.parse_args()
    require((args.phase in ('train','evaluate'))==bool(args.arm),'Unexpected/missing arm')
    a=light()
    if args.worker or args.phase:
        faulthandler.enable(all_threads=True)
        try:allocated(a,args)
        except BaseException:
            failure=traceback.format_exc()
            print(failure,file=sys.stderr,flush=True);print('LONG RUN FAILURE:\n'+failure,flush=True)
            try:write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}_{args.phase or "worker"}.txt',failure)
            except BaseException:print(traceback.format_exc(),file=sys.stderr,flush=True)
            raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from login node')
    parent_complete()
    if args.plan_only:
        print(json.dumps(dict(recipe=RECIPE,arms=ARMS,output=str(a.output),sbatch=command(a),
                             full_input_checks='inside GPU allocation; fresh B1 init, no pilot continuation'),indent=2))
        return
    with run_lock(a.output/'submission'):
        launch=a.output/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'Existing long-run source changed')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket=a.output/'submission/latest_job.json'
        if ticket.exists():
            old=read_json(ticket)['job_id']
            queue=subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split()
            if old in queue:print('Already queued/running',old);return
        storage_probe(a.output)
        save_json(launch,a.launch_identity)
        (a.output/'logs').mkdir(parents=True,exist_ok=True)
        job=subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        print('Slurm accepted 3-arm 10k job',job,flush=True)
        save_json(ticket,dict(job_id=job,pilot_job=PILOT_JOB,identity=a.launch_identity))
        print('Logs:',a.output/'logs','\nReport:',a.output/'results/latent_report.md',flush=True)


if __name__=='__main__':main()
