#!/usr/bin/env python3
"""Recover job166501 outputs and add fixed-window zoom+EDSR x ASF x frozen Q.

Old sources, receipts and completed results are preserved. Only baseline evaluation
and unfinished B-Clean zoom are resumed in their original roots. The corrected PRC2
uses a new root and a GT snapshot; no thresholds are relaxed and no TEST is read.
"""
import argparse
import copy
from datetime import datetime,timezone
import faulthandler
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback

from lowres.common import read_json,run_lock,sha256,identity
from rev4 import require
from rev4.durable_io import save_json,storage_probe,write_text
import submit_research_bundle as research
import submit_quality_combo as quality

REPO=Path(__file__).resolve().parent
OUTPUT=REPO/'.runtime/monkey_recovery/v1'
PHASES=('zoom_probe','B1_s43','BClean_s43','zoom_quality','zoom_BClean','prc2_snapshot_train','prc2_snapshot_evaluate','report')
FILES=('submit_monkey_recovery.py','rev4/prc2_snapshot.py','rev4/zoom_quality.py',
       'tests/test_monkey_recovery.py','docs/monkey-recovery-v1.md')


def light():
    a=quality.light();a.output=OUTPUT;a.repo=REPO
    a.launch_identity=dict(stage='recovery166501+zoom factorial+PRC2 snapshot v1',no_test=True,
        phases=list(PHASES),parent_job='166501',time_limit='1-12:00:00',
        baseline_policy='reuse both trained seed43 checkpoints; resume original evaluation, no baseline training',
        zoom='Clean/LR2 only, same B1 windows; B1/ASF x raw/frozenQ; EDSR on crops; unchanged replace fusion',
        prc2='new snapshot head/root; preserve GT before DB target mutation; all thresholds and guards unchanged',
        old_launches=dict(quality=quality.light().launch_identity,research=research.light().launch_identity),
        source={name:sha256(REPO/name) for name in FILES})
    return a


def old_context(a):
    old=research.light()
    require(read_json(old.output/'submission/launch.json')==old.launch_identity==a.launch_identity['old_launches']['research'],
            'Original research source changed')
    ctx=research.verified(old)
    require(ctx.trial_identity==read_json(ctx.output/'identity.json'),'Original research identity changed')
    return ctx


def verified(a):
    q=quality.light()
    require(read_json(q.output/'submission/launch.json')==q.launch_identity==a.launch_identity['old_launches']['quality'],
            'Completed quality source changed')
    ctx=quality.verified(q)
    complete=read_json(q.output/'results/complete.json')
    original=read_json(q.output/'identity.json')
    require(complete['identity']==original==ctx.trial_identity and complete['no_test'] and complete['no_training'],
            'Quality reference identity differs')
    for file,key in [('quality_combo_report.md','report_sha256'),('quality_combo_summary.json','summary_sha256'),
                     ('records_sha256.json','records_sha256')]:
        require(sha256(q.output/'results'/file)==complete[key],'Completed scorer artifact changed')
    old=old_context(a);seeds={}
    for arm in ('B1_s43','BClean_s43'):
        p=old.output/'train'/arm/'trained.json';t=read_json(p)
        require(t['identity']==old.trial_identity and t['optimizer_updates']==10000 and t['micro_iterations']==20000,
                'Incomplete seed43 training')
        for checkpoint,digest in t['checkpoints'].items():
            require(sha256(Path(checkpoint))==digest,'Saved seed43 checkpoint changed')
        seeds[arm]=sha256(p)
    zoom=read_json(old.output/'zoom/B1/complete.json')
    require(zoom['tasks']==7524 and zoom['records_sha256']==sha256(old.output/'zoom/B1/records.json'),
            'Incomplete original B1 zoom study')
    ctx.repo=REPO;ctx.output=OUTPUT;ctx.research_reference=old.output
    ctx.quality_reference=q.output;ctx.quality_identity_sha=identity(original)
    ctx.trial_identity=dict(launch=a.launch_identity,quality_completion_sha256=sha256(q.output/'results/complete.json'),
        old_research_identity_sha256=identity(old.trial_identity),seed_receipts=seeds,
        zoom_receipt_sha256=sha256(old.output/'zoom/B1/complete.json'),
        b1_checkpoint_sha256=ctx.b1_digest,asf_checkpoint_sha256=ctx.asf_digest,
        edsr_checkpoint_sha256=sha256(ctx.edsr_weight),validation_sha256=old.trial_identity['validation_sha256'])
    return ctx


def snapshot_context(a):
    base=old_context(a);ctx=copy.copy(base);ctx.output=OUTPUT/'prc2_snapshot'
    ctx.trial_identity=dict(parent=base.trial_identity,recovery=a.launch_identity,
        correction='GT snapshot before DBModuleLoss target mutation; all PRC2 constants unchanged')
    return ctx


def snapshot_phase(a,phase):
    from unittest.mock import patch
    from rev4 import bundle_training as training
    ctx=snapshot_context(a);signature=ctx.output/'identity.json'
    if signature.exists():require(read_json(signature)==ctx.trial_identity,'Snapshot run identity changed')
    else:save_json(signature,ctx.trial_identity)
    if phase=='prc2_snapshot_evaluate':
        if (ctx.output/'stopped.json').exists():
            print('PRC2 snapshot stopped by unchanged scientific gate; evaluation not applicable',flush=True);return
        training.evaluate(ctx,'PRC2');return
    original=training.configuration
    def configuration(context,arm,resume):
        cfg=original(context,arm,resume)
        require(arm=='PRC2','Only PRC2 is trained in recovery')
        cfg.custom_imports['imports']=list(cfg.custom_imports['imports'])+['rev4.prc2_snapshot']
        cfg.model.det_head.type='Rev4PRC2SnapshotDBHead'
        return cfg
    try:
        with patch.object(training,'configuration',side_effect=configuration):training.fit(ctx,'PRC2')
    except ValueError as exc:
        message=str(exc)
        if not re.fullmatch(r'PRC2 active in only \d+/\d+ early microbatches; stopping',message):raise
        save_json(ctx.output/'stopped.json',dict(identity=ctx.trial_identity,outcome='STOPPED_LOW_ACTIVE',
            reason=message,thresholds_relaxed=False,no_test=True))
        print('SCIENTIFIC STOP:',message,flush=True)


def run_phase(a,phase):
    ctx=verified(a)
    if phase in ('zoom_probe','zoom_quality'):
        from rev4.zoom_quality import run
        run(ctx,probe_only=phase=='zoom_probe')
    elif phase in ('B1_s43','BClean_s43'):
        from rev4.bundle_training import evaluate
        evaluate(old_context(a),phase)
    elif phase=='zoom_BClean':
        from rev4.zoom_study import run
        run(old_context(a),'B-Clean')
    elif phase.startswith('prc2_snapshot_'):snapshot_phase(a,phase)
    else:report(a,ctx)


def report(a,ctx):
    old=old_context(a)
    from rev4.bundle_report import training_section
    baselines=training_section(old)
    snapshots=snapshot_context(a);stop=snapshots.output/'stopped.json'
    if stop.exists():prc=dict(status='STOPPED_LOW_ACTIVE',receipt=read_json(stop))
    else:
        p=snapshots.output/'train/PRC2/evaluation_summary.json'
        require(p.is_file(),'Snapshot PRC2 evaluation is missing')
        prc=dict(status='COMPLETE',evaluation=read_json(p))
    zoom=read_json(OUTPUT/'zoom_quality/results/complete.json')
    require(zoom['identity']==ctx.trial_identity and zoom['report_sha256']==sha256(OUTPUT/'zoom_quality/results/report.md'),
            'Zoom quality completion invalid')
    bc=read_json(old.output/'zoom/B-Clean/complete.json')
    require(bc['tasks']==4524 and bc['records_sha256']==sha256(old.output/'zoom/B-Clean/records.json'),
            'B-Clean zoom incomplete')
    require(all(baselines[k]['status']=='COMPLETE' for k in ('B1_s43','BClean_s43')),'Baseline evaluations incomplete')
    root=OUTPUT/'results'
    save_json(root/'recovery_summary.json',dict(identity=ctx.trial_identity,baseline_evaluations=baselines,
        prc2_snapshot=prc,old_prc2_status='STOPPED:7/200; preserve original failure',zoom_quality_complete=zoom,
        quality_reference=str(ctx.quality_reference/'results/quality_combo_report.md'),no_test=True))
    lines=['# Monkey recovery166501 and fixed-window zoom+EDSR x ASF x Q','',
        'Original completed quality results and both trained seed43 baselines were reused. No TEST.',
        'Original job ended during B1 early-checkpoint evaluation; final termination cause was not established.',
        'DB target mutation is independently reproduced as a cause of the PRC2 mask failure, not proof of the whole-job termination cause.','',
        '## Seed43 baselines','', '| Model | Selected updates | Clean H at .40/.60 | LR2 H at .40/.60 |',
        '|---|---:|---:|---:|']
    for arm in ('B1_s43','BClean_s43'):
        s=baselines[arm]['summary'];fixed=s['fixed']['primary']
        lines.append(f'| {arm} | {s["native_selection"]["selected_updates"]} | {fixed["clean"]["hmean"]:.2%} | {fixed["lr2"]["hmean"]:.2%} |')
    lines+=['',f'Corrected PRC2 outcome: **{prc["status"]}**. Original2 thresholds and the200-microbatch gate are unchanged.',
        'Corrected teacher supervision uses a deep copy of aligned GT before the student detection loss mutates polygons.',
        'See `prc2_snapshot/train/PRC2/evaluation_summary.json` or `prc2_snapshot/stopped.json`.','',
        '## Zoom factorial','',
        'See `zoom_quality/results/report.md` and `summary.json` for all four zoom arms, paired CIs and whole-image controls.',
        'Both detectors use the same B1-selected crop windows, EDSR only on crops and unchanged replace fusion.',
        'Q uses originating crop P-map features before coordinate mapping; no scorer fitting or threshold search.',
        'B-Clean original zoom records are complete under the original research root.','',
        'Development data reused, limited seeds, source CIs and multiple exploratory endpoints remain limitations.','']
    write_text(root/'recovery_report.md','\n'.join(lines))
    save_json(root/'complete.json',dict(identity=ctx.trial_identity,no_test=True,
        report_sha256=sha256(root/'recovery_report.md'),summary_sha256=sha256(root/'recovery_summary.json')))


def child(a,phase):
    log=OUTPUT/'logs'/f'{phase}_{os.environ.get("SLURM_JOB_ID","unknown")}.log'
    print('START recovery phase',phase,'log',log,flush=True)
    with log.open('ab',buffering=0) as stream:
        proc=subprocess.Popen([str(a.python),'-u',str(REPO/'submit_monkey_recovery.py'),'--phase',phase],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                save_json(OUTPUT/'heartbeat.json',dict(phase=phase,pid=proc.pid,
                    utc=datetime.now(timezone.utc).isoformat(),log=str(log)))
                time.sleep(10)
        finally:
            if proc.poll() is None:proc.terminate()
    print('END recovery phase',phase,'returncode',proc.returncode,flush=True)
    require(proc.returncode==0,f'Recovery phase {phase} failed ({proc.returncode}); inspect {log}')


def worker(a,args):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Require a Slurm allocation')
    require(read_json(OUTPUT/'submission/launch.json')==a.launch_identity,'Recovery code changed')
    ctx=verified(a);p=OUTPUT/'identity.json'
    if p.exists():require(read_json(p)==ctx.trial_identity,'Recovery run identity changed')
    else:save_json(p,ctx.trial_identity)
    if args.phase:
        if args.phase!='zoom_probe':
            require(read_json(OUTPUT/'zoom_quality/preflight.json')['status']=='PASS','Zoom probe must PASS first')
        run_phase(a,args.phase);return
    def interruption(sig,frame):
        print('Received signal',sig,'; writing recovery interruption receipt',file=sys.stderr,flush=True)
        try:save_json(OUTPUT/f'signal_{os.environ["SLURM_JOB_ID"]}.json',dict(signal=sig,utc=datetime.now(timezone.utc).isoformat()))
        except BaseException:traceback.print_exc()
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,interruption)
    with (OUTPUT/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);storage_probe(OUTPUT)
        subprocess.run([str(a.python),'-m','unittest','tests.test_monkey_recovery','-v'],cwd=REPO,check=True)
        status_path=OUTPUT/'status.json';status=read_json(status_path) if status_path.exists() else {}
        for phase in PHASES:
            if status.get(phase)=='PASS' and phase!='report':continue
            status[phase]='RUNNING';save_json(status_path,status)
            try:child(a,phase)
            except BaseException:
                status[phase]='FAIL';save_json(status_path,status);raise
            status[phase]='PASS';save_json(status_path,status)


def command(a):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    return ['sbatch','--parsable','--job-name=monkey_v2_recovery_zoom','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=1-12:00:00','--signal=B:USR1@120','--chdir='+str(REPO),'--output='+str(OUTPUT/'logs/job_%j.stdout.log'),
        '--error='+str(OUTPUT/'logs/job_%j.stderr.log'),'--wrap',
        env+shlex.join([str(a.python),'-u',str(REPO/'submit_monkey_recovery.py'),'--worker'])]


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--plan-only',action='store_true');group.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    group.add_argument('--phase',choices=PHASES,help=argparse.SUPPRESS)
    args=parser.parse_args();a=light()
    if args.worker or args.phase:
        faulthandler.enable(all_threads=True)
        try:worker(a,args)
        except BaseException:
            error=traceback.format_exc();print(error,file=sys.stderr,flush=True)
            try:write_text(OUTPUT/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}_{args.phase or "worker"}.txt',error)
            except BaseException:traceback.print_exc()
            raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login')
    if args.plan_only:
        print(json.dumps(dict(phases=PHASES,output=str(OUTPUT),sbatch=command(a),checks='inside allocation; fail before full on bad probe'),indent=2));return
    with run_lock(OUTPUT/'submission'):
        launch=OUTPUT/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'Existing recovery source changed')
        if (OUTPUT/'results/complete.json').exists():print('Already complete',OUTPUT/'results');return
        ticket=OUTPUT/'submission/latest_job.json'
        if ticket.exists():
            old=read_json(ticket)['job_id']
            if old in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',old);return
        (OUTPUT/'logs').mkdir(parents=True,exist_ok=True);storage_probe(OUTPUT);save_json(launch,a.launch_identity)
        job=subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        save_json(ticket,dict(job_id=job,parent_job='166501',identity=a.launch_identity))
        print('Submitted ONE recovery+zoom job',job,'\nReport:',OUTPUT/'results/recovery_report.md',flush=True)


if __name__=='__main__':main()
