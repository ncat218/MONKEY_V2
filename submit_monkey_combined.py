#!/usr/bin/env python3
"""One allocation for frozen quality scoring followed by the original research bundle.

This orchestration does not alter either study, launch identity, output directory or training recipe.
The original queued jobs are replaced operationally only after the new held submission is verified.
"""
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
import submit_quality_combo as quality
import submit_research_bundle as research

REPO=Path(__file__).resolve().parent
TIME_LIMIT='2-06:00:00'  # 12h + 40h child caps, with2h orchestration margin; QoS max72h.
STUDIES=('quality','research')
OLD_JOBS={'quality':'166132','research':'165276'}
OWN_FILES=('submit_monkey_combined.py','tests/test_monkey_combined.py','docs/monkey-combined-v1.md')


def children():
    return {'quality':quality.light(),'research':research.light()}


def light():
    a=quality.light()
    a.output=REPO/'.runtime/monkey_combined/v1'
    snapshots=children()
    a.launch_identity=dict(stage='one allocation: quality then research; unchanged studies',no_test=True,
        order=list(STUDIES),parent_job=quality.PARENT_JOB,time_limit=TIME_LIMIT,
        children={name:ctx.launch_identity for name,ctx in snapshots.items()},
        source={p:sha256(REPO/p) for p in OWN_FILES})
    return a


def verify_child_launches(a):
    snapshots=children()
    for name,ctx in snapshots.items():
        require(ctx.launch_identity==a.launch_identity['children'][name],name+' source changed')
        require(read_json(ctx.output/'submission/launch.json')==ctx.launch_identity,
                name+' original queued launch differs; do not merge changed studies')
    return snapshots


def command(a,dependency=None,hold=False):
    env='export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env+=' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    args=['sbatch','--parsable','--job-name=monkey_v2_combined','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time='+TIME_LIMIT,'--chdir='+str(REPO),'--output='+str(a.output/'logs/job_%j.stdout.log'),
        '--error='+str(a.output/'logs/job_%j.stderr.log')]
    if dependency:args.append('--dependency=afterok:'+dependency)
    if hold:args.append('--hold')
    return args+['--wrap',env+shlex.join([str(a.python),'-u',str(REPO/'submit_monkey_combined.py'),'--worker'])]


def completion(name,ctx):
    marker=ctx.output/'results/complete.json'
    require(marker.is_file(),name+' child exited without a completion receipt')
    value=read_json(marker)
    require(value['identity']==read_json(ctx.output/'identity.json') and value['no_test'],
            name+' child identity or TEST policy changed')
    report=ctx.output/'results'/('quality_combo_report.md' if name=='quality' else 'bundle_report.md')
    require(sha256(report)==value['report_sha256'],name+' report bytes changed')
    if name=='quality':
        summary=ctx.output/'results/quality_combo_summary.json'
        require(value['no_training'] and value['summary_sha256']==sha256(summary),
                'Quality scorer must remain inference-only')
    else:
        require(value['final'] is True,'Research bundle has incomplete/failed phases; preserve partial reports')
    return dict(completion_sha256=sha256(marker),report=str(report),report_sha256=sha256(report))


def run_studies(a,snapshots):
    """Child processes release all model/optimizer memory between the two studies."""
    status_path=a.output/'status.json'
    status=read_json(status_path) if status_path.exists() else {}
    results={}
    for name in STUDIES:
        ctx=snapshots[name]
        if status.get(name,{}).get('state')=='PASS':
            results[name]=completion(name,ctx);continue
        log=a.output/'logs'/f'{name}_{os.environ.get("SLURM_JOB_ID","unknown")}.log'
        status[name]=dict(state='RUNNING',log=str(log));save_json(status_path,status)
        launcher='submit_quality_combo.py' if name=='quality' else 'submit_research_bundle.py'
        print('START COMBINED CHILD',name,'log:',log,flush=True)
        try:
            with log.open('ab',buffering=0) as stream:
                result=subprocess.run([str(a.python),'-u',str(REPO/launcher),'--worker'],cwd=REPO,
                                      stdout=stream,stderr=subprocess.STDOUT)
            require(result.returncode==0,name+' child failed; next study will not run. See '+str(log))
            results[name]=completion(name,ctx)
        except BaseException:
            status[name].update(state='FAIL');save_json(status_path,status);raise
        status[name].update(state='PASS',**results[name]);save_json(status_path,status)
        print('END COMBINED CHILD',name,'PASS',flush=True)
    lines=['# Combined Monkey studies','',
        'One allocation; frozen quality scoring first, then the original research bundle.',
        'Study recipes, launch identities and output roots are unchanged. No TEST.','',
        '| Study | Status | Report |','|---|---|---|']
    for name,result in results.items():lines.append(f'| {name} | PASS | {result["report"]} |')
    output=a.output/'results'
    write_text(output/'combined_report.md','\n'.join(lines)+'\n')
    save_json(output/'complete.json',dict(identity=a.launch_identity,no_test=True,children=results,
        report_sha256=sha256(output/'combined_report.md')))


def worker(a):
    import fcntl
    require(os.environ.get('SLURM_JOB_ID'),'Combined workflow requires a Slurm allocation')
    require(read_json(a.output/'submission/launch.json')==a.launch_identity,'Combined source changed')
    snapshots=verify_child_launches(a)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage_probe(a.output)
        subprocess.run([str(a.python),'-m','unittest','tests.test_monkey_combined','-v'],cwd=REPO,check=True)
        run_studies(a,snapshots)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--plan-only',action='store_true');group.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--hold',action='store_true',help='Keep replacement held until original pending jobs are cancelled')
    args=parser.parse_args();a=light()
    if args.worker:
        faulthandler.enable(all_threads=True)
        try:worker(a)
        except BaseException:
            failure=traceback.format_exc();write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID","unknown")}.txt',failure)
            print(failure,file=sys.stderr,flush=True);raise
        return
    require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login')
    snapshots=verify_child_launches(a)
    dependency=quality.dependency()
    if args.plan_only:
        print(json.dumps(dict(time_limit=TIME_LIMIT,parent=dependency,order=STUDIES,
            outputs={k:str(v.output) for k,v in snapshots.items()},sbatch=command(a,dependency,args.hold)),indent=2));return
    with run_lock(a.output/'submission'):
        launch=a.output/'submission/launch.json'
        if launch.exists():require(read_json(launch)==a.launch_identity,'Existing combined source changed')
        if (a.output/'results/complete.json').exists():print('Already complete',a.output/'results');return
        ticket=a.output/'submission/latest_job.json'
        if ticket.exists():
            old=read_json(ticket)['job_id']
            if old in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',old);return
        (a.output/'logs').mkdir(parents=True,exist_ok=True);storage_probe(a.output);save_json(launch,a.launch_identity)
        job=subprocess.check_output(command(a,dependency,args.hold),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected Slurm response')
        save_json(ticket,dict(job_id=job,dependency=dependency,held_at_submission=args.hold,
            replaces=OLD_JOBS,identity=a.launch_identity))
        print('Submitted ONE combined job',job,'held:',args.hold,'dependency:',dependency or 'parent complete',flush=True)


if __name__=='__main__':main()
