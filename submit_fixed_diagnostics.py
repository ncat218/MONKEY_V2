#!/usr/bin/env python3
"""Submit fixed .40/.60 inference diagnostics; no threshold/model selection."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from lowres.common import identity as fingerprint, read_json, run_lock, save_json, sha256
from rev4 import require
from rev4.adaptation_jobs import REPO
import submit_decision_recovery as parent

RUN=REPO/'.runtime/rev4_fixed_diagnostics/v1'


def context():
    a=parent.context()
    previous_identity=a.recovery_identity
    completion=parent.RUN/'results/complete.json'
    receipt=read_json(completion)
    require(receipt['identity']==fingerprint(previous_identity) and
            receipt['report_sha256']==sha256(parent.RUN/'results/decision_audit_report.md'),
            'Require completed, unchanged job 63985 diagnostics')
    a.output=RUN
    a.recovery_identity=dict(stage='fixed postprocess residual diagnostics v1',parent_job='63985',
        parent_run_identity=previous_identity,completion_sha256=sha256(completion),
        pair=[.40,.60],models=['B-Clean','B1'],training=False,no_test=True,threshold_sweep=False,
        diagnostic_kernel_threshold=.40,weak_kernel_fraction=.30,
        source={name:sha256(REPO/name) for name in
            ('rev4/fixed_diagnostic_core.py','rev4/fixed_diagnostics.py','rev4/fixed_review.py','submit_fixed_diagnostics.py',
             'tests/test_fixed_diagnostics.py')})
    return a


def sbatch_args(a):
    command=parent.sbatch_args(a)
    for i,value in enumerate(command):
        if value.startswith('--job-name='):
            command[i]='--job-name=monkey_v2_fixed_diagnosis'
        elif value.startswith('--time='):
            command[i]='--time=03:00:00'
        elif value.startswith('--output='):
            command[i]='--output='+str(RUN/'logs/job_%j.stdout.log')
        elif value.startswith('--error='):
            command[i]='--error='+str(RUN/'logs/job_%j.stderr.log')
    command[-1]=command[-1].replace(str(REPO/'submit_decision_recovery.py'),str(Path(__file__).resolve()))
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--plan-only',action='store_true')
    action.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    a=context()
    if args.plan_only:
        from rev4.calibration import validation_cohort
        from rev4.decision_audit import installed_postprocessor, model_contexts
        from rev4.decision_plan import grid
        from rev4.decision_recovery import load_counts
        rows,tasks=validation_cohort(a,a.calibration_prepared)
        require(len(rows)==2262 and len(tasks)==4524,'Unexpected cohort size')
        installed_postprocessor()
        model_contexts(a)
        for name in ('B-Clean','B1'):
            load_counts(a.recovery_source/f'{name}_source_counts.npz',sorted(rows),grid())
        print(json.dumps(dict(stage=a.recovery_identity['stage'],sources=len(rows),
            tasks_per_model=len(tasks),models=['B-Clean','B1'],fixed_mask=.40,fixed_score=.60,
            output=str(RUN),no_test=True,training=False,threshold_sweep=False,
            historical_count_references='PASS',sbatch=sbatch_args(a)),indent=2),flush=True)
    elif args.worker:
        import fcntl
        require(os.environ.get('SLURM_JOB_ID'),'Worker requires Slurm GPU allocation')
        require(read_json(RUN/'submission/launch.json')['identity']==a.recovery_identity,
                'Fixed diagnostic source/input changed after submit')
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(600,repeat=True)
        with (RUN/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                from rev4.fixed_diagnostics import run
                run(a)
            except BaseException:
                failure=traceback.format_exc()
                print(failure,file=sys.stderr,flush=True)
                (RUN/f'failure_{os.environ["SLURM_JOB_ID"]}.txt').write_text(failure,encoding='utf-8')
                raise
            finally:
                faulthandler.cancel_dump_traceback_later()
    else:
        require(sys.platform=='linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login node')
        with run_lock(RUN/'submission'):
            ticket=RUN/'submission/launch.json'
            if ticket.is_file():
                require(read_json(ticket)['identity']==a.recovery_identity,'Existing diagnostic source changed; stop')
            if (RUN/'results/complete.json').is_file():
                print('Already complete:',RUN/'results/fixed_diagnostics_report.md')
                return
            receipt=RUN/'submission/latest_job.json'
            if receipt.is_file():
                job=read_json(receipt)['job_id']
                queue=subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split()
                if job in queue:
                    print('Already queued/running:',job)
                    return
            (RUN/'logs').mkdir(parents=True,exist_ok=True)
            save_json(ticket,dict(identity=a.recovery_identity))
            job=subprocess.check_output(sbatch_args(a),text=True).strip().split(';')[0]
            require(job.isdecimal(),f'Unexpected sbatch response: {job}')
            save_json(receipt,dict(job_id=job,identity=a.recovery_identity))
            print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/fixed_diagnostics_report.md"}',flush=True)


if __name__=='__main__':
    main()
