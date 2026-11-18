#!/usr/bin/env python3
"""User-operated launcher for exploratory PR/quad and paired error diagnostics."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lowres.common import read_json, run_lock, save_json, sha256
from rev4 import require
from rev4.adaptation_jobs import REPO
from rev4.decision_plan import STAGE
import submit_b1_diagnostics


RUN = REPO/'.runtime/rev4_decision_audit/v1'
EDA_ROOT = REPO/'.runtime/paired_eda/v1'
SOURCE_FILES = ('rev4/decision_plan.py', 'rev4/decision_core.py', 'rev4/decision_review.py',
                'rev4/decision_audit.py', 'submit_decision_audit.py')


def source_identity():
    return {name: sha256(REPO/name) for name in SOURCE_FILES}


def verified_inputs():
    a = submit_b1_diagnostics.verified_inputs()
    a.output = RUN
    a.eda_root = EDA_ROOT
    a.decision_source_identity = source_identity()
    require(a.python.is_file(), f'Historical Python runtime missing: {a.python}')
    require((EDA_ROOT/'results/B0_BClean_B1/complete.json').is_file(),
            'Paired EDA B0/B-Clean/B1 output is missing')
    require((a.calibration_prepared/'validation_remaining.jsonl').is_file() and
            (a.calibration_prepared/'validation_tasks.jsonl').is_file(),
            'Locked remaining-validation manifests are missing')
    return a


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    env = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
           '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    args = [str(a.python), '-u', str(Path(__file__).resolve()), '--worker']
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_decision', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=08:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', env+shlex.join(args)]


def launch_identity(a):
    return dict(stage=STAGE, code=source_identity(),
        validation_sha256=sha256(a.calibration_prepared/'validation_remaining.jsonl'),
        tasks_sha256=sha256(a.calibration_prepared/'validation_tasks.jsonl'),
        evaluation_policy_sha256=sha256(a.evaluation_protocol),
        eda_complete_sha256=sha256(EDA_ROOT/'results/B0_BClean_B1/complete.json'))


def submit(a):
    identity = launch_identity(a)
    with run_lock(RUN/'submission'):
        ticket = RUN/'submission/launch.json'
        if ticket.is_file():
            require(read_json(ticket)['identity'] == identity,
                    'A prior run exists with a different identity; use a new versioned output path')
        complete = RUN/'results/complete.json'
        if complete.is_file():
            print('Decision audit already complete:', RUN/'results/decision_audit_report.md')
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('Decision audit already queued or running:', prior)
                return
        RUN.joinpath('logs').mkdir(parents=True, exist_ok=True)
        RUN.joinpath('submission').mkdir(parents=True, exist_ok=True)
        save_json(ticket, dict(identity=identity, status='SUBMITTED', no_test=True, training=False))
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, identity=identity, stage=STAGE))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/decision_audit_report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan-only', action='store_true', help='Verify inputs and print the Slurm plan')
    action.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    identity = launch_identity(a)
    if args.plan_only:
        from rev4.decision_audit import installed_postprocessor, model_contexts
        from rev4.calibration import validation_cohort
        rows, tasks = validation_cohort(a, a.calibration_prepared)
        require(len(rows) == 2262 and len(tasks) == 4524, 'Unexpected remaining-validation cohort size')
        models, _ = model_contexts(a)
        pp_path = installed_postprocessor()
        print(json.dumps(dict(stage=STAGE, sources=len(rows), tasks=len(tasks), models=['B0','B-Clean','B1'],
            geometry_settings=9, score_thresholds=8, grid_combinations=72, output=str(RUN),
            inputs={'eda':str(EDA_ROOT), 'validation_sha256':identity['validation_sha256'],
                    'evaluator_sha256':identity['evaluation_policy_sha256'],
                    'models':[{m['name']:m['checkpoint_sha256']} for m in models],
                    'db_postprocessor':str(pp_path)},
            limits={'training':False, 'test':False, 'protocol_changes':False, 'time':'08:00:00'},
            sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        import fcntl
        from rev4 import decision_audit

        require(os.environ.get('SLURM_JOB_ID'), 'Decision audit worker requires a Slurm allocation')
        require(read_json(RUN/'submission/launch.json')['identity'] == identity,
                'Submitted source/input identity changed before worker start')
        RUN.mkdir(parents=True, exist_ok=True)
        with (RUN/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            decision_audit.run(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'),
                'Submit from the UIT-HPC login node')
        submit(a)


if __name__ == '__main__':
    main()
