#!/usr/bin/env python3
"""User-operated UIT-HPC launcher for B0/B1 remaining-validation calibration."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lowres.common import read_json, run_lock, save_json
from rev4 import require
from rev4.b1_training import validate_prepared
from rev4.calibration import GRID, validation_cohort
from rev4.training_data import VAL_SOURCES
import submit_b1
from submit_s2a import check_hash


RUN = submit_b1.REPO/'.runtime/rev4_calibration/run'
B1_JOB_ID = '12187'
B1_SUMMARY_SHA256 = 'a61093355d7177b94caa7c7ae7461873a358dc832e7aa51bb4d7e88dce4bf260'
B1_IDENTITY_SHA256 = 'ebfc09930328b1d0a49f20d000c5960f0fdaec44bf77cd3a9bac1d975a36a31b'


def verified_inputs():
    a = submit_b1.verified_context()
    a.calibration_run = RUN
    receipt = read_json(a.output/'submission/latest_job.json')
    require(receipt['job_id'] == B1_JOB_ID, 'Calibration requires B1 job 12187')
    summary_path, identity_path = a.output/'b1_summary.json', a.output/'b1_identity.json'
    check_hash(summary_path, B1_SUMMARY_SHA256)
    check_hash(identity_path, B1_IDENTITY_SHA256)
    a.b1_summary_sha256, a.b1_identity_sha256 = B1_SUMMARY_SHA256, B1_IDENTITY_SHA256
    a.b1_summary, a.b1_identity = read_json(summary_path), read_json(identity_path)
    selected = a.b1_summary['selected_checkpoint']
    a.b1_checkpoint = a.output/'weights/iter_20000.pth'
    require(a.b1_summary['stage'] == 'B1' and selected['sources'] == VAL_SOURCES and
            selected['identity']['updates'] == 10_000 and Path(selected['checkpoint']) == a.b1_checkpoint,
            'Expected the selected checkpoint from job 12187')
    a.b1_digest = selected['identity']['checkpoint_sha256']
    check_hash(a.b1_checkpoint, a.b1_digest)
    a.calibration_prepared = validate_prepared(a)
    check_hash(a.calibration_prepared/'prepared.json', a.b1_identity['prepared_sha256'])
    check_hash(a.calibration_prepared/'validation_remaining.jsonl', selected['identity']['validation_sha256'])
    check_hash(a.calibration_prepared/'validation_tasks.jsonl', selected['identity']['tasks_sha256'])
    return a


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    command = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
               '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command += shlex.join([str(a.python), '-u', str(Path(__file__).resolve()), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_cal', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=12:00:00', '--chdir='+str(submit_b1.REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        report = RUN/'results/calibration_report.md'
        if report.with_name('calibration_summary.json').is_file():
            print('Calibration complete:', report)
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('Calibration already queued or running:', prior)
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, b1_job_id=B1_JOB_ID, b1_summary_sha256=B1_SUMMARY_SHA256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify inputs and print the calibration job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        rows, tasks = validation_cohort(a, a.calibration_prepared)
        print(json.dumps(dict(stage='B0/B1 calibration', b1_job=B1_JOB_ID, sources=len(rows), tasks=len(tasks),
                              grid=GRID, output=str(RUN), sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        import fcntl
        from rev4.calibration import run

        require(os.environ.get('SLURM_JOB_ID'), 'Calibration worker requires a Slurm allocation')
        with (RUN/'worker.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'),
                'Submit calibration from the UIT-HPC login node')
        submit(a)


if __name__ == '__main__':
    main()
