#!/usr/bin/env python3
"""User-operated Slurm launcher for B1-cal residual failure analysis."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lowres.common import read_json, run_lock, save_json
from rev4 import b1_diagnostics, calibration, require
from rev4.calibration import validation_cohort
from rev4.training_data import VAL_SOURCES
from submit_b1 import REPO
import submit_calibration
from submit_s2a import check_hash


RUN = REPO/'.runtime/rev4_b1_diagnostics/run'
CALIBRATION_JOB = '12396'
CALIBRATION_SHA256 = '208720b8f8aa9901cac54144e0c070108971d4bef1b49dba55487990169b92e3'
CALIBRATION_COUNTS_SHA256 = '6301415fa0120573c817ebfce4046fc563c30227c986e2d965ef81e80e91dab9'


def verified_inputs():
    a = submit_calibration.verified_inputs()
    receipt = read_json(a.calibration_run/'submission/latest_job.json')
    require(receipt['job_id'] == CALIBRATION_JOB and receipt['b1_job_id'] == submit_calibration.B1_JOB_ID and
            receipt['b1_summary_sha256'] == a.b1_summary_sha256, 'Require calibration job 12396 on B1 job 12187')
    summary = a.calibration_run/'results/calibration_summary.json'
    a.calibration_counts = a.calibration_run/'counts/B1.json'
    check_hash(summary, CALIBRATION_SHA256)
    check_hash(a.calibration_counts, CALIBRATION_COUNTS_SHA256)
    a.calibration_sha256, a.calibration_counts_sha256 = CALIBRATION_SHA256, CALIBRATION_COUNTS_SHA256
    a.calibration_summary = read_json(summary)
    locked = a.calibration_summary['identity']
    selected = a.calibration_summary['models']['B1']['calibrated']
    require(a.calibration_summary['stage'] == 'B0/B1 calibration' and a.calibration_summary['sources'] == VAL_SOURCES and
            a.calibration_summary['b1_native_reproduced'] is True and locked['b1_checkpoint_sha256'] == a.b1_digest and
            locked['evaluator_policy_sha256'] == a.lock['evaluator_policy_sha256'] and
            (selected['mask_thr'], selected['min_text_score']) == b1_diagnostics.THRESHOLDS,
            'Unexpected B1 calibration selection or identity')
    check_hash(Path(calibration.__file__), locked['calibration_code_sha256'])
    check_hash(a.calibration_prepared/'validation_remaining.jsonl', locked['validation_sha256'])
    check_hash(a.calibration_prepared/'validation_tasks.jsonl', locked['tasks_sha256'])
    a.diagnostics_run = RUN
    return a


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    command = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
               '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command += shlex.join([str(a.python), '-u', str(Path(__file__).resolve()), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_b1diag', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=04:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        report = RUN/'results/b1_cal_diagnostics.md'
        if report.with_suffix('.json').is_file():
            print('B1-cal diagnostics complete:', report)
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('B1-cal diagnostics already queued or running:', prior)
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, calibration_job=CALIBRATION_JOB, calibration_sha256=CALIBRATION_SHA256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify frozen inputs and print the diagnostic job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        rows, tasks = validation_cohort(a, a.calibration_prepared)
        print(json.dumps(dict(stage='B1-cal diagnostics', calibration_job=CALIBRATION_JOB, sources=len(rows),
                              tasks=len(tasks), thresholds=b1_diagnostics.THRESHOLDS, output=str(RUN),
                              sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        import fcntl

        require(os.environ.get('SLURM_JOB_ID'), 'Diagnostic worker requires a Slurm allocation')
        with (RUN/'worker.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            b1_diagnostics.run(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
        submit(a)


if __name__ == '__main__':
    main()
