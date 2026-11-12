#!/usr/bin/env python3
"""Submit the paired LR2 restoration experiment on UIT-HPC."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lowres.common import read_json, run_lock, save_json, sha256
from rev4 import require, restoration
from rev4.calibration import validation_cohort
from submit_b1 import REPO
import submit_b1_diagnostics


RUN = REPO/'.runtime/lr2_restoration/run'
DIAGNOSTICS_RECORDS_SHA256 = 'c4b05bdc1d1fc8e86d842009a9e2427e37087b5420f7009cbbf65c0ff1ed777e'


def verified_inputs():
    a = submit_b1_diagnostics.verified_inputs()
    a.restoration_run = RUN
    a.restoration_weights = RUN.parent/'weights/edsr64_x2-torchsr-v1.0.3.pt'
    a.diagnostics_records = a.diagnostics_run/'paired_records.json'
    submit_b1_diagnostics.check_hash(a.diagnostics_records, DIAGNOSTICS_RECORDS_SHA256)
    return a


def download_weights(path):
    if path.is_file():
        require(sha256(path) == restoration.WEIGHT_SHA256, f'Unexpected EDSR checkpoint at {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    try:
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location', '--max-time', '90',
                        '--output', str(temporary), restoration.WEIGHT_URL], check=True)
        require(sha256(temporary) == restoration.WEIGHT_SHA256, 'Downloaded EDSR SHA-256 mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    command = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
               '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command += shlex.join([str(a.python), '-u', str(Path(__file__).resolve()), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_lr2_sr', '--ntasks=1', '--account=uit',
            '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35', '--cpus-per-task=8', '--mem=40G',
            '--time=04:00:00', '--chdir='+str(REPO), '--output='+str(RUN/'logs/job_%j.stdout.log'),
            '--error='+str(RUN/'logs/job_%j.stderr.log'), '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        report = RUN/'restoration_report.md'
        if (RUN/'restoration_summary.json').is_file():
            print('Restoration complete:', report)
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True)
            if prior in queued.split():
                print('Restoration already queued or running:', prior)
                return
        download_weights(a.restoration_weights)
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, calibration_job='12396', edsr_sha256=restoration.WEIGHT_SHA256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify inputs and print the job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        rows, _ = validation_cohort(a, a.calibration_prepared)
        print(json.dumps(dict(sources=len(rows), inputs=restoration.BRANCHES, edsr_sha256=restoration.WEIGHT_SHA256,
                              sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        import fcntl

        require(os.environ.get('SLURM_JOB_ID'), 'Restoration worker requires a Slurm allocation')
        require(sha256(a.restoration_weights) == restoration.WEIGHT_SHA256, 'EDSR checkpoint SHA-256 mismatch')
        with (RUN/'worker.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restoration.run(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
        submit(a)


if __name__ == '__main__':
    main()
