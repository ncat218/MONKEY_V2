#!/usr/bin/env python3
"""User-operated CPU Slurm launcher for the lost FULL_R1 TRAIN manifest."""
import argparse
import getpass
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

from lowres.common import read_json, run_lock, save_json, sha256
from rev4 import require
from rev4.full_r1_generator import REQUIREMENTS
from rev4.training_data import TRAIN_NAME, TRAIN_SHA256
from submit_b1 import REPO, verified_context


def install_packages(a):
    destination = a.recovery/'packages'
    if destination.is_dir():
        return
    with TemporaryDirectory(prefix='.packages-', dir=a.recovery) as temporary:
        target = Path(temporary)/'packages'
        subprocess.run([str(a.python), '-m', 'pip', '--isolated', 'install', '--no-user', '--no-deps',
                        '--only-binary=:all:', '--disable-pip-version-check', '--target', str(target),
                        *REQUIREMENTS], check=True)
        target.rename(destination)


def sbatch_args(a):
    log = a.recovery/'logs'
    command = 'export PYTHONDONTWRITEBYTECODE=1; exec '+shlex.join(
        [str(a.python), '-u', str(Path(__file__).resolve()), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_recover_train', '--ntasks=1', '--account=uit',
            '--partition=normal', '--qos=mig2-35', '--cpus-per-task=8', '--mem=32G', '--time=08:00:00',
            '--chdir='+str(REPO), '--output='+str(log/'job_%j.stdout.log'),
            '--error='+str(log/'job_%j.stderr.log'), '--wrap', command]


def submit(a):
    with run_lock(a.recovery/'submission'):
        recovered = a.recovery/'verified'/TRAIN_NAME
        if recovered.is_file():
            require(sha256(recovered) == TRAIN_SHA256, 'Existing recovered TRAIN SHA-256 mismatch')
            print('TRAIN ready. Submit B1 with python submit_b1.py')
            return
        receipt = a.recovery/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('TRAIN recovery already queued or running:', prior)
                return
        install_packages(a)
        (a.recovery/'logs').mkdir(exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, expected_train_sha256=TRAIN_SHA256))
        print(f'Submitted recovery job {job}\nLogs: {a.recovery/"logs"}\nReceipt: {a.recovery/"verified/recovery.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_context()
    if args.worker:
        import fcntl

        require(os.environ.get('SLURM_JOB_ID'), 'Recovery worker requires a Slurm allocation')
        from rev4.train_recovery import recover

        with (a.recovery/'worker.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recover(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
        for key in ('clean_json', 'textocr_json'):
            require(a.raw_sources[key].is_file(), f'Missing original annotation {a.raw_sources[key]}')
        require((a.clean_root/'train_images').is_dir(), 'Missing original clean train images')
        submit(a)


if __name__ == '__main__':
    main()
