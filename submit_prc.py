#!/usr/bin/env python3
"""User-operated H200 launcher for the single Rev.4 M/PRC trial."""
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
from rev4.prc_training import PRC
from rev4.training_data import RECIPE, TRAIN_SOURCES, VAL_SOURCES
from submit_b1 import REPO
import submit_b1_diagnostics
from submit_s2a import check_hash


RUN = REPO/'.runtime/rev4_prc/run'
DIAGNOSTICS_SHA256 = 'c7ae24063921a96a3736b6157994f7fdac05116b8b6646d23886530f85e0904d'
G4_SHA256 = '3bb77656a5c95fb9ee8db00a76eac3c4b10d5d722c7cc3488f03bc23976d0e50'


def verified_inputs():
    a = submit_b1_diagnostics.verified_inputs()
    diagnostics = a.diagnostics_run/'results/b1_cal_diagnostics.json'
    g4 = a.output.parent/'g4_review/evidence.json'
    check_hash(diagnostics, DIAGNOSTICS_SHA256)
    check_hash(g4, G4_SHA256)
    require(read_json(a.diagnostics_run/'submission/latest_job.json')['job_id'] == '12434',
            'Require the B1-cal diagnostics job 12434')
    a.diagnostics_sha256, a.g4_sha256 = DIAGNOSTICS_SHA256, G4_SHA256
    a.output = RUN
    return a


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    env = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
           '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command = env+shlex.join([str(a.python), '-u', str(Path(__file__).resolve()), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_prc', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=12:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        report = RUN/'prc_report.md'
        if (RUN/'prc_summary.json').is_file():
            print('M PRC complete:', report)
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('M PRC already queued or running:', prior)
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, diagnostics_job='12434', diagnostics_sha256=DIAGNOSTICS_SHA256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify inputs and print the M trial')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        print(json.dumps(dict(stage='M PRC', train_sources=TRAIN_SOURCES, validation_sources=VAL_SOURCES,
                              training=RECIPE, prc=PRC, sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        import fcntl
        from rev4.prc_training import run

        require(os.environ.get('SLURM_JOB_ID'), 'PRC worker requires a Slurm allocation')
        with (RUN/'worker.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
        submit(a)


if __name__ == '__main__':
    main()
