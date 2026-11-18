#!/usr/bin/env python3
"""Prepare or submit the exploratory equal-budget LR2 feature-distillation pilot."""
import sys

if __name__ == '__main__':
    print('MONKEY_V2 feature pilot: starting launcher...', file=sys.stderr, flush=True)

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

from lowres.common import canonical, read_json, run_lock, save_json, sha256
from rev4 import require
from rev4.feature_plan import ARMS, PAIRS, PILOT


REPO = Path(__file__).resolve().parent
RUN = REPO/'.runtime/lr2_feature_distillation/pilot_v1'
CODE_FILES = ('submit_feature_pilot.py', 'rev4/feature_plan.py', 'rev4/feature_pilot.py', 'rev4/feature_distillation.py',
              'rev4/feature_distillation_plugins.py', 'rev4/prc_geometry.py', 'rev4/prc_plugins.py',
              'tests/test_feature_distillation.py', 'tests/test_feature_pilot.py',
              'docs/lr2-feature-distillation-pilot.md')


def launch_identity():
    return json.loads(canonical(dict(stage='exploratory feature pilot v1', recipe=PILOT,
                                    arms=ARMS, thresholds=PAIRS,
                                    code={path: sha256(REPO/path) for path in CODE_FILES})))


def launch_inputs():
    """Read only small repository files; no legacy imports/data/checkpoint reads."""
    stack = REPO.parent/'monkeyocr_workspace/environment_and_models__phase1_verified_stack'
    return SimpleNamespace(python=stack/'python311_venv/bin/python',
                           cuda=stack/'cuda_toolkit_12_8_1__private_nvidia_conda_environment',
                           s1=REPO/'.runtime/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0',
                           output=RUN, launch_identity=launch_identity())


def verified_inputs():
    # Full historical imports and SHA-256 audit belong inside the allocation.
    print('GPU worker: verifying frozen B1 inputs and historical SHA-256 chain...', flush=True)
    import submit_b1_diagnostics
    from rev4.training_data import TRAIN_NAME

    a = submit_b1_diagnostics.verified_inputs()
    a.output = RUN
    a.pilot_identity = json.loads(canonical(dict(stage='exploratory feature pilot v1', recipe=PILOT,
                    arms=ARMS, thresholds=PAIRS, b1_checkpoint_sha256=a.b1_digest,
                    b1_identity_sha256=a.b1_identity_sha256, calibration_sha256=a.calibration_sha256,
                    calibration_counts_sha256=a.calibration_counts_sha256,
                    evaluator_policy_sha256=a.lock['evaluator_policy_sha256'],
                    prepared_sha256=sha256(a.calibration_prepared/'prepared.json'),
                    train_sha256=sha256(a.calibration_prepared/TRAIN_NAME),
                    validation_sha256=sha256(a.calibration_prepared/'validation_remaining.jsonl'),
                    validation_tasks_sha256=sha256(a.calibration_prepared/'validation_tasks.jsonl'),
                    code={path: sha256(REPO/path) for path in CODE_FILES})))
    a.launch_identity = launch_identity()
    print('GPU worker: full input verification passed.', flush=True)
    return a


def sbatch_args(a):
    env = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))+
           ' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command = env+shlex.join([str(a.python), '-u', str(REPO/'submit_feature_pilot.py'), '--worker'])
    return ['sbatch', '--parsable', '--job-name=monkey_v2_lr2_fkd', '--ntasks=1', '--account=uit',
            '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35', '--cpus-per-task=8',
            '--mem=40G', '--time=06:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def lock_identity(a):
    target = RUN/'identity.json'
    if target.is_file():
        require(read_json(target) == a.pilot_identity, 'Pilot identity changed; stop and use a new versioned run')
    else:
        save_json(target, a.pilot_identity)


def submit(a):
    with run_lock(RUN/'submission'):
        launch = RUN/'submission/launch_identity.json'
        if launch.is_file():
            require(read_json(launch) == a.launch_identity, 'Submitted source changed; stop for version review')
        else:
            save_json(launch, a.launch_identity)
        completed = RUN/'results/pilot_summary.json'
        if completed.is_file():
            require(read_json(completed)['identity']['code'] == a.launch_identity['code'], 'Completed pilot source changed')
            print('Pilot already complete:', RUN/'results/pilot_report.md')
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            previous = read_json(receipt)
            queued = subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split()
            if previous['job_id'] in queued:
                print('Pilot already queued or running:', previous['job_id'])
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        require(a.python.is_file(), f'Historical Python executable is missing: {a.python}')
        print('Submitting Slurm job; full input/GPU checks run before training inside the allocation.', flush=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, launch_identity=a.launch_identity, input_verification='pending GPU worker'))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/pilot_report.md"}')


def child(a, *args):
    print('Starting phase:', ' '.join(args), flush=True)
    subprocess.run([str(a.python), '-u', str(REPO/'submit_feature_pilot.py'), *args], check=True, cwd=REPO)


def worker(a):
    import fcntl

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(read_json(RUN/'submission/launch_identity.json') == a.launch_identity,
                'Source/recipe changed after submission; stopping before GPU work')
        lock_identity(a)
        subprocess.run([str(a.python), '-m', 'unittest', 'tests.test_feature_distillation',
                        'tests.test_feature_pilot', '-v'], check=True, cwd=REPO)
        child(a, '--phase', 'probe')
        for arm in ARMS:
            child(a, '--phase', 'train', '--arm', arm)
            child(a, '--phase', 'evaluate', '--arm', arm)
        child(a, '--phase', 'report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Print a lightweight plan; full data audit is deferred to the GPU worker')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--phase', choices=('probe', 'train', 'evaluate', 'report'), help=argparse.SUPPRESS)
    parser.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not args.arm or args.phase in ('train', 'evaluate'), '--arm is only valid for train/evaluate')
    require(args.phase not in ('train', 'evaluate') or args.arm, 'train/evaluate require --arm')
    if args.plan_only:
        a = launch_inputs()
        print(json.dumps(dict(launch_identity=a.launch_identity, output=str(RUN),
                              input_verification='PENDING: full SHA-256 and GPU checks run in the allocation before training',
                              sbatch=sbatch_args(a)), indent=2))
    elif args.worker or args.phase:
        require(sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID'), 'Worker phases require a Slurm GPU allocation')
        a = verified_inputs()
        require(read_json(RUN/'submission/launch_identity.json') == a.launch_identity,
                'Source changed after submission; stopping before training')
        if args.worker:
            worker(a)
        else:
            from rev4 import feature_pilot

            require((RUN/'identity.json').is_file(), 'Missing submitted pilot identity')
            lock_identity(a)
            if args.phase != 'probe':
                receipt = read_json(RUN/'preflight.json')
                require(receipt['status'] == 'PASS' and receipt['identity'] == a.pilot_identity,
                        'GPU preflight must PASS before training/evaluation')
            if args.phase == 'probe':
                feature_pilot.probe(a)
            elif args.phase == 'train':
                feature_pilot.train(a, args.arm)
            elif args.phase == 'evaluate':
                feature_pilot.evaluate(a, args.arm)
            else:
                feature_pilot.summarize(a)
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
        submit(launch_inputs())


if __name__ == '__main__':
    main()
