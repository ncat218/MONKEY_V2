#!/usr/bin/env python3
"""User-operated UIT-HPC launcher for Rev.4 B1 Clean + LR2."""
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
from rev4.training_data import LEGACY_COMMIT, RECIPE, TRAIN_NAME, TRAIN_SHA256
import submit_s2a
import submit_s2b

REPO = Path(__file__).resolve().parent
RUN = REPO/'.runtime/rev4_b1/run'
S2B_SUMMARY_SHA256 = '0ca4156736a4c96dca10c2cae276b29c9491b74ecaaf2bd022c17f9b569f5b51'


def verified_inputs(train_manifest=None, full_r1_root=None):
    a = submit_s2b.verified_inputs()
    a.s2b_run = a.output
    a.output = RUN
    s2b_summary = a.s2b_run/'results/s2b_summary.json'
    submit_s2a.check_hash(s2b_summary, S2B_SUMMARY_SHA256)
    b = read_json(s2b_summary)
    require(b['stage'] == 'S2-B' and b['mode'] == 'descriptive' and b['selected_condition'] == 'lr2' and
            b['g3'] == 'NOT ASSESSED' and b['s2_ids_sha256'] == a.lock['s2_ids_sha256'],
            'Require the locked LR2 branch after S2-B')
    a.s2b_summary_sha256 = S2B_SUMMARY_SHA256
    a.original_remaining_ids = read_json(a.s1/'frozen_v2_0/split_lock.json')['remaining_validation']
    raw = a.clean_root.parents[1]
    a.legacy_repo = a.legacy_root.parent/'MonKeyOCR-'
    a.full_r1_root = (full_r1_root.expanduser().resolve() if full_r1_root else raw/'full_textvqa_degraded')
    a.train_manifest = (train_manifest.expanduser().resolve() if train_manifest else
                        a.validation_manifest.with_name(TRAIN_NAME))
    a.raw_sources = dict(clean_json=a.clean_root/'TextVQA_0.5.1_train.json',
                         textocr_json=raw/'textocr_0_1_train_json/TextOCR_0.1_train.json',
                         clean_images_root=a.clean_root, degraded_root=a.full_r1_root,
                         main_image_manifest=a.full_r1_root/'main/image_manifest.csv')
    require(a.legacy_repo.is_dir(), f'Require historical source repository {a.legacy_repo}')
    revision = subprocess.check_output(['git', '-C', str(a.legacy_repo), 'rev-parse',
                                        f'{LEGACY_COMMIT}^{{commit}}'], text=True).strip()
    require(revision == LEGACY_COMMIT, 'Historical reconstruction revision mismatch')
    if train_manifest:
        require(a.train_manifest.is_file(), f'TRAIN file not found: {a.train_manifest}')
    if a.train_manifest.is_file():
        require(sha256(a.train_manifest) == TRAIN_SHA256,
                f'Frozen TRAIN SHA-256 mismatch at {a.train_manifest}. Expected {TRAIN_SHA256}')
    elif not (RUN/'prepared'/TRAIN_NAME).is_file():
        for source in (
            a.raw_sources['clean_json'], a.raw_sources['textocr_json'], a.raw_sources['main_image_manifest'],
            a.full_r1_root/'config/frozen_degradation_config.json',
            a.full_r1_root/'metadata/split_policy.json', a.full_r1_root/'metadata/locked_eval_manifest.csv',
        ):
            require(source.is_file(), f'Missing frozen TRAIN and reconstruction metadata: {source}. '
                    'Provide the original file with --train-manifest or a FULL_R1 metadata backup with '
                    f'--full-r1-root. Expected TRAIN SHA-256 {TRAIN_SHA256}.')
    return a


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    pillow = shlex.quote(str(a.s1/'pillow-11.3.0'))
    command = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+pillow+' LD_LIBRARY_PATH='+lib+
               '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    args = [str(a.python), '-u', str(Path(__file__).resolve()), '--worker', '--full-r1-root', str(a.full_r1_root)]
    if a.train_manifest.is_file():
        args += ['--train-manifest', str(a.train_manifest)]
    command += shlex.join(args)
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_b1', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=12:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        if (RUN/'b1_summary.json').is_file():
            print('B1 complete:', RUN/'b1_report.md')
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('B1 already queued or running:', prior)
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, stage='B1', train_sha256=TRAIN_SHA256,
                                stage_a_summary_sha256=a.s2a_summary_sha256,
                                stage_b_summary_sha256=a.s2b_summary_sha256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"b1_report.md"}')


def worker(a):
    import fcntl
    from rev4 import b1_training, training_data

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe_file = RUN/'locked_recipe.json'
        if recipe_file.is_file():
            require(read_json(recipe_file) == RECIPE, 'Existing B1 run has a different recipe')
        else:
            save_json(recipe_file, RECIPE)
        if not (RUN/'prepared/prepared.json').is_file():
            training_data.prepare(a)
        b1_training.run(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify frozen inputs and print the B1 job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--train-manifest', type=Path, help='Original TRAIN JSONL from any backup location')
    parser.add_argument('--full-r1-root', type=Path, help='FULL_R1 backup with config, metadata and main/image_manifest.csv')
    args = parser.parse_args()
    a = verified_inputs(args.train_manifest, args.full_r1_root)
    if args.plan_only:
        print(json.dumps(dict(stage='B1', selected_condition='lr2', train_sha256=TRAIN_SHA256,
                              train_sources=15_222, remaining_validation_sources=2_262,
                              output=str(RUN), recipe=RECIPE, sbatch=sbatch_args(a)), indent=2))
    elif args.worker:
        require(os.environ.get('SLURM_JOB_ID'), 'B1 worker requires a Slurm allocation')
        worker(a)
    elif sys.platform != 'linux' or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit B1 from the UIT-HPC login node')
    else:
        submit(a)


if __name__ == '__main__':
    main()
