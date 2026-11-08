#!/usr/bin/env python3
"""User-operated UIT-HPC launcher for locked Rev.4 TASK 2A."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lowres.common import read_json, run_lock, save_json, sha256
from rev4 import CONDITIONS, require
from run_all import options as historical_options, worker_environment

REPO = Path(__file__).resolve().parent
TASK1 = REPO/'.runtime/rev4_task1'
TASK2A = REPO/'.runtime/rev4_task2a'
EVIDENCE = TASK2A/'handoff'
RUN = TASK2A/'run'
AUDIT = TASK1/'recheck-20261108-120822'
DRY_RUN = TASK1/'generator-dryrun-20261108-125519'
LR_GENERATOR = TASK1/'handoff/generate_lr_only.py'
BLUR_GENERATOR = TASK1/'handoff/generate_blur_jpeg_s2_audited.py'
PROTOCOL = EVIDENCE/'CS431_Pilot_Protocol_v2.1_FINAL.md'
RECEIPT = EVIDENCE/'pre_s2_lock_receipt.md'
LOCK = EVIDENCE/'stage_a_lock.json'


def check_hash(path, expected):
    actual = sha256(path)
    require(actual == expected, f'Frozen SHA-256 mismatch for {path}: {actual}')


def verified_inputs():
    lock = read_json(LOCK)
    require(lock['status'] == 'PRE-S2 TECHNICAL GO' and '**PRE-S2 TECHNICAL GO**' in RECEIPT.read_text(encoding='utf-8'),
            'Require a technical GO receipt')
    a = historical_options([])
    a.s1 = a.output
    a.output = RUN
    a.lock = lock
    a.lock_path = LOCK
    a.lr_generator = LR_GENERATOR
    a.launcher_path = Path(__file__).resolve()
    for path, key in (
        (RECEIPT, 'receipt_sha256'), (PROTOCOL, 'rev4_protocol_sha256'),
        (AUDIT/'s2_audit.json', 'audit_json_sha256'), (AUDIT/'S2_source_ids.txt', 's2_ids_sha256'),
        (AUDIT/'S2_sources.txt', 's2_paths_sha256'),
        (LR_GENERATOR, 'lr_generator_sha256'), (BLUR_GENERATOR, 'blur_jpeg_generator_sha256'),
        (a.validation_manifest, 'validation_sha256'), (a.evaluation_protocol, 'evaluator_policy_sha256'),
        (REPO/'lowres/evaluate_shared.py', 'evaluator_code_sha256'),
        (a.model_config, 'config_sha256'), (a.checkpoint, 'checkpoint_sha256'),
        (a.s1/'frozen_v2_0/split_lock.json', 's1_split_sha256'),
        (a.s1/'results/clean_identity.json', 's1_identity_sha256'),
        (a.s1/'results/pmap_equivalence.json', 's1_pmap_probe_sha256'),
        (a.s1/'results/inference_identity.json', 's1_inference_identity_sha256'),
    ):
        check_hash(path, lock[key])
    a.s1_identity = read_json(a.s1/'results/inference_identity.json')
    for name, expected in a.s1_identity['source'].items():
        check_hash(REPO/'lowres'/name, expected)
    evidence = read_json(AUDIT/'s2_audit.json')
    require(not evidence['s1']['exif_not_1'] and not evidence['s2']['exif_not_1'] and
            not evidence['gt']['round_trip_failures'], 'Recheck EXIF/geometry did not pass')
    dry = read_json(DRY_RUN/'generator_dry_run.json')
    require(dry['status'] == 'PASS_GENERATOR_DRY_RUN_ONLY' and dry['audit_json_sha256'] == lock['audit_json_sha256'] and
            dry['sha256_on_hpc'] == {LR_GENERATOR.name: lock['lr_generator_sha256'],
                                     BLUR_GENERATOR.name: lock['blur_jpeg_generator_sha256']} and
            dry['blur_jpeg_rows'] == 8 and dry['lr_rows'] == 6 and dry['pillow'] == '11.3.0' and
            dry['blur_sigmas'] == ['1.88', '3.75'] and dry['jpeg_qualities'] == [30, 10],
            'Generator dry-run receipt does not match the frozen audit')
    require(set(dry['dry_run_source_ids']) <= set(evidence['s1_ids']) and len(set(dry['dry_run_source_ids'])) == 2,
            'Dry-run receipt must contain two distinct S1 sources')
    s1_files = {row[0]: (row[1], row[2]) for row in evidence['s1']['images']}
    require(dry['dry_run_source_sha256'] == dict(s1_files[source] for source in dry['dry_run_source_ids']),
            'Dry-run source hashes must match the audited S1 images')
    test_log = (DRY_RUN/'ssim_unit_tests.log').read_text(encoding='utf-8').splitlines()
    require(sum(line.endswith(' ... ok') for line in test_log) == 10 and
            any(line.startswith('Ran 10 tests ') for line in test_log) and
            test_log[-1].strip() == 'OK', 'Require 10 passing synthetic SSIM unit tests')
    require((DRY_RUN/'orientation_preview.png').is_file(), 'Missing dry-run orientation preview')
    require((a.s1/'pillow-11.3.0/PIL').is_dir(), 'Missing audited Pillow 11.3.0 overlay')
    a.s2_ids = (AUDIT/'S2_source_ids.txt').read_text(encoding='utf-8').splitlines()
    a.s2_paths = (AUDIT/'S2_sources.txt').read_text(encoding='utf-8').splitlines()
    a.s1_ids = read_json(a.s1/'frozen_v2_0/split_lock.json')['diagnostic']
    require(a.s2_ids == sorted(set(a.s2_ids)) == evidence['s2_ids'] and len(a.s2_ids) == len(a.s2_paths) == 500 and
            not set(a.s2_ids) & set(a.s1_ids) and a.s2_paths == evidence['s2_paths'],
            'Unexpected S1/S2 membership or ordering')
    return a


def stage_command(a, name):
    return [str(a.python), '-u', str(Path(__file__).resolve()), name]


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    command = 'export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=' + lib + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    command += shlex.join(stage_command(a, '--worker'))
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_s2a', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=08:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        if (RUN/'results/s2a_summary.json').is_file():
            print('TASK 2A complete:', RUN/'results/s2a_report.md')
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if read_json(receipt)['job_id'] in queued:
                print('TASK 2A already queued or running:', read_json(receipt)['job_id'])
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, stage='S2-A', s2_ids_sha256=a.lock['s2_ids_sha256']))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/s2a_report.md"}')


def worker(a):
    import fcntl
    import shutil

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = RUN/'pre_s2_snapshot'
        snapshot.mkdir(exist_ok=True)
        for path in (RECEIPT, PROTOCOL, LOCK, AUDIT/'S2_source_ids.txt', AUDIT/'S2_sources.txt',
                     AUDIT/'s2_audit.json', DRY_RUN/'generator_dry_run.json', DRY_RUN/'ssim_unit_tests.log',
                     DRY_RUN/'orientation_preview.png'):
            target = snapshot/path.name
            if target.exists():
                check_hash(target, sha256(path))
            else:
                temp = target.with_suffix(target.suffix+'.tmp')
                shutil.copyfile(path, temp)
                temp.replace(target)
        if not (RUN/'prepared/prepared.json').is_file():
            env = worker_environment(a)
            env['PYTHONPATH'] = str(a.s1/'pillow-11.3.0')
            subprocess.run(stage_command(a, '--prepare'), env=env, check=True)
        from rev4.stage_a import infer
        infer(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify the GO and print the planned job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--prepare', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        print(json.dumps(dict(status='PRE-S2 TECHNICAL GO', stage='S2-A', conditions=CONDITIONS,
                              sources=len(a.s2_ids), output=str(RUN), sbatch=sbatch_args(a)), indent=2))
    elif args.worker or args.prepare:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Worker stages require a Slurm allocation')
        if args.prepare:
            from rev4.stage_a import prepare
            prepare(a)
        else:
            worker(a)
    elif sys.platform != 'linux' or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit TASK 2A from the UIT-HPC login node')
    else:
        submit(a)


if __name__ == '__main__':
    main()
