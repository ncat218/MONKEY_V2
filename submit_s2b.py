#!/usr/bin/env python3
"""User-operated UIT-HPC launcher for the locked descriptive Rev.4 S2-B branch."""
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
from rev4 import stage_a, stage_b
import submit_s2a
from run_all import worker_environment

REPO = Path(__file__).resolve().parent
RUN = REPO/'.runtime/rev4_task2b/run'
S2A_JOB_ID = '11907'
S2A_SUMMARY_SHA256 = '1286efe8b926f26ded452ffcc7d22bc6485575360cdb9e67cf096f9e67caf8ae'


def check_descriptive_branch(summary, expected_ids):
    require(summary['stage'] == 'S2-A' and summary['source_count'] == 500 and
            summary['s2_ids_sha256'] == expected_ids, 'S2-A summary has a different cohort')
    gates = summary['gates']
    require(set(gates) == {'G1', 'G2', 'G3'} and all(gates[gate] is True for gate in ('G1', 'G2', 'G3')),
            'Descriptive S2-B requires LR2 to pass G1, G2 and G3')
    require(summary['next_branch'] == 'LR2 selected. Blur/JPEG remains descriptive',
            'Unexpected registered S2-A branch decision')


def verified_inputs():
    a = submit_s2a.verified_inputs()
    a.s2a_run = a.output
    a.output = RUN
    summary_path = a.s2a_run/'results/s2a_summary.json'
    submit_s2a.check_hash(summary_path, S2A_SUMMARY_SHA256)
    a.s2a_summary = read_json(summary_path)
    check_descriptive_branch(a.s2a_summary, a.lock['s2_ids_sha256'])
    submitted = read_json(a.s2a_run/'submission/latest_job.json')
    require(submitted['job_id'] == S2A_JOB_ID and submitted['s2_ids_sha256'] == a.lock['s2_ids_sha256'],
            'Expected the verified S2-A job receipt')
    identity_path = a.s2a_run/'results/inference_identity.json'
    identity = read_json(identity_path)
    for path, key in (
        (a.lock_path, 'lock_sha256'),
        (Path(stage_a.__file__), 'worker_sha256'),
        (Path(submit_s2a.__file__), 'launcher_sha256'),
        (a.s2a_run/'prepared/prepared.json', 'prepared_sha256'),
    ):
        submit_s2a.check_hash(path, identity[key])
    require(identity['runtime'] == a.s1_identity['runtime'] and identity['gpu'] == a.s1_identity['gpu'],
            'S2-A inference identity differs from the audited B0 environment')
    s1_receipt = read_json(a.s2a_run/'results/s1_clean_identity.json')
    probe = read_json(a.s2a_run/'results/pmap_equivalence.json')
    require(s1_receipt == dict(sources=500, counts_and_matching_equal=True) and
            probe['cohort'] == 'S1' and probe['exact_polygons_and_scores_equal'] is True,
            'Require S2-A clean identity and P-map probe')
    a.s2a_summary_sha256 = S2A_SUMMARY_SHA256
    a.s2a_identity_sha256 = sha256(identity_path)
    a.s2a_job_id = S2A_JOB_ID
    a.blur_generator = submit_s2a.BLUR_GENERATOR
    a.source_list = submit_s2a.AUDIT/'S2_sources.txt'
    a.s2b_launcher = Path(__file__).resolve()
    return a


def stage_command(a, name):
    return [str(a.python), '-u', str(Path(__file__).resolve()), name]


def sbatch_args(a):
    lib = shlex.quote(str(a.cuda/'lib'))
    command = 'export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=' + lib + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    command += shlex.join(stage_command(a, '--worker'))
    return ['sbatch', '--parsable', '--job-name=monkey_v2_rev4_s2b', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=08:00:00', '--chdir='+str(REPO),
            '--output='+str(RUN/'logs/job_%j.stdout.log'), '--error='+str(RUN/'logs/job_%j.stderr.log'),
            '--wrap', command]


def submit(a):
    with run_lock(RUN/'submission'):
        if (RUN/'results/s2b_summary.json').is_file():
            print('TASK 2B complete:', RUN/'results/s2b_report.md')
            return
        receipt = RUN/'submission/latest_job.json'
        if receipt.is_file():
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            prior = read_json(receipt)['job_id']
            if prior in queued:
                print('TASK 2B already queued or running:', prior)
                return
        (RUN/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(sbatch_args(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, stage='S2-B descriptive', s2_ids_sha256=a.lock['s2_ids_sha256'],
                                stage_a_summary_sha256=a.s2a_summary_sha256))
        print(f'Submitted job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/s2b_report.md"}')


def worker(a):
    import fcntl

    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = dict(stage_a_job_id=a.s2a_job_id, stage_a_summary_sha256=a.s2a_summary_sha256,
                    stage_a_identity_sha256=a.s2a_identity_sha256,
                    s2_ids_sha256=a.lock['s2_ids_sha256'], selected_condition='lr2',
                    gates=a.s2a_summary['gates'], mode='descriptive')
        gate_path = RUN/'stage_b_gate.json'
        if gate_path.is_file():
            require(read_json(gate_path) == gate, 'Existing run has a different stage gate')
        else:
            save_json(gate_path, gate)
        if not (RUN/'prepared/prepared.json').is_file():
            env = worker_environment(a)
            env['PYTHONPATH'] = str(a.s1/'pillow-11.3.0')
            subprocess.run(stage_command(a, '--prepare'), env=env, check=True)
        stage_b.infer(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true', help='Verify the branch gate and print the job')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--prepare', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = verified_inputs()
    if args.plan_only:
        print(json.dumps(dict(stage='S2-B descriptive', selected_condition='lr2', g3='NOT ASSESSED',
                              conditions=tuple(stage_b.CONDITIONS), sources=len(a.s2_ids),
                              output=str(RUN), sbatch=sbatch_args(a)), indent=2))
    elif args.worker or args.prepare:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Worker stages require a Slurm allocation')
        if args.prepare:
            stage_b.prepare(a)
        else:
            worker(a)
    elif sys.platform != 'linux' or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit TASK 2B from the UIT-HPC login node')
    else:
        submit(a)


if __name__ == '__main__':
    main()
