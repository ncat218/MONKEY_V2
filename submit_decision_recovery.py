#!/usr/bin/env python3
"""Resume decision diagnostics from completed job 54308 sweep artifacts."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from lowres.common import read_json, run_lock, save_json, sha256
from rev4 import require
from rev4.adaptation_jobs import REPO
import submit_decision_audit as original

RUN = REPO/'.runtime/rev4_decision_audit/recovery_v1'
ORIGINAL_HASHES = {
    'B0_source_counts.npz':'4cd94be0b092d6d752b278d8758c9dca3bca749730f23214dca3815d976e71e0',
    'B-Clean_source_counts.npz':'6cfad71f328685240111b7a72636dafe296f42da1846e1306c891e97b84ef001',
    'B1_source_counts.npz':'70b5914b3f5adac01932d4224b8e55c4c342cec3a965bf15baa133f5685fca54',
    'results/pr_curves.csv':'93fb5f182e8554a234131c988c9dfaed97a3cf249a3a0ca607a39a3bba77e47d',
    'results/quad_filter_audit.csv':'0678985bc08e9965dd8a154da6f8ca10e33fbc1d60e097e98fc330e2b21b7957',
    'results/threshold_frontier.csv':'af3e28ec828169b824cb08111c6cd026909b66683608050aa014f6f2661a0962',
}


def context():
    a = original.verified_inputs()
    a.recovery_source, a.output = original.RUN, RUN
    for name,digest in ORIGINAL_HASHES.items():
        require(sha256(a.recovery_source/name) == digest, f'Job 54308 artifact changed: {name}')
    previous = read_json(a.recovery_source/'submission/latest_job.json')
    require(previous['job_id'] == '54308', 'Recovery requires original job 54308')
    source = read_json(a.recovery_source/'run_identity.json')
    require(source['source_code'] == original.source_identity(),
            'Original sweep implementation changed; cannot reuse counts without review')
    a.recovery_identity = dict(stage='decision review recovery v1', parent_job='54308',
        original=original.launch_identity(a), original_artifacts=ORIGINAL_HASHES,
        original_run_identity_sha256=sha256(a.recovery_source/'run_identity.json'),
        recovery_source={name:sha256(REPO/name) for name in
            ('rev4/decision_recovery.py','submit_decision_recovery.py','tests/test_decision_recovery.py')},
        no_test=True,training=False,full_sweep_repeated=False,chunk_tasks=64)
    return a


def sbatch_args(a):
    command = original.sbatch_args(a)
    for i,value in enumerate(command):
        if value.startswith('--job-name='):
            command[i] = '--job-name=monkey_v2_review_recovery'
        elif value.startswith('--time='):
            command[i] = '--time=02:00:00'
        elif value.startswith('--output='):
            command[i] = '--output='+str(RUN/'logs/job_%j.stdout.log')
        elif value.startswith('--error='):
            command[i] = '--error='+str(RUN/'logs/job_%j.stderr.log')
    command[-1] = command[-1].replace(str(REPO/'submit_decision_audit.py'),str(Path(__file__).resolve()))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    a = context()
    if args.plan_only:
        from rev4.decision_audit import installed_postprocessor, model_contexts
        from rev4.calibration import validation_cohort
        from rev4.decision_recovery import load_counts
        from rev4.decision_plan import grid

        rows,tasks = validation_cohort(a,a.calibration_prepared)
        require(len(rows) == 2262 and len(tasks) == 4524,'Unexpected cohort size')
        installed_postprocessor()
        model_contexts(a)
        for name in ('B0','B-Clean','B1'):
            load_counts(a.recovery_source/f'{name}_source_counts.npz',sorted(rows),grid())
        print(json.dumps(dict(identity=a.recovery_identity,sources=len(rows),tasks=len(tasks),
            output=str(RUN),reused_counts='PASS: all three model caches',sbatch=sbatch_args(a)),indent=2),flush=True)
    elif args.worker:
        import fcntl
        require(os.environ.get('SLURM_JOB_ID'),'Recovery worker requires Slurm allocation')
        require(read_json(RUN/'submission/launch.json')['identity'] == a.recovery_identity,
                'Recovery source/input identity changed after submit')
        faulthandler.enable(all_threads=True)
        faulthandler.dump_traceback_later(300,repeat=True)
        with (RUN/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                from rev4.decision_recovery import run
                run(a)
            except BaseException:
                failure = traceback.format_exc()
                print(failure,file=sys.stderr,flush=True)
                (RUN/f'failure_{os.environ["SLURM_JOB_ID"]}.txt').write_text(failure,encoding='utf-8')
                raise
            finally:
                faulthandler.cancel_dump_traceback_later()
    else:
        require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'),'Submit from HPC login node')
        with run_lock(RUN/'submission'):
            ticket = RUN/'submission/launch.json'
            if ticket.is_file():
                require(read_json(ticket)['identity'] == a.recovery_identity,'Recovery identity changed; stop')
            if (RUN/'results/complete.json').is_file():
                print('Recovery already complete:',RUN/'results/decision_audit_report.md')
                return
            receipt = RUN/'submission/latest_job.json'
            if receipt.is_file():
                previous = read_json(receipt)['job_id']
                queue = subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split()
                if previous in queue:
                    print('Recovery already queued/running:',previous)
                    return
            (RUN/'logs').mkdir(parents=True,exist_ok=True)
            save_json(ticket,dict(identity=a.recovery_identity))
            job = subprocess.check_output(sbatch_args(a),text=True).strip().split(';')[0]
            require(job.isdecimal(),f'Unexpected sbatch response: {job}')
            save_json(receipt,dict(job_id=job,identity=a.recovery_identity))
            print(f'Submitted recovery job {job}\nLogs: {RUN/"logs"}\nReport: {RUN/"results/decision_audit_report.md"}',flush=True)


if __name__ == '__main__':
    main()
