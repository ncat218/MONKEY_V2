#!/usr/bin/env python3
"""Submit the FROZEN v2.0 validation-only pilot to one UIT-HPC GPU."""
import argparse
import getpass
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from lowres.common import read_json, run_lock, save_json

REPO = Path(__file__).resolve().parent
RUN_NAME = 'MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0'
PHASE4 = 'phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_mixed_10000updates_final_weights_only_seed42_v2'
PHASE5 = 'phase5_FULL_R1_E0_original_vs_Aprime_FP32_job9793__inference_EDA_native_selected_seed42_v1'


def options(argv=None):
    datastore = Path('/datastore')/getpass.getuser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, default=datastore/'textvqa/Tuan/monkeyocr_workspace')
    parser.add_argument('--clean-root', type=Path,
                        default=datastore/'textvqa/data/benchmark_v1/raw/textvqa_clean/TextVQA')
    parser.add_argument('--runtime-root', type=Path, default=datastore/'_workspace_cs431_monkey/lowres')
    parser.add_argument('--plan-only', action='store_true', help='Print the plan without reading data or submitting')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--prepare', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.legacy_root = args.legacy_root.expanduser().resolve()
    args.clean_root = args.clean_root.expanduser().resolve()
    args.runtime_root = args.runtime_root.expanduser().resolve()
    args.output = args.runtime_root/RUN_NAME
    for source in (args.legacy_root, args.clean_root, REPO):
        if args.output.is_relative_to(source) or source.is_relative_to(args.output):
            raise ValueError('Use a runtime outside the code, legacy workspace and clean image roots')
    stack = args.legacy_root/'environment_and_models__phase1_verified_stack'
    args.python = stack/'python311_venv/bin/python'
    args.cuda = stack/'cuda_toolkit_12_8_1__private_nvidia_conda_environment'
    args.checkpoint = (stack/'upstream_MonkeyOCRv2/detection/mmocr/mmocr/model_weight'
                       /'dbnet_mkv2vitae_icdar2015.pth')
    phase4, phase5 = args.legacy_root/PHASE4, args.legacy_root/PHASE5
    args.model_config = phase5/'inputs/INFERENCE_CONFIG__identical_to_Aprime_job9793__only_checkpoint_differs.py'
    args.validation_manifest = (phase4/'inputs/03_frozen_detection_source_splits__official_train_intersection__70_15_15'
                                /'manifest__validation__textvqa_clean_degraded_same_textocr_word_gt.jsonl')
    args.evaluation_protocol = (phase4/'inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT'
                                /'evaluation_protocol.json')
    args.e0_validation_predictions = (phase5/'outputs/E0_ORIGINAL_CHECKPOINT__global_predictions/validation/mask_0p30'
                                      /'predictions__global__clean_and_degraded.jsonl')
    return args


def worker_command(args, stage):
    return [str(args.python), '-u', str(Path(__file__).resolve()), stage,
            '--legacy-root', str(args.legacy_root), '--clean-root', str(args.clean_root),
            '--runtime-root', str(args.runtime_root)]


def worker_environment(args):
    old = os.environ.get('LD_LIBRARY_PATH')
    lib = str(args.cuda/'lib')
    return {**os.environ, 'LD_LIBRARY_PATH': lib + (os.pathsep + old if old else ''),
            'PYTHONDONTWRITEBYTECODE': '1'}


def submission(args):
    logs = args.output/'logs'
    lib = shlex.quote(str(args.cuda/'lib'))
    command = 'export PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=' + lib + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    command += shlex.join(worker_command(args, '--worker'))
    return ['sbatch', '--parsable', '--job-name=monkey_v2_lowres_pilot', '--ntasks=1',
            '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
            '--cpus-per-task=8', '--mem=40G', '--time=08:00:00',
            '--output='+str(logs/'job_%j.stdout.log'), '--error='+str(logs/'job_%j.stderr.log'),
            '--wrap', command]


def submit(args):
    for path in (args.clean_root, args.python, args.cuda, args.checkpoint, args.model_config,
                 args.validation_manifest, args.evaluation_protocol, args.e0_validation_predictions):
        if not path.exists():
            raise FileNotFoundError(path)
    with run_lock(args.output/'submission'):
        if (args.output/'results/pilot_summary.json').exists():
            print('Pilot complete:', args.output/'results/pilot_summary.json')
            return
        receipt = args.output/'submission/latest_job.json'
        if receipt.is_file():
            prior = read_json(receipt)['job_id']
            queued = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                              '--format=%i'], text=True).split()
            if prior in queued:
                print('Pilot already queued or running:', prior)
                return
        (args.output/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(submission(args), text=True).strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError(f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job))
        print(f'Submitted job {job}\nLogs: {args.output/"logs"}\nResults: {args.output/"results"}')


def run_worker(args):
    import fcntl
    import tempfile

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.output/'frozen_v2_0/preflight.json').is_file():
            target = args.output/'pillow-11.3.0'
            if not target.is_dir():
                with tempfile.TemporaryDirectory(dir=args.output) as temporary:
                    staging = Path(temporary)/'pillow'
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                                    '--disable-pip-version-check',
                                    '--target', str(staging), 'Pillow==11.3.0'],
                                   env={**os.environ, 'PIP_USER': 'false'}, check=True)
                    staging.rename(target)
            env = {**os.environ, 'PYTHONPATH': str(target)}
            subprocess.run(worker_command(args, '--prepare'), env=env, check=True)
        from lowres.inference import pilot
        pilot(args)


def main(argv=None):
    args = options(argv)
    if args.plan_only:
        print(json.dumps(dict(output=str(args.output), validation=str(args.validation_manifest),
                              precision='FP32', sbatch=submission(args)), indent=2))
    elif args.worker or args.prepare:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Worker stages require a Slurm allocation')
        if args.prepare:
            from lowres.prepare import prepare
            def stop(_signum, _frame):
                raise InterruptedError('Data preparation interrupted')
            signal.signal(signal.SIGTERM, stop)
            print(prepare(args, args.output))
        else:
            run_worker(args)
    elif os.environ.get('SLURM_JOB_ID'):
        subprocess.run(worker_command(args, '--worker'), env=worker_environment(args), check=True)
    elif sys.platform != 'linux':
        raise RuntimeError('Submit on UIT-HPC Linux or use --plan-only')
    else:
        submit(args)


if __name__ == '__main__':
    main()
