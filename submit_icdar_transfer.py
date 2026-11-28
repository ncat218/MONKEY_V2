#!/usr/bin/env python3
"""Verify exact reproduction dataset and submit frozen ICDAR2015 inference only."""
import argparse
import faulthandler
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
from types import SimpleNamespace

from lowres.common import identity, read_json, run_lock, sha256
from rev4 import require
from rev4.durable_io import save_json, storage_probe, write_text
from rev4.icdar_transfer import (RECIPE, DATA_HASHES, MODEL_HASHES, EDSR_SHA,
                                BASE_CONFIG_SHA, ASF_CONFIG_SHA, audit_data)

REPO = Path(__file__).resolve().parent
FILES = ('submit_icdar_transfer.py','rev4/icdar_transfer.py','tests/test_icdar_transfer.py','docs/icdar2015-transfer-v1.md')


def context(data=None):
    workspace = REPO.parent/'monkeyocr_workspace'
    stack = workspace/'environment_and_models__phase1_verified_stack'
    a = SimpleNamespace(repo=REPO, stack=stack, python=stack/'python311_venv/bin/python',
        data=Path(data).resolve() if data else workspace/'ICDAR2015__Monkey_E0_Aprime_FP32_transfer/inputs',
        output=REPO/'.runtime/icdar2015_transfer/v1',
        edsr_weight=REPO/'.runtime/lr2_restoration/weights/edsr64_x2-torchsr-v1.0.3.pt')
    a.configs = dict(base=REPO/'.runtime/fusion_study/recovery_v2/evaluation/B1/config.py',
                     ASF=REPO/'.runtime/fusion_study/recovery_v2/evaluation/asf/config.py')
    a.weights = {'B0':stack/'upstream_MonkeyOCRv2/detection/mmocr/mmocr/model_weight/dbnet_mkv2vitae_icdar2015.pth',
                 'B-Clean':REPO/'.runtime/bclean/v1/weights/iter_20000.pth',
                 'B1':REPO/'.runtime/rev4_b1/run/weights/iter_20000.pth',
                 'Nearest':REPO/'.runtime/fusion_study/v1/arms/nearest/weights/iter_20000.pth',
                 'ASF':REPO/'.runtime/fusion_study/v1/arms/asf/weights/iter_20000.pth'}
    # Pin all Python helpers, including inherited capture/decode/SR code, without importing GPU libraries on login.
    sources = set(FILES)
    for folder in ('rev4','lowres','mixab'):
        sources.update(str(p.relative_to(REPO)).replace('\\','/') for p in (REPO/folder).rglob('*.py'))
    assets = read_json(REPO/'docs/handoff/artifacts/tri_quality_frozen/provenance.json')
    sources.update(assets['files'])
    a.launch = dict(recipe=RECIPE, dataset_path=str(a.data), no_training=True,
                    source={f:sha256(REPO/f) for f in sorted(sources)})
    return a


def verify(a):
    rows, audit = audit_data(a.data)
    for name, path in a.weights.items(): require(sha256(path) == MODEL_HASHES[name], 'Checkpoint changed: '+name)
    require(sha256(a.configs['base']) == BASE_CONFIG_SHA and sha256(a.configs['ASF']) == ASF_CONFIG_SHA,
            'Frozen inference config changed')
    require(sha256(a.edsr_weight) == EDSR_SHA, 'EDSR checkpoint changed')
    metric = a.stack/'upstream_MonkeyOCRv2/detection/mmocr/mmocr/mmocr/evaluation/metrics/hmean_iou_metric.py'
    a.signature = dict(launch=a.launch, data_audit=audit, model_sha256=MODEL_HASHES,
        config_sha256={k:sha256(p) for k,p in a.configs.items()}, edsr_sha256=EDSR_SHA,
        installed_metric_sha256=sha256(metric))
    a.digest = identity(a.signature)
    return audit


def command(a):
    cuda = a.stack/'cuda_toolkit_12_8_1__private_nvidia_conda_environment/lib'
    pillow = REPO/'.runtime/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0/pillow-11.3.0'
    env = 'export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(pillow))
    env += ' LD_LIBRARY_PATH='+shlex.quote(str(cuda))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    return ['sbatch','--parsable','--job-name=monkey_v2_icdar2015','--ntasks=1','--account=uit',
        '--partition=normal','--gres=gpu:2g.35gb:1','--qos=mig2-35','--cpus-per-task=8','--mem=40G',
        '--time=08:00:00','--chdir='+str(REPO),'--output='+str(a.output/'logs/job_%j.stdout.log'),
        '--error='+str(a.output/'logs/job_%j.stderr.log'),'--wrap',
        env+shlex.join([str(a.python),'-u',str(REPO/'submit_icdar_transfer.py'),'--worker','--data',str(a.data)])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--plan-only',action='store_true'); g.add_argument('--verify-only',action='store_true')
    g.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args(); a = context(args.data)
    if args.worker:
        import fcntl
        faulthandler.enable(all_threads=True)
        require(os.environ.get('SLURM_JOB_ID'), 'Require Slurm GPU allocation')
        require(read_json(a.output/'submission/launch.json') == a.launch, 'Code/plan changed since submission')
        try:
            with (a.output/'worker.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                verify(a)
                require(a.signature == read_json(a.output/'identity.json'), 'Input identity changed since submission')
                storage_probe(a.output)
                subprocess.run([str(a.python),'-m','unittest','tests.test_icdar_transfer','-v'],cwd=REPO,check=True)
                from rev4.icdar_transfer import run
                run(a)
        except BaseException:
            error=traceback.format_exc();write_text(a.output/f'failure_{os.environ.get("SLURM_JOB_ID")}.txt',error)
            raise
        return
    audit = verify(a)
    if args.plan_only or args.verify_only:
        print(json.dumps(dict(data_audit=audit, recipe=RECIPE, identity_sha256=a.digest,
                             output=str(a.output), sbatch=command(a) if args.plan_only else None),indent=2))
        return
    require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from HPC login')
    with run_lock(a.output/'submission'):
        if (a.output/'results/complete.json').exists():
            print('Already completed:',a.output/'results/icdar2015_report.md');return
        ticket = a.output/'submission/latest_job.json'
        if ticket.exists():
            prior = read_json(ticket)['job_id']
            if prior in subprocess.check_output(['squeue','--me','--noheader','--format=%i'],text=True).split():
                print('Already queued/running',prior);return
        prior = a.output/'identity.json'
        if prior.exists(): require(read_json(prior) == a.signature, 'Existing version differs; preserve it')
        storage_probe(a.output); (a.output/'logs').mkdir(parents=True,exist_ok=True)
        save_json(prior,a.signature);save_json(a.output/'submission/launch.json',a.launch)
        job = subprocess.check_output(command(a),text=True).strip().split(';')[0]
        require(job.isdecimal(),'Unexpected sbatch response')
        save_json(ticket,dict(job_id=job, identity_sha256=a.digest, no_training=True))
        print('Submitted ICDAR2015 inference-only job',job,'\nReport:',a.output/'results/icdar2015_report.md',flush=True)


if __name__ == '__main__': main()
