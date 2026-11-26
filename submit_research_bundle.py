#!/usr/bin/env python3
"""ONE Slurm allocation: probes, query-free zoom study, seed replicates, PRC2 and one report.

Phases run as child processes inside the same allocation. An independent phase that fails is
recorded in status.json and the remaining phases continue; the technical probe must PASS first.
Whole-image EDSR / bicubic on fine-tuned B1 is job 163012 (submit_edsr_recheck.py) and is not repeated.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from lowres.common import read_json, run_lock, sha256
from rev4 import require
from rev4.bundle_plan import (BCLEAN_CHECKPOINT_SHA, EDSR_RECHECK_JOB, OUTPUT_NAME, PRC2, STRESS_VARIANTS, TIME_LIMIT,
                              TRAIN_ARMS, ZOOM, ZOOM_MODELS, ZOOM_RULES, ZOOM_VARIANTS)
from rev4.durable_io import save_json, storage_probe, write_text
import submit_module_trials as original

REPO = Path(__file__).resolve().parent
FILES = ['submit_research_bundle.py', 'rev4/bundle_plan.py', 'rev4/zoom_core.py', 'rev4/zoom_study.py',
         'rev4/prc2_plugins.py', 'rev4/bundle_training.py', 'rev4/bundle_report.py', 'tests/test_research_bundle.py',
         'docs/research-bundle-v1.md']
# Repository code the bundle executes (hashed so a change between submission and run stops the job).
REUSED = ['rev4/__init__.py', 'rev4/adaptation_plan.py', 'rev4/b1_diagnostics.py', 'rev4/b1_plugins.py',
          'rev4/b1_training.py', 'rev4/bclean_plugins.py', 'rev4/calibration.py', 'rev4/decision_core.py',
          'rev4/durable_io.py', 'rev4/edsr.py', 'rev4/fusion_evaluation.py', 'rev4/fusion_plan.py',
          'rev4/local_scratch.py', 'rev4/paired_eda_core.py', 'rev4/prc_geometry.py', 'rev4/prc_plugins.py',
          'rev4/restoration.py', 'rev4/ssim_calibration.py', 'rev4/stage_a.py', 'rev4/training_data.py',
          'lowres/common.py', 'lowres/evaluate_shared.py', 'lowres/generate_lr_only.py', 'lowres/inference.py',
          'submit_module_trials.py', 'submit_b1_diagnostics.py', 'submit_calibration.py', 'submit_b1.py',
          'submit_s2a.py']
PHASES = ('probe', 'zoom', 'train', 'evaluate', 'report')


def light():
    a = original.light('latent')
    a.output = REPO/'.runtime'/OUTPUT_NAME
    a.launch_identity = dict(stage='research bundle v1; single allocation', no_test=True,
                             plan=dict(train=list(TRAIN_ARMS), zoom_models=list(ZOOM_MODELS),
                                       zoom_variants={k: list(v) for k, v in ZOOM_VARIANTS.items()},
                                       stress_variants=list(STRESS_VARIANTS),
                                       rules={k: list(v) for k, v in ZOOM_RULES.items()}, zoom=ZOOM, prc2=PRC2,
                                       whole_image_edsr=f'job {EDSR_RECHECK_JOB}, not repeated here'),
                             source={name: sha256(REPO/name) for name in FILES+REUSED})
    return a


def verified(a):
    from rev4 import restoration

    ctx = original.verified(a)
    ctx.bclean_checkpoint = REPO/'.runtime/bclean/v1/weights/iter_20000.pth'
    ctx.bclean_digest = BCLEAN_CHECKPOINT_SHA
    require(sha256(ctx.bclean_checkpoint) == BCLEAN_CHECKPOINT_SHA, 'B-Clean checkpoint changed')
    ctx.edsr_weights = REPO/'.runtime/lr2_restoration/weights/edsr64_x2-torchsr-v1.0.3.pt'
    require(sha256(ctx.edsr_weights) == restoration.WEIGHT_SHA256, 'EDSR weights changed')
    prepared = REPO/'.runtime/bclean/v1/prepared'
    receipt = read_json(prepared/'prepared.json')
    ctx.bclean_annotations = prepared/'train_annotations_clean.json'
    require(ctx.bclean_annotations.is_file(), 'B-Clean prepared annotations missing')
    ctx.trial_identity.update(bclean_checkpoint_sha256=BCLEAN_CHECKPOINT_SHA, edsr_sha256=restoration.WEIGHT_SHA256,
                              bclean_annotation_sha256=receipt['annotation_sha256'],
                              bclean_annotation_source_sha256=receipt['source_sha256'])
    return ctx


def command(a):
    env = 'export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))
    env += ' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec '
    args = [str(a.python), '-u', str(REPO/'submit_research_bundle.py'), '--worker']
    return ['sbatch', '--parsable', '--job-name=monkey_v2_research_bundle', '--ntasks=1', '--account=uit',
            '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35', '--cpus-per-task=8', '--mem=40G',
            '--time='+TIME_LIMIT, '--chdir='+str(REPO), '--output='+str(a.output/'logs/job_%j.stdout.log'),
            '--error='+str(a.output/'logs/job_%j.stderr.log'), '--wrap', env+shlex.join(args)]


def child(a, phase, arm=None):
    args = [str(a.python), '-u', str(REPO/'submit_research_bundle.py'), '--phase', phase]
    if arm:
        args += ['--arm', arm]
    print('START', phase, arm or '', flush=True)
    code = subprocess.run(args, cwd=REPO).returncode
    print('END', phase, arm or '', code, flush=True)
    return code == 0


def allocated(a, args):
    import fcntl

    require(os.environ.get('SLURM_JOB_ID'), 'Require a Slurm allocation')
    require(read_json(a.output/'submission/launch.json') == a.launch_identity, 'Bundle source changed after submission')
    ctx = verified(a)
    path = ctx.output/'identity.json'
    if path.exists():
        require(read_json(path) == ctx.trial_identity, 'Bundle identity changed')
    else:
        save_json(path, ctx.trial_identity)
    if args.phase:
        from rev4 import bundle_report, bundle_training, zoom_study

        if args.phase != 'probe':
            require(read_json(ctx.output/'preflight.json')['identity'] == ctx.trial_identity, 'Probe must run first')
        if args.phase == 'probe':
            bundle_training.probe(ctx)
        elif args.phase == 'zoom':
            zoom_study.run(ctx, args.arm)
        elif args.phase == 'train':
            bundle_training.fit(ctx, args.arm)
        elif args.phase == 'evaluate':
            bundle_training.evaluate(ctx, args.arm)
        else:
            bundle_report.report(ctx)
        return
    with (ctx.output/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        storage_probe(ctx.output)
        subprocess.run([str(a.python), '-m', 'unittest', 'tests.test_research_bundle', '-v'], cwd=REPO, check=True)
        status_path = ctx.output/'status.json'
        status = read_json(status_path) if status_path.exists() else {}

        def step(phase, arm=None):
            key = f'{phase}:{arm or ""}'
            if status.get(key) == 'PASS' and phase != 'report':
                return True
            ok = child(a, phase, arm)
            status[key] = 'PASS' if ok else 'FAIL'
            save_json(status_path, status)
            return ok

        def skip(key, reason):
            status[key] = f'SKIPPED ({reason})'
            save_json(status_path, status)

        require(step('probe'), 'Technical probe failed; no study phase runs')
        preflight = read_json(ctx.output/'preflight.json')
        zoom_ok = preflight['zoom']['status'] == 'PASS'
        if zoom_ok:
            step('zoom', 'B1')
        else:
            skip('zoom:B1', 'zoom identity probe FAIL')
            skip('zoom:B-Clean', 'zoom identity probe FAIL')
        trained = []
        for arm in TRAIN_ARMS:
            if preflight['configs'][arm]['status'] != 'PASS':
                skip(f'train:{arm}', 'configuration probe FAIL')
                continue
            if arm == 'PRC2' and preflight['prc2']['status'] != 'PASS':
                skip('train:PRC2', 'PRC2 probe FAIL')
                continue
            if step('train', arm):
                trained.append(arm)
        for arm in trained:
            step('evaluate', arm)
        if zoom_ok:
            step('zoom', 'B-Clean')
        require(step('report'), 'Report phase failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan-only', action='store_true')
    actions.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--phase', choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument('--arm', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == 'zoom':
        require(args.arm in ZOOM_MODELS, 'Zoom phase needs a zoom model')
    elif args.phase in ('train', 'evaluate'):
        require(args.arm in TRAIN_ARMS, 'Training/evaluation phase needs a bundle arm')
    else:
        require(not args.arm, 'Unexpected --arm')
    a = light()
    if args.worker or args.phase:
        faulthandler.enable(all_threads=True)
        try:
            allocated(a, args)
        except BaseException:
            failure = traceback.format_exc()
            print(failure, file=sys.stderr, flush=True)
            print('BUNDLE FAILURE:\n'+failure, flush=True)
            try:
                name = f'failure_{os.environ.get("SLURM_JOB_ID", "unknown")}_{args.phase or "worker"}_{args.arm or ""}.txt'
                write_text(a.output/name, failure)
            except BaseException:
                print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        return
    require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the UIT-HPC login node')
    if args.plan_only:
        print(json.dumps(dict(plan=a.launch_identity['plan'], output=str(a.output), sbatch=command(a)), indent=2))
        return
    with run_lock(a.output/'submission'):
        launch = a.output/'submission/launch.json'
        if launch.exists():
            require(read_json(launch) == a.launch_identity, 'Bundle code changed after an earlier submission; use a new version')
        if (a.output/'results/complete.json').exists():
            print('Already complete:', a.output/'results')
            return
        ticket = a.output/'submission/latest_job.json'
        if ticket.exists():
            old = read_json(ticket)['job_id']
            if old in subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split():
                print('Already queued/running:', old)
                return
        storage_probe(a.output)
        save_json(launch, a.launch_identity)
        (a.output/'logs').mkdir(parents=True, exist_ok=True)
        job = subprocess.check_output(command(a), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(ticket, dict(job_id=job, identity=a.launch_identity))
        print('Submitted ONE research bundle job', job, flush=True)
        print('Logs:', a.output/'logs', '\nReport:', a.output/'results/bundle_report.md', flush=True)


if __name__ == '__main__':
    main()
