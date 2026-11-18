"""Standard-library-only launcher plumbing; heavyweight audits stay on GPU nodes."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from lowres.common import canonical, read_json, run_lock, save_json, sha256
from rev4 import require

REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('rev4/adaptation_plan.py', 'rev4/adaptation_jobs.py', 'rev4/bclean.py',
                'rev4/bclean_plugins.py', 'rev4/paired_eda_core.py', 'rev4/paired_eda.py', 'rev4/paired_eda_report.py',
                'submit_bclean.py', 'submit_paired_eda.py', 'tests/test_adaptation.py',
                'docs/bclean-and-paired-eda.md')


def source_identity():
    return {name: sha256(REPO/name) for name in SOURCE_FILES}


def light_inputs(kind, include_bclean=False):
    require(kind in ('bclean', 'eda'), 'Unknown research job')
    stack = REPO.parent/'monkeyocr_workspace/environment_and_models__phase1_verified_stack'
    run = REPO/'.runtime'/('bclean/v1' if kind == 'bclean' else 'paired_eda/v1')
    mode = 'B0_BClean_B1' if include_bclean else 'B0_B1'
    return SimpleNamespace(kind=kind, include_bclean=include_bclean, output=run, mode=mode,
                submission=run/'submission'/(mode if kind == 'eda' else 'B-Clean'),
                python=stack/'python311_venv/bin/python',
                cuda=stack/'cuda_toolkit_12_8_1__private_nvidia_conda_environment',
                s1=REPO/'.runtime/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0',
                launch_identity=dict(kind=kind, code=source_identity()))


def sbatch_args(a, dependencies=()):
    name = 'submit_bclean.py' if a.kind == 'bclean' else 'submit_paired_eda.py'
    args = [str(a.python), '-u', str(REPO/name), '--worker']
    if a.kind == 'eda' and a.include_bclean:
        args.append('--include-bclean')
    env = ('export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(a.s1/'pillow-11.3.0'))+
           ' LD_LIBRARY_PATH='+shlex.quote(str(a.cuda/'lib'))+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; exec ')
    command = ['sbatch', '--parsable', '--job-name=monkey_v2_'+('bclean' if a.kind == 'bclean' else 'eda_'+a.mode),
               '--ntasks=1', '--account=uit', '--partition=normal', '--gres=gpu:2g.35gb:1', '--qos=mig2-35',
               '--cpus-per-task=8', '--mem=40G', '--time='+('08:00:00' if a.kind == 'bclean' else '06:00:00'),
               '--chdir='+str(REPO), '--output='+str(a.output/'logs/job_%j.stdout.log'),
               '--error='+str(a.output/'logs/job_%j.stderr.log')]
    if dependencies:
        require(all(str(job).isdecimal() for job in dependencies), 'Dependencies must be Slurm numeric job IDs')
        command.append('--dependency=afterok:'+':'.join(map(str, dependencies)))
    return command+['--wrap', env+shlex.join(args)]


def submit(a, dependencies=()):
    require(sys.platform == 'linux' and not os.environ.get('SLURM_JOB_ID'), 'Submit from the HPC login node')
    with run_lock(a.submission):
        ticket = a.submission/'launch.json'
        if ticket.is_file():
            require(read_json(ticket)['identity'] == a.launch_identity, 'Run source changed; stop for review')
        receipt = a.submission/'latest_job.json'
        if receipt.is_file():
            previous = read_json(receipt)
            queued = subprocess.check_output(['squeue', '--me', '--noheader', '--format=%i'], text=True).split()
            if previous['job_id'] in queued:
                print('Already queued/running:', previous['job_id'], flush=True)
                return
        completed = (a.output/'results/bclean_summary.json' if a.kind == 'bclean' else
                     a.output/'results'/a.mode/'complete.json')
        if completed.is_file():
            print('Already complete:', completed, flush=True)
            return
        require(a.python.is_file(), f'Historical Python missing: {a.python}')
        (a.output/'logs').mkdir(parents=True, exist_ok=True)
        save_json(ticket, dict(identity=a.launch_identity, mode=a.mode, dependencies=list(dependencies),
                              full_input_checks='pending allocated worker'))
        job = subprocess.check_output(sbatch_args(a, dependencies), text=True).strip().split(';')[0]
        require(job.isdecimal(), f'Unexpected sbatch response: {job}')
        save_json(receipt, dict(job_id=job, identity=a.launch_identity, mode=a.mode, dependencies=list(dependencies)))
        print(f'Submitted job {job}\nLogs: {a.output/"logs"}\nOutput: {a.output}', flush=True)


def allocated_context(light):
    require(sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID'), 'Worker requires a Slurm GPU allocation')
    require(read_json(light.submission/'launch.json')['identity'] == light.launch_identity,
            'Source changed after submission')
    print('Running synthetic control/evaluator regression checks inside allocation...', flush=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_adaptation', 'tests.test_b1',
                    'tests.test_calibration', '-v'], cwd=REPO, check=True)
    print('Verifying frozen B0/B1 inputs, split and evaluator...', flush=True)
    import submit_b1_diagnostics
    from rev4.b1_training import environment_identity
    from rev4.calibration import validation_cohort

    a = submit_b1_diagnostics.verified_inputs()
    require(environment_identity(a, a.calibration_prepared) == a.b1_identity, 'Runtime differs from historical B1')
    validation_cohort(a, a.calibration_prepared)
    a.output, a.mode, a.include_bclean = light.output, light.mode, light.include_bclean
    a.launch_identity = light.launch_identity
    a.code_identity = source_identity()
    print('Frozen inputs/runtime PASS.', flush=True)
    return a


def lock_record(path, identity):
    identity = json.loads(canonical(identity))
    if path.is_file():
        require(read_json(path) == identity, f'Identity changed at {path}; stop for review')
    else:
        save_json(path, identity)
    return identity

