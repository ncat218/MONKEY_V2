#!/usr/bin/env python3
"""Inference-only paired before/after EDA on remaining-validation."""
import argparse
import json

from rev4.adaptation_jobs import allocated_context, light_inputs, sbatch_args, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--include-bclean', action='store_true')
    parser.add_argument('--after-job', action='append', default=[], help='Run after successful Slurm job(s)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    light = light_inputs('eda', args.include_bclean)
    if args.plan_only:
        print(json.dumps(dict(models=light.mode, input_verification='pending allocated worker',
                              sbatch=sbatch_args(light, args.after_job)), indent=2))
    elif args.worker:
        import fcntl
        from rev4 import paired_eda

        light.output.mkdir(parents=True, exist_ok=True)
        with (light.output/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            paired_eda.run(allocated_context(light))
    else:
        submit(light, args.after_job)


if __name__ == '__main__':
    main()
