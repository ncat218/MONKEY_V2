#!/usr/bin/env python3
"""Submit the versioned 100% Clean control, initialized from B0."""
import argparse
import json
import os

from rev4.adaptation_jobs import allocated_context, light_inputs, sbatch_args, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    light = light_inputs('bclean')
    if args.plan_only:
        print(json.dumps(dict(recipe='B1 recipe, B0 init, 100% Clean, 10k updates',
                              input_verification='pending allocated worker', sbatch=sbatch_args(light)), indent=2))
    elif args.worker:
        import fcntl
        from rev4 import bclean

        light.output.mkdir(parents=True, exist_ok=True)
        with (light.output/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bclean.run(allocated_context(light))
    else:
        submit(light)


if __name__ == '__main__':
    main()
