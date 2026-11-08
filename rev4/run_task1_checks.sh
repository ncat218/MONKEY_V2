#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task="$repo/.runtime/rev4_task1"

stack="/datastore/$USER/textvqa/Tuan/monkeyocr_workspace/environment_and_models__phase1_verified_stack"
overlay="$repo/.runtime/MONKEY_V2__FULL_R1_LowRes_pilot_FROZEN_v2_0/pillow-11.3.0"
python="$stack/python311_venv/bin/python"
audit=$(find "$task" -mindepth 2 -maxdepth 2 -path '*/recheck-*/s2_audit.json' | sort | tail -n 1)
if [[ ! -f "$audit" ]]; then
  printf 'No completed audit recheck found in %s\n' "$task" >&2
  exit 2
fi
work="$task/generator-dryrun-$(date +%Y%m%d-%H%M%S)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo:$overlay"
cd "$repo"
printf 'Using audit %s\n' "$audit"
"$python" -m rev4.verify_task1_generators --audit-json "$audit" --output-dir "$work"
"$python" -m unittest discover -s tests -p 'test_rev4_ssim_calibration.py' -v 2>&1 | tee "$work/ssim_unit_tests.log"
printf 'Review %s/orientation_preview.png and retain %s/generator_dry_run.json.\n' "$work" "$work"
