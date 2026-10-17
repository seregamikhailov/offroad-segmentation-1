#!/usr/bin/env bash
# Run the Mac-sized screening sweep (restartable) and aggregate the results.
# caffeinate keeps the Mac awake (display may sleep) while the sweep runs; keep it plugged in.
#   nohup bash scripts/run_screening_mac.sh > sweep_mac.log 2>&1 &
#   GROUP=loss bash scripts/run_screening_mac.sh        # one group only
# Progress:  tail sweep_mac.log   |   per-run:  tail -c 300 runs/screening_mac/<run>/train.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export PYTHONWARNINGS=ignore NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
ARGS=(--sweep configs/sweeps/screening_mac.yaml)
[ -n "${GROUP:-}" ] && ARGS+=(--group "$GROUP")

caffeinate -i python -m offroad_seg.sweep "${ARGS[@]}"
python -m offroad_seg.aggregate --runs runs/screening_mac --out results/screening_mac
