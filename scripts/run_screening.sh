#!/usr/bin/env bash
# Run the screening sweep (restartable: finished runs are skipped) and aggregate the results.
# Use inside tmux/screen or with nohup:   nohup bash scripts/run_screening.sh > sweep.log 2>&1 &
# Run a single group:  GROUP=arch bash scripts/run_screening.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ARGS=(--sweep configs/sweeps/screening.yaml)
[ -n "${GROUP:-}" ] && ARGS+=(--group "$GROUP")

python -m offroad_seg.sweep "${ARGS[@]}"
python -m offroad_seg.aggregate --runs runs/screening --out results/screening
