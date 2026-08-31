#!/bin/bash
# Smoke for e0281_gate: last 2 folds, 2 training anchors, nothing logged (CLUSTER.md gotcha 8).
# NOTE: --max-train-anchors 2 shrinks the training set to ~400k rows, so the subsample arms
# are NOT the real gate here -- this only checks the code path end to end.
#SBATCH --job-name=e0900smoke
#SBATCH --partition=compute
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

"$PY" src/run_gate.py --config configs/e0900_gate.yaml --screen --max-train-anchors 2 --no-log
echo "=== SMOKE DONE ==="
