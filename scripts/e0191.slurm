#!/bin/bash
#SBATCH --job-name=e0191
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
"$PY" src/metrics.py
"$PY" src/folds.py
"$PY" src/run.py --config configs/e0191_tsfresh.yaml ${SCREEN:+--screen}
echo "=== TRAIN DONE ==="
