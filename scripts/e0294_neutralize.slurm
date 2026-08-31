#!/bin/bash
# e0294 -- feature-neutralisation diagnostic (IDEAS.md §I21). No training: rebuilds each fold's
# validation features and orthogonalises e0049's OOF prediction against the drift-feature subset.
# CPU work -> compute (no GPU contention, unlike the corr-loss run). ~1-2 min/fold feature build.
#SBATCH --job-name=e0294neut
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
"$PY" src/run_neutralize.py --oof oof/e0049.parquet --exp-id e0294
echo "=== NEUTRALIZE DONE ==="
