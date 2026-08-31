#!/bin/bash
#SBATCH --job-name=e0063_impall
#SBATCH --partition=compute
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs reports
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  "$PY" src/run.py --config configs/e0063_all.yaml
else
  "$PY" src/importance.py --config configs/e0063_all.yaml --tag _all
fi
