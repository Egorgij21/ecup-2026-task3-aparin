#!/bin/bash
#SBATCH --job-name=pred_e0049_e0020
#SBATCH --partition=compute
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
"$PY" src/predict.py --config "${CONFIG:-configs/e0049_nomoment.yaml}"
echo "=== PREDICT DONE ==="
