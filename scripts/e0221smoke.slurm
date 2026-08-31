#!/bin/bash
#SBATCH --job-name=e0221smoke
#SBATCH --partition=computeshort
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
"$PY" src/run.py --config configs/e0221smoke.yaml --screen
echo "=== SMOKE DONE ==="
