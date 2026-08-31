#!/bin/bash
#SBATCH --job-name=e0011_e0014
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0011_diff.yaml configs/e0012_cumshare.yaml configs/e0013_ewm.yaml configs/e0014_com.yaml)

"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== DONE ${CFGS[$SLURM_ARRAY_TASK_ID]} ==="
