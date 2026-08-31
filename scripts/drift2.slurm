#!/bin/bash
#SBATCH --job-name=drift_compare
#SBATCH --partition=compute
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs reports/eda
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0057_nototals.yaml configs/e0056_nodrift.yaml)
"$PY" src/eda/anchor_drift.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
