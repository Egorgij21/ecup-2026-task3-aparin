#!/bin/bash
#SBATCH --job-name=e0058_e0062
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0058_filt_top100.yaml \
      configs/e0059_filt_top200.yaml \
      configs/e0060_filt_top400.yaml \
      configs/e0061_filt_positive.yaml \
      configs/e0062_filt_positive_strict.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
