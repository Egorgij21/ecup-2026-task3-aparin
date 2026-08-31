#!/bin/bash
#SBATCH --job-name=e0051_e0055
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
CFGS=(configs/e0051_fcomega.yaml \
      configs/e0052_fcstl.yaml \
      configs/e0053_fcshift.yaml \
      configs/e0054_fctrans.yaml \
      configs/e0055_fcacf.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
