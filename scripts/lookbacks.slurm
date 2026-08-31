#!/bin/bash
#SBATCH --job-name=e0029_e0034
#SBATCH --partition=compute
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0029_lookback15.yaml configs/e0030_lookback60.yaml configs/e0031_lookback120.yaml \
      configs/e0032_lookback180.yaml configs/e0033_lookback240.yaml configs/e0034_lookback360.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
