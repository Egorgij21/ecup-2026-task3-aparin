#!/bin/bash
#SBATCH --job-name=e0027_e0029_e0030_e0033
#SBATCH --partition=compute
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0027_lookback30.yaml configs/e0029_lookback15.yaml \
      configs/e0030_lookback60.yaml configs/e0033_lookback270.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
