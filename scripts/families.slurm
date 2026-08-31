#!/bin/bash
#SBATCH --job-name=e0020r_e0041_e0048
#SBATCH --partition=compute
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-8
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFGS=(configs/e0020r_repro.yaml \
      configs/e0041_sbcshape.yaml \
      configs/e0042_sbcmoment.yaml \
      configs/e0043_sbcorder.yaml \
      configs/e0044_sbcnorm.yaml \
      configs/e0045_sbcspec.yaml \
      configs/e0046_sbcts.yaml \
      configs/e0047_sbcent.yaml \
      configs/e0048_tsfeat.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
