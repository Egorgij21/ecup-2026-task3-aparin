#!/bin/bash
#SBATCH --job-name=e0001_seeds
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
CFGS=(configs/e0001s1_seed.yaml configs/e0001s2_seed.yaml configs/e0001s3_seed.yaml \
      configs/e0001s4_seed.yaml)
"$PY" src/run.py --config "${CFGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== DONE ${CFGS[$SLURM_ARRAY_TASK_ID]} ==="
