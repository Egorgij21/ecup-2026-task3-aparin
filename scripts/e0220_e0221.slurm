#!/bin/bash
#SBATCH --job-name=e0220_e0221
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=11:00:00
#SBATCH --array=0-1
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
CFG=(configs/e0220_l2_es.yaml configs/e0221_hlgauss.yaml)
"$PY" src/run.py --config "${CFG[$SLURM_ARRAY_TASK_ID]}"
echo "=== DONE ${CFG[$SLURM_ARRAY_TASK_ID]} ==="
