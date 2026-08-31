#!/bin/bash
# Like scripts/predict.slurm but FORWARDS every argument, so --train-through works.
# A separate file on purpose: predict.slurm is shared and overwriting a teammate's script
# has cost this project once already (EXPERIMENTS.md 6b).
#
#   sbatch -J e0251 scripts/predict_ext.slurm --config configs/e0251_gbdtext.yaml \
#          --rounds 587 --train-through 2026-01-14
#SBATCH --job-name=pred_ext
#SBATCH --partition=compute
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
echo "=== $SLURM_JOB_NAME ($SLURM_JOB_ID): predict.py $@ ==="
"$PY" src/predict.py "$@"
echo "=== PREDICT DONE ==="
