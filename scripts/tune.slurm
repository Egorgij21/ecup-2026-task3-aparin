#!/bin/bash
# Optuna screen for the LightGBM member. Workers share one JournalStorage file.
#   sbatch scripts/tune.slurm
#SBATCH --job-name=tune_lgb
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
"$ROOT/envs/ecup/bin/python" src/tune.py \
    --config configs/e0049_nomoment.yaml --study lgb_v1 \
    --trials 14 --folds 3 --subsample 0.5 --cache
