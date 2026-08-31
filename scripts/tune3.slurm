#!/bin/bash
#SBATCH --job-name=tune_lgb3
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
"$ROOT/envs/ecup/bin/python" src/tune.py \
    --config configs/e0049_nomoment.yaml --study lgb_v3 \
    --trials 16 --folds 3 --subsample 0.5 --cache --space v3
