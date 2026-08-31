#!/bin/bash
# Confirm the best screened parameter sets on all 5 folds, full data.
#SBATCH --job-name=tune_confirm
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
"$ROOT/envs/ecup/bin/python" src/tune.py --config configs/e0049_nomoment.yaml \
    --study lgb_v1 --cache --confirm-top "${TOPN:-3}"
