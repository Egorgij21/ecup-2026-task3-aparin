#!/bin/bash
#SBATCH --job-name=pred_tuned_lgb
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
"$ROOT/envs/ecup/bin/python" src/predict_tuned.py \
  --config configs/e0049_nomoment.yaml --params configs/e0093_lgb3_params.json \
  --out e0093_lgb3 --cache
