#!/bin/bash
#SBATCH --job-name=lgb3_confirm
#SBATCH --partition=compute
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
"$ROOT/envs/ecup/bin/python" src/tune.py --config configs/e0049_nomoment.yaml \
    --model lgb --study lgb_v3 --cache --confirm-top 3 --save-oof e0093_lgb3
