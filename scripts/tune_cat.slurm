#!/bin/bash
# CatBoost tuning. GPU is a genuine requirement (task_type=GPU), so apini is correct per 5b.
#SBATCH --job-name=tune_cat
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
"$ROOT/envs/ecup/bin/python" src/tune.py --config configs/e0049_nomoment.yaml \
    --model cat --study cat_v1 --trials 12 --folds 3 --subsample 0.5 --cache
