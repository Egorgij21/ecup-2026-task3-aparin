#!/bin/bash
#SBATCH --job-name=cat_confirm
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
"$ROOT/envs/ecup/bin/python" src/tune.py --config configs/e0049_nomoment.yaml \
    --model cat --study cat_v1 --cache --confirm-top 3 --save-oof e0092_cat
