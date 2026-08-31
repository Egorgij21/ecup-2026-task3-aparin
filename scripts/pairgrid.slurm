#!/bin/bash
# 20x20 pair-plot matrix of the top features, coloured by target.
#   sbatch scripts/pairgrid.slurm --config configs/e0039_sbc.yaml
#   sbatch scripts/pairgrid.slurm --config configs/e0039_sbc.yaml --dpi 900 --n-points 125000
# At 900 dpi the canvas is 46,800 px/side = 8.8 GB of RGBA before PNG encoding, and
# bbox_inches="tight" renders it twice -- hence the large memory ask.
#SBATCH --job-name=pairgrid
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$ROOT/data/mplcache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # apini has no internet
"$ROOT/envs/ecup/bin/python" src/eda/pairgrid.py "$@"
ls -la "$ROOT/reports/eda/"pairgrid_*.png
