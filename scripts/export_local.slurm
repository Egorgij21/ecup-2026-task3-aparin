#!/bin/bash
# Build one anchor's features and ship a compact parquet for the local notebook.
#SBATCH --job-name=export_local
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
"$ROOT/envs/ecup/bin/python" src/eda/export_local.py "$@"
