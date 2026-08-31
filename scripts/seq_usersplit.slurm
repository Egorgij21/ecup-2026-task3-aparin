#!/bin/bash
#SBATCH --job-name=seq_usersplit
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$ROOT/envs/seq/bin/python" src/seq_usersplit.py "$@"
echo "=== SEQ USERSPLIT DONE ==="
