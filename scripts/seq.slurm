#!/bin/bash
# The `seq` approach: causal sequence model on the raw daily panel.
#   sbatch scripts/seq.slurm --config configs/e0100_seq_tcn.yaml --screen
#
# This is the ONE job type in this project that legitimately holds a GPU (CLUSTER.md rule 5b):
# it trains a torch model, unlike LightGBM/EDA/feature building, which belong on `compute`.
# The --gres flag is REQUIRED or the job may land on the L40S node instead (CLUSTER.md §4).
# apini has no outbound internet; envs/seq is pre-built on `compute`, so nothing is fetched.
#SBATCH --job-name=seq
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
cd "$ROOT"; mkdir -p logs runs oof subs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PY="$ROOT/envs/seq/bin/python"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" src/run_seq.py "$@"
echo "=== SEQ DONE ==="
