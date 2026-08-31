#!/bin/bash
# Multi-anchor test-time augmentation for the seq family.
#   sbatch scripts/seq_anchors.slurm --config configs/e0101_seq_gru.yaml --anchors 30
#
# Same GPU justification as scripts/seq.slurm (CLUSTER.md rule 5b): it trains a torch model.
# Longer wall clock than seq.slurm because scoring is 30 forward passes per fold instead of 1.
#SBATCH --job-name=seqanch
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs oof subs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PY="$ROOT/envs/seq/bin/python"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

"$PY" src/run_seq_anchors.py "$@"
echo "=== SEQ ANCHORS DONE ==="
