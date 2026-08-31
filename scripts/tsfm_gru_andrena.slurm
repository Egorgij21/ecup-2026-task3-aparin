#!/bin/bash
#SBATCH --job-name=tsfm_gru_a
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
# Weights were pre-fetched by scripts/ttm_gate.slurm on `computeshort`; run offline so a hub
# lookup cannot stall or fail the GPU job (CLUSTER.md 4: apini has no internet at all).
export HF_HOME="$ROOT/envs/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ROOT/envs/tsfm/bin/python" src/run_tsfm_gru.py "$@"
echo "=== TSFM GRU DONE ==="
