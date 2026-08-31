#!/bin/bash
# Feasibility smoke for IDEAS.md §I19: last 2 folds only, control (lambda=0) + one treatment
# (lambda=0.5). Confirms the within-day Pearson term (a) does not NaN under bf16 AMP, (b) leaves
# the MSE path byte-identical at lambda=0, and (c) actually changes the trajectory at lambda>0.
# --screen writes a tier=screen row; nothing here decides (CLAUDE.md §4.2).
#SBATCH --job-name=e0290smA
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs oof
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$ROOT/envs/seq/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== SMOKE e0290 control (lambda=0, must reproduce the MSE path) ==="
"$PY" src/run_seq.py --config configs/e0290_corrloss.yaml --screen
echo "=== SMOKE e0292 treatment (lambda=0.5) ==="
"$PY" src/run_seq.py --config configs/e0292_corrloss.yaml --screen
echo "=== SMOKE DONE ==="
