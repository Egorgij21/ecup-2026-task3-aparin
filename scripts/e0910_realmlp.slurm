#!/bin/bash
# e0910 -- RealMLP ONLY (drop TabM, which crawled on CPU at 4.5M rows), on a 1.2M-row subsample
# so a CPU-only NN is feasible. IDEAS.md §I22: is a modern tabular NN the decorrelated STRONG
# member that closes the top-1 gap? TabICLv2 (teammate e0340) already showed a non-tree tabular
# model reaches r=0.995 but rho only 0.659 (just short of positive excess). RealMLP is documented
# stronger than TabICL -> the test is whether it holds rho >= ~0.659 at that decorrelation.
# --no-refit-full keeps the model on the subsample (refit would re-expand to full data).
# Logged as e0910 to stay clear of the teammate's e03xx frontier.
#SBATCH --job-name=e0910nn
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
PY="$ROOT/envs/ag/bin/python"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FOLD=${SLURM_ARRAY_TASK_ID:-4}
echo "=== e0910 RealMLP (1.2M subsample, no refit), fold $FOLD ==="
"$PY" src/run_ag.py --exp-id e0910 --config configs/e0049_nomoment.yaml --fold "$FOLD" \
  --presets medium_quality --time-limit 18000 --subsample 1200000 --no-refit-full \
  --hyperparameters '{"REALMLP":{}}'
echo "=== e0910 fold $FOLD DONE ==="
