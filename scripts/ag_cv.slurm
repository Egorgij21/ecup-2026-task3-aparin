#!/bin/bash
# AutoGluon TabularPredictor on the frozen folds -- one array task per fold.
#   sbatch scripts/ag_cv.slurm
# apini: no outbound internet, but envs/ag is already installed so nothing is fetched.
# The --gres flag is REQUIRED here or the job may land on the L40S node (CLUSTER.md §4).
#SBATCH --job-name=e0064_agcv
#SBATCH --partition=apini
#SBATCH --account=pilot_apini
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag data/ag
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # apini has no internet; fail fast, not hang
# Array index -> fold, deliberately reversed. SLURM dispatches low task IDs first, and on a
# busy GPU partition the later tasks may wait hours. Fold 4 is both the largest (25 anchors
# vs fold 0's 8) and the most test-like (CLAUDE.md §3.2), so it must not be last in line --
# its score alone is a usable screen against e0060's fold-4 value of 1.73236.
FOLD_ORDER=(4 3 2 1 0)
FOLD=${FOLD_ORDER[$SLURM_ARRAY_TASK_ID]}
echo "array task $SLURM_ARRAY_TASK_ID -> fold $FOLD"
"$ROOT/envs/ag/bin/python" src/run_ag.py \
    --config configs/e0060_filt_top400.yaml \
    --exp-id e0064 --mode cv --fold "$FOLD" \
    --presets medium_quality --time-limit 9000 --cache "$@"
echo "=== AG CV FOLD $FOLD DONE ==="
