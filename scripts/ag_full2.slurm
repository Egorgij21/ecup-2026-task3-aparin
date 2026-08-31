#!/bin/bash
# AutoGluon TabularPredictor on the frozen folds -- one array task per fold.
#   sbatch scripts/ag_cv.slurm
# apini: no outbound internet, but envs/ag is already installed so nothing is fetched.
# The --gres flag is REQUIRED here or the job may land on the L40S node (CLUSTER.md §4).
#SBATCH --job-name=e0065_agfull
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
cd "$ROOT"; mkdir -p logs runs/ag data/ag
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # apini has no internet; fail fast, not hang
# Array index -> fold, deliberately reversed. SLURM dispatches low task IDs first, and on a
# busy GPU partition the later tasks may wait hours. Fold 4 is both the largest (25 anchors
# vs fold 0's 8) and the most test-like (CLAUDE.md §3.2), so it must not be last in line --
# its score alone is a usable screen against e0060's fold-4 value of 1.73236.



"$ROOT/envs/ag/bin/python" src/run_ag.py \
    --config configs/e0060_filt_top400.yaml \
    --exp-id e0065 --mode full \
    --presets good_quality --time-limit 28800 --exclude KNN RF XT --cache "$@"
echo "=== AG FULL DONE ==="
