#!/bin/bash
# SMOKE for the e0930-e0938 training-regime sweep (CLUSTER.md gotcha 8: never launch a full
# array without one). 1 fold, 1 seed, 3 epochs -- validates plumbing, NOT a result (4.2).
#
# THREE TASKS, and task 1 is the one that matters:
#   0  control, no regime flags
#   1  the SAME config with every new flag passed EXPLICITLY AT ITS DEFAULT
#      (--curriculum none --day-sample 1.0 --day-weight-halflife 0)
#      => PORT GATE. andrena is bit-deterministic within its GPU architecture (CLUSTER.md 10),
#         so task 1's rho must equal task 0's EXACTLY. Any difference means a new branch is
#         not inert at its default and every historical run on this path is invalidated.
#         This is the same gate --loss passed (corr_lambda=0 == e0101 byte-for-byte).
#   2  all three knobs ON at once. NOT an experiment (three changes) -- it exercises the code
#      paths and, more usefully, prints the per-epoch `cells` counter so the schedule claims
#      are MEASURED: `slide` must hold cells/epoch constant, which is the property the whole
#      "constant number of batches across epochs" design rests on.
#
# Read the .out for:  "CURRICULUM 'slide'" / "DAY RECENCY WEIGHTING" / "(USER,DATE) SAMPLING"
# lines, then the per-epoch `cells N.NNM` field.
#
#SBATCH --job-name=e0930smk
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --array=0-3
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8

COMMON="--variant full --hidden 48 --folds 4 --seeds 1 --epochs 5 --fixed-epochs --no-save-weights"

# Task 2 is slide ALONE (not stacked with day-sample) so the per-epoch `cells` counter reads
# the schedule and nothing else -- that is the number the "constant work per epoch" claim
# lives or dies on, and the first smoke showed a day-based schedule drifting +26% on it.
ARMS=(
  "smk0ctl|"
  "smk1gate|--curriculum none --day-sample 1.0 --day-weight-halflife 0"
  "smk2slide|--curriculum slide --curr-start 0.5"
  "smk3rest|--curriculum expand --curr-start 0.5 --day-sample 0.5 --day-weight-halflife 90"
)
ENTRY="${ARMS[$SLURM_ARRAY_TASK_ID]}"
EXP="${ENTRY%%|*}"; FLAGS="${ENTRY#*|}"

echo "=== SMOKE $EXP (task $SLURM_ARRAY_TASK_ID) : $COMMON --tag-suffix $EXP $FLAGS ==="
"$ROOT/envs/seq/bin/python" src/run_usercv.py $COMMON --tag-suffix "$EXP" $FLAGS
echo "=== SMOKE $EXP DONE ==="
