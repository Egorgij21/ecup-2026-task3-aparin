#!/bin/bash
# e0930-e0938 -- the TRAINING-REGIME sweep on the causal (user-split GRU) path.
#
# WHY THIS AXIS EXISTS. run_usercv.py batches over USERS and supervises every valid day of the
# sampled users in one masked mean. So there is no curriculum and no date sampling anywhere in
# this project, and every calendar day carries EQUAL gradient weight -- the position that
# actually matters at submission (the final anchor) gets ~1/365th of the loss, the same as a
# day in April 2025. Nothing on record tests that. The nearest evidence is the GBDT anchor
# decay (hl 60d/180d, flat) whose grid spans 6.5 months against this path's 12, and e0115
# (supervising every 7th day costs +0.00012), which prices THINNING days, not RE-WEIGHTING them.
#
# ⚠ READ BEFORE INTERPRETING ANY RESULT HERE.
#   * PRE-REGISTERED PRIOR: all arms nil. e0115 says positions are ~redundant (targets overlap
#     29/30 days) and the GBDT decay surface is flat. Recorded before the numbers exist.
#   * PASS-THROUGH CAPS THE UPSIDE. The usercv slot's measured member->blend pass-through is
#     ~0.16x (d48: +0.00039 member rho -> +0.000064 blend). Buying the +0.000633 that reaches
#     1.6450 would need ~+0.0040 member rho, about 10x the largest gain this path has produced.
#     This is a MECHANISM TEST, not a route to top-1. Do not let a positive arm imply otherwise.
#   * --fixed-epochs IS MANDATORY and every arm carries it. 1j: scoring the MIN of a
#     variable-length early-stopped curve is biased by ~sigma*sqrt(2 ln N), so a curriculum that
#     trains "easier" early would win on evaluation count alone. This axis is maximally exposed
#     to that trap; a fixed budget scored once is the only honest comparison.
#   * REPORT WITHIN-ANCHOR RHO (1z-A). Raw RMSLE and pooled rho both pay a model for the
#     calendar level, which 1b proves is free at submission time.
#   * MATCHED SAME-SESSION CONTROL. e0930 runs in this array, not from a logged number:
#     cross-session drift is +0.00027..+0.00046 for configs with ZERO changes (e0195), which
#     exceeds every effect being tested here.
#
# ARMS -- each differs from e0930 by EXACTLY ONE declared thing (4.1). Multi-arm != bundled.
#   e0930  CONTROL                                    (the reference; no regime flags)
#   e0931  --day-weight-halflife 270   \
#   e0932  --day-weight-halflife 90     >  recency weighting, 9x half-life bracket
#   e0933  --day-weight-halflife 30    /      (the GBDT decay test spanned only 3x)
#   e0934  --curriculum expand         -- fresh data ADDED each epoch, old data never removed
#   e0935  --curriculum slide          -- old data dropped as fresh arrives, CONSTANT work/batch
#   e0936  --curriculum slide --curr-shuffle  -- ORDERING NULL (e0214 discipline): identical
#          per-epoch volumes, progression destroyed. If e0935 == e0936 the ORDER carries
#          nothing and only the volume schedule matters.
#   e0937  --day-sample 0.5            \  (user,date) sampling: Bernoulli day mask resampled
#   e0938  --day-sample 0.25           /  EVERY STEP, so the gradient stops being a
#                                         deterministic function of the user permutation
#
# Submit:  sbatch scripts/e0930_regime.slurm
# Smoke first (CLUSTER.md gotcha 8):  sbatch scripts/e0930_smoke.slurm
#
#SBATCH --job-name=e0930reg
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --array=0-8
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8

# The champion's usercv config (hidden=48, the d48 capacity optimum that produced e0301),
# scored at a FIXED budget of 32 epochs = the CV median best-epoch used by predict_usercv.py.
COMMON="--variant full --hidden 48 --folds 0 1 2 3 4 --seeds 3 --epochs 32 --fixed-epochs --no-save-weights"

ARMS=(
  "e0930|"
  "e0931|--day-weight-halflife 270"
  "e0932|--day-weight-halflife 90"
  "e0933|--day-weight-halflife 30"
  "e0934|--curriculum expand --curr-start 0.5"
  "e0935|--curriculum slide --curr-start 0.5"
  "e0936|--curriculum slide --curr-start 0.5 --curr-shuffle"
  "e0937|--day-sample 0.5"
  "e0938|--day-sample 0.25"
)

ENTRY="${ARMS[$SLURM_ARRAY_TASK_ID]}"
EXP="${ENTRY%%|*}"
FLAGS="${ENTRY#*|}"

# --tag-suffix gives every arm its own artefact tag. REVIEW_NOTES A3 asked for this and the
# 2026-08-20 incident that destroyed the usercv_extra OOF is why: the CONTROL carries no
# regime flags, so without a suffix its tag would collide with any prior identical run and
# silently overwrite the reference this whole array is measured against (rule 10).
echo "=== $EXP (array task $SLURM_ARRAY_TASK_ID, job $SLURM_JOB_ID) ==="
echo "=== run_usercv.py $COMMON --tag-suffix $EXP $FLAGS ==="
"$ROOT/envs/seq/bin/python" src/run_usercv.py $COMMON --tag-suffix "$EXP" $FLAGS
echo "=== $EXP DONE ==="
