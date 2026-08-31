#!/bin/bash
# e0210-e0214: XGBoost family reference + the three FEATURES.md confirm candidates + the
# noise control. One array task per experiment; each is a full 5-fold frozen-fold run.
#
# CLUSTER.md: `compute`, no --account, no --gres (CPU work). Logs use %x so the filename
# always equals the job name. The feature cache stays OFF -- scratch is at 2.87T of a 3T soft
# limit and one generation of cache is ~45G (CLUSTER.md §7.1).
#SBATCH --job-name=e0210_e0214
#SBATCH --partition=compute
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

CFGS=(configs/e0210_xgb_base.yaml \
      configs/e0211_xgb_agebucket.yaml \
      configs/e0212_xgb_cartbacklog.yaml \
      configs/e0213_xgb_cohortrel.yaml \
      configs/e0214_xgb_noisectl.yaml)

CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID -> $CFG"
"$PY" src/run_xgb.py --config "$CFG"
echo "=== DONE $CFG"
