#!/bin/bash
# Smoke test for run_xgb.py + the FEATURES.md confirm blocks (CLUSTER.md §7.8: always
# smoke-test on a subsample before a full array).
#
# Usage:  sbatch scripts/e0210_smoke.slurm [configs...]
# No args -> all five e021x configs.  One config -> that config only (used for a re-run
# of a single config after the rest already passed).
#
# 2026-08-20: --time raised 1:00:00 -> 3:00:00. The first smoke on `compute` still carried
# the computeshort 1 h limit and TIMEOUT'd at 01:00:10 after validating 4 of 5 configs
# (e0210-e0213 all green; e0214 was the one cut off). The limit now covers a full sweep.
#
# This is NOT an experiment: --no-log keeps every run out of runs/ and experiments.csv.
# --screen (2 folds) + --max-train-anchors 2 prove end to end that (1) the runner imports
# and trains, (2) assert_no_lookahead passes with each new block installed, (3) each block
# adds exactly one column, (4) the LightGBM->XGB parameter mapping gives a sane RMSLE.
#SBATCH --job-name=e0210smoke
#SBATCH --partition=compute
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

CFGS=("$@")
if [ ${#CFGS[@]} -eq 0 ]; then
  CFGS=(configs/e0210_xgb_base.yaml \
        configs/e0211_xgb_agebucket.yaml \
        configs/e0212_xgb_cartbacklog.yaml \
        configs/e0213_xgb_cohortrel.yaml \
        configs/e0214_xgb_noisectl.yaml)
fi

for CFG in "${CFGS[@]}"; do
  echo "############ SMOKE $CFG"
  "$PY" src/run_xgb.py --config "$CFG" --screen --max-train-anchors 2 --no-log
done
echo "=== SMOKE DONE ==="
