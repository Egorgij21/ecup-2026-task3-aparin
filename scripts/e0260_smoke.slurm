#!/bin/bash
# Smoke for run_regime.py (CLUSTER.md §7.8). 2 folds x 2 anchors, --no-log: writes nothing to
# runs/ or experiments.csv. Proves all 7 arms train end to end, the leak guard passes, the
# sample-weight vectors reach LightGBM with the right length, and multi-seed arms actually
# produce DIFFERENT per-seed predictions (bagging_seed/feature_fraction_seed are set
# explicitly -- if they were not, a seed change would be a no-op and e0260 would silently be
# an expensive copy of the reference).
#
# --time is 3h, not 1h: the e0210 smoke TIMEOUT'd at 01:00:10 because a computeshort limit rode
# along onto `compute`. Do not lower this without measuring a fold first.
#SBATCH --job-name=e0260smoke
#SBATCH --partition=compute
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

"$PY" src/run_regime.py --config configs/e0260_regime.yaml \
      --screen --max-train-anchors 2 --no-log
echo "=== SMOKE DONE ==="
