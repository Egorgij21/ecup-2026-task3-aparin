#!/bin/bash
# e0260-e0266 -- the training-regime sweep. 7 arms + 1 same-session reference per fold = 18
# fits/fold x 5 folds = 90 fits, sharing one feature build per fold.
#
# Wall clock: feature building is ~25 min/fold and the 178-round fits are ~2 min each, but
# e0266 is 1162 rounds x 3 seeds at lr 0.0077 -- that arm alone is roughly half the fit budget.
# 16h with the e0210 lesson applied (a partition change must carry a matching --time).
#SBATCH --job-name=e0260_e0266
#SBATCH --partition=compute
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

"$PY" src/run_regime.py --config configs/e0260_regime.yaml
echo "=== REGIME SWEEP DONE ==="
