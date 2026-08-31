#!/bin/bash
# Smoke for run_magnitude.py (CLUSTER.md §7.8). 2 folds x 2 anchors, --no-log: writes nothing
# to runs/ or experiments.csv. Proves the runner trains all 8 arms end to end, that the leak
# guard passes, that the buyers-only restriction bites (~56% of rows kept), and that the
# rank-Gaussian inverse returns predictions on the L scale rather than in normal-score units.
#
# --time is 3h, not 1h: the e0210 smoke TIMEOUT'd at 01:00:10 because a computeshort limit rode
# along onto `compute`. Do not lower this without measuring a fold first.
#SBATCH --job-name=e0250smoke
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

"$PY" src/run_magnitude.py --config configs/e0250_magnitude.yaml \
      --screen --max-train-anchors 2 --no-log
echo "=== SMOKE DONE ==="
