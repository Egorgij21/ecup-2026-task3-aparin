#!/bin/bash
# e0250-e0257 -- the magnitude sweep (IDEAS.md §I13). Eight arms sharing one feature build
# per fold; each writes its own OOF and its own runs/<exp_id>.json.
#
# CLUSTER.md: `compute`, no --account/--gres (CPU work), %x logging, feature cache OFF
# (scratch at ~2.9T of a 3T soft limit). --time covers 5 folds x (1 reference + 8 arms);
# feature building dominates at ~25 min/fold, the fits are minutes each.
#SBATCH --job-name=e0250_e0257
#SBATCH --partition=compute
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

"$PY" src/metrics.py
"$PY" src/run_magnitude.py --config configs/e0250_magnitude.yaml
echo "=== MAGNITUDE SWEEP DONE ==="
