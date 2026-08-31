#!/bin/bash
# e0900-e0327 -- the scale-penalty gate + pseudo-label arms (IDEAS.md I17/I18). Eight arms
# sharing one feature build per fold; each writes its own OOF and runs/<exp_id>.json.
#
# CLUSTER.md: `compute`, no --account/--gres (CPU work), %x logging. Feature cache OFF (the
# cluster cache generations predate the current features.py hash, so there would be no hits).
# Wall clock: feature building ~25 min/fold dominates; the fits are seconds to ~3 min each.
#SBATCH --job-name=e0900_e0327
#SBATCH --partition=compute
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof runs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1

"$PY" src/metrics.py
"$PY" src/run_gate.py --config configs/e0900_gate.yaml "$@"
echo "=== GATE SWEEP DONE ==="
