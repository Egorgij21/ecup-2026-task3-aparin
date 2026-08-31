#!/bin/bash
#SBATCH --job-name=gifters_build
#SBATCH --partition=compute
#SBATCH --mem=140G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"
mkdir -p logs reports/eda reports/figs

# Call the interpreter DIRECTLY -- never `conda activate` (see CLUSTER.md gotcha 2).
PY="$ROOT/envs/ecup/bin/python"
export POLARS_MAX_THREADS=16
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

"$PY" src/gifters.py
echo "=== EDA DONE ==="
