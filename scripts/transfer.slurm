#!/bin/bash
#SBATCH --job-name=transfer_test
#SBATCH --partition=compute
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00

#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs reports/eda
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
"$PY" src/eda/transfer_test.py
