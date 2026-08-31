#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --partition=computeshort
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs reports
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8 PYTHONUNBUFFERED=1
"$PY" scripts/noise_ceiling.py --full --max-gb 60 2>&1 | tee reports/noise_ceiling_full.log
echo "=== CEILING DONE ==="
