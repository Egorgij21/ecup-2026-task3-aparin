#!/bin/bash
#SBATCH --job-name=e0222
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs oof subs
PY="$ROOT/envs/ecup/bin/python"
export OMP_NUM_THREADS=16 POLARS_MAX_THREADS=16 PYTHONUNBUFFERED=1
"$PY" src/run.py --config configs/e0222_twopart_full.yaml
echo "=== e0222 DONE ==="
