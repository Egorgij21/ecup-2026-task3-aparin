#!/bin/bash
#SBATCH --job-name=alpha
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
"$ROOT/envs/ecup/bin/python" src/blend_fixed.py \
  --group gbdt e0049 e0064 \
  --group seq e0100 e0101 e0101s1 e0101s2 e0101s3 e0102 e0108
echo "=== ALPHA DONE ==="
