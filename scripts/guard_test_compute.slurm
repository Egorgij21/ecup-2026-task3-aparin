#!/bin/bash
#SBATCH --job-name=guard_test
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
ROOT=/path/to/ecup
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
"$ROOT/envs/ecup/bin/python" src/eda/guard_test.py --cache "$@"
