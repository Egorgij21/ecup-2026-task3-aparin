#!/bin/bash
# Leave-one-fold-out log-space blend across approaches.  CPU-only -> `compute`.
#   sbatch scripts/blend_seq.slurm e0049 e0100 e0101
#
# This is the measurement the whole `seq` approach exists to produce.  EXPERIMENTS.md §2
# closed the blending door for our existing models: corr(log e0049, log e0064) = 0.9989, so
# AutoGluon-vs-LightGBM on the same 400 columns is not diversity and the blend buys ~nothing.
# The number to read here is the pairwise correlation column, not just the blend CV -- a
# sequence model that lands 0.003 behind while correlating at ~0.95 is worth more than one
# that ties while correlating at 0.999.
#SBATCH --job-name=blend_seq
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
"$ROOT/envs/ecup/bin/python" src/blend.py "$@"
echo "=== BLEND DONE ==="
