#!/bin/bash
# Correctness gate for the `seq` approach's data layer.  CPU-only work -> `compute`, never a
# GPU partition (CLUSTER.md rule 5b).
#
# What it proves: the dense (user x channel x day) tensor agrees with src/data.py's
# independently-written prefix-sum panel on targets, population masks, channel window sums and
# the geo3 naive reference.  Two implementations built from the same parquet by different
# routes; if they agree the tensor is right, and if they do not, no CV number from this
# approach means anything.
#SBATCH --job-name=seq_selftest
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:40:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 POLARS_MAX_THREADS=8
PY="$ROOT/envs/seq/bin/python"

"$PY" src/metrics.py
"$PY" src/seqnet.py
"$PY" src/seqdata.py
echo "=== SEQ SELFTEST DONE ==="
