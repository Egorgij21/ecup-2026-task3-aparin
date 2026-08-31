#!/bin/bash
# e0919 -- Chronos-Bolt-small per-token GRU + abs-scale-stats (IDEAS.md §I29, strengthen of e0915).
# Reuses envs/chronos (built + embed-gated by chronos_env, job 24145943). CPU on `compute` -- Chronos-
# Bolt-small is light. Runs the --smoke end-to-end check FIRST (CLUSTER.md gotcha 8); once it passes,
# drop --smoke from the invocation below and resubmit for the full fold-4 feasibility.
#SBATCH --job-name=e0919chr
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/chronos/bin/python"
"$PY" -c "import chronos" || { echo "chronos env broken"; exit 3; }
"$PY" src/run_tsfm_chronos.py --exp-id e0919 --fold 4
echo "=== e0919 full fold-4 DONE ==="
