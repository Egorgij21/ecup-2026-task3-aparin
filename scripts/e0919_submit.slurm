#!/bin/bash
# e0919 SUBMIT -- Chronos-Bolt + scale-stats per-token GRU predicted at the REAL test anchor
# (2026-02-13, longest context). Pooled rho_partial +0.0054 (positive!); test-anchor value should be
# higher (per-anchor trend). Trains clean-only (<= 2025-10-16), writes subs/e0919.csv in sample_submit
# order. Calibrate locally (§1b log-mean shift) then read test-time rho from the LB score.
#SBATCH --job-name=e0919sub
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs subs
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/chronos/bin/python"
"$PY" -c "import chronos" || { echo "chronos env broken"; exit 3; }
"$PY" src/run_tsfm_chronos.py --exp-id e0919 --submit
echo "=== e0919 SUBMIT DONE -> subs/e0919.csv (calibrate locally, §1b shift) ==="
