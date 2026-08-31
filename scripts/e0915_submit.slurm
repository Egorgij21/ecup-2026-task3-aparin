#!/bin/bash
# e0915 SUBMIT -- the frozen-MOMENT per-patch GRU predicted at the REAL test anchor (2026-02-13,
# longest histories = where the FM is strongest; IDEAS.md §I28 per-anchor trend). Trains clean-only
# (<= 2025-10-16, guard-safe), writes subs/e0915.csv in sample_submit order. Then calibrate_lb applies
# the §1b log-mean shift and we read the member's TEST-TIME rho from the LB score via rho_from_lb.
# Same env (envs/tsfm) + code path as the validated fold-4 confirm; compute/CPU for reliability.
#SBATCH --job-name=e0915sub
#SBATCH --partition=compute
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs subs
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/tsfm/bin/python"
"$PY" -c "import momentfm" || { echo "tsfm env broken"; exit 3; }
"$PY" src/run_tsfm_gru.py --mode patch --exp-id e0915 --submit
echo "=== e0915 raw submission written; calibrating (log-mean shift, §1b) ==="
"$PY" src/calibrate_lb.py --members e0915 --out e0915_cal --mode shift --assume-rho e0915=0.66
echo "=== e0915 SUBMIT DONE -> subs/e0915_cal.csv ==="
