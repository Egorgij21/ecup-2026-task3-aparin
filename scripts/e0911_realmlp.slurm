#!/bin/bash
# e0911 -- RealMLP retry: the AG env lacks pytabkit (RealMLP's backend). Install it ADDITIVELY
# with --no-deps (no version changes to any shared package -> zero risk to teammates' AG jobs;
# torch/sklearn already present), then run RealMLP on the 1.2M subsample. `compute` has internet.
# If RealMLP still can't import, the job fails fast without having altered the shared env.
#SBATCH --job-name=e0911nn
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
PY="$ROOT/envs/ag/bin/python"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== installing pytabkit (--no-deps, additive) ==="
"$ROOT/envs/ag/bin/pip" install --no-deps pytabkit 2>&1 | tail -5 || echo "pip pytabkit failed"
"$ROOT/envs/ag/bin/pip" install --no-deps skorch 2>&1 | tail -3 || true
"$PY" -c "import pytabkit; print('pytabkit', pytabkit.__version__, 'import OK')" || { echo "pytabkit import FAILED -- aborting"; exit 3; }

FOLD=${SLURM_ARRAY_TASK_ID:-4}
echo "=== e0911 RealMLP (1.2M subsample, no refit), fold $FOLD ==="
"$PY" src/run_ag.py --exp-id e0911 --config configs/e0049_nomoment.yaml --fold "$FOLD" \
  --presets medium_quality --time-limit 18000 --subsample 1200000 --no-refit-full \
  --hyperparameters '{"REALMLP":{}}'
echo "=== e0911 fold $FOLD DONE ==="
