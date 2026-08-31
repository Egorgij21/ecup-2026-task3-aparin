#!/bin/bash
# e0912 -- tabular NN member, AG-NATIVE NN_TORCH (pure-torch MLP; no pytabkit, no torchvision ->
# avoids the env conflict that blocked RealMLP in e0911). 1.2M-row subsample, no refit. The last
# attempt on the "modern tabular NN as a decorrelated member" direction (I22); if this is a
# marginal twin like TabICLv2 (e0340), the direction is closed with an actual number.
#SBATCH --job-name=e0912nn
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
PY="$ROOT/envs/ag/bin/python"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FOLD=${SLURM_ARRAY_TASK_ID:-4}
echo "=== e0912 NN_TORCH (AG-native MLP, 1.2M subsample, no refit), fold $FOLD ==="
"$PY" src/run_ag.py --exp-id e0912 --config configs/e0049_nomoment.yaml --fold "$FOLD" \
  --presets medium_quality --time-limit 14400 --subsample 1200000 --no-refit-full \
  --hyperparameters '{"NN_TORCH":{}}'
echo "=== e0912 fold $FOLD DONE ==="
