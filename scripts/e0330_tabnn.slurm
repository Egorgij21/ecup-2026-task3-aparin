#!/bin/bash
# e0330 -- modern tabular NN (RealMLP + TabM) as a candidate DECORRELATED blend member
# (IDEAS.md §I22). The one unoccupied point in model-space: a feedforward net on the 665
# aggregate features is a different function class from the GBDT AND a different input
# representation from the GRU-on-raw-sequence. AutoGluon 1.6.1's env torch is CPU-only, so this
# runs on `compute` (no GPU-quota fight) -- exactly where the co-tenant contention is absent.
# Trees and TFMs are excluded via the explicit hyperparameters set; TFMs are already refuted (I17).
#
# Feasibility first: ONE fold (the most test-like, fold 4) before spending 5. Array-friendly:
# pass --array=0-4 later for the full confirm.
#SBATCH --job-name=e0330nn
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
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
echo "=== e0330 RealMLP+TabM NN member, fold $FOLD ==="
"$PY" src/run_ag.py --exp-id e0330 --config configs/e0049_nomoment.yaml --fold "$FOLD" \
  --presets medium_quality --time-limit 21600 \
  --hyperparameters '{"REALMLP":{},"TABM":{}}'
echo "=== e0330 fold $FOLD DONE ==="
